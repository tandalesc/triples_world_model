#!/usr/bin/env python3
"""Convert CommitPackFT commits to length-2 JEPA chains (code-edit transitions).

WORLD-STATE READING (survey §4a / transition_corpora_survey.md rank 3):
A commit is a genuine (state_t, action, state_{t+1}) transition: the file before
the change is state_t, the file after is state_{t+1}, and the commit message is
the explicit natural-language action. We encode each commit as a LENGTH-2 chain:

    state_t   = "BEFORE: <old code, truncated to N tokens>"
    state_t+1 = "AFTER: <new code, truncated to N tokens>"
    action    = cleaned commit subject line

    {"chain": [state_t, state_t1]}                       # plain
    {"chain": [...], "actions": ["<commit msg>@0"]}      # labeled twin

SCHEMA DECISION (documented per task): we keep the BEFORE:/AFTER: prefixes inside
the state text so a single shared BPE/decoder can tell which side of the edit it
is generating, and so the JEPA target encoder sees "AFTER" surface structure that
contrasts with the "BEFORE" source — this strengthens the cross-state signal that
data.py requires (online sees state_t, EMA target sees state_{t+1}). Code is
truncated to --max-code-tokens *whitespace tokens* (a cheap proxy that bounds the
downstream BPE length; the BPE coverage report converts this to true subword
lengths). Truncation is on the changed region's neighborhood when possible (see
_truncate_around_change) so the snippet actually contains the diff, not just the
file header.

data.py handles length-2 chains: a length-L chain yields L-1 pairs, so L=2 -> 1
pair (the single transition). In `triples` mode length-2 chains are SKIPPED
(needs >=3 states) — so these configs MUST use pairing mode `pairs` (data.mode
default "pairs"). Recorded in the manifest + the corpus config.

FILTERS (survey + task):
  - single-file commit (old_file == new_file, both present)        [structural]
  - small diff: < --max-changed-lines changed lines                [transition density]
  - one language (--langs python by default; add more to scale)    [domain control]
  - permissive license subset (--licenses)                         [reuse safety]
  - non-trivial: before != after, message non-empty after cleaning [usefulness]

SCALE: --limit-rows streams only the first N rows per language (HF streaming) so
the SAME command validates locally (e.g. --limit-rows 2000) and scales server-side
(drop the flag). CommitPackFT python alone is ~56K rows -> ~40-50K post-filter;
to reach the 100K target add languages: --langs python,javascript,java,go,ruby.

Usage (local sample):
    uv run python scripts/convert_commitpack.py \
        --limit-rows 2000 --n-train 1500 --n-test 200 \
        --out-dir data/commitpack --seed 7

Usage (full, server-side, 100K target):
    uv run python scripts/convert_commitpack.py \
        --langs python,javascript,java,go,ruby \
        --n-train 100000 --n-test 5000 --out-dir data/commitpack --seed 7
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from pathlib import Path

# Permissive licenses (per survey §4a). CommitPackFT's `license` is the repo license.
DEFAULT_LICENSES = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "mpl-2.0",
    "cc0-1.0", "unlicense", "0bsd",
}

# CommitPackFT configs are full language names ("Python"), but the config key passed
# to load_dataset is lowercased ("python"). We accept either spelling on the CLI.
_LANG_ALIAS = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "java": "java", "go": "go", "golang": "go",
    "ruby": "ruby", "rust": "rust", "c": "c", "cpp": "c++", "c++": "c++",
    "typescript": "typescript", "ts": "typescript",
}


def clean_message(subject: str, message: str) -> str:
    """Clean a commit message into a one-line action label.

    Prefer `subject` (the first line); fall back to the first line of `message`.
    Strip trailing whitespace, drop trailing issue refs / signoffs, collapse space.
    """
    text = (subject or "").strip()
    if not text:
        text = (message or "").strip().splitlines()[0] if message else ""
    text = text.splitlines()[0] if text else ""
    # Drop common noise suffixes: "(#123)", "[skip ci]", trailing issue tags.
    text = re.sub(r"\s*\(#\d+\)\s*$", "", text)
    text = re.sub(r"\s*\[[^\]]*\]\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Commit contents in the wild contain leaked credentials (GitHub push
# protection flags them, and they should never enter training data anyway).
_SECRET_PATTERNS = [
    re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"(?i)(?:aws)?_?secret_?(?:access)?_?key[\"'\s:=]+[A-Za-z0-9/+=]{40}"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api|auth)_?(?:key|token)[\"'\s:=]+[A-Za-z0-9_\-]{32,}"),
]


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_PATTERNS)


def count_changed_lines(old: str, new: str) -> int:
    """Number of added+removed lines in the unified diff (cheap diff-size proxy)."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    n = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
        if line[:3] in ("---", "+++", "@@ "):
            continue
        if line and line[0] in "+-":
            n += 1
    return n


def _first_change_line(old: str, new: str) -> int:
    """Index of the first differing line (for change-centered truncation)."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            return i1
    return 0


def _truncate_around_change(code: str, max_tokens: int, center_line: int) -> str:
    """Truncate code to ~max_tokens whitespace tokens, windowed around center_line.

    Keeps the snippet centered on the changed region so the BEFORE/AFTER text
    actually contains the diff rather than just the file's import header. Falls
    back to a head truncation when centering isn't informative.
    """
    lines = code.splitlines()
    if not lines:
        return ""
    # Expand a line window outward from center until we hit the token budget.
    lo = hi = max(0, min(center_line, len(lines) - 1))
    tok = len(lines[lo].split())
    while tok < max_tokens and (lo > 0 or hi < len(lines) - 1):
        if lo > 0:
            lo -= 1
            tok += len(lines[lo].split())
        if hi < len(lines) - 1 and tok < max_tokens:
            hi += 1
            tok += len(lines[hi].split())
    window = "\n".join(lines[lo : hi + 1])
    toks = window.split()
    if len(toks) > max_tokens:
        window = " ".join(toks[:max_tokens])
    return window.strip()


def make_state_texts(old: str, new: str, max_tokens: int) -> tuple[str, str]:
    """Build ("BEFORE: ...", "AFTER: ...") truncated around the change."""
    center = _first_change_line(old, new)
    old_snip = _truncate_around_change(old, max_tokens, center)
    new_snip = _truncate_around_change(new, max_tokens, center)
    return f"BEFORE: {old_snip}", f"AFTER: {new_snip}"


def iter_rows(lang_key: str, limit: int | None):
    """Stream rows for one CommitPackFT language config (streaming; bounded).

    CommitPackFT ships a (now-unsupported) loading script, so we point `datasets`
    at the per-language jsonl file directly via the hub resolve URL. The data lives
    at data/<lang>/data.jsonl in the repo. Streaming keeps memory bounded for the
    full server run.
    """
    from datasets import load_dataset

    url = f"hf://datasets/bigcode/commitpackft/data/{lang_key}/data.jsonl"
    ds = load_dataset("json", data_files=url, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield row


def build_chains(
    langs: list[str],
    licenses: set[str],
    limit: int | None,
    max_changed_lines: int,
    max_code_tokens: int,
) -> tuple[list[dict], list[dict], dict]:
    plain: list[dict] = []
    labeled: list[dict] = []
    n_seen = 0
    drop = {"multi_file": 0, "big_diff": 0, "license": 0, "no_change": 0, "empty_msg": 0, "secret": 0}

    for lang in langs:
        key = _LANG_ALIAS.get(lang.lower().strip(), lang.lower().strip())
        for row in iter_rows(key, limit):
            n_seen += 1
            old_file = str(row.get("old_file") or "")
            new_file = str(row.get("new_file") or "")
            old = str(row.get("old_contents") or "")
            new = str(row.get("new_contents") or "")
            lic = str(row.get("license") or "").lower().strip()

            if not old_file or old_file != new_file:
                drop["multi_file"] += 1
                continue
            if licenses and lic not in licenses:
                drop["license"] += 1
                continue
            if old == new:
                drop["no_change"] += 1
                continue
            if count_changed_lines(old, new) >= max_changed_lines:
                drop["big_diff"] += 1
                continue
            msg = clean_message(str(row.get("subject") or ""), str(row.get("message") or ""))
            if not msg:
                drop["empty_msg"] += 1
                continue
            if contains_secret(old) or contains_secret(new) or contains_secret(msg):
                drop["secret"] += 1
                continue

            s0, s1 = make_state_texts(old, new, max_code_tokens)
            plain.append({"chain": [s0, s1]})
            labeled.append({
                "chain": [s0, s1],
                "actions": [f"{msg}@0"],
                "lang": str(row.get("lang") or key),
                "license": lic,
            })

    stats = {
        "rows_seen": n_seen,
        "chains_kept": len(plain),
        "dropped": drop,
        "max_changed_lines": max_changed_lines,
        "max_code_tokens": max_code_tokens,
        "transitions": len(plain),  # length-2 chains => 1 transition each
    }
    return plain, labeled, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="python", help="comma-separated language configs")
    ap.add_argument("--licenses", default=",".join(sorted(DEFAULT_LICENSES)),
                    help="comma-separated permissive licenses to keep (empty = all)")
    ap.add_argument("--out-dir", default="data/commitpack")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-test", type=int, default=5_000)
    ap.add_argument("--limit-rows", type=int, default=0,
                    help="stream only first N rows PER LANGUAGE (0 = all)")
    ap.add_argument("--max-changed-lines", type=int, default=30)
    ap.add_argument("--max-code-tokens", type=int, default=96,
                    help="whitespace-token budget per code side (proxy bounding BPE length)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    langs = [x for x in args.langs.split(",") if x.strip()]
    licenses = {x.strip().lower() for x in args.licenses.split(",") if x.strip()}
    limit = args.limit_rows or None
    print(f"Source: bigcode/commitpackft langs={langs} licenses={sorted(licenses) or 'ALL'}")
    print(f"Streaming{f' first {limit}/lang' if limit else ' ALL rows'}...")

    plain, labeled, stats = build_chains(
        langs, licenses, limit, args.max_changed_lines, args.max_code_tokens
    )
    print(f"Kept {len(plain)} transitions from {stats['rows_seen']} rows. Dropped: {stats['dropped']}")

    rng = random.Random(args.seed)
    order = list(range(len(plain)))
    rng.shuffle(order)
    plain = [plain[i] for i in order]
    labeled = [labeled[i] for i in order]

    n_test = min(args.n_test, len(plain))
    test_plain, test_labeled = plain[:n_test], labeled[:n_test]
    rest_plain, rest_labeled = plain[n_test:], labeled[n_test:]
    n_train = min(args.n_train, len(rest_plain))
    train_plain, train_labeled = rest_plain[:n_train], rest_labeled[:n_train]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train_plain)
    write_jsonl(out / "train_labeled.jsonl", train_labeled)
    write_jsonl(out / "test.jsonl", test_plain)
    write_jsonl(out / "test_labeled.jsonl", test_labeled)

    manifest = {
        "corpus": "commitpack",
        "source": "bigcode/commitpackft",
        "license": "permissive per-repo (filtered): " + ", ".join(sorted(licenses)),
        "schema": {
            "state_t": "BEFORE: <old code truncated to max_code_tokens around change>",
            "state_t+1": "AFTER: <new code truncated to max_code_tokens around change>",
            "action": "cleaned commit subject line (labeled twin only)",
        },
        "chain_length": 2,
        "data_mode": "pairs (length-2 chains are skipped in triples mode)",
        "langs": langs,
        "filters": {
            "single_file": True,
            "max_changed_lines": args.max_changed_lines,
            "permissive_only": bool(licenses),
        },
        "seed": args.seed,
        "limit_rows": args.limit_rows,
        "splits": {
            "train": {"chains": len(train_plain), "transitions": len(train_plain)},
            "test": {"chains": len(test_plain), "transitions": len(test_plain)},
        },
        "build_stats": stats,
        "files": ["train.jsonl", "train_labeled.jsonl", "test.jsonl", "test_labeled.jsonl"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote to {out}/:")
    print(f"  train.jsonl          {len(train_plain):>7} chains")
    print(f"  test.jsonl           {len(test_plain):>7} chains")
    print(f"  + _labeled twins + manifest.json")
    if train_plain:
        print("\nSample transition:")
        ex = train_labeled[0]
        print(f"  action: {ex['actions'][0]}")
        print(f"  s0: {ex['chain'][0][:90]}")
        print(f"  s1: {ex['chain'][1][:90]}")


if __name__ == "__main__":
    main()
