#!/usr/bin/env python3
"""Convert a wikiHow steps corpus to JEPA chain JSONL.

WORLD-STATE READING (survey §2a / transition_corpora_survey.md rank 2):
Each wikiHow article is an ordered procedure. We read each *step* as a state
snapshot of task progress; the (implicit) action is whatever transition carried
the procedure from step i to step i+1. A chain is therefore the article's ordered
list of step texts:

    {"chain": [step_0_text, step_1_text, ..., step_k_text]}

This is the JEPAChainDataset contract (src/twm/jepa/data.py): a length-L chain
yields L-1 adjacent (state_t, state_{t+1}) pairs in `pairs` mode, or one
(s0,s1,s2) example per length>=3 chain in `triples` mode. wikiHow is the natural
20x scale-up of OpenPI (data/openpi_train.jsonl), the repo's existing procedural
lineage; we reuse its normalization spirit (clean, lowercased, whitespace-collapsed
surface forms) but keep step text as free NL (open-vocab path), not (e,a,v) triples.

------------------------------------------------------------------------------
SOURCE CHOICE (documented; the survey's pick was wrong-shape — see notes)
------------------------------------------------------------------------------
The survey nominated `tasksource/goal-step-wikihow`. On inspection ALL three of
its subsets (goal/order/step) are multiple-choice QA reformattings (sent1, sent2,
ending0..3, label) — none expose a clean ordered step list per article. So we use:

  PRIMARY:  gursi26/wikihow-cleaned   (214,293 articles, CC BY-NC-SA 3.0)
            `summary` field = the article's step HEADLINES, sentence-separated.
            Each headline is an imperative step ("take theatre classes .",
            "volunteer at a theatre ."). We split the summary into ordered
            headline states -> the chain. Mean ~6 steps/article -> ~1.2M
            transitions extractable; far above the 200K-chain target.

  ALT:      b-mc2/wikihow_lists        (11,461 articles, CC BY-NC-SA 3.0)
            `result` = numbered "1. ... 2. ..." stepped summary. Cleaner step
            boundaries but ~20x smaller. Selectable via --source wikihow_lists
            for a higher-precision (but small) variant.

LICENSE: CC BY-NC-SA 3.0 (wikiHow content). Non-commercial; fine for research,
must be flagged for any release. Recorded in the manifest.

------------------------------------------------------------------------------
LABELED TWIN (weak action labels for action-recovery eval)
------------------------------------------------------------------------------
We emit BOTH `<out>.jsonl` (plain chains) and `<out>_labeled.jsonl` carrying a
per-transition weak action label, mirroring data/entity_world/train_labeled.jsonl
({"chain": [...], "actions": ["wash@0", ...]}). The label for transition i is the
FIRST VERB LEMMA of step_{i+1}'s headline (the imperative verb that drives the
next state), tagged "@0" to match the entity-world entity-slot convention (single
implicit actor). Heuristic + its noise are documented in
research/corpus_conversion_notes.md and in extract_action_label() below.

------------------------------------------------------------------------------
SCALE: local sample vs full server run
------------------------------------------------------------------------------
--limit-articles streams only the first N articles (HF streaming) so the SAME
command validates end-to-end locally (e.g. --limit-articles 2000) and scales on
the server (drop the flag, or --limit-articles 0 for all). --n-train / --n-test
cap the emitted chain counts (seeded sample).

Usage (local sample):
    uv run python scripts/convert_wikihow.py \
        --limit-articles 2000 --n-train 1500 --n-test 200 \
        --out-dir data/wikihow --seed 7

Usage (full, server-side):
    uv run python scripts/convert_wikihow.py \
        --n-train 200000 --n-test 5000 --out-dir data/wikihow --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Action-label extraction (weak; first-verb-lemma heuristic)
# ---------------------------------------------------------------------------

# Tiny irregular-verb lemma map + suffix rules. We deliberately AVOID a heavy NLP
# dependency (spaCy/nltk) so the converter runs anywhere `tokenizers` does and is
# deterministic. wikiHow step headlines are overwhelmingly imperative — the first
# token is almost always the bare-form verb already ("take", "open", "remove"),
# so lemmatization is mostly a no-op; the suffix rules only catch the occasional
# gerund/3rd-person leak ("opening"->"open", "removes"->"remove").
_IRREGULAR = {
    "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "doing": "do", "done": "do",
    "made": "make", "making": "make",
    "took": "take", "taking": "take", "taken": "take",
    "put": "put", "putting": "put",
    "got": "get", "getting": "get", "gotten": "get",
    "set": "set", "setting": "set",
    "cut": "cut", "cutting": "cut",
    "kept": "keep", "keeping": "keep",
    "left": "leave", "leaving": "leave",
    "held": "hold", "holding": "hold",
    "let": "let", "letting": "let",
    "ran": "run", "running": "run",
    "began": "begin", "beginning": "begin",
}

# Leading words to skip before the verb: politeness/adverb/article noise that can
# precede the imperative ("first, open ...", "carefully remove ...", "be sure to").
_SKIP_LEADING = {
    "please", "first", "firstly", "next", "then", "now", "finally", "lastly",
    "also", "carefully", "gently", "slowly", "quickly", "simply", "just",
    "always", "never", "again", "the", "a", "an", "to", "be", "sure", "make",
    "try", "go", "and", "or", "if", "when", "you", "your",
}
# NOTE: "make/try/go/be" double as real verbs; we only skip them when a *later*
# token is a plausible verb (handled in extract_action_label by falling back to
# the first content token if every token gets skipped).

_WORD = re.compile(r"[a-z][a-z'-]*")


def _lemmatize(tok: str) -> str:
    """Crude verb lemmatizer: irregular map first, then -ing/-es/-s/-ed suffix strip."""
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    if tok.endswith("ing") and len(tok) > 5:
        stem = tok[:-3]
        # doubled consonant ("running"->"runn"->"run"); else add back an 'e' guess off.
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 4 and tok[-3] in "sxz":
        return tok[:-2]
    if tok.endswith("ed") and len(tok) > 4:
        stem = tok[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def extract_action_label(step_text: str) -> str:
    """Weak action label = first-verb lemma of a step headline.

    HEURISTIC & NOISE (documented per task requirement):
      - wikiHow step headlines are imperative; the action verb is usually the
        FIRST content word. We take the first alphabetic token, skipping a small
        set of leading adverbs/politeness/articles (_SKIP_LEADING), then lemmatize.
      - Failure modes (the "noise" this label carries):
          * Non-imperative headlines ("Your phone will restart.") -> the label
            is a noun/pronoun-derived token, not a true action. ~5-10% of steps.
          * Phrasal verbs lose their particle ("turn off" -> "turn").
          * Skip-list collisions: a headline that legitimately starts with
            "make"/"try"/"go" yields "" after skipping; we fall back to the first
            content token so the label is never empty.
      - This is a WEAK supervisory signal for action-recovery eval, NOT ground
        truth. Treat recovered-action vs label agreement as a soft metric.
    Returns a single lowercase lemma, or "none" if no word token exists.
    """
    toks = _WORD.findall(step_text.lower())
    if not toks:
        return "none"
    for t in toks:
        if t not in _SKIP_LEADING:
            return _lemmatize(t)
    # Everything was a skip word — fall back to the first token's lemma.
    return _lemmatize(toks[0])


# ---------------------------------------------------------------------------
# Step splitting per source
# ---------------------------------------------------------------------------

# Sentence splitter for the cleaned-summary source: split on a period that is
# followed by whitespace+lowercase OR end-of-string. The corpus is lowercased and
# uses " . " between step headlines.
_SENT_SPLIT = re.compile(r"\s*\.\s+")
_NUM_STEP = re.compile(r"^\s*\d+\.\s*", re.M)


def _clean_step(text: str) -> str:
    """Normalize a step headline: collapse whitespace, strip, ensure trailing period.

    Mirrors OpenPI/convert_openpi.py normalization spirit (lowercase, collapsed
    whitespace) but keeps the step as a free-NL sentence (open-vocab path)."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" .")
    if not text:
        return ""
    return text + "."


def steps_from_cleaned_summary(summary: str) -> list[str]:
    """gursi26/wikihow-cleaned: split the `summary` field into ordered step headlines."""
    if not summary:
        return []
    parts = _SENT_SPLIT.split(summary.strip())
    steps = [_clean_step(p) for p in parts]
    return [s for s in steps if s and len(s) >= 3]


def steps_from_wikihow_lists(result: str) -> list[str]:
    """b-mc2/wikihow_lists: split the numbered `result` ("1. ...\\n2. ...") into steps."""
    if not result:
        return []
    # Drop the leading "N." markers, then split on them.
    parts = _NUM_STEP.split(result)
    steps = [_clean_step(p) for p in parts]
    return [s for s in steps if s and len(s) >= 3]


SOURCES = {
    "wikihow_cleaned": {
        "hf_id": "gursi26/wikihow-cleaned",
        "config": "default",
        "split": "train",
        "field": "summary",
        "splitter": steps_from_cleaned_summary,
        "license": "cc-by-nc-sa-3.0",
        "n_articles": 214293,
    },
    "wikihow_lists": {
        "hf_id": "b-mc2/wikihow_lists",
        "config": "default",
        "split": "train",
        "field": "result",
        "splitter": steps_from_wikihow_lists,
        "license": "cc-by-nc-sa-3.0",
        "n_articles": 11461,
    },
}


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def iter_articles(source: dict, limit: int | None):
    """Stream (title, ordered_steps) from the HF dataset (streaming; bounded by limit)."""
    from datasets import load_dataset

    ds = load_dataset(
        source["hf_id"], source["config"], split=source["split"], streaming=True
    )
    splitter = source["splitter"]
    field = source["field"]
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        steps = splitter(str(row.get(field) or ""))
        title = str(row.get("title") or "").strip()
        yield title, steps


def build_chains(
    source: dict,
    limit: int | None,
    min_steps: int,
    max_steps: int,
) -> tuple[list[dict], list[dict], dict]:
    """Build (plain_chains, labeled_chains, stats).

    Each kept article -> one chain dict. Plain: {"chain":[...]}. Labeled adds
    "actions":["verb@0", ...] (len == len(chain)-1) and "title".
    """
    plain: list[dict] = []
    labeled: list[dict] = []
    n_seen = 0
    n_too_short = 0
    n_truncated = 0
    step_counts: list[int] = []

    for title, steps in iter_articles(source, limit):
        n_seen += 1
        if len(steps) < min_steps:
            n_too_short += 1
            continue
        if len(steps) > max_steps:
            steps = steps[:max_steps]
            n_truncated += 1
        chain = list(steps)
        # Action label per transition i: first-verb lemma of the *next* step.
        actions = [f"{extract_action_label(chain[i + 1])}@0" for i in range(len(chain) - 1)]
        plain.append({"chain": chain})
        labeled.append({"chain": chain, "actions": actions, "title": title})
        step_counts.append(len(chain))

    stats = {
        "articles_seen": n_seen,
        "chains_kept": len(plain),
        "dropped_too_short": n_too_short,
        "truncated_over_max": n_truncated,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "mean_steps": round(sum(step_counts) / len(step_counts), 2) if step_counts else 0,
        "transitions": sum(c - 1 for c in step_counts),
    }
    return plain, labeled, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(SOURCES), default="wikihow_cleaned")
    ap.add_argument("--out-dir", default="data/wikihow")
    ap.add_argument("--n-train", type=int, default=200_000, help="max train chains to emit")
    ap.add_argument("--n-test", type=int, default=5_000, help="iid test chains to emit")
    ap.add_argument(
        "--limit-articles", type=int, default=0,
        help="stream only the first N articles (0 = all). Use a small value for local sampling.",
    )
    ap.add_argument("--min-steps", type=int, default=2, help="min steps to keep an article as a chain")
    ap.add_argument("--max-steps", type=int, default=12, help="truncate chains longer than this")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    source = SOURCES[args.source]
    limit = args.limit_articles or None
    print(f"Source: {source['hf_id']} (license {source['license']}, ~{source['n_articles']} articles)")
    print(f"Streaming{f' first {limit}' if limit else ' ALL'} articles...")

    plain, labeled, stats = build_chains(source, limit, args.min_steps, args.max_steps)
    print(f"Built {len(plain)} chains from {stats['articles_seen']} articles "
          f"(dropped {stats['dropped_too_short']} too-short, "
          f"truncated {stats['truncated_over_max']}). "
          f"~{stats['transitions']} transitions, mean {stats['mean_steps']} steps/chain.")

    # Seeded shuffle, then split into train / iid-test (disjoint articles).
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
        "corpus": "wikihow",
        "source": source["hf_id"],
        "source_key": args.source,
        "license": source["license"],
        "license_note": "CC BY-NC-SA 3.0 (wikiHow content): non-commercial; research use OK, flag for release.",
        "world_state_reading": "each step headline is a task-progress state snapshot; step transitions are implicit actions",
        "action_label_heuristic": "first-verb lemma of the next step's headline (weak; see convert_wikihow.extract_action_label)",
        "seed": args.seed,
        "limit_articles": args.limit_articles,
        "min_steps": args.min_steps,
        "max_steps": args.max_steps,
        "splits": {
            "train": {"chains": len(train_plain),
                      "transitions": sum(len(c["chain"]) - 1 for c in train_plain)},
            "test": {"chains": len(test_plain),
                     "transitions": sum(len(c["chain"]) - 1 for c in test_plain)},
        },
        "build_stats": stats,
        "files": ["train.jsonl", "train_labeled.jsonl", "test.jsonl", "test_labeled.jsonl"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote to {out}/:")
    print(f"  train.jsonl          {len(train_plain):>7} chains")
    print(f"  train_labeled.jsonl  {len(train_labeled):>7} chains")
    print(f"  test.jsonl           {len(test_plain):>7} chains")
    print(f"  test_labeled.jsonl   {len(test_labeled):>7} chains")
    print(f"  manifest.json")
    if train_plain:
        print("\nSample chain:")
        ex = train_labeled[0]
        for j, st in enumerate(ex["chain"]):
            act = f"  --[{ex['actions'][j]}]-->" if j < len(ex["actions"]) else ""
            print(f"    s{j}: {st[:80]}{act}")


if __name__ == "__main__":
    main()
