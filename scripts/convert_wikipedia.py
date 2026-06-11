#!/usr/bin/env python3
"""Convert Wikipedia article text to JEPA chain JSONL (v4.4 scale-up).

REFRAMING (v4.4, user-directed): this arm DROPS the action-labeled world-model
requirement. The question is no longer "can the model recover a latent action"; it
is "can the slot-encoder architecture learn to reproduce DIVERSE free-form text under
masking — building metric pooled geometry — when surface diversity is UNBOUNDED?"
Wikipedia is the limit case of the data-pressure hypothesis: where the single-template
entity world stalled separation-AUC at ~0.52 (and 18x paraphrase renderings bought only
+0.03-0.04), Wikipedia has effectively infinite surface variety.

CHAIN READING
-------------
A "chain" is a window of 2-4 CONSECUTIVE sentences drawn from a single article
paragraph:

    {"chain": [sent_0, sent_1, sent_2]}

There is NO causal/action structure here (unlike wikiHow's ordered steps). Adjacent
sentences are simply neighbours in running prose. The JEPAChainDataset contract is
unchanged: in `triples` mode a length-3 chain is one (s0,s1,s2) example; in `pairs`
mode it yields adjacent (state_t, state_{t+1}) pairs. The pairing the model trains on
is "predict/reconstruct an adjacent sentence" — a masked-reconstruction signal over
real, diverse prose, NOT a state transition.

NO LABELED TWIN (deliberate)
----------------------------
wikiHow/CommitPackFT emit `<out>_labeled.jsonl` with weak action labels. Wikipedia
sentences carry no action; we emit ONLY `train.jsonl` / `test.jsonl` (plain chains).
The eval path must degrade gracefully without `_labeled` files — entity-oracle metrics
(action-NMI, OOD ladder, rollout) are SKIPPED, separation-AUC still emits off the plain
chains (see src/twm/jepa/diagnostics.py _separation_auc plain-chain fallback).

SOURCE
------
HuggingFace `wikimedia/wikipedia`, config `20231101.en` (the cleaned, prose-only dump;
infoboxes/templates already stripped). STREAMING — we never download the full ~20GB
parquet locally; `--limit-articles` bounds how many articles we pull. The SAME command
validates locally (small `--limit-articles`) and scales on the server (large/0).

FILTERING (skip stubs/lists/tables)
-----------------------------------
Per paragraph we:
  - split into sentences (regex sentence splitter; no heavy NLP dep),
  - drop sentences that look like list/table/markup debris (bullets, pipe-tables,
    heading lines, ref/coordinate junk, mostly-non-alpha),
  - cap sentence length at ~48 words (long run-on sentences blow the token budget),
  - require >= `--min-sent` clean sentences in a window to emit a chain.
Articles shorter than `--min-article-chars` (stubs) are skipped wholesale.

Usage (local sample):
    uv run python scripts/convert_wikipedia.py \
        --limit-articles 4000 --n-train 20000 --n-test 2000 \
        --out-dir data/wikipedia --seed 7

Usage (full, server-side):
    uv run python scripts/convert_wikipedia.py \
        --n-train 500000 --n-test 5000 --out-dir data/wikipedia --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Sentence splitting + cleaning
# ---------------------------------------------------------------------------

# Split on sentence-final punctuation followed by whitespace + a capital/quote/digit.
# Deliberately simple (no spaCy/nltk dependency, deterministic, runs anywhere
# `datasets` does). It over-splits on abbreviations occasionally — acceptable noise
# for a masked-reconstruction corpus.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]")

# Lines/sentences that are list/table/markup debris rather than prose.
_BULLET = re.compile(r"^\s*([*\-•·#]|\d+[.)])\s")
_HEADING = re.compile(r"^\s*={2,}.*={2,}\s*$")  # == Heading ==
_PIPE_TABLE = re.compile(r"\|")                  # wikitable cell separators
_REF_JUNK = re.compile(r"(\{\{|\}\}|\[\[|\]\]|<ref|</ref|http[s]?://|www\.)")


def _clean_sentence(text: str) -> str:
    """Collapse whitespace and strip; return '' if empty."""
    return _WS.sub(" ", text).strip()


def is_prose_sentence(sent: str, min_words: int, max_words: int) -> bool:
    """True iff `sent` looks like a real prose sentence (not list/table/markup debris).

    Filters (skip stubs/lists/tables, per spec):
      - too short (< min_words) or too long (> max_words),
      - bullet/numbered-list lead, == heading == lines, pipe-table rows,
      - template/ref/url markup leakage ({{...}}, [[...]], <ref>, http),
      - mostly-non-alphabetic (coordinates, pure numbers, symbol runs): require the
        fraction of alphabetic characters to clear 0.5.
    """
    if not sent:
        return False
    n_words = len(sent.split())
    if n_words < min_words or n_words > max_words:
        return False
    if _BULLET.match(sent) or _HEADING.match(sent):
        return False
    if _PIPE_TABLE.search(sent) or _REF_JUNK.search(sent):
        return False
    # Mostly-alphabetic gate: count letters vs total non-space chars.
    non_space = [c for c in sent if not c.isspace()]
    if not non_space:
        return False
    alpha = sum(1 for c in non_space if _WORD.match(c))
    if alpha / len(non_space) < 0.5:
        return False
    # Must end with sentence punctuation (drops dangling fragments).
    return sent[-1] in ".!?"


def sentences_from_paragraph(
    para: str, min_words: int, max_words: int
) -> list[str]:
    """Split one paragraph into ordered clean prose sentences (list/table debris dropped)."""
    para = para.strip()
    if not para:
        return []
    raw = _SENT_SPLIT.split(para)
    out: list[str] = []
    for r in raw:
        s = _clean_sentence(r)
        if is_prose_sentence(s, min_words, max_words):
            out.append(s)
    return out


def windows_from_sentences(
    sents: list[str], min_len: int, max_len: int, rng: random.Random
) -> list[list[str]]:
    """Tile a sentence list into NON-OVERLAPPING consecutive-sentence windows.

    Each window is a chain of `min_len`..`max_len` consecutive sentences. Window
    lengths are sampled per window (seeded) in [min_len, max_len], then we walk the
    sentence list left to right consuming that many at a time. A trailing remainder
    shorter than `min_len` is dropped. Non-overlapping so the same sentence never
    appears in two chains (no train/test leakage within an article, and no trivially
    duplicated reconstruction targets)."""
    chains: list[list[str]] = []
    i = 0
    n = len(sents)
    while i + min_len <= n:
        w = rng.randint(min_len, max_len)
        w = min(w, n - i)
        if w < min_len:
            break
        chains.append(sents[i : i + w])
        i += w
    return chains


# ---------------------------------------------------------------------------
# Streaming + build
# ---------------------------------------------------------------------------


def iter_articles(hf_id: str, hf_config: str, split: str, limit: int | None):
    """Stream article `text` strings from the HF dataset (streaming; bounded by limit)."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, hf_config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield str(row.get("text") or "")


def build_chains(
    hf_id: str,
    hf_config: str,
    split: str,
    limit: int | None,
    min_len: int,
    max_len: int,
    min_sent_words: int,
    max_sent_words: int,
    min_article_chars: int,
    target_chains: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Stream articles -> consecutive-sentence-window chains + stats.

    Stops once `target_chains` chains have been collected (so the local sample run is
    fast and bounded even with --limit-articles 0). Returns (chains, stats)."""
    rng = random.Random(seed)
    chains: list[dict] = []
    n_articles = 0
    n_stub = 0
    n_para = 0
    sent_word_counts: list[int] = []
    chain_len_counts: list[int] = []
    vocab: set[str] = set()
    n_tokens = 0

    for text in iter_articles(hf_id, hf_config, split, limit):
        n_articles += 1
        if len(text) < min_article_chars:
            n_stub += 1
            continue
        # Paragraphs are blank-line separated in the cleaned dump.
        for para in text.split("\n\n"):
            n_para += 1
            sents = sentences_from_paragraph(para, min_sent_words, max_sent_words)
            if len(sents) < min_len:
                continue
            for w in windows_from_sentences(sents, min_len, max_len, rng):
                chains.append({"chain": w})
                chain_len_counts.append(len(w))
                for s in w:
                    toks = s.split()
                    sent_word_counts.append(len(toks))
                    n_tokens += len(toks)
                    vocab.update(t.lower() for t in toks)
                if len(chains) >= target_chains:
                    break
            if len(chains) >= target_chains:
                break
        if len(chains) >= target_chains:
            break

    def _pct(xs: list[int], p: float) -> int:
        if not xs:
            return 0
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * p))]

    stats = {
        "articles_seen": n_articles,
        "dropped_stub_articles": n_stub,
        "paragraphs_seen": n_para,
        "chains_built": len(chains),
        "min_len": min_len,
        "max_len": max_len,
        "min_sent_words": min_sent_words,
        "max_sent_words": max_sent_words,
        "mean_chain_len": round(statistics.mean(chain_len_counts), 2) if chain_len_counts else 0,
        "mean_sent_words": round(statistics.mean(sent_word_counts), 2) if sent_word_counts else 0,
        "sent_words_p50": _pct(sent_word_counts, 0.50),
        "sent_words_p90": _pct(sent_word_counts, 0.90),
        "sent_words_p99": _pct(sent_word_counts, 0.99),
        "sent_words_max": max(sent_word_counts) if sent_word_counts else 0,
        # Vocabulary richness: distinct lowercased word types / total word tokens
        # (type-token ratio). A proxy for surface diversity — the v4.4 hypothesis's
        # whole point (Wikipedia >> single-template entity world).
        "n_word_tokens": n_tokens,
        "n_word_types": len(vocab),
        "type_token_ratio": round(len(vocab) / n_tokens, 5) if n_tokens else 0,
    }
    return chains, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hf-id", default="wikimedia/wikipedia")
    ap.add_argument("--hf-config", default="20231101.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", default="data/wikipedia")
    ap.add_argument("--n-train", type=int, default=20_000, help="max train chains to emit")
    ap.add_argument("--n-test", type=int, default=2_000, help="iid test chains to emit")
    ap.add_argument(
        "--limit-articles", type=int, default=0,
        help="stream only the first N articles (0 = stream until target chains hit).",
    )
    ap.add_argument("--min-len", type=int, default=2, help="min sentences per chain window")
    ap.add_argument("--max-len", type=int, default=4, help="max sentences per chain window")
    ap.add_argument("--min-sent-words", type=int, default=4, help="drop sentences shorter than this")
    ap.add_argument("--max-sent-words", type=int, default=48, help="drop/skip sentences longer than this")
    ap.add_argument("--min-article-chars", type=int, default=600, help="skip stub articles shorter than this")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    limit = args.limit_articles or None
    # Over-collect by n_test so the disjoint test split does not starve train.
    target = args.n_train + args.n_test
    print(f"Source: {args.hf_id} ({args.hf_config}, split={args.split})")
    print(f"Streaming{f' first {limit}' if limit else ''} articles until {target} chains...")

    chains, stats = build_chains(
        hf_id=args.hf_id,
        hf_config=args.hf_config,
        split=args.split,
        limit=limit,
        min_len=args.min_len,
        max_len=args.max_len,
        min_sent_words=args.min_sent_words,
        max_sent_words=args.max_sent_words,
        min_article_chars=args.min_article_chars,
        target_chains=target,
        seed=args.seed,
    )
    print(
        f"Built {len(chains)} chains from {stats['articles_seen']} articles "
        f"(dropped {stats['dropped_stub_articles']} stubs). "
        f"mean {stats['mean_chain_len']} sents/chain, "
        f"mean {stats['mean_sent_words']} words/sent (p99={stats['sent_words_p99']}). "
        f"TTR={stats['type_token_ratio']}."
    )

    # Seeded shuffle, then split into disjoint train / iid-test.
    rng = random.Random(args.seed)
    rng.shuffle(chains)
    n_test = min(args.n_test, len(chains))
    test_chains = chains[:n_test]
    train_chains = chains[n_test : n_test + args.n_train]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train_chains)
    write_jsonl(out / "test.jsonl", test_chains)

    manifest = {
        "corpus": "wikipedia",
        "source": args.hf_id,
        "hf_config": args.hf_config,
        "license": "cc-by-sa-4.0",
        "license_note": "Wikipedia text: CC BY-SA 4.0 + GFDL. Attribution + share-alike.",
        "world_state_reading": (
            "v4.4 reframing: NO action/world-state. A chain is a window of 2-4 consecutive "
            "sentences from one article paragraph; the training signal is masked "
            "reconstruction of adjacent diverse prose (data-pressure limit case)."
        ),
        "labeled_twin": False,
        "labeled_twin_note": (
            "No action labels (Wikipedia sentences carry no action). Eval degrades "
            "gracefully: entity-oracle metrics skipped, separation-AUC runs off plain chains."
        ),
        "seed": args.seed,
        "limit_articles": args.limit_articles,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_sent_words": args.min_sent_words,
        "max_sent_words": args.max_sent_words,
        "min_article_chars": args.min_article_chars,
        "splits": {
            "train": {"chains": len(train_chains)},
            "test": {"chains": len(test_chains)},
        },
        "build_stats": stats,
        "files": ["train.jsonl", "test.jsonl"],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote to {out}/:")
    print(f"  train.jsonl  {len(train_chains):>7} chains")
    print(f"  test.jsonl   {len(test_chains):>7} chains")
    print(f"  manifest.json")
    if train_chains:
        print("\nSample chain:")
        for j, st in enumerate(train_chains[0]["chain"]):
            print(f"    s{j}: {st[:90]}")


if __name__ == "__main__":
    main()
