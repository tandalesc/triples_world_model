#!/usr/bin/env python3
"""Build a large-vocab domain BPE for the wikiHow / CommitPackFT corpora.

WHY ~8192 (not 512): the existing domain BPEs (data/glucose/jepa_bpe_512.json,
data/entity_world/bpe_512.json) cover a *tiny, low-entropy* surface vocabulary
(placeholder-heavy GLUCOSE, a fixed entity/attribute pool). wikiHow is open-domain
English prose and CommitPackFT is source code — both have a far larger type
vocabulary (identifiers, punctuation runs, rare words). A 512 BPE shatters that
into byte fragments (the entity-world report measured ~13 byte-frags/state at 512;
open-domain text is worse), inflating tokens-per-state past any reasonable
max_text_tokens. We target ~8192 and *measure* the fragmentation to justify it.

This script (mirroring scripts/build_jepa_bpe.py + build_entity_world_bpe.py):
  - trains a ByteLevel BPE with the SAME special-token order so <pad>=0 matches
    DomainBPETokenizer.PAD_ID (the trainer/data path hardcodes pad id 0),
  - reports tokens-per-state distribution + overflow rate vs a candidate
    max_text_tokens, and RECOMMENDS a max per corpus,
  - can train PER-CORPUS (default) or a SHARED BPE over both, and reports both so
    the shared-vs-separate tradeoff is measurable.

Usage (per-corpus, after the converters):
    uv run python scripts/build_corpus_bpe.py --corpus wikihow \
        --data-dir data/wikihow --vocab-size 8192 --max-text-tokens 128
    uv run python scripts/build_corpus_bpe.py --corpus commitpack \
        --data-dir data/commitpack --vocab-size 8192 --max-text-tokens 160

Usage (shared BPE over both corpora + comparison):
    uv run python scripts/build_corpus_bpe.py --corpus shared \
        --data-dir data/wikihow,data/commitpack --vocab-size 8192 --max-text-tokens 160
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

SPECIAL_TOKENS = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]  # <pad>=0 convention
INITIAL_ALPHABET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"()[]{}<>-/_=+*&%$#@\\|~`^ \t"
)


def collect_texts(data_dirs: list[Path]) -> list[str]:
    """Collect all state texts from train + test JSONL chains across the given dirs.

    Test splits are included at BUILD time for vocabulary coverage only (character
    statistics, no dynamics) — same rationale as build_entity_world_bpe.py."""
    texts: list[str] = []
    for data_dir in data_dirs:
        for fname in ["train.jsonl", "test.jsonl"]:
            path = data_dir / fname
            if not path.exists():
                continue
            n_before = len(texts)
            with open(path) as f:
                for line in f:
                    for step_text in json.loads(line)["chain"]:
                        texts.append(step_text)
            print(f"  Read {data_dir.name}/{fname}: +{len(texts) - n_before} states")
    return texts


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=INITIAL_ALPHABET,
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def coverage_report(tok: Tokenizer, texts: list[str], max_tokens: int, label: str) -> dict:
    """Token-count distribution + overflow rate. Returns a stats dict for the report file."""
    lengths = [len(tok.encode(t).ids) for t in texts]
    n = len(lengths)
    sl = sorted(lengths)
    n_over = sum(1 for l in lengths if l > max_tokens)
    rep = {
        "label": label,
        "states": n,
        "vocab_size": tok.get_vocab_size(),
        "mean_tokens": round(statistics.mean(lengths), 2),
        "p50": sl[n // 2],
        "p90": sl[int(n * 0.90)],
        "p95": sl[int(n * 0.95)],
        "p99": sl[int(n * 0.99)],
        "max": max(lengths),
        "max_text_tokens": max_tokens,
        "n_over_max": n_over,
        "pct_over_max": round(100.0 * n_over / n, 3) if n else 0.0,
    }
    print(f"\n=== Coverage [{label}] (vocab={rep['vocab_size']}, max_text_tokens={max_tokens}) ===")
    print(f"  States   : {n:,}")
    print(f"  Mean tok : {rep['mean_tokens']}")
    print(f"  P50/P90/P95/P99 : {rep['p50']}/{rep['p90']}/{rep['p95']}/{rep['p99']}")
    print(f"  Max tok  : {rep['max']}")
    print(f"  > {max_tokens} tok: {n_over:,} ({rep['pct_over_max']}%)")
    return rep


def recommend_max(rep: dict) -> tuple[int, str]:
    """Recommend a max_text_tokens covering ~p99 (rounded up to a multiple of 16)."""
    p99 = rep["p99"]
    rec = ((p99 + 15) // 16) * 16
    verdict = (
        f"p99={p99} -> max_text_tokens={rec} covers >=99% of states; "
        f"current candidate {rep['max_text_tokens']} drops {rep['pct_over_max']}%."
    )
    return rec, verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["wikihow", "commitpack", "gharchive", "shared"], required=True)
    ap.add_argument("--data-dir", required=True,
                    help="data dir (comma-separated for --corpus shared)")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-text-tokens", type=int, default=128,
                    help="candidate max for the overflow report")
    ap.add_argument("--out", default=None, help="override output BPE json path")
    args = ap.parse_args()

    data_dirs = [Path(d) for d in args.data_dir.split(",")]
    out_path = Path(args.out) if args.out else data_dirs[0] / f"bpe_{args.vocab_size}.json"

    print(f"Building {args.corpus} BPE (vocab={args.vocab_size}) -> {out_path}")
    texts = collect_texts(data_dirs)
    if not texts:
        raise RuntimeError(f"No texts under {data_dirs}. Run the converter first.")
    print(f"  Total state texts: {len(texts):,}")

    tok = build_tokenizer(texts, args.vocab_size)
    pad_id = tok.token_to_id("<pad>")
    print(f"\nTrained. vocab={tok.get_vocab_size()}  <pad>={pad_id}")
    if pad_id != 0:
        print("  WARNING: <pad> != 0; DomainBPETokenizer.PAD_ID=0 expects id 0.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"Saved: {out_path}")

    # Round-trip sanity
    sample = texts[0]
    enc = tok.encode(sample)
    dec = tok.decode(enc.ids)
    print(f"\nRound-trip:\n  in : {sample[:80]}\n  out: {dec.strip()[:80]}")

    # Coverage: overall + per-corpus when shared.
    reports = [coverage_report(tok, texts, args.max_text_tokens, args.corpus)]
    if args.corpus == "shared":
        for d in data_dirs:
            sub = collect_texts([d])
            reports.append(coverage_report(tok, sub, args.max_text_tokens, d.name))

    rec, verdict = recommend_max(reports[0])
    print(f"\nRECOMMENDATION: {verdict}")

    report_path = out_path.parent / f"bpe_{args.vocab_size}_coverage.json"
    with open(report_path, "w") as f:
        json.dump({
            "corpus": args.corpus,
            "vocab_size": args.vocab_size,
            "bpe_path": str(out_path),
            "recommended_max_text_tokens": rec,
            "recommendation": verdict,
            "reports": reports,
        }, f, indent=2)
    print(f"Wrote coverage report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
