#!/usr/bin/env python3
"""Build a 512-token domain BPE on GLUCOSE chain text -> data/glucose/jepa_bpe_512.json.

Usage (from repo root):
    uv run python scripts/build_jepa_bpe.py

Spec reference: jepa_operator_v1_design.md §6.
The 512-vocab BPE covers GLUCOSE's placeholder-heavy, low-entropy surface forms
(Someone_A, Something_B, common verbs) and keeps token_emb at 512·d.
max_text_tokens=64 is verified here: reports tokens-per-state distribution and
the fraction of states exceeding 64 tokens.
"""

import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


DATA_DIR = Path("data/glucose")
OUT_PATH = DATA_DIR / "jepa_bpe_512.json"
VOCAB_SIZE = 512
MAX_TEXT_TOKENS = 64
SPECIAL_TOKENS = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]
# PAD_ID = 0 (position of <pad> in special_tokens list)


def collect_texts(data_dir: Path) -> list[str]:
    """Collect all state texts from GLUCOSE chain_general_train.jsonl."""
    texts: list[str] = []
    # Primary training file
    train_path = data_dir / "chain_general_train.jsonl"
    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                data = json.loads(line)
                for step_text in data["chain"]:
                    texts.append(step_text)
        print(f"  Read {train_path.name}: {len(texts)} state texts so far")

    # Also include test file if present (for fuller vocabulary coverage at build time)
    test_path = data_dir / "chain_general_test.jsonl"
    if test_path.exists():
        n_before = len(texts)
        with open(test_path) as f:
            for line in f:
                data = json.loads(line)
                for step_text in data["chain"]:
                    texts.append(step_text)
        print(f"  Read {test_path.name}: +{len(texts) - n_before} state texts")

    return texts


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    """Train a BPE tokenizer on the provided texts."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel pre-tokenization: handles arbitrary Unicode via byte fallback,
    # preserves placeholder patterns like Someone_A / Something_B as single runs.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Characters that appear frequently in GLUCOSE text; ensures they are kept
    # as single-character tokens (not decomposed to bytes) from the start.
    initial_alphabet = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,;:!?'\"()-/_@# "
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    return tokenizer


def coverage_report(tokenizer: Tokenizer, texts: list[str], max_tokens: int) -> None:
    """Print token-count distribution and coverage stats for the given texts."""
    lengths: list[int] = []
    for text in texts:
        enc = tokenizer.encode(text)
        lengths.append(len(enc.ids))

    n = len(lengths)
    n_over = sum(1 for l in lengths if l > max_tokens)
    pct_over = 100.0 * n_over / n if n > 0 else 0.0

    sorted_lens = sorted(lengths)
    p50 = sorted_lens[n // 2]
    p90 = sorted_lens[int(n * 0.90)]
    p95 = sorted_lens[int(n * 0.95)]
    p99 = sorted_lens[int(n * 0.99)]
    mean = statistics.mean(lengths)

    print(f"\n=== Coverage report (vocab={tokenizer.get_vocab_size()}, max_tokens={max_tokens}) ===")
    print(f"  States analysed : {n:,}")
    print(f"  Mean tokens     : {mean:.1f}")
    print(f"  P50  tokens     : {p50}")
    print(f"  P90  tokens     : {p90}")
    print(f"  P95  tokens     : {p95}")
    print(f"  P99  tokens     : {p99}")
    print(f"  Max  tokens     : {max(lengths)}")
    print(f"  States > {max_tokens} tok : {n_over:,} / {n:,}  ({pct_over:.1f}%)")

    # Histogram buckets: 0-15, 16-31, 32-47, 48-63, 64-79, 80+
    buckets = [(0, 15), (16, 31), (32, 47), (48, 63), (64, 79), (80, 999)]
    print("  Distribution:")
    for lo, hi in buckets:
        cnt = sum(1 for l in lengths if lo <= l <= hi)
        bar = "#" * (cnt * 40 // n)
        label = f"  [{lo:3d}-{hi:3d}]" if hi < 999 else f"  [{lo:3d}+    ]"
        print(f"    {label}: {cnt:5d} ({100*cnt/n:5.1f}%) {bar}")


def main() -> None:
    print("Building JEPA domain BPE (vocab=512) on GLUCOSE chain text...")
    print(f"  Output: {OUT_PATH}")

    texts = collect_texts(DATA_DIR)
    if not texts:
        raise RuntimeError(
            f"No texts found under {DATA_DIR}. "
            "Run from the repo root and ensure data/glucose/chain_general_train.jsonl exists."
        )
    print(f"  Total state texts for training: {len(texts):,}")

    tokenizer = build_tokenizer(texts, VOCAB_SIZE)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"\nTrained tokenizer — actual vocab size: {actual_vocab}")

    # Verify special token IDs match the convention in domain_bpe.py (PAD=0)
    pad_id = tokenizer.token_to_id("<pad>")
    mask_id = tokenizer.token_to_id("<mask>")
    unk_id = tokenizer.token_to_id("<unk>")
    print(f"  <pad>={pad_id}  <mask>={mask_id}  <unk>={unk_id}")
    if pad_id != 0:
        print(
            f"  WARNING: <pad> got id={pad_id} (expected 0). "
            "DomainBPETokenizer.PAD_ID=0 hardcodes id 0 — this mismatch means "
            "padding masks built from `ids == 0` will be incorrect. "
            "Consider re-ordering SPECIAL_TOKENS or patching PAD_ID."
        )

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_PATH))
    print(f"\nSaved: {OUT_PATH}")

    # Round-trip sanity check
    test_text = "Someone_A puts Something_A on Something_B. Someone_A feel(s) happy."
    enc = tokenizer.encode(test_text)
    decoded = tokenizer.decode(enc.ids)
    print(f"\nRound-trip check:")
    print(f"  Input   : {test_text}")
    print(f"  Tokens  : {enc.tokens[:12]}{'...' if len(enc.tokens) > 12 else ''}")
    print(f"  IDs     : {enc.ids[:12]}{'...' if len(enc.ids) > 12 else ''}")
    print(f"  Decoded : {decoded.strip()}")

    # Coverage report on training texts only
    coverage_report(tokenizer, texts, MAX_TEXT_TOKENS)

    print("\nDone.")


if __name__ == "__main__":
    main()
