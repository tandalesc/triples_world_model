#!/usr/bin/env python3
"""Build a BPE tokenizer from ATOMIC training data.

Scans all entity and value strings, trains a BPE tokenizer with the
`tokenizers` library, and saves it for use with the domain vocab decoder.

Usage:
    uv run python scripts/build_domain_vocab.py \
        --data-dir data/atomic_10000 \
        --vocab-size 1500 \
        --out-dir data/atomic_10000
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def collect_phrases(data_dir: Path) -> list[str]:
    """Collect all entity and value strings from training data."""
    phrases = []
    train_path = data_dir / "train.jsonl"
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            for t in ex.get("state_t", []) + ex.get("state_t+1", []):
                phrases.append(t[0])  # entity
                phrases.append(t[2])  # value
    return phrases


def main():
    ap = argparse.ArgumentParser(description="Build domain BPE tokenizer")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--vocab-size", type=int, default=1500)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    phrases = collect_phrases(data_dir)
    unique = sorted(set(phrases))
    print(f"Collected {len(phrases)} phrases ({len(unique)} unique)")

    # Normalize: replace underscores with spaces for BPE training
    normalized = [p.replace("_", " ").lower().strip() for p in unique]

    # Train BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(normalized, trainer=trainer)

    # Set post-processor for padding support
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

    out_path = out_dir / "domain_bpe_tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Report statistics
    from collections import Counter
    all_lengths_entity = []
    all_lengths_value = []

    train_path = data_dir / "train.jsonl"
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            for t in ex.get("state_t", []) + ex.get("state_t+1", []):
                entity = t[0].replace("_", " ").lower().strip()
                value = t[2].replace("_", " ").lower().strip()
                e_enc = tokenizer.encode(entity)
                v_enc = tokenizer.encode(value)
                all_lengths_entity.append(len(e_enc.ids))
                all_lengths_value.append(len(v_enc.ids))

    print(f"\nEntity token lengths: avg={sum(all_lengths_entity)/len(all_lengths_entity):.1f}, "
          f"max={max(all_lengths_entity)}, p95={sorted(all_lengths_entity)[int(len(all_lengths_entity)*0.95)]}")
    print(f"Value token lengths:  avg={sum(all_lengths_value)/len(all_lengths_value):.1f}, "
          f"max={max(all_lengths_value)}, p95={sorted(all_lengths_value)[int(len(all_lengths_value)*0.95)]}")

    # Suggest max_tokens
    max_tok = max(max(all_lengths_entity), max(all_lengths_value))
    suggested = min(max_tok + 2, 16)
    print(f"\nSuggested max_tokens: {suggested} (max observed: {max_tok})")

    # Show some example encodings
    print("\nExample encodings:")
    samples = unique[:10]
    for phrase in samples:
        norm = phrase.replace("_", " ").lower().strip()
        enc = tokenizer.encode(norm)
        dec = tokenizer.decode(enc.ids)
        print(f"  {phrase[:40]:<40s} → {len(enc.ids)} tokens: {enc.tokens[:8]}{'...' if len(enc.tokens) > 8 else ''}")
        print(f"    decode: '{dec}'")


if __name__ == "__main__":
    main()
