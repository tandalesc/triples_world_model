#!/usr/bin/env python3
"""Train a BPE tokenizer on GLUCOSE chain text data.

Usage:
  uv run python scripts/build_glucose_tokenizer.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


def main():
    data_dir = Path("data/glucose")
    out_path = data_dir / "glucose_bpe_tokenizer.json"

    # Collect all text from chain files
    texts = []
    for f in sorted(data_dir.glob("*chain*train*.jsonl")):
        print(f"Reading {f.name}...")
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                for step_text in data["chain"]:
                    texts.append(step_text)

    print(f"Total texts: {len(texts)}")
    print(f"Sample: {texts[0][:80]}")

    # Train tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    special_tokens = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]
    initial_alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'-()/_ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

    trainer = trainers.BpeTrainer(
        vocab_size=2048,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(str(out_path))

    print(f"\nSaved: {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Test encode/decode
    test = "Someone_A feel(s) happy. Someone_A is at Somewhere_A."
    ids = tokenizer.encode(test).ids
    decoded = tokenizer.decode(ids)
    print(f"\nTest: {test}")
    print(f"IDs:  {ids}")
    print(f"Back: {decoded}")
    print(f"Tokens: {tokenizer.encode(test).tokens}")


if __name__ == "__main__":
    main()
