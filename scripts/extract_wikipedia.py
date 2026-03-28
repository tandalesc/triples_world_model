#!/usr/bin/env python3
"""Extract sequential chains from Wikipedia articles.

Frames Wikipedia sections as state transitions:
  "Early life" → "Career" → "Major works" → "Legacy"

Each section becomes a state in a chain. Filters for articles with
clear sequential structure (biographies, events, processes).

Outputs chains compatible with train_chain.py.

Usage:
  uv run python scripts/extract_wikipedia.py --max-articles 50000
  uv run python scripts/extract_wikipedia.py --max-articles 50000 --category biography
"""

import argparse
import json
import random
import re
from pathlib import Path



def parse_paragraphs(text: str, min_len: int = 50, max_len: int = 300) -> list[str]:
    """Split article text into paragraph chunks.

    The wikimedia dataset has headers stripped — text is just paragraphs
    separated by newlines. We split on double newlines and filter/truncate.
    """
    paragraphs = []
    for para in re.split(r"\n\n+", text):
        para = para.strip()
        if len(para) < min_len:
            continue
        # Truncate long paragraphs to first few sentences
        if len(para) > max_len:
            sentences = para.split(". ")
            truncated = ""
            for s in sentences:
                if len(truncated) + len(s) > max_len:
                    break
                truncated += s + ". "
            para = truncated.strip()
        if para:
            paragraphs.append(para)
    return paragraphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=50000)
    parser.add_argument("--out-dir", default="data/wikipedia")
    parser.add_argument("--min-sections", type=int, default=3)
    parser.add_argument("--max-chain-len", type=int, default=6)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Wikipedia dataset...")
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                      streaming=True)

    chains = []
    n_processed = 0

    for article in ds:
        if n_processed >= args.max_articles:
            break
        n_processed += 1

        if n_processed % 10000 == 0:
            print(f"  Processed {n_processed}, chains: {len(chains)}")

        text = article["text"]

        paragraphs = parse_paragraphs(text)
        if len(paragraphs) < args.min_sections:
            continue

        steps = paragraphs[:args.max_chain_len]
        # Continue mode: all prefix sub-chains of length 2+
        for end in range(2, len(steps) + 1):
            chains.append({"chain": steps[:end], "mode": 0})
        # Identity: 1 per article (first paragraph, 3-step)
        chains.append({"chain": [steps[0], steps[0], steps[0]], "mode": 2})

    print(f"\nProcessed: {n_processed}")
    print(f"Chains: {len(chains)}")

    # Chain length distribution
    from collections import Counter
    lengths = Counter(len(c["chain"]) for c in chains)
    for l, cnt in sorted(lengths.items()):
        print(f"  {l}-step: {cnt}")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(chains)
    n_test = int(len(chains) * args.test_frac)

    train = chains[n_test:]
    test = chains[:n_test]

    for name, data in [("wiki_train", train), ("wiki_test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(data)}")

    # Samples
    for c in train[:3]:
        print(f"\nChain ({len(c['chain'])} steps):")
        for i, s in enumerate(c["chain"]):
            print(f"  {i}: {s[:100]}")


if __name__ == "__main__":
    main()
