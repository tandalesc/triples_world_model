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


# Section orderings that suggest sequential/causal structure
BIOGRAPHICAL_SECTIONS = {
    "early life", "childhood", "education", "early career", "career",
    "personal life", "later life", "death", "legacy", "awards",
    "filmography", "discography", "bibliography", "works",
}

EVENT_SECTIONS = {
    "background", "causes", "prelude", "events", "battle", "aftermath",
    "consequences", "legacy", "reactions", "impact", "timeline",
}

PROCESS_SECTIONS = {
    "history", "development", "design", "production", "release",
    "reception", "legacy", "influence",
}

# Sections to skip
SKIP_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "sources", "citations", "bibliography",  # as a references section
}


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Parse Wikipedia article text into (section_title, section_text) pairs.

    Handles == Section == and === Subsection === markup.
    """
    sections = []
    current_title = "Introduction"
    current_text = []

    for line in text.split("\n"):
        # Match section headers: == Title == or === Title ===
        m = re.match(r"^=+\s*(.+?)\s*=+$", line.strip())
        if m:
            # Save previous section
            text_joined = " ".join(current_text).strip()
            if text_joined and len(text_joined) > 50:
                sections.append((current_title.lower(), text_joined))
            current_title = m.group(1)
            current_text = []
        else:
            if line.strip():
                current_text.append(line.strip())

    # Last section
    text_joined = " ".join(current_text).strip()
    if text_joined and len(text_joined) > 50:
        sections.append((current_title.lower(), text_joined))

    return sections


def is_sequential(sections: list[tuple[str, str]]) -> bool:
    """Check if sections suggest a sequential/temporal ordering."""
    titles = {t for t, _ in sections}
    bio_overlap = len(titles & BIOGRAPHICAL_SECTIONS)
    event_overlap = len(titles & EVENT_SECTIONS)
    process_overlap = len(titles & PROCESS_SECTIONS)
    return max(bio_overlap, event_overlap, process_overlap) >= 2


def sections_to_chain(sections: list[tuple[str, str]], max_steps: int = 6,
                       max_text_len: int = 300) -> list[str]:
    """Convert sections to chain steps, truncating long sections."""
    steps = []
    for title, text in sections:
        if title in SKIP_SECTIONS:
            continue
        # Truncate long sections to first few sentences
        if len(text) > max_text_len:
            sentences = text.split(". ")
            truncated = ""
            for s in sentences:
                if len(truncated) + len(s) > max_text_len:
                    break
                truncated += s + ". "
            text = truncated.strip()
        if text:
            steps.append(f"{title.title()}: {text}")
        if len(steps) >= max_steps:
            break
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=50000)
    parser.add_argument("--out-dir", default="data/wikipedia")
    parser.add_argument("--min-sections", type=int, default=3)
    parser.add_argument("--max-chain-len", type=int, default=6)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequential-only", action="store_true",
                        help="Only keep articles with sequential section structure")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Wikipedia dataset...")
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en", split="train",
                      streaming=True, trust_remote_code=True)

    chains = []
    n_processed = 0
    n_sequential = 0

    for article in ds:
        if n_processed >= args.max_articles:
            break
        n_processed += 1

        if n_processed % 10000 == 0:
            print(f"  Processed {n_processed}, chains: {len(chains)}")

        text = article["text"]
        title = article.get("title", "")

        sections = parse_sections(text)
        if len(sections) < args.min_sections:
            continue

        if args.sequential_only and not is_sequential(sections):
            continue
        n_sequential += 1

        steps = sections_to_chain(sections, max_steps=args.max_chain_len)
        if len(steps) >= args.min_sections:
            chains.append({"chain": steps, "mode": 0})
            # Also add reverse
            chains.append({"chain": steps[::-1], "mode": 1})

    print(f"\nProcessed: {n_processed}")
    print(f"Sequential: {n_sequential}")
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
