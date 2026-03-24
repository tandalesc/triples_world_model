#!/usr/bin/env python3
"""Convert GLUCOSE stories to next-state prediction format.

Each story becomes a sequence of sentence-states:
  {"states": ["sentence 1", "sentence 2", "sentence 3", ...]}

The dynamics core learns to predict state_{t+1} from state_t (or a window
of previous states), with no mode labels — the transformation is inferred
from context.

Usage:
  uv run python scripts/convert_glucose_nsp.py
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def extract_stories(csv_path: str) -> list[list[str]]:
    """Extract unique stories as sentence sequences."""
    stories = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            story_id = row.get("story_id", "")
            story_text = row.get("story", "")
            if story_id and story_text and story_id not in stories:
                stories[story_id] = story_text

    sequences = []
    for text in stories.values():
        # GLUCOSE uses **** as sentence separator in some fields
        sents = [s.strip() for s in text.replace("****", ".").split(".") if s.strip()]
        if len(sents) >= 2:
            sequences.append(sents)

    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/glucose/GLUCOSE_training_data_final.csv")
    parser.add_argument("--out-dir", default="data/glucose")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stories = extract_stories(args.csv)
    print(f"Stories: {len(stories)}")
    print(f"Avg length: {sum(len(s) for s in stories)/len(stories):.1f} sentences")

    # Convert to JSONL
    examples = [{"states": sents} for sents in stories]

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_test = int(len(examples) * args.test_frac)

    train = examples[n_test:]
    test = examples[:n_test]

    for name, data in [("nsp_train", train), ("nsp_test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {path.name}: {len(data)} examples")

    # Print samples
    for ex in train[:3]:
        print(f"\n  Story ({len(ex['states'])} sentences):")
        for i, s in enumerate(ex["states"]):
            print(f"    {i}: {s[:80]}")


if __name__ == "__main__":
    main()
