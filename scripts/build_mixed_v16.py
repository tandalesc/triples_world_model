#!/usr/bin/env python3
"""Build mixed_v16 dataset: Wikipedia (sub-chains) + TextWorld.

Extracts Wikipedia with all prefix sub-chains, merges with TextWorld,
and writes train/test splits.

Usage:
  uv run python scripts/build_mixed_v16.py
"""

import json
import random
from pathlib import Path


def main():
    data_dir = Path("data")
    out_train = data_dir / "mixed_v16_train.jsonl"
    out_test = data_dir / "mixed_v16_test.jsonl"
    seed = 42
    test_frac = 0.1

    chains = []

    # 1. TextWorld chains (all modes)
    tw_path = data_dir / "textworld_kg" / "tw_augmented_train.jsonl"
    tw_test_path = data_dir / "textworld_kg" / "tw_augmented_test.jsonl"
    tw_count = 0
    for p in [tw_path, tw_test_path]:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        with open(p) as f:
            for line in f:
                chains.append(json.loads(line))
                tw_count += 1
    print(f"TextWorld: {tw_count} chains")

    # 2. Wikipedia chains (from pre-extracted files)
    wiki_train = data_dir / "wikipedia" / "wiki_train.jsonl"
    wiki_test = data_dir / "wikipedia" / "wiki_test.jsonl"
    wiki_count = 0
    for p in [wiki_train, wiki_test]:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        with open(p) as f:
            for line in f:
                chains.append(json.loads(line))
                wiki_count += 1
    print(f"Wikipedia: {wiki_count} chains")

    print(f"Total: {len(chains)} chains")

    # Mode distribution
    mode_counts = {}
    for c in chains:
        m = c.get("mode", "?")
        mode_counts[m] = mode_counts.get(m, 0) + 1
    for m, cnt in sorted(mode_counts.items()):
        print(f"  mode {m}: {cnt} ({100*cnt/len(chains):.1f}%)")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(chains)
    n_test = int(len(chains) * test_frac)
    test = chains[:n_test]
    train = chains[n_test:]

    for path, data in [(out_train, train), (out_test, test)]:
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {path}: {len(data)}")


if __name__ == "__main__":
    main()
