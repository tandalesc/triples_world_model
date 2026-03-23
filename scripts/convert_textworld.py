#!/usr/bin/env python3
"""Convert TextWorld KG dataset to chain format for TWM training.

Each example has:
  - observation: natural language game text
  - previous_action: player action
  - previous_triplets: current KG state as [[entity, loc, relation], ...]
  - target_commands: KG deltas ("add, entity, property, relation" or "delete, ...")

We reconstruct game trajectories by grouping consecutive steps,
then build chains where each step is the observation text + action.

Usage:
  uv run python scripts/convert_textworld.py --data-dir data/textworld_kg
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def triplets_to_text(triplets: list[list[str]]) -> str:
    """Convert KG triplets to natural language-ish text."""
    if not triplets:
        return ""
    parts = []
    for t in triplets:
        if len(t) == 3:
            # [entity, location, relation] → "entity is relation location"
            parts.append(f"{t[0]} is {t[2]} {t[1]}")
        elif len(t) == 4:
            # [entity, location1, location2, relation] → "entity relation location1 location2"
            parts.append(f"{t[0]} {t[3]} {t[1]} to {t[2]}")
    return ". ".join(parts) + "." if parts else ""


def apply_commands(triplets: list[list[str]], commands: list[str]) -> list[list[str]]:
    """Apply target_commands to get new KG state."""
    # Convert to set of tuples for easy add/delete
    state = set()
    for t in triplets:
        state.add(tuple(t))

    for cmd in commands:
        parts = [p.strip() for p in cmd.split(",")]
        if len(parts) < 3:
            continue
        op = parts[0]
        triple = tuple(parts[1:])
        if op == "add":
            state.add(triple)
        elif op == "delete":
            state.discard(triple)

    return [list(t) for t in sorted(state)]


def build_chains(data_path: str, max_chain_len: int = 6) -> list[dict]:
    """Build text chains from TextWorld KG transitions.

    Each chain step is: "observation. action: previous_action"
    This gives the model both the game text and what happened.
    """
    chains = []

    with open(data_path) as f:
        # Group into game episodes by tracking state continuity
        current_chain = []
        prev_triplets = None

        for line in f:
            ex = json.loads(line)
            obs = ex["observation"].strip()
            action = ex["previous_action"].strip()
            triplets = ex["previous_triplets"]
            commands = ex["target_commands"]

            # New episode detection: restart or empty previous triplets
            if action == "restart" or not triplets:
                # Save current chain if long enough
                if len(current_chain) >= 2:
                    chains.append({"chain": current_chain[:max_chain_len], "mode": 0})
                current_chain = [obs]
                prev_triplets = apply_commands(triplets, commands)
                continue

            # Build step text: observation + action context
            step_text = f"{obs} Action: {action}."
            current_chain.append(step_text)

            # Update state
            prev_triplets = apply_commands(triplets, commands)

            # Cap chain length and start new
            if len(current_chain) >= max_chain_len:
                chains.append({"chain": current_chain, "mode": 0})
                # Start new chain from last state
                current_chain = [current_chain[-1]]

        # Don't forget last chain
        if len(current_chain) >= 2:
            chains.append({"chain": current_chain[:max_chain_len], "mode": 0})

    return chains


def augment_chains(chains: list[dict]) -> list[dict]:
    """Add reverse (query) and identity chains."""
    augmented = list(chains)

    for c in chains:
        # Reverse chain (query mode)
        augmented.append({"chain": c["chain"][::-1], "mode": 1})

    # Identity: sample one step per chain
    for c in chains:
        pick = hash(c["chain"][0]) % len(c["chain"])
        step = c["chain"][pick]
        augmented.append({"chain": [step, step], "mode": 2})

    return augmented


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/textworld_kg")
    parser.add_argument("--out-dir", default="data/textworld_kg")
    parser.add_argument("--max-chain-len", type=int, default=6)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    for split, filename in [("train", "train_dataset.json"), ("test", "test_dataset.json")]:
        print(f"\nProcessing {split}...")
        chains = build_chains(str(data_dir / filename), args.max_chain_len)
        print(f"  Raw chains: {len(chains)}")

        if args.augment:
            chains = augment_chains(chains)
            print(f"  After augmentation: {len(chains)}")

        # Chain length distribution
        from collections import Counter
        lengths = Counter(len(c["chain"]) for c in chains)
        for l, cnt in sorted(lengths.items()):
            print(f"    {l}-step: {cnt}")

        rng = random.Random(args.seed)
        rng.shuffle(chains)

        prefix = "tw_augmented" if args.augment else "tw_chain"
        path = out_dir / f"{prefix}_{split}.jsonl"
        with open(path, "w") as f:
            for c in chains:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  Wrote {path.name}: {len(chains)}")

        # Samples
        for c in chains[:3]:
            mode_names = {0: "advance", 1: "query", 2: "identity"}
            print(f"\n    mode={mode_names[c['mode']]}, steps={len(c['chain'])}")
            for i, s in enumerate(c["chain"]):
                print(f"      {i}: {s[:100]}")


if __name__ == "__main__":
    main()
