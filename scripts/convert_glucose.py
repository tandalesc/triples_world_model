#!/usr/bin/env python3
"""Convert GLUCOSE causal reasoning annotations to text chains for multi-turn dynamics.

GLUCOSE provides 10 dimensions of causal knowledge per story event:
  Dims 1-5 (preconditions): cause, motivation, location, possession, attribute
  Dims 6-10 (consequences): effect, emotion, location change, possession change, attribute change

Each dimension has a natural-language annotation with a causal connective
(e.g., "X >Causes/Enables> Y"). We extract the LHS/RHS text and build chains:
  precondition texts → event texts → consequence texts

Output format:
  {"chain": ["text_0", "text_1", "text_2"], "mode": 0}

Modes:
  0 = advance:  preconditions → event → consequences
  1 = query:    consequences → event → preconditions (reverse)
  2 = identity: step → step → step (reconstruct)

Usage:
  uv run python scripts/convert_glucose.py --annotation both
  uv run python scripts/convert_glucose.py --annotation general --augment
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PRE_DIMS = [1, 2, 3, 4, 5]
POST_DIMS = [6, 7, 8, 9, 10]

CONNECTIVE_RE = re.compile(
    r"\s*>(Causes/Enables|Causes|Enables|Motivates|Results in)>\s*"
)


def split_connective(text: str) -> tuple[str, str] | None:
    """Split 'LHS >Connective> RHS' into (lhs, rhs) text."""
    m = CONNECTIVE_RE.search(text)
    if not m:
        return None
    lhs = text[: m.start()].strip()
    rhs = text[m.end() :].strip()
    if lhs and rhs:
        return (lhs, rhs)
    return None


def process_row(row: dict, annotation: str) -> dict | None:
    """Extract precondition/consequence texts from a GLUCOSE row.

    Returns {"pre_texts": [...], "post_texts": [...], "event_texts": [...]}
    or None if insufficient data.
    """
    suffix = "NL"
    mode = "specific" if annotation == "specific" else "general"

    pre_texts = []   # LHS of precondition dims (causes)
    event_texts = []  # RHS of pre dims / LHS of post dims (the event)
    post_texts = []   # RHS of consequence dims (effects)

    seen_pre = set()
    seen_event = set()
    seen_post = set()

    for dim in PRE_DIMS:
        key = f"{dim}_{mode}{suffix}"
        val = row.get(key, "")
        if not val or val == "escaped":
            continue
        pair = split_connective(val)
        if pair:
            lhs, rhs = pair
            if lhs not in seen_pre:
                pre_texts.append(lhs)
                seen_pre.add(lhs)
            if rhs not in seen_event:
                event_texts.append(rhs)
                seen_event.add(rhs)

    for dim in POST_DIMS:
        key = f"{dim}_{mode}{suffix}"
        val = row.get(key, "")
        if not val or val == "escaped":
            continue
        pair = split_connective(val)
        if pair:
            lhs, rhs = pair
            if lhs not in seen_event:
                event_texts.append(lhs)
                seen_event.add(lhs)
            if rhs not in seen_post:
                post_texts.append(rhs)
                seen_post.add(rhs)

    if not pre_texts or not event_texts or not post_texts:
        return None

    return {
        "pre_texts": pre_texts,
        "event_texts": event_texts,
        "post_texts": post_texts,
    }


def texts_to_sentence(texts: list[str]) -> str:
    """Join multiple annotation texts into a single sentence."""
    return ". ".join(t.rstrip(".") for t in texts) + "."


def generate_chain(processed: dict) -> dict:
    """Generate a 3-step text chain (advance mode)."""
    return {
        "chain": [
            texts_to_sentence(processed["pre_texts"]),
            texts_to_sentence(processed["event_texts"]),
            texts_to_sentence(processed["post_texts"]),
        ],
        "mode": 0,
    }


def generate_augmented(processed: dict) -> list[dict]:
    """Generate advance + query + identity chains from one example."""
    pre = texts_to_sentence(processed["pre_texts"])
    event = texts_to_sentence(processed["event_texts"])
    post = texts_to_sentence(processed["post_texts"])

    examples = []
    # Advance: pre → event → post
    examples.append({"chain": [pre, event, post], "mode": 0})
    # Query: post → event → pre (reverse causal chain)
    examples.append({"chain": [post, event, pre], "mode": 1})
    # Identity: reconstruct one step (rotate through to balance)
    steps = [pre, event, post]
    pick = hash(pre) % 3  # deterministic but varied
    examples.append({"chain": [steps[pick], steps[pick]], "mode": 2})
    return examples


def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Convert GLUCOSE to text chains")
    parser.add_argument(
        "--csv",
        default="data/glucose/GLUCOSE_training_data_final.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/glucose",
    )
    parser.add_argument(
        "--annotation",
        choices=["specific", "general", "both"],
        default="both",
    )
    parser.add_argument("--min-quality", type=int, default=2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment", action="store_true",
        help="Generate advance + query + identity augmentations",
    )
    parser.add_argument(
        "--long-chains", action="store_true",
        help="Build longer chains from multi-sentence stories",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    annotations = (
        ["specific", "general"] if args.annotation == "both" else [args.annotation]
    )

    for annotation in annotations:
        print(f"\nProcessing {annotation} annotations...")

        chains = []
        total = 0
        skipped = 0

        # Group by story + sentence index for long chains
        from collections import defaultdict
        story_events: dict[str, dict[int, dict]] = defaultdict(dict)

        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                quality = int(row.get("worker_quality_rating", "0") or "0")
                if quality < args.min_quality:
                    skipped += 1
                    continue

                processed = process_row(row, annotation)
                if not processed:
                    continue

                if args.augment:
                    chains.extend(generate_augmented(processed))
                else:
                    chains.append(generate_chain(processed))

                # Collect for long chains
                if args.long_chains:
                    story_id = row.get("story_id", "")
                    sent_idx = int(row.get("selected_sentence_index", "0") or "0")
                    if story_id and sent_idx > 0:
                        # Keep the best quality annotation per story+sentence
                        existing = story_events[story_id].get(sent_idx)
                        if existing is None or quality > existing.get("_quality", 0):
                            processed["_quality"] = quality
                            story_events[story_id][sent_idx] = processed

        # Build long chains from story-level event sequences
        if args.long_chains:
            long_count = 0
            for story_id, events in story_events.items():
                if len(events) < 2:
                    continue
                # Sort by sentence index
                sorted_idxs = sorted(events.keys())
                # Build chain: event_text from each sentence in order
                steps = []
                for idx in sorted_idxs:
                    ev = events[idx]
                    steps.append(texts_to_sentence(ev["event_texts"]))
                if len(steps) >= 3:
                    chains.append({"chain": steps, "mode": 0})
                    # Reverse long chain for query mode
                    if args.augment:
                        chains.append({"chain": steps[::-1], "mode": 1})
                    long_count += 1
            print(f"  Long chains (3+ steps): {long_count}")

        print(f"  Total rows: {total}, skipped (quality < {args.min_quality}): {skipped}")

        # Shuffle and split
        rng = random.Random(args.seed)
        prefix = "augmented_chain" if args.augment else "chain"

        rng.shuffle(chains)
        n_test = int(len(chains) * args.test_frac)
        write_jsonl(chains[n_test:], out_dir / f"{prefix}_{annotation}_train.jsonl")
        write_jsonl(chains[:n_test], out_dir / f"{prefix}_{annotation}_test.jsonl")

        # Print mode distribution
        if args.augment:
            from collections import Counter
            mode_counts = Counter(c["mode"] for c in chains)
            mode_names = {0: "advance", 1: "query", 2: "identity"}
            for m, cnt in sorted(mode_counts.items()):
                print(f"  {mode_names[m]}: {cnt}")

        # Print samples
        if chains:
            print(f"\n  Sample ({annotation}):")
            for c in chains[:3]:
                mode_names = {0: "advance", 1: "query", 2: "identity"}
                print(f"    mode={mode_names[c['mode']]}, steps={len(c['chain'])}")
                for i, step in enumerate(c["chain"]):
                    print(f"      {i}: {step[:80]}")


if __name__ == "__main__":
    main()
