#!/usr/bin/env python3
"""Convert GLUCOSE causal reasoning annotations to text chains for multi-turn dynamics.

GLUCOSE provides 10 dimensions of causal knowledge per story event:
  Dims 1-5 (preconditions): cause, motivation, location, possession, attribute
  Dims 6-10 (consequences): effect, emotion, location change, possession change, attribute change

Each dimension has a natural-language annotation with a causal connective
(e.g., "X >Causes/Enables> Y"). We extract the LHS/RHS text and build chains:
  precondition texts → event texts → consequence texts

Output format:
  {"chain": ["precondition sentence", "event sentence", "consequence sentence"]}

Usage:
  uv run python scripts/convert_glucose.py --annotation both
  uv run python scripts/convert_glucose.py --annotation general
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
    """Generate a 3-step text chain."""
    return {
        "chain": [
            texts_to_sentence(processed["pre_texts"]),
            texts_to_sentence(processed["event_texts"]),
            texts_to_sentence(processed["post_texts"]),
        ]
    }


def generate_pairs(processed: dict) -> list[dict]:
    """Generate single-turn text pairs (for baseline comparison)."""
    pairs = []
    pre = texts_to_sentence(processed["pre_texts"])
    event = texts_to_sentence(processed["event_texts"])
    post = texts_to_sentence(processed["post_texts"])
    # pre → event
    pairs.append({"text_t": pre, "text_t1": event})
    # event → post
    pairs.append({"text_t": event, "text_t1": post})
    return pairs


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    annotations = (
        ["specific", "general"] if args.annotation == "both" else [args.annotation]
    )

    for annotation in annotations:
        print(f"\nProcessing {annotation} annotations...")

        chains = []
        pairs = []
        total = 0
        skipped = 0

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

                chains.append(generate_chain(processed))
                pairs.extend(generate_pairs(processed))

        print(f"  Total rows: {total}, skipped (quality < {args.min_quality}): {skipped}")

        # Shuffle and split
        rng = random.Random(args.seed)

        for name, data in [("chain", chains), ("pair", pairs)]:
            rng.shuffle(data)
            n_test = int(len(data) * args.test_frac)
            write_jsonl(data[n_test:], out_dir / f"{name}_{annotation}_train.jsonl")
            write_jsonl(data[:n_test], out_dir / f"{name}_{annotation}_test.jsonl")

        # Print samples
        if chains:
            print(f"\n  Sample chain ({annotation}):")
            c = chains[0]["chain"]
            for i, step in enumerate(c):
                print(f"    step {i}: {step[:100]}")

        if pairs:
            print(f"\n  Sample pair ({annotation}):")
            p = pairs[0]
            print(f"    in:  {p['text_t'][:100]}")
            print(f"    out: {p['text_t1'][:100]}")


if __name__ == "__main__":
    main()
