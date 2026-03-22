#!/usr/bin/env python3
"""Convert Playground Series S5E7 (Extrovert vs Introvert) CSV to TWM triples.

Attributes: time_alone, stage_fear, social_events, going_outside,
            drained_social, friends, post_freq
Target: personality (extrovert/introvert)
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# --- Discretization ---

def time_alone_bin(val: float) -> str:
    if val <= 1:
        return "low"
    elif val <= 3:
        return "moderate"
    elif val <= 5:
        return "high"
    else:
        return "very_high"


def social_events_bin(val: float) -> str:
    if val <= 2:
        return "rarely"
    elif val <= 5:
        return "sometimes"
    elif val <= 7:
        return "often"
    else:
        return "very_often"


def going_outside_bin(val: float) -> str:
    if val <= 1:
        return "rarely"
    elif val <= 3:
        return "sometimes"
    elif val <= 5:
        return "often"
    else:
        return "very_often"


def friends_bin(val: float) -> str:
    if val <= 3:
        return "few"
    elif val <= 7:
        return "some"
    elif val <= 12:
        return "many"
    else:
        return "lots"


def post_freq_bin(val: float) -> str:
    if val <= 1:
        return "rarely"
    elif val <= 4:
        return "sometimes"
    elif val <= 7:
        return "often"
    else:
        return "very_often"


# --- Triple builders ---

MODE_ADVANCE = ["#mode", "type", "advance"]

ATTR_KEYS = [
    "time_alone", "stage_fear", "social_events", "going_outside",
    "drained_social", "friends", "post_freq",
]


def row_to_attrs(row: dict) -> dict:
    attrs = {}

    if row.get("Time_spent_Alone"):
        attrs["time_alone"] = time_alone_bin(float(row["Time_spent_Alone"]))
    if row.get("Stage_fear"):
        attrs["stage_fear"] = row["Stage_fear"].lower()
    if row.get("Social_event_attendance"):
        attrs["social_events"] = social_events_bin(float(row["Social_event_attendance"]))
    if row.get("Going_outside"):
        attrs["going_outside"] = going_outside_bin(float(row["Going_outside"]))
    if row.get("Drained_after_socializing"):
        attrs["drained_social"] = row["Drained_after_socializing"].lower()
    if row.get("Friends_circle_size"):
        attrs["friends"] = friends_bin(float(row["Friends_circle_size"]))
    if row.get("Post_frequency"):
        attrs["post_freq"] = post_freq_bin(float(row["Post_frequency"]))

    return attrs


def attrs_to_triples(attrs: dict, include_target: bool = False) -> list[list[str]]:
    triples = [["person", k, attrs[k]] for k in ATTR_KEYS if k in attrs]
    if include_target:
        triples.append(["person", "personality", attrs["personality"]])
    return triples


def make_advance(attrs: dict) -> dict:
    state_t = [MODE_ADVANCE] + attrs_to_triples(attrs, include_target=False)
    state_t1 = attrs_to_triples(attrs, include_target=True)
    return {"state_t": state_t, "state_t+1": state_t1}


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


# --- Main ---

def main():
    src = Path("data/playground-series-s5e7/train.csv")
    out_dir = Path("data/playground-series-s5e7")

    with open(src) as f:
        rows = list(csv.DictReader(f))

    all_attrs = []
    for row in rows:
        if not row.get("Personality"):
            continue
        attrs = row_to_attrs(row)
        attrs["personality"] = row["Personality"].lower()
        all_attrs.append(attrs)

    train_examples = [make_advance(attrs) for attrs in all_attrs]
    random.shuffle(train_examples)

    print(f"Personality → TWM Triples")
    print(f"  Total rows: {len(rows)}")
    print(f"  Valid rows: {len(all_attrs)}")
    print()
    write_jsonl(out_dir / "train.jsonl", train_examples)

    # Stats
    max_in = max_out = 0
    tokens = set()
    for line in open(out_dir / "train.jsonl"):
        ex = json.loads(line)
        max_in = max(max_in, len(ex["state_t"]))
        max_out = max(max_out, len(ex["state_t+1"]))
        for t in ex["state_t"] + ex["state_t+1"]:
            tokens.update(t)

    print(f"\n  Train: {len(train_examples)}")
    print(f"\n  Max input triples: {max_in}")
    print(f"  Max output triples: {max_out}")
    print(f"  Unique tokens: {len(tokens)}")
    print(f"  Tokens: {sorted(tokens)}")

    print("\n  Value distributions:")
    for attr in ATTR_KEYS + ["personality"]:
        counts = Counter(a.get(attr, "<missing>") for a in all_attrs)
        dist = ", ".join(f"{v}={c}" for v, c in counts.most_common())
        print(f"    {attr}: {dist}")


if __name__ == "__main__":
    main()
