#!/usr/bin/env python3
"""Convert Spaceship Titanic CSV to TWM triple-transition JSONL.

Frames tabular classification as state transition:
  state_t: passenger attributes + mode
  state_t+1: same attributes + transported prediction

Attributes (all optional — missing fields are simply omitted as triples):
  home_planet, cryo_sleep, destination, age_group,
  room_service, food_court, shopping, spa, vrdeck,
  deck, side, cabin_region, group_size
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# --- Discretization ---

def age_group(age: float) -> str:
    if age <= 12:
        return "child"
    elif age <= 17:
        return "teen"
    elif age <= 30:
        return "young_adult"
    elif age <= 50:
        return "adult"
    else:
        return "senior"


def spend_bin(val: str) -> str | None:
    """Bin a single spending column: zero vs nonzero is the real signal."""
    if not val:
        return None
    return "zero" if float(val) == 0 else "nonzero"


def cabin_region(num_str: str) -> str:
    """Bin cabin number into regions."""
    n = int(num_str)
    if n <= 300:
        return "forward"
    elif n <= 600:
        return "midship"
    else:
        return "aft"


def parse_cabin(cabin: str) -> dict:
    """Parse 'B/0/P' -> dict with deck, side, cabin_region."""
    if not cabin:
        return {}
    parts = cabin.split("/")
    if len(parts) != 3:
        return {}
    result = {
        "deck": parts[0].lower(),
        "side": "port" if parts[2] == "P" else "starboard",
        "cabin_region": cabin_region(parts[1]),
    }
    return result


DESTINATION_MAP = {
    "TRAPPIST-1e": "trappist-1e",
    "PSO J318.5-22": "pso-j318",
    "55 Cancri e": "55-cancri-e",
}

SPENDING_COLS = [
    ("RoomService", "room_service"),
    ("FoodCourt", "food_court"),
    ("ShoppingMall", "shopping"),
    ("Spa", "spa"),
    ("VRDeck", "vrdeck"),
]

# --- Compositional holdout combos ---

HOLDOUT_COMBOS = [
    {"home_planet": "europa", "cryo_sleep": "true", "destination": "55-cancri-e"},
    {"home_planet": "earth", "age_group": "senior", "room_service": "high"},
    {"home_planet": "mars", "deck": "d"},
]


def matches_holdout(attrs: dict) -> bool:
    for combo in HOLDOUT_COMBOS:
        if all(attrs.get(k) == v for k, v in combo.items()):
            return True
    return False


# --- Triple builders ---

MODE_ADVANCE = ["#mode", "type", "advance"]
MODE_IDENTITY = ["#mode", "type", "identity"]

# Order for consistent triple output
ATTR_ORDER = [
    "home_planet", "cryo_sleep", "destination", "age_group",
    "room_service", "food_court", "shopping", "spa", "vrdeck",
    "deck", "side", "cabin_region", "group_size",
]


def attrs_to_triples(attrs: dict, include_transported: bool = False) -> list[list[str]]:
    triples = [["passenger", k, attrs[k]] for k in ATTR_ORDER if k in attrs]
    if include_transported:
        triples.append(["passenger", "transported", attrs["transported"]])
    return triples


def make_advance(attrs: dict) -> dict:
    state_t = [MODE_ADVANCE] + attrs_to_triples(attrs, include_transported=False)
    state_t1 = attrs_to_triples(attrs, include_transported=True)
    return {"state_t": state_t, "state_t+1": state_t1}


def make_identity(attrs: dict) -> dict:
    triples = attrs_to_triples(attrs, include_transported=True)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


def row_to_attrs(row: dict, group_counts: dict) -> dict | None:
    """Convert CSV row to attribute dict. Only returns None if Transported is missing."""
    attrs = {}

    if row.get("HomePlanet"):
        attrs["home_planet"] = row["HomePlanet"].lower()
    if row.get("CryoSleep"):
        attrs["cryo_sleep"] = row["CryoSleep"].lower()
    if row.get("Destination"):
        attrs["destination"] = DESTINATION_MAP.get(row["Destination"], row["Destination"].lower())
    if row.get("Age"):
        attrs["age_group"] = age_group(float(row["Age"]))

    # Individual spending columns
    for csv_col, attr_name in SPENDING_COLS:
        val = spend_bin(row.get(csv_col, ""))
        if val is not None:
            attrs[attr_name] = val

    # Cabin → deck, side, cabin_region
    cabin_attrs = parse_cabin(row.get("Cabin", ""))
    attrs.update(cabin_attrs)

    # Group size
    group_id = row["PassengerId"].split("_")[0]
    gc = group_counts.get(group_id, 1)
    if gc == 1:
        attrs["group_size"] = "solo"
    elif gc == 2:
        attrs["group_size"] = "pair"
    else:
        attrs["group_size"] = "family"

    return attrs


# --- Main ---

def main():
    src = Path("data/spaceship-titanic/train.csv")
    out_dir = Path("data/spaceship-titanic")

    # 1. Read CSV and compute group sizes
    with open(src) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    group_counts: Counter[str] = Counter()
    for row in rows:
        group_id = row["PassengerId"].split("_")[0]
        group_counts[group_id] += 1

    # 2. Convert rows to attribute dicts (no dropping — missing fields just omit triples)
    all_attrs = []
    dropped = 0
    for row in rows:
        if not row.get("Transported"):
            dropped += 1
            continue
        attrs = row_to_attrs(row, group_counts)
        attrs["transported"] = row["Transported"].lower()
        all_attrs.append(attrs)

    print(f"Spaceship Titanic → TWM Triples (v3)")
    print(f"  Total rows: {len(rows)}")
    print(f"  Dropped (no Transported label): {dropped}")
    print(f"  Valid rows: {len(all_attrs)}")

    # All advance examples go to train (no identity, no internal splits)
    train_examples = [make_advance(attrs) for attrs in all_attrs]
    random.shuffle(train_examples)

    # Remove old split files if they exist
    for old_file in ["test_comp.jsonl", "test_seen.jsonl"]:
        p = out_dir / old_file
        if p.exists():
            p.unlink()
            print(f"  Removed old {old_file}")

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

    # Value distributions
    print("\n  Value distributions:")
    for attr in ATTR_ORDER + ["transported"]:
        counts = Counter(a.get(attr, "<missing>") for a in all_attrs)
        dist = ", ".join(f"{v}={c}" for v, c in counts.most_common())
        print(f"    {attr}: {dist}")


if __name__ == "__main__":
    main()
