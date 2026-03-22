#!/usr/bin/env python3
"""Convert classic Titanic CSV to TWM triple-transition JSONL.

Frames survival prediction as state transition:
  state_t: passenger attributes (7 triples + mode)
  state_t+1: same attributes + survived prediction

Attributes: class, sex, age_group, family_size, fare, embarked, cabin_known
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

# --- Discretization ---

def age_group(age: float) -> str:
    if age <= 17:
        return "child"
    elif age <= 30:
        return "young_adult"
    elif age <= 50:
        return "adult"
    else:
        return "senior"


def family_size(sibsp: int, parch: int) -> str:
    total = sibsp + parch
    if total == 0:
        return "solo"
    elif total <= 3:
        return "small"
    else:
        return "large"


def fare_level(fare: float) -> str:
    if fare <= 10:
        return "low"
    elif fare <= 30:
        return "medium"
    elif fare <= 100:
        return "high"
    else:
        return "premium"


EMBARKED_MAP = {
    "S": "southampton",
    "C": "cherbourg",
    "Q": "queenstown",
}

CLASS_MAP = {
    "1": "first",
    "2": "second",
    "3": "third",
}

# --- Compositional holdout combos ---

HOLDOUT_COMBOS = [
    {"class": "first", "sex": "male", "age_group": "senior"},
    {"class": "third", "sex": "female", "embarked": "queenstown"},
    {"class": "second", "family_size": "large"},
]


def matches_holdout(attrs: dict) -> bool:
    for combo in HOLDOUT_COMBOS:
        if all(attrs.get(k) == v for k, v in combo.items()):
            return True
    return False


# --- Triple builders ---

MODE_ADVANCE = ["#mode", "type", "advance"]
MODE_IDENTITY = ["#mode", "type", "identity"]

ATTR_KEYS = ["class", "sex", "age_group", "family_size", "fare", "embarked", "cabin_known"]


def attrs_to_triples(attrs: dict, include_survived: bool = False) -> list[list[str]]:
    triples = [["passenger", k, attrs[k]] for k in ATTR_KEYS if k in attrs]
    if include_survived:
        triples.append(["passenger", "survived", attrs["survived"]])
    return triples


def make_advance(attrs: dict) -> dict:
    state_t = [MODE_ADVANCE] + attrs_to_triples(attrs, include_survived=False)
    state_t1 = attrs_to_triples(attrs, include_survived=True)
    return {"state_t": state_t, "state_t+1": state_t1}


def make_identity(attrs: dict) -> dict:
    triples = attrs_to_triples(attrs, include_survived=True)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


def row_to_attrs(row: dict) -> dict | None:
    """Convert a CSV row to attribute dict. Skips missing fields instead of dropping rows.

    Returns None only if Sex and Pclass are both missing (too little signal).
    """
    if not row.get("Sex") and not row.get("Pclass"):
        return None

    attrs = {}

    if row.get("Pclass") and row["Pclass"] in CLASS_MAP:
        attrs["class"] = CLASS_MAP[row["Pclass"]]
    if row.get("Sex"):
        attrs["sex"] = row["Sex"].lower()
    if row.get("Age"):
        attrs["age_group"] = age_group(float(row["Age"]))
    if row.get("SibSp") is not None and row.get("Parch") is not None:
        attrs["family_size"] = family_size(int(row["SibSp"]), int(row["Parch"]))
    if row.get("Fare"):
        attrs["fare"] = fare_level(float(row["Fare"]))
    if row.get("Embarked") and row["Embarked"] in EMBARKED_MAP:
        attrs["embarked"] = EMBARKED_MAP[row["Embarked"]]
    attrs["cabin_known"] = "yes" if row.get("Cabin") else "no"

    return attrs


# --- Main ---

def main():
    src = Path("data/titanic/train.csv")
    out_dir = Path("data/titanic")

    # 1. Read CSV
    with open(src) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 2. Convert rows to attribute dicts
    all_attrs = []
    dropped = 0
    for row in rows:
        attrs = row_to_attrs(row)
        if attrs is None:
            dropped += 1
            continue
        attrs["survived"] = "yes" if row["Survived"] == "1" else "no"
        all_attrs.append(attrs)

    print(f"Classic Titanic → TWM Triples")
    print(f"  Total rows: {len(rows)}")
    print(f"  Dropped (missing fields): {dropped}")
    print(f"  Valid rows: {len(all_attrs)}")

    # 3. Split into holdout vs train pool
    test_comp_examples = []
    train_pool = []
    for attrs in all_attrs:
        ex = make_advance(attrs)
        if matches_holdout(attrs):
            test_comp_examples.append(ex)
        else:
            train_pool.append(ex)

    # 4. Split train pool: 90% train, 10% test_seen
    random.shuffle(train_pool)
    n_seen = len(train_pool) // 10
    test_seen_examples = train_pool[:n_seen]
    train_examples = train_pool[n_seen:]

    # 5. Add identity examples (~20% of train count)
    n_identity = len(train_examples) // 5
    identity_indices = random.sample(range(len(all_attrs)), min(n_identity, len(all_attrs)))
    for idx in identity_indices:
        train_examples.append(make_identity(all_attrs[idx]))

    # 6. Shuffle and write
    random.shuffle(train_examples)
    random.shuffle(test_comp_examples)
    random.shuffle(test_seen_examples)

    print()
    write_jsonl(out_dir / "train.jsonl", train_examples)
    write_jsonl(out_dir / "test_comp.jsonl", test_comp_examples)
    write_jsonl(out_dir / "test_seen.jsonl", test_seen_examples)

    # 7. Stats
    max_in = max_out = 0
    tokens = set()
    for f in [out_dir / "train.jsonl", out_dir / "test_comp.jsonl", out_dir / "test_seen.jsonl"]:
        for line in open(f):
            ex = json.loads(line)
            max_in = max(max_in, len(ex["state_t"]))
            max_out = max(max_out, len(ex["state_t+1"]))
            for t in ex["state_t"] + ex["state_t+1"]:
                tokens.update(t)

    print(f"\n  Train: {len(train_examples)} (incl. {n_identity} identity)")
    print(f"  Test comp: {len(test_comp_examples)}")
    print(f"  Test seen: {len(test_seen_examples)}")
    print(f"\n  Max input triples: {max_in}")
    print(f"  Max output triples: {max_out}")
    print(f"  Unique tokens: {len(tokens)}")
    print(f"  Tokens: {sorted(tokens)}")

    # Value distributions
    print("\n  Value distributions:")
    for attr in ATTR_KEYS + ["survived"]:
        counts = Counter(a.get(attr, "<missing>") for a in all_attrs)
        dist = ", ".join(f"{v}={c}" for v, c in counts.most_common())
        print(f"    {attr}: {dist}")


if __name__ == "__main__":
    main()
