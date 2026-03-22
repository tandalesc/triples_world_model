#!/usr/bin/env python3
"""Convert classic Titanic CSV to TWM triple-transition JSONL (v3).

Trains on ALL of train.csv (no internal test split — Kaggle submission is the only real eval).

Attributes (all optional — missing fields omitted as triples):
  title, class, sex, age_group, is_child, age_estimated,
  sibsp, parch, is_alone, fare_pp, embarked, cabin_deck, cabin_known
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(42)

# --- Feature extraction ---

def extract_title(name: str) -> str | None:
    match = re.search(r', (\w+)\.', name)
    if not match:
        return None
    title = match.group(1)
    if title in ("Mr",):
        return "mr"
    elif title in ("Mrs", "Mme", "Ms"):
        return "mrs"
    elif title in ("Miss", "Mlle"):
        return "miss"
    elif title in ("Master",):
        return "master"
    else:
        return "rare"


def age_group(age: float) -> str:
    if age <= 5:
        return "infant"
    elif age <= 12:
        return "child"
    elif age <= 17:
        return "teen"
    elif age <= 30:
        return "young_adult"
    elif age <= 50:
        return "adult"
    else:
        return "senior"


def sibsp_bin(n: int) -> str:
    if n == 0:
        return "none"
    elif n == 1:
        return "one"
    else:
        return "many"


def parch_bin(n: int) -> str:
    if n == 0:
        return "none"
    elif n <= 2:
        return "one_two"
    else:
        return "many"


def fare_pp_level(fare: float, family_total: int) -> str:
    """Fare per person."""
    fpp = fare / max(1, 1 + family_total)
    if fpp <= 8:
        return "very_low"
    elif fpp <= 15:
        return "low"
    elif fpp <= 30:
        return "medium"
    elif fpp <= 60:
        return "high"
    else:
        return "premium"


def extract_cabin_deck(cabin: str) -> str | None:
    if not cabin:
        return None
    return cabin[0].lower()


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

# --- Triple builders ---

MODE_ADVANCE = ["#mode", "type", "advance"]
MODE_IDENTITY = ["#mode", "type", "identity"]

ATTR_KEYS = [
    "title", "class", "sex", "age_group", "is_child", "age_estimated",
    "sibsp", "parch", "is_alone", "fare_pp", "embarked", "cabin_deck", "cabin_known",
]


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
    """Convert a CSV row to attribute dict. Skips missing fields."""
    if not row.get("Sex") and not row.get("Pclass"):
        return None

    attrs = {}

    # Title
    if row.get("Name"):
        title = extract_title(row["Name"])
        if title:
            attrs["title"] = title

    # Class
    if row.get("Pclass") and row["Pclass"] in CLASS_MAP:
        attrs["class"] = CLASS_MAP[row["Pclass"]]

    # Sex
    if row.get("Sex"):
        attrs["sex"] = row["Sex"].lower()

    # Age features
    if row.get("Age"):
        age = float(row["Age"])
        attrs["age_group"] = age_group(age)
        attrs["is_child"] = "yes" if age < 15 else "no"
        attrs["age_estimated"] = "yes" if age != int(age) and (age * 2) == int(age * 2) else "no"

    # Family features
    sibsp = int(row["SibSp"]) if row.get("SibSp") else None
    parch = int(row["Parch"]) if row.get("Parch") else None
    if sibsp is not None:
        attrs["sibsp"] = sibsp_bin(sibsp)
    if parch is not None:
        attrs["parch"] = parch_bin(parch)
    if sibsp is not None and parch is not None:
        attrs["is_alone"] = "yes" if sibsp == 0 and parch == 0 else "no"

    # Fare per person
    if row.get("Fare"):
        family_total = (sibsp or 0) + (parch or 0)
        attrs["fare_pp"] = fare_pp_level(float(row["Fare"]), family_total)

    # Embarked
    if row.get("Embarked") and row["Embarked"] in EMBARKED_MAP:
        attrs["embarked"] = EMBARKED_MAP[row["Embarked"]]

    # Cabin features
    if row.get("Cabin"):
        deck = extract_cabin_deck(row["Cabin"])
        if deck:
            attrs["cabin_deck"] = deck
    attrs["cabin_known"] = "yes" if row.get("Cabin") else "no"

    return attrs


# --- Main ---

def main():
    src = Path("data/titanic/train.csv")
    out_dir = Path("data/titanic")

    with open(src) as f:
        rows = list(csv.DictReader(f))

    # Convert all rows
    all_attrs = []
    dropped = 0
    for row in rows:
        attrs = row_to_attrs(row)
        if attrs is None:
            dropped += 1
            continue
        attrs["survived"] = "yes" if row["Survived"] == "1" else "no"
        all_attrs.append(attrs)

    print(f"Classic Titanic → TWM Triples (v3)")
    print(f"  Total rows: {len(rows)}")
    print(f"  Dropped: {dropped}")
    print(f"  Valid rows: {len(all_attrs)}")

    # All advance examples go to train (no identity — every example teaches classification)
    train_examples = [make_advance(attrs) for attrs in all_attrs]
    random.shuffle(train_examples)

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
    for attr in ATTR_KEYS + ["survived"]:
        counts = Counter(a.get(attr, "<missing>") for a in all_attrs)
        dist = ", ".join(f"{v}={c}" for v, c in counts.most_common())
        print(f"    {attr}: {dist}")


if __name__ == "__main__":
    main()
