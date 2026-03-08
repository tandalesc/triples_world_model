#!/usr/bin/env python3
"""Generate pet simulator dataset for validating identity + world-advance modes.

Mode is encoded as a triple: ["#mode", "type", "identity"] or ["#mode", "type", "advance"].
This is the first triple in every example — the model learns to condition on it
through regular attention, no special architecture needed.

Pets: Buddy, Luna, Max, Daisy, Rocky (unique names = unique tokens)
Attributes: hunger, energy, mood, cleanliness (3 levels each, ordered low→high)
Solo actions: feed, play, bathe, nap, pet, walk, ignore
Interactions: play_with (both pets get mood up, energy down),
              compete (one wins food, other stays hungry),
              cuddle (both mood up)

max_triples=12: fits two full pets (4+4) + mode + action(s) comfortably.

Compositional generalization holdout:
  - Buddy, Luna, Max: all solo actions in training
  - Daisy: train on {bathe, nap, pet, ignore}, test on {feed, play, walk}
  - Rocky: train on {feed, play, walk}, test on {bathe, nap, pet, ignore}
  - Interactions: hold out Daisy+Rocky pairs for test

Output:
  data/pet_sim/train.jsonl      — combined identity + dynamics + interactions
  data/pet_sim/test_comp.jsonl  — compositional holdout (unseen combos)
  data/pet_sim/test_seen.jsonl  — seen combos held out for validation
"""

import json
import random
from pathlib import Path
from itertools import product, combinations

random.seed(42)

# --- Domain definition ---

PETS = ["Buddy", "Luna", "Max", "Daisy", "Rocky"]

# Values ordered: index 0 = best/highest, index -1 = worst/lowest
ATTRIBUTES = {
    "hunger":      ["full", "hungry", "starving"],
    "energy":      ["rested", "tired", "exhausted"],
    "mood":        ["happy", "content", "sad"],
    "cleanliness": ["clean", "messy", "dirty"],
}

# Solo action → {attr: "up" (toward index 0) or "down" (toward index -1)}
SOLO_ACTIONS = {
    "feed":   {"hunger": "up"},
    "play":   {"mood": "up", "energy": "down"},
    "bathe":  {"cleanliness": "up"},
    "nap":    {"energy": "up"},
    "pet":    {"mood": "up"},
    "walk":   {"mood": "up", "energy": "down", "cleanliness": "down"},
    "ignore": {"mood": "down"},
}

# Interaction effects: applied to BOTH pets
INTERACTIONS = {
    "play_with": {"mood": "up", "energy": "down"},
    "cuddle":    {"mood": "up"},
    "compete":   None,  # special: winner hunger up, loser mood down
}

# Compositional holdout
SOLO_HOLDOUT = {
    "Daisy": {"feed", "play", "walk"},
    "Rocky": {"bathe", "nap", "pet", "ignore"},
}
# Hold out these pairs for interaction tests
INTERACTION_HOLDOUT_PAIRS = {("Daisy", "Rocky"), ("Rocky", "Daisy")}

MODE_ADVANCE = ["#mode", "type", "advance"]
MODE_IDENTITY = ["#mode", "type", "identity"]


def random_state() -> dict[str, str]:
    return {attr: random.choice(vals) for attr, vals in ATTRIBUTES.items()}


def shift(state: dict[str, str], attr: str, direction: str) -> str:
    """Shift one attribute value up or down, clamped."""
    vals = ATTRIBUTES[attr]
    idx = vals.index(state[attr])
    if direction == "up":
        idx = max(0, idx - 1)
    else:
        idx = min(len(vals) - 1, idx + 1)
    return vals[idx]


def apply_solo(state: dict[str, str], action: str) -> dict[str, str]:
    new = dict(state)
    for attr, direction in SOLO_ACTIONS[action].items():
        new[attr] = shift(state, attr, direction)
    return new


def apply_interaction(state1: dict[str, str], state2: dict[str, str],
                      interaction: str) -> tuple[dict[str, str], dict[str, str]]:
    """Apply interaction, return (new_state1, new_state2)."""
    n1, n2 = dict(state1), dict(state2)

    if interaction == "compete":
        # Pet1 "wins" food, pet2 gets mood down
        n1["hunger"] = shift(state1, "hunger", "up")
        n2["mood"] = shift(state2, "mood", "down")
    else:
        effects = INTERACTIONS[interaction]
        for attr, direction in effects.items():
            n1[attr] = shift(state1, attr, direction)
            n2[attr] = shift(state2, attr, direction)

    return n1, n2


def state_to_triples(pet: str, state: dict[str, str]) -> list[list[str]]:
    return [[pet, attr, val] for attr, val in state.items()]


def is_solo_holdout(pet: str, action: str) -> bool:
    return pet in SOLO_HOLDOUT and action in SOLO_HOLDOUT[pet]


def is_interaction_holdout(pet1: str, pet2: str) -> bool:
    return (pet1, pet2) in INTERACTION_HOLDOUT_PAIRS


# --- Example builders ---

def make_solo(pet: str, state: dict[str, str], action: str) -> dict:
    next_state = apply_solo(state, action)
    inp = [MODE_ADVANCE] + state_to_triples(pet, state) + [[pet, "action", action]]
    out = state_to_triples(pet, next_state)
    return {"state_t": inp, "state_t+1": out}


def make_two_pet_solo(pet1: str, state1: dict[str, str], action: str,
                      pet2: str, state2: dict[str, str]) -> dict:
    """pet1 acts, pet2 unchanged. Full 4 attrs each."""
    next1 = apply_solo(state1, action)
    inp = ([MODE_ADVANCE]
           + state_to_triples(pet1, state1)
           + state_to_triples(pet2, state2)
           + [[pet1, "action", action]])
    out = state_to_triples(pet1, next1) + state_to_triples(pet2, state2)
    return {"state_t": inp, "state_t+1": out}


def make_interaction(pet1: str, state1: dict[str, str],
                     pet2: str, state2: dict[str, str],
                     interaction: str) -> dict:
    """Both pets affected by interaction."""
    next1, next2 = apply_interaction(state1, state2, interaction)
    inp = ([MODE_ADVANCE]
           + state_to_triples(pet1, state1)
           + state_to_triples(pet2, state2)
           + [[pet1, "action", interaction], [pet2, "action", interaction]])
    out = state_to_triples(pet1, next1) + state_to_triples(pet2, next2)
    return {"state_t": inp, "state_t+1": out}


def make_identity(pet: str, state: dict[str, str]) -> dict:
    triples = state_to_triples(pet, state)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def make_two_pet_identity(pet1: str, state1: dict[str, str],
                          pet2: str, state2: dict[str, str]) -> dict:
    triples = state_to_triples(pet1, state1) + state_to_triples(pet2, state2)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def write_jsonl(path: Path, examples: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


def main():
    out_dir = Path("data/pet_sim")

    attr_names = list(ATTRIBUTES.keys())
    all_val_combos = list(product(*[ATTRIBUTES[a] for a in attr_names]))

    train_examples = []
    test_comp = []

    # --- 1. Single-pet solo actions (exhaustive) ---
    solo_train = []
    for pet in PETS:
        for vals in all_val_combos:
            state = dict(zip(attr_names, vals))
            for action in SOLO_ACTIONS:
                ex = make_solo(pet, state, action)
                if is_solo_holdout(pet, action):
                    test_comp.append(ex)
                else:
                    solo_train.append(ex)

    random.shuffle(solo_train)
    n_seen_test = len(solo_train) // 10
    test_seen = solo_train[:n_seen_test]
    solo_train = solo_train[n_seen_test:]
    train_examples.extend(solo_train)

    # --- 2. Two-pet solo (one acts, other passive) — full 4 attrs ---
    all_pairs = list(combinations(PETS, 2))
    # Include both orderings (who acts)
    ordered_pairs = [(a, b) for a, b in all_pairs] + [(b, a) for a, b in all_pairs]

    for pet1, pet2 in ordered_pairs:
        for _ in range(30):
            state1 = random_state()
            state2 = random_state()
            action = random.choice(list(SOLO_ACTIONS.keys()))
            ex = make_two_pet_solo(pet1, state1, action, pet2, state2)
            if is_solo_holdout(pet1, action):
                test_comp.append(ex)
            else:
                train_examples.append(ex)

    # --- 3. Interactions ---
    for pet1, pet2 in ordered_pairs:
        for interaction in INTERACTIONS:
            for _ in range(30):
                state1 = random_state()
                state2 = random_state()
                ex = make_interaction(pet1, state1, pet2, state2, interaction)
                if is_interaction_holdout(pet1, pet2):
                    test_comp.append(ex)
                else:
                    train_examples.append(ex)

    # --- 4. Identity (single + two-pet) ---
    identity_examples = []
    for pet in PETS:
        for vals in all_val_combos:
            state = dict(zip(attr_names, vals))
            identity_examples.append(make_identity(pet, state))

    for pet1, pet2 in all_pairs:
        for _ in range(20):
            identity_examples.append(
                make_two_pet_identity(pet1, random_state(), pet2, random_state()))

    train_examples.extend(identity_examples)

    # --- Shuffle and write ---
    random.shuffle(train_examples)
    random.shuffle(test_comp)
    random.shuffle(test_seen)

    # Count by type
    n_identity = len(identity_examples)
    n_solo_train = len(solo_train)
    n_two_pet = sum(1 for _ in train_examples)  # approximate

    print(f"\nPet Simulator Dataset (v2):")
    print(f"  Vocab tokens: {len(PETS)} pets, {len(ATTRIBUTES)} attrs, "
          f"{sum(len(v) for v in ATTRIBUTES.values())} values, "
          f"{len(SOLO_ACTIONS)} solo actions, {len(INTERACTIONS)} interactions")
    print(f"  Mode triple: ['#mode', 'type', 'identity'|'advance']")
    print(f"  max_triples needed: 12")
    print()

    write_jsonl(out_dir / "train.jsonl", train_examples)
    write_jsonl(out_dir / "test_comp.jsonl", test_comp)
    write_jsonl(out_dir / "test_seen.jsonl", test_seen)

    # Stats
    print(f"\n  Train total: {len(train_examples)}")
    print(f"    ~{n_solo_train} solo dynamics")
    print(f"    ~{n_identity} identity")
    print(f"    + two-pet solo + interactions")
    print(f"  Test comp: {len(test_comp)}")
    print(f"  Test seen: {len(test_seen)}")

    # Verify max triples
    max_in = max_out = 0
    tokens = set()
    for f in [out_dir / "train.jsonl", out_dir / "test_comp.jsonl", out_dir / "test_seen.jsonl"]:
        for line in open(f):
            ex = json.loads(line)
            max_in = max(max_in, len(ex["state_t"]))
            max_out = max(max_out, len(ex["state_t+1"]))
            for t in ex["state_t"] + ex["state_t+1"]:
                tokens.update(t)
    print(f"\n  Max input triples: {max_in}")
    print(f"  Max output triples: {max_out}")
    print(f"  Unique tokens: {len(tokens)}")
    print(f"  Tokens: {sorted(tokens)}")

    # Samples
    print(f"\n  --- Sample interaction ---")
    for ex in train_examples:
        actions = [t for t in ex["state_t"] if t[1] == "action"]
        if len(actions) == 2:
            print(f"    {json.dumps(ex)}")
            break
    print(f"\n  --- Sample two-pet solo ---")
    for ex in train_examples:
        pets_in = set(t[0] for t in ex["state_t"] if t[0] not in ("#mode",))
        actions = [t for t in ex["state_t"] if t[1] == "action"]
        if len(pets_in) == 2 and len(actions) == 1:
            print(f"    {json.dumps(ex)}")
            break


if __name__ == "__main__":
    main()
