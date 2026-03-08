#!/usr/bin/env python3
"""Generate pet simulator dataset v3.

Expanded from v2:
- 6 attributes × 4 levels (was 4 × 3)
- Conditional effects: play-when-exhausted, feed-when-stuffed, etc.
- Vocalizations: bark/meow as output action triples
- Energy-based compete winner (not fixed)
- Anger contagion through play_with
- Sampled generation (state space too large for exhaustive)

max_triples=16: fits mode(1) + 2 pets × 6 attrs(12) + 2 actions + 1 vocalization.
"""

import json
import random
from pathlib import Path
from itertools import combinations

random.seed(42)

# --- Domain ---

PETS = ["Buddy", "Luna", "Max", "Daisy", "Rocky"]
PET_SPECIES = {
    "Buddy": "dog", "Max": "dog", "Rocky": "dog",
    "Luna": "cat", "Daisy": "cat",
}

# index 0 = best, index -1 = worst
ATTRIBUTES = {
    "hunger":      ["stuffed", "full", "hungry", "starving"],
    "energy":      ["energized", "rested", "tired", "exhausted"],
    "mood":        ["ecstatic", "happy", "content", "sad"],
    "cleanliness": ["spotless", "clean", "messy", "dirty"],
    "boredom":     ["entertained", "engaged", "bored", "restless"],
    "anger":       ["calm", "annoyed", "angry", "furious"],
}
ATTR_NAMES = list(ATTRIBUTES.keys())

# Base effects: "up" = toward index 0 (better), "down" = toward last (worse)
BASE_EFFECTS = {
    "feed":   {"hunger": "up"},
    "play":   {"mood": "up", "energy": "down", "boredom": "up"},
    "bathe":  {"cleanliness": "up"},
    "nap":    {"energy": "up", "boredom": "down"},
    "pet":    {"mood": "up"},
    "walk":   {"mood": "up", "energy": "down", "cleanliness": "down", "boredom": "up"},
    "ignore": {"mood": "down", "boredom": "down"},
}
SOLO_ACTIONS = list(BASE_EFFECTS.keys())

# Conditional overrides: list of (condition, override_effects)
# condition: {attr: value_or_set}. "species" is a special key.
# First match wins; if none, base effects apply.
CONDITIONALS = {
    "play": [
        ({"energy": {"exhausted"}}, {"mood": "down", "energy": "down"}),
    ],
    "feed": [
        ({"hunger": {"stuffed"}}, {"cleanliness": "down"}),
    ],
    "walk": [
        ({"energy": {"exhausted"}},
         {"mood": "up", "energy": "down", "cleanliness": "down", "boredom": "up", "anger": "down"}),
    ],
    "pet": [
        ({"anger": {"angry", "furious"}}, {"mood": "up", "anger": "up"}),
    ],
    "bathe": [
        ({"species": "cat"}, {"cleanliness": "up", "anger": "down"}),
    ],
}

# Interaction base effects
INTERACTION_EFFECTS = {
    "play_with": {"mood": "up", "energy": "down", "boredom": "up"},
    "cuddle":    {"mood": "up", "anger": "up"},
    "compete":   None,
}
INTERACTIONS = list(INTERACTION_EFFECTS.keys())

# Compositional holdout
SOLO_HOLDOUT = {
    "Daisy": {"feed", "play", "walk"},
    "Rocky": {"bathe", "nap", "pet", "ignore"},
}
INTERACTION_HOLDOUT_PAIRS = {("Daisy", "Rocky"), ("Rocky", "Daisy")}

MODE_ADVANCE = ["#mode", "type", "advance"]
MODE_IDENTITY = ["#mode", "type", "identity"]


# --- Helpers ---

def random_state():
    return {attr: random.choice(vals) for attr, vals in ATTRIBUTES.items()}


def random_state_with(**overrides):
    state = random_state()
    state.update(overrides)
    return state


def shift(state, attr, direction):
    vals = ATTRIBUTES[attr]
    idx = vals.index(state[attr])
    if direction == "up":
        idx = max(0, idx - 1)
    else:
        idx = min(len(vals) - 1, idx + 1)
    return vals[idx]


def matches_condition(state, condition, pet):
    for key, expected in condition.items():
        if key == "species":
            if PET_SPECIES[pet] != expected:
                return False
        elif isinstance(expected, set):
            if state[key] not in expected:
                return False
        else:
            if state[key] != expected:
                return False
    return True


def get_effects(pet, state, action):
    for condition, overrides in CONDITIONALS.get(action, []):
        if matches_condition(state, condition, pet):
            return dict(overrides)
    return dict(BASE_EFFECTS[action])


def get_vocalizations(pet, state, action=None):
    species = PET_SPECIES[pet]
    if species == "dog":
        if state.get("boredom") in ("bored", "restless") or \
           state.get("anger") in ("angry", "furious") or \
           action == "compete":
            return [[pet, "action", "bark"]]
    else:
        if state.get("hunger") in ("hungry", "starving") or \
           state.get("anger") in ("angry", "furious") or \
           action == "pet":
            return [[pet, "action", "meow"]]
    return []


def apply_solo(pet, state, action):
    effects = get_effects(pet, state, action)
    new = dict(state)
    for attr, direction in effects.items():
        new[attr] = shift(state, attr, direction)
    vocs = get_vocalizations(pet, state, action)
    return new, vocs


def apply_interaction(pet1, state1, pet2, state2, interaction):
    n1, n2 = dict(state1), dict(state2)
    vocs = []

    if interaction == "compete":
        e1 = ATTRIBUTES["energy"].index(state1["energy"])
        e2 = ATTRIBUTES["energy"].index(state2["energy"])
        if e1 < e2:
            pet1_wins = True
        elif e2 < e1:
            pet1_wins = False
        else:
            pet1_wins = random.random() < 0.5

        if pet1_wins:
            n1["hunger"] = shift(state1, "hunger", "up")
            n1["mood"] = shift(state1, "mood", "up")
            n2["mood"] = shift(state2, "mood", "down")
            n2["anger"] = shift(state2, "anger", "down")
        else:
            n2["hunger"] = shift(state2, "hunger", "up")
            n2["mood"] = shift(state2, "mood", "up")
            n1["mood"] = shift(state1, "mood", "down")
            n1["anger"] = shift(state1, "anger", "down")

        vocs += get_vocalizations(pet1, state1, "compete")
        vocs += get_vocalizations(pet2, state2, "compete")

    elif interaction == "play_with":
        effects = INTERACTION_EFFECTS[interaction]
        for attr, direction in effects.items():
            n1[attr] = shift(state1, attr, direction)
            n2[attr] = shift(state2, attr, direction)
        # Anger contagion
        if state1["anger"] in ("angry", "furious"):
            n2["anger"] = shift(state2, "anger", "down")
        if state2["anger"] in ("angry", "furious"):
            n1["anger"] = shift(state1, "anger", "down")
        vocs += get_vocalizations(pet1, state1, interaction)
        vocs += get_vocalizations(pet2, state2, interaction)

    else:  # cuddle
        effects = INTERACTION_EFFECTS[interaction]
        for attr, direction in effects.items():
            n1[attr] = shift(state1, attr, direction)
            n2[attr] = shift(state2, attr, direction)

    return n1, n2, vocs


def state_to_triples(pet, state):
    return [[pet, attr, state[attr]] for attr in ATTR_NAMES]


def is_solo_holdout(pet, action):
    return pet in SOLO_HOLDOUT and action in SOLO_HOLDOUT[pet]


def is_interaction_holdout(pet1, pet2):
    return (pet1, pet2) in INTERACTION_HOLDOUT_PAIRS


# --- Example builders ---

def make_solo(pet, state, action):
    next_state, vocs = apply_solo(pet, state, action)
    inp = [MODE_ADVANCE] + state_to_triples(pet, state) + [[pet, "action", action]]
    out = state_to_triples(pet, next_state) + vocs
    return {"state_t": inp, "state_t+1": out}


def make_two_pet_solo(pet1, state1, action, pet2, state2):
    next1, vocs = apply_solo(pet1, state1, action)
    inp = ([MODE_ADVANCE]
           + state_to_triples(pet1, state1)
           + state_to_triples(pet2, state2)
           + [[pet1, "action", action]])
    out = state_to_triples(pet1, next1) + state_to_triples(pet2, state2) + vocs
    return {"state_t": inp, "state_t+1": out}


def make_interaction(pet1, state1, pet2, state2, interaction):
    next1, next2, vocs = apply_interaction(pet1, state1, pet2, state2, interaction)
    inp = ([MODE_ADVANCE]
           + state_to_triples(pet1, state1)
           + state_to_triples(pet2, state2)
           + [[pet1, "action", interaction], [pet2, "action", interaction]])
    out = state_to_triples(pet1, next1) + state_to_triples(pet2, next2) + vocs
    return {"state_t": inp, "state_t+1": out}


def make_identity(pet, state):
    triples = state_to_triples(pet, state)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def make_two_pet_identity(pet1, state1, pet2, state2):
    triples = state_to_triples(pet1, state1) + state_to_triples(pet2, state2)
    return {"state_t": [MODE_IDENTITY] + triples, "state_t+1": triples}


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  {path.name}: {len(examples)} examples")


# --- Main ---

def main():
    out_dir = Path("data/pet_sim")

    train = []
    test_comp = []
    all_pairs = list(combinations(PETS, 2))
    ordered_pairs = [(a, b) for a, b in all_pairs] + [(b, a) for a, b in all_pairs]

    # --- 1. Single-pet solo (sampled) ---
    solo_train = []
    for pet in PETS:
        for action in SOLO_ACTIONS:
            for _ in range(200):
                state = random_state()
                ex = make_solo(pet, state, action)
                if is_solo_holdout(pet, action):
                    test_comp.append(ex)
                else:
                    solo_train.append(ex)

    # --- 2. Targeted conditional examples ---
    for pet in PETS:
        # play when exhausted
        for _ in range(60):
            ex = make_solo(pet, random_state_with(energy="exhausted"), "play")
            (test_comp if is_solo_holdout(pet, "play") else solo_train).append(ex)

        # feed when stuffed
        for _ in range(60):
            ex = make_solo(pet, random_state_with(hunger="stuffed"), "feed")
            (test_comp if is_solo_holdout(pet, "feed") else solo_train).append(ex)

        # walk when exhausted
        for _ in range(60):
            ex = make_solo(pet, random_state_with(energy="exhausted"), "walk")
            (test_comp if is_solo_holdout(pet, "walk") else solo_train).append(ex)

        # pet when angry
        for _ in range(60):
            anger_val = random.choice(["angry", "furious"])
            ex = make_solo(pet, random_state_with(anger=anger_val), "pet")
            (test_comp if is_solo_holdout(pet, "pet") else solo_train).append(ex)

        # bathe cats when any mood (cats always get angrier)
        if PET_SPECIES[pet] == "cat":
            for _ in range(60):
                ex = make_solo(pet, random_state(), "bathe")
                (test_comp if is_solo_holdout(pet, "bathe") else solo_train).append(ex)

    # Vocalization triggers (ensure model sees bark/meow)
    for pet in PETS:
        species = PET_SPECIES[pet]
        for _ in range(40):
            if species == "dog":
                state = random_state_with(boredom=random.choice(["bored", "restless"]))
            else:
                state = random_state_with(hunger=random.choice(["hungry", "starving"]))
            action = random.choice(SOLO_ACTIONS)
            ex = make_solo(pet, state, action)
            (test_comp if is_solo_holdout(pet, action) else solo_train).append(ex)

        for _ in range(40):
            state = random_state_with(anger=random.choice(["angry", "furious"]))
            action = random.choice(SOLO_ACTIONS)
            ex = make_solo(pet, state, action)
            (test_comp if is_solo_holdout(pet, action) else solo_train).append(ex)

    random.shuffle(solo_train)
    n_seen_test = len(solo_train) // 10
    test_seen = solo_train[:n_seen_test]
    solo_train = solo_train[n_seen_test:]
    train.extend(solo_train)

    # --- 3. Two-pet solo ---
    for pet1, pet2 in ordered_pairs:
        for _ in range(40):
            state1, state2 = random_state(), random_state()
            action = random.choice(SOLO_ACTIONS)
            ex = make_two_pet_solo(pet1, state1, action, pet2, state2)
            (test_comp if is_solo_holdout(pet1, action) else train).append(ex)

    # --- 4. Interactions (play_with, cuddle) ---
    for pet1, pet2 in ordered_pairs:
        for interaction in ["play_with", "cuddle"]:
            for _ in range(40):
                state1, state2 = random_state(), random_state()
                ex = make_interaction(pet1, state1, pet2, state2, interaction)
                (test_comp if is_interaction_holdout(pet1, pet2) else train).append(ex)

    # Anger contagion examples for play_with
    for pet1, pet2 in ordered_pairs:
        for _ in range(30):
            angry_pet = random.choice([pet1, pet2])
            angry_val = random.choice(["angry", "furious"])
            s1 = random_state_with(anger=angry_val) if angry_pet == pet1 else random_state()
            s2 = random_state_with(anger=angry_val) if angry_pet == pet2 else random_state()
            ex = make_interaction(pet1, s1, pet2, s2, "play_with")
            (test_comp if is_interaction_holdout(pet1, pet2) else train).append(ex)

    # --- 5. Compete (energy-balanced) ---
    energy_vals = ATTRIBUTES["energy"]
    for pet1, pet2 in ordered_pairs:
        # pet1 has more energy (wins)
        for _ in range(25):
            e1_idx = random.randint(0, 2)
            e2_idx = random.randint(e1_idx + 1, 3)
            s1 = random_state_with(energy=energy_vals[e1_idx])
            s2 = random_state_with(energy=energy_vals[e2_idx])
            ex = make_interaction(pet1, s1, pet2, s2, "compete")
            (test_comp if is_interaction_holdout(pet1, pet2) else train).append(ex)

        # pet2 has more energy (pet1 loses)
        for _ in range(25):
            e2_idx = random.randint(0, 2)
            e1_idx = random.randint(e2_idx + 1, 3)
            s1 = random_state_with(energy=energy_vals[e1_idx])
            s2 = random_state_with(energy=energy_vals[e2_idx])
            ex = make_interaction(pet1, s1, pet2, s2, "compete")
            (test_comp if is_interaction_holdout(pet1, pet2) else train).append(ex)

        # Equal energy (both outcomes)
        for _ in range(15):
            e_idx = random.randint(0, 3)
            s1 = random_state_with(energy=energy_vals[e_idx])
            s2 = random_state_with(energy=energy_vals[e_idx])
            ex = make_interaction(pet1, s1, pet2, s2, "compete")
            (test_comp if is_interaction_holdout(pet1, pet2) else train).append(ex)

    # --- 6. Identity ---
    identity = []
    for pet in PETS:
        for _ in range(200):
            identity.append(make_identity(pet, random_state()))
    for pet1, pet2 in all_pairs:
        for _ in range(30):
            identity.append(make_two_pet_identity(pet1, random_state(), pet2, random_state()))
    train.extend(identity)

    # --- Shuffle and write ---
    random.shuffle(train)
    random.shuffle(test_comp)
    random.shuffle(test_seen)

    print(f"\nPet Simulator Dataset (v3):")
    print(f"  {len(PETS)} pets, {len(ATTRIBUTES)} attrs × {len(next(iter(ATTRIBUTES.values())))} levels")
    print(f"  {len(SOLO_ACTIONS)} solo + {len(INTERACTIONS)} interactions")
    print(f"  {sum(len(v) for v in CONDITIONALS.values())} conditional rules")
    print()

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test_comp.jsonl", test_comp)
    write_jsonl(out_dir / "test_seen.jsonl", test_seen)

    print(f"\n  Train: {len(train)}")
    print(f"  Test comp: {len(test_comp)}")
    print(f"  Test seen: {len(test_seen)}")

    # Verify
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


if __name__ == "__main__":
    main()
