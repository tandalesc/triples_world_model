"""Tests for the synthetic entity-world trace generator (scripts/generate_entity_world.py).

Covers the four properties the engram-wm OOD-entity experiment depends on:
  - Determinism under seed: same config + seed -> byte-identical splits.
  - Split disjointness: train / near-OOD / far-OOD use disjoint entity TYPE sets.
  - Oracle consistency: replaying a chain's action labels from the initial state
    reproduces the rendered state texts exactly.
  - Action-label alignment: len(actions) == len(chain) - 1, labels parse, actors valid.

Run (pytest not vendored locally):
    uv run --with pytest python -m pytest tests/jepa/test_entity_world.py -q
"""

import importlib.util
import json
import random
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the generator module by path (scripts/ is not a package).
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
GEN_PATH = REPO / "scripts" / "generate_entity_world.py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("generate_entity_world", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_chain_determinism_same_seed(gen):
    cfg = gen.CONFIG
    train_types = gen._types_for_role("train")

    def run():
        rng = random.Random(123)
        return gen.build_split(rng, train_types, 50, cfg)[0]

    a = run()
    b = run()
    assert a == b, "same seed must yield byte-identical chains"


def test_generate_chain_deterministic(gen):
    types = ["dog", "kettle"]
    out1 = gen.generate_chain(random.Random(99), types, 6)
    out2 = gen.generate_chain(random.Random(99), types, 6)
    assert out1 == out2


def test_different_seed_differs(gen):
    types = ["dog"]
    a = gen.generate_chain(random.Random(1), types, 6)
    b = gen.generate_chain(random.Random(2), types, 6)
    assert a != b


# ---------------------------------------------------------------------------
# Split disjointness
# ---------------------------------------------------------------------------

def test_type_role_disjointness(gen):
    train = set(gen._types_for_role("train"))
    near = set(gen._types_for_role("near_ood"))
    far = set(gen._types_for_role("far_ood"))
    assert train and near and far
    assert train.isdisjoint(near)
    assert train.isdisjoint(far)
    assert near.isdisjoint(far)


def test_split_files_use_only_their_types(gen, tmp_path):
    """Every chain in a split mentions only display names from that split's type set."""
    cfg = dict(gen.CONFIG)
    roles = {
        "train": gen._types_for_role("train"),
        "near": gen._types_for_role("near_ood"),
        "far": gen._types_for_role("far_ood"),
    }
    for role, types in roles.items():
        plain, _ = gen.build_split(random.Random(5), types, 30, cfg)
        allowed_displays = {gen.TYPE_LIBRARY[t]["display"] for t in types}
        other_displays = {d["display"] for d in gen.TYPE_LIBRARY.values()} - allowed_displays
        text = " ".join(t for r in plain for t in r["chain"]).lower()
        for disp in other_displays:
            # A foreign display must not appear (display names are distinct noun phrases).
            assert disp.lower() not in text, f"{role} split leaked '{disp}'"


def test_near_ood_shares_schema_far_does_not(gen):
    """near-OOD reuses a train schema (interpolation); far-OOD is a novel recombination."""
    train_schemas = {tuple(d["schema"]) for n, d in gen.TYPE_LIBRARY.items()
                     if d["split_role"] == "train"}
    for name, d in gen.TYPE_LIBRARY.items():
        if d["split_role"] == "near_ood":
            assert tuple(d["schema"]) in train_schemas, f"near {name} should reuse a schema"
            assert "derived_from" in d
        if d["split_role"] == "far_ood":
            assert tuple(d["schema"]) not in train_schemas, f"far {name} should be novel"


# ---------------------------------------------------------------------------
# Oracle consistency (replay reproduces states)
# ---------------------------------------------------------------------------

def test_oracle_replay_reproduces_chain(gen):
    """Replaying action labels from the recorded initial states reproduces every
    rendered state text in the chain. This is the contract the action-recovery eval
    relies on: labels + initial state == the trace."""
    rng = random.Random(2026)
    for _ in range(200):
        types = gen._sample_type_names(rng, gen._types_for_role("train"), (1, 2))
        chain_len = rng.randint(4, 8)
        # generate_chain seeds its own entity states internally; to replay we need those
        # initial states, so reproduce generation with a captured rng and pull state_0.
        sub = random.Random(rng.random())
        # Reconstruct what generate_chain does, capturing initial states + labels.
        entities = [(tn, gen._random_state(sub, tn)) for tn in types]
        initial_states = [dict(st) for _, st in entities]
        texts = [gen.render_state(entities)]
        labels = []
        for _ in range(chain_len - 1):
            actor = sub.randrange(len(entities))
            tn = entities[actor][0]
            action = sub.choice(gen._applicable_actions(tn))
            new = gen.apply_action(tn, entities[actor][1], action)
            entities[actor] = (tn, new)
            texts.append(f"{gen.render_action(entities, action, actor)} {gen.render_state(entities)}")
            labels.append(f"{action}@{actor}")

        # Now replay purely from initial_states + labels via the public replay path.
        snapshots = gen.replay_chain(types, initial_states, labels)
        # Re-render each snapshot's state and compare to the trace's state portion.
        assert len(snapshots) == len(texts)
        # state_0
        assert gen.render_state(list(zip(types, snapshots[0]))) == texts[0]
        # subsequent states (text = "<action sentence> <state sentence>")
        for i in range(1, len(texts)):
            rendered_state = gen.render_state(list(zip(types, snapshots[i])))
            assert texts[i].endswith(rendered_state), f"replay mismatch at step {i}"


def test_apply_action_outside_schema_is_noop(gen):
    """An action with no profile entry for a type leaves the state unchanged."""
    state = gen._random_state(random.Random(0), "lamp")  # device: no 'feed'
    after = gen.apply_action("lamp", state, "feed")
    assert after == state


def test_apply_action_pure(gen):
    """apply_action must not mutate the input state."""
    state = gen._random_state(random.Random(0), "dog")
    snapshot = dict(state)
    gen.apply_action("dog", state, "feed")
    assert state == snapshot


# ---------------------------------------------------------------------------
# Action-label alignment
# ---------------------------------------------------------------------------

def test_action_label_alignment(gen):
    rng = random.Random(11)
    for _ in range(100):
        types = gen._sample_type_names(rng, gen._types_for_role("train"), (1, 2))
        chain_len = rng.randint(4, 8)
        texts, actions = gen.generate_chain(rng, types, chain_len)
        assert len(texts) == chain_len
        assert len(actions) == chain_len - 1
        for label in actions:
            action, actor = label.rsplit("@", 1)
            assert action in gen.ACTIONS
            assert 0 <= int(actor) < len(types)


def test_labeled_records_carry_actions_and_types(gen):
    cfg = dict(gen.CONFIG)
    _, labeled = gen.build_split(random.Random(3), gen._types_for_role("train"), 20, cfg)
    for r in labeled:
        assert "chain" in r and "actions" in r and "types" in r
        assert len(r["actions"]) == len(r["chain"]) - 1
        for label in r["actions"]:
            action, actor = label.rsplit("@", 1)
            assert action in gen.ACTIONS
            assert 0 <= int(actor) < len(r["types"])


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_describes_all_types(gen):
    m = gen.build_manifest(gen.CONFIG)
    # Manifest only includes types active under the config's world_version.
    # For world_version=1 (the default CONFIG) this is the v1 types only.
    world_version = gen.CONFIG.get("world_version", 1)
    expected_types = {name for name, d in gen.TYPE_LIBRARY.items()
                      if d.get("world_version_min", 1) <= world_version}
    assert set(m["types"].keys()) == expected_types
    for name, t in m["types"].items():
        assert t["schema"]
        assert t["profile"]
        assert t["split_role"] in {"train", "near_ood", "far_ood"}
    # Splits documented with discriminating purpose.
    assert set(m["splits"].keys()) == {"train", "test_iid", "test_ood_near", "test_ood_far"}
    for s in m["splits"].values():
        assert s["discriminates"]
