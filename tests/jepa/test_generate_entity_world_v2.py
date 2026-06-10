"""Tests for the v4 entity-world v2 generator extension (jepa_v4_design.md §4).

Covers:
  - world_version=1 byte-reproduces the campaign (determinism regression test)
  - world_version=2 generates >=12 train types, 4 near-OOD, 4 far-OOD
  - world_version=2 chains have 3-5 entities and lengths 6-12
  - oracle_dist is present in labeled records when world_version=2
  - stochastic_v2=False (default) produces degenerate oracle_dist (single entry, prob=1.0)
  - split disjointness holds for v2 types too
  - backward-compatible: world_version=1 ignores all *_v2 keys

Run:
    uv run --with pytest python -m pytest tests/jepa/test_generate_entity_world_v2.py -q
"""

import importlib.util
import json
import random
from pathlib import Path

import pytest

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
# world_version=1 byte-reproducibility (campaign regression)
# ---------------------------------------------------------------------------

def test_world_v1_default_config_unchanged(gen):
    """Default CONFIG must still have world_version=1 (bitwise backward-compat gate)."""
    assert gen.CONFIG.get("world_version", 1) == 1


def test_world_v1_determinism_regression(gen):
    """world_version=1 with a fixed seed must produce byte-identical results on two runs."""
    cfg_v1 = dict(gen.CONFIG)
    cfg_v1["world_version"] = 1
    train_types = gen._types_for_role("train", 1)

    def run():
        rng = random.Random(42)
        plain, labeled = gen.build_split(rng, train_types, 20, cfg_v1)
        return plain

    assert run() == run(), "world_version=1 must be byte-identical across two runs"


def test_world_v1_ignores_v2_keys(gen):
    """Providing *_v2 keys with world_version=1 must not change output."""
    cfg_base = dict(gen.CONFIG)
    cfg_base["world_version"] = 1

    cfg_with_v2 = dict(cfg_base)
    cfg_with_v2["entities_per_chain_v2"] = (3, 5)
    cfg_with_v2["chain_len_min_v2"] = 6
    cfg_with_v2["chain_len_max_v2"] = 12

    train_types = gen._types_for_role("train", 1)
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    plain1, _ = gen.build_split(rng1, train_types, 20, cfg_base)
    plain2, _ = gen.build_split(rng2, train_types, 20, cfg_with_v2)
    assert plain1 == plain2, "v2 keys must be ignored when world_version=1"


# ---------------------------------------------------------------------------
# world_version=2 type library
# ---------------------------------------------------------------------------

def test_world_v2_has_12_plus_train_types(gen):
    """world_version=2 must have at least 12 train types."""
    train_v2 = gen._types_for_role("train", 2)
    assert len(train_v2) >= 12, f"expected >=12 train types in v2, got {len(train_v2)}: {train_v2}"


def test_world_v2_has_4_near_ood_types(gen):
    """world_version=2 must have exactly 4 near-OOD types."""
    near_v2 = gen._types_for_role("near_ood", 2)
    assert len(near_v2) >= 4, f"expected >=4 near-OOD types in v2, got {len(near_v2)}: {near_v2}"


def test_world_v2_has_4_far_ood_types(gen):
    """world_version=2 must have exactly 4 far-OOD types."""
    far_v2 = gen._types_for_role("far_ood", 2)
    assert len(far_v2) >= 4, f"expected >=4 far-OOD types in v2, got {len(far_v2)}: {far_v2}"


def test_world_v1_still_has_7_train_types(gen):
    """world_version=1 must retain exactly the 7 campaign train types."""
    train_v1 = gen._types_for_role("train", 1)
    assert len(train_v1) == 7, f"expected 7 train types in v1, got {len(train_v1)}: {train_v1}"
    expected = {"dog", "cat", "horse", "fern", "lamp", "kettle", "box"}
    assert set(train_v1) == expected


def test_world_v2_type_disjointness(gen):
    """train / near-OOD / far-OOD must remain disjoint in v2."""
    train = set(gen._types_for_role("train", 2))
    near = set(gen._types_for_role("near_ood", 2))
    far = set(gen._types_for_role("far_ood", 2))
    assert train.isdisjoint(near), f"train/near overlap: {train & near}"
    assert train.isdisjoint(far),  f"train/far overlap: {train & far}"
    assert near.isdisjoint(far),   f"near/far overlap: {near & far}"


def test_world_v2_new_types_have_valid_schemas(gen):
    """All v2-only types must have valid schemas (all attrs in ATTRIBUTE_POOL)."""
    attr_pool = set(gen.ATTRIBUTE_POOL.keys())
    for name, d in gen.TYPE_LIBRARY.items():
        if d.get("world_version_min", 1) >= 2:
            for attr in d["schema"]:
                assert attr in attr_pool, f"type '{name}' has unknown attr '{attr}'"


def test_world_v2_near_ood_have_derived_from(gen):
    """Near-OOD types (v2 included) must document derived_from."""
    for name, d in gen.TYPE_LIBRARY.items():
        if d["split_role"] == "near_ood":
            assert "derived_from" in d, f"near-OOD type '{name}' missing derived_from"


# ---------------------------------------------------------------------------
# world_version=2 chain properties
# ---------------------------------------------------------------------------

def _make_cfg_v2():
    """Minimal v2 config for tests."""
    return {
        "world_version": 2,
        "chain_len_min": 4,
        "chain_len_max": 8,
        "chain_len_min_v2": 6,
        "chain_len_max_v2": 12,
        "entities_per_chain": (1, 2),
        "entities_per_chain_v2": (3, 5),
        "wait_weight": 0.15,
        "stochastic_v2": False,
        "stochastic_p": 0.15,
    }


def test_world_v2_chain_lengths_in_range(gen):
    """v2 chains must have lengths in [chain_len_min_v2, chain_len_max_v2]."""
    cfg = _make_cfg_v2()
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(7)
    plain, labeled = gen.build_split(rng, train_types, 100, cfg)
    for r in plain:
        n = len(r["chain"])
        assert cfg["chain_len_min_v2"] <= n <= cfg["chain_len_max_v2"], \
            f"chain length {n} outside [{cfg['chain_len_min_v2']}, {cfg['chain_len_max_v2']}]"


def test_world_v2_entities_per_chain_in_range(gen):
    """v2 labeled records must have 3-5 entity types per chain."""
    cfg = _make_cfg_v2()
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(7)
    _, labeled = gen.build_split(rng, train_types, 100, cfg)
    min_e, max_e = cfg["entities_per_chain_v2"]
    for rec in labeled:
        n_ent = len(rec["types"])
        assert min_e <= n_ent <= max_e, \
            f"expected {min_e}-{max_e} entities but got {n_ent}: {rec['types']}"


def test_world_v2_oracle_dist_present(gen):
    """v2 labeled records must have oracle_dist with the right length."""
    cfg = _make_cfg_v2()
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(7)
    _, labeled = gen.build_split(rng, train_types, 30, cfg)
    for rec in labeled:
        assert "oracle_dist" in rec, "oracle_dist missing from v2 labeled record"
        assert len(rec["oracle_dist"]) == len(rec["actions"]), \
            "oracle_dist length must equal actions length"


def test_world_v2_oracle_dist_deterministic_is_single_entry(gen):
    """When stochastic_v2=False, oracle_dist entries must be degenerate (single branch prob=1)."""
    cfg = _make_cfg_v2()
    cfg["stochastic_v2"] = False
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(7)
    _, labeled = gen.build_split(rng, train_types, 50, cfg)
    for rec in labeled:
        for step_dist in rec["oracle_dist"]:
            dist = step_dist["dist"]
            assert len(dist) == 1, \
                f"deterministic dist should have 1 entry, got {len(dist)}"
            assert abs(dist[0]["prob"] - 1.0) < 1e-6, \
                f"deterministic prob should be 1.0, got {dist[0]['prob']}"


def test_world_v2_oracle_replay_still_works(gen):
    """Oracle replay must reproduce the chain texts with v2 multi-entity chains."""
    cfg = _make_cfg_v2()
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(77)
    _, labeled = gen.build_split(rng, train_types, 50, cfg)
    for rec in labeled[:20]:
        type_names = rec["types"]
        initial_states = rec["initial_states"]
        actions = rec["actions"]
        texts = rec["chain"]
        snapshots = gen.replay_chain(type_names, initial_states, actions)
        # state_0
        assert gen.render_state(list(zip(type_names, snapshots[0]))) == texts[0]
        # subsequent states
        for i in range(1, len(texts)):
            rendered = gen.render_state(list(zip(type_names, snapshots[i])))
            assert texts[i].endswith(rendered), f"replay mismatch at step {i}"


def test_world_v2_labeled_has_initial_states(gen):
    """v2 labeled records must carry initial_states with correct length."""
    cfg = _make_cfg_v2()
    train_types = gen._types_for_role("train", 2)
    rng = random.Random(7)
    _, labeled = gen.build_split(rng, train_types, 30, cfg)
    for rec in labeled:
        assert "initial_states" in rec
        assert len(rec["initial_states"]) == len(rec["types"])


# ---------------------------------------------------------------------------
# world_version=1 backward-compat: no oracle_dist
# ---------------------------------------------------------------------------

def test_world_v1_no_oracle_dist(gen):
    """world_version=1 labeled records must NOT have oracle_dist (backward-compat)."""
    cfg = dict(gen.CONFIG)
    cfg["world_version"] = 1
    train_types = gen._types_for_role("train", 1)
    rng = random.Random(7)
    _, labeled = gen.build_split(rng, train_types, 20, cfg)
    for rec in labeled:
        assert "oracle_dist" not in rec, "oracle_dist must not appear in v1 records"


# ---------------------------------------------------------------------------
# build_manifest v2 correctness
# ---------------------------------------------------------------------------

def test_build_manifest_v2_lists_v2_types(gen):
    """build_manifest with world_version=2 must include v2 types and world_version field."""
    cfg = _make_cfg_v2()
    m = gen.build_manifest(cfg)
    assert m.get("world_version") == 2
    # All v2 train types should be in the manifest
    train_v2 = gen._types_for_role("train", 2)
    for tn in train_v2:
        assert tn in m["types"], f"train type '{tn}' missing from v2 manifest"


def test_build_manifest_v1_omits_v2_types(gen):
    """build_manifest with world_version=1 must not include v2-only types."""
    cfg = dict(gen.CONFIG)
    cfg["world_version"] = 1
    m = gen.build_manifest(cfg)
    # V2-only types must not appear.
    v2_only = [name for name, d in gen.TYPE_LIBRARY.items()
               if d.get("world_version_min", 1) >= 2]
    for tn in v2_only:
        assert tn not in m["types"], f"v2-only type '{tn}' leaked into v1 manifest"


# ---------------------------------------------------------------------------
# Config parse check for all 12 v4 configs
# ---------------------------------------------------------------------------

def test_v4_configs_parse():
    """All v4-family JSON configs must parse cleanly.

    The v4.0 family is the 8 `jepa_v4_*` configs ({v4,v4_blackbox}×{s0,s1,s2}+smokes);
    later campaigns (v4.0.1, v4.1, v4.2) add more `jepa_v4*` configs over time, so we
    assert the v4.0 set is present AND that EVERY matched v4-family config parses, rather
    than a brittle exact count that grows each campaign.
    """
    from twm.jepa.config import JEPAConfig
    configs_dir = REPO / "configs" / "jepa"
    # The frozen v4.0 set (exactly 8) must always exist.
    v40 = sorted(configs_dir.glob("jepa_v4_*.json"))
    assert len(v40) == 8, f"expected 8 v4.0 configs, found {len(v40)}: {[c.name for c in v40]}"
    # Every v4-family config (v4.0, v4.0.1, v4.1, v4.2, ...) must parse cleanly.
    v4_configs = sorted(configs_dir.glob("jepa_v4*.json"))
    for p in v4_configs:
        cfg = JEPAConfig.from_json(str(p))
        assert cfg is not None, f"config {p.name} failed to parse"


def test_v4_configs_seeds(gen):
    """v4_s0/s1/s2 configs must have the right seeds."""
    import json as _json
    configs_dir = REPO / "configs" / "jepa"
    for seed in [0, 1, 2]:
        p = configs_dir / f"jepa_v4_s{seed}.json"
        d = _json.loads(p.read_text())
        assert d["seed"] == seed, f"jepa_v4_s{seed}.json has wrong seed {d['seed']}"


def test_v4_smoke_is_fast(gen):
    """Smoke configs must have small max_chains and few epochs."""
    import json as _json
    for name in ["jepa_v4_smoke.json", "jepa_v4_blackbox_smoke.json"]:
        p = REPO / "configs" / "jepa" / name
        d = _json.loads(p.read_text())
        assert d["data"].get("max_chains", 9999) <= 64, f"{name}: max_chains too large"
        assert d["optim"]["epochs"] <= 5, f"{name}: epochs too large for smoke"


def test_v4_structured_has_targeted_actions(gen):
    """v4 structured (rotation_scale) configs must have use_targeted_actions=true."""
    import json as _json
    for seed in [0, 1, 2]:
        p = REPO / "configs" / "jepa" / f"jepa_v4_s{seed}.json"
        d = _json.loads(p.read_text())
        assert d["model"].get("use_targeted_actions") is True, \
            f"jepa_v4_s{seed} must have use_targeted_actions=true"


def test_v4_blackbox_has_no_targeted_actions(gen):
    """v4 blackbox configs must have use_targeted_actions=false (mask noop)."""
    import json as _json
    for seed in [0, 1, 2]:
        p = REPO / "configs" / "jepa" / f"jepa_v4_blackbox_s{seed}.json"
        d = _json.loads(p.read_text())
        assert d["model"].get("use_targeted_actions") is False, \
            f"jepa_v4_blackbox_s{seed} must have use_targeted_actions=false"
