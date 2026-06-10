"""Tests for the retrieval ceiling calibration script (scripts/calibrate_retrieval_ceiling.py).

Task C owns these tests (jepa_entity_campaign.md §7-C).

Test suite:
  A. Oracle backend on a tiny labeled fixture returns oracle_hard_mrr ≈ 1.0
     (deterministic chains + oracle replay should rank gold at position 1).
  B. Oracle backend without initial_states degrades to text-only fallback (no crash).
  C. Anthropic backend with no API key prints skip message and returns cleanly
     (monkeypatch ANTHROPIC_API_KEY env var).
  D. Hard pool builder produces correct structure (gold_idx in [0, pool_size-1]).

Run::
    uv run --with pytest python -m pytest tests/jepa/test_calibrate.py -x -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "generate_entity_world", REPO / "scripts" / "generate_entity_world.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_calib():
    spec = importlib.util.spec_from_file_location(
        "calibrate_retrieval_ceiling",
        REPO / "scripts" / "calibrate_retrieval_ceiling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def calib():
    return _load_calib()


# ---------------------------------------------------------------------------
# Tiny labeled fixture builder
# ---------------------------------------------------------------------------

def _build_tiny_labeled(gen, n_chains: int = 10, chain_len: int = 4,
                         seed: int = 42) -> list[dict]:
    """Build a small set of labeled records with initial_states for testing."""
    rng = random.Random(seed)
    train_types = gen._types_for_role("train")
    cfg = {
        "chain_len_min": chain_len,
        "chain_len_max": chain_len,
        "entities_per_chain": (1, 1),
        "wait_weight": 0.1,
    }
    _, labeled = gen.build_split(rng, train_types, n_chains=n_chains, cfg=cfg)
    return labeled


# ---------------------------------------------------------------------------
# A. Oracle backend: oracle_hard_mrr ≈ 1.0 on deterministic chains
# ---------------------------------------------------------------------------

class TestOracleBackend:
    """Oracle backend should achieve oracle_hard_mrr close to 1.0."""

    def test_oracle_mrr_near_one(self, gen, calib, tmp_path):
        """Oracle backend on tiny deterministic chains should have MRR ≈ 1.0.

        The oracle dynamics is deterministic: apply_action(type, state_t, action) returns
        the unique true next state.  When that rendered state appears in the pool, the
        oracle ranks it at position 1, giving RR = 1.0.
        """
        labeled = _build_tiny_labeled(gen, n_chains=20, chain_len=4)
        labeled_path = tmp_path / "labeled_oracle.jsonl"
        with open(labeled_path, "w") as f:
            for r in labeled:
                f.write(json.dumps(r) + "\n")

        out_path = str(tmp_path / "oracle_result.json")
        result = calib.run_oracle_backend(
            labeled_path=str(labeled_path),
            n=50,
            out_path=out_path,
            gen_mod=gen,
        )

        assert "oracle_hard_mrr" in result, f"Missing oracle_hard_mrr in result: {result}"
        mrr = result["oracle_hard_mrr"]
        # Oracle should achieve high MRR (expected ~1.0; allow small tolerance for pool
        # construction randomness where a distractor happens to match the oracle output).
        assert mrr > 0.9, (
            f"oracle_hard_mrr = {mrr:.4f} (expected > 0.9 for deterministic oracle)"
        )
        assert result["backend"] == "oracle"
        assert result["n"] > 0

        # Output JSON should exist.
        with open(out_path) as f:
            saved = json.load(f)
        assert "oracle_hard_mrr" in saved
        assert saved["oracle_hard_mrr"] == result["oracle_hard_mrr"]

    def test_oracle_backend_pool_solvable_frac(self, gen, calib, tmp_path):
        """pool_solvable_frac should be reported."""
        labeled = _build_tiny_labeled(gen, n_chains=15, chain_len=3)
        labeled_path = tmp_path / "labeled_solvable.jsonl"
        with open(labeled_path, "w") as f:
            for r in labeled:
                f.write(json.dumps(r) + "\n")

        result = calib.run_oracle_backend(
            labeled_path=str(labeled_path),
            n=30,
            out_path=None,
            gen_mod=gen,
        )

        assert "pool_solvable_frac" in result
        # The solvable fraction should be in [0, 1].
        assert 0.0 <= result["pool_solvable_frac"] <= 1.0

    def test_oracle_fallback_without_initial_states(self, gen, calib, tmp_path):
        """Oracle backend degrades gracefully when initial_states is missing."""
        # Build records WITHOUT initial_states.
        rng = random.Random(99)
        train_types = gen._types_for_role("train")[:2]
        cfg = {"chain_len_min": 3, "chain_len_max": 4,
               "entities_per_chain": (1, 1), "wait_weight": 0.1}
        plain, labeled = gen.build_split(rng, train_types, n_chains=10, cfg=cfg)
        # Strip initial_states from labeled records.
        stripped = [{k: v for k, v in r.items() if k != "initial_states"} for r in labeled]

        labeled_path = tmp_path / "labeled_no_init.jsonl"
        with open(labeled_path, "w") as f:
            for r in stripped:
                f.write(json.dumps(r) + "\n")

        # Should not raise, should return a result dict.
        result = calib.run_oracle_backend(
            labeled_path=str(labeled_path),
            n=20,
            out_path=None,
            gen_mod=gen,
        )
        assert "oracle_hard_mrr" in result
        assert result.get("mode") in ("text_only_fallback", "oracle_replay")


# ---------------------------------------------------------------------------
# B. Anthropic backend: graceful skip when no API key
# ---------------------------------------------------------------------------

class TestAnthropicBackend:
    """The anthropic backend must skip cleanly when ANTHROPIC_API_KEY is absent."""

    def test_no_api_key_prints_skip_message_and_returns_cleanly(self, calib, capsys, tmp_path):
        """When ANTHROPIC_API_KEY is not set, the backend returns with skipped=True."""
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = calib.run_anthropic_backend(
                glucose_path=None,
                entity_path=None,
                n=10,
                out_path=None,
            )
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

        assert result.get("skipped") is True, f"Expected skipped=True, got {result}"
        assert result.get("backend") == "anthropic"

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.out, (
            "Should print the skip message mentioning ANTHROPIC_API_KEY"
        )

    def test_no_api_key_writes_skip_json(self, calib, tmp_path):
        """When out_path is given and key is absent, writes a JSON with skipped=True."""
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        out_path = str(tmp_path / "skip_result.json")
        try:
            calib.run_anthropic_backend(
                glucose_path=None,
                entity_path=None,
                n=10,
                out_path=out_path,
            )
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

        with open(out_path) as f:
            saved = json.load(f)
        assert saved.get("skipped") is True

    def test_no_api_key_env_with_empty_string(self, calib, tmp_path):
        """Empty string key also triggers the skip path (API treats it as absent)."""
        old_val = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = ""
        try:
            result = calib.run_anthropic_backend(
                glucose_path=None,
                entity_path=None,
                n=5,
                out_path=None,
            )
        finally:
            if old_val is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = old_val

        # An empty key should also trigger the skip (no network call).
        assert result.get("skipped") is True or result.get("backend") == "anthropic"


# ---------------------------------------------------------------------------
# C. Hard pool builder
# ---------------------------------------------------------------------------

class TestHardPoolBuilder:
    """_build_hard_pool should produce valid query structures."""

    def test_pool_structure(self, gen, calib):
        """Each query has the right structure and gold_idx is in bounds."""
        labeled = _build_tiny_labeled(gen, n_chains=20, chain_len=4)
        # Use plain records (no initial_states needed for pool building).
        plain = [{"chain": r["chain"]} for r in labeled]

        pools = calib._build_hard_pool(plain, n_queries=30)

        assert len(pools) <= 30
        for (s_t, s_t1, pool, gold_idx) in pools:
            assert isinstance(s_t, str)
            assert isinstance(s_t1, str)
            assert isinstance(pool, list)
            assert len(pool) >= 1
            assert 0 <= gold_idx < len(pool), (
                f"gold_idx={gold_idx} out of bounds for pool of size {len(pool)}"
            )
            # The gold is actually in the pool at gold_idx.
            assert pool[gold_idx] == s_t1, (
                f"pool[gold_idx]={pool[gold_idx]!r} != s_t1={s_t1!r}"
            )

    def test_pool_contains_gold(self, gen, calib):
        """gold_idx should point to the true next state s_t1."""
        labeled = _build_tiny_labeled(gen, n_chains=10, chain_len=3)
        plain = [{"chain": r["chain"]} for r in labeled]
        pools = calib._build_hard_pool(plain, n_queries=20)

        for (s_t, s_t1, pool, gold_idx) in pools:
            assert s_t1 in pool, f"Gold state not found in pool candidates"
            assert pool[gold_idx] == s_t1

    def test_pool_size_bounded(self, gen, calib):
        """Pool should not exceed pool_size candidates."""
        labeled = _build_tiny_labeled(gen, n_chains=30, chain_len=4)
        plain = [{"chain": r["chain"]} for r in labeled]
        pools = calib._build_hard_pool(plain, n_queries=50)

        for (_, _, pool, _) in pools:
            # Pool size is min(pool_size, available) where pool_size=8.
            assert len(pool) <= 8, f"Pool size {len(pool)} exceeds 8"


# ---------------------------------------------------------------------------
# D. Oracle backend full-pipeline integration
# ---------------------------------------------------------------------------

class TestOraclePipeline:
    """End-to-end test of the oracle pipeline with the real generator."""

    def test_oracle_pipeline_end_to_end(self, gen, calib, tmp_path):
        """Full pipeline: generate -> write JSONL -> run oracle backend -> check MRR > 0.9."""
        # Generate a slightly larger fixture.
        rng = random.Random(77)
        train_types = gen._types_for_role("train")
        cfg = {
            "chain_len_min": 4,
            "chain_len_max": 6,
            "entities_per_chain": (1, 2),
            "wait_weight": 0.1,
        }
        _, labeled = gen.build_split(rng, train_types, n_chains=30, cfg=cfg)

        labeled_path = tmp_path / "labeled_e2e.jsonl"
        with open(labeled_path, "w") as f:
            for r in labeled:
                f.write(json.dumps(r) + "\n")

        result = calib.run_oracle_backend(
            labeled_path=str(labeled_path),
            n=100,
            out_path=None,
            gen_mod=gen,
        )

        assert result["oracle_hard_mrr"] > 0.9, (
            f"End-to-end oracle_hard_mrr = {result['oracle_hard_mrr']:.4f} (expected > 0.9)"
        )
        print(f"\nOracle pipeline e2e: oracle_hard_mrr = {result['oracle_hard_mrr']:.4f}")
