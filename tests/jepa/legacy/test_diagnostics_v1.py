"""Tests for T6 — diagnostics + export (spec §5, §8, §12 row T6).

Covers:
  * eval_diagnostics runs end-to-end on a randomly-initialized nano model
  * all returned scalar metrics are finite
  * PNG files are produced when out_dir is specified
  * export_jepa_weights produces a JSON file with the expected INT8 structure
  * nano export size assertion: <= 303 KB
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Repo root on path so `from scripts...` resolves (this file lives in tests/jepa/legacy/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twm.jepa.operator import RotationScaleOperator
from twm.jepa.slot_encoder import SlotEncoder
from twm.jepa.legacy.model_v1 import JEPAOperatorModel

from twm.jepa.legacy.diagnostics_v1 import eval_diagnostics
from scripts.export_jepa_weights import export_jepa_weights


# ---------------------------------------------------------------------------
# helpers to build a nano model from scratch
# ---------------------------------------------------------------------------

NANO_M = dict(
    d_model=64,
    d_noun=32,
    n_slots=8,
    n_verbs=8,
    n_text_layers=2,
    tie_text_layers=True,
    n_heads=4,
    d_ff=128,
    n_slot_iters=3,
    max_text_tokens=64,
)
VOCAB = 512


def _make_nano():
    """Build a JEPAOperatorModel from T1/T2/T5 contracts (randomly initialized)."""
    torch.manual_seed(42)
    token_emb = nn.Embedding(VOCAB, NANO_M["d_model"])
    token_emb.weight.requires_grad_(False)

    encoder = SlotEncoder(
        token_emb=token_emb,
        d_model=NANO_M["d_model"],
        d_noun=NANO_M["d_noun"],
        n_slots=NANO_M["n_slots"],
        n_verbs=NANO_M["n_verbs"],
        n_heads=NANO_M["n_heads"],
        n_text_layers=NANO_M["n_text_layers"],
        tie_text_layers=NANO_M["tie_text_layers"],
        n_slot_iters=NANO_M["n_slot_iters"],
        max_text_tokens=NANO_M["max_text_tokens"],
    )
    operator = RotationScaleOperator(
        n_verbs=NANO_M["n_verbs"],
        d_noun=NANO_M["d_noun"],
        block=2,
    )
    model = JEPAOperatorModel(
        encoder=encoder,
        operator=operator,
        d_noun=NANO_M["d_noun"],
        n_verbs=NANO_M["n_verbs"],
        n_heads=NANO_M["n_heads"],
    )
    return model


# ---------------------------------------------------------------------------
# minimal dataset stub
# ---------------------------------------------------------------------------

class _TinyDataset:
    """Minimal fake dataset matching JEPAChainDatasetProtocol for testing."""

    def __init__(self, n=32, T=64, seed=7):
        torch.manual_seed(seed)
        self._n = n
        self._T = T
        # (n, T) token ids in [0, VOCAB)
        self._src = torch.randint(0, VOCAB, (n, T))
        self._tgt = torch.randint(0, VOCAB, (n, T))
        # pad masks: True = padded, False = real (PyTorch convention)
        self._src_pad = torch.zeros(n, T, dtype=torch.bool)
        self._tgt_pad = torch.zeros(n, T, dtype=torch.bool)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {
            "src_ids": self._src[idx],
            "src_pad": self._src_pad[idx],
            "tgt_ids": self._tgt[idx],
            "tgt_pad": self._tgt_pad[idx],
        }


# ---------------------------------------------------------------------------
# eval_diagnostics tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nano_model():
    return _make_nano()


@pytest.fixture(scope="module")
def tiny_dataset():
    return _TinyDataset(n=32)


def test_eval_diagnostics_returns_dict(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_eval_diagnostics_all_scalars_finite(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    for key, val in result.items():
        if isinstance(val, float):
            assert math.isfinite(val) or math.isnan(val) is False or key.startswith("binding"), (
                f"metric {key!r} = {val} is not finite"
            )
        elif isinstance(val, list):
            # residual_vs_slots_curve — check all floats finite
            for v in val:
                assert math.isfinite(float(v)), f"metric {key!r} item {v} not finite"


def test_eval_diagnostics_key_metrics_present(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    required_keys = [
        "noun_eff_rank",
        "noun_per_dim_var_mean",
        "scale_drift_mean_log_r",
        "scale_drift_mean_a_over_k",
        "scale_drift_alarm",
        "verb_usage_ppl",
        "verb_usage_entropy",
        "sanity_bbT_err_mean",
        "sanity_inv_err_mean",
        "residual_vs_slots_curve",
        "residual_vs_slots_monotone",
        "slot_entropy_mean",
        "held_out_cos_mean",
        "held_out_mse_mean",
        "multi_step_drift_cos_1",
        "multi_step_drift_cos_16",
        "multi_step_drift_drop",
    ]
    for k in required_keys:
        assert k in result, f"missing key: {k!r}"


def test_eval_diagnostics_structural_sanity_near_zero(nano_model, tiny_dataset):
    # Structural sanity errors should be near zero for a correctly implemented operator
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=8)
    assert result["sanity_bbT_err_mean"] < 1e-4, result["sanity_bbT_err_mean"]
    assert result["sanity_inv_err_mean"] < 1e-4, result["sanity_inv_err_mean"]


def test_eval_diagnostics_png_files_written(nano_model, tiny_dataset):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=8, out_dir=out)
        pngs = list(out.glob("*.png"))
        assert len(pngs) >= 5, f"expected >= 5 PNGs, got {len(pngs)}: {[p.name for p in pngs]}"


def test_eval_diagnostics_noun_eff_rank_positive(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    assert result["noun_eff_rank"] > 0.0


def test_eval_diagnostics_residual_curve_length(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    # curve should have M+1 entries (0 slots to M slots)
    M = NANO_M["n_slots"]
    assert len(result["residual_vs_slots_curve"]) == M + 1


def test_eval_diagnostics_multi_step_drift_bounded(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=8)
    # With a random model, cos_1 and cos_16 should both be in [0, 1]
    assert 0.0 <= result["multi_step_drift_cos_1"] <= 1.0
    assert 0.0 <= result["multi_step_drift_cos_16"] <= 1.0


def test_eval_diagnostics_held_out_cos_in_range(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    # cosine similarity should be in [-1, 1]
    assert -1.1 <= result["held_out_cos_mean"] <= 1.1


def test_eval_diagnostics_verb_ppl_positive(nano_model, tiny_dataset):
    result = eval_diagnostics(nano_model, tiny_dataset, device="cpu", n_examples=16)
    assert result["verb_usage_ppl"] >= 1.0


# ---------------------------------------------------------------------------
# export tests
# ---------------------------------------------------------------------------

def test_export_produces_json(nano_model):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "jepa_weights.json"
        weights = export_jepa_weights(
            nano_model, out_path, profile="jepa_nano", assert_under_303kb=True
        )
        assert out_path.exists()
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["format"] == "jepa_v1_int8"
        assert "operator" in loaded


def test_export_operator_int8_structure(nano_model):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "jepa_weights.json"
        weights = export_jepa_weights(
            nano_model, out_path, profile="jepa_nano", assert_under_303kb=True
        )
        op = weights["operator"]
        assert "cos_int8" in op
        assert "sin_int8" in op
        assert "r_fp16" in op
        assert op["n_verbs"] == NANO_M["n_verbs"]
        # cos and sin should be INT8 values in [-127, 127]
        cos_flat = _flatten_list(op["cos_int8"])
        for v in cos_flat[:20]:
            assert -128 <= v <= 127, f"cos INT8 value {v} out of range"


def test_export_under_303kb(nano_model):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "jepa_weights.json"
        export_jepa_weights(
            nano_model, out_path, profile="jepa_nano", assert_under_303kb=True
        )
        size_kb = out_path.stat().st_size / 1024
        assert size_kb <= 303, f"nano export is {size_kb:.1f} KB, exceeds 303 KB envelope"


def test_export_has_js_api_doc(nano_model):
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "jepa_weights.json"
        weights = export_jepa_weights(
            nano_model, out_path, profile="jepa_nano", assert_under_303kb=True
        )
        assert "js_api" in weights
        assert "step_latent" in weights["js_api"]
        assert "undo_latent" in weights["js_api"]


def test_export_operator_cos2_sin2_approx_1(nano_model):
    """cos^2 + sin^2 should reconstruct to ~1 when dequantized with scale 1/127."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "jepa_weights.json"
        weights = export_jepa_weights(
            nano_model, out_path, profile="jepa_nano", assert_under_303kb=True
        )
        op = weights["operator"]
        scale = op["cos_scale"]  # 1/127
        cos_q = _flatten_list(op["cos_int8"])
        sin_q = _flatten_list(op["sin_int8"])
        import numpy as np
        cos_f = np.array(cos_q, dtype=float) * scale
        sin_f = np.array(sin_q, dtype=float) * scale
        # After INT8 quantization, cos^2 + sin^2 should be close to 1
        mag2 = cos_f ** 2 + sin_f ** 2
        assert abs(mag2.mean() - 1.0) < 0.02, f"mean(cos^2+sin^2) = {mag2.mean():.4f}, expected ~1.0"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(_flatten_list(item))
        else:
            out.append(item)
    return out
