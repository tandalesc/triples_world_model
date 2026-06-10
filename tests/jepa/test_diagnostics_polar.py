"""Unit tests for the v2.1 polar per-factor diagnostics (research/jepa_v21_polar.md §5.2, §8.1).

These exercise the new diagnostic helpers directly (fast, no dataset/tokenizer):
  - _block_modulus_np / _block_phase_np match the torch forward-path definitions.
  - _phase_uniformity: 1.0 for a uniform phase distribution, ~0 + collapsed-block flags
    for a concentrated one (mean resultant length R). [§5.2.1]
  - _identity_persistence: a RotationOperator preserves modulus to < 1e-5 for all verbs
    WITH polar conditioning on (the load-bearing assertion); a scaling operator reports
    nonzero drift but no pure-rotation failure. [§8.1]
  - modulus effective rank via the shared _effective_rank helper. [§5.2.2]
"""

import numpy as np
import torch

from twm.jepa.diagnostics import (
    _block_modulus_np,
    _block_phase_np,
    _phase_uniformity,
    _identity_persistence,
    _effective_rank,
)
from twm.jepa import RotationScaleOperator, RotationOperator, PolarConditioner, block_modulus


def test_block_modulus_np_matches_torch():
    torch.manual_seed(0)
    k = torch.randn(5, 8, 32)
    m_torch = block_modulus(k).numpy()
    m_np = _block_modulus_np(k.numpy())
    assert np.allclose(m_torch, m_np, atol=1e-5)


def test_block_phase_np_range():
    k = np.random.RandomState(0).randn(10, 32)
    phi = _block_phase_np(k)
    assert phi.shape == (10, 16)
    assert phi.min() >= -np.pi - 1e-6 and phi.max() <= np.pi + 1e-6


def test_phase_uniformity_high_for_uniform():
    """Phases drawn uniformly -> mean resultant length R ≈ 0 -> phase_uniformity ≈ 1."""
    rng = np.random.RandomState(1)
    n, nb = 20000, 4
    phi = rng.uniform(-np.pi, np.pi, size=(n, nb))
    # build k from unit-modulus complex coords with these phases.
    k = np.empty((n, 2 * nb))
    k[:, 0::2] = np.cos(phi)
    k[:, 1::2] = np.sin(phi)
    out = _phase_uniformity(k, n=n)
    assert out["phase_uniformity"] > 0.95
    assert out["phase_collapsed_blocks"] == 0


def test_phase_uniformity_flags_collapse():
    """A concentrated phase -> R ≈ 1 -> low uniformity and a Rayleigh-flagged block."""
    n, nb = 20000, 4
    phi = np.full((n, nb), 0.3) + np.random.RandomState(2).randn(n, nb) * 0.01
    k = np.empty((n, 2 * nb))
    k[:, 0::2] = np.cos(phi)
    k[:, 1::2] = np.sin(phi)
    out = _phase_uniformity(k, n=n)
    assert out["phase_uniformity"] < 0.2
    assert out["phase_collapsed_blocks"] == nb


def test_modulus_effective_rank_helper():
    """Modulus-profile cov eff-rank: full-rank isotropic profiles -> rank ≈ nb."""
    rng = np.random.RandomState(3)
    M = np.abs(rng.randn(5000, 8))  # 8 independent modulus dims
    Mc = M - M.mean(0, keepdims=True)
    C = Mc.T @ Mc / (Mc.shape[0] - 1)
    er = _effective_rank(C)
    assert er > 6.0  # close to 8 for independent dims


class _ModelStub:
    """Minimal model surface for _identity_persistence: just .operator and .conditioner."""
    def __init__(self, operator, conditioner=None):
        self.operator = operator
        self.conditioner = conditioner


def test_identity_persistence_rotation_passes_with_conditioning():
    """RotationOperator (log_r≡0) preserves modulus to < 1e-5 for ALL verbs, even with a
    nonzero conditioned phase offset — the load-bearing §8.1 assertion."""
    rop = RotationOperator(n_verbs=8, d_noun=32)
    cond = PolarConditioner(n_blocks=16)
    cond.H.weight.data.normal_(0, 0.5)  # genuinely conditioned phase
    model = _ModelStub(rop, cond)
    torch.manual_seed(4)
    k = torch.randn(16, 8, 32)
    out = _identity_persistence(model, k)
    assert out["n_pure_rotation_verbs"] == 8
    assert out["identity_persistence_err"] < 1e-5
    assert out["identity_persistence_pass"] is True


def test_identity_persistence_scaling_reports_drift():
    """A RotationScaleOperator with nonzero log_r has no pure-rotation verbs, so the
    assertion passes vacuously but per-verb modulus drift is reported and nonzero."""
    op = RotationScaleOperator(n_verbs=4, d_noun=32)
    op.log_r.data.fill_(0.4)  # every verb scales -> no pure-rotation verb
    model = _ModelStub(op, conditioner=None)
    torch.manual_seed(5)
    k = torch.randn(16, 8, 32)
    out = _identity_persistence(model, k)
    assert out["n_pure_rotation_verbs"] == 0
    assert out["identity_persistence_pass"] is True  # vacuous (no rotation verbs)
    assert max(out["modulus_drift_per_verb"]) > 0.1  # scaling moves modulus
