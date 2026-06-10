"""Unit tests for JEPA losses (T3). Spec §3 + §12 row T3.

Covers:
- SIGReg(standard normal) is small; SIGReg(rank-collapsed) >> it.
- SIGReg gradient is nonzero on standardized inputs.
- Negative test: sphere-projected input saturates the GoF (documents WHY we
  standardize instead of L2-projecting — the decisive judge flaw).
- L_div usage entropy penalizes single-verb routing.
- Gumbel anneal schedule endpoints.
- JEPALoss end-to-end shapes / components.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from twm.jepa.legacy.losses_v1 import (
    JEPALoss,
    anneal_tau,
    gumbel_softmax_sample,
    sigreg_loss,
    spread_penalty,
    usage_entropy,
)


def _seed():
    torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# SIGReg                                                                       #
# --------------------------------------------------------------------------- #

def test_sigreg_isotropic_small_vs_collapsed_large():
    _seed()
    dn = 32
    iso = torch.randn(4096, dn)
    # Bottleneck collapse looks like a few discrete codes, NOT a low-rank Gaussian
    # (random projections of a low-rank *Gaussian* stay Gaussian, so SIGReg can't
    # and shouldn't fire on that). The real failure SIGReg detects is the
    # non-Gaussianity of a clustered / discrete bottleneck — per-dim standardize
    # cannot Gaussianize a multimodal distribution.
    centers = torch.randn(4, dn) * 3.0
    assign = torch.randint(0, 4, (4096,))
    collapsed = centers[assign] + 0.05 * torch.randn(4096, dn)

    l_iso = sigreg_loss(iso, n_slices=256)
    l_collapsed = sigreg_loss(collapsed, n_slices=256)

    # Isotropic Gaussian should be near zero after standardization.
    assert l_iso.item() < 5e-3, f"isotropic SIGReg not small: {l_iso.item()}"
    # Clustered collapse must be detectable — much larger than isotropic.
    assert l_collapsed.item() > 10.0 * l_iso.item(), (
        f"collapsed {l_collapsed.item()} not >> isotropic {l_iso.item()}"
    )


def test_sigreg_gradient_nonzero_on_standardized():
    _seed()
    dn = 32
    # Anisotropic so there is something for the gradient to push on.
    x = torch.randn(1024, dn, requires_grad=True)
    scale = torch.linspace(0.2, 4.0, dn)
    loss = sigreg_loss(x * scale, n_slices=128)
    loss.backward()
    assert x.grad is not None
    g = x.grad.norm().item()
    assert g > 1e-6, f"SIGReg gradient vanished on standardized input: {g}"


def test_sigreg_sphere_projection_saturates_negative_test():
    """DOCUMENTED NEGATIVE TEST (spec §3): if nouns are L2-projected onto the unit
    sphere, every 1D projection has std ≈ 1/√dn and the N(0,1)-calibrated GoF fires
    constantly with a near-dead gradient. We do NOT project to the sphere in the
    real loss (standardize=True); this test exists to prove the failure mode and
    justify the standardize precondition.
    """
    _seed()
    dn = 32
    x = torch.randn(2048, dn)
    sphere = x / x.norm(dim=-1, keepdim=True)

    # standardize=False simulates feeding the raw (sphere-projected) samples
    # straight into the GoF — the un-preconditioned path the judges flagged.
    l_sphere_raw = sigreg_loss(sphere, n_slices=256, standardize=False)
    # With standardize=True the loss is brought back to a healthy scale.
    l_sphere_std = sigreg_loss(sphere, n_slices=256, standardize=True)

    # The un-standardized sphere path saturates (large, calibrated to N(0,1) but
    # the projections have std 1/√dn ≈ 0.18) — much larger than the standardized
    # path. This is exactly why we standardize.
    assert l_sphere_raw.item() > l_sphere_std.item(), (
        f"sphere-projected raw GoF should saturate above standardized: "
        f"raw={l_sphere_raw.item()} std={l_sphere_std.item()}"
    )
    assert l_sphere_raw.item() > 0.1, (
        f"sphere-projected raw GoF expected to be saturated/large: {l_sphere_raw.item()}"
    )


# --------------------------------------------------------------------------- #
# L_div                                                                        #
# --------------------------------------------------------------------------- #

def test_usage_entropy_penalizes_single_verb_routing():
    v = 8
    # Single-verb routing: all slots route to verb 0.
    single = torch.zeros(64, v)
    single[:, 0] = 1.0
    # Uniform routing: all verbs used equally.
    uniform = torch.full((64, v), 1.0 / v)

    pen_single = usage_entropy(single)
    pen_uniform = usage_entropy(uniform)

    # Penalty is -H(p̄). Single-verb => H≈0 => penalty≈0 (worst).
    # Uniform => H=log V => penalty=-log V (best, most negative).
    assert pen_single.item() > pen_uniform.item(), (
        f"single-verb penalty {pen_single.item()} should exceed uniform "
        f"{pen_uniform.item()}"
    )
    assert math.isclose(pen_uniform.item(), -math.log(v), rel_tol=1e-4)
    assert abs(pen_single.item()) < 1e-4


def test_spread_penalty_identity_vs_spread():
    dn = 32
    half = dn // 2
    # All verbs at identity (θ=0, log r=0): max identity-proximity penalty.
    theta_id = torch.zeros(4, half)
    log_r_id = torch.zeros(4, half)
    # Spread-out verbs: distinct, away from origin and each other.
    g = torch.Generator().manual_seed(1)
    theta_spread = torch.randn(4, half, generator=g) * 1.5
    log_r_spread = torch.randn(4, half, generator=g) * 1.5

    pen_id = spread_penalty(theta_id, log_r_id)
    pen_spread = spread_penalty(theta_spread, log_r_spread)
    assert pen_id.item() > pen_spread.item(), (
        f"identity operators {pen_id.item()} should be penalized over spread "
        f"{pen_spread.item()}"
    )


# --------------------------------------------------------------------------- #
# Gumbel anneal + sampling                                                     #
# --------------------------------------------------------------------------- #

def test_anneal_schedule_endpoints():
    total = 1000
    # Start of training -> tau_start.
    assert math.isclose(anneal_tau(0, total, 2.0, 0.5, 0.3), 2.0, rel_tol=1e-6)
    # End of anneal window (30% of steps) -> tau_end, and held flat after.
    assert math.isclose(anneal_tau(300, total, 2.0, 0.5, 0.3), 0.5, rel_tol=1e-6)
    assert math.isclose(anneal_tau(999, total, 2.0, 0.5, 0.3), 0.5, rel_tol=1e-6)
    # Midpoint of the anneal window -> halfway between.
    mid = anneal_tau(150, total, 2.0, 0.5, 0.3)
    assert math.isclose(mid, 1.25, rel_tol=1e-6), mid
    # Monotone non-increasing.
    prev = float("inf")
    for s in range(0, total, 50):
        cur = anneal_tau(s, total, 2.0, 0.5, 0.3)
        assert cur <= prev + 1e-9
        prev = cur


def test_gumbel_softmax_shapes_and_hard():
    _seed()
    logits = torch.randn(4, 8, 16)  # (B, M, V)
    soft = gumbel_softmax_sample(logits, tau=1.0, hard=False)
    assert soft.shape == logits.shape
    assert torch.allclose(soft.sum(-1), torch.ones(4, 8), atol=1e-4)

    hard = gumbel_softmax_sample(logits, tau=0.5, hard=True)
    assert hard.shape == logits.shape
    # Straight-through hard forward value is one-hot.
    assert torch.allclose(hard.sum(-1), torch.ones(4, 8), atol=1e-4)
    maxv, _ = hard.max(-1)
    assert torch.allclose(maxv, torch.ones(4, 8), atol=1e-4)


def test_gumbel_softmax_carries_gradient():
    _seed()
    logits = torch.randn(4, 8, 16, requires_grad=True)
    soft = gumbel_softmax_sample(logits, tau=1.0, hard=False)
    soft.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# JEPALoss end-to-end                                                          #
# --------------------------------------------------------------------------- #

class _FakeOperator(nn.Module):
    """Minimal stand-in for RotationScaleOperator exposing theta / log_r."""

    def __init__(self, v: int, dn: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(v, dn // 2) * 0.5)
        self.log_r = nn.Parameter(torch.randn(v, dn // 2) * 0.1)


def test_jepaloss_forward_components_and_backward():
    _seed()
    b, m, dn, v = 8, 6, 32, 8
    op = _FakeOperator(v, dn)
    loss_fn = JEPALoss(operator=op, n_slices=64)

    k = torch.randn(b, m, dn, requires_grad=True)
    verb_logits = torch.randn(b, m, v, requires_grad=True)
    zhat = torch.randn(b, dn, requires_grad=True)
    z_target = torch.randn(b, dn)

    total, comp = loss_fn(k, verb_logits, zhat, z_target, gumbel_tau=1.0)
    for key in ("loss", "L_pred", "L_sigreg", "L_div", "L_entropy", "L_spread"):
        assert key in comp, f"missing component {key}"
    assert total.requires_grad
    total.backward()
    # Gradient must reach the prediction (L_pred), the nouns (L_sigreg) and the
    # verb logits (L_div entropy) and the operator codebook (L_div spread).
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert verb_logits.grad is not None and verb_logits.grad.abs().sum() > 0
    assert op.theta.grad is not None and op.theta.grad.abs().sum() > 0


def test_jepaloss_zero_target_zero_pred_has_zero_l_pred():
    _seed()
    b, m, dn, v = 4, 4, 32, 8
    op = _FakeOperator(v, dn)
    loss_fn = JEPALoss(operator=op, n_slices=32, w_sigreg=0.0, w_div=0.0)
    k = torch.randn(b, m, dn)
    verb_logits = torch.randn(b, m, v)
    z = torch.randn(b, dn)
    total, comp = loss_fn(k, verb_logits, z, z, gumbel_tau=1.0)
    assert math.isclose(comp["L_pred"], 0.0, abs_tol=1e-6)


def test_jepaloss_no_operator_safe():
    """L_div spread degrades gracefully to entropy-only when no operator given."""
    _seed()
    b, m, dn, v = 4, 4, 32, 8
    loss_fn = JEPALoss(operator=None, n_slices=32)
    k = torch.randn(b, m, dn)
    verb_logits = torch.randn(b, m, v)
    zhat = torch.randn(b, dn)
    z = torch.randn(b, dn)
    total, comp = loss_fn(k, verb_logits, zhat, z, gumbel_tau=1.0)
    assert comp["L_spread"] == 0.0
    assert torch.isfinite(total)
