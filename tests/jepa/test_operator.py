"""Unit tests for the JEPA rotation+scale operator (spec §12 T1, §1).

Covers the structural-inverse guarantee (no consistency loss is needed because the
inverse is exact), the soft-mix expected-operator identity, the T=1 integrate seam,
the fp32-under-autocast guard, and bake round-trip.
"""

import math

import pytest
import torch

from twm.jepa.operator import (
    RotationScaleOperator,
    RotationOperator,
    SOnCayleyOperator,
)


def _make(n_verbs=8, d_noun=32, seed=0):
    torch.manual_seed(seed)
    return RotationScaleOperator(n_verbs=n_verbs, d_noun=d_noun, block=2)


def test_apply_shape_and_hard_verbs():
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a = op.apply(k, v)
    assert a.shape == k.shape


def test_structural_inverse_BBinv_identity():
    # ||B B^{-1} - I|| < 1e-5 for every verb.
    op = _make()
    for v in range(op.n_verbs):
        san = op.structural_sanity(v)
        assert san["bbT_err"] < 1e-5, (v, san["bbT_err"])
        assert san["inv_err"] < 1e-5, (v, san["inv_err"])


def test_exact_undo_round_trip():
    # k -> B_v k -> B_v^{-1}(B_v k) recovers k exactly (structural inverse).
    op = _make()
    k = torch.randn(4, 8, 32, dtype=torch.float64).float()
    v = torch.randint(0, op.n_verbs, (4, 8))
    a = op.apply(k, v)
    k_rt = op.inverse_apply(a, v)
    assert torch.allclose(k_rt, k, atol=1e-5), (k_rt - k).abs().max().item()


def test_inverse_is_negate_theta_and_logr():
    # inverse_apply must equal applying a verb with (-theta, -log_r).
    op = _make()
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))
    inv = op.inverse_apply(k, v)

    # Build a mirror operator with negated params and compare to its forward apply.
    mirror = RotationScaleOperator(op.n_verbs, op.d_noun, block=2)
    with torch.no_grad():
        mirror.theta.copy_(-op.theta)
        mirror.log_r.copy_(-op.log_r)
    fwd = mirror.apply(k, v)
    assert torch.allclose(inv, fwd, atol=1e-6)


def test_integrate_T1_equals_apply_bitwise_close():
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a_apply = op.apply(k, v)
    a_int = op.integrate(k, v, T=1)
    # T=1 fast-path is the same call -> exactly equal.
    assert torch.equal(a_apply, a_int)


def test_integrate_T_greater_than_1_runs():
    # Loop + per-step generator hook present and dormant; just verify it executes.
    op = _make()
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, op.n_verbs, (2, 8))
    out = op.integrate(k, v, T=4)
    assert out.shape == k.shape


def test_soft_mix_onehot_equals_hard_apply():
    # Soft-mix with a one-hot distribution must equal the hard-index apply.
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    p = torch.nn.functional.one_hot(v, num_classes=op.n_verbs).float()  # (B,M,V)
    a_hard = op.apply(k, v)
    a_soft = op.apply(k, p)
    assert torch.allclose(a_hard, a_soft, atol=1e-6), (a_hard - a_soft).abs().max().item()


def test_soft_mix_is_expected_operator():
    # E_v[r·R(θ)] k = sum_v p_v (B_v k) for the block-linear form (documented in §1.6).
    op = _make()
    k = torch.randn(2, 3, 32)
    p = torch.rand(2, 3, op.n_verbs)
    p = p / p.sum(-1, keepdim=True)  # normalize to a distribution

    a_soft = op.apply(k, p)

    # Explicit expectation: sum_v p_v * apply(k, verb=v).
    acc = torch.zeros_like(a_soft)
    for v in range(op.n_verbs):
        idx = torch.full((2, 3), v, dtype=torch.long)
        acc = acc + p[..., v].unsqueeze(-1) * op.apply(k, idx)
    assert torch.allclose(a_soft, acc, atol=1e-5), (a_soft - acc).abs().max().item()


def test_fp32_autocast_guard():
    # Under bf16 autocast the operator math must still run in fp32 and return finite,
    # correct results (mirrors the VQ gotcha). CPU bf16 autocast is the portable check.
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        a = op.apply(k, v)
        k_rt = op.inverse_apply(a, v)
    assert torch.isfinite(a).all()
    # Round-trip still exact despite the outer autocast context.
    assert torch.allclose(k_rt, k, atol=1e-4), (k_rt - k).abs().max().item()


def test_bake_round_trip():
    # bake() returns (cos, sin, r) per verb/block; reconstructing apply from them
    # must reproduce op.apply for hard verbs.
    op = _make()
    baked = op.bake()
    cos, sin, r = baked["cos"], baked["sin"], baked["r"]
    assert cos.shape == (op.n_verbs, op.d_noun // 2)
    assert sin.shape == (op.n_verbs, op.d_noun // 2)
    assert r.shape == (op.n_verbs, op.d_noun // 2)
    # cos^2 + sin^2 == 1 (structural rotation orthogonality).
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-6)

    k = torch.randn(2, 8, 32)
    v = torch.randint(0, op.n_verbs, (2, 8))
    a_ref = op.apply(k, v)

    # Reconstruct apply from baked tables: (x', y') = (r(x cos - y sin), r(x sin + y cos)).
    a_blk = cos[v]  # (2,8,n_blocks)
    b_blk = sin[v]
    r_blk = r[v]
    xpair = k.reshape(2, 8, op.d_noun // 2, 2)
    xc, yc = xpair[..., 0], xpair[..., 1]
    out_x = r_blk * (xc * a_blk - yc * b_blk)
    out_y = r_blk * (xc * b_blk + yc * a_blk)
    out = torch.stack([out_x, out_y], dim=-1).reshape(2, 8, op.d_noun)
    assert torch.allclose(out, a_ref, atol=1e-5), (out - a_ref).abs().max().item()


def test_init_excludes_near_identity_angles():
    # θ init excludes |θ| < 0.1 (avoid identity rotation), within (-π/2, π/2).
    op = _make(seed=3)
    assert (op.theta.abs() >= 0.1 - 1e-6).all()
    assert (op.theta.abs() <= math.pi / 2 + 1e-6).all()
    # log_r ~ N(0, 0.1): small, centered near 0 (r near 1).
    assert op.log_r.abs().mean() < 0.5


def test_rotation_operator_is_norm_preserving():
    # RotationOperator freezes log_r=0 -> r=1 -> norm preserving.
    torch.manual_seed(0)
    op = RotationOperator(n_verbs=8, d_noun=32, block=2)
    assert not op.log_r.requires_grad
    assert torch.equal(op.log_r, torch.zeros_like(op.log_r))
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a = op.apply(k, v)
    assert torch.allclose(k.norm(dim=-1), a.norm(dim=-1), atol=1e-5)


def test_son_cayley_stub_raises():
    op = SOnCayleyOperator(n_verbs=8, d_noun=32)
    k = torch.randn(1, 8, 32)
    v = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        op.apply(k, v)
    with pytest.raises(NotImplementedError):
        op.bake()


def test_odd_d_noun_and_bad_block_rejected():
    with pytest.raises(ValueError):
        RotationScaleOperator(n_verbs=4, d_noun=31, block=2)
    with pytest.raises(ValueError):
        RotationScaleOperator(n_verbs=4, d_noun=32, block=4)


def test_n_verbs_property():
    op = _make(n_verbs=16)
    assert op.n_verbs == 16
