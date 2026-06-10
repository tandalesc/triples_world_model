"""Unit tests for GatedMLPTransition — the black-box transition baseline (v3 §4).

Owned by Task C. Covers the Operator-ABC contract (apply/velocity/integrate), the NO
INVERSE contract (inverse_apply/bake raise; structural_sanity NaN), the theta_offset
accepted-and-ignored signature parity, near-identity init, the param-match budget, and
the operator_group="gated_mlp" end-to-end build + unroll through the identical pipeline.
"""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from twm.jepa.baseline_transition import GatedMLPTransition


def _make(n_verbs=8, d_noun=32, d_e=4, d_h=8, seed=0):
    torch.manual_seed(seed)
    return GatedMLPTransition(n_verbs=n_verbs, d_noun=d_noun, d_e=d_e, d_h=d_h)


# =========================================================================== apply
def test_apply_shape_hard_and_soft():
    op = _make()
    k = torch.randn(4, 8, 32)
    v_hard = torch.randint(0, op.n_verbs, (4, 8))
    a_hard = op.apply(k, v_hard)
    assert a_hard.shape == k.shape
    assert torch.isfinite(a_hard).all()
    # soft (B,M,V) float path (Gumbel ST) is accepted.
    p = F.one_hot(v_hard, num_classes=op.n_verbs).float()
    a_soft = op.apply(k, p)
    assert a_soft.shape == k.shape
    # one-hot soft == hard apply (expected-verb-embedding identity for a one-hot).
    assert torch.allclose(a_hard, a_soft, atol=1e-5), (a_hard - a_soft).abs().max().item()


def test_theta_offset_accepted_and_ignored():
    """apply accepts theta_offset for polar-call-site signature parity but IGNORES it:
    a nonzero offset must NOT change the output (the MLP has no phase split, §4.1)."""
    op = _make()
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))
    a_none = op.apply(k, v)
    a_off = op.apply(k, v, theta_offset=torch.randn(3, 8, 16) * 5.0)
    assert torch.equal(a_none, a_off), (a_none - a_off).abs().max().item()


def test_near_identity_at_init():
    """Large-negative gate init ⟹ sigmoid(gate) ≈ 0 ⟹ a* ≈ k at init (§4.2)."""
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a = op.apply(k, v)
    rel = (a - k).norm() / k.norm()
    assert rel < 0.1, rel.item()  # sigmoid(-4)≈0.018 -> small perturbation of k


def test_gate_init_is_large_negative():
    op = _make()
    assert torch.allclose(op.gate, torch.full_like(op.gate, GatedMLPTransition.GATE_INIT))


def test_apply_is_differentiable_into_params():
    op = _make()
    k = torch.randn(2, 8, 32, requires_grad=True)
    v = torch.randint(0, op.n_verbs, (2, 8))
    op.apply(k, v).pow(2).mean().backward()
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert op.W1.weight.grad is not None and op.W1.weight.grad.abs().sum() > 0
    assert op.gate.grad is not None  # gate participates in the output


def test_soft_path_gradient_into_verb_logits():
    """The soft (B,M,V) path keeps the ST gradient flowing into the verb distribution."""
    op = _make()
    k = torch.randn(2, 8, 32)
    p = torch.randn(2, 8, op.n_verbs, requires_grad=True).softmax(-1)
    p.retain_grad()
    op.apply(k, p).pow(2).mean().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


# =========================================================================== NO INVERSE
def test_inverse_apply_raises():
    op = _make()
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, op.n_verbs, (2, 8))
    a = op.apply(k, v)
    with pytest.raises(NotImplementedError, match="no structural inverse"):
        op.inverse_apply(a, v)


def test_bake_raises():
    op = _make()
    with pytest.raises(NotImplementedError, match="not JS-exportable"):
        op.bake()


def test_structural_sanity_is_nan():
    """No inverse ⟹ invertibility metrics undefined: NaN-filled (must NOT crash)."""
    op = _make()
    san = op.structural_sanity(0)
    assert set(san) == {"bbT_err", "inv_err"}
    assert math.isnan(san["bbT_err"])
    assert math.isnan(san["inv_err"])


# =========================================================================== seam parity
def test_velocity_matches_apply_minus_k():
    op = _make()
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))
    assert torch.allclose(op.velocity(k, v), op.apply(k, v) - k, atol=1e-6)


def test_integrate_T1_equals_apply_and_T_gt_1_raises():
    op = _make()
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, op.n_verbs, (2, 8))
    assert torch.equal(op.integrate(k, v, T=1), op.apply(k, v))
    with pytest.raises(NotImplementedError, match="no multi-step integrator"):
        op.integrate(k, v, T=2)


# =========================================================================== budget / type
def test_param_count_within_2x_of_operator():
    """832 transition params (1.6×) — within the ~2× band over the operator's
    op-codebook(256)+polar-H(256)=512 transition params (§4.2)."""
    op = _make()
    n = sum(p.numel() for p in op.parameters())
    # documented figure: verb_emb 32 + W1 288 + W2 256 + gate 256 = 832.
    assert n == 832, n
    operator_transition_params = 256 + 256  # op codebook (2·V·nb=2·8·16) + polar H (nb²=16²)
    assert 512 <= n <= 2 * operator_transition_params + 1, n  # within ~2× (≤ 1024-ish band)


def test_param_breakdown():
    op = _make()
    assert op.verb_emb.weight.numel() == 32   # V·d_e = 8·4
    assert op.W1.weight.numel() == (32 + 4) * 8  # (dn+d_e)·d_h = 36·8 = 288, bias-free
    assert op.W1.bias is None                 # bias-free to match the §4.2 832 budget
    assert op.W2.weight.numel() == 8 * 32     # d_h·dn = 256
    assert op.W2.bias is None
    assert op.gate.numel() == 8 * 32          # V·dn = 256


def test_n_verbs_property():
    assert _make(n_verbs=16).n_verbs == 16


def test_is_operator_subclass():
    from twm.jepa import Operator
    assert isinstance(_make(), Operator)


def test_fp32_autocast_guard():
    op = _make()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        a = op.apply(k, v)
    assert torch.isfinite(a).all()


# ============================================== end-to-end build through the pipeline (§4.3)
def _build_real(cfg):
    from twm.jepa.model import build_jepa_model_v2
    emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    emb.weight.requires_grad_(False)
    return build_jepa_model_v2(cfg, emb)


def test_operator_group_gated_mlp_builds_and_runs():
    """operator_group='gated_mlp' selects GatedMLPTransition and the model builds + runs a
    forward through the identical pipeline (§4.3)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"operator_group": "gated_mlp"},
    })
    model = _build_real(cfg)
    assert isinstance(model.operator, GatedMLPTransition)
    B, T = 3, 10
    src = torch.randint(5, 512, (B, T)); tgt = torch.randint(5, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    out = model(src, pad, tgt, pad, tau=1.0, hard=True)
    assert out["a"].shape == (B, cfg.model.n_slots, cfg.model.d_noun)
    assert torch.isfinite(out["a"]).all()


def test_gated_mlp_default_d_e_d_h():
    """build reads model.gated_mlp.d_e/d_h; absent -> defaults 4/8 -> 832 params."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"operator_group": "gated_mlp"},
    })
    model = _build_real(cfg)
    assert sum(p.numel() for p in model.operator.parameters()) == 832


def test_gated_mlp_undo_latent_raises_through_model():
    """The model's pet/demo undo path delegates to operator.inverse_apply -> RAISES for
    the black-box baseline (documented; never used in v3 training)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"operator_group": "gated_mlp"},
    })
    model = _build_real(cfg)
    k = torch.randn(2, cfg.model.n_slots, cfg.model.d_noun)
    a = model.step_latent(k, 1)
    with pytest.raises(NotImplementedError, match="no structural inverse"):
        model.undo_latent(a, 1)
