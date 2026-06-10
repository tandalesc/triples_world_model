"""Unit tests for JEPAOperatorModelV2.forward_unroll — the v3 two-hop unroll (§2).

Owned by Task C. Reuses the mock encoder/operator/posterior/prior/decoder from
test_model.py (FROZEN interfaces, §11) so these tests run without the sibling Task A/B
modules. Critical contracts:

  - n_unroll_steps=1 (hop 1 alone) is BITWISE-IDENTICAL to the current single-hop forward.
  - composed two-hop a2 = B_v2(B_v1 k0) matches MANUAL angle-addition for the operator
    path, with polar conditioning reading the hop's INPUT modulus (|k0| then |a1|, §2.3).
  - leakage (§2.5): s2 reaches hop-2 memory a2 ONLY through the discrete v2.
  - both hops are trainable (token CE summed over hops reaches both posteriors).
  - the gated-MLP baseline composes by re-application with no special-casing.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for test_model mock reuse

# Reuse the frozen-interface mocks + helpers from the Task C model tests.
from test_model import _make_model, _build_real  # noqa: E402


def _triple(B=4, T=12):
    s0 = torch.randint(5, 512, (B, T))
    s1 = torch.randint(5, 512, (B, T))
    s2 = torch.randint(5, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    return s0, s1, s2, pad


# =========================================================================== shapes
def test_returns_two_hops_with_shapes():
    torch.manual_seed(0)
    m = _make_model()
    B, T, M, dn, V = 4, 12, 8, 32, 8
    s0, s1, s2, pad = _triple(B, T)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    assert isinstance(hops, list) and len(hops) == 2
    for h in hops:
        assert h["k"].shape == (B, M, dn)
        assert h["a"].shape == (B, M, dn)
        assert h["v"].shape == (B,)
        assert h["v_onehot"].shape == (B, V)
        assert h["v_logits"].shape == (B, V)
        assert h["p_logits"].shape == (B, V)
        assert h["zhat"].shape == (B, dn)
        assert h["z_target"].shape == (B, dn)
        assert h["logits"].shape[:2] == (B, T)
    # composition: hop-2 input k IS hop-1 output a (a2 = B_v2(B_v1 k0)).
    assert torch.equal(hops[1]["k"], hops[0]["a"])


def test_use_pred_false_skips_anchor_both_hops():
    m = _make_model(use_pred=False)
    s0, s1, s2, pad = _triple(3, 8)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad)
    for h in hops:
        assert h["zhat"] is None and h["z_target"] is None
        assert h["logits"].shape[:2] == (3, 8)


# =========================================================================== CRITICAL: hop-1 == forward
def test_hop1_bitwise_identical_to_single_hop_forward():
    """n_unroll_steps=1 (hop 1 alone) MUST be BITWISE-IDENTICAL to forward() with the
    same (s0, s1) — same encoder/posterior/prior/operator/decoder/anchor calls."""
    torch.manual_seed(0)
    m = _make_model()
    m.eval()
    s0, s1, s2, pad = _triple(4, 12)
    with torch.no_grad():
        torch.manual_seed(999)  # identical Gumbel noise for both calls
        fwd = m(s0, pad, s1, pad, tau=1.0, hard=True)
        torch.manual_seed(999)
        hops = m.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    h1 = hops[0]
    for key in ("k", "a", "v_onehot", "v_logits", "p_logits", "logits", "zhat", "z_target"):
        assert torch.equal(fwd[key], h1[key]), key


# =========================================================================== CRITICAL: composition
def test_composition_matches_manual_angle_addition():
    """Composed two-hop a2 = B_v2(B_v1 k0) MUST equal the manual angle-additive
    composition (REAL RotationScaleOperator, polar conditioning ON). Hop-2 conditioning
    reads the POST-step modulus |a1| (§2.3)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    model = _build_real(cfg)
    model.eval()
    # nonzero H so conditioning is genuinely state-dependent (|k0| vs |a1| differ).
    model.conditioner.H.weight.data.normal_(0, 0.2)
    B, T = 3, 10
    s0, s1, s2, pad = _triple(B, T)
    with torch.no_grad():
        hops = model.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    k0, a1, a2 = hops[0]["k"], hops[0]["a"], hops[1]["a"]
    v1, v2 = hops[0]["v"], hops[1]["v"]   # (B,) argmax verbs

    op, cond = model.operator, model.conditioner

    def manual_apply(k, vidx, offset):
        # θ_eff = θ_v + offset (per block); r_v unconditioned. RoPE-style block apply.
        theta = op.theta.float()[vidx]                 # (B, nb)
        r = torch.exp(op.log_r.float())[vidx]          # (B, nb)
        M, nb = k.shape[1], theta.shape[-1]
        theta = theta.unsqueeze(1).expand(B, M, nb)
        r = r.unsqueeze(1).expand(B, M, nb)
        theta_eff = theta + offset
        a = r * torch.cos(theta_eff); b = r * torch.sin(theta_eff)
        xp = k.float().reshape(B, M, nb, 2)
        xc, yc = xp[..., 0], xp[..., 1]
        return torch.stack([a * xc - b * yc, b * xc + a * yc], dim=-1).reshape(k.shape)

    a1_manual = manual_apply(k0, v1, cond(k0))         # hop-1 offset H(|k0|)
    assert torch.allclose(a1, a1_manual, atol=1e-4), (a1 - a1_manual).abs().max().item()
    a2_manual = manual_apply(a1_manual, v2, cond(a1))  # hop-2 offset H(|a1|) — POST-step
    assert torch.allclose(a2, a2_manual, atol=1e-4), (a2 - a2_manual).abs().max().item()


def test_hop2_conditioning_reads_post_step_modulus():
    """The hop-2 offset MUST be H(|a1|), not the stale H(|k0|) (§2.3). With a scaling
    verb |a1| ≠ |k0|, so reading k0 would give a different (wrong) a2."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    model = _build_real(cfg)
    model.eval()
    model.conditioner.H.weight.data.normal_(0, 0.4)
    s0, s1, s2, pad = _triple(3, 10)
    with torch.no_grad():
        hops = model.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
        k0, a1, a2 = hops[0]["k"], hops[0]["a"], hops[1]["a"]
        v2 = hops[1]["v_onehot"]
        # recompute a2 with the CORRECT (|a1|) and the WRONG (|k0|) offsets.
        a2_correct = model.operator.apply(a1, v2.unsqueeze(1).expand(-1, a1.shape[1], -1),
                                          theta_offset=model.conditioner(a1))
        a2_wrong = model.operator.apply(a1, v2.unsqueeze(1).expand(-1, a1.shape[1], -1),
                                        theta_offset=model.conditioner(k0))
    assert torch.allclose(a2, a2_correct, atol=1e-5)
    # if |a1| genuinely differs from |k0| (scaling verb), the wrong offset diverges.
    if not torch.allclose(model.conditioner(a1), model.conditioner(k0), atol=1e-6):
        assert not torch.allclose(a2, a2_wrong, atol=1e-4)


# =========================================================================== leakage (§2.5)
def test_leakage_s2_only_via_v2():
    """Perturbing s2 changes hop-2 memory a2 ONLY through the discrete v2. Where the hop-2
    posterior picks the SAME verb for both s2 variants, a2 must be identical."""
    torch.manual_seed(0)
    m = _make_model()
    m.eval()
    B, T = 3, 10
    s0 = torch.randint(5, 512, (B, T)); s1 = torch.randint(5, 512, (B, T))
    s2a = torch.randint(5, 512, (B, T)); s2b = torch.randint(5, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(7)
        ha = m.forward_unroll(s0, pad, s1, pad, s2a, pad, tau=1.0, hard=True)
        torch.manual_seed(7)
        hb = m.forward_unroll(s0, pad, s1, pad, s2b, pad, tau=1.0, hard=True)
    same = ha[1]["v"] == hb[1]["v"]
    if same.any():
        assert torch.allclose(ha[1]["a"][same], hb[1]["a"][same], atol=1e-5)
    # hop-1 (a1) never sees s2 at all -> always identical.
    assert torch.equal(ha[0]["a"], hb[0]["a"])


def test_leakage_decoder_memory_is_a_star_each_hop():
    """Each hop's decoder memory IS that hop's a* (spy id check). SpyDecoder records the
    LAST memory id -> hop-2's a*."""
    m = _make_model()
    s0, s1, s2, pad = _triple(2, 8)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad)
    assert m.decoder.last_memory_id == id(hops[1]["a"])


# =========================================================================== trainability
def test_gradient_reaches_both_posteriors():
    """Token CE summed over hops reaches v1 AND v2 through the ST action (§2.4)."""
    torch.manual_seed(0)
    m = _make_model()
    s0, s1, s2, pad = _triple(4, 10)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    loss = (hops[0]["logits"].float().pow(2).mean()
            + 0.5 * hops[1]["logits"].float().pow(2).mean())
    loss.backward()
    grad = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.transition.mlp.parameters())
    assert grad, "unroll token loss did not reach the posterior through the ST action"


def test_hop2_gradient_reaches_hop1_operator_input():
    """Composition pressure: a hop-2-only loss must flow back through a1 into k0/encoder."""
    torch.manual_seed(0)
    m = _make_model()
    s0, s1, s2, pad = _triple(3, 10)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    hops[1]["logits"].float().pow(2).mean().backward()
    grad = any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.noun_head.parameters())
    assert grad, "hop-2 loss did not propagate through the composed a1 into the encoder"


# =========================================================================== gated-MLP unroll
def test_gated_mlp_unroll_composes_by_reapplication():
    """The gated-MLP baseline composes by re-application in the unroll loop (no inverse,
    no special-casing) — forward_unroll runs end-to-end (§4.3)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"operator_group": "gated_mlp"},
    })
    model = _build_real(cfg)
    s0, s1, s2, pad = _triple(2, 8)
    hops = model.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    assert len(hops) == 2
    assert torch.equal(hops[1]["k"], hops[0]["a"])  # a2 composes on a1
    assert torch.isfinite(hops[1]["a"]).all()


def test_kind_ids_surfaced_per_hop_when_enabled():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"use_polar_conditioning": True, "use_kind_head": True,
                  "kind_codebook_size": 8},
    })
    model = _build_real(cfg)
    s0, s1, s2, pad = _triple(3, 12)
    hops = model.forward_unroll(s0, pad, s1, pad, s2, pad, tau=1.0, hard=True)
    for h in hops:
        assert "kind_ids" in h
        assert h["kind_ids"].shape == (3, cfg.model.n_slots)
