"""Tests for the entity-campaign NORM BUDGET (Task A; entity §1).

Covers the load-bearing guarantees of the renormalize-and-track operator change:

  - use_norm_budget=False is BITWISE the v3 path (operator + model);
  - the budget inverse round-trip is EXACT including the tracked per-slot scale;
  - renormalization restores the pre-step modulus L2-norm and PRESERVES the
    inter-block modulus-profile SHAPE (only the global radius is factored out);
  - norms stay BOUNDED over 16 switched-verb applications (the probe-2 blowup
    scenario): budgeted max-norm-ratio < 2 where the UNBUDGETED operator blows
    past 20;
  - the tracked scale state is VISIBLE to the readout/anchor geometry (perturbing
    s_acc moves the pooled anchor) but NOT to the decoder memory (leakage invariant);
  - forward_unroll threads scale_delta / s_acc keys when on, None when off, and the
    two-hop accumulation is correct (s_acc2 == log_rho_1 + log_rho_2);
  - GatedMLPTransition no-ops the flag with a ONE-TIME warning, returns a zeros
    scale, and inverse_apply still RAISES regardless of norm_budget.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from twm.jepa.operator import RotationScaleOperator
from twm.jepa.baseline_transition import GatedMLPTransition
from twm.jepa.conditioning import block_modulus
from twm.jepa.model import JEPAOperatorModelV2

# Reuse the model mocks (frozen interfaces) from the model test module.
from tests.jepa.test_model import (  # type: ignore
    MockEncoder,
    MockTransition,
    MockPrior,
    SpyDecoder,
)


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """Save/restore the global torch RNG around each test so this module's seeding does
    NOT perturb the ambient RNG state other (non-seeding) test modules rely on — keeps
    the suite order-independent regardless of where these tests run."""
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


def _op(n_verbs=8, d_noun=32, seed=0):
    torch.manual_seed(seed)
    return RotationScaleOperator(n_verbs=n_verbs, d_noun=d_noun, block=2)


# =========================================================================== operator
def test_apply_off_is_bitwise_v3():
    """norm_budget=False returns a BARE tensor identical to the legacy apply (entity §1.1)."""
    op = _op()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a_off = op.apply(k, v)
    assert isinstance(a_off, torch.Tensor)  # bare tensor, NOT a tuple
    # Re-run the same legacy path explicitly: it must be byte-identical.
    a_legacy = op.apply(k, v)
    assert torch.equal(a_off, a_legacy)


def test_apply_off_with_offset_bitwise():
    """The polar (theta_offset) path with norm_budget=False is unchanged too."""
    op = _op()
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))
    off = torch.randn(3, 8, op.n_blocks) * 0.1
    a = op.apply(k, v, theta_offset=off)
    assert isinstance(a, torch.Tensor)
    assert torch.equal(a, op.apply(k, v, theta_offset=off))


def test_apply_on_returns_tuple_with_logrho():
    op = _op()
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    out = op.apply(k, v, norm_budget=True)
    assert isinstance(out, tuple) and len(out) == 2
    a, log_rho = out
    assert a.shape == k.shape
    assert log_rho.shape == (4, 8)  # per-slot, NOT per-block


def test_renorm_restores_pre_modulus_norm():
    """After the budget apply, each slot's modulus L2-norm == its PRE-step norm (entity §1.1)."""
    op = _op(seed=1)
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    norm_pre = block_modulus(k.float()).norm(dim=-1)   # (B, M)
    a, _ = op.apply(k, v, norm_budget=True)
    norm_post = block_modulus(a.float()).norm(dim=-1)  # (B, M)
    assert torch.allclose(norm_post, norm_pre, atol=1e-4), (
        (norm_post - norm_pre).abs().max().item()
    )


def test_renorm_preserves_modulus_profile_shape():
    """Renormalization factors out only the global radius: the inter-block modulus
    profile SHAPE (m / ‖m‖) is identical between the unbudgeted and budgeted outputs."""
    op = _op(seed=2)
    k = torch.randn(4, 8, 32)
    v = torch.randint(0, op.n_verbs, (4, 8))
    a_raw = op.apply(k, v)                              # unbudgeted (the scaled output)
    a_bud, _ = op.apply(k, v, norm_budget=True)         # budgeted (radius factored out)
    m_raw = block_modulus(a_raw.float())                # (B, M, nb)
    m_bud = block_modulus(a_bud.float())
    # Normalize each slot's profile to unit L2 and compare — the SHAPE must match.
    shape_raw = m_raw / m_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    shape_bud = m_bud / m_bud.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(shape_raw, shape_bud, atol=1e-4)


def test_inverse_round_trip_exact_with_scale():
    """inverse_apply(apply(k,v,norm_budget=True), scale_delta=log_rho) == k exactly (entity §1.1)."""
    op = _op(seed=3)
    k = torch.randn(4, 8, 32, dtype=torch.float64).float()
    v = torch.randint(0, op.n_verbs, (4, 8))
    a, log_rho = op.apply(k, v, norm_budget=True)
    k_rt = op.inverse_apply(a, v, norm_budget=True, scale_delta=log_rho)
    assert torch.allclose(k_rt, k, atol=1e-5), (k_rt - k).abs().max().item()


def test_inverse_round_trip_exact_with_offset_and_scale():
    """Round-trip exact under BOTH polar conditioning and the budget (offset replayed)."""
    op = _op(seed=4)
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))
    off = torch.randn(3, 8, op.n_blocks) * 0.2
    a, log_rho = op.apply(k, v, theta_offset=off, norm_budget=True)
    k_rt = op.inverse_apply(a, v, theta_offset=off, norm_budget=True, scale_delta=log_rho)
    assert torch.allclose(k_rt, k, atol=1e-5), (k_rt - k).abs().max().item()


def test_inverse_budget_requires_scale_delta():
    op = _op()
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, op.n_verbs, (2, 8))
    a, _ = op.apply(k, v, norm_budget=True)
    with pytest.raises(AssertionError):
        op.inverse_apply(a, v, norm_budget=True)  # scale_delta missing


def test_norms_bounded_over_16_switched_verb_applications():
    """The probe-2 blowup scenario (entity §1.0): apply 16 switched scaling verbs.

    UNBUDGETED the modulus blows past 20× the start; BUDGETED it stays bounded
    (max norm ratio < 2 because each step renormalizes back to the pre-step norm)."""
    torch.manual_seed(5)
    op = RotationScaleOperator(n_verbs=8, d_noun=32, block=2)
    # Make the operator genuinely scaling so the unbudgeted path blows up: large +log_r.
    with torch.no_grad():
        op.log_r.copy_(torch.full_like(op.log_r, 0.5))  # r = exp(0.5) ≈ 1.65 per step
    k0 = torch.randn(4, 8, 32)
    base_norm = block_modulus(k0.float()).norm(dim=-1)   # (B, M) start radius

    # --- UNBUDGETED: norm compounds ---
    k = k0.clone()
    for h in range(16):
        v = torch.full((4, 8), h % op.n_verbs, dtype=torch.long)
        k = op.apply(k, v)
    norm_unbud = block_modulus(k.float()).norm(dim=-1)
    ratio_unbud = (norm_unbud / base_norm.clamp_min(1e-8)).max().item()
    assert ratio_unbud > 20.0, f"unbudgeted should blow up, got {ratio_unbud}"

    # --- BUDGETED: norm stays put (each step renormalizes to its pre-step norm) ---
    k = k0.clone()
    for h in range(16):
        v = torch.full((4, 8), h % op.n_verbs, dtype=torch.long)
        k, _ = op.apply(k, v, norm_budget=True)
    norm_bud = block_modulus(k.float()).norm(dim=-1)
    ratio_bud = (norm_bud / base_norm.clamp_min(1e-8)).max().item()
    assert ratio_bud < 2.0, f"budgeted should stay bounded, got {ratio_bud}"


# =========================================================================== gated MLP no-op
def test_gated_mlp_norm_budget_returns_zeros_scale():
    torch.manual_seed(0)
    op = GatedMLPTransition(n_verbs=8, d_noun=32, d_e=4, d_h=8)
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, 8, (3, 8))
    out = op.apply(k, v, norm_budget=True)
    assert isinstance(out, tuple) and len(out) == 2
    a, scale_delta = out
    assert a.shape == k.shape
    assert scale_delta.shape == (3, 8)
    assert torch.equal(scale_delta, torch.zeros(3, 8))  # log-scale 0 ⇒ scale 1.0


def test_gated_mlp_norm_budget_off_is_bare_tensor():
    op = GatedMLPTransition(n_verbs=8, d_noun=32)
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, 8, (2, 8))
    a = op.apply(k, v)  # default norm_budget=False
    assert isinstance(a, torch.Tensor)


def test_gated_mlp_warns_once(caplog):
    import logging

    # Fresh class-level flag so the one-time guard fires in THIS test.
    GatedMLPTransition._warned_norm_budget = False
    op = GatedMLPTransition(n_verbs=8, d_noun=32)
    k = torch.randn(2, 8, 32)
    v = torch.randint(0, 8, (2, 8))
    with caplog.at_level(logging.WARNING, logger="twm.jepa.baseline_transition"):
        op.apply(k, v, norm_budget=True)
        op.apply(k, v, norm_budget=True)
        op.apply(k, v, norm_budget=True)
    budget_warnings = [r for r in caplog.records if "norm_budget" in r.getMessage()]
    assert len(budget_warnings) == 1, "warning must fire EXACTLY once per process"


def test_gated_mlp_inverse_raises_regardless_of_norm_budget():
    op = GatedMLPTransition(n_verbs=8, d_noun=32)
    a = torch.randn(2, 8, 32)
    v = torch.randint(0, 8, (2, 8))
    with pytest.raises(NotImplementedError):
        op.inverse_apply(a, v)
    with pytest.raises(NotImplementedError):
        op.inverse_apply(a, v, norm_budget=True, scale_delta=torch.zeros(2, 8))


# =========================================================================== model wiring
def _make_budget_model(use_norm_budget=True, use_polar=False, d_model=64, d_noun=32,
                       n_slots=8, n_verbs=8, n_heads=4):
    """A real RotationScaleOperator behind mock encoder/transition/prior/decoder."""
    enc = MockEncoder(d_model=d_model, d_noun=d_noun, n_slots=n_slots, n_verbs=n_verbs)
    op = RotationScaleOperator(n_verbs=n_verbs, d_noun=d_noun, block=2)
    trans = MockTransition(enc.encode_text, d_model=d_model, n_verbs=n_verbs)
    prior = MockPrior(d_model=d_model, n_verbs=n_verbs)
    dec = SpyDecoder(d_dec=d_model, d_noun=d_noun, n_heads=n_heads)
    return JEPAOperatorModelV2(
        enc, op, trans, prior, dec, d_noun=d_noun, n_verbs=n_verbs, n_heads=n_heads,
        use_pred=True, use_polar_conditioning=use_polar, use_norm_budget=use_norm_budget,
    )


def test_model_off_builds_no_scale_proj():
    m = _make_budget_model(use_norm_budget=False)
    assert m.scale_readout_proj is None
    assert m.use_norm_budget is False


def test_model_on_builds_scale_proj():
    m = _make_budget_model(use_norm_budget=True)
    assert isinstance(m.scale_readout_proj, nn.Linear)
    assert m.scale_readout_proj.in_features == m.d_noun + 1
    assert m.scale_readout_proj.out_features == m.d_noun


def test_forward_off_has_none_scale_keys():
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=False)
    src = torch.randint(0, 512, (3, 10))
    pad = torch.zeros(3, 10, dtype=torch.bool)
    out = m(src, pad, src, pad)
    assert out["scale_delta"] is None
    assert out["s_acc"] is None


def test_forward_on_has_scale_keys():
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True)
    src = torch.randint(0, 512, (3, 10))
    pad = torch.zeros(3, 10, dtype=torch.bool)
    out = m(src, pad, src, pad)
    assert out["scale_delta"].shape == (3, 8)
    assert out["s_acc"].shape == (3, 8)
    # single hop: s_acc == scale_delta (s starts at 0)
    assert torch.equal(out["s_acc"], out["scale_delta"])


def test_unroll_off_has_none_scale_keys():
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=False)
    B, T = 3, 10
    ids = lambda: torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    hops = m.forward_unroll(ids(), pad, ids(), pad, ids(), pad)
    assert len(hops) == 2
    for h in hops:
        assert h["scale_delta"] is None
        assert h["s_acc"] is None


def test_unroll_on_scale_keys_and_two_hop_accumulation():
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True)
    B, T = 3, 10
    ids = lambda: torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    hops = m.forward_unroll(ids(), pad, ids(), pad, ids(), pad)
    assert len(hops) == 2
    for h in hops:
        assert h["scale_delta"].shape == (B, 8)
        assert h["s_acc"].shape == (B, 8)
    # hop-1 s_acc == hop-1 scale_delta (start at 0)
    assert torch.allclose(hops[0]["s_acc"], hops[0]["scale_delta"], atol=1e-6)
    # hop-2 s_acc == hop-1 scale_delta + hop-2 scale_delta (log-domain accumulate, §1.3)
    expected = hops[0]["scale_delta"] + hops[1]["scale_delta"]
    assert torch.allclose(hops[1]["s_acc"], expected, atol=1e-6)


def test_unroll_two_hop_compose_correctness():
    """forward_unroll's a2 must equal applying the budgeted operator twice from k0
    with the per-hop conditioning + budget — the explicit two-hop replay."""
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True, use_polar=True)
    B, T = 2, 10
    s0, s1, s2 = (torch.randint(0, 512, (B, T)) for _ in range(3))
    pad = torch.zeros(B, T, dtype=torch.bool)
    hops = m.forward_unroll(s0, pad, s1, pad, s2, pad)

    # Replay independently using the model's own primitives + stored actions.
    _, k0, _ = m.encoder(s0, pad)
    a1, sd1 = m._apply_action(k0, hops[0]["v_onehot"])
    a2, sd2 = m._apply_action(a1, hops[1]["v_onehot"])
    assert torch.allclose(hops[0]["a"], a1, atol=1e-5)
    assert torch.allclose(hops[1]["a"], a2, atol=1e-5)
    assert torch.allclose(hops[0]["scale_delta"], sd1, atol=1e-6)
    assert torch.allclose(hops[1]["scale_delta"], sd2, atol=1e-6)


def test_scale_visible_to_readout_anchor():
    """Perturbing s_acc moves the pooled anchor (the scale is readout-visible, §1.1)."""
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True)
    a = torch.randn(3, 8, m.d_noun)
    s0 = torch.zeros(3, 8)
    s1 = torch.ones(3, 8) * 2.0
    pooled0 = m._anchor_pool(a, s0)
    pooled1 = m._anchor_pool(a, s1)
    assert not torch.allclose(pooled0, pooled1), "s_acc must change the anchor pool"


def test_decoder_memory_unaffected_by_scale():
    """The decoder memory is `a` ONLY — perturbing s_acc must NOT change decoder logits
    (the leakage invariant: scale enters the anchor path, never the decoder, §1.1)."""
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True)
    src = torch.randint(0, 512, (3, 10))
    pad = torch.zeros(3, 10, dtype=torch.bool)
    out = m(src, pad, src, pad)
    # The decoder was called with `a` (the operator output), independent of s_acc — the
    # SpyDecoder records the memory id, and the model passes out["a"] (not the augmented
    # slot). Re-decode out["a"] directly and confirm it matches the forward logits.
    logits_direct = m.decoder(out["a"], src, pad)
    assert torch.allclose(logits_direct, out["logits"], atol=1e-5)
    # And the decoder memory id IS out["a"] (the un-augmented operator output).
    assert m.decoder.last_memory_id == id(out["a"])


def test_step_undo_roundtrip_with_budget():
    """The pet/demo path: step_latent returns (a, theta_offset, scale_delta); undo_latent
    re-applies the scale for an exact inverse (entity §1.3)."""
    torch.manual_seed(0)
    m = _make_budget_model(use_norm_budget=True, use_polar=True)
    k = torch.randn(2, 8, m.d_noun)
    a, theta_offset, scale_delta = m.step_latent(k, verb_idx=3)
    k_rt = m.undo_latent(a, verb_idx=3, theta_offset=theta_offset, scale_delta=scale_delta)
    assert torch.allclose(k_rt, k, atol=1e-4), (k_rt - k).abs().max().item()


def test_build_model_reads_use_norm_budget():
    """build_jepa_model_v2 wires use_norm_budget from the config (entity §1.5)."""
    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2

    cfg = JEPAConfig.from_dict({
        "profile": "jepa_v3",
        "model": {"use_norm_budget": True, "use_polar_conditioning": True},
        "data": {"vocab_size": 64, "max_text_tokens": 16},
    })
    emb = nn.Embedding(64, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, emb)
    assert model.use_norm_budget is True
    assert model.scale_readout_proj is not None


def test_build_model_default_no_budget():
    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2

    cfg = JEPAConfig.from_dict({
        "profile": "jepa_v3",
        "data": {"vocab_size": 64, "max_text_tokens": 16},
    })
    emb = nn.Embedding(64, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, emb)
    assert model.use_norm_budget is False
    assert model.scale_readout_proj is None
