"""v6 §B contract test: L_lam_inv surface-augmentation invariance (arXiv:2506.15691).

The load-bearing leakage invariant of the augmented forward: the posterior infers the
discrete action `v` ONLY from the φ frame of the (s_t, s_{t+1}) pair; the φ' frame enters
ONLY as the decoder's teacher-forced CE target. So perturbing the φ' decoder-target frame
while holding the φ pair fixed must NOT change v's argmax (and must not change a*, k, or the
posterior logits). This test pins that contract against the REAL build_jepa_model_v2 forward.

It also confirms the unaugmented (decoder_target=None) forward is bitwise the single-frame
path, and that the φ' target DOES change the decoder logits (the target is doing work).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _build_model():
    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2
    cfg = JEPAConfig.from_json(str(ROOT / "configs" / "jepa" / "jepa_v6_smoke.json"))
    torch.manual_seed(0)
    emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    emb.weight.requires_grad_(False)
    model = build_jepa_model_v2(cfg, emb)
    model.eval()  # deterministic (hard ST argmax path, no dropout) for the contract
    return model, cfg


def _rand_batch(cfg, seed=1):
    g = torch.Generator().manual_seed(seed)
    B, T = 6, cfg.data.max_text_tokens
    ids = torch.randint(5, cfg.data.vocab_size, (B, T), generator=g)
    pad = torch.zeros(B, T, dtype=torch.bool)
    return ids, pad


def test_v_invariant_to_decoder_target_frame():
    """Changing the φ' decoder-target frame must NOT change v's argmax / posterior / a*."""
    model, cfg = _build_model()
    s_ids, s_pad = _rand_batch(cfg, seed=1)   # φ s_t
    t_ids, t_pad = _rand_batch(cfg, seed=2)   # φ s_{t+1}
    # Two DIFFERENT φ' decoder-target frames (the "same underlying transition" is fixed by
    # the φ pair; only the surface frame of the CE target differs).
    d1_ids, d1_pad = _rand_batch(cfg, seed=3)
    d2_ids, d2_pad = _rand_batch(cfg, seed=4)

    # Seed identically before each forward so the posterior's Gumbel draw is the SAME for
    # both — the ONLY differing input is then the φ' decoder-target frame. (The contract is
    # about the φ' frame not routing into v; controlling the shared stochastic draw isolates
    # it from posterior sampling noise.)
    with torch.no_grad():
        torch.manual_seed(42)
        out1 = model(s_ids, s_pad, t_ids, t_pad, tau=1.0, hard=True,
                     decoder_target=(d1_ids, d1_pad))
        torch.manual_seed(42)
        out2 = model(s_ids, s_pad, t_ids, t_pad, tau=1.0, hard=True,
                     decoder_target=(d2_ids, d2_pad))

    # INVARIANCE: v's argmax is identical regardless of the φ' decoder target.
    assert torch.equal(out1["v"], out2["v"]), "v argmax changed with the φ' decoder frame"
    # And the upstream geometry (posterior logits, nouns k, operator output a*) is identical.
    assert torch.allclose(out1["v_logits"], out2["v_logits"], atol=0, rtol=0)
    assert torch.allclose(out1["k"], out2["k"], atol=0, rtol=0)
    assert torch.allclose(out1["a"], out2["a"], atol=0, rtol=0)
    # SANITY: the φ' target DID reach the decoder (logits differ) — the target is load-bearing.
    assert not torch.allclose(out1["logits"], out2["logits"]), \
        "decoder logits did not change with the φ' target — target not wired"


def test_unaugmented_forward_matches_single_frame():
    """decoder_target=None must be bitwise the single-frame v4 forward (decoder CE on tgt)."""
    model, cfg = _build_model()
    s_ids, s_pad = _rand_batch(cfg, seed=1)
    t_ids, t_pad = _rand_batch(cfg, seed=2)
    with torch.no_grad():
        torch.manual_seed(7)
        base = model(s_ids, s_pad, t_ids, t_pad, tau=1.0, hard=True)
        # Explicitly passing the SAME frame as decoder_target must reproduce `base`.
        torch.manual_seed(7)
        same = model(s_ids, s_pad, t_ids, t_pad, tau=1.0, hard=True,
                     decoder_target=(t_ids, t_pad))
    assert torch.equal(base["v"], same["v"])
    assert torch.allclose(base["logits"], same["logits"], atol=0, rtol=0)
    assert torch.allclose(base["a"], same["a"], atol=0, rtol=0)


def test_posterior_inputs_override_routes_phi_pair():
    """posterior_inputs override: v / k come from the φ pair, decoder target stays separate."""
    model, cfg = _build_model()
    phi_s, phi_sp = _rand_batch(cfg, seed=10)
    phi_t, phi_tp = _rand_batch(cfg, seed=11)
    dec_t, dec_tp = _rand_batch(cfg, seed=12)
    with torch.no_grad():
        # Route the φ pair via posterior_inputs; decoder target is a different frame.
        torch.manual_seed(3)
        out = model(phi_s, phi_sp, phi_t, phi_tp, tau=1.0, hard=True,
                    posterior_inputs=(phi_s, phi_sp, phi_t, phi_tp),
                    decoder_target=(dec_t, dec_tp))
        # Reference: posterior on the φ pair directly (no override) must give the same v / k.
        torch.manual_seed(3)
        ref = model(phi_s, phi_sp, phi_t, phi_tp, tau=1.0, hard=True)
    assert torch.equal(out["v"], ref["v"])
    assert torch.allclose(out["k"], ref["k"], atol=0, rtol=0)
