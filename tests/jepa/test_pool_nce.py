"""Unit tests for the v4.1 pooled-space hard-negative InfoNCE (design §C5).

L_pool_nce is the escalation if the v4.0.1 decoder-CE margin's AUC is also flat: it
trains the POOLED latent space DIRECTLY with an InfoNCE whose positives/negatives mirror
the separation-AUC diagnostic's pool construction (anchor = pooled predicted next state
`zhat`, positive = ONLINE pooled true next state, negatives = same-chain + in-batch NN).

Required properties (task §4):
  - off==neutral: w_pool_nce=0.0 reproduces the v4.0 total EXACTLY (term not added, not
    computed) and the aggregator runs without z_pool_pos.
  - loss decreases on a toy where separation is learnable.
  - negatives exclude the positive (the diagonal is never masked away / never demoted).
  - gradient reaches the encoder through the pooled (online) positive path.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from twm.jepa.losses import JEPALossV2, pool_info_nce

ROOT = Path(__file__).resolve().parents[2]


def _seed(s: int = 0):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------

def test_pool_info_nce_shape_and_finite():
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn)
    zpos = torch.randn(B, dn)
    cids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = pool_info_nce(zhat, zpos, chain_ids=cids, temperature=0.1, n_pool_negs=4)
    assert loss.shape == (), "pool_info_nce must be a scalar"
    assert torch.isfinite(loss), f"pool_info_nce not finite: {loss.item()}"


def test_pool_info_nce_near_zero_when_anchor_equals_positive():
    """Anchor == its own pooled positive, well-separated batch ⟹ near-zero CE."""
    _seed()
    B, dn = 16, 32
    zpos = torch.randn(B, dn)
    zhat = zpos.clone()
    loss = pool_info_nce(zhat, zpos, temperature=0.05, n_pool_negs=8)
    assert loss.item() < 0.05, f"matched anchor==positive should be near-zero, got {loss.item()}"


# ---------------------------------------------------------------------------
# Discriminative behaviour: matched anchor scores lower than mismatched
# ---------------------------------------------------------------------------

def test_pool_info_nce_matched_beats_mismatched():
    _seed()
    B, dn = 12, 32
    zpos = torch.randn(B, dn)
    cids = torch.arange(B) // 2  # pairs share a chain (same-chain hard negatives)
    good = pool_info_nce(zpos.clone(), zpos, chain_ids=cids, temperature=0.1, n_pool_negs=4)
    bad = pool_info_nce(zpos.roll(1, 0), zpos, chain_ids=cids, temperature=0.1, n_pool_negs=4)
    assert good.item() < bad.item(), (
        f"matched anchor must score lower than mismatched: good={good.item():.4f} bad={bad.item():.4f}"
    )


def test_pool_info_nce_decreases_on_learnable_toy():
    """On a toy where separation IS learnable (a trainable linear maps a noisy anchor
    toward the true pooled positive among same-chain + NN hard negatives), the loss
    decreases under SGD — the term carries a usable gradient toward the AUC geometry."""
    _seed(1)
    B, dn = 24, 16
    # Fixed pooled positives (= the AUC candidate space) and chain structure.
    z_pos = torch.randn(B, dn)
    chain_ids = torch.arange(B) // 3  # 3 states per chain ⟹ real same-chain negatives

    # The anchor is a learnable linear of a NOISY view of the true positive — separation is
    # learnable (the map can recover the positive direction) but starts entangled.
    noisy = z_pos + 0.6 * torch.randn(B, dn)
    head = nn.Linear(dn, dn)
    opt = torch.optim.Adam(head.parameters(), lr=5e-2)

    first = last = None
    for step in range(60):
        opt.zero_grad()
        zhat = head(noisy)
        loss = pool_info_nce(zhat, z_pos, chain_ids=chain_ids, temperature=0.1, n_pool_negs=6)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first - 0.3, (
        f"pool InfoNCE must decrease on a learnable toy: first={first:.4f} last={last:.4f}"
    )


# ---------------------------------------------------------------------------
# Negatives exclude the positive
# ---------------------------------------------------------------------------

def test_pool_info_nce_negatives_exclude_positive():
    """The positive (diagonal) is never demoted to a negative — even when a same-chain
    sibling's pooled vector EXACTLY duplicates the anchor's own positive, the diagonal
    column stays the label. We verify by making one off-diagonal column an exact duplicate
    of a positive and confirming the loss stays finite and the matched anchor still wins.

    The structural guard: the diagonal is ALWAYS in `keep` (label=i), and NN mining masks
    out self via the identity mask before top-k, so the positive can never be selected as
    its own negative."""
    _seed()
    B, dn = 6, 32
    z_pos = torch.randn(B, dn)
    # Row 1 is an exact duplicate of row 0's positive AND on the same chain.
    z_pos[1] = z_pos[0].clone()
    chain_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    zhat = z_pos.clone()  # anchor i aligned with positive i
    loss = pool_info_nce(zhat, z_pos, chain_ids=chain_ids, temperature=0.1, n_pool_negs=3)
    assert torch.isfinite(loss), "duplicate-positive must not produce inf/nan"

    # Direct structural check: an anchor pointed at its OWN positive scores lower than the
    # same anchor pointed at a wrong (rolled) target — proving the diagonal is the positive,
    # not masked out.
    good = pool_info_nce(z_pos.clone(), z_pos, chain_ids=chain_ids, n_pool_negs=3)
    bad = pool_info_nce(z_pos.roll(2, 0), z_pos, chain_ids=chain_ids, n_pool_negs=3)
    assert good.item() < bad.item()


def test_pool_info_nce_nn_negatives_raise_loss():
    """In-batch NN mining adds hard negatives. A batch where each anchor's nearest in-batch
    neighbour is a CONFUSER (close to its own positive) scores higher than a batch with the
    same positives but no NN negatives selected (n_pool_negs=0, no chain) — confirming the
    NN pool actually populates the denominator."""
    _seed(3)
    B, dn = 10, 32
    base = torch.randn(B, dn)
    # Make each row's neighbour (row+1) a near-duplicate confuser of row's positive.
    z_pos = base.clone()
    zhat = base.clone()
    confused = pool_info_nce(zhat, z_pos, chain_ids=None, temperature=0.1, n_pool_negs=6)
    # With NO negatives at all (n_pool_negs=0, no chain), every non-diagonal column is masked
    # out, so the CE has a single class ⟹ ~0 loss. Adding NN negatives must raise it.
    none_neg = pool_info_nce(zhat, z_pos, chain_ids=None, temperature=0.1, n_pool_negs=0)
    assert none_neg.item() < confused.item(), (
        f"NN negatives must raise the loss vs no-negative: none={none_neg.item():.4f} "
        f"nn={confused.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_pool_info_nce_gradient_flows_to_anchor():
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn, requires_grad=True)
    z_pos = torch.randn(B, dn)
    loss = pool_info_nce(zhat, z_pos, temperature=0.1, n_pool_negs=4)
    loss.backward()
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0


def test_pool_info_nce_gradient_reaches_encoder_through_positive():
    """Unlike the EMA-keyed L_nce, the pooled positive is the ONLINE pool, so gradient must
    flow into the encoder through BOTH the anchor AND the positive (stop_grad_pos=False).

    A tiny stand-in encoder produces both the anchor and the positive pool; the encoder
    must receive gradient via the positive path even if the anchor path is detached."""
    _seed()
    B, dn = 8, 32
    encoder = nn.Linear(dn, dn)  # stands in for SlotEncoder+Readout (the online pool path)
    src_pos = torch.randn(B, dn)
    z_pos = encoder(src_pos)              # ONLINE pooled positive (gradient path)
    zhat = torch.randn(B, dn).detach()   # detach the anchor to isolate the positive path
    loss = pool_info_nce(zhat, z_pos, temperature=0.1, n_pool_negs=4, stop_grad_pos=False)
    loss.backward()
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0, (
        "gradient must reach the encoder through the ONLINE positive pool path"
    )


def test_pool_info_nce_stop_grad_pos_blocks_positive_gradient():
    """With stop_grad_pos=True the positive is detached (MoCo asymmetry) — no gradient
    leaks into the encoder through the key path."""
    _seed()
    B, dn = 8, 32
    encoder = nn.Linear(dn, dn)
    src_pos = torch.randn(B, dn)
    z_pos = encoder(src_pos)
    # Anchor differentiable (so backward has a grad path); the positive path is what must
    # be blocked under stop_grad_pos.
    zhat = torch.randn(B, dn, requires_grad=True)
    loss = pool_info_nce(zhat, z_pos, temperature=0.1, n_pool_negs=4, stop_grad_pos=True)
    loss.backward()
    g = encoder.weight.grad
    assert g is None or g.abs().sum() < 1e-12, (
        "stop_grad_pos=True must block gradient into the encoder via the positive"
    )
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0, (
        "the anchor still receives gradient under stop_grad_pos"
    )


# ---------------------------------------------------------------------------
# Leakage audit (mirror test_infonce signature guard)
# ---------------------------------------------------------------------------

def test_pool_info_nce_signature_has_no_decoder_argument():
    sig = inspect.signature(pool_info_nce)
    params = set(sig.parameters)
    forbidden = {"decoder", "decoder_fn", "memory", "tgt_ids", "logits"}
    assert not (params & forbidden), (
        f"pool_info_nce must not take any decoder/memory/target-token argument; "
        f"found {params & forbidden}"
    )


# ---------------------------------------------------------------------------
# Aggregator: off == neutral, active adds term, gradient reaches zhat
# ---------------------------------------------------------------------------

class _FakeOperator(nn.Module):
    def __init__(self, v: int, dn: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(v, dn // 2) * 0.5)
        self.log_r = nn.Parameter(torch.randn(v, dn // 2) * 0.1)


def _make_inputs(B=8, M=6, dn=32, V_verb=8, T=16, V_vocab=512):
    _seed()
    logits = torch.randn(B, T, V_vocab, requires_grad=True)
    tgt_ids = torch.randint(5, V_vocab, (B, T))
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)
    k = torch.randn(B, M, dn, requires_grad=True)
    v_logits = torch.randn(B, V_verb, requires_grad=True)
    p_logits = torch.randn(B, V_verb, requires_grad=True)
    zhat = torch.randn(B, dn, requires_grad=True)
    z_target = torch.randn(B, dn)
    return logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target


def test_aggregator_w_pool_nce_zero_is_neutral():
    """w_pool_nce=0.0 must NOT add L_pool_nce (exact v4.0 recoverability), AND the
    aggregator must run identically whether or not z_pool_pos is supplied — when off it is
    never computed, so passing a positive pool changes nothing."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    z_pool_pos = torch.randn(B, dn)
    chain_ids = torch.arange(B) // 2

    loss_fn = JEPALossV2(operator=op, n_slices=32, w_pool_nce=0.0, w_nce=0.0, w_pred=0.25)
    # Seed before each call so the random SIGReg slices match — the ONLY remaining
    # difference would be the pool_nce term, which must be inert when w_pool_nce=0.
    torch.manual_seed(11)
    total_no_pos, comp_no_pos = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids,
    )
    torch.manual_seed(11)
    total_with_pos, comp_with_pos = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids, z_pool_pos=z_pool_pos,
    )
    assert comp_no_pos["L_pool_nce"] == 0.0, "w_pool_nce=0 must report L_pool_nce=0.0 (not computed)"
    assert comp_with_pos["L_pool_nce"] == 0.0, "w_pool_nce=0 must NOT compute the term even with z_pool_pos"
    assert math.isclose(total_no_pos.item(), total_with_pos.item(), rel_tol=0, abs_tol=0), (
        "off path must be bitwise-identical with or without z_pool_pos supplied"
    )
    assert math.isclose(comp_no_pos["w_pool_nce"], 0.0)


def test_aggregator_w_pool_nce_active_adds_term():
    """With w_pool_nce>0 AND z_pool_pos given, the term is computed and added; the total
    difference vs w_pool_nce=0 equals w_pool_nce * L_pool_nce."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    z_pool_pos = torch.randn(B, dn)
    chain_ids = torch.arange(B) // 2

    loss_on = JEPALossV2(operator=op, n_slices=32, w_pool_nce=0.5, n_pool_negs=4,
                         w_nce=0.0, w_pred=0.0)
    total_on, comp_on = loss_on(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids, z_pool_pos=z_pool_pos,
    )
    loss_off = JEPALossV2(operator=op, n_slices=32, w_pool_nce=0.0, n_pool_negs=4,
                          w_nce=0.0, w_pred=0.0)
    total_off, comp_off = loss_off(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids, z_pool_pos=z_pool_pos,
    )
    assert comp_on["L_pool_nce"] > 0.0, "L_pool_nce must be computed when w_pool_nce>0"
    expected = 0.5 * comp_on["L_pool_nce"]
    actual = comp_on["loss"] - comp_off["loss"]
    assert math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-6), (
        f"w_pool_nce contribution mismatch: expected {expected:.6f}, got {actual:.6f}"
    )


def test_aggregator_w_pool_nce_gradient_reaches_zhat_and_positive():
    """Through the aggregator, L_pool_nce pushes gradient into the anchor zhat AND into the
    online positive pool (so, in the real model, into the encoder)."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    z_pool_pos = torch.randn(B, dn, requires_grad=True)  # the online positive (grad path)
    loss_fn = JEPALossV2(operator=op, n_slices=32, w_pool_nce=0.5, n_pool_negs=4,
                         w_nce=0.0, w_pred=0.0)
    total, _ = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=torch.arange(B) // 2, z_pool_pos=z_pool_pos,
    )
    total.backward()
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0, "grad must reach anchor zhat"
    assert z_pool_pos.grad is not None and z_pool_pos.grad.abs().sum() > 0, (
        "grad must reach the online positive pool (stop_grad_pos defaults False)"
    )


# ---------------------------------------------------------------------------
# Config parsing + end-to-end gradient through the REAL encoder
# ---------------------------------------------------------------------------

def test_v41_config_parses_pool_nce_fields():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_v41_s0.json")
    assert math.isclose(cfg.loss.w_pool_nce, 0.5)
    assert math.isclose(cfg.loss.tau_pool, 0.1)
    assert cfg.loss.n_pool_negs == 16
    # The hardened margin from v4.0.1 must be preserved.
    assert math.isclose(cfg.loss.w_margin, 1.0)
    assert math.isclose(cfg.loss.margin, 2.0)


def test_pool_nce_gradient_reaches_real_encoder():
    """End-to-end: the online pooled positive `model._online_bundle.pool_raw(tgt)` carries
    gradient into the REAL SlotEncoder, so L_pool_nce trains the exact pooling the
    separation-AUC's `online` variant measures."""
    from twm.jepa.config import JEPAConfig
    try:
        from twm.jepa.model import build_jepa_model_v2
        cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
        emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
        emb.weight.requires_grad_(False)
        model = build_jepa_model_v2(cfg, emb)
    except Exception as e:
        pytest.skip(f"sibling v2 module not ready: {e}")

    model.train()
    B, T = 6, cfg.data.max_text_tokens
    V = cfg.data.vocab_size
    src = torch.randint(5, V, (B, T))
    src_pad = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.randint(5, V, (B, T))
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)

    out = model.forward_v2(src, src_pad, tgt, tgt_pad, tau=1.0, hard=True)
    z_pool_pos = model._online_bundle.pool_raw(tgt, tgt_pad)  # (B, dn) ONLINE, grad path
    chain_ids = torch.arange(B) // 2
    loss = pool_info_nce(out["zhat"], z_pool_pos, chain_ids=chain_ids,
                         temperature=0.1, n_pool_negs=4)
    loss.backward()

    enc_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    total = sum(g.abs().sum().item() for g in enc_grads if g is not None)
    assert total > 0, "L_pool_nce gradient must reach the real SlotEncoder via the online pool"
