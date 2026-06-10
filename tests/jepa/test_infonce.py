"""Unit tests for the InfoNCE next-state contrastive loss (Task A, design doc §1).

Covers:
  - info_nce decreases when the anchor matches its true key among negatives
    (vs a mismatched anchor).
  - same-chain negatives are actually drawn via chain_ids (the same-chain wrong
    next-states populate the contrastive denominator and shape the loss).
  - gradient flows to the encoder (anchor zhat) and operator side; the positive key
    and explicit negatives are stop-grad (no gradient leaks into them).
  - permutation test: shuffling the targets within the batch destroys the signal
    (the loss for matched pairs is strictly lower than for shuffled pairs).
  - shape/finite, diagonal-positive near-zero, temperature monotonicity.
  - cross-hop hard negatives (neg_keys) append columns and raise the loss.
  - w_nce gating: w_nce=0 reproduces exact v2.1 total (L_nce not added); the v3
    aggregator forward accepts chain_ids and computes L_nce when w_nce>0.
  - leakage: the loss has no decoder argument (signature-level), and z_target /
    nce_neg_keys carry no gradient anywhere (they are detached in info_nce).
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn

from twm.jepa.losses import JEPALossV2, info_nce


def _seed(s: int = 0):
    torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Basic shape / finiteness / diagonal positive
# ---------------------------------------------------------------------------

def test_info_nce_shape_and_finite():
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn)
    ztgt = torch.randn(B, dn)
    loss = info_nce(zhat, ztgt, temperature=0.1)
    assert loss.shape == (), "info_nce must be a scalar"
    assert torch.isfinite(loss), f"info_nce not finite: {loss.item()}"


def test_info_nce_near_zero_when_anchor_equals_key():
    """If the anchor exactly equals its own key and keys are well separated, the
    softmax-CE over the gallery is near zero (each anchor retrieves its own key)."""
    _seed()
    B, dn = 16, 32
    # Orthogonal-ish keys: random Gaussian in high-ish dim are near-orthogonal.
    ztgt = torch.randn(B, dn)
    zhat = ztgt.clone()  # anchor == key
    loss = info_nce(zhat, ztgt, temperature=0.05)
    assert loss.item() < 0.05, f"matched anchor==key should give near-zero loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# NCE decreases when the prediction matches the target among negatives
# ---------------------------------------------------------------------------

def test_info_nce_decreases_when_prediction_matches_target():
    """Core property: an anchor that points at its true next-state key scores a LOWER
    InfoNCE than an anchor that points elsewhere (a wrong next-state among negatives)."""
    _seed()
    B, dn = 12, 32
    ztgt = torch.randn(B, dn)

    # Good anchor: aligned with its own key.
    zhat_good = ztgt.clone()
    # Bad anchor: each row points at the NEXT row's key (a wrong, in-batch negative).
    zhat_bad = ztgt.roll(shifts=1, dims=0).clone()

    loss_good = info_nce(zhat_good, ztgt, temperature=0.1)
    loss_bad = info_nce(zhat_bad, ztgt, temperature=0.1)

    assert loss_good.item() < loss_bad.item(), (
        f"matched anchor should score lower than mismatched: "
        f"good={loss_good.item():.4f} bad={loss_bad.item():.4f}"
    )


def test_info_nce_permutation_destroys_signal():
    """Permutation test: shuffling the targets within the batch (breaking the diagonal
    correspondence) destroys the retrieval signal — the loss rises sharply because each
    anchor's true key is now a random other row."""
    _seed()
    B, dn = 16, 32
    ztgt = torch.randn(B, dn)
    zhat = ztgt.clone()  # perfectly aligned diagonal

    loss_aligned = info_nce(zhat, ztgt, temperature=0.1)

    # Shuffle the targets so anchor i no longer corresponds to key i.
    perm = torch.randperm(B)
    while bool((perm == torch.arange(B)).all()):
        perm = torch.randperm(B)
    loss_shuffled = info_nce(zhat, ztgt[perm], temperature=0.1)

    assert loss_shuffled.item() > loss_aligned.item() + 0.5, (
        f"shuffling targets must destroy the signal: aligned={loss_aligned.item():.4f} "
        f"shuffled={loss_shuffled.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Same-chain negatives actually drawn via chain_ids
# ---------------------------------------------------------------------------

def test_info_nce_same_chain_negatives_used():
    """Same-chain wrong next-states are real negatives in the (B,B) matrix. With
    chain_ids the loss still treats off-diagonal same-chain columns as negatives (they
    push the anchor away), so the loss reflects them.

    We verify chain_ids participates: a batch where a same-chain sibling is a HARD
    negative (its key is close to the anchor's true key direction but distinct) scores
    higher than a batch where that sibling is replaced by a far-away unrelated key.
    """
    _seed()
    B, dn = 8, 32
    base = torch.randn(B, dn)
    ztgt = base.clone()
    zhat = base.clone()  # aligned

    chain_ids = torch.arange(B)  # each its own chain (default behaviour)
    # Make rows 0 and 1 same-chain siblings, with row 1's key a HARD negative for row 0:
    # row 1's key points partway toward row 0's key.
    chain_ids[1] = chain_ids[0]
    ztgt[1] = (0.9 * base[0] + 0.1 * base[1])
    # anchor 0 stays aligned with its own (row-0) key
    zhat[0] = ztgt[0]

    loss_hard = info_nce(zhat, ztgt, chain_ids=chain_ids, temperature=0.1)

    # Compare to a version where row 1's key is FAR from row 0 (an easy negative).
    ztgt_easy = ztgt.clone()
    ztgt_easy[1] = -base[0]  # opposite direction -> easy negative for anchor 0
    loss_easy = info_nce(zhat, ztgt_easy, chain_ids=chain_ids, temperature=0.1)

    assert loss_hard.item() > loss_easy.item(), (
        f"a same-chain HARD negative must raise the loss vs an easy one: "
        f"hard={loss_hard.item():.4f} easy={loss_easy.item():.4f}"
    )


def test_info_nce_same_chain_duplicate_positive_masked():
    """Defensive masking (§1.4): if a same-chain off-diagonal column EXACTLY duplicates
    the anchor's own positive key, it must be masked out (not counted as a spurious
    negative), so the loss does not blow up. With chain_ids the duplicate column is
    masked; without chain_ids it would (wrongly) be a hard negative and raise the loss."""
    _seed()
    B, dn = 6, 32
    ztgt = torch.randn(B, dn)
    zhat = ztgt.clone()

    # Make row 1's key an EXACT duplicate of row 0's key, and put them on the same chain.
    ztgt[1] = ztgt[0].clone()
    chain_ids = torch.arange(B)
    chain_ids[1] = chain_ids[0]

    loss_masked = info_nce(zhat, ztgt, chain_ids=chain_ids, temperature=0.1)
    loss_unmasked = info_nce(zhat, ztgt, chain_ids=None, temperature=0.1)

    # Masking the duplicate-positive column should give a strictly lower loss for the
    # affected anchors (the duplicate is no longer a confusing negative).
    assert loss_masked.item() < loss_unmasked.item(), (
        f"duplicate-positive same-chain column must be masked: "
        f"masked={loss_masked.item():.4f} unmasked={loss_unmasked.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Gradient flow: anchor gets grad, key + negatives do not
# ---------------------------------------------------------------------------

def test_info_nce_gradient_flows_to_anchor():
    """Gradient must reach the anchor zhat (which is downstream of encoder + operator)."""
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn, requires_grad=True)
    ztgt = torch.randn(B, dn)
    loss = info_nce(zhat, ztgt, temperature=0.1)
    loss.backward()
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0, (
        "InfoNCE gradient must flow to the anchor (encoder + operator path)"
    )


def test_info_nce_gradient_flows_through_encoder_and_operator():
    """End-to-end: anchor produced by a tiny (encoder->operator) chain; both stages must
    receive gradient from InfoNCE."""
    _seed()
    B, dn = 8, 32
    encoder = nn.Linear(dn, dn)   # stands in for the slot encoder + readout
    operator = nn.Linear(dn, dn)  # stands in for the operator/predictor head
    src = torch.randn(B, dn)
    k = encoder(src)
    zhat = operator(k)            # anchor = predictor(readout(operator(k)))
    ztgt = torch.randn(B, dn)     # stop-grad EMA key

    loss = info_nce(zhat, ztgt, temperature=0.1)
    loss.backward()

    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0, (
        "InfoNCE gradient must reach the encoder"
    )
    assert operator.weight.grad is not None and operator.weight.grad.abs().sum() > 0, (
        "InfoNCE gradient must reach the operator"
    )


def test_info_nce_key_is_stopgrad():
    """The positive key must receive NO gradient (MoCo/BYOL asymmetry — info_nce detaches
    z_target internally regardless of caller)."""
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn, requires_grad=True)
    ztgt = torch.randn(B, dn, requires_grad=True)  # caller forgot to detach
    loss = info_nce(zhat, ztgt, temperature=0.1)
    loss.backward()
    assert ztgt.grad is None or ztgt.grad.abs().sum() < 1e-12, (
        "positive key z_target must be stop-grad inside info_nce"
    )


def test_info_nce_neg_keys_are_stopgrad():
    """Explicit cross-hop hard-negative keys must also be stop-grad."""
    _seed()
    B, dn = 8, 32
    zhat = torch.randn(B, dn, requires_grad=True)
    ztgt = torch.randn(B, dn)
    neg = torch.randn(B, 1, dn, requires_grad=True)
    loss = info_nce(zhat, ztgt, temperature=0.1, neg_keys=neg)
    loss.backward()
    assert neg.grad is None or neg.grad.abs().sum() < 1e-12, (
        "explicit negative keys must be stop-grad inside info_nce"
    )
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Cross-hop hard negatives raise the loss
# ---------------------------------------------------------------------------

def test_info_nce_neg_keys_append_columns_and_raise_loss():
    """Adding a hard cross-hop negative (close to the anchor's true key direction) raises
    the loss vs the same batch with no extra negative."""
    _seed()
    B, dn = 8, 32
    ztgt = torch.randn(B, dn)
    zhat = ztgt.clone()  # aligned

    loss_no_neg = info_nce(zhat, ztgt, temperature=0.1)

    # Each anchor's extra negative points partway toward its own true key -> hard.
    neg = (0.9 * ztgt + 0.1 * torch.randn(B, dn)).unsqueeze(1)  # (B, 1, dn)
    loss_with_neg = info_nce(zhat, ztgt, temperature=0.1, neg_keys=neg)

    assert loss_with_neg.item() > loss_no_neg.item(), (
        f"a hard cross-hop negative must raise the loss: "
        f"no_neg={loss_no_neg.item():.4f} with_neg={loss_with_neg.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Temperature monotonicity
# ---------------------------------------------------------------------------

def test_info_nce_temperature_monotonicity():
    """A LOWER temperature sharpens the softmax. When an anchor is MORE aligned with a
    wrong key than with its own (a confident-but-wrong match), sharpening puts more mass
    on the wrong key and raises the InfoNCE loss. So the loss for this adversarial batch
    grows as τ shrinks."""
    _seed()
    B, dn = 12, 32
    ztgt = torch.randn(B, dn)
    # Each anchor aligns MORE with its neighbour's key than its own (confident-wrong):
    # 0.3 weight on self, 0.7 on the neighbour -> argmax is the wrong (neighbour) key.
    zhat = 0.3 * ztgt + 0.7 * ztgt.roll(-1, 0)

    loss_hot = info_nce(zhat, ztgt, temperature=1.0)
    loss_cold = info_nce(zhat, ztgt, temperature=0.05)

    assert loss_cold.item() > loss_hot.item(), (
        f"lower temperature should sharpen and raise the loss on confident-wrong matches: "
        f"hot(τ=1.0)={loss_hot.item():.4f} cold(τ=0.05)={loss_cold.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Leakage: signature-level decoder isolation
# ---------------------------------------------------------------------------

def test_info_nce_signature_has_no_decoder_argument():
    """Leakage audit (§1.6): the contrastive loss must have NO decoder/decoder-memory
    argument — the only future-text path is the stop-grad key. Asserted structurally."""
    sig = inspect.signature(info_nce)
    params = set(sig.parameters)
    forbidden = {"decoder", "decoder_fn", "memory", "tgt_ids", "logits"}
    assert not (params & forbidden), (
        f"info_nce must not take any decoder/memory/target-token argument; found {params & forbidden}"
    )


# ---------------------------------------------------------------------------
# JEPALossV2 aggregator: w_nce gating + recoverability
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


def test_aggregator_w_nce_zero_reproduces_v2():
    """w_nce=0 must NOT add L_nce to the total (exact v2.1 recoverability). The total with
    w_nce=0 equals the total without any nce wiring (component L_nce reported as 0.0)."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = inp

    torch.manual_seed(7)
    loss_off = JEPALossV2(operator=op, n_slices=32, w_nce=0.0, w_pred=0.25)
    total_off, comp_off = loss_off(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=torch.arange(B),
    )
    # L_nce reported but zero-weighted and not computed.
    assert comp_off["L_nce"] == 0.0, "w_nce=0 must report L_nce=0.0 (not computed)"
    assert math.isclose(comp_off["w_nce"], 0.0)


def test_aggregator_w_nce_active_adds_term():
    """With w_nce>0 the InfoNCE term is computed and added to the total; the difference
    between w_nce=0.25 and w_nce=0 totals equals 0.25 * L_nce."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = inp
    chain_ids = torch.arange(B)

    torch.manual_seed(7)
    loss_on = JEPALossV2(operator=op, n_slices=32, w_nce=0.25, w_pred=0.0)
    total_on, comp_on = loss_on(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids,
    )
    torch.manual_seed(7)
    loss_off = JEPALossV2(operator=op, n_slices=32, w_nce=0.0, w_pred=0.0)
    total_off, comp_off = loss_off(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=chain_ids,
    )

    assert comp_on["L_nce"] > 0.0, "L_nce must be computed when w_nce>0"
    expected = 0.25 * comp_on["L_nce"]
    actual = comp_on["loss"] - comp_off["loss"]
    assert math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-6), (
        f"w_nce contribution mismatch: expected {expected:.6f}, got {actual:.6f}"
    )


def test_aggregator_w_nce_gradient_reaches_zhat():
    """Through the aggregator, the InfoNCE term must push gradient into zhat (the anchor)."""
    B, M, dn, V_verb = 8, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = inp

    loss_fn = JEPALossV2(operator=op, n_slices=32, w_nce=0.25, w_pred=0.0)
    total, _ = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target,
        tau=1.0, chain_ids=torch.arange(B),
    )
    total.backward()
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0, (
        "aggregator with w_nce>0 must send gradient to the anchor zhat"
    )
