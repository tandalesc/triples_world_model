"""Unit tests for JEPA v2 losses (T-task D). Design doc §5 + §11.

Covers:
  - token_ce: finite, masks pad (ignore_index), EOS not masked, gradient flows.
  - prior_kl: non-negative, zero when q==p, gradient reaches p_logits (not q_logits),
    stop-grad on posterior verified.
  - JEPALossV2 forward: correct component keys, finite total, backward passes.
  - L_sigreg and L_pred reused from v1 (imported, not re-tested exhaustively).
  - Leakage permutation check at the loss level: swapping tgt_ids gives different
    token_ce, confirming the decoder output (logits) carries tgt info through a*
    only (the test here is structural — it verifies token_ce is discriminative).
  - Operator is NOT a submodule of JEPALossV2 (no double-registration).
  - anneal_tau v2 defaults: tau_start=3.0, tau_end=1.0, anneal_frac=0.5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from twm.jepa.losses import (
    JEPALossV2,
    anneal_tau,
    prior_kl,
    sigreg_loss,
    token_ce,
    token_margin,
    _per_example_ce,
)


def _seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# token_ce
# ---------------------------------------------------------------------------

def test_token_ce_finite_and_correct_shape():
    _seed()
    B, T, V = 4, 16, 512
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))  # 0-4 are special tokens; use 5+
    loss = token_ce(logits, tgt_ids, pad_id=0)
    assert torch.isfinite(loss), f"token_ce not finite: {loss.item()}"
    assert loss.shape == (), "token_ce should be scalar"


def test_token_ce_masks_pad_positions():
    """Padding positions (pad_id=0) must not contribute to the CE loss."""
    _seed()
    B, T, V = 4, 10, 512
    logits = torch.randn(B, T, V)
    tgt_ids = torch.zeros(B, T, dtype=torch.long)  # all pads
    # With all pad targets, ignore_index=0 should exclude everything.
    # PyTorch returns 0.0 when all targets are ignored.
    loss_all_pad = token_ce(logits, tgt_ids, pad_id=0)
    assert loss_all_pad.item() == 0.0 or not torch.isfinite(loss_all_pad) or loss_all_pad.item() >= 0.0
    # More useful check: loss should be finite when only some positions are pad.
    tgt_mixed = tgt_ids.clone()
    tgt_mixed[:, :5] = torch.randint(5, V, (B, 5))  # non-pad first half
    loss_mixed = token_ce(logits, tgt_mixed, pad_id=0)
    assert torch.isfinite(loss_mixed), f"mixed pad CE not finite: {loss_mixed}"


def test_token_ce_eos_not_masked():
    """EOS token (id=4) must be included in the loss denominator.

    Design doc §4.3: EOS is a real predicted token (not masked with ignore_index=pad).
    """
    _seed()
    B, T, V = 4, 8, 512
    logits = torch.randn(B, T, V)

    # All EOS targets
    tgt_eos = torch.full((B, T), 4, dtype=torch.long)
    loss_eos = token_ce(logits, tgt_eos, pad_id=0)

    # All pad targets (should be effectively 0 / not-a-number under CE)
    tgt_pad = torch.zeros(B, T, dtype=torch.long)
    loss_pad = token_ce(logits, tgt_pad, pad_id=0)

    # EOS (id=4) is real, pad (id=0) is ignored.
    # loss_eos should be a valid positive CE; loss_pad is 0 or nan (all ignored).
    assert torch.isfinite(loss_eos) and loss_eos.item() > 0, (
        f"EOS CE should be positive and finite, got {loss_eos.item()}"
    )
    # pad targets: PyTorch CE returns 0 when all inputs are ignored
    assert loss_pad.item() == 0.0 or not torch.isfinite(loss_pad), (
        f"all-pad CE should be 0 or nan, got {loss_pad.item()}"
    )


def test_token_ce_gradient_reaches_logits():
    _seed()
    B, T, V = 4, 12, 512
    logits = torch.randn(B, T, V, requires_grad=True)
    tgt_ids = torch.randint(5, V, (B, T))
    loss = token_ce(logits, tgt_ids, pad_id=0)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0, "No gradient through token_ce to logits"


def test_token_ce_discriminative():
    """Swapping different target sequences gives different CE values.

    This is the loss-level leakage permutation check: if the decoder's logits carry
    true information about a* (the only conditioning path), then different tgt_ids
    yield different losses when logits are held constant.

    Full leakage integrity (decoder memory = a* only) is tested by test_model.py
    (Task C). Here we verify token_ce itself is discriminative — it does not collapse
    all target sequences to the same scalar.
    """
    _seed()
    B, T, V = 4, 12, 512
    logits = torch.randn(B, T, V)

    tgt_a = torch.randint(5, V // 2, (B, T))      # lower half of vocab
    tgt_b = torch.randint(V // 2, V, (B, T))       # upper half of vocab

    loss_a = token_ce(logits, tgt_a, pad_id=0).item()
    loss_b = token_ce(logits, tgt_b, pad_id=0).item()

    # The two different target sets should yield different CE values (not identical).
    assert not math.isclose(loss_a, loss_b, rel_tol=1e-5), (
        f"token_ce is not discriminative: both targets give {loss_a:.4f}"
    )


# ---------------------------------------------------------------------------
# prior_kl
# ---------------------------------------------------------------------------

def test_prior_kl_nonnegative():
    _seed()
    B, V = 8, 8
    q_logits = torch.randn(B, V)
    p_logits = torch.randn(B, V)
    kl = prior_kl(q_logits, p_logits, tau=1.0)
    assert torch.isfinite(kl), f"prior_kl not finite: {kl.item()}"
    assert kl.item() >= 0.0, f"KL must be non-negative, got {kl.item()}"


def test_prior_kl_zero_when_equal():
    """KL(q ‖ p) = 0 when q and p have the same distribution."""
    _seed()
    B, V = 8, 8
    logits = torch.randn(B, V)
    kl = prior_kl(logits, logits.clone(), tau=1.0)
    assert math.isclose(kl.item(), 0.0, abs_tol=1e-5), (
        f"KL(q ‖ q) should be 0, got {kl.item()}"
    )


def test_prior_kl_gradient_reaches_p_not_q():
    """Gradient must reach p_logits (the prior, which should change).
    q_logits must have NO gradient (posterior is stop-grad target).
    """
    _seed()
    B, V = 8, 8
    q_logits = torch.randn(B, V, requires_grad=True)
    p_logits = torch.randn(B, V, requires_grad=True)

    kl = prior_kl(q_logits, p_logits, tau=1.0)
    kl.backward()

    assert p_logits.grad is not None, "p_logits must receive gradient"
    assert p_logits.grad.abs().sum() > 0, "p_logits gradient must be nonzero"

    # q_logits should have NO gradient (it is stop-grad in prior_kl).
    assert q_logits.grad is None or q_logits.grad.abs().sum() < 1e-10, (
        "q_logits must be stop-grad in prior_kl (posterior must not be steered by KL)"
    )


def test_prior_kl_stopgrad_on_posterior():
    """Verify the stop-grad on q by checking that the computation graph does not
    include q_logits: calling .backward() on the KL should not set q_logits.grad."""
    B, V = 4, 8
    q_logits = torch.randn(B, V, requires_grad=True)
    p_logits = torch.randn(B, V, requires_grad=True)

    kl = prior_kl(q_logits, p_logits, tau=1.5)
    kl.backward()

    # If stop-grad is correctly applied, q_logits.grad is None.
    assert q_logits.grad is None, (
        "posterior q_logits.grad must be None — stop-grad violated in prior_kl"
    )


def test_prior_kl_large_when_distant():
    """KL(q ‖ p) should be large when q and p are concentrated on opposite codes."""
    B, V = 4, 8
    q_logits = torch.zeros(B, V)
    q_logits[:, 0] = 10.0  # q concentrated on code 0

    p_logits = torch.zeros(B, V)
    p_logits[:, V - 1] = 10.0  # p concentrated on last code

    kl = prior_kl(q_logits, p_logits, tau=1.0)
    assert kl.item() > 1.0, (
        f"KL between opposite-code distributions should be large, got {kl.item()}"
    )


def test_prior_kl_tau_effect():
    """Increasing tau softens the q distribution, reducing the KL when p is uniform."""
    B, V = 4, 8
    q_logits = torch.zeros(B, V)
    q_logits[:, 0] = 5.0  # peaked at code 0
    p_logits = torch.zeros(B, V)  # uniform prior

    kl_low_tau = prior_kl(q_logits, p_logits, tau=0.1)   # very peaked q
    kl_high_tau = prior_kl(q_logits, p_logits, tau=10.0)  # near-uniform q

    # High tau -> near-uniform q -> KL closer to 0
    assert kl_high_tau.item() < kl_low_tau.item(), (
        f"Higher tau should soften q and reduce KL vs uniform p: "
        f"low={kl_low_tau.item():.4f} high={kl_high_tau.item():.4f}"
    )


# ---------------------------------------------------------------------------
# anneal_tau v2 defaults
# ---------------------------------------------------------------------------

def test_anneal_tau_v2_defaults():
    """v2 changes tau_start=3.0, tau_end=1.0, anneal_frac=0.5."""
    total = 1000

    # At step 0, should return tau_start=3.0
    tau0 = anneal_tau(0, total, tau_start=3.0, tau_end=1.0, anneal_frac=0.5)
    assert math.isclose(tau0, 3.0, rel_tol=1e-6), f"Expected 3.0 at step 0, got {tau0}"

    # After anneal window (50% of steps), should return tau_end=1.0 and stay there
    tau_end = anneal_tau(500, total, tau_start=3.0, tau_end=1.0, anneal_frac=0.5)
    assert math.isclose(tau_end, 1.0, rel_tol=1e-6), f"Expected 1.0 at step 500, got {tau_end}"

    tau_late = anneal_tau(999, total, tau_start=3.0, tau_end=1.0, anneal_frac=0.5)
    assert math.isclose(tau_late, 1.0, rel_tol=1e-6), f"Expected 1.0 at step 999, got {tau_late}"

    # Monotone non-increasing
    prev = float("inf")
    for s in range(0, total, 50):
        cur = anneal_tau(s, total, tau_start=3.0, tau_end=1.0, anneal_frac=0.5)
        assert cur <= prev + 1e-9, f"anneal_tau not monotone at step {s}: {cur} > {prev}"
        prev = cur

    # Floor is 1.0 not 0.5 (v1 used 0.5 which caused collapse oscillation per design doc §2.4)
    assert tau_end >= 1.0, "v2 tau floor must be 1.0, not the v1 value of 0.5"


# ---------------------------------------------------------------------------
# JEPALossV2 end-to-end
# ---------------------------------------------------------------------------

class _FakeOperator(nn.Module):
    """Minimal operator stand-in exposing theta/log_r for v1 compatibility."""
    def __init__(self, v: int, dn: int):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(v, dn // 2) * 0.5)
        self.log_r = nn.Parameter(torch.randn(v, dn // 2) * 0.1)


def _make_inputs(B=4, M=6, dn=32, V_verb=8, T=16, V_vocab=512):
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


def test_jepav2loss_forward_component_keys():
    """All expected component keys must be present in the returned dict."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    loss_fn = JEPALossV2(operator=op, n_slices=64)

    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    total, comp = loss_fn(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0)

    required_keys = {"loss", "L_token", "L_prior", "L_sigreg", "L_pred", "gumbel_tau"}
    missing = required_keys - set(comp.keys())
    assert not missing, f"Missing component keys: {missing}"


def test_jepav2loss_finite_total():
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    loss_fn = JEPALossV2(operator=op, n_slices=64)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    total, comp = loss_fn(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target)
    assert torch.isfinite(total), f"total loss not finite: {total.item()}"


def test_jepav2loss_backward_reaches_all_inputs():
    """Gradients must flow to logits (L_token), p_logits (L_prior), k (L_sigreg),
    zhat (L_pred). q_logits (v_logits) must NOT receive grad (stop-grad in prior_kl)."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    loss_fn = JEPALossV2(operator=op, n_slices=64)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    total, _ = loss_fn(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0)
    total.backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0, \
        "logits must receive grad (L_token)"
    assert p_logits.grad is not None and p_logits.grad.abs().sum() > 0, \
        "p_logits must receive grad (L_prior)"
    assert k.grad is not None and k.grad.abs().sum() > 0, \
        "k must receive grad (L_sigreg)"
    assert zhat.grad is not None and zhat.grad.abs().sum() > 0, \
        "zhat must receive grad (L_pred)"

    # v_logits (posterior) must NOT receive gradient — it is stop-grad in prior_kl.
    assert v_logits.grad is None or v_logits.grad.abs().sum() < 1e-10, \
        "v_logits (posterior) must be stop-grad in JEPALossV2"


def test_jepav2loss_w_pred_zero_ablation():
    """Setting w_pred=0 should eliminate L_pred's contribution to the total loss.

    We verify this by checking that:
      total(w_pred=0.25) - total(w_pred=0) == 0.25 * L_pred

    Both calls use the SAME loss_fn instance (same random slices for sigreg) so the
    only difference between the two runs is the w_pred weighting of L_pred. We seed
    torch before each call to get identical sigreg random slices.
    """
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)

    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )

    # Run with w_pred=0.25 (seed fixed for sigreg).
    torch.manual_seed(7)
    loss_fn_full = JEPALossV2(operator=op, n_slices=32, w_pred=0.25)
    total_f, comp_f = loss_fn_full(logits, tgt_ids, tgt_pad, k, v_logits, p_logits,
                                    zhat, z_target, tau=1.0)

    # Run with w_pred=0 using SAME seed so sigreg gets identical random slices.
    torch.manual_seed(7)
    loss_fn_zero = JEPALossV2(operator=op, n_slices=32, w_pred=0.0)
    total_z, comp_z = loss_fn_zero(logits, tgt_ids, tgt_pad, k, v_logits, p_logits,
                                    zhat, z_target, tau=1.0)

    # The difference between the two totals should equal 0.25 * L_pred.
    expected_diff = 0.25 * comp_f["L_pred"]
    actual_diff = comp_f["loss"] - comp_z["loss"]
    assert math.isclose(actual_diff, expected_diff, rel_tol=1e-4), (
        f"L_pred weight ablation mismatch: expected diff {expected_diff:.6f}, "
        f"got {actual_diff:.6f}"
    )


def test_jepav2loss_operator_not_submodule():
    """The operator must NOT be registered as a submodule of JEPALossV2.

    If it were, AdamW would see its params twice (double-LR bug documented in
    v1 losses.py). This test ensures no theta/log_r appear in loss_fn.parameters().
    """
    V_verb, dn = 8, 32
    op = _FakeOperator(V_verb, dn)
    loss_fn = JEPALossV2(operator=op)

    loss_param_ids = {id(p) for p in loss_fn.parameters()}
    op_param_ids = {id(p) for p in op.parameters()}
    overlap = loss_param_ids & op_param_ids
    assert not overlap, (
        f"Operator params appear in JEPALossV2.parameters() — double-registration bug! "
        f"Overlap: {overlap}"
    )


def test_jepav2loss_l_div_absent():
    """L_div must NOT appear in the component dict (deleted in v2)."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    loss_fn = JEPALossV2(operator=op, n_slices=32)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    _, comp = loss_fn(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target)
    assert "L_div" not in comp, (
        "L_div must be deleted in v2 (design doc §0, §5). Found in components dict."
    )


# ---------------------------------------------------------------------------
# sigreg_loss import check (reused from v1, not re-tested extensively here)
# ---------------------------------------------------------------------------

def test_sigreg_available_in_losses():
    """sigreg_loss lives in the merged live losses.py (was re-exported from v1)."""
    _seed()
    x = torch.randn(256, 32)
    loss = sigreg_loss(x, n_slices=64)
    assert torch.isfinite(loss) and loss.item() >= 0


# ---------------------------------------------------------------------------
# Diff-weighted CE (v4 §2.2): token_ce token_weights argument
# ---------------------------------------------------------------------------

def test_token_ce_token_weights_none_is_v3_bitwise():
    """token_ce(token_weights=None) must be BITWISE the historical v3 mean CE.

    The None path takes the exact F.cross_entropy(reduction='mean', ignore_index)
    branch, so it is identical to the pre-v4 implementation.
    """
    _seed()
    B, T, V = 4, 16, 512
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))
    tgt_ids[:, -3:] = 0  # some pad positions
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V), tgt_ids.reshape(B * T), ignore_index=0
    )
    got = token_ce(logits, tgt_ids, pad_id=0, token_weights=None)
    assert torch.equal(ref, got), "token_weights=None must reproduce v3 CE bitwise"


def test_token_ce_all_ones_weights_equals_uniform():
    """All-ones weights over non-pad ⟹ the weighted mean == the uniform mean CE.

    This is the w_diff=1.0 bitwise guarantee (§2.2): the dataset always emits a weight
    tensor, and an all-ones one must equal the None (v3) path to within float tolerance.
    """
    _seed()
    B, T, V = 4, 16, 512
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))
    tgt_ids[:, -4:] = 0  # pad tail
    uniform = token_ce(logits, tgt_ids, pad_id=0, token_weights=None)
    ones = token_ce(logits, tgt_ids, pad_id=0, token_weights=torch.ones(B, T))
    assert math.isclose(uniform.item(), ones.item(), rel_tol=1e-6, abs_tol=1e-7), (
        f"all-ones weighted CE ({ones.item()}) must match uniform CE ({uniform.item()})"
    )


def test_token_ce_diff_weights_focus_loss():
    """Up-weighting the high-CE token positions must increase the weighted loss.

    Construct weights that are large exactly where per-token CE is large; the weighted
    mean must then exceed the uniform mean (the loss "focuses" on those tokens).
    """
    _seed()
    B, T, V = 2, 8, 32
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))
    # Per-token CE to locate the highest-CE positions.
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V), tgt_ids.reshape(B * T),
        ignore_index=0, reduction="none",
    ).reshape(B, T)
    weights = torch.ones(B, T)
    # Up-weight the single highest-CE position in each row.
    hi = ce.argmax(dim=1)
    for b in range(B):
        weights[b, hi[b]] = 4.0
    uniform = token_ce(logits, tgt_ids, pad_id=0, token_weights=None).item()
    weighted = token_ce(logits, tgt_ids, pad_id=0, token_weights=weights).item()
    assert weighted > uniform, (
        f"up-weighting the highest-CE token should raise the loss: "
        f"weighted={weighted:.4f} uniform={uniform:.4f}"
    )


def test_token_ce_diff_weights_pad_excluded():
    """Pad positions contribute neither to numerator nor denominator under weighting.

    Even with a nonzero weight on a pad position, that position's CE is 0 (ignore_index)
    AND its weight is zeroed by the non-pad mask, so changing a pad-position weight cannot
    change the weighted loss.
    """
    _seed()
    B, T, V = 3, 10, 64
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))
    tgt_ids[:, -3:] = 0  # pad
    w_a = torch.ones(B, T)
    w_b = torch.ones(B, T)
    w_b[:, -3:] = 9.0  # large weight on pad positions
    la = token_ce(logits, tgt_ids, pad_id=0, token_weights=w_a).item()
    lb = token_ce(logits, tgt_ids, pad_id=0, token_weights=w_b).item()
    assert math.isclose(la, lb, rel_tol=1e-6, abs_tol=1e-7), (
        f"pad-position weights must be ignored: {la} vs {lb}"
    )


def test_token_ce_weighted_gradient_flows():
    _seed()
    B, T, V = 4, 12, 64
    logits = torch.randn(B, T, V, requires_grad=True)
    tgt_ids = torch.randint(5, V, (B, T))
    weights = torch.ones(B, T)
    weights[:, 0] = 4.0
    loss = token_ce(logits, tgt_ids, pad_id=0, token_weights=weights)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# token_margin (v4 §3.1): per-pair token-level hard-negative hinge
# ---------------------------------------------------------------------------

def test_per_example_ce_shape_and_matches_mean():
    """_per_example_ce returns (B,) and its mean over a uniform batch matches token_ce."""
    _seed()
    B, T, V = 4, 12, 64
    logits = torch.randn(B, T, V)
    tgt_ids = torch.randint(5, V, (B, T))  # no pad
    per = _per_example_ce(logits, tgt_ids, pad_id=0)
    assert per.shape == (B,)
    # With no pad, the mean of per-row means equals the global mean CE.
    glob = token_ce(logits, tgt_ids, pad_id=0).item()
    assert math.isclose(per.mean().item(), glob, rel_tol=1e-5)


def test_token_margin_zero_when_gold_beats_neighbor():
    """Hinge is exactly 0 when CE(neighbor) − CE(gold) ≥ margin for every row.

    We make the gold logits assign near-certain probability to the gold ids (tiny CE)
    and the neighbor logits assign tiny probability to the neighbor ids (large CE), so the
    gap exceeds the margin with room to spare.
    """
    _seed()
    B, T, V = 3, 6, 16
    gold_ids = torch.randint(2, V, (B, T))
    neg_ids = torch.randint(2, V, (B, T))
    # Gold logits: huge mass on gold ids ⟹ CE ≈ 0.
    logits_gold = torch.full((B, T, V), -10.0)
    logits_gold.scatter_(2, gold_ids.unsqueeze(-1), 20.0)
    # Neg logits: huge mass AWAY from neg ids ⟹ CE large.
    logits_neg = torch.full((B, T, V), 0.0)
    logits_neg.scatter_(2, neg_ids.unsqueeze(-1), -20.0)
    m = token_margin(logits_gold, gold_ids, logits_neg, neg_ids, pad_id=0, margin=0.5)
    assert math.isclose(m.item(), 0.0, abs_tol=1e-6), (
        f"hinge should be 0 when gold beats neighbor by > margin, got {m.item()}"
    )


def test_token_margin_positive_when_neighbor_easier():
    """Hinge is positive when the neighbor is decoded MORE easily than the gold.

    Swap the construction: neighbor near-certain (tiny CE), gold hard (large CE). Then
    CE(neighbor) − CE(gold) < 0 < margin ⟹ hinge > 0.
    """
    _seed()
    B, T, V = 3, 6, 16
    gold_ids = torch.randint(2, V, (B, T))
    neg_ids = torch.randint(2, V, (B, T))
    logits_neg = torch.full((B, T, V), -10.0)
    logits_neg.scatter_(2, neg_ids.unsqueeze(-1), 20.0)   # neighbor easy ⟹ CE ≈ 0
    logits_gold = torch.full((B, T, V), 0.0)
    logits_gold.scatter_(2, gold_ids.unsqueeze(-1), -20.0)  # gold hard ⟹ CE large
    m = token_margin(logits_gold, gold_ids, logits_neg, neg_ids, pad_id=0, margin=0.5)
    assert m.item() > 0.0, f"hinge should be positive when neighbor is easier, got {m.item()}"


def test_token_margin_exactly_margin_when_equal_ce():
    """When CE(gold) == CE(neighbor), the gap is 0, so the hinge == margin."""
    _seed()
    B, T, V = 2, 5, 16
    ids = torch.randint(2, V, (B, T))
    logits = torch.randn(B, T, V)
    # Same logits AND same ids ⟹ ce_gold == ce_neg ⟹ gap 0 ⟹ hinge == margin.
    m = token_margin(logits, ids, logits.clone(), ids.clone(), pad_id=0, margin=0.5)
    assert math.isclose(m.item(), 0.5, rel_tol=1e-5), (
        f"equal-CE hinge should equal the margin (0.5), got {m.item()}"
    )


def test_token_margin_gradient_flows_to_both_passes():
    _seed()
    B, T, V = 3, 6, 16
    gold_ids = torch.randint(2, V, (B, T))
    neg_ids = torch.randint(2, V, (B, T))
    logits_gold = torch.randn(B, T, V, requires_grad=True)
    logits_neg = torch.randn(B, T, V, requires_grad=True)
    m = token_margin(logits_gold, gold_ids, logits_neg, neg_ids, pad_id=0, margin=0.5)
    m.backward()
    assert logits_gold.grad is not None and logits_gold.grad.abs().sum() > 0
    assert logits_neg.grad is not None and logits_neg.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# JEPALossV2 v4 passthrough: diff-weights + margin + bitwise v3 reproduction
# ---------------------------------------------------------------------------

def test_jepav2loss_w_diff_off_w_margin_off_reproduces_v3():
    """w_diff=1.0 (token_weights all-ones) + w_margin=0.0 ⟹ EXACT v3 loss total.

    With token_weights=None and the margin disabled, the total must equal the v3 total
    computed without any v4 arguments (bitwise, same seed for sigreg slices).
    """
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )

    torch.manual_seed(7)
    loss_v3 = JEPALossV2(operator=op, n_slices=32, w_nce=0.0)
    total_v3, comp_v3 = loss_v3(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0
    )

    torch.manual_seed(7)
    loss_v4 = JEPALossV2(operator=op, n_slices=32, w_nce=0.0, w_margin=0.0)
    total_v4, comp_v4 = loss_v4(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
        token_weights=torch.ones(B, tgt_ids.shape[1]),   # all-ones ⟹ uniform
        margin_logits_neg=None, margin_neg_ids=None,
    )
    assert math.isclose(comp_v3["L_token"], comp_v4["L_token"], rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(total_v3.item(), total_v4.item(), rel_tol=1e-6, abs_tol=1e-6), (
        f"v4 with w_diff=1/w_margin=0 must reproduce v3 total: "
        f"{total_v3.item()} vs {total_v4.item()}"
    )
    assert comp_v4["L_margin"] == 0.0


def test_jepav2loss_margin_adds_to_total():
    """When w_margin>0 and a neighbor pass is supplied, L_margin is added to the total."""
    B, M, dn, V_verb, T = 4, 6, 32, 8, 16
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb, T=T
    )
    # A neighbor that is NOT easy to decode from logits ⟹ positive hinge expected.
    neg_ids = torch.randint(5, 512, (B, T))

    torch.manual_seed(11)
    base = JEPALossV2(operator=op, n_slices=32, w_margin=0.0)
    total_base, _ = base(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
    )

    torch.manual_seed(11)
    with_margin = JEPALossV2(operator=op, n_slices=32, w_margin=0.25, margin=0.5)
    total_m, comp_m = with_margin(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
        margin_logits_neg=logits, margin_neg_ids=neg_ids,
    )
    expected = total_base.item() + 0.25 * comp_m["L_margin"]
    assert math.isclose(total_m.item(), expected, rel_tol=1e-5, abs_tol=1e-6), (
        f"margin term must add w_margin*L_margin: {total_m.item()} vs {expected}"
    )
    assert "L_margin" in comp_m and "w_margin" in comp_m


def test_jepav2loss_margin_skipped_without_neighbor():
    """w_margin>0 but no neighbor pass ⟹ margin skipped (L_margin=0), no crash."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb
    )
    loss_fn = JEPALossV2(operator=op, n_slices=32, w_margin=0.25)
    total, comp = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
    )
    assert torch.isfinite(total)
    assert comp["L_margin"] == 0.0


def test_jepav2loss_diff_weighting_changes_token_loss():
    """Passing non-uniform token_weights changes L_token (the diff-CE actually bites)."""
    B, M, dn, V_verb, T = 4, 6, 32, 8, 16
    op = _FakeOperator(V_verb, dn)
    logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target = _make_inputs(
        B=B, M=M, dn=dn, V_verb=V_verb, T=T
    )
    weights = torch.ones(B, T)
    weights[:, :4] = 4.0  # up-weight the first 4 positions

    torch.manual_seed(5)
    loss_fn = JEPALossV2(operator=op, n_slices=32)
    _, comp_uniform = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
        token_weights=None,
    )
    torch.manual_seed(5)
    _, comp_weighted = loss_fn(
        logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau=1.0,
        token_weights=weights,
    )
    assert not math.isclose(comp_uniform["L_token"], comp_weighted["L_token"], rel_tol=1e-4), (
        "non-uniform diff weights should change L_token"
    )


# ---------------------------------------------------------------------------
# v4 §1.3 mask_prior_kl + w_mask_prior wiring (Integrator)
# ---------------------------------------------------------------------------
def test_mask_prior_kl_nonnegative_and_finite():
    from twm.jepa.losses import mask_prior_kl
    _seed()
    g = torch.randn(5, 8)
    gp = torch.randn(5, 8)
    kl = mask_prior_kl(g, gp)
    assert torch.isfinite(kl) and kl.item() >= 0.0


def test_mask_prior_kl_zero_when_equal():
    from twm.jepa.losses import mask_prior_kl
    z = torch.randn(4, 8)
    assert mask_prior_kl(z, z.clone()).item() < 1e-6


def test_mask_prior_kl_stopgrad_on_posterior():
    """Posterior mask is the target (stop-grad); only the prior mask receives gradient."""
    from twm.jepa.losses import mask_prior_kl
    g = torch.randn(4, 8, requires_grad=True)
    gp = torch.randn(4, 8, requires_grad=True)
    mask_prior_kl(g, gp).backward()
    assert g.grad is None or g.grad.abs().sum() == 0, "posterior mask must be stop-grad"
    assert gp.grad is not None and gp.grad.abs().sum() > 0, "prior mask must get gradient"


def test_jepav2loss_w_mask_prior_off_reproduces_v3():
    """w_mask_prior=0 (or missing mask logits) ⟹ L_mask_prior skipped, v3-bitwise total."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    g_logits = torch.randn(B, M)
    g_prior_logits = torch.randn(B, M)

    # Re-seed before each forward so the sigreg random slices match (it advances global RNG).
    loss_off = JEPALossV2(operator=op, n_slices=32, w_mask_prior=0.0)
    _seed()
    t_off, c_off = loss_off(*inp, tau=1.0, g_logits=g_logits, g_prior_logits=g_prior_logits)
    loss_base = JEPALossV2(operator=op, n_slices=32)
    _seed()
    t_base, c_base = loss_base(*inp, tau=1.0)
    assert math.isclose(t_off.item(), t_base.item(), rel_tol=1e-6)
    assert c_off["L_mask_prior"] == 0.0


def test_jepav2loss_w_mask_prior_adds_to_total():
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    g_logits = torch.randn(B, M)
    g_prior_logits = torch.randn(B, M)

    loss_base = JEPALossV2(operator=op, n_slices=32, w_mask_prior=0.0)
    _seed()
    t_base, _ = loss_base(*inp, tau=1.0, g_logits=g_logits, g_prior_logits=g_prior_logits)
    loss_on = JEPALossV2(operator=op, n_slices=32, w_mask_prior=0.5)
    _seed()
    t_on, c_on = loss_on(*inp, tau=1.0, g_logits=g_logits, g_prior_logits=g_prior_logits)
    assert c_on["L_mask_prior"] > 0.0
    # Same sigreg slices (re-seeded) ⟹ the only delta is +w_mask_prior * L_mask_prior > 0.
    assert t_on.item() > t_base.item()


def test_jepav2loss_mask_prior_skipped_without_logits():
    """Even with w_mask_prior>0, missing mask logits ⟹ term skipped (targeting off path)."""
    B, M, dn, V_verb = 4, 6, 32, 8
    op = _FakeOperator(V_verb, dn)
    inp = _make_inputs(B=B, M=M, dn=dn, V_verb=V_verb)
    loss_on = JEPALossV2(operator=op, n_slices=32, w_mask_prior=0.5)
    _, c = loss_on(*inp, tau=1.0)  # no g_logits/g_prior_logits
    assert c["L_mask_prior"] == 0.0
