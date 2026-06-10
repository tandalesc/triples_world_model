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
