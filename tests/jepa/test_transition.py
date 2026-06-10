"""Tests for the v2 action heads (Task A). Spec: research/jepa_v2_latent_actions.md
§2 (TransitionEncoder), §3 (PriorHead), §11 Task A.

Covers the frozen interface and the load-bearing v2 invariants:
  * forward shapes: v_onehot (B,V), v_logits (B,V), pool_t (B,d); p_logits (B,V)
  * hard one-hot path: v_onehot is a true one-hot (sums to 1, single 1 per row)
  * gradient flows to v_logits AND into the shared trunk through the soft path
  * posterior depends on text_{t+1}: shuffling t+1 within the batch changes the
    v distribution (the v2 raison-d'etre — v1's verb was a function of t only)
  * prior depends ONLY on text_t: identical pool_t -> identical p_logits, and
    p_logits is independent of any text_{t+1} (it never sees it by construction)
  * delta channel on/off changes the MLP input width (3d vs 2d) and is wired
  * param counts land near the §9 budget line items (26K / 4.7K nano)
"""

import torch
import torch.nn as nn

from twm.jepa.slot_encoder import SlotEncoder
from twm.jepa.transition import TransitionEncoder, PriorHead, masked_mean


# nano-v2 (spec §10 model block): d=64, V=8, transition.mlp_hidden=128, prior.mlp_hidden=64.
D_MODEL = 64
N_VERBS = 8
MLP_HIDDEN_POST = 128
MLP_HIDDEN_PRIOR = 64
VOCAB = 512
T_TEXT = 64

# §9 param-budget line items for nano-v2.
# NOTE: the doc's §2.2 arithmetic for the posterior MLP (25,864) undercounts the
# LayerNorm(128) by 128 — it counted only the LN weight, not the bias. The actual
# architecture as specified (Linear(3d->h) + GELU + LayerNorm(h) + Linear(h->V))
# is: fc1 24,704 + LN 256 + fc2 1,032 = 25,992. We assert the true architectural
# count (still well under the 250K total budget); the 128-param slip is the doc's.
SECTION9_TRANSITION = 25_992  # 3d->h GELU LN h->V, d=64 h=128 V=8 (LN weight+bias)
SECTION9_PRIOR = 4_680        # d->h GELU h->V, d=64 h=64 V=8


def _make_trunk(freeze_emb=True):
    """A real SlotEncoder so the posterior shares the actual text trunk (§2.1)."""
    emb = nn.Embedding(VOCAB, D_MODEL)
    if freeze_emb:
        emb.weight.requires_grad_(False)
    enc = SlotEncoder(
        emb,
        d_model=D_MODEL,
        d_noun=32,
        n_slots=8,
        n_verbs=N_VERBS,
        n_text_layers=2,
        tie_text_layers=True,
        n_heads=4,
        d_ff=128,
        n_slot_iters=3,
        max_text_tokens=T_TEXT,
    )
    return enc


def _make_posterior(use_delta=True, trunk=None):
    enc = trunk if trunk is not None else _make_trunk()
    post = TransitionEncoder(
        enc.encode_text,
        d_model=D_MODEL,
        n_verbs=N_VERBS,
        mlp_hidden=MLP_HIDDEN_POST,
        use_delta=use_delta,
    )
    return enc, post


def _batch(B=8, T=T_TEXT, pad_frac=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    src_ids = torch.randint(0, VOCAB, (B, T), generator=g)
    tgt_ids = torch.randint(0, VOCAB, (B, T), generator=g)
    src_pad = torch.zeros(B, T, dtype=torch.bool)
    tgt_pad = torch.zeros(B, T, dtype=torch.bool)
    cut = int(T * pad_frac)
    src_pad[:, cut:] = True
    tgt_pad[:, cut:] = True
    return src_ids, src_pad, tgt_ids, tgt_pad


# ---------------------------------------------------------------------------
# masked_mean helper
# ---------------------------------------------------------------------------

def test_masked_mean_ignores_pad():
    B, T, d = 3, 5, 4
    ctx = torch.randn(B, T, d)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, 3:] = True  # last 2 positions are pad
    pooled = masked_mean(ctx, pad)
    # Should equal the mean over the first 3 positions only.
    expected = ctx[:, :3].mean(dim=1)
    assert torch.allclose(pooled, expected, atol=1e-6)
    assert pooled.shape == (B, d)


def test_masked_mean_all_pad_is_zero_not_nan():
    ctx = torch.randn(2, 4, 6)
    pad = torch.ones(2, 4, dtype=torch.bool)  # entirely pad
    pooled = masked_mean(ctx, pad)
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled, torch.zeros_like(pooled))


def test_masked_mean_none_pad():
    ctx = torch.randn(2, 4, 6)
    pooled = masked_mean(ctx, None)
    assert torch.allclose(pooled, ctx.mean(dim=1), atol=1e-6)


# ---------------------------------------------------------------------------
# TransitionEncoder — shapes and hard one-hot
# ---------------------------------------------------------------------------

def test_posterior_forward_shapes():
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=8)
    v_onehot, v_logits, pool_t = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
    assert v_onehot.shape == (8, N_VERBS)
    assert v_logits.shape == (8, N_VERBS)
    assert pool_t.shape == (8, D_MODEL)
    assert torch.isfinite(v_onehot).all()
    assert torch.isfinite(v_logits).all()
    assert torch.isfinite(pool_t).all()


def test_posterior_hard_onehot_sums_to_one():
    """hard=True -> straight-through hard one-hot: each row sums to 1 with a single 1."""
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=16)
    with torch.no_grad():
        v_onehot, _, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
    row_sums = v_onehot.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    # Exactly one entry equal to 1 per row, rest 0.
    is_onehot = ((v_onehot == 0) | (v_onehot == 1)).all()
    assert is_onehot
    assert (v_onehot == 1).sum(dim=-1).eq(1).all()


def test_posterior_soft_path_is_distribution():
    """hard=False -> soft Gumbel sample: rows sum to 1, entries in (0,1)."""
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=8)
    with torch.no_grad():
        v_soft, _, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=False)
    row_sums = v_soft.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert (v_soft >= 0).all() and (v_soft <= 1).all()
    # Soft sample is not a hard one-hot (almost surely has fractional mass).
    assert ((v_soft > 1e-4) & (v_soft < 1 - 1e-4)).any()


# ---------------------------------------------------------------------------
# Gradient flow through the straight-through path
# ---------------------------------------------------------------------------

def test_gradient_reaches_v_logits_through_hard_st():
    """Hard straight-through must still pass gradient to v_logits (and the head)."""
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=8)
    v_onehot, v_logits, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
    # A scalar that depends on the (ST) sample. The ST estimator routes grad from
    # v_onehot back through the soft sample into v_logits.
    loss = v_onehot.sum()
    loss.backward()
    # fc2 produces v_logits; its weight/bias must receive gradient.
    assert post.fc2.weight.grad is not None
    assert post.fc2.weight.grad.abs().sum() > 0


def test_gradient_flows_into_shared_trunk():
    """Posterior gradient flows back into the SlotEncoder text trunk (§2.1, intended)."""
    enc = _make_trunk(freeze_emb=True)
    _, post = _make_posterior(trunk=enc)
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=8)
    v_onehot, _, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
    v_onehot.sum().backward()
    # The shared text self-attn block (trunk) must receive gradient — both the
    # text_t and text_{t+1} passes flow through it.
    trunk_block = enc.text_blocks[0]
    grads = [p.grad for p in trunk_block.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_gradient_soft_path():
    """Soft path (hard=False) also delivers gradient to v_logits."""
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=8)
    v_soft, _, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=False)
    v_soft.sum().backward()
    assert post.fc2.weight.grad is not None
    assert post.fc2.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Posterior DEPENDS on text_{t+1} (the v2 raison-d'etre)
# ---------------------------------------------------------------------------

def test_posterior_depends_on_target_permutation():
    """Shuffling text_{t+1} within the batch must change the v distribution.

    This is the central v2 invariant: the action is q(v | t, t+1), so permuting
    the t+1 partner of each t changes which action is inferred. (v1's verb was a
    function of t alone — this test would be meaningless there.)
    """
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=64, seed=1)

    with torch.no_grad():
        _, logits_a, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
        # Permute the t+1 partners across the batch (src fixed, tgt shuffled).
        perm = torch.randperm(src_ids.shape[0])
        # Ensure it is a real derangement-ish permutation (not the identity).
        while torch.equal(perm, torch.arange(src_ids.shape[0])):
            perm = torch.randperm(src_ids.shape[0])
        _, logits_b, _ = post(
            src_ids, src_pad, tgt_ids[perm], tgt_pad[perm], tau=1.0, hard=True
        )

    # Per-example logits must change when the t+1 partner changes.
    assert not torch.allclose(logits_a, logits_b, atol=1e-4), (
        "Posterior v_logits did not change when text_{t+1} was permuted — the "
        "posterior is ignoring the next state (v1 failure mode)."
    )

    # The argmax-v distribution over the batch should also shift.
    argmax_a = logits_a.argmax(dim=-1)
    argmax_b = logits_b.argmax(dim=-1)
    hist_a = torch.bincount(argmax_a, minlength=N_VERBS).float()
    hist_b = torch.bincount(argmax_b, minlength=N_VERBS).float()
    assert not torch.equal(hist_a, hist_b) or not torch.equal(argmax_a, argmax_b)


def test_posterior_pool_t_independent_of_target():
    """pool_t (the returned text_t pool) must NOT depend on text_{t+1}."""
    _, post = _make_posterior()
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=16, seed=2)
    with torch.no_grad():
        _, _, pool_a = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
        perm = torch.randperm(16)
        _, _, pool_b = post(src_ids, src_pad, tgt_ids[perm], tgt_pad[perm], tau=1.0, hard=True)
    # pool_t reads only text_t -> identical regardless of the t+1 partner.
    assert torch.allclose(pool_a, pool_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Delta channel
# ---------------------------------------------------------------------------

def test_delta_channel_changes_input_width():
    enc = _make_trunk()
    _, post_delta = _make_posterior(use_delta=True, trunk=enc)
    _, post_nodelta = _make_posterior(use_delta=False, trunk=enc)
    assert post_delta.fc1.in_features == 3 * D_MODEL
    assert post_nodelta.fc1.in_features == 2 * D_MODEL
    # Both still produce valid logits.
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=4)
    for post in (post_delta, post_nodelta):
        _, logits, _ = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
        assert logits.shape == (4, N_VERBS)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# PriorHead — shape and t-only dependence
# ---------------------------------------------------------------------------

def test_prior_forward_shape():
    prior = PriorHead(d_model=D_MODEL, n_verbs=N_VERBS, mlp_hidden=MLP_HIDDEN_PRIOR)
    pool_t = torch.randn(8, D_MODEL)
    p_logits = prior(pool_t)
    assert p_logits.shape == (8, N_VERBS)
    assert torch.isfinite(p_logits).all()


def test_prior_deterministic_in_pool_t():
    """Prior is a pure function of pool_t — same pool_t -> same logits."""
    prior = PriorHead(d_model=D_MODEL, n_verbs=N_VERBS, mlp_hidden=MLP_HIDDEN_PRIOR)
    prior.eval()
    pool_t = torch.randn(8, D_MODEL)
    with torch.no_grad():
        a = prior(pool_t)
        b = prior(pool_t.clone())
    assert torch.allclose(a, b)


def test_prior_depends_only_on_t_not_target():
    """End-to-end: prior built on pool_t is invariant to text_{t+1} permutation.

    The model feeds the posterior's pool_t (text_t only) into the prior; since
    pool_t is independent of text_{t+1} (proven above), the prior logits cannot
    depend on the next state. Mirrors the model's wiring (§3: reuse pool_t).
    """
    _, post = _make_posterior()
    prior = PriorHead(d_model=D_MODEL, n_verbs=N_VERBS, mlp_hidden=MLP_HIDDEN_PRIOR)
    src_ids, src_pad, tgt_ids, tgt_pad = _batch(B=16, seed=3)
    with torch.no_grad():
        _, _, pool_a = post(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
        perm = torch.randperm(16)
        _, _, pool_b = post(src_ids, src_pad, tgt_ids[perm], tgt_pad[perm], tau=1.0, hard=True)
        p_a = prior(pool_a)
        p_b = prior(pool_b)
    assert torch.allclose(p_a, p_b, atol=1e-6), (
        "Prior logits changed when text_{t+1} was permuted — the prior is "
        "leaking next-state info (must read text_t only)."
    )


def test_prior_no_layernorm():
    """Spec §3: prior is minimal — no LayerNorm (reads already-LN'd trunk pool)."""
    prior = PriorHead(d_model=D_MODEL, n_verbs=N_VERBS, mlp_hidden=MLP_HIDDEN_PRIOR)
    assert not any(isinstance(m, nn.LayerNorm) for m in prior.modules())


# ---------------------------------------------------------------------------
# Param budget (§9): posterior MLP ~25.9K, prior ~4.7K. Shared trunk = 0 new.
# ---------------------------------------------------------------------------

def test_posterior_param_count_no_trunk():
    """TransitionEncoder owns ONLY its MLP head — the shared trunk is not its param."""
    _, post = _make_posterior(use_delta=True)
    n = sum(p.numel() for p in post.parameters())
    assert n == SECTION9_TRANSITION, (
        f"TransitionEncoder param count {n:,} != §9 line item {SECTION9_TRANSITION:,}. "
        f"If this changed, the shared trunk may have been accidentally registered."
    )


def test_prior_param_count():
    prior = PriorHead(d_model=D_MODEL, n_verbs=N_VERBS, mlp_hidden=MLP_HIDDEN_PRIOR)
    n = sum(p.numel() for p in prior.parameters())
    assert n == SECTION9_PRIOR, f"PriorHead param count {n:,} != §9 line item {SECTION9_PRIOR:,}."


def test_trunk_not_registered_as_submodule():
    """The encode_text callable must NOT register the SlotEncoder under the posterior."""
    enc = _make_trunk()
    _, post = _make_posterior(trunk=enc)
    # No child module of `post` should be the SlotEncoder (no trunk params captured).
    for _, m in post.named_modules():
        assert not isinstance(m, SlotEncoder)
