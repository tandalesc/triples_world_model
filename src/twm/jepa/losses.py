"""JEPA losses (the live v2 path). Spec: research/jepa_v2_latent_actions.md §5.

Combined loss:
    L = w_token·L_token + w_prior·L_prior + w_sigreg·L_sigreg + w_pred·L_pred
        + w_nce·L_nce

  - L_token (CE, weight 1.0): cross-entropy of text_{t+1} tokens given operator-
    transformed slots a*. This is THE primary loss — it forces the discrete latent
    action v to carry information about the causal step. Token CE is the grounding
    loss; v1's anti-goal "no token decoder" is explicitly revoked (design doc §0).
  - L_prior (KL, weight 0.1): KL(stopgrad(softmax(q/τ)) ‖ softmax(p)) — distills the
    posterior action distribution into the prior (which runs on state_t only). Enables
    autonomous rollout without the posterior at inference.
  - L_sigreg (0.05): isotropic-Gaussian GoF on standardized nouns. Keeps the noun
    space isotropic; standardize, never L2.
  - L_pred (0.25, optional): MSE(zhat, z.detach()) EMA aux objective. Keeps the JEPA
    latent objective alive as a regularizer on a*. Setting w_pred=0 is a supported
    ablation (pure token-grounding).
  - L_nce (0.25 in v3, default 0.0): InfoNCE next-state contrastive. The DISCRIMINATIVE
    upgrade of L_pred — same pooled anchor zhat vs stop-grad EMA key z_target, turned into
    a softmax-CE that pulls toward the true next-state AND pushes away in-batch + same-chain
    negatives (the retrieval objective hard_mrr needs). Computed only when w_nce>0; in the
    v3 recipe it takes over w_pred's 0.25 slot (mutually exclusive). w_nce=0 reproduces
    exact v2.1 behavior (recoverability — design doc §1.7).

L_div is GONE: verb informativeness comes from necessity (the decoder needs v's bits)
not a gameable regularizer. Codebook usage is a diagnostic only (diagnostics.py). The
dead v1 L_div helpers (usage_entropy, spread_penalty, old JEPALoss) live in
legacy/losses_v1.py and are never imported on the live path.

Leakage constraint (§6): the decoder must see ONLY {a*_i} as memory — never raw
text_{t+1} encodings or the posterior features. token_ce() is called with logits
produced by a decoder whose memory is a*, never with any shortcut conditioning.
This module enforces the constraint at the signature level: token_ce and prior_kl
have no parameter that could carry text_{t+1} activations directly.

Shared utilities `sigreg_loss`, `_standardize`, `_epps_pulley_gof`, `anneal_tau`,
and `gumbel_softmax_sample` live in THIS file (the merged live losses module).
transition.py imports `gumbel_softmax_sample` from here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gumbel-softmax verb sampling + temperature anneal
# ---------------------------------------------------------------------------

def anneal_tau(
    step: int,
    total_steps: int,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
    anneal_frac: float = 0.3,
) -> float:
    """Annealed Gumbel temperature τ_g: tau_start -> tau_end, linearly, over the
    first `anneal_frac` of training, then held flat at tau_end.
    """
    anneal_steps = max(1, int(round(anneal_frac * total_steps)))
    if step >= anneal_steps:
        return float(tau_end)
    frac = step / anneal_steps  # in [0, 1)
    return float(tau_start + (tau_end - tau_start) * frac)


def gumbel_softmax_sample(
    logits: torch.Tensor,
    tau: float,
    hard: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Gumbel-softmax over the verb axis (last dim).

    Args:
        logits: (..., V) verb logits.
        tau:    Gumbel temperature (annealed during training).
        hard:   if True, straight-through hard one-hot (eval/export); the forward
                value is a hard argmax one-hot while the gradient flows through the
                soft sample.

    Returns:
        (..., V) soft (or straight-through-hard) assignment that sums to 1 over V.
    """
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    y = F.softmax((logits + gumbel) / tau, dim=-1)
    if not hard:
        return y
    idx = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    return y_hard + (y - y.detach())


# ---------------------------------------------------------------------------
# SIGReg: sliced isotropic-Gaussian goodness-of-fit
# ---------------------------------------------------------------------------

def _standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Center + scale by BATCH statistics (zero-mean / unit-var per dim).

    NOT per-vector L2-norm. Projecting nouns onto the unit sphere makes every 1D
    projection have std ≈ 1/√dn, which saturates the N(0,1)-calibrated GoF and
    kills its gradient. Standardizing keeps the test on a scale where it has a
    live gradient.
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


def _epps_pulley_gof(
    proj: torch.Tensor,
    n_knots: int = 17,
    knot_max: float = 3.0,
) -> torch.Tensor:
    """Epps-Pulley characteristic-function GoF of 1D samples `proj` vs N(0,1).

    The empirical characteristic function is φ_emp(t) = (1/N) Σ_n exp(i t x_n);
    for N(0,1) the target is φ_0(t) = exp(-t²/2) (real, imag 0). We integrate the
    squared modulus of the difference over a knot grid on [0, knot_max] with the
    trapezoid rule. Zero iff the standardized samples are unit Gaussian.
    """
    n = proj.shape[0]
    t = torch.linspace(0.0, knot_max, n_knots, device=proj.device, dtype=proj.dtype)
    dt = knot_max / (n_knots - 1)
    w = torch.full((n_knots,), dt, device=proj.device, dtype=proj.dtype)
    w[0] *= 0.5
    w[-1] *= 0.5  # trapezoid endpoint weights

    arg = proj.unsqueeze(-1) * t.view(1, 1, -1)        # (N, S, K)
    re = torch.cos(arg).mean(dim=0)                     # (S, K)  Re φ_emp
    im = torch.sin(arg).mean(dim=0)                     # (S, K)  Im φ_emp
    target_re = torch.exp(-0.5 * t * t).view(1, -1)     # (1, K)  Re φ_0 (real)

    diff2 = (re - target_re).pow(2) + im.pow(2)         # (S, K)
    integral = (diff2 * w.view(1, -1)).sum(dim=-1)      # (S,)  ∫ over t per slice
    return integral.mean()


def sigreg_loss(
    nouns: torch.Tensor,
    n_slices: int = 256,
    n_knots: int = 17,
    knot_max: float = 3.0,
    standardize: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced isotropic-Gaussian GoF on a batch of nouns.

    Args:
        nouns: (..., dn) — flattened to (N, dn) over all leading dims.
        n_slices: number of random unit directions to project onto.
        standardize: MUST be True. False is a documented negative path — on
            un-standardized / sphere-projected inputs the GoF saturates and the
            gradient vanishes; kept only so tests can demonstrate the failure.

    Returns:
        scalar SIGReg loss (≈ 0 when nouns are isotropic unit Gaussian after
        standardization, >> 0 under rank collapse / anisotropy).
    """
    x = nouns.reshape(-1, nouns.shape[-1])
    dn = x.shape[-1]
    if standardize:
        x = _standardize(x)

    dirs = torch.randn(dn, n_slices, device=x.device, dtype=x.dtype, generator=generator)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    proj = x @ dirs                                     # (N, S)
    return _epps_pulley_gof(proj, n_knots=n_knots, knot_max=knot_max)


# ---------------------------------------------------------------------------
# token_ce: cross-entropy of text_{t+1} given a* memory (primary loss)
# ---------------------------------------------------------------------------

def token_ce(
    logits: torch.Tensor,       # (B, T, V_vocab) — output of TokenDecoder.forward(a*, tgt_ids, tgt_pad)
    tgt_ids: torch.Tensor,      # (B, T) target token ids
    pad_id: int = 0,
    token_weights: torch.Tensor | None = None,  # (B, T) per-token diff weights (§2.2); None ⟹ uniform
) -> torch.Tensor:
    """Cross-entropy loss for autoregressive token prediction.

    ARDecoder.forward(a*, tgt_ids, tgt_pad) returns logits (B, T, V_vocab) aligned to
    tgt_ids positions 0..T-1: internally it prepends a learned BOS embedding and feeds
    [bos, tgt[:-1]] to the decoder, so logits[b, t] predicts tgt_ids[b, t].

    Loss = CE(logits.reshape(-1, V), tgt_ids.reshape(-1), ignore_index=pad_id).
    Pad positions are excluded from the denominator (ignore_index). EOS (id 4) is
    included as a real predicted token (not masked), so the decoder learns to stop.

    Diff-weighting (v4 §2.2): when `token_weights` is given, each non-pad position's CE
    is scaled by its weight (diff tokens get `w_diff`, boilerplate 1.0) and the loss is a
    WEIGHTED mean (`Σ w·ce / Σ w` over non-pad positions). When `token_weights` is None
    — OR an all-ones tensor over the non-pad positions — this is BITWISE the v3 mean CE
    (same `ignore_index`, same denominator), so `w_diff=1.0` reproduces v3 exactly.

    Args:
        logits:  (B, T, V_vocab) — decoder logits (no manual shift needed; the shift
                 is baked into ARDecoder's [bos]+tgt[:-1] construction).
        tgt_ids: (B, T) target token ids (EOS appended before pad, as §7 requires).
        pad_id:  id of the padding token (0 in jepa_bpe_512.json).
        token_weights: optional (B, T) float per-token weights. None ⟹ uniform (v3 path).

    Returns:
        scalar (weighted) mean CE over non-pad positions.
    """
    B, T, V = logits.shape
    if token_weights is None:
        # v3-bitwise path: identical to the historical implementation.
        return F.cross_entropy(
            logits.reshape(B * T, V),
            tgt_ids.reshape(B * T),
            ignore_index=pad_id,
        )
    # Weighted path: per-token CE (no reduction), then a weighted mean over non-pad.
    ce = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_ids.reshape(B * T),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, T)                                   # (B, T) — 0 at ignored (pad) positions
    valid = (tgt_ids != pad_id).to(ce.dtype)          # (B, T) non-pad mask
    w = token_weights.to(ce.dtype) * valid            # zero out pad weights defensively
    return (ce * w).sum() / w.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# token_margin: token-level hard-negative contrastive (the encoder-AIM fix, §3)
# ---------------------------------------------------------------------------

def _per_example_ce(
    logits: torch.Tensor,       # (B, T, V_vocab)
    tgt_ids: torch.Tensor,      # (B, T)
    pad_id: int = 0,
) -> torch.Tensor:
    """Per-example mean CE: the non-pad mean cross-entropy of each row (B,).

    Reuses the `reduction="none"` CE path of `token_ce`, reduced PER ROW so the margin
    (§3.1) is on the same scale (nats) as `L_token`. Pad positions are excluded from
    both numerator and denominator. A row that is entirely pad contributes 0.0 (the
    denominator is clamped to 1.0), which never happens for a real target.
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_ids.reshape(B * T),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, T)                                   # (B, T) — 0 at pad positions
    valid = (tgt_ids != pad_id).to(ce.dtype)          # (B, T)
    return (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)   # (B,)


def token_margin(
    logits_gold: torch.Tensor,  # (B, T, V) decoder run on a* teacher-forced on the GOLD target
    gold_ids: torch.Tensor,     # (B, T) gold target ids
    logits_neg: torch.Tensor,   # (B, T, V) decoder run on the SAME a* teacher-forced on a NEIGHBOR
    neg_ids: torch.Tensor,      # (B, T) same-chain neighbor target ids
    pad_id: int = 0,
    margin: float = 0.5,
) -> torch.Tensor:
    """Per-pair token-level hard-negative hinge (design v4 §3.1).

        L_margin = mean_i  max(0,  m − ( CE(neighbor_i | a*_i) − CE(gold_i | a*_i) ) )

    Decoding the WRONG (same-chain neighbor) next-state from a*_i should cost MORE CE than
    decoding the right one, by at least `margin` nats. Both decoder runs use the SAME a*_i
    memory (`logits_gold` and `logits_neg` are produced from the gold's operator output);
    only the teacher-forced target ids differ. This is the encoder-AIM fix (separation
    0.53 → >0.7): unlike pooled InfoNCE, the hinge bites at the token level where the
    discriminative clause lives.

    The hinge is zero whenever `ce_neg − ce_gold ≥ margin` (gold already beats the
    neighbor by the margin) and positive otherwise. Gradient flows into BOTH passes' a*
    geometry (push gold-likely / neighbor-unlikely) and into the decoder.

    Args:
        logits_gold/logits_neg: (B, T, V) — decoder logits on the SAME a* memory.
        gold_ids/neg_ids:       (B, T) — gold vs neighbor teacher-forced target ids.
        pad_id:  padding id (excluded from the per-example CE, as in token_ce).
        margin:  hinge margin in nats (default 0.5).

    Returns:
        scalar mean hinge over the batch (≥ 0).
    """
    ce_gold = _per_example_ce(logits_gold, gold_ids, pad_id)   # (B,)
    ce_neg = _per_example_ce(logits_neg, neg_ids, pad_id)      # (B,)
    return F.relu(margin - (ce_neg - ce_gold)).mean()


# ---------------------------------------------------------------------------
# info_nce: discriminative next-state contrastive (replaces vestigial L_pred)
# ---------------------------------------------------------------------------

def info_nce(
    zhat: torch.Tensor,                       # (B, dn) anchor (gradient) = Predictor(Readout(a*))
    z_target: torch.Tensor,                   # (B, dn) positive key (stop-grad) = EMA.pool_raw(text_{t+1})
    chain_ids: torch.Tensor | None = None,    # (B,) long — same-chain hard-negative bookkeeping (§1.4)
    temperature: float = 0.1,
    neg_keys: torch.Tensor | None = None,     # (B, n_neg, dn) extra hard-neg keys (unroll cross-hop, §2.4)
) -> torch.Tensor:
    """InfoNCE next-state contrastive loss (design doc §1.3).

    Standard InfoNCE with cosine similarity and temperature τ. The anchor `zhat` is the
    model's predicted next-state pool over the operator-transformed slots a*; the positive
    key `z_target` is the stop-grad EMA raw-noun readout pool of the TRUE next state. The
    loss pulls each anchor toward its own key (the diagonal) and pushes it away from every
    other key in the batch (in-batch negatives) plus any explicit `neg_keys` (cross-hop
    hard negatives in unroll mode).

    Negatives (§1.4):
      - In-batch: every off-diagonal key kn[j], j≠i.
      - Same-chain (`chain_ids` given): off-diagonal columns that share row i's chain are
        already hard negatives in the (B,B) matrix — kept as negatives (they are wrong
        next-states on the same narrative). The mask is only used DEFENSIVELY to drop a
        column that would duplicate the anchor's own positive (a chain contributing the
        same target twice); such columns are masked to -inf so they never count as a
        spurious negative. The diagonal positive is never masked.
      - `neg_keys` (§2.4): explicit per-anchor hard negatives (the cross-hop sibling key,
        e.g. z_2 when scoring hop 1), appended as extra logit columns.

    The positive key is stop-grad: only the anchor receives gradient (MoCo/BYOL asymmetry
    inherited from L_pred — cannot drive representational collapse). z_target is detached
    defensively here regardless of the caller.

    Args:
        zhat:       (B, dn) anchor (receives gradient).
        z_target:   (B, dn) positive key (detached internally).
        chain_ids:  optional (B,) long; same-chain bookkeeping (see above). When None,
                    plain in-batch InfoNCE.
        temperature: τ_nce (cosine-sim divisor). 0.1 by default (SimCLR/MoCo).
        neg_keys:   optional (B, n_neg, dn) explicit hard-negative keys (detached
                    internally), appended as extra columns of the logits matrix.

    Returns:
        scalar InfoNCE loss (≈ 0 when each anchor's normalized direction matches its own
        key and is orthogonal to all negatives).
    """
    B = zhat.shape[0]
    # Positive key (and explicit negatives) are stop-grad — only the anchor gets gradient.
    z_target = z_target.detach()

    qn = F.normalize(zhat, dim=-1)               # (B, dn)
    kn = F.normalize(z_target, dim=-1)           # (B, dn)

    logits = (qn @ kn.t()) / temperature         # (B, B) cosine sim to every in-batch key

    # Defensive same-chain handling (§1.4): mask only OFF-DIAGONAL columns that exactly
    # duplicate the anchor's own positive key (a chain contributing the same target twice).
    # We never mask the diagonal (the true positive). Plain same-chain wrong-next-states
    # stay as negatives by design.
    if chain_ids is not None:
        same = chain_ids.view(-1, 1) == chain_ids.view(1, -1)   # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        # A column j (j≠i) duplicates i's positive iff same-chain AND its key matches i's
        # key direction (cosine ≈ 1). Detect duplicate keys to avoid a false negative.
        key_sim = (kn @ kn.t())                                 # (B, B) key-key cosine
        dup_pos = same & (~eye) & (key_sim > 1.0 - 1e-4)
        logits = logits.masked_fill(dup_pos, float("-inf"))

    # Explicit cross-hop hard negatives (§2.4): append (B, n_neg) extra columns.
    if neg_keys is not None:
        neg = F.normalize(neg_keys.detach(), dim=-1)            # (B, n_neg, dn)
        neg_logits = torch.einsum("bd,bnd->bn", qn, neg) / temperature  # (B, n_neg)
        logits = torch.cat([logits, neg_logits], dim=1)         # (B, B + n_neg)

    labels = torch.arange(B, device=logits.device)              # diagonal is the positive
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# prior_kl: KL(stopgrad q ‖ p) — distill posterior into prior
# ---------------------------------------------------------------------------

def prior_kl(
    q_logits: torch.Tensor,    # (B, V) posterior logits (from TransitionEncoder)
    p_logits: torch.Tensor,    # (B, V) prior logits (from PriorHead)
    tau: float = 1.0,
) -> torch.Tensor:
    """KL(stopgrad(softmax(q/τ)) ‖ softmax(p)).

    The posterior is the target (stop-grad): the prior learns to imitate the
    posterior's action distribution from state_t only. Forward KL (mode-covering)
    so the prior keeps mass on all actions the posterior uses — correct for
    multi-modal rollout (design doc §3.1).

    τ = the current annealed posterior temperature, so prior and posterior are
    compared at the same sharpness. Using the same τ prevents spurious gradient
    when the posterior is still soft (large τ) from being amplified by the prior.

    Args:
        q_logits: (B, V) logits from the posterior TransitionEncoder.
        p_logits: (B, V) logits from the PriorHead.
        tau:      current Gumbel/posterior temperature.

    Returns:
        scalar mean KL, ≥ 0; = 0 iff q and p are identical distributions.
    """
    # Posterior target distribution (stop gradient — prior must follow, not steer).
    with torch.no_grad():
        q_dist = F.softmax(q_logits / tau, dim=-1)  # (B, V)

    # Prior distribution (receives gradient).
    p_log = F.log_softmax(p_logits, dim=-1)          # (B, V)

    # KL(q ‖ p) = Σ q * (log q − log p).
    # F.kl_div expects (log_predictions, targets): KL(target ‖ input).
    kl = F.kl_div(p_log, q_dist, reduction="batchmean", log_target=False)
    return kl.clamp_min(0.0)


def mask_prior_kl(
    g_logits: torch.Tensor,        # (B, M) posterior per-slot mask logits (from s_t, s_{t+1})
    g_prior_logits: torch.Tensor,  # (B, M) prior per-slot mask logits (from s_t alone)
) -> torch.Tensor:
    """KL( stopgrad(Bernoulli(σ(g_logits))) ‖ Bernoulli(σ(g_prior_logits)) ), per slot (§1.3).

    The targeted-action mask is the analogue of the verb prior: at rollout the model has
    only s_t, so the PriorHead must predict WHICH slots an action will touch from the start
    state alone. The posterior mask (which reads s_{t+1}) is the stop-grad target; the prior
    learns to imitate it. Per-slot independent Bernoulli KL, averaged over slots and batch.

    Forward KL (mode-covering, matching `prior_kl`): the prior keeps mass on every slot the
    posterior touches. = 0 iff the prior matches the posterior gate probability on every slot.

    Args:
        g_logits:       (B, M) posterior mask logits — the target (stop-grad).
        g_prior_logits: (B, M) prior mask logits — receives gradient.

    Returns:
        scalar mean per-slot Bernoulli KL, ≥ 0.
    """
    eps = 1e-6
    with torch.no_grad():
        q = torch.sigmoid(g_logits).clamp(eps, 1.0 - eps)  # (B, M) posterior gate prob
    p = torch.sigmoid(g_prior_logits).clamp(eps, 1.0 - eps)  # (B, M) prior gate prob
    # KL(q ‖ p) for Bernoulli = q log(q/p) + (1−q) log((1−q)/(1−p)), per slot.
    kl = q * (q.log() - p.log()) + (1.0 - q) * ((1.0 - q).log() - (1.0 - p).log())
    return kl.mean().clamp_min(0.0)


# ---------------------------------------------------------------------------
# JEPALossV2 aggregator
# ---------------------------------------------------------------------------

class JEPALossV2(nn.Module):
    """Combined JEPA v2 training loss (design doc §5).

    The operator is held by reference (not owned as a submodule) so its
    theta/log_r parameters are not double-registered in AdamW — same pattern
    as the v1 JEPALoss._operator_ref pattern (legacy/losses_v1.py).

    Forward signature (frozen per design doc §11, Task D):
        forward(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau)
        -> (total: Tensor, components: dict)
    """

    def __init__(
        self,
        operator: nn.Module | None = None,
        w_token: float = 1.0,
        w_prior: float = 0.1,
        w_sigreg: float = 0.05,
        w_pred: float = 0.25,
        w_nce: float = 0.0,
        w_margin: float = 0.0,
        margin: float = 0.5,
        w_mask_prior: float = 0.0,
        nce_temperature: float = 0.1,
        n_slices: int = 256,
        n_knots: int = 17,
        knot_max: float = 3.0,
        standardize: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        # Store operator by reference without submodule registration.
        # (plain self.operator = op would register its params into JEPALossV2,
        # causing duplicate AdamW entries — see legacy/losses_v1.py for the rationale.)
        object.__setattr__(self, "_operator_ref", [operator])
        self.w_token = w_token
        self.w_prior = w_prior
        self.w_sigreg = w_sigreg
        self.w_pred = w_pred
        self.w_nce = w_nce
        # v4 §3: token-level hard-negative margin. w_margin=0.0 ⟹ off (v3 bitwise).
        self.w_margin = w_margin
        self.margin = margin
        # v4 §1.3: targeted-mask prior KL. w_mask_prior=0.0 ⟹ off (v3 bitwise); skipped
        # entirely when the model does not emit prior-mask logits (targeting off).
        self.w_mask_prior = w_mask_prior
        self.nce_temperature = nce_temperature
        self.n_slices = n_slices
        self.n_knots = n_knots
        self.knot_max = knot_max
        self.standardize = standardize
        self.pad_id = pad_id

    def forward(
        self,
        logits: torch.Tensor,        # (B, T, V_vocab) decoder output for text_{t+1}
        tgt_ids: torch.Tensor,       # (B, T) target token ids
        tgt_pad: torch.Tensor,       # (B, T) bool padding mask (True = pad; unused here,
                                     #   CE uses ignore_index instead — kept in signature
                                     #   for caller symmetry with the dataset dict keys)
        k: torch.Tensor,             # (B, M, dn) nouns (for L_sigreg)
        v_logits: torch.Tensor,      # (B, V) posterior action logits (q)
        p_logits: torch.Tensor,      # (B, V) prior action logits (p)
        zhat: torch.Tensor,          # (B, dn) predicted latent (anchor; L_pred + L_nce)
        z_target: torch.Tensor,      # (B, dn) EMA target latent, stop-grad (positive key)
        tau: float = 1.0,            # current Gumbel temperature (for L_prior KL sharpness)
        chain_ids: torch.Tensor | None = None,   # (B,) long — same-chain negatives (§1.4)
        nce_neg_keys: torch.Tensor | None = None,  # (B, n_neg, dn) cross-hop hard negs (§2.4)
        token_weights: torch.Tensor | None = None,   # (B, T) diff-CE per-token weights (§2.2)
        margin_logits_neg: torch.Tensor | None = None,  # (B, T, V) decoder on a* + neighbor ids (§3)
        margin_neg_ids: torch.Tensor | None = None,      # (B, T) same-chain neighbor target ids (§3)
        g_logits: torch.Tensor | None = None,         # (B, M) posterior mask logits (§1.3)
        g_prior_logits: torch.Tensor | None = None,   # (B, M) prior mask logits (§1.3)
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss and per-term components.

        The InfoNCE term (`L_nce`) is computed ONLY when `w_nce > 0` (skips the similarity
        matmul otherwise, so v2.1 configs pay zero cost). `w_nce` takes over `w_pred`'s slot
        in the v3 recipe; the default v3 config sets `w_pred=0.0, w_nce=0.25`, and setting
        `w_nce=0.0, w_pred=0.25` recovers exact v2.1 behavior. This call is hop-agnostic:
        the unroll trainer (Task B) calls it once per hop and sums with hop weights
        outside.

        Returns:
            (total, components) where components is a flat dict of scalar floats for
            logging (plus 'loss' = total.item() for convenience).
        """
        # L_token: primary grounding loss — CE of text_{t+1} tokens given a* memory.
        # Diff-weighted when token_weights is given (§2.2); None ⟹ v3-bitwise mean CE.
        l_token = token_ce(logits, tgt_ids, pad_id=self.pad_id, token_weights=token_weights)

        # L_prior: KL from posterior to prior, enables autonomous rollout.
        l_prior = prior_kl(v_logits, p_logits, tau=tau)

        # L_sigreg: isotropic-Gaussian GoF on standardized nouns (unchanged from v1).
        l_sigreg = sigreg_loss(
            k,
            n_slices=self.n_slices,
            n_knots=self.n_knots,
            knot_max=self.knot_max,
            standardize=self.standardize,
        )

        # L_pred: JEPA latent MSE aux objective (optional; w_pred=0 ablates it).
        l_pred = F.mse_loss(zhat, z_target.detach())

        # L_nce: discriminative next-state contrastive (replaces L_pred in v3). Only
        # computed when active — when w_nce=0 we skip the matmul and report 0.0.
        if self.w_nce > 0:
            l_nce = info_nce(
                zhat,
                z_target,
                chain_ids=chain_ids,
                temperature=self.nce_temperature,
                neg_keys=nce_neg_keys,
            )
        else:
            l_nce = None

        # L_margin: token-level hard-negative hinge (§3). Computed only when active AND the
        # train loop supplies the neighbor pass (logits/ids on the SAME a* memory). When
        # w_margin=0.0 (or the neighbor pass is absent) it is skipped ⟹ v3-bitwise total.
        if self.w_margin > 0 and margin_logits_neg is not None and margin_neg_ids is not None:
            l_margin = token_margin(
                logits, tgt_ids,
                margin_logits_neg, margin_neg_ids,
                pad_id=self.pad_id,
                margin=self.margin,
            )
        else:
            l_margin = None

        # L_mask_prior: per-slot Bernoulli KL teaching the PriorHead's mask to imitate the
        # posterior mask (§1.3). Computed only when targeting is ON (both mask logits given)
        # AND w_mask_prior>0; otherwise skipped ⟹ v3-bitwise total.
        if self.w_mask_prior > 0 and g_logits is not None and g_prior_logits is not None:
            l_mask_prior = mask_prior_kl(g_logits, g_prior_logits)
        else:
            l_mask_prior = None

        total = (
            self.w_token  * l_token
            + self.w_prior  * l_prior
            + self.w_sigreg * l_sigreg
            + self.w_pred   * l_pred
        )
        if l_nce is not None:
            total = total + self.w_nce * l_nce
        if l_margin is not None:
            total = total + self.w_margin * l_margin
        if l_mask_prior is not None:
            total = total + self.w_mask_prior * l_mask_prior

        components = {
            "loss":       total.item(),
            "L_token":    l_token.item(),
            "L_prior":    l_prior.item(),
            "L_sigreg":   l_sigreg.item(),
            "L_pred":     l_pred.item(),
            "L_nce":      (l_nce.item() if l_nce is not None else 0.0),
            "L_margin":   (l_margin.item() if l_margin is not None else 0.0),
            "L_mask_prior": (l_mask_prior.item() if l_mask_prior is not None else 0.0),
            "gumbel_tau": float(tau),
            # weights logged for interpretability
            "w_token":    float(self.w_token),
            "w_prior":    float(self.w_prior),
            "w_sigreg":   float(self.w_sigreg),
            "w_pred":     float(self.w_pred),
            "w_nce":      float(self.w_nce),
            "w_margin":   float(self.w_margin),
            "w_mask_prior": float(self.w_mask_prior),
        }
        return total, components


# Suffix-drop alias (R.1): the live aggregator is `JEPALoss`; `JEPALossV2` stays a
# back-compat alias so the frozen `from twm.jepa import JEPALossV2` entry point and
# scripts/train_jepa_v2.py keep working.
JEPALoss = JEPALossV2
