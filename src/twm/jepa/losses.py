"""JEPA losses (the live v2 path). Spec: research/jepa_v2_latent_actions.md §5.

Combined loss:
    L = w_token·L_token + w_prior·L_prior + w_sigreg·L_sigreg + w_pred·L_pred

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
) -> torch.Tensor:
    """Cross-entropy loss for autoregressive token prediction.

    ARDecoder.forward(a*, tgt_ids, tgt_pad) returns logits (B, T, V_vocab) aligned to
    tgt_ids positions 0..T-1: internally it prepends a learned BOS embedding and feeds
    [bos, tgt[:-1]] to the decoder, so logits[b, t] predicts tgt_ids[b, t].

    Loss = CE(logits.reshape(-1, V), tgt_ids.reshape(-1), ignore_index=pad_id).
    Pad positions are excluded from the denominator (ignore_index). EOS (id 4) is
    included as a real predicted token (not masked), so the decoder learns to stop.

    Args:
        logits:  (B, T, V_vocab) — decoder logits (no manual shift needed; the shift
                 is baked into ARDecoder's [bos]+tgt[:-1] construction).
        tgt_ids: (B, T) target token ids (EOS appended before pad, as §7 requires).
        pad_id:  id of the padding token (0 in jepa_bpe_512.json).

    Returns:
        scalar mean CE over non-pad positions.
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_ids.reshape(B * T),
        ignore_index=pad_id,
    )


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
        zhat: torch.Tensor,          # (B, dn) predicted latent (for L_pred aux)
        z_target: torch.Tensor,      # (B, dn) EMA target latent, stop-grad
        tau: float = 1.0,            # current Gumbel temperature (for L_prior KL sharpness)
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss and per-term components.

        Returns:
            (total, components) where components is a flat dict of scalar floats for
            logging (plus 'loss' = total.item() for convenience).
        """
        # L_token: primary grounding loss — CE of text_{t+1} tokens given a* memory.
        l_token = token_ce(logits, tgt_ids, pad_id=self.pad_id)

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

        total = (
            self.w_token  * l_token
            + self.w_prior  * l_prior
            + self.w_sigreg * l_sigreg
            + self.w_pred   * l_pred
        )

        components = {
            "loss":       total.item(),
            "L_token":    l_token.item(),
            "L_prior":    l_prior.item(),
            "L_sigreg":   l_sigreg.item(),
            "L_pred":     l_pred.item(),
            "gumbel_tau": float(tau),
            # weights logged for interpretability
            "w_token":    float(self.w_token),
            "w_prior":    float(self.w_prior),
            "w_sigreg":   float(self.w_sigreg),
            "w_pred":     float(self.w_pred),
        }
        return total, components


# Suffix-drop alias (R.1): the live aggregator is `JEPALoss`; `JEPALossV2` stays a
# back-compat alias so the frozen `from twm.jepa import JEPALossV2` entry point and
# scripts/train_jepa_v2.py keep working.
JEPALoss = JEPALossV2
