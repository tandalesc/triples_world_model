"""JEPA losses (T3). Spec §3 + §12 row T3.

Three irreducible losses, nothing else (no consistency / invertibility / VAE /
spectral / CKA / orthogonality / composition / token-recon losses — those are
regressions if they ever appear, spec §3 "explicitly absent").

    L = w_pred·L_pred + w_sigreg·L_sigreg + w_div·L_div  (+ w_scale_reg·‖log r‖₂)

- L_pred:   MSE(zhat, z_target.detach()).
- L_sigreg: sliced isotropic-Gaussian goodness-of-fit on STANDARDIZED nouns.
            The decisive shared flaw the judges caught: do NOT L2-project nouns
            to the unit sphere before the GoF. On the sphere every random 1D
            projection has std ≈ 1/√dn, so the Epps-Pulley test (calibrated to
            N(0,1)) fires constantly and supplies NO gradient. We center + scale
            by BATCH statistics instead (zero-mean / unit-var per dim).
- L_div:    verb non-triviality over the Gumbel-softmax assignment: usage entropy
            (penalize unused codes) + spread (push |θ|, |log r| away from identity
            and verbs away from each other).

VerbHead gradient fix (spec §3): the soft Gumbel-softmax mix Σ_v softmax_v·B_v k
is what carries L_pred into the verb logits. That mixing happens in the model
forward (it needs the operator); this module owns the Gumbel-softmax *sampling*
(`gumbel_softmax_sample`) and the temperature anneal schedule (`anneal_tau`).
At eval/export, hard argmax (straight-through) is used.

Operator math (the codebook θ / log r read for L_div spread) is touched only as
detached/penalty terms here; the operator's own fp32-autocast apply lives in
operator.py (T1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gumbel-softmax verb sampling + temperature anneal (spec §3 VerbHead fix)
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

    Spec §3: 2.0 -> 0.5 over the first 30% of steps.
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
                soft sample. At eval we still use the soft sample for the backward
                path so the anneal-tail STE matches training, per spec.

    Returns:
        (..., V) soft (or straight-through-hard) assignment that sums to 1 over V.
    """
    # Sample Gumbel(0,1) noise. -log(-log(U)) with U ~ Uniform(0,1).
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    y = F.softmax((logits + gumbel) / tau, dim=-1)
    if not hard:
        return y
    # Straight-through: hard one-hot forward, soft gradient backward.
    idx = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    return y_hard + (y - y.detach())


# ---------------------------------------------------------------------------
# SIGReg: sliced isotropic-Gaussian goodness-of-fit (spec §3 L_sigreg)
# ---------------------------------------------------------------------------

def _standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Center + scale by BATCH statistics (zero-mean / unit-var per dim).

    NOT per-vector L2-norm. Projecting nouns onto the unit sphere makes every 1D
    projection have std ≈ 1/√dn, which saturates the N(0,1)-calibrated GoF and
    kills its gradient (spec §3, the decisive shared flaw). Standardizing keeps
    the test on a scale where it has a live gradient.
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

    The empirical characteristic function is
        φ_emp(t) = (1/N) Σ_n exp(i t x_n)
    and for N(0,1) the target is φ_0(t) = exp(-t²/2) (real, imag 0). We integrate
    the squared modulus of the difference over a knot grid on [0, knot_max] with
    the trapezoid rule. Zero iff the standardized samples are unit Gaussian.

    Args:
        proj:    (N, S) projected samples — N points along S slices.
        n_knots: number of t-grid knots.
        knot_max: upper limit of the t integration.

    Returns:
        scalar mean discrepancy over slices.
    """
    n = proj.shape[0]
    # t-grid (n_knots,) on [0, knot_max]; trapezoid weights.
    t = torch.linspace(0.0, knot_max, n_knots, device=proj.device, dtype=proj.dtype)
    dt = knot_max / (n_knots - 1)
    w = torch.full((n_knots,), dt, device=proj.device, dtype=proj.dtype)
    w[0] *= 0.5
    w[-1] *= 0.5  # trapezoid endpoint weights

    # arg[n, s, k] = t_k * x_{n,s}
    arg = proj.unsqueeze(-1) * t.view(1, 1, -1)        # (N, S, K)
    re = torch.cos(arg).mean(dim=0)                     # (S, K)  Re φ_emp
    im = torch.sin(arg).mean(dim=0)                     # (S, K)  Im φ_emp
    target_re = torch.exp(-0.5 * t * t).view(1, -1)     # (1, K)  Re φ_0 (real)

    # |φ_emp - φ_0|² = (Re_emp - Re_0)² + (Im_emp - 0)²
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
        standardize: MUST be True (spec). False is a documented negative path —
            on un-standardized / sphere-projected inputs the GoF saturates and the
            gradient vanishes; kept only so tests can demonstrate the failure.

    Returns:
        scalar SIGReg loss (≈ 0 when nouns are isotropic unit Gaussian after
        standardization, >> 0 under rank collapse / anisotropy).
    """
    x = nouns.reshape(-1, nouns.shape[-1])
    dn = x.shape[-1]
    if standardize:
        x = _standardize(x)

    # Random unit slicing directions (dn, S). Match x's device/dtype.
    dirs = torch.randn(dn, n_slices, device=x.device, dtype=x.dtype, generator=generator)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    proj = x @ dirs                                     # (N, S)
    return _epps_pulley_gof(proj, n_knots=n_knots, knot_max=knot_max)


# ---------------------------------------------------------------------------
# L_div: verb non-triviality (spec §3 L_div)
# ---------------------------------------------------------------------------

def usage_entropy(
    assign: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Usage entropy over the batch verb assignment, returned as a *penalty*
    (negative entropy) so minimizing it MAXIMIZES code usage.

    Args:
        assign: (..., V) soft (Gumbel-softmax) per-slot assignment over verbs.

    Returns:
        scalar = -H(p̄), where p̄ is the batch-mean assignment over verbs. This is
        minimized (most negative) when p̄ is uniform — i.e. all codes used. A
        single-verb routing makes p̄ a near one-hot => H≈0 => penalty≈0 (the worst,
        least-negative value), so the term penalizes verb collapse.
    """
    p = assign.reshape(-1, assign.shape[-1]).mean(dim=0)  # (V,) batch-mean usage
    p = p / (p.sum() + eps)
    h = -(p * (p + eps).log()).sum()
    return -h


def spread_penalty(
    theta: torch.Tensor,
    log_r: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize operators collapsing to identity or onto each other.

    Args:
        theta: (V, dn//2) per-verb per-block angles.
        log_r: (V, dn//2) per-verb per-block log-scales.

    Returns:
        scalar = identity-proximity term + pairwise-proximity term.

        identity:  mean exp(-(|θ|² + |log r|²))  — large when a verb sits at the
                   identity (θ→0, log r→0), small when it is a non-trivial operator.
        pairwise:  mean over verb pairs of exp(-‖(θ_u,log r_u)-(θ_v,log r_v)‖²) —
                   large when two verbs coincide, small when they spread apart.
    """
    v = theta.shape[0]
    # Per-verb feature = concat(θ, log r) flattened over blocks. (V, dn)
    feat = torch.cat([theta, log_r], dim=-1).reshape(v, -1)

    # Identity proximity: distance of each verb from the origin (identity op).
    id_dist2 = feat.pow(2).sum(dim=-1)                  # (V,)
    identity_term = torch.exp(-id_dist2).mean()

    # Pairwise proximity over distinct verb pairs.
    if v > 1:
        # ‖f_u - f_v‖² via the standard expansion. (V, V)
        sq = feat.pow(2).sum(dim=-1, keepdim=True)
        pair_dist2 = sq + sq.T - 2.0 * (feat @ feat.T)
        pair_dist2 = pair_dist2.clamp_min(0.0)
        prox = torch.exp(-pair_dist2)
        # Exclude the diagonal (a verb vs itself).
        off_diag = prox - torch.eye(v, device=feat.device, dtype=feat.dtype)
        pairwise_term = off_diag.sum() / (v * (v - 1))
    else:
        pairwise_term = feat.new_zeros(())

    return identity_term + pairwise_term


# ---------------------------------------------------------------------------
# JEPALoss
# ---------------------------------------------------------------------------

class JEPALoss(nn.Module):
    """Combined JEPA training loss (spec §3).

    Holds a reference to the operator codebook so L_div can read the per-verb
    angle / scale params (θ, log r) for the spread penalty. The operator is NOT
    owned here — it is constructed by the model (T1/T5) and passed in.

    `forward` matches the frozen JEPALossProtocol signature: it receives the
    already-mixed `zhat` (the soft Gumbel mix over verbs happens in the model so
    L_pred flows into the verb logits) and the stop-grad `z_target`.
    """

    def __init__(
        self,
        operator: nn.Module | None = None,
        w_pred: float = 1.0,
        w_sigreg: float = 0.05,
        w_div: float = 0.1,
        w_scale_reg: float = 0.0,
        n_slices: int = 256,
        n_knots: int = 17,
        knot_max: float = 3.0,
        standardize: bool = True,
    ):
        super().__init__()
        # Store the operator WITHOUT registering it as a submodule. A plain
        # `self.operator = operator` would trigger nn.Module.__setattr__ and
        # auto-register the operator's theta/log_r in `JEPALoss.parameters()`.
        # Those params already live in the model's online_parameters(); double-
        # registration puts them in AdamW twice (2x effective LR + a duplicate-
        # parameter error). We only need READ access to theta/log_r for the L_div
        # spread penalty, so stash the reference in a 1-element list (object.
        # __setattr__ bypasses nn.Module's submodule capture) and unwrap on read.
        object.__setattr__(self, "_operator_ref", [operator])
        self.w_pred = w_pred
        self.w_sigreg = w_sigreg
        self.w_div = w_div
        self.w_scale_reg = w_scale_reg
        self.n_slices = n_slices
        self.n_knots = n_knots
        self.knot_max = knot_max
        self.standardize = standardize

    def _operator_params(self):
        """Fetch (theta, log_r) from the operator codebook for L_div, or None.

        RotationScaleOperator exposes `theta` (V, dn//2) and `log_r` (V, dn//2).
        """
        op = self._operator_ref[0]
        if op is None:
            return None
        theta = getattr(op, "theta", None)
        log_r = getattr(op, "log_r", None)
        if theta is None or log_r is None:
            return None
        return theta, log_r

    def forward(
        self,
        k: torch.Tensor,            # (B, M, dn) standardized nouns
        verb_logits: torch.Tensor,  # (B, M, V)
        zhat: torch.Tensor,         # (B, dn) predicted next-state pool from a*
        z_target: torch.Tensor,     # (B, dn) EMA-encoder(next_text) pool, stop-grad
        gumbel_tau: float,          # current annealed Gumbel temperature
        hard: bool = False,         # eval/export: hard argmax (straight-through)
    ) -> tuple[torch.Tensor, dict]:
        # --- L_pred: MSE to the stop-grad EMA target ------------------------
        l_pred = F.mse_loss(zhat, z_target.detach())

        # --- L_sigreg: standardized sliced-Gaussian GoF on nouns ------------
        l_sigreg = sigreg_loss(
            k,
            n_slices=self.n_slices,
            n_knots=self.n_knots,
            knot_max=self.knot_max,
            standardize=self.standardize,
        )

        # --- L_div: usage entropy + operator spread -------------------------
        assign = gumbel_softmax_sample(verb_logits, tau=gumbel_tau, hard=hard)
        l_entropy = usage_entropy(assign)

        params = self._operator_params()
        if params is not None:
            theta, log_r = params
            l_spread = spread_penalty(theta, log_r)
            scale_reg = log_r.norm()
        else:
            l_spread = verb_logits.new_zeros(())
            scale_reg = verb_logits.new_zeros(())
        l_div = l_entropy + l_spread

        total = (
            self.w_pred * l_pred
            + self.w_sigreg * l_sigreg
            + self.w_div * l_div
            + self.w_scale_reg * scale_reg
        )

        components = {
            "loss": total.item(),
            "L_pred": l_pred.item(),
            "L_sigreg": l_sigreg.item(),
            "L_div": l_div.item(),
            "L_entropy": l_entropy.item(),
            "L_spread": l_spread.item() if torch.is_tensor(l_spread) else float(l_spread),
            "scale_reg": scale_reg.item() if torch.is_tensor(scale_reg) else float(scale_reg),
            "gumbel_tau": float(gumbel_tau),
        }
        return total, components
