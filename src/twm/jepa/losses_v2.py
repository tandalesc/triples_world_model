"""JEPA v2 losses (T-task D). Spec §5.

Combined loss:
    L = w_token·L_token + w_prior·L_prior + w_sigreg·L_sigreg + w_pred·L_pred

Differences from v1 JEPALoss:
  - L_token (CE, weight 1.0): cross-entropy of text_{t+1} tokens given operator-
    transformed slots a*. This is THE primary loss — it forces the discrete latent
    action v to carry information about the causal step. Token CE is the grounding
    loss; v1's anti-goal "no token decoder" is explicitly revoked (design doc §0).
  - L_prior (KL, weight 0.1): KL(stopgrad(softmax(q/τ)) ‖ softmax(p)) — distills the
    posterior action distribution into the prior (which runs on state_t only). Enables
    autonomous rollout without the posterior at inference.
  - L_sigreg (0.05): REUSED unchanged from v1 losses.py via import. Keeps the noun
    space isotropic; standardize, never L2.
  - L_pred (0.25, optional): REUSED MSE(zhat, z.detach()) EMA aux objective. Keeps
    the JEPA latent objective alive as a regularizer on a*. Setting w_pred=0 is a
    supported ablation (pure token-grounding).
  - L_div: DELETED. Verb informativeness comes from necessity (decoder needs v's bits)
    not a gameable regularizer. Codebook usage is a diagnostic only (diagnostics_v2).

Leakage constraint (§6): the decoder must see ONLY {a*_i} as memory — never raw
text_{t+1} encodings or the posterior features. token_ce() is called with logits
produced by a decoder whose memory is a*, never with any shortcut conditioning.
This module enforces the constraint at the signature level: token_ce and prior_kl
have no parameter that could carry text_{t+1} activations directly.

`anneal_tau` and `sigreg_loss` are imported from v1 losses.py — no duplication, no
edit to v1. The v2 helpers added here are `token_ce`, `prior_kl`, and `JEPALossV2`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import unchanged v1 utilities — do not duplicate, do not edit v1 losses.py.
from .losses import sigreg_loss, anneal_tau  # noqa: F401  (re-exported for callers)


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
    as v1 JEPALoss._operator_ref (see v1 losses.py header comment).

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
        # causing duplicate AdamW entries — see v1 losses.py for the rationale.)
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
