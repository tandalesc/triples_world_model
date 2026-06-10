"""GatedMLPTransition — the black-box transition baseline (engram-wm, v3 §4).

Authoritative spec: research/jepa_v3_design.md §4. This is the engram-wm program's
black-box transition behind the SAME `apply(k, v)`-style interface as the structured
RotationScaleOperator, used to isolate "what does the polar/rotation structure buy over
a generic learned transition?". It is a drop-in `Operator` subclass selectable via
`operator_group="gated_mlp"`, and trains through the IDENTICAL pipeline (forward, unroll,
decoder, InfoNCE, losses) — only `inverse_apply`/`bake` raise (the baseline is
deliberately NON-REVERSIBLE; that is the point of contrasting it with the invertible
operator).

Architecture (per verb v, on noun k ∈ ℝ^dn — §4.2):

    e_v   = verb_emb[v]                              # (..., d_e)
    h     = GELU( W1([k ; e_v]) )                    # (..., d_h)
    a*    = k + sigmoid(gate[v]) ⊙ W2(h)             # residual + per-verb gate

Params (nano dn=32, V=8, d_e=4, d_h=8): verb_emb 32 + W1 288 + W2 256 + gate 256 = **832**
transition params, vs the operator's op-codebook(256)+polar-H(256)=512 ⟹ **1.6×**, within
the ~2× requirement (§4.2). The gate is the operator-style per-verb `(V, dn)` table (a
full `Linear(dn+d_e, dn)` gate would blow the budget).

Numerics: `apply` runs in fp32 under `autocast(enabled=False)`, mirroring the operator/VQ
gotcha (GELU/sigmoid are bf16-unstable at large magnitude).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Operator

_LOG = logging.getLogger(__name__)


def _autocast_off(device_type: str):
    return torch.amp.autocast(device_type=device_type, enabled=False)


class GatedMLPTransition(Operator):
    """Black-box verb-conditioned gated-MLP transition (engram-wm baseline, v3 §4).

    A DROP-IN `Operator` subclass behind the SAME `apply(k, v)` interface as the
    structured RotationScaleOperator, so `model._apply_action` and the unroll loop are
    byte-identical across operator families. Selected via `operator_group="gated_mlp"`.

    `apply` accepts a `theta_offset=None` kwarg for SIGNATURE COMPATIBILITY with the
    polar-conditioning call site (`model._apply_action` passes `theta_offset=...` for
    every operator family) but IGNORES it — the gated MLP has no phase/modulus split.
    Documented and asserted by a test (passing a nonzero offset changes nothing); this
    keeps `model._apply_action` branch-free across operator families.

    NO INVERSE (the whole point of the contrast): `inverse_apply` and `bake` RAISE.
    `structural_sanity` returns a NaN-filled dict (invertibility metrics undefined for a
    black-box). `velocity` = `apply(k,v) - k` for the dormant T-step seam parity;
    `integrate` is defined only at T==1 (== apply) and raises otherwise.

    Near-identity init (matching the operator's small-θ / zero-init-H spirit, §4.2): the
    gate `Parameter(V, dn)` is init to a LARGE NEGATIVE constant (-4.0) so
    `sigmoid(-4) ≈ 0.018` ⟹ `a* ≈ k` at step 0. A zero-init gate gives `sigmoid(0)=0.5`,
    a half-residual, NOT identity — so the negative-bias init is deliberate.
    """

    GATE_INIT = -4.0  # sigmoid(-4) ≈ 0.018 ⟹ near-identity transition at init (§4.2)
    _warned_norm_budget = False  # class-level: one warning per process, not per step

    def __init__(
        self,
        n_verbs: int,
        d_noun: int,
        block: int = 2,
        d_e: int = 4,
        d_h: int = 8,
    ):
        super().__init__()
        self._n_verbs = n_verbs
        self.d_noun = d_noun
        self.d_e = d_e
        self.d_h = d_h

        # W1/W2 are BIAS-FREE so the param budget matches the §4.2 arithmetic exactly
        # (verb_emb 32 + W1 288 + W2 256 + gate 256 = 832); biases would add d_h+dn and
        # break the documented 832 figure. The per-verb gate is the only additive bias-like
        # term, and it is the operator-style (V, dn) table, not a generic affine.
        self.verb_emb = nn.Embedding(n_verbs, d_e)                       # (V, d_e)
        self.W1 = nn.Linear(d_noun + d_e, d_h, bias=False)              # (dn+d_e) -> d_h
        self.W2 = nn.Linear(d_h, d_noun, bias=False)                    # d_h -> dn
        self.gate = nn.Parameter(torch.empty(n_verbs, d_noun))          # per-verb gate table
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.verb_emb.weight, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
            # near-identity at init: large-negative gate ⟹ sigmoid(gate) ≈ 0 ⟹ a* ≈ k.
            self.gate.fill_(self.GATE_INIT)

    @property
    def n_verbs(self) -> int:
        return self._n_verbs

    # ---- verb resolution -----------------------------------------------------------

    def _gather_verb(self, v: torch.Tensor):
        """Resolve verb arg -> (e_v (..., d_e), gate_v (..., dn)) per position.

        v hard: (B, M) long      -> embedding lookup (the trained path; v2/v3 train hard).
        v soft: (B, M, V) float  -> expected verb embedding `p @ verb_emb` and expected
                gate `p @ gate` (keeps the ST gradient flowing into v_logits; documented
                as the soft-mix path, mirroring the operator's expected-coefficient form).
        """
        if not torch.is_floating_point(v):
            idx = v.long()
            e = self.verb_emb(idx)                          # (..., d_e)
            g = F.embedding(idx, self.gate)                 # (..., dn)
        else:
            p = v.float()                                   # (..., V)
            e = p @ self.verb_emb.weight                    # (..., d_e)
            g = p @ self.gate                               # (..., dn)
        return e, g

    # ---- Operator interface --------------------------------------------------------

    def apply(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        theta_offset: torch.Tensor | None = None,
        *,
        norm_budget: bool = False,
    ) -> torch.Tensor:
        """a* = k + sigmoid(gate[v]) ⊙ W2(GELU(W1([k ; e_v]))).

        v hard (B,M) long OR soft (B,M,V) float. `theta_offset` is ACCEPTED for
        signature parity with the polar call site and IGNORED (the gated MLP has no
        phase split — §4.1). fp32 under autocast-off (numerics gotcha).

        norm_budget (entity §1.4): the black-box has no modulus/phase split and no
        scale notion, so the flag is ACCEPTED-AND-IGNORED (signature parity with the
        structured operator, exactly like `theta_offset`). It logs a ONE-TIME warning
        (guarded by the class-level `_warned_norm_budget`) and, to keep
        `model._apply_action` branch-free across operator families, returns the SAME
        `(a, scale_delta)` arity as the structured operator with `scale_delta =
        zeros(B, M)` (log-scale 0 ⟹ scale 1.0 — "this transition exposes no scale").
        """
        del theta_offset  # accepted-and-ignored (signature parity with the polar call site)
        if norm_budget and not type(self)._warned_norm_budget:
            type(self)._warned_norm_budget = True
            _LOG.warning(
                "GatedMLPTransition received norm_budget=True but has no modulus/phase "
                "split or scale notion (entity §1.4): the flag is ignored and the "
                "tracked scale stays 0 (the black-box exposes no irreversibility scalar)."
            )
        with _autocast_off(k.device.type):
            kf = k.float()
            e, g = self._gather_verb(v)                     # (...,d_e), (...,dn)
            e = e.float()
            inp = torch.cat([kf, e], dim=-1)                # (..., dn+d_e)
            h = F.gelu(self.W1(inp))                        # (..., d_h)
            delta = self.W2(h)                              # (..., dn)
            out = kf + torch.sigmoid(g.float()) * delta     # residual + per-verb gate
        out = out.to(k.dtype)
        if norm_budget:
            B, M = k.shape[0], k.shape[1]
            scale_delta = torch.zeros(B, M, dtype=k.dtype, device=k.device)
            return out, scale_delta                         # (a, zeros) — branch-free arity
        return out

    def inverse_apply(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        theta_offset: torch.Tensor | None = None,
        *,
        norm_budget: bool = False,
        scale_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NO INVERSE — black-box transition (engram-wm baseline). RAISES.

        The structured operator's whole pitch is a STRUCTURAL inverse; the black-box
        baseline deliberately has none (that is the point of the contrast).
        `model.undo_latent` / the pet-demo path is not used in v3 training, so this never
        executes in the train loop.

        `norm_budget`/`scale_delta` are ACCEPTED for signature parity with the structured
        operator but change NOTHING — there is no inverse regardless (entity §1.4). The
        retraction probe (§4) relies on this raise: the structured operator can retract an
        event, the black-box cannot, and that asymmetry IS the experiment.
        """
        raise NotImplementedError(
            "GatedMLPTransition is a black-box transition with no structural inverse "
            "(engram-wm baseline)"
        )

    def velocity(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Single-step displacement `apply(k,v) - k` (T-step seam parity, §4.1)."""
        with _autocast_off(k.device.type):
            out = self.apply(k, v).float() - k.float()
        return out.to(k.dtype)

    def integrate(self, k: torch.Tensor, v: torch.Tensor, T: int = 1) -> torch.Tensor:
        """T-step seam. Black-box has no integrator; only T==1 (== apply) is defined."""
        if T != 1:
            raise NotImplementedError(
                "GatedMLPTransition has no multi-step integrator; T must be 1 "
                "(black-box baseline)"
            )
        return self.apply(k, v)

    @torch.no_grad()
    def structural_sanity(self, v: int) -> dict:
        """No inverse ⟹ invertibility metrics are UNDEFINED. Returns NaN-filled so the
        diagnostics harness (which may call it) does not crash (§4.1)."""
        return {"bbT_err": float("nan"), "inv_err": float("nan")}

    @torch.no_grad()
    def bake(self) -> dict:
        """NOT JS-exportable (black-box MLP, no closed-form per-verb tables). RAISES."""
        raise NotImplementedError(
            "GatedMLPTransition is a black-box transition and is not JS-exportable "
            "(no per-verb cos/sin/r tables to bake)"
        )
