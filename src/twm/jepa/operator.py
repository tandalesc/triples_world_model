"""JEPA operator algebra — block-diagonal 2×2 rotation+scale.

Authoritative spec: research/jepa_operator_v1_design.md §1 (the centerpiece) and
§12 row T1. The empirical group-fit verdict (§1.1) mandates the rotation+scale
family `r·R(θ)` per 2×2 block; pure rotation is retained only as a frozen-`log_r`
ablation, and SO(n)-via-Cayley is an interface stub.

Per verb v, per 2×2 block b (block size b=2 EVERYWHERE in v1, §1.3):

    B_v = diag_blocks( r_b · R(θ_b) ),   R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]
    B_v^{-1} = diag_blocks( R(θ_b)ᵀ / r_b )      # STRUCTURAL — no consistency loss

`apply` is RoPE-style elementwise — no matrix is ever materialized (§1.3 GRAFT D2):

    (x', y') = ( r·(x cosθ − y sinθ),  r·(x sinθ + y cosθ) )

Architectural contracts (verbatim per §1.4):

  - 1-to-N is the encoder's job, NOT the operator's. `B_v k` is deterministic;
    multiple outcomes from one state are routed into different slots by the slot
    encoder's competitive assignment. The operator never solves 1-to-N.

  - Abelian commutativity is silent. `B_u B_v = B_v B_u` (diagonal-block scale +
    same-plane rotation commute). Harmless in v1 (no multiturn, no composition
    loss — anti-goals). Non-abelian expressiveness enters only via the v2
    state-dependent `velocity(k, v)`.

Numerics: all operator math runs in fp32 under `autocast(enabled=False)` (§1.3) —
cos/sin and exp are bf16-unstable at large magnitude, mirroring the VQ gotcha
(see vq_layer.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Operator


def _autocast_off(device_type: str):
    return torch.amp.autocast(device_type=device_type, enabled=False)


class RotationScaleOperator(Operator):
    """v1 mandated family C*^(d/2): per-verb angle θ and log-scale log_r per 2×2 block.

    Params:
        theta:  (V, dn//2)  block angles
        log_r:  (V, dn//2)  block log-scales (r = exp(log_r) > 0 always — no clamp,
                            gradient-stable; stored as log so positivity is structural)

    `apply` / `inverse_apply` are RoPE-style elementwise. The verb argument may be
    either hard indices `(B, M)` (long) or soft probabilities `(B, M, V)` (float) —
    the latter is the Gumbel-softmax soft-mix path (§3). For the block-linear form,

        E_v[ r_v · R(θ_v) ] = R-block built from ( Σ_v p_v · r_v cosθ_v,
                                                   Σ_v p_v · r_v sinθ_v )

    i.e. the expected operator is the block whose (a, b) = (E[r cosθ], E[r sinθ]).
    This is EXACT for the 2×2 block-linear operator: each output coordinate is a
    linear function of (r cosθ, r sinθ), so the expectation commutes with apply.
    (Documented per §1.6 soft-mix note.)
    """

    def __init__(self, n_verbs: int, d_noun: int, block: int = 2):
        super().__init__()
        if block != 2:
            # v1 keeps b=2 everywhere (§1.3 FIX): RoPE-style pairs, JS/INT8-trivial.
            raise ValueError(f"v1 operator requires block=2, got block={block}")
        if d_noun % 2 != 0:
            raise ValueError(f"d_noun must be even for 2×2 blocks, got {d_noun}")
        self._n_verbs = n_verbs
        self.d_noun = d_noun
        self.n_blocks = d_noun // 2
        self.block = block

        self.theta = nn.Parameter(torch.empty(n_verbs, self.n_blocks))
        self.log_r = nn.Parameter(torch.empty(n_verbs, self.n_blocks))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init (§1.3): θ ~ Uniform(−π/2, π/2) EXCLUDING |θ| < 0.1 (avoid identity);
        # log r ~ Normal(0, 0.1) (near 1.0, gradient discovers contraction).
        with torch.no_grad():
            theta = torch.empty_like(self.theta).uniform_(-math.pi / 2, math.pi / 2)
            # Push any |θ| < 0.1 out to ±0.1 (sign-preserving) to avoid the identity rotation.
            small = theta.abs() < 0.1
            theta[small] = 0.1 * torch.sign(theta[small].sign() + 0.5)  # 0 maps to +0.1
            self.theta.copy_(theta)
            self.log_r.normal_(0.0, 0.1)

    @property
    def n_verbs(self) -> int:
        return self._n_verbs

    # ---- block parameter gathering -------------------------------------------------

    def _gather_blocks(self, v: torch.Tensor):
        """Resolve verb arg -> per-position (r·cosθ, r·sinθ, r) of shape (..., n_blocks).

        v hard:  (B, M) long      -> gather rows of theta/log_r.
        v soft:  (B, M, V) float  -> expected block coefficients (block-linear, exact).

        Returns (a, b, r) where a = E[r cosθ], b = E[r sinθ], r = E[r] (r is the
        scale used for the inverse / for diagnostics; for the soft path r is the
        mixed scale, only consistent with (a, b) when verbs share a block, which is
        the documented expected-operator semantics).
        """
        theta = self.theta.float()   # (V, n_blocks)
        log_r = self.log_r.float()   # (V, n_blocks)
        r = torch.exp(log_r)         # (V, n_blocks)
        rcos = r * torch.cos(theta)  # (V, n_blocks)
        rsin = r * torch.sin(theta)  # (V, n_blocks)

        if not torch.is_floating_point(v):
            # Hard indices (B, M) -> (B, M, n_blocks)
            idx = v.long()
            a = F.embedding(idx, rcos)
            b = F.embedding(idx, rsin)
            rr = F.embedding(idx, r)
        else:
            # Soft probabilities (B, M, V) -> expected coefficients.
            p = v.float()                       # (B, M, V)
            a = p @ rcos                         # (B, M, n_blocks)
            b = p @ rsin
            rr = p @ r
        return a, b, rr

    def _apply_blocks(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Elementwise block-linear apply given per-position (a, b) = (rcosθ, rsinθ).

        x: (..., dn). a, b: (..., n_blocks). Output: (..., dn).
            (x', y') = (a·x − b·y,  b·x + a·y)
        """
        xpair = x.float().reshape(*x.shape[:-1], self.n_blocks, 2)
        xc, yc = xpair[..., 0], xpair[..., 1]  # (..., n_blocks)
        out_x = a * xc - b * yc
        out_y = b * xc + a * yc
        out = torch.stack([out_x, out_y], dim=-1).reshape(*x.shape)
        return out

    # ---- Operator interface --------------------------------------------------------

    def apply(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """a* = B_v k.  RoPE-style, no matrix materialized. fp32 under autocast.

        v hard (B,M) long OR soft (B,M,V) float (Gumbel soft-mix; expected operator).
        """
        with _autocast_off(k.device.type):
            a, b, _ = self._gather_blocks(v)
            out = self._apply_blocks(k, a, b)
        return out.to(k.dtype)

    def inverse_apply(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """k = B_v^{-1} a = diag_blocks(R(θ)ᵀ / r) a.  STRUCTURAL — exact undo.

        Equivalent to negating θ and negating log_r:
            R(−θ)/r has block coefficients (cosθ/r, −sinθ/r).
        """
        with _autocast_off(a.device.type):
            theta = self.theta.float()
            log_r = self.log_r.float()
            inv_r = torch.exp(-log_r)             # 1/r
            cos_over_r = inv_r * torch.cos(theta)  # cosθ / r
            sin_over_r = inv_r * torch.sin(theta)  # sinθ / r

            if not torch.is_floating_point(v):
                idx = v.long()
                acoef = F.embedding(idx, cos_over_r)
                # inverse rotation is R(−θ): coefficients (cosθ/r, −sinθ/r)
                bcoef = -F.embedding(idx, sin_over_r)
            else:
                p = v.float()
                acoef = p @ cos_over_r
                bcoef = -(p @ sin_over_r)
            out = self._apply_blocks(a, acoef, bcoef)
        return out.to(a.dtype)

    def velocity(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Generator action for the T-step seam (§1.6).  v1: static block generator.

        v1 returns the single-step displacement `B_v k − k` so that one Euler step
        of size dt=1 reproduces `apply` exactly. v2 replaces this with a
        state-dependent MLP -> skew generator. v1 keeps this dormant but consistent.
        """
        with _autocast_off(k.device.type):
            out = self.apply(k, v).float() - k.float()
        return out.to(k.dtype)

    def _integrate_step(self, x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
        """Single integration step. At dt=1 (T=1) this is the exact exponential map.

        v1 static generator: x + dt·(B_v x − x). With T=1, dt=1 -> B_v x == apply().
        """
        with _autocast_off(x.device.type):
            disp = self.apply(x, v).float() - x.float()
            out = x.float() + dt * disp
        return out.to(x.dtype)

    def integrate(self, k: torch.Tensor, v: torch.Tensor, T: int = 1) -> torch.Tensor:
        """T-step integration seam (§1.6). T hard-set to 1 in v1.

        At T=1 this is bitwise-close to `apply` (single exp map). The loop and the
        per-step generator hook are present and dormant for v2.
        """
        if T == 1:
            # Fast path: identical to apply() (single exp map), no residual arithmetic
            # so T=1 is bitwise-close to apply.
            return self.apply(k, v)
        x = k
        for _ in range(T):
            x = self._integrate_step(x, v, dt=1.0 / T)
        return x

    @torch.no_grad()
    def structural_sanity(self, v: int) -> dict:
        """Runtime invariant check for verb v -> {"bbT_err", "inv_err"}. Diagnostic, NOT a loss.

        bbT_err: ‖B_v B_v^{-1} − I‖_F over a random probe (should be ~0).
        inv_err: ‖B_v^{-1} B_v k − k‖ round-trip error (should be ~0).
        """
        device = self.theta.device
        idx = torch.full((1, 1), int(v), dtype=torch.long, device=device)
        k = torch.randn(1, 1, self.d_noun, device=device)

        # inv_err: round-trip k -> B_v k -> B_v^{-1}(B_v k) should recover k.
        a = self.apply(k, idx)
        k_rt = self.inverse_apply(a, idx)
        inv_err = (k_rt - k).norm().item()

        # bbT_err: ‖B B^{-1} − I‖_F via materializing the action on the identity basis.
        eye = torch.eye(self.d_noun, device=device).unsqueeze(0)  # (1, dn, dn) as M=dn slots
        eye_slots = eye.reshape(1, self.d_noun, self.d_noun)
        idx_full = torch.full((1, self.d_noun), int(v), dtype=torch.long, device=device)
        # B^{-1} applied to columns of I gives B^{-1}; then B applied gives B B^{-1}.
        b_inv = self.inverse_apply(eye_slots, idx_full)          # (1, dn, dn)
        bbinv = self.apply(b_inv, idx_full)                      # (1, dn, dn) == B B^{-1}
        bbT_err = (bbinv - eye).norm().item()

        return {"bbT_err": bbT_err, "inv_err": inv_err}

    @torch.no_grad()
    def bake(self) -> dict:
        """Export-ready per-verb (cos, sin, r) per block for JS / INT8 (§8).

        Returns float32 tensors of shape (V, n_blocks):
            cos = cosθ, sin = sinθ, r = exp(log_r).
        The JS apply is (x', y') = (r·(x·cos − y·sin), r·(x·sin + y·cos)).
        """
        theta = self.theta.detach().float()
        log_r = self.log_r.detach().float()
        return {
            "cos": torch.cos(theta).cpu(),
            "sin": torch.sin(theta).cpu(),
            "r": torch.exp(log_r).cpu(),
        }


class RotationOperator(RotationScaleOperator):
    """Pure-rotation ablation (U(1)^(d/2)): log_r frozen at 0 so r ≡ 1 (norm-preserving).

    Config ablation only — the empirical fit (§1.1) shows pure rotation is no better
    than identity on GLUCOSE, so this exists to reproduce that negative result and as
    the frozen-`log_r` config flip referenced in O3.
    """

    def __init__(self, n_verbs: int, d_noun: int, block: int = 2):
        super().__init__(n_verbs, d_noun, block)
        with torch.no_grad():
            self.log_r.zero_()
        self.log_r.requires_grad_(False)


class SOnCayleyOperator(Operator):
    """SO(n)-via-Cayley operator — interface stub (anti-goal §13). Not built in v1."""

    def __init__(self, n_verbs: int, d_noun: int, block: int = 2):
        super().__init__()
        self._n_verbs = n_verbs
        self.d_noun = d_noun

    @property
    def n_verbs(self) -> int:
        return self._n_verbs

    def apply(self, k, v):
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")

    def inverse_apply(self, a, v):
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")

    def velocity(self, k, v):
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")

    def integrate(self, k, v, T: int = 1):
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")

    def structural_sanity(self, v: int) -> dict:
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")

    def bake(self) -> dict:
        raise NotImplementedError("SOnCayleyOperator is a v1 interface stub")
