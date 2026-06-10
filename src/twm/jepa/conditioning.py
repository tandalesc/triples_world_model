"""JEPA v2.1 polar conditioning — the "adjective" replacement (design §3, §7).

Authoritative spec: research/jepa_v21_polar.md §3 (H conditioning map) and §7
(optional discrete "kind" head). Builds on the polar reading of the operator's
2×2 blocks: a noun `k ∈ ℝ^dn` is `nb = dn/2` complex coordinates `z_b = x_b + i·y_b`.
The MODULUS profile `m_b = |z_b|` is the object's persistent identity; the PHASE
profile `arg(z_b)` is its mutable state.

Two pieces live here:

  - `PolarConditioner` (the H map): a single shared `Linear(nb -> nb, bias=False)`,
    ZERO-INITIALIZED, that maps a slot's modulus profile to a per-block phase offset
    `θ_offset = H(m_i)`. Zero-init ⟹ `H(m) = 0` at step 0 ⟹ v2.1 == v2.0 exactly
    (the behavior-preservation guarantee, design §0/§11). Conditions PHASE only;
    `log_r` (irreversibility) stays a global verb property (design §3.2).

  - `KindHead` (optional, default OFF): an argmax/VQ readout over the modulus profile
    assigning each slot a discrete "kind" id. Diagnostic/demo ONLY — NEVER load-bearing
    for routing (does not gate H, does not select a verb, does not touch the operator;
    design §7). It is a microscope, not a gear.

Modulus helper `block_modulus(k)` is shared by the forward path, the diagnostics
(phase-uniformity / modulus-eff-rank, design §5.2), and the kind head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block_modulus(k: torch.Tensor) -> torch.Tensor:
    """Per-block complex modulus profile m_b = |z_b| = sqrt(x_b² + y_b²).

    Reads `k (..., dn)` as `nb = dn/2` complex coordinates `(x_b, y_b)` (the same 2×2
    block layout `operator._apply_blocks` uses) and returns `(..., nb)` moduli.

    NOTE on gradient (design §3.1): the forward path keeps the gradient on `k` so the
    encoder learns to shape the modulus profile as a useful conditioning signal. Only
    the §8.1 identity-persistence *diagnostic* detaches; never the forward.
    """
    dn = k.shape[-1]
    if dn % 2 != 0:
        raise ValueError(f"block_modulus requires even last dim (2×2 blocks), got {dn}")
    pair = k.reshape(*k.shape[:-1], dn // 2, 2)  # (..., nb, 2)
    return pair.pow(2).sum(dim=-1).sqrt()        # (..., nb)


class PolarConditioner(nn.Module):
    """The H map: modulus profile -> per-block phase offset (design §3).

        θ_offset = H(m_i),   m_i = |z_i| ∈ ℝ^nb,   H = Linear(nb -> nb, bias=False)

    ZERO-INITIALIZED (`weight.zero_()`): at init `H(m) = 0` for any input, so the
    conditioned operator angle `θ_eff = θ_v + H(m)` equals `θ_v` ⟹ v2.1 == v2.0 at
    step 0. `bias=False` is mandatory for this guarantee (a nonzero bias would offset
    the angle even at a zero modulus; design §6).

    Shared across verbs (one `(nb, nb)` matrix, `nb²` params): the adjective-style
    modulation is a property of the *space*, not of individual verbs (design §3.1).
    """

    def __init__(self, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.H = nn.Linear(n_blocks, n_blocks, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # ZERO-INIT (design §3.1) — the v2.1==v2.0-at-init guarantee. Do NOT change to a
        # nonzero init without also re-deriving the §11 behavior-preservation gate.
        with torch.no_grad():
            self.H.weight.zero_()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """k (B, M, dn) -> theta_offset (B, M, nb).

        Computes the modulus profile of `k` (gradient kept, design §3.1) and maps it
        through the zero-init H. Returns the per-block phase offset to add to the verb
        angle inside `operator.apply(..., theta_offset=...)`.
        """
        m = block_modulus(k)        # (B, M, nb), gradient kept
        return self.H(m)            # (B, M, nb), zero at init


class KindHead(nn.Module):
    """Optional discrete "kind" readout over the modulus profile (design §7).

    A codebook `(K, nb)` of K modulus prototypes; each slot's kind id is the nearest
    prototype to its modulus profile:

        kind_id(k_i) = argmin_j || m(k_i) − kind_codebook[j] ||²

    STRICT CONSTRAINTS (design §7):
      - NEVER load-bearing for routing — does not gate H, select a verb, or touch the
        operator. Read OFF the modulus profile purely to LABEL slots in diagnostics /
        the demo.
      - Codebook init `normal(0, 1)` (repo VQ gotcha: match expected encoder
        statistics, not the tiny VQ-VAE uniform init that snap-collapses).
      - The optional commitment loss (`commitment_loss`) is computed in fp32 and only
        used when the kind head is enabled with a nonzero λ_vq (default off / untrained).

    This exists to make the "modulus = identity" claim legible to a human, not to make
    the model work.
    """

    def __init__(self, n_blocks: int, codebook_size: int = 16):
        super().__init__()
        self.n_blocks = n_blocks
        self.codebook_size = codebook_size
        # normal(0,1) init per the repo VQ gotcha (vq_layer.py): match the expected
        # modulus-profile magnitudes, not the collapse-prone uniform(-1/N, 1/N).
        self.codebook = nn.Parameter(torch.randn(codebook_size, n_blocks))

    @torch.no_grad()
    def assign(self, k: torch.Tensor) -> torch.Tensor:
        """k (B, M, dn) -> kind ids (B, M) long. Argmin over codebook (fp32, no grad).

        Diagnostic readout — never routes (design §7).
        """
        m = block_modulus(k).float()                    # (B, M, nb)
        cb = self.codebook.float()                       # (K, nb)
        # ||m - cb||² = ||m||² + ||cb||² - 2 m·cb (fp32, mirrors VQ distance math).
        dists = (
            m.pow(2).sum(-1, keepdim=True)               # (B, M, 1)
            + cb.pow(2).sum(-1)                          # (K,)
            - 2.0 * m @ cb.t()                           # (B, M, K)
        )
        return dists.argmin(dim=-1)                      # (B, M)

    def commitment_loss(self, k: torch.Tensor) -> torch.Tensor:
        """VQ commitment loss for the modulus profile (fp32; design §7 / repo VQ gotcha).

        MSE between the modulus profile and its nearest (stop-grad) codebook entry,
        symmetrized with the codebook-update term. Only ever called when the kind head
        is enabled with a nonzero weight; default config leaves the kind head untrained.
        """
        m = block_modulus(k).float()                    # (B, M, nb)
        ids = self.assign(k)                            # (B, M)
        chosen = F.embedding(ids, self.codebook.float())  # (B, M, nb)
        # commitment (pull m to codebook) + codebook (pull codebook to m), each w/ a
        # stop-grad on the other side — standard VQ-VAE straight-through losses.
        commit = F.mse_loss(m, chosen.detach())
        codebook = F.mse_loss(chosen, m.detach())
        return commit + codebook
