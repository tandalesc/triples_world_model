"""Conditional flow matching dynamics over bottleneck space.

Replaces deterministic forward_dynamics with a learned velocity field that
samples z_target ~ p(z | z_prev, mode) by integrating from N(0, I) at t=0
to the data distribution at t=1.

Architectural bet: the chrF=49 wall on TextWorld advance is a manifold
mismatch between deterministic dynamics output and compressor output.
Identity decode hits chrF 96 from the compressor manifold; advance hits
chrF 24 from the dynamics manifold. Training the flow's target to be
compressor(target) puts dynamics output on the same manifold by
construction, so the frozen identity decoder can decode it cleanly.

Training (rectified flow / OT-CFM):
    z_prev   = compressor(input).detach()
    z_target = compressor(target).detach()
    t ~ U(0, 1); eps ~ N(0, I)
    z_t      = (1 - t) * eps + t * z_target
    target_v = z_target - eps
    loss     = MSE(v_theta(z_t, t, z_prev, mode), target_v)

Sampling (Euler):
    z = randn_like(z_prev)
    for i in range(n_steps):
        v = v_theta(z, t=i/n_steps, z_prev, mode)
        z = z + v * (1 / n_steps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_decoder import AdaLNZeroLayer, TimestepEmbedding


class FlowDynamics(nn.Module):
    """Velocity field for conditional flow matching in bottleneck space.

    Operates on (B, N, d) tensors where N = max_triples * 3. The bottleneck
    sequence has no positional encoding — set-equivariance is preserved
    inside the dynamics, identical to the deterministic dynamics core.

    Conditioning is split:
      - timestep:  AdaLN context, broadcast across positions
      - z_prev:    cross-attention memory (no pos enc)
      - mode_ids:  prefixed to z_prev as a 3-token mode triple
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        num_modes: int,
        max_triples: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_triples = max_triples
        self.num_modes = num_modes

        self.mode_emb = nn.Embedding(num_modes * 3, d_model)
        self.mode_role_emb = nn.Embedding(3, d_model)

        self.time_emb = TimestepEmbedding(d_model)

        self.layers = nn.ModuleList([
            AdaLNZeroLayer(
                d_model=d_model,
                n_heads=n_heads,
                context_dim=d_model,
                d_ff=d_ff or d_model * 4,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj_out = nn.Linear(d_model, d_model)
        # Zero-init so the field starts at v ≡ 0; trained from there.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _build_mode_triple(self, mode_ids: torch.Tensor) -> torch.Tensor:
        device = mode_ids.device
        base = mode_ids * 3
        slot_ids = base.unsqueeze(1) + torch.arange(3, device=device)
        return self.mode_emb(slot_ids) + self.mode_role_emb(
            torch.arange(3, device=device)
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_prev: torch.Tensor,
        mode_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at point z_t on the flow path.

        z_t:      (B, N, d) point on flow path between noise (t=0) and data (t=1)
        t:        (B,) scalar time in [0, 1]
        z_prev:   (B, N, d) conditioning bottleneck (compressor of input)
        mode_ids: (B,) mode index (advance/query/identity)
        """
        B, N, d = z_t.shape

        mode_triple = self._build_mode_triple(mode_ids)  # (B, 3, d)
        memory = torch.cat([mode_triple, z_prev], dim=1)  # (B, 3+N, d)

        t_emb = self.time_emb(t)  # (B, d)
        t_ctx = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, d)

        x = z_t
        for layer in self.layers:
            x = layer(x, context=t_ctx, memory=memory)
        x = self.norm_out(x)
        return self.proj_out(x)

    @torch.no_grad()
    def sample(
        self,
        z_prev: torch.Tensor,
        mode_ids: torch.Tensor,
        n_steps: int = 10,
        z_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler ODE integration from N(0, I) at t=0 to data at t=1."""
        B, N, d = z_prev.shape
        device = z_prev.device
        z = torch.randn_like(z_prev) if z_init is None else z_init
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=z.dtype)
            v = self(z, t, z_prev, mode_ids)
            z = z + dt * v
        return z


def flow_loss(
    flow: FlowDynamics,
    z_prev: torch.Tensor,
    z_target: torch.Tensor,
    mode_ids: torch.Tensor,
) -> torch.Tensor:
    """Rectified flow MSE loss.

    Sample t uniformly per example, interpolate linearly, regress the
    constant velocity z_target - eps.
    """
    B = z_prev.shape[0]
    device = z_prev.device

    t = torch.rand(B, device=device, dtype=z_prev.dtype)
    eps = torch.randn_like(z_target)
    t_b = t.view(B, 1, 1)
    z_t = (1.0 - t_b) * eps + t_b * z_target
    target_v = z_target - eps

    pred_v = flow(z_t, t, z_prev, mode_ids)
    return F.mse_loss(pred_v, target_v)
