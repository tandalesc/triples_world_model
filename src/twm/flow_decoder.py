"""Flow matching decoder for TWM latent-to-text generation.

Optimal Transport Conditional Flow Matching (OT-CFM) in the frozen
BPE embedding space. Learns a velocity field that transports noise
to token embeddings via deterministic ODE trajectories.

Advantages over diffusion:
  - Deterministic trajectories (no train/gen mismatch)
  - Straight interpolation paths (fewer ODE steps needed)
  - No stochastic noise schedule to tune

Advantages over AR:
  - Parallel prediction (no exposure bias from sequential decoding)
  - No left-to-right ordering bias

Architecture:
    bottleneck (B, N*3, d) → cross-attention memory (NO pos enc on K/V)
    noise (B, T, d) + timestep → adaLN-Zero denoiser → velocity prediction
    Euler ODE: noise → clean embeddings → NN decode → token IDs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_decoder import (
    AdaLNZeroLayer,
    TimestepEmbedding,
)


class FlowDecoder(nn.Module):
    """Flow matching decoder: velocity field in frozen embedding space.

    Training: predict velocity v_t at random interpolation point x_t.
    Generation: Euler ODE integration from noise to clean, then NN decode.
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_text_tokens: int = 128,
        dropout: float = 0.1,
        bottleneck_dim: int | None = None,
        n_ode_steps: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_text_tokens = max_text_tokens
        self.bottleneck_dim = bottleneck_dim or d_model
        self.n_ode_steps = n_ode_steps

        # Shared frozen embedding table
        self.token_emb = token_emb

        # Position embeddings for output token positions
        self.pos_emb = nn.Embedding(max_text_tokens, d_model)

        # Timestep conditioning
        self.time_embed = TimestepEmbedding(d_model, embed_dim=d_model)

        # Memory projection: bottleneck → cross-attention K/V
        # NO positional encoding — preserves set invariance of dynamics core
        bn_d = self.bottleneck_dim
        self.bn_input_proj = nn.Linear(bn_d, d_model) if bn_d != d_model else nn.Identity()
        self.memory_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Conditioning pool for length head
        self.cond_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cond_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.cond_proj = nn.Linear(d_model, d_model)

        # adaLN-Zero denoiser layers with two context signals:
        #   - timestep embedding (d_model): flow progress
        #   - position embedding (d_model): where in the sequence
        self.layers = nn.ModuleList([
            AdaLNZeroLayer(
                d_model=d_model,
                n_heads=n_heads,
                context_dims=[d_model, d_model],
                d_ff=d_model * 4,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Length prediction head
        self.length_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _project_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.bn_input_proj(bottleneck)

    def _build_memory(self, bottleneck, pre_dynamics=None):
        """Build cross-attention memory. No positional encoding (set invariance)."""
        projected = self._project_bottleneck(bottleneck)
        if pre_dynamics is not None:
            pre_projected = self._project_bottleneck(pre_dynamics)
            combined = torch.cat([pre_projected, projected], dim=1)
            return self.memory_proj(combined)
        return self.memory_proj(projected)

    def _pool_conditioning(self, bottleneck):
        """Attention-pool bottleneck to single conditioning vector."""
        projected = self._project_bottleneck(bottleneck)
        B = projected.shape[0]
        query = self.cond_query.expand(B, -1, -1)
        pooled, _ = self.cond_attn(query, projected, projected)
        return self.cond_proj(pooled.squeeze(1))

    def _make_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Unit-normalized noise matching embedding magnitude."""
        noise = torch.randn_like(like)
        return F.normalize(noise, dim=-1)

    def _predict_velocity(self, x_t, t_batch, memory, T):
        """Run denoiser to predict velocity at interpolation point x_t."""
        B = x_t.shape[0]
        device = x_t.device

        t_emb = self.time_embed(t_batch)  # (B, d)
        pos_emb = self.pos_emb(torch.arange(T, device=device))  # (T, d)

        ctx_time = t_emb.unsqueeze(1).expand(B, T, -1)
        ctx_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)

        x = x_t + pos_emb.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, [ctx_time, ctx_pos], memory)

        return self.ln_f(x)

    def forward(
        self,
        bottleneck: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pre_dynamics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Training forward: compute velocity MSE loss.

        Args:
            bottleneck: (B, N*3, d) post-dynamics latent
            target_ids: (B, T) target token IDs
            target_pad_mask: (B, T) True where padding
            pre_dynamics: (B, N*3, d) optional pre-dynamics

        Returns:
            velocity_loss: scalar
            length_loss: scalar
            metrics: dict with velocity_mse, length_mse, tok_acc
        """
        B, T = target_ids.shape
        device = target_ids.device

        memory = self._build_memory(bottleneck, pre_dynamics)

        # Clean target embeddings
        x_0 = self.token_emb(target_ids)  # (B, T, d)

        # Sample noise (unit-normalized)
        x_1 = self._make_noise(x_0)

        # Sample timestep t ~ U(0, 1)
        # Convention: t=0 is noise, t=1 is clean
        t = torch.rand(B, device=device)

        # Interpolate: x_t = (1-t)*noise + t*clean
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_1 + t_expand * x_0

        # Target velocity: v = clean - noise (constant along straight path)
        v_target = x_0 - x_1

        # Predict velocity
        v_pred = self._predict_velocity(x_t, t, memory, T)

        # Velocity MSE on non-pad positions
        non_pad = ~target_pad_mask
        if non_pad.any():
            velocity_loss = F.mse_loss(v_pred[non_pad], v_target[non_pad])
        else:
            velocity_loss = torch.tensor(0.0, device=device)

        # Length prediction
        cond = self._pool_conditioning(bottleneck)
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True) / self.max_text_tokens
        len_input = torch.cat([cond, norm_hint], dim=-1)
        len_pred = self.length_head(len_input).squeeze(-1)
        target_len = non_pad.sum(dim=-1).float()
        length_loss = F.mse_loss(len_pred, target_len)

        # Metrics: NN-decode accuracy at t=1 (clean, no ODE needed)
        with torch.no_grad():
            emb_norm = F.normalize(self.token_emb.weight, dim=-1)
            # Check velocity prediction quality: apply v_pred to x_t, decode
            x_corrected = x_t + (1 - t_expand) * v_pred  # single-step correction to t=1
            x_corrected_norm = F.normalize(x_corrected[non_pad], dim=-1)
            nn_ids = torch.matmul(x_corrected_norm, emb_norm.T).argmax(-1)
            tok_acc = (nn_ids == target_ids[non_pad]).float().mean().item()

        metrics = {
            "velocity_mse": velocity_loss.item(),
            "length_mse": length_loss.item(),
            "tok_acc": tok_acc,
        }

        return velocity_loss, length_loss, metrics

    def forward_length(self, bottleneck):
        """Predict text length from bottleneck."""
        cond = self._pool_conditioning(bottleneck)
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True) / self.max_text_tokens
        return self.length_head(torch.cat([cond, norm_hint], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        bottleneck: torch.Tensor,
        n_steps: int | None = None,
        max_tokens: int | None = None,
        pre_dynamics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate token IDs via Euler ODE integration.

        Args:
            bottleneck: (B, N*3, d) conditioning
            n_steps: ODE integration steps (default: self.n_ode_steps)
            max_tokens: override text length
            pre_dynamics: (B, N*3, d) optional

        Returns:
            (B, T) generated token IDs
        """
        B = bottleneck.shape[0]
        device = bottleneck.device
        steps = n_steps or self.n_ode_steps

        memory = self._build_memory(bottleneck, pre_dynamics)

        # Predict length
        if max_tokens is None:
            T = self.forward_length(bottleneck).round().long().clamp(
                1, self.max_text_tokens
            ).max().item()
        else:
            T = max_tokens

        # Start from noise (t=0)
        x = self._make_noise(torch.zeros(B, T, self.d_model, device=device))
        dt = 1.0 / steps

        # Euler ODE integration: t=0 (noise) → t=1 (clean)
        for i in range(steps):
            t = i * dt
            t_batch = torch.full((B,), t, device=device)

            # Predict velocity at current point
            v = self._predict_velocity(x, t_batch, memory, T)

            # Euler step
            x = x + dt * v

            # Re-normalize to unit sphere to prevent drift
            x = F.normalize(x, dim=-1)

        # NN decode: cosine similarity against frozen embedding table
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        x_norm = F.normalize(x, dim=-1)
        sims = torch.matmul(x_norm, emb_norm.T)
        token_ids = sims.argmax(dim=-1)

        return token_ids

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
