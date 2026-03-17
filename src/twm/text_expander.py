"""Text Expander: triple-level 256d bottleneck → free text via diffusion.

Takes conditioning vectors from the shared bottleneck (N*3 vectors at 256d)
and generates a variable-length fluent text sequence using continuous diffusion
in the frozen embedding space — same framework as the triple expander.

Key differences from triple expander (DiffusionDecoder):
  - Operates over longer sequences (up to 64 text tokens vs 12 per slot)
  - Cross-attends to ALL triple-position vectors (N×3 memory tokens)
  - adaLN conditioning from pooled bottleneck vectors
  - Length head predicts text length from conditioning
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_decoder import (
    AdaLNZeroLayer,
    TimestepEmbedding,
    cosine_noise_schedule,
    importance_sample_timesteps,
)


class TextExpander(nn.Module):
    """Generates free text from triple-level bottleneck via continuous diffusion.

    Architecture:
        bottleneck vectors (N*3, 256d)
        → pool to adaLN conditioning
        → cross-attention memory
        → adaLN-Zero denoiser (3L) over text tokens
        → MSE x₀-prediction + NN decode
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        max_text_tokens: int = 64,
        max_triples: int = 8,
        dropout: float = 0.1,
        alpha_min: float = 0.01,
        timestep_bias_power: float = 1.0,
        use_decode_proj: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_text_tokens = max_text_tokens
        self.max_triples = max_triples
        self.alpha_min = alpha_min
        self.timestep_bias_power = timestep_bias_power

        # Shared frozen embedding table
        self.token_emb = token_emb

        # Text position embeddings
        self.pos_emb = nn.Embedding(max_text_tokens, d_model)

        # Timestep embedding
        # Context dim: d_model (from pooled conditioning)
        self.time_embed = TimestepEmbedding(d_model, embed_dim=d_model)

        # Conditioning: learned attention pool over bottleneck vectors.
        # Single learned query cross-attends to all N*3 positions, preserving
        # per-position gradient flow (no mean-pool division by 36).
        self.cond_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cond_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.cond_proj = nn.Linear(d_model, d_model)

        # Cross-attention memory projection: bottleneck → memory tokens
        self.memory_proj = nn.Linear(d_model, d_model)

        # adaLN-Zero denoiser layers with independent context signals:
        #   - pooled conditioning (d_model): what to denoise toward
        #   - timestep embedding (d_model): denoising scale/aggressiveness
        #   - position embedding (d_model): where in the sequence
        self.layers = nn.ModuleList([
            AdaLNZeroLayer(
                d_model=d_model,
                n_heads=n_heads,
                context_dims=[d_model, d_model, d_model],
                d_ff=d_model * 4,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Length head: predict text token count from pooled conditioning + bottleneck norm hint.
        # The norm hint gives the MLP a proxy for content density / slot occupancy
        # so it can distinguish "12 tokens" from "10 tokens" of similar content.
        self.length_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Decode projection for NN decode sharpening
        self.use_decode_proj = use_decode_proj
        if use_decode_proj:
            self.decode_proj = nn.Linear(d_model, d_model)

    def _pool_conditioning(
        self,
        bottleneck: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention-pool bottleneck vectors to a single conditioning vector.

        A learned query cross-attends to all N*3 bottleneck positions. This
        preserves per-position gradient flow (no mean-pool division by N*3)
        and lets the model decide what's globally relevant.

        Args:
            bottleneck: (B, N*3, d_model) triple-level vectors
            triple_pad_mask: (B, N) True where triple is pad

        Returns:
            (B, d_model) pooled conditioning
        """
        B = bottleneck.shape[0]

        # Build key padding mask for attention (over N*3 positions)
        key_pad_mask = None
        if triple_pad_mask is not None:
            N = triple_pad_mask.shape[1]
            key_pad_mask = triple_pad_mask.unsqueeze(-1).expand(B, N, 3).reshape(B, N * 3)

        query = self.cond_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.cond_attn(
            query=query, key=bottleneck, value=bottleneck,
            key_padding_mask=key_pad_mask,
        )  # (B, 1, d_model)

        return self.cond_proj(pooled.squeeze(1))  # (B, d_model)

    def _make_noise(self, like: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(like)
        emb_norm = self.token_emb.weight.norm(dim=-1).mean()
        noise = noise * emb_norm / (noise.norm(dim=-1, keepdim=True) + 1e-8)
        return noise

    def _nn_decode(self, pred_emb: torch.Tensor) -> torch.Tensor:
        emb = self.decode_proj(pred_emb) if self.use_decode_proj else pred_emb
        pred_norm = F.normalize(emb, dim=-1)
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.T)
        return sims.argmax(dim=-1)

    def decode_proj_logits(self, pred_emb: torch.Tensor) -> torch.Tensor:
        emb = self.decode_proj(pred_emb) if self.use_decode_proj else pred_emb
        pred_norm = F.normalize(emb, dim=-1)
        emb_norm = F.normalize(self.token_emb.weight.detach(), dim=-1)
        return torch.matmul(pred_norm, emb_norm.T)

    def forward(
        self,
        bottleneck: torch.Tensor,
        target_text_ids: torch.Tensor,
        target_text_pad_mask: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: corrupt target text, predict clean embeddings.

        Args:
            bottleneck: (B, N*3, d_model) conditioning from compressor
            target_text_ids: (B, T) target text BPE token IDs
            target_text_pad_mask: (B, T) True where text token is padding
            triple_pad_mask: (B, N) True where triple is padding
            timestep: (B,) diffusion timesteps, sampled if None

        Returns:
            pred_emb: (B, T, d_model) predicted clean embeddings
            all_mask: (B, T) all-True mask (continuous mode)
        """
        B, T = target_text_ids.shape
        device = target_text_ids.device

        # Pool conditioning
        cond = self._pool_conditioning(bottleneck, triple_pad_mask)  # (B, d_model)

        # Cross-attention memory: project bottleneck vectors
        memory = self.memory_proj(bottleneck)  # (B, N*3, d_model)

        # Sample timestep
        if timestep is None:
            timestep = importance_sample_timesteps(B, device, self.timestep_bias_power)

        # Noise schedule
        alpha_t = cosine_noise_schedule(timestep, alpha_min=self.alpha_min)
        alpha_t = alpha_t.view(B, 1, 1)

        # Get clean embeddings and corrupt
        original_emb = self.token_emb(target_text_ids)
        noise = self._make_noise(original_emb)
        corrupted = torch.sqrt(alpha_t) * original_emb + torch.sqrt(1 - alpha_t) * noise

        # Add position embeddings
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        x = corrupted + self.pos_emb(pos_ids)

        # Build independent context signals for adaLN
        t_emb = self.time_embed(timestep)  # (B, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=device))  # (T, d_model)
        ctx_cond = cond.unsqueeze(1).expand(B, T, -1)      # (B, T, d_model)
        ctx_time = t_emb.unsqueeze(1).expand(B, T, -1)     # (B, T, d_model)
        ctx_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)   # (B, T, d_model)

        # Run denoiser with factored context signals
        for layer in self.layers:
            x = layer(x, [ctx_cond, ctx_time, ctx_pos], memory)

        decoded = self.ln_f(x)
        all_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        return decoded, all_mask

    def _length_input(self, bottleneck: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Build length head input: pooled conditioning + bottleneck norm hint.

        The norm hint is the mean L2 norm of bottleneck slots, normalized by
        max_text_tokens. This gives the length head a scale-invariant signal
        for how "full" the bottleneck is, helping it distinguish similar
        content at different lengths.
        """
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)
        norm_hint = norm_hint / self.max_text_tokens  # normalize to ~[0, 1]
        return torch.cat([cond, norm_hint], dim=-1)  # (B, d_model + 1)

    def forward_length(
        self,
        bottleneck: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict text length from conditioning.

        Returns:
            (B,) predicted text token counts (raw regression)
        """
        cond = self._pool_conditioning(bottleneck, triple_pad_mask)
        return self.length_head(self._length_input(bottleneck, cond)).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        bottleneck: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
        n_steps: int = 10,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        """Generate text token IDs via iterative denoising.

        Args:
            bottleneck: (B, N*3, d_model) conditioning
            triple_pad_mask: (B, N) True where triple is pad
            n_steps: denoising steps
            max_tokens: override text length (else uses length head)

        Returns:
            (B, T) generated token IDs
        """
        B = bottleneck.shape[0]
        device = bottleneck.device

        # Pool once, reuse for both length prediction and conditioning
        cond = self._pool_conditioning(bottleneck, triple_pad_mask)

        if max_tokens is None:
            len_input = self._length_input(bottleneck, cond)
            T = self.length_head(len_input).squeeze(-1).round().long().clamp(1, self.max_text_tokens).max().item()
        else:
            T = max_tokens
        memory = self.memory_proj(bottleneck)

        # Start from noise
        x = self._make_noise(torch.zeros(B, T, self.d_model, device=device))
        pos_emb = self.pos_emb(torch.arange(T, device=device))

        schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_now = schedule[i]
            t_next = schedule[i + 1]
            alpha_next = cosine_noise_schedule(t_next.unsqueeze(0), alpha_min=self.alpha_min)

            t_batch = t_now.expand(B)
            t_emb = self.time_embed(t_batch)

            ctx_cond = cond.unsqueeze(1).expand(B, T, -1)
            ctx_time = t_emb.unsqueeze(1).expand(B, T, -1)
            ctx_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)

            x_input = x + pos_emb.unsqueeze(0)
            for layer in self.layers:
                x_input = layer(x_input, [ctx_cond, ctx_time, ctx_pos], memory)
            pred_emb = self.ln_f(x_input)

            # Re-noise to next level
            if i < n_steps - 1:
                noise = self._make_noise(pred_emb)
                alpha_n = alpha_next.view(1, 1, 1)
                x = torch.sqrt(alpha_n) * pred_emb + torch.sqrt(1 - alpha_n) * noise
            else:
                x = pred_emb

        # Final clean prediction
        t_batch = torch.zeros(B, device=device)
        t_emb = self.time_embed(t_batch)
        ctx_cond = cond.unsqueeze(1).expand(B, T, -1)
        ctx_time = t_emb.unsqueeze(1).expand(B, T, -1)
        ctx_pos = pos_emb.unsqueeze(0).expand(B, -1, -1)

        x_input = x + pos_emb.unsqueeze(0)
        for layer in self.layers:
            x_input = layer(x_input, [ctx_cond, ctx_time, ctx_pos], memory)
        decoded = self.ln_f(x_input)

        return self._nn_decode(decoded)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
