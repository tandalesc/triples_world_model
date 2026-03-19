"""Diffusion Compressor: iterative denoising from noise → W-space bottleneck.

Instead of one-shot compression (learned queries → cross-attention → bottleneck),
starts from Gaussian noise in W-space and iteratively denoises conditioned on
encoded BPE text. Each denoising step refines the bottleneck representation,
giving the model multiple "thinking steps" to find good latent representations.

This mirrors the TextExpander (which denoises noise → BPE tokens conditioned on
bottleneck), creating a symmetric architecture:
  Compressor: noise in W → denoise conditioned on BPE → bottleneck
  Expander:   noise in BPE → denoise conditioned on bottleneck → BPE

No per-step diffusion loss. The K denoising steps form a differentiable graph,
trained end-to-end via the downstream expander reconstruction loss.
"""

import torch
import torch.nn as nn

from .diffusion_decoder import (
    AdaLNZeroLayer,
    TimestepEmbedding,
    cosine_noise_schedule,
)


class DiffusionCompressor(nn.Module):
    """Compresses free text into triple-level bottleneck via iterative denoising.

    Architecture:
        Stage A — Text encoding (computed once):
            frozen_token_emb → + text_pos_emb → self_attention (N layers) → encoded text

        Stage B — Iterative denoising (K steps):
            noise ~ N(0, I) in (B, N*3, d_model)
            for each step:
                adaLN-Zero layers cross-attending to encoded text
                predict clean bottleneck, re-noise for next step
            output: denoised (B, N*3, d_model) bottleneck
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_denoise_layers: int = 3,
        n_denoise_steps: int = 5,
        max_triples: int = 8,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_triples = max_triples
        self.n_denoise_steps = n_denoise_steps

        # === Stage A: Text encoding (same as TextCompressor) ===
        self.token_emb = token_emb
        self.text_pos_emb = nn.Embedding(max_text_tokens, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # === Conditioning: pool encoded text to single vector ===
        self.cond_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cond_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.cond_proj = nn.Linear(d_model, d_model)

        # Cross-attention memory projection
        self.memory_proj = nn.Linear(d_model, d_model)

        # === Structural embeddings for bottleneck positions ===
        self.role_emb = nn.Embedding(3, d_model)
        self.triple_pos_emb = nn.Embedding(max_triples, d_model)

        # === Denoiser (mirrors TextExpander) ===
        self.time_embed = TimestepEmbedding(d_model, embed_dim=d_model)

        # 3 independent context signals: pooled text cond, timestep, structural position
        self.layers = nn.ModuleList([
            AdaLNZeroLayer(
                d_model=d_model,
                n_heads=n_heads,
                context_dims=[d_model, d_model, d_model],
                d_ff=d_model * 4,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(n_denoise_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Output normalization (matches TextCompressor)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_pad_mask: torch.Tensor,
        n_triples: int,
    ) -> torch.Tensor:
        """Compress text to bottleneck via iterative denoising.

        Args:
            text_token_ids: (B, T) BPE token IDs
            text_pad_mask: (B, T) True where padding
            n_triples: number of triples to extract

        Returns:
            (B, max_triples * 3, d_model) bottleneck vectors
        """
        B = text_token_ids.shape[0]
        T_text = text_token_ids.shape[1]
        device = text_token_ids.device
        N = n_triples
        n_queries = N * 3

        # === Stage A: Encode text (once) ===
        pos_ids = torch.arange(T_text, device=device).unsqueeze(0)
        text_emb = self.token_emb(text_token_ids) + self.text_pos_emb(pos_ids)
        encoded_text = self.self_attn(text_emb, src_key_padding_mask=text_pad_mask)

        # === Pool conditioning ===
        cond_q = self.cond_query.expand(B, -1, -1)
        pooled, _ = self.cond_attn(
            cond_q, encoded_text, encoded_text,
            key_padding_mask=text_pad_mask,
        )
        cond = self.cond_proj(pooled.squeeze(1))  # (B, d_model)

        # Cross-attention memory
        memory = self.memory_proj(encoded_text)  # (B, T_text, d_model)

        # === Structural embeddings ===
        triple_idx = torch.arange(N, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(N)
        struct_emb = self.role_emb(role_idx) + self.triple_pos_emb(triple_idx)  # (n_queries, d_model)

        # === Iterative denoising ===
        x = torch.randn(B, n_queries, self.d_model, device=device)

        schedule = torch.linspace(1.0, 0.0, self.n_denoise_steps + 1, device=device)

        for i in range(self.n_denoise_steps):
            t_now = schedule[i]
            t_next = schedule[i + 1]

            # Timestep embedding
            t_batch = t_now.expand(B)
            t_emb = self.time_embed(t_batch)  # (B, d_model)

            # Build context signals (expanded to per-position)
            ctx_cond = cond.unsqueeze(1).expand(B, n_queries, -1)
            ctx_time = t_emb.unsqueeze(1).expand(B, n_queries, -1)
            ctx_struct = struct_emb.unsqueeze(0).expand(B, -1, -1)

            # Add structural embedding to noisy input
            x_input = x + struct_emb.unsqueeze(0)

            # Run denoiser layers with cross-attention to encoded text
            for layer in self.layers:
                x_input = layer(x_input, [ctx_cond, ctx_time, ctx_struct], memory)

            pred_clean = self.ln_f(x_input)

            # Re-noise for next step (unless last)
            if i < self.n_denoise_steps - 1:
                alpha_next = cosine_noise_schedule(t_next.unsqueeze(0))
                alpha_n = alpha_next.view(1, 1, 1)
                noise = torch.randn_like(pred_clean)
                x = torch.sqrt(alpha_n) * pred_clean + torch.sqrt(1 - alpha_n) * noise
            else:
                x = pred_clean

        # Pad to max_triples * 3 if needed
        if N < self.max_triples:
            pad_size = (self.max_triples - N) * 3
            pad = torch.zeros(B, pad_size, self.d_model, device=device)
            x = torch.cat([x, pad], dim=1)

        return self.out_ln(x)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
