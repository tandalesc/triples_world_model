"""Set-to-sequence transformer decoder.

Takes bottleneck vectors (set of triple-level latents) and generates
a token sequence via autoregressive cross-attention decoding.
Standard CE loss — no diffusion, no noise, train=gen by construction.

Architecture:
    bottleneck (B, N*3, d) → memory for cross-attention
    <bos> + target tokens → causal self-attention + cross-attention → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqDecoder(nn.Module):
    """Autoregressive transformer decoder over bottleneck memory."""

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_text_tokens: int = 128,
        dropout: float = 0.1,
        bottleneck_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_text_tokens = max_text_tokens
        self.bottleneck_dim = bottleneck_dim or d_model

        # Shared frozen embedding (same as compressor/expander)
        self.token_emb = token_emb
        self.vocab_size = token_emb.num_embeddings

        # Input projection if token embedding dim != d_model
        emb_dim = token_emb.embedding_dim
        self.input_proj = nn.Linear(emb_dim, d_model) if emb_dim != d_model else nn.Identity()

        # Position embeddings for decoder sequence
        self.pos_emb = nn.Embedding(max_text_tokens + 1, d_model)  # +1 for BOS

        # Project bottleneck to memory for cross-attention
        bn_d = self.bottleneck_dim
        self.memory_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.vocab_size)

        # BOS token embedding (learned)
        self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Length head: predict token count from bottleneck
        self.length_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _pool_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Mean-pool bottleneck for length prediction."""
        return bottleneck.mean(dim=1)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pre_dynamics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            bottleneck: (B, N*3, d) post-dynamics bottleneck
            target_ids: (B, T) target token IDs
            target_pad_mask: (B, T) True where padding
            pre_dynamics: (B, N*3, d) optional pre-dynamics for multi-level

        Returns:
            logits: (B, T, vocab_size) prediction logits for each position
        """
        B, T = target_ids.shape
        device = target_ids.device

        # Build memory from bottleneck (+ optional pre-dynamics)
        projected = self.memory_proj(bottleneck)
        if pre_dynamics is not None:
            pre_projected = self.memory_proj(pre_dynamics)
            memory = torch.cat([pre_projected, projected], dim=1)
        else:
            memory = projected

        # Embed target tokens: BOS + target[:-1] (shifted right)
        tok_emb = self.input_proj(self.token_emb(target_ids))  # (B, T, d)
        bos = self.bos_emb.expand(B, -1, -1)  # (B, 1, d)
        decoder_input = torch.cat([bos, tok_emb[:, :-1]], dim=1)  # (B, T, d)

        # Add position embeddings
        pos = self.pos_emb(torch.arange(T, device=device))
        decoder_input = decoder_input + pos

        # Causal mask
        causal_mask = self._causal_mask(T, device)

        # Decode
        out = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_pad_mask,
        )

        logits = self.output_proj(self.ln_f(out))
        return logits

    def forward_length(
        self,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """Predict text length from bottleneck."""
        pooled = self._pool_bottleneck(bottleneck)
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True)
        norm_hint = norm_hint / self.max_text_tokens
        return self.length_head(torch.cat([pooled, norm_hint], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        bottleneck: torch.Tensor,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        pre_dynamics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            bottleneck: (B, N*3, d) conditioning
            max_tokens: max output length (uses length head if None)
            temperature: sampling temperature (0 = greedy)
            pre_dynamics: (B, N*3, d) optional pre-dynamics

        Returns:
            (B, T) generated token IDs
        """
        B = bottleneck.shape[0]
        device = bottleneck.device
        pad_id = self.token_emb.weight.shape[0] - 3  # hack: assume pad is near end

        # Build memory
        projected = self.memory_proj(bottleneck)
        if pre_dynamics is not None:
            pre_projected = self.memory_proj(pre_dynamics)
            memory = torch.cat([pre_projected, projected], dim=1)
        else:
            memory = projected

        # Predict length
        if max_tokens is None:
            pooled = self._pool_bottleneck(bottleneck)
            norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True)
            norm_hint = norm_hint / self.max_text_tokens
            T = self.length_head(
                torch.cat([pooled, norm_hint], dim=-1)
            ).squeeze(-1).round().long().clamp(1, self.max_text_tokens).max().item()
        else:
            T = max_tokens

        # Start with BOS
        generated = torch.zeros(B, T, dtype=torch.long, device=device)
        current = self.bos_emb.expand(B, -1, -1)  # (B, 1, d)

        for t in range(T):
            pos = self.pos_emb(torch.arange(current.shape[1], device=device))
            decoder_input = current + pos[:current.shape[1]]

            causal_mask = self._causal_mask(current.shape[1], device)

            out = self.decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=causal_mask,
            )

            logits = self.output_proj(self.ln_f(out[:, -1:]))  # (B, 1, V)

            if temperature == 0:
                next_id = logits.argmax(-1)  # (B, 1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs.squeeze(1), 1)  # (B, 1)

            generated[:, t] = next_id.squeeze(-1)

            # Append to sequence
            next_emb = self.input_proj(self.token_emb(next_id))  # (B, 1, d)
            current = torch.cat([current, next_emb], dim=1)

        return generated

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
