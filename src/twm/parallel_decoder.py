"""Parallel set-to-sequence decoder (DETR-style).

Generates all token positions simultaneously via learned position
queries that cross-attend to bottleneck memory. No autoregressive
ordering, no diffusion noise — permutation-invariant over input
triples, parallel over output positions.

Architecture:
    bottleneck (B, N*3, d) → memory for cross-attention
    learned position queries (T, d) → self-attention + cross-attention → logits

Like DETR object queries but for token positions. Each query learns
to attend to the bottleneck for its content and to other queries for
coordination (avoiding duplicates, ensuring coverage).

Train = gen by construction: one forward pass, CE loss per position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelDecoder(nn.Module):
    """Non-autoregressive parallel decoder over bottleneck memory."""

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

        # Shared frozen embedding
        self.token_emb = token_emb
        self.vocab_size = token_emb.num_embeddings

        # Learned position queries — one per output position
        # These are the "what goes in position i?" queries
        self.pos_queries = nn.Embedding(max_text_tokens, d_model)

        # Project bottleneck to memory for cross-attention
        bn_d = self.bottleneck_dim
        self.memory_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Decoder layers: self-attention (queries coordinate) + cross-attention (read bottleneck)
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

        # Length head
        self.length_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _build_memory(self, bottleneck, pre_dynamics=None):
        """Build cross-attention memory, optionally multi-level."""
        projected = self.memory_proj(bottleneck)
        if pre_dynamics is not None:
            pre_projected = self.memory_proj(pre_dynamics)
            return torch.cat([pre_projected, projected], dim=1)
        return projected

    def forward(
        self,
        bottleneck: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        pre_dynamics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel forward pass — predict all positions at once.

        Args:
            bottleneck: (B, N*3, d) post-dynamics bottleneck
            target_ids: (B, T) target token IDs (for loss computation)
            target_pad_mask: (B, T) True where padding
            pre_dynamics: (B, N*3, d) optional pre-dynamics

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = target_ids.shape
        device = target_ids.device

        memory = self._build_memory(bottleneck, pre_dynamics)

        # Position queries for T positions
        queries = self.pos_queries(torch.arange(T, device=device))  # (T, d)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, T, d)

        # Decode: self-attention among queries + cross-attention to memory
        # No causal mask — all positions see each other
        out = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_key_padding_mask=target_pad_mask,
        )

        logits = self.output_proj(self.ln_f(out))
        return logits

    def forward_length(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict text length from bottleneck."""
        pooled = bottleneck.mean(dim=1)
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True)
        norm_hint = norm_hint / self.max_text_tokens
        return self.length_head(torch.cat([pooled, norm_hint], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        bottleneck: torch.Tensor,
        max_tokens: int | None = None,
        pre_dynamics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate — same as forward but without targets.

        Args:
            bottleneck: (B, N*3, d)
            max_tokens: override length (else uses length head)
            pre_dynamics: (B, N*3, d) optional

        Returns:
            (B, T) generated token IDs
        """
        B = bottleneck.shape[0]
        device = bottleneck.device

        memory = self._build_memory(bottleneck, pre_dynamics)

        if max_tokens is None:
            T = self.forward_length(bottleneck).round().long().clamp(
                1, self.max_text_tokens
            ).max().item()
        else:
            T = max_tokens

        queries = self.pos_queries(torch.arange(T, device=device))
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        out = self.decoder(tgt=queries, memory=memory)
        logits = self.output_proj(self.ln_f(out))

        return logits.argmax(-1)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
