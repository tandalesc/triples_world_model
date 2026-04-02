"""Discrete entity prediction head for dynamics latents.

Predicts key entity tokens (objects, locations, actions) directly
from the dynamics latent as hard categorical classifications.
These become prefix conditioning for the AR decoder, providing
discrete disambiguation that soft cross-attention cannot resolve.

The dynamics latent encodes "take food-item from kitchen-surface"
as a continuous region. This head resolves it to specific discrete
tokens: ["white", "onion", "table"] which the AR decoder assembles
into "You take the white onion from the table."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityHead(nn.Module):
    """Predict discrete entity tokens from dynamics latent.

    Architecture:
        latent (B, N*3, d) → attention-pool to K entity queries
        → per-query MLP → vocab logits → argmax → entity tokens

    The predicted tokens become prefix conditioning for the AR decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_entity_slots: int = 8,
        n_heads: int = 4,
        bottleneck_dim: int | None = None,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_entity_slots = n_entity_slots
        self.pad_id = pad_id
        bn_d = bottleneck_dim or d_model

        # Learned entity queries — each query extracts one entity token
        self.entity_queries = nn.Parameter(torch.randn(n_entity_slots, d_model) * 0.02)

        # Project latent to d_model for cross-attention
        self.latent_proj = nn.Linear(bn_d, d_model)

        # Cross-attention: entity queries attend to latent positions
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1,
        )
        self.ln = nn.LayerNorm(d_model)

        # Per-slot classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, vocab_size),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict entity token logits from latent.

        Args:
            bottleneck: (B, N*3, d) dynamics latent

        Returns:
            logits: (B, n_entity_slots, vocab_size)
        """
        B = bottleneck.shape[0]
        device = bottleneck.device

        memory = self.latent_proj(bottleneck)  # (B, N*3, d)
        queries = self.entity_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        attended, _ = self.cross_attn(queries, memory, memory)
        attended = self.ln(attended + queries)  # residual

        return self.classifier(attended)  # (B, K, V)

    @torch.no_grad()
    def predict(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict entity tokens (hard argmax).

        Args:
            bottleneck: (B, N*3, d)

        Returns:
            entity_ids: (B, n_entity_slots) predicted token IDs
        """
        logits = self.forward(bottleneck)
        return logits.argmax(-1)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
