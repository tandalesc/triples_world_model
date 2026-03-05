"""Triple World Model — set-to-set transformer encoder with input residual."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_triples: int = 8
    dropout: float = 0.1
    pretrained_embed_dim: int | None = None  # set when using pretrained embeddings

    @property
    def max_positions(self) -> int:
        return self.max_triples * 3

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            return cls(**json.load(f))


class TripleWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained_embeds: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        embed_dim = config.pretrained_embed_dim or config.d_model
        if pretrained_embeds is not None:
            embed_dim = pretrained_embeds.shape[1]
            config.pretrained_embed_dim = embed_dim

        self.token_emb = nn.Embedding(config.vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeds is not None:
            self.token_emb.weight.data.copy_(pretrained_embeds)

        if embed_dim != config.d_model:
            self.embed_proj = nn.Linear(embed_dim, config.d_model, bias=False)
        else:
            self.embed_proj = nn.Identity()

        self.triple_pos_emb = nn.Embedding(config.max_triples, config.d_model)
        self.role_emb = nn.Embedding(3, config.d_model)  # entity=0, relation=1, value=2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def _build_position_encoding(self, device: torch.device) -> torch.Tensor:
        """(1, max_positions, d_model) positional encoding from triple_index + role."""
        triple_idx = torch.arange(self.config.max_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(self.config.max_triples)
        return (self.triple_pos_emb(triple_idx) + self.role_emb(role_idx)).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, max_triples * 3) token IDs, <pad>=0 for empty slots
        Returns:
            logits: (B, max_triples * 3, vocab_size)
        """
        pos_enc = self._build_position_encoding(input_ids.device)
        input_emb = self.embed_proj(self.token_emb(input_ids))
        x = input_emb + pos_enc

        # Padding mask: pad positions CAN attend to non-pad (useful representations),
        # but non-pad positions do NOT attend to pad (no useful info there)
        pad_mask = input_ids == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)

        # Input residual: biases output toward copying input tokens.
        # Most of the world persists — the encoder only needs to learn the delta.
        # For <pad> input positions (embedding is zero), this adds nothing.
        x = x + input_emb

        return self.head(x)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return predicted token IDs (greedy argmax)."""
        logits = self.forward(input_ids)
        return logits.argmax(dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
