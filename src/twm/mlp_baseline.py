"""MLP baseline for Triple World Model.

Same GloVe embeddings, same input/output format, but replaces the
transformer encoder with a simple MLP. This tests whether the transformer's
attention mechanism actually contributes beyond the pretrained embeddings.
"""

import torch
import torch.nn as nn

from .model import ModelConfig


class MLPWorldModel(nn.Module):
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

        # Positional encoding (same as transformer model)
        self.triple_pos_emb = nn.Embedding(config.max_triples, config.d_model)
        self.role_emb = nn.Embedding(3, config.d_model)

        # MLP: per-position (no cross-position interaction)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        triple_idx = torch.arange(self.config.max_triples, device=input_ids.device).repeat_interleave(3)
        role_idx = torch.arange(3, device=input_ids.device).repeat(self.config.max_triples)
        pos_enc = (self.triple_pos_emb(triple_idx) + self.role_emb(role_idx)).unsqueeze(0)

        input_emb = self.embed_proj(self.token_emb(input_ids))
        x = input_emb + pos_enc

        # MLP processes each position independently (no attention)
        x = self.mlp(x)
        x = self.ln(x)

        # Input residual (same as transformer model)
        x = x + input_emb

        return self.head(x)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids).argmax(dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
