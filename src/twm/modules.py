"""Decomposed TWM components: Encoder, Dynamics, Decoder.

The triple world model is split into three reusable pieces:
- TripleEncoder: tokens -> latent vectors (embedding + positional encoding)
- TransformerDynamics: latent_t -> latent_t+1 (pure transformer, no token knowledge)
- TripleDecoder: latent vectors -> logits (LayerNorm + output heads + input residual)

This separation lets you swap the encoder (e.g., sentence-transformer, image encoder,
sensor encoder) while reusing the same dynamics core.
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class TripleEncoder(nn.Module):
    """Maps token IDs to positionally-encoded latent vectors.

    Returns both the full encoded representation (embedding + position)
    and the raw embedding (for the input residual skip connection).
    """

    def __init__(self, config: ModelConfig, pretrained_embeds: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        if config.use_split_embeddings:
            self.entity_emb = nn.Embedding(config.n_entities, config.d_model, padding_idx=0)
            self.attr_emb = nn.Embedding(config.n_attrs, config.d_model, padding_idx=0)
            self.value_emb = nn.Embedding(config.n_values, config.d_model, padding_idx=0)
            self.embed_proj = nn.Identity()
        else:
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
        self.role_emb = nn.Embedding(3, config.d_model)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.use_split_embeddings:
            return self.embed_proj(self.token_emb(input_ids))

        B, T = input_ids.shape
        role_pattern = torch.tensor([0, 1, 2], device=input_ids.device).repeat(T // 3)
        out = torch.zeros(B, T, self.config.d_model, device=input_ids.device)

        ent_mask = role_pattern == 0
        attr_mask = role_pattern == 1
        val_mask = role_pattern == 2

        out[:, ent_mask] = self.entity_emb(input_ids[:, ent_mask])
        out[:, attr_mask] = self.attr_emb(input_ids[:, attr_mask])
        out[:, val_mask] = self.value_emb(input_ids[:, val_mask])
        return out

    def _build_position_encoding(self, device: torch.device) -> torch.Tensor:
        triple_idx = torch.arange(self.config.max_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(self.config.max_triples)
        return (self.triple_pos_emb(triple_idx) + self.role_emb(role_idx)).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent: (B, T, d_model) — embedding + positional encoding, ready for dynamics
            raw_emb: (B, T, d_model) — raw token embedding, for input residual
        """
        raw_emb = self._embed_tokens(input_ids)
        pos_enc = self._build_position_encoding(input_ids.device)
        return raw_emb + pos_enc, raw_emb


class TransformerDynamics(nn.Module):
    """Pure latent-space transformer: latent_t -> latent_t+1.

    No knowledge of tokens, triples, or vocabularies. Any encoder that produces
    (B, T, d_model) vectors can feed into this.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def extract_attention_weights(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Extract per-layer attention weights for visualization.

        Returns list of (n_heads, T, T) tensors, one per layer.
        """
        attn_weights = []
        for layer in self.encoder.layers:
            x_norm = layer.norm1(x)
            attn_out, weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=pad_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + attn_out
            x = x + layer._ff_block(layer.norm2(x))
            attn_weights.append(weights.detach().cpu())
        return attn_weights


class TripleDecoder(nn.Module):
    """Maps latent vectors back to token logits.

    Applies LayerNorm, optional input residual skip connection,
    and projects to vocabulary logits (shared or split per role).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ln_f = nn.LayerNorm(config.d_model)

        if config.use_split_embeddings:
            self.entity_head = nn.Linear(config.d_model, config.n_entities)
            self.attr_head = nn.Linear(config.d_model, config.n_attrs)
            self.value_head = nn.Linear(config.d_model, config.n_values)
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, latent: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            latent: (B, T, d_model) output from dynamics
            skip: (B, T, d_model) optional residual from encoder (raw embeddings)
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.ln_f(latent)
        if skip is not None:
            x = x + skip
        return self._output_logits(x)

    def _output_logits(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.use_split_embeddings:
            return self.head(x)

        B, T, D = x.shape
        max_v = max(self.config.n_entities, self.config.n_attrs, self.config.n_values)
        logits = torch.full((B, T, max_v), float('-inf'), device=x.device)

        role_pattern = torch.tensor([0, 1, 2], device=x.device).repeat(T // 3)

        ent_mask = role_pattern == 0
        attr_mask = role_pattern == 1
        val_mask = role_pattern == 2

        logits[:, ent_mask, :self.config.n_entities] = self.entity_head(x[:, ent_mask])
        logits[:, attr_mask, :self.config.n_attrs] = self.attr_head(x[:, attr_mask])
        logits[:, val_mask, :self.config.n_values] = self.value_head(x[:, val_mask])

        return logits
