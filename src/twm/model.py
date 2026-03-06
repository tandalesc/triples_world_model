"""Triple World Model — set-to-set transformer encoder with input residual."""

import torch
import torch.nn as nn

from .config import ModelConfig


class TripleWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained_embeds: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        if config.use_split_embeddings:
            self._init_split_embeddings(config)
        else:
            self._init_shared_embeddings(config, pretrained_embeds)

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

        if config.use_split_embeddings:
            self.entity_head = nn.Linear(config.d_model, config.n_entities)
            self.attr_head = nn.Linear(config.d_model, config.n_attrs)
            self.value_head = nn.Linear(config.d_model, config.n_values)
        else:
            self.head = nn.Linear(config.d_model, config.vocab_size)

    def _init_shared_embeddings(self, config: ModelConfig, pretrained_embeds: torch.Tensor | None):
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

    def _init_split_embeddings(self, config: ModelConfig):
        self.entity_emb = nn.Embedding(config.n_entities, config.d_model, padding_idx=0)
        self.attr_emb = nn.Embedding(config.n_attrs, config.d_model, padding_idx=0)
        self.value_emb = nn.Embedding(config.n_values, config.d_model, padding_idx=0)
        self.embed_proj = nn.Identity()

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens, routing to correct table if using split embeddings."""
        if not self.config.use_split_embeddings:
            return self.embed_proj(self.token_emb(input_ids))

        B, T = input_ids.shape
        # Build role mask: 0=entity, 1=attr, 2=value repeating
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
        """(1, max_positions, d_model) positional encoding from triple_index + role."""
        triple_idx = torch.arange(self.config.max_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(self.config.max_triples)
        return (self.triple_pos_emb(triple_idx) + self.role_emb(role_idx)).unsqueeze(0)

    def _output_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project to output logits, using split heads if configured."""
        if not self.config.use_split_embeddings:
            return self.head(x)

        B, T, D = x.shape
        # For split heads, we need to produce a unified logits tensor.
        # Use the max vocab size across roles and scatter into it.
        max_v = max(self.config.n_entities, self.config.n_attrs, self.config.n_values)
        logits = torch.full((B, T, max_v), float('-inf'), device=x.device)

        role_pattern = torch.tensor([0, 1, 2], device=x.device).repeat(T // 3)

        ent_mask = role_pattern == 0
        attr_mask = role_pattern == 1
        val_mask = role_pattern == 2

        ent_logits = self.entity_head(x[:, ent_mask])  # (B, n_ent_pos, n_entities)
        attr_logits = self.attr_head(x[:, attr_mask])
        val_logits = self.value_head(x[:, val_mask])

        logits[:, ent_mask, :self.config.n_entities] = ent_logits
        logits[:, attr_mask, :self.config.n_attrs] = attr_logits
        logits[:, val_mask, :self.config.n_values] = val_logits

        return logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, max_triples * 3) token IDs, <pad>=0 for empty slots
        Returns:
            logits: (B, max_triples * 3, vocab_size) or max role vocab size for split
        """
        pos_enc = self._build_position_encoding(input_ids.device)
        input_emb = self._embed_input(input_ids)
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

        return self._output_logits(x)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return predicted token IDs (greedy argmax)."""
        logits = self.forward(input_ids)
        return logits.argmax(dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
