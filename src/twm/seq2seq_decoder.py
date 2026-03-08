"""Seq2seq decoder for Triple World Model.

Replaces nearest-neighbor SentenceDecoder with a cross-attention
transformer decoder that produces per-role logits over phrase vocabularies.

The decoder cross-attends to the dynamics latent representation and
outputs logits for each position, enabling training with cross-entropy
and other discrete loss functions (DPO, contrastive, etc.).
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class Seq2SeqDecoder(nn.Module):
    """Cross-attention transformer decoder with per-role output heads.

    Architecture:
        - Learned query embeddings (one per output position)
        - N transformer decoder layers cross-attending to dynamics output
        - Per-role linear heads: entity, attr, value

    The decoder receives the dynamics latent as "memory" and produces
    logits over phrase vocabularies at each output position.
    """

    def __init__(
        self,
        config: ModelConfig,
        n_entity_phrases: int,
        n_attr_phrases: int,
        n_value_phrases: int,
        n_decoder_layers: int = 2,
    ):
        super().__init__()
        self.config = config
        self.n_decoder_layers = n_decoder_layers
        T = config.max_triples * 3

        # Learned query embeddings for output positions
        self.query_emb = nn.Embedding(T, config.d_model)
        # Reuse the same triple/role positional encoding scheme
        self.triple_pos_emb = nn.Embedding(config.max_triples, config.d_model)
        self.role_emb = nn.Embedding(3, config.d_model)

        # Cross-attention decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_decoder_layers,
        )
        self.ln_f = nn.LayerNorm(config.d_model)

        # Per-role output heads
        self.entity_head = nn.Linear(config.d_model, n_entity_phrases)
        self.attr_head = nn.Linear(config.d_model, n_attr_phrases)
        self.value_head = nn.Linear(config.d_model, n_value_phrases)

    def _build_queries(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build position-encoded query embeddings for all output slots."""
        T = self.config.max_triples * 3
        pos_idx = torch.arange(T, device=device)
        triple_idx = torch.arange(self.config.max_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(self.config.max_triples)

        queries = (
            self.query_emb(pos_idx)
            + self.triple_pos_emb(triple_idx)
            + self.role_emb(role_idx)
        )
        return queries.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            memory: (B, T, d_model) dynamics output (encoder latent)
            memory_key_padding_mask: (B, T) True where padded in encoder

        Returns:
            dict with per-role logits:
                "entity": (B, n_triples, n_entity_phrases)
                "attr":   (B, n_triples, n_attr_phrases)
                "value":  (B, n_triples, n_value_phrases)
        """
        B = memory.shape[0]
        queries = self._build_queries(B, memory.device)

        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoded = self.ln_f(decoded)  # (B, T, d_model)

        # Split by role: positions 0,3,6,... are entity; 1,4,7,... are attr; 2,5,8,... are value
        T = self.config.max_triples * 3
        role_idx = torch.arange(3, device=memory.device).repeat(self.config.max_triples)

        entity_pos = (role_idx == 0)
        attr_pos = (role_idx == 1)
        value_pos = (role_idx == 2)

        return {
            "entity": self.entity_head(decoded[:, entity_pos]),  # (B, max_triples, n_entity)
            "attr": self.attr_head(decoded[:, attr_pos]),         # (B, max_triples, n_attr)
            "value": self.value_head(decoded[:, value_pos]),      # (B, max_triples, n_value)
        }

    def decode_greedy(self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Greedy decode: return argmax phrase IDs per position.

        Returns:
            (B, max_triples, 3) tensor of phrase IDs
        """
        logits = self.forward(memory, memory_key_padding_mask)
        entity_ids = logits["entity"].argmax(dim=-1)  # (B, max_triples)
        attr_ids = logits["attr"].argmax(dim=-1)
        value_ids = logits["value"].argmax(dim=-1)
        return torch.stack([entity_ids, attr_ids, value_ids], dim=-1)
