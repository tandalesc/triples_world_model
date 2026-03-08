"""Compressor: BPE token sequences → single 256d vector per triple slot.

Per slot (entity, attr, or value): takes S=12 BPE tokens, runs self-attention
for intra-slot contextualization, then role-conditioned cross-attention pooling
to produce one vector.

Output shape matches SentenceEncoder: (B, max_triples * 3, d_model).
The dynamics core sees no difference.
"""

import torch
import torch.nn as nn


class TripleCompressor(nn.Module):
    """Compresses BPE token sequences to single vectors per triple slot.

    Architecture per slot:
        frozen_token_emb → + token_position_emb → self_attention (2L) → role_pool → 256d
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_roles: int = 3,
        max_seq_len: int = 12,
        max_triples: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_roles = n_roles
        self.max_seq_len = max_seq_len
        self.max_triples = max_triples

        # Shared frozen embedding (same table as decoder/expander)
        self.token_emb = token_emb

        # Token position within slot: 0..S-1
        self.token_pos_emb = nn.Embedding(max_seq_len, d_model)

        # Self-attention for intra-slot contextualization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Role-conditioned pool: one learned query per role + cross-attention
        self.pool_queries = nn.Embedding(n_roles, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.pool_ln = nn.LayerNorm(d_model)

        # Output projection: bring compressor output to unit-norm scale
        # matching the embedding space the decoder expects
        self.out_ln = nn.LayerNorm(d_model)

        # Triple-level positional encoding (same scheme as SentenceEncoder)
        self.triple_pos_emb = nn.Embedding(max_triples, d_model)
        self.role_pos_emb = nn.Embedding(n_roles, d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_pad_mask: torch.Tensor,
        triple_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compress BPE tokens to triple-level vectors.

        Args:
            token_ids: (B, M, 3, S) BPE token IDs per slot
            token_pad_mask: (B, M, 3, S) True where token is padding
            triple_pad_mask: (B, M) True where triple is padding

        Returns:
            (B, M*3, d_model) — one vector per triple position,
            same shape as SentenceEncoder output
        """
        B, M, R, S = token_ids.shape
        D = self.d_model
        device = token_ids.device

        # Identify valid (non-pad) slots: (B, M) -> (B, M, 3) -> (B*M*3,)
        slot_valid = ~triple_pad_mask.unsqueeze(-1).expand(B, M, R).reshape(B * M * R)
        n_valid = slot_valid.sum().item()

        # Output buffer: zeros for pad slots
        out = torch.zeros(B, M * R, D, device=device)

        if n_valid == 0:
            # All padding -- return zeros + positional encoding
            triple_idx = torch.arange(M, device=device).repeat_interleave(R)
            role_idx = torch.arange(R, device=device).repeat(M)
            pos_enc = (self.triple_pos_emb(triple_idx) + self.role_pos_emb(role_idx)).unsqueeze(0)
            return out + pos_enc

        # Flatten and select only valid slots
        ids_flat = token_ids.reshape(B * M * R, S)
        mask_flat = token_pad_mask.reshape(B * M * R, S)

        ids_valid = ids_flat[slot_valid]    # (n_valid, S)
        mask_valid = mask_flat[slot_valid]  # (n_valid, S)

        # Embed + token position
        x = self.token_emb(ids_valid)  # (n_valid, S, D)
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        x = x + self.token_pos_emb(pos_ids)

        # Self-attention contextualization
        x = self.self_attn(x, src_key_padding_mask=mask_valid)  # (n_valid, S, D)

        # Role-conditioned pool
        # Role IDs for valid slots: [0,1,2,0,1,2,...] but only keep valid ones
        all_role_ids = torch.arange(R, device=device).repeat(B * M)  # (B*M*3,)
        role_ids_valid = all_role_ids[slot_valid]  # (n_valid,)
        queries = self.pool_queries(role_ids_valid).unsqueeze(1)  # (n_valid, 1, D)

        pooled, _ = self.cross_attn(
            query=queries, key=x, value=x,
            key_padding_mask=mask_valid,
        )  # (n_valid, 1, D)
        pooled = self.pool_ln(pooled.squeeze(1))  # (n_valid, D)

        # Scatter valid results back into full output
        out_flat = out.reshape(B * M * R, D)
        out_flat[slot_valid] = pooled
        out = out_flat.reshape(B, M * R, D)

        # Normalize to match embedding scale
        out = self.out_ln(out)

        # Add triple-level positional encoding
        triple_idx = torch.arange(M, device=device).repeat_interleave(R)
        role_idx = torch.arange(R, device=device).repeat(M)
        pos_enc = (self.triple_pos_emb(triple_idx) + self.role_pos_emb(role_idx)).unsqueeze(0)
        out = out + pos_enc

        return out

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
