"""Text Compressor: free text BPE tokens → triple-level 256d bottleneck vectors.

Takes a variable-length BPE text sequence and produces the same bottleneck
format as TripleCompressor: (B, N_triples × 3, d_model).

The text has no explicit triple structure. Learned extraction queries with
role conditioning cross-attend to the contextualized text to discover and
extract triple-structured information.
"""

import torch
import torch.nn as nn


class TextCompressor(nn.Module):
    """Compresses free text into triple-level bottleneck vectors.

    Architecture:
        frozen_token_emb → + text_pos_emb → self_attention (4L)
        → learned extraction queries (N*3) cross-attend to text
        → (B, N*3, d_model) bottleneck vectors
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_triples: int = 8,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_triples = max_triples
        self.max_text_tokens = max_text_tokens

        # Shared frozen embedding (same table as compressor/expander)
        self.token_emb = token_emb

        # Text position embeddings
        self.text_pos_emb = nn.Embedding(max_text_tokens, d_model)

        # Self-attention to contextualize the full text
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learned extraction queries — one per triple position (max_triples * 3)
        self.extract_queries = nn.Parameter(
            torch.randn(max_triples * 3, d_model) * 0.02
        )

        # Cross-attention: queries extract triple-structured info from text
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )

        # Query self-attention: slots coordinate after extraction to avoid
        # redundancy (two entity queries attending to the same text region)
        # and ensure coverage (some text going unattended)
        self.query_self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Role and triple position encoding for extraction queries
        self.role_emb = nn.Embedding(3, d_model)
        self.triple_pos_emb = nn.Embedding(max_triples, d_model)

        # Output normalization
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_pad_mask: torch.Tensor,
        n_triples: int,
    ) -> torch.Tensor:
        """Compress text to triple-level bottleneck vectors.

        Args:
            text_token_ids: (B, T) BPE token IDs
            text_pad_mask: (B, T) True where padding
            n_triples: number of triples to extract (max_triples during training)

        Returns:
            (B, n_triples * 3, d_model) — same shape as TripleCompressor output
        """
        B = text_token_ids.shape[0]
        T = text_token_ids.shape[1]
        device = text_token_ids.device

        # Embed text tokens + position
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(text_token_ids) + self.text_pos_emb(pos_ids)

        # Contextualize with self-attention
        x = self.self_attn(x, src_key_padding_mask=text_pad_mask)

        # Build extraction queries with role + triple position encoding
        n_queries = n_triples * 3
        queries = self.extract_queries[:n_queries].unsqueeze(0).expand(B, -1, -1)

        # Add structural encoding to queries
        triple_idx = torch.arange(n_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(n_triples)
        struct_enc = self.role_emb(role_idx) + self.triple_pos_emb(triple_idx)
        queries = queries + struct_enc.unsqueeze(0)

        # Cross-attend: queries extract from text independently
        extracted, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=text_pad_mask,
        )

        # Query self-attention: slots coordinate to reduce redundancy
        # and improve coverage. No internal LayerNorms — preserves magnitude
        # information from extraction (strong vs weak matches). Only out_ln
        # at the interface boundary normalizes for downstream consumers.
        sa_out, _ = self.query_self_attn(extracted, extracted, extracted)
        extracted = extracted + sa_out
        extracted = extracted + self.query_ffn(extracted)

        # Pad to max_triples * 3 if n_triples < max_triples
        if n_triples < self.max_triples:
            pad_size = (self.max_triples - n_triples) * 3
            pad = torch.zeros(B, pad_size, self.d_model, device=device)
            extracted = torch.cat([extracted, pad], dim=1)

        return self.out_ln(extracted)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
