"""Dual-memory autoregressive decoder for TWM.

Separates dense (compressor) and sparse (dynamics) information channels
with distinct cross-attention heads. The decoder learns which channel
to consult at each generation step:

  Dense channel (compressor output):
    - Token-grounded, entity-level detail
    - Carries spelling, exact phrasing, surface forms
    - "white onion", "counter", "kitchen"
    - Dropped out during training to prevent copying

  Sparse channel (dynamics output):
    - Structural transformation signal
    - Carries what transition happened
    - "take X from Y", "room change", "score up"
    - Always available

Architecture per decoder layer:
    1. Causal self-attention (output tokens coordinate)
    2. Cross-attention to SPARSE memory (dynamics output, always on)
    3. Cross-attention to DENSE memory (compressor output, dropout during training)
    4. FFN

Both cross-attention heads use NO positional encoding on K/V
to preserve the dynamics core's set invariance.

The dense dropout forces the decoder to learn from dynamics when the
dense channel is unavailable, while allowing it to ground entities
when the dense channel is present. This is analogous to skip
connections with stochastic depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCrossAttentionLayer(nn.Module):
    """Transformer decoder layer with two separate cross-attention heads.

    1. Causal self-attention
    2. Cross-attention to sparse (dynamics) memory — always on
    3. Cross-attention to dense (compressor) memory — dropped during training
    4. FFN
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Sparse cross-attention (dynamics — structural transform)
        self.sparse_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Dense cross-attention (compressor — entity grounding)
        self.dense_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Learnable gate for dense channel contribution
        # Initialized to 0.5 so both channels contribute equally at start
        self.dense_gate = nn.Parameter(torch.tensor(0.5))

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x, sparse_memory, dense_memory, causal_mask=None,
                tgt_key_padding_mask=None, dense_available=True):
        """
        Args:
            x: (B, T, d) decoder hidden state
            sparse_memory: (B, N*3, d) dynamics output — always used
            dense_memory: (B, N*3, d) compressor output — dropped during training
            causal_mask: (T, T) causal attention mask
            tgt_key_padding_mask: (B, T) padding mask
            dense_available: whether the dense channel is active this step
        """
        # 1. Causal self-attention (pre-norm)
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + sa_out

        # 2. Sparse cross-attention (dynamics — always on)
        x_norm = self.norm2(x)
        sparse_out, _ = self.sparse_attn(x_norm, sparse_memory, sparse_memory)
        x = x + sparse_out

        # 3. Dense cross-attention (compressor — conditional)
        if dense_available and dense_memory is not None:
            x_norm = self.norm3(x)
            dense_out, _ = self.dense_attn(x_norm, dense_memory, dense_memory)
            x = x + self.dense_gate * dense_out

        # 4. FFN (pre-norm)
        x_norm = self.norm4(x)
        x = x + self.ffn(x_norm)

        return x


class DualARDecoder(nn.Module):
    """AR decoder with separate dense and sparse memory channels.

    Dense (compressor): entity grounding, token-level detail.
    Sparse (dynamics): structural transformation signal.
    Dense channel dropout during training prevents copy shortcut.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_text_tokens: int = 128,
        dropout: float = 0.1,
        bottleneck_dim: int | None = None,
        pad_id: int = 0,
        dense_dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.pad_id = pad_id
        self.dense_dropout = dense_dropout
        bn_d = bottleneck_dim or d_model

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding for output tokens only
        self.pos_emb = nn.Embedding(max_text_tokens + 1, d_model)

        # Sparse memory projection (dynamics output)
        # No positional encoding — set invariance
        self.sparse_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Dense memory projection (compressor output)
        # Separate projection so the two channels live in different subspaces
        # No positional encoding — set invariance
        self.dense_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Dual cross-attention decoder layers
        self.layers = nn.ModuleList([
            DualCrossAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # BOS embedding
        self.bos_emb = nn.Parameter(torch.randn(d_model) * 0.02)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        dynamics_out: torch.Tensor,
        compressor_out: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward.

        Args:
            dynamics_out: (B, N*3, d) post-dynamics latent (sparse channel)
            compressor_out: (B, N*3, d) compressor output (dense channel)
            target_ids: (B, T) target token IDs
            target_pad_mask: (B, T) True where padding

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = target_ids.shape
        device = target_ids.device

        # Build memories — separate projections, no pos encoding
        sparse_memory = self.sparse_proj(dynamics_out)
        dense_memory = self.dense_proj(compressor_out)

        # Dense channel dropout during training
        if self.training and self.dense_dropout > 0:
            drop_mask = torch.rand(B, device=device) < self.dense_dropout
            if drop_mask.all():
                dense_available = False
            elif drop_mask.any():
                # Zero out dense memory for dropped examples
                dense_memory = dense_memory.clone()
                dense_memory[drop_mask] = 0.0
                dense_available = True
            else:
                dense_available = True
        else:
            dense_available = True

        # Decoder input: BOS + target[:-1]
        bos = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tok_emb = self.token_emb(target_ids[:, :-1])
        decoder_input = torch.cat([bos, tok_emb], dim=1)

        # Positional encoding on output tokens only
        pos = self.pos_emb(torch.arange(T, device=device))
        x = decoder_input + pos

        # Causal mask
        causal_mask = self._causal_mask(T, device)

        # Run dual cross-attention layers
        for layer in self.layers:
            x = layer(
                x, sparse_memory, dense_memory,
                causal_mask=causal_mask,
                tgt_key_padding_mask=target_pad_mask,
                dense_available=dense_available,
            )

        return self.output_proj(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        dynamics_out: torch.Tensor,
        compressor_out: torch.Tensor,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive generation with both memory channels active.

        At inference, the dense channel is always on (no dropout).
        """
        B = dynamics_out.shape[0]
        device = dynamics_out.device
        T = max_tokens or self.max_text_tokens

        sparse_memory = self.sparse_proj(dynamics_out)
        dense_memory = self.dense_proj(compressor_out)

        current_emb = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        generated = []

        for t in range(T):
            seq_len = current_emb.shape[1]
            pos = self.pos_emb(torch.arange(seq_len, device=device))
            x = current_emb + pos
            causal_mask = self._causal_mask(seq_len, device)

            for layer in self.layers:
                x = layer(x, sparse_memory, dense_memory,
                          causal_mask=causal_mask, dense_available=True)

            logits = self.output_proj(self.ln_f(x[:, -1:]))

            if temperature == 0:
                next_id = logits.argmax(-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs.squeeze(1), 1)

            generated.append(next_id.squeeze(-1))

            # Stop conditions
            if (next_id.squeeze(-1) == self.pad_id).all():
                break

            if len(generated) >= 10:
                last = torch.stack(generated, dim=1)
                stopped = False
                for plen in [2, 3, 5, 8]:
                    if last.shape[1] >= plen * 3:
                        tail = last[:, -plen * 3:]
                        p1, p2, p3 = tail[:, :plen], tail[:, plen:plen*2], tail[:, plen*2:]
                        if (p1 == p2).all(-1).all() and (p2 == p3).all(-1).all():
                            stopped = True
                            break
                if stopped:
                    break

            next_emb = self.token_emb(next_id)
            current_emb = torch.cat([current_emb, next_emb], dim=1)

        if generated:
            return torch.stack(generated, dim=1)
        return torch.zeros(B, 1, dtype=torch.long, device=device)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
