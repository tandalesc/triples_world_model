"""Autoregressive decoder for TWM latent-to-text generation.

Cross-attends into frozen dynamics latents (set of triple-level vectors)
and generates token sequences left-to-right with causal self-attention.

Key design constraint: NO positional encoding on cross-attention K/V.
The dynamics core is set-based (permutation-invariant over triple positions).
Adding positional encoding to latent positions would break this invariance.
The decoder attends to latent positions by CONTENT only — it learns to
retrieve what it needs based on semantic similarity, not position.

Positional encoding is used ONLY in the decoder's causal self-attention
over output tokens, where left-to-right ordering is meaningful.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARDecoder(nn.Module):
    """Autoregressive transformer decoder over frozen latent memory.

    Architecture:
        latent (B, N*3, d) → memory K/V (NO positional encoding)
        BOS + tokens → causal self-attention (WITH positional encoding)
                     → cross-attention to memory
                     → linear → vocab logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_text_tokens: int = 128,
        dropout: float = 0.1,
        bottleneck_dim: int | None = None,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.pad_id = pad_id
        bn_d = bottleneck_dim or d_model

        # Token embedding for decoder input
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding for OUTPUT token positions only.
        # NOT applied to latent memory — intentional for set invariance.
        self.pos_emb = nn.Embedding(max_text_tokens + 1, d_model)  # +1 for BOS

        # Project latent to memory for cross-attention.
        # No positional encoding added — cross-attention retrieves by content only.
        # This preserves the dynamics core's permutation invariance over triple positions.
        self.memory_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # BOS embedding (learned)
        self.bos_emb = nn.Parameter(torch.randn(d_model) * 0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _build_memory(self, bottleneck, pre_dynamics=None):
        """Build cross-attention memory from latent. No positional encoding."""
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
        """Teacher-forced forward pass.

        Args:
            bottleneck: (B, N*3, d) frozen latent
            target_ids: (B, T) target token IDs
            target_pad_mask: (B, T) True where padding
            pre_dynamics: (B, N*3, d) optional pre-dynamics for multi-level

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = target_ids.shape
        device = target_ids.device

        # Memory from latent — no positional encoding (set invariance)
        memory = self._build_memory(bottleneck, pre_dynamics)

        # Decoder input: BOS + target[:-1] (shifted right)
        bos = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, d)
        tok_emb = self.token_emb(target_ids[:, :-1])  # (B, T-1, d)
        decoder_input = torch.cat([bos, tok_emb], dim=1)  # (B, T, d)

        # Add positional encoding to decoder tokens ONLY
        pos = self.pos_emb(torch.arange(T, device=device))
        decoder_input = decoder_input + pos

        # Causal mask for autoregressive decoding
        causal_mask = self._causal_mask(T, device)

        # Decode with cross-attention to latent memory
        out = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_pad_mask,
        )

        return self.output_proj(self.ln_f(out))

    @torch.no_grad()
    def generate(
        self,
        bottleneck: torch.Tensor,
        max_tokens: int | None = None,
        pre_dynamics: torch.Tensor | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive greedy generation.

        Args:
            bottleneck: (B, N*3, d) frozen latent
            max_tokens: max output length
            pre_dynamics: (B, N*3, d) optional
            temperature: 0 = greedy (default)

        Returns:
            (B, T) generated token IDs
        """
        B = bottleneck.shape[0]
        device = bottleneck.device
        T = max_tokens or self.max_text_tokens

        memory = self._build_memory(bottleneck, pre_dynamics)

        # Start with BOS
        current_emb = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        generated = []

        for t in range(T):
            seq_len = current_emb.shape[1]
            pos = self.pos_emb(torch.arange(seq_len, device=device))
            decoder_input = current_emb + pos

            causal_mask = self._causal_mask(seq_len, device)

            out = self.decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=causal_mask,
            )

            logits = self.output_proj(self.ln_f(out[:, -1:]))  # (B, 1, V)

            if temperature == 0:
                next_id = logits.argmax(-1)  # (B, 1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs.squeeze(1), 1)

            generated.append(next_id.squeeze(-1))  # (B,)

            # Stop if all sequences have hit pad
            if (next_id.squeeze(-1) == self.pad_id).all():
                break

            # Append token embedding for next step
            next_emb = self.token_emb(next_id)  # (B, 1, d)
            current_emb = torch.cat([current_emb, next_emb], dim=1)

        # Stack and pad to consistent length
        if generated:
            result = torch.stack(generated, dim=1)  # (B, generated_len)
        else:
            result = torch.zeros(B, 1, dtype=torch.long, device=device)

        return result

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
