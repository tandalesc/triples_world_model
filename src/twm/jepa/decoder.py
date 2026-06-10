"""JEPA v2 token decoder — the grounding loss (design doc §4).

A thin adapter around the repo's `ARDecoder` (`src/twm/ar_decoder.py`). The decoder
cross-attends a single memory channel: the operator-transformed slots `a* = B_v k`
of width `d_noun`. It generates `text_{t+1}` tokens left-to-right with causal
self-attention and teacher forcing.

WHY this is the v2 grounding mechanism: token CE is the primary loss. The decoder's
ONLY conditioning on the future is `a*`, and since `a* = B_v k` with `k` a function of
`text_t` alone, the only `text_{t+1}` info reaching the decoder is the discrete action
`v`'s ~log2(V) bits. This forces `v` to carry the causal-step identity (design doc §6).

LEAKAGE CONTRACT (design doc §6, L1/L2): the constructor has NO posterior /
`text_{t+1}`-encoding argument, and `forward`/`generate` accept ONLY `a_star`
(+ teacher-forced target prefix). The decoder must never see raw `text_{t+1}`
encodings, posterior features, untransformed `k`, or `slots_t`.

DECIDED (design doc §4.1): single-memory `ARDecoder`, NOT `DualARDecoder` (its dense
channel would bypass `v` with raw `text_t`/`k` info — a leak — and costs more params).
"""

import torch
import torch.nn as nn

from twm.ar_decoder import ARDecoder


class TokenDecoder(nn.Module):
    """v2 token decoder adapter over `ARDecoder` (single memory = `a*` slots).

    Memory is `a*` of width `d_noun`; `ARDecoder.memory_proj` maps `d_noun -> d_dec`,
    so `bottleneck_dim` is set to `d_noun`. No positional encoding on the memory —
    `a*` is a permutation-invariant slot set, which `ARDecoder` already enforces.
    """

    def __init__(
        self,
        vocab_size: int,
        d_dec: int = 64,
        n_layers: int = 1,
        n_heads: int = 4,
        d_ff: int = 128,
        d_noun: int = 32,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_dec = d_dec
        self.d_noun = d_noun
        self.max_text_tokens = max_text_tokens
        self.pad_id = pad_id

        # The decoder's own width (d_dec) is decoupled from the encoder d_model so we
        # can hit the nano-v2 param budget (design doc §4.2). bottleneck_dim = d_noun
        # because the memory is a* of width d_noun.
        self.decoder = ARDecoder(
            vocab_size=vocab_size,
            d_model=d_dec,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_text_tokens=max_text_tokens,
            dropout=dropout,
            bottleneck_dim=d_noun,
            pad_id=pad_id,
        )

    def forward(
        self,
        a_star: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward.

        Args:
            a_star: (B, M, d_noun) operator-transformed slot memory. The ONLY future
                    conditioning channel (leakage contract §6).
            tgt_ids: (B, T) target token IDs for text_{t+1} (no leading BOS; ARDecoder
                     prepends its learned bos_emb internally).
            tgt_pad: (B, T) True where padding.

        Returns:
            logits: (B, T, vocab_size). Position t predicts tgt_ids[t] (the shift is
                    baked into ARDecoder's [bos] + tgt[:-1] input). CE is therefore
                    cross_entropy(logits.reshape(-1, V), tgt_ids.reshape(-1),
                    ignore_index=pad_id) with no manual shift (design doc §4.3).
        """
        return self.decoder(
            bottleneck=a_star,
            target_ids=tgt_ids,
            target_pad_mask=tgt_pad,
        )

    @torch.no_grad()
    def generate(
        self,
        a_star: torch.Tensor,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressive generation conditioned on `a*` only.

        Args:
            a_star: (B, M, d_noun) operator-transformed slot memory.
            max_tokens: max output length (defaults to max_text_tokens).
            temperature: 0.0 = greedy (default); > 0 = temperature sampling for the
                         diagnostics text samples (design doc §8).

        Returns:
            (B, T) generated token IDs (excluding BOS).
        """
        return self.decoder.generate(
            bottleneck=a_star,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
