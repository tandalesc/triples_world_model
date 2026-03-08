"""Token-level W-space encoder for Triple World Model.

Instead of encoding each phrase as a single sentence-transformer vector,
this encoder BPE-tokenizes each phrase and looks up frozen W-space
embeddings per token. The TWM sees token-level input, preserving
sub-phrase distinctions that sentence-level compression loses.

Input triple (PersonX, xReact, loses their nerve):
  entity: [PersonX]          → 2 BPE tokens → 2 W-space vectors
  attr:   [xReact]           → 2 BPE tokens → 2 W-space vectors
  value:  [loses their nerve] → 3 BPE tokens → 3 W-space vectors

Each position is padded to max_tokens_per_slot, giving a fixed sequence
length of max_triples * 3 * max_tokens_per_slot.

Positional encoding: triple_pos + role_emb + token_pos_emb (3-level).
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class TokenEncoder(nn.Module):
    """Encodes BPE-tokenized triples into TWM latent space using frozen W-space embeddings.

    The token embeddings come from the diffusion decoder's token_emb (frozen,
    W-space initialized). This encoder adds positional structure on top.
    """

    def __init__(self, config: ModelConfig, max_tokens_per_slot: int = 12):
        super().__init__()
        self.config = config
        self.max_tokens_per_slot = max_tokens_per_slot

        # 3-level positional encoding
        self.triple_pos_emb = nn.Embedding(config.max_triples, config.d_model)
        self.role_emb = nn.Embedding(3, config.d_model)  # entity=0, attr=1, value=2
        self.token_pos_emb = nn.Embedding(max_tokens_per_slot, config.d_model)

    def forward(
        self,
        token_embeds: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeds: (B, max_triples * 3 * max_tokens, d_model) W-space embeddings
            pad_mask: (B, max_triples * 3 * max_tokens) True = pad

        Returns:
            latent: (B, T, d_model) — embeddings + positional encoding
            raw_emb: (B, T, d_model) — raw embeddings only
        """
        B, T, D = token_embeds.shape
        device = token_embeds.device
        M = self.config.max_triples
        S = self.max_tokens_per_slot

        # Build 3-level positional encoding
        # For position p: which triple, which role, which token within slot
        triple_idx = torch.arange(M, device=device).repeat_interleave(3 * S)  # (M*3*S,)
        role_idx = torch.arange(3, device=device).repeat_interleave(S).repeat(M)  # (M*3*S,)
        token_idx = torch.arange(S, device=device).repeat(M * 3)  # (M*3*S,)

        pos_enc = (
            self.triple_pos_emb(triple_idx)
            + self.role_emb(role_idx)
            + self.token_pos_emb(token_idx)
        ).unsqueeze(0)  # (1, T, d_model)

        return token_embeds + pos_enc, token_embeds
