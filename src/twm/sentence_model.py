"""Sentence-level Triple World Model.

Composes SentenceEncoder + TransformerDynamics + SentenceDecoder for
free-text triple transitions. Uses cosine similarity loss instead of
cross-entropy since outputs are continuous embeddings, not token logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .modules import TransformerDynamics
from .sentence_encoder import SentenceEncoder, SentenceDecoder, PhraseBank


class SentenceTripleWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, st_dim: int):
        super().__init__()
        self.config = config
        self.st_dim = st_dim

        self.encoder = SentenceEncoder(config, st_dim)
        self.dynamics = TransformerDynamics(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.decoder = SentenceDecoder(config, st_dim)

    def forward(
        self, input_embeds: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            input_embeds: (B, T, st_dim) sentence-transformer embeddings
            pad_mask: (B, T) True where padded

        Returns:
            output_embeds: (B, T, st_dim) predicted next-state embeddings
        """
        latent, raw_emb = self.encoder(input_embeds)
        latent = self.dynamics(latent, src_key_padding_mask=pad_mask)
        return self.decoder(latent, skip=raw_emb)

    @torch.no_grad()
    def predict(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning predicted ST-space embeddings."""
        return self.forward(input_embeds, pad_mask)

    def load_dynamics_from_checkpoint(self, checkpoint_path: str, legacy: bool = True):
        """Load dynamics weights from a pretrained TWM checkpoint.

        Args:
            checkpoint_path: path to .pt file
            legacy: if True, remap flat 'encoder.*' keys to 'encoder.*'
        """
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        dynamics_sd = {}
        for key, val in sd.items():
            if legacy and key.startswith("encoder."):
                # Legacy flat format: encoder.layers.* -> encoder.layers.*
                dynamics_sd[key] = val
            elif key.startswith("dynamics.encoder."):
                # New nested format: dynamics.encoder.* -> encoder.*
                new_key = key.removeprefix("dynamics.")
                dynamics_sd[new_key] = val

        self.dynamics.load_state_dict(dynamics_sd, strict=True)

    def freeze_dynamics(self):
        """Freeze dynamics weights — only train encoder/decoder projections."""
        for param in self.dynamics.parameters():
            param.requires_grad = False

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def cosine_embedding_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine similarity loss for sentence-level predictions.

    Args:
        pred: (B, T, st_dim) predicted embeddings
        target: (B, T, st_dim) target embeddings
        pad_mask: (B, T) True where padded — these positions are excluded

    Returns:
        scalar loss (1 - mean cosine similarity over non-pad positions)
    """
    # Normalize
    pred_norm = F.normalize(pred, dim=-1)
    tgt_norm = F.normalize(target, dim=-1)

    # Per-position cosine similarity
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, T)

    if pad_mask is not None:
        # Zero out pad positions
        cos_sim = cos_sim.masked_fill(pad_mask, 0.0)
        n_valid = (~pad_mask).sum().clamp(min=1)
        return 1.0 - cos_sim.sum() / n_valid
    else:
        return 1.0 - cos_sim.mean()
