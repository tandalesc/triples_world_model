"""Triple World Model — composed from Encoder + Dynamics + Decoder.

This is the high-level wrapper that composes the three modules from modules.py.
For most uses, this is the only class you need. Use the modules directly when
swapping in a custom encoder or decoder.
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .modules import TripleEncoder, TransformerDynamics, TripleDecoder

# Keys that belong to the encoder in legacy (pre-refactor) checkpoints
_ENCODER_PREFIXES = (
    "token_emb.", "embed_proj.", "triple_pos_emb.", "role_emb.",
    "entity_emb.", "attr_emb.", "value_emb.",
)
# Keys that belong to the decoder in legacy checkpoints
_DECODER_PREFIXES = (
    "ln_f.", "head.", "entity_head.", "attr_head.", "value_head.",
)


def _remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap flat legacy checkpoint keys to the new nested structure."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            # encoder.layers.* -> dynamics.encoder.layers.*
            remapped[f"dynamics.{key}"] = value
        elif any(key.startswith(p) for p in _DECODER_PREFIXES):
            remapped[f"triple_decoder.{key}"] = value
        elif any(key.startswith(p) for p in _ENCODER_PREFIXES):
            remapped[f"triple_encoder.{key}"] = value
        else:
            # Unknown key — pass through (will error on load if truly unknown)
            remapped[key] = value
    return remapped


def _is_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Check if a state dict uses the old flat key format."""
    return any(k.startswith("encoder.") for k in state_dict) and not any(
        k.startswith("dynamics.") for k in state_dict
    )


class TripleWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained_embeds: torch.Tensor | None = None):
        super().__init__()
        self.config = config

        self.triple_encoder = TripleEncoder(config, pretrained_embeds)
        self.dynamics = TransformerDynamics(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.triple_decoder = TripleDecoder(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, max_triples * 3) token IDs, <pad>=0 for empty slots
        Returns:
            logits: (B, max_triples * 3, vocab_size)
        """
        latent, raw_emb = self.triple_encoder(input_ids)
        pad_mask = input_ids == 0
        latent = self.dynamics(latent, src_key_padding_mask=pad_mask)
        return self.triple_decoder(latent, skip=raw_emb)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return predicted token IDs (greedy argmax)."""
        logits = self.forward(input_ids)
        return logits.argmax(dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict, auto-detecting and remapping legacy format."""
        if _is_legacy_state_dict(state_dict):
            state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
