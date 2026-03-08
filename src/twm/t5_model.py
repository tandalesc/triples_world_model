"""Hybrid TWM with T5 value decoder.

Entity/attr: discrete seq2seq heads (small vocab, fast convergence)
Value: frozen T5 decoder generates free-text from projected TWM embeddings

For each triple, the full context (entity + attr + value positions from
the dynamics output) is concatenated and projected into a multi-token
sequence in T5 space. This gives T5 multiple cross-attention keys to
attend to, and each token can specialize on different aspects of the
prediction (semantic category, specificity, sentiment, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .modules import TransformerDynamics
from .sentence_encoder import SentenceEncoder
from .t5_decoder import T5ValueDecoder
from .phrase_vocab import PhraseVocab


class HybridT5WorldModel(nn.Module):
    """TWM with discrete entity/attr heads and T5-based value decoder."""

    def __init__(
        self,
        config: ModelConfig,
        st_dim: int,
        vocab: PhraseVocab,
        t5_model_name: str = "t5-small",
        n_proj_tokens: int = 8,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.config = config
        self.st_dim = st_dim
        self.vocab = vocab

        sizes = vocab.vocab_sizes
        self.encoder = SentenceEncoder(config, st_dim)
        self.dynamics = TransformerDynamics(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Discrete heads for entity/attr
        self.ln_f = nn.LayerNorm(config.d_model)
        self.entity_head = nn.Linear(config.d_model, sizes["entity"])
        self.attr_head = nn.Linear(config.d_model, sizes["attr"])

        # T5 decoder for values — receives full triple context
        self.value_decoder = T5ValueDecoder(
            twm_dim=config.d_model,
            t5_model_name=t5_model_name,
            n_proj_tokens=n_proj_tokens,
            unfreeze_last_n=unfreeze_last_n,
        )

    def encode_dynamics(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run encoder + dynamics, return latent output."""
        latent, _raw = self.encoder(input_embeds)
        return self.dynamics(latent, src_key_padding_mask=pad_mask)

    def _extract_triple_context(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate (entity, attr, value) for each triple.

        Args:
            latent: (B, T, d_model) dynamics output where T = max_triples * 3

        Returns:
            (B, M, 3*d_model) concatenated entity+attr+value per triple
        """
        M = self.config.max_triples
        entity_idx = torch.arange(0, M * 3, 3, device=latent.device)
        attr_idx = torch.arange(1, M * 3, 3, device=latent.device)
        value_idx = torch.arange(2, M * 3, 3, device=latent.device)

        entity_emb = latent[:, entity_idx]  # (B, M, d_model)
        attr_emb = latent[:, attr_idx]
        value_emb = latent[:, value_idx]

        return torch.cat([entity_emb, attr_emb, value_emb], dim=-1)  # (B, M, 3*d_model)

    def forward_discrete(
        self,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Entity/attr logits from dynamics output."""
        x = self.ln_f(latent)
        M = self.config.max_triples
        entity_idx = torch.arange(0, M * 3, 3, device=latent.device)
        attr_idx = torch.arange(1, M * 3, 3, device=latent.device)

        return {
            "entity": self.entity_head(x[:, entity_idx]),
            "attr": self.attr_head(x[:, attr_idx]),
        }

    def forward_value(
        self,
        latent: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """T5 value decoder with full triple context.

        Args:
            latent: (B, T, d_model) dynamics output
            target_ids: (B*M, seq_len) tokenized target values (flattened)

        Returns:
            logits: (n_valid, seq_len, t5_vocab_size)
        """
        triple_ctx = self._extract_triple_context(latent)  # (B, M, 3*d_model)
        B, M, D = triple_ctx.shape
        ctx_flat = triple_ctx.reshape(B * M, D)

        return self.value_decoder(ctx_flat, target_ids, target_attention_mask)

    @torch.no_grad()
    def generate_values(
        self,
        latent: torch.Tensor,
        max_length: int = 32,
    ) -> list[list[str]]:
        """Generate value texts for all triple slots."""
        triple_ctx = self._extract_triple_context(latent)
        B, M, D = triple_ctx.shape
        ctx_flat = triple_ctx.reshape(B * M, D)
        texts = self.value_decoder.generate(ctx_flat, max_length=max_length)
        return [texts[i * M:(i + 1) * M] for i in range(B)]

    def load_dynamics_from_checkpoint(self, checkpoint_path: str, legacy: bool = True):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        dynamics_sd = {}
        for key, val in sd.items():
            if legacy and key.startswith("encoder."):
                dynamics_sd[key] = val
            elif key.startswith("dynamics.encoder."):
                new_key = key.removeprefix("dynamics.")
                dynamics_sd[new_key] = val
        self.dynamics.load_state_dict(dynamics_sd, strict=True)

    def load_encoder_from_sentence_model(self, checkpoint_path: str):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_sd = {
            k.removeprefix("encoder."): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        if encoder_sd:
            self.encoder.load_state_dict(encoder_sd, strict=True)

    def freeze_dynamics(self):
        for param in self.dynamics.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
