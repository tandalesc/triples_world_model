"""Seq2seq Triple World Model.

Composes SentenceEncoder + TransformerDynamics + Seq2SeqDecoder.
Unlike SentenceTripleWorldModel which outputs continuous embeddings
for NN lookup, this outputs per-role logits over phrase vocabularies.
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .modules import TransformerDynamics
from .sentence_encoder import SentenceEncoder
from .seq2seq_decoder import Seq2SeqDecoder
from .phrase_vocab import PhraseVocab


class Seq2SeqTripleWorldModel(nn.Module):
    def __init__(self, config: ModelConfig, st_dim: int, vocab: PhraseVocab):
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
        self.decoder = Seq2SeqDecoder(
            config,
            n_entity_phrases=sizes["entity"],
            n_attr_phrases=sizes["attr"],
            n_value_phrases=sizes["value"],
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_embeds: (B, T, st_dim) sentence-transformer embeddings
            pad_mask: (B, T) True where padded

        Returns:
            dict with per-role logits:
                "entity": (B, max_triples, n_entity_phrases)
                "attr":   (B, max_triples, n_attr_phrases)
                "value":  (B, max_triples, n_value_phrases)
        """
        latent, _raw_emb = self.encoder(input_embeds)
        latent = self.dynamics(latent, src_key_padding_mask=pad_mask)
        return self.decoder(latent, memory_key_padding_mask=pad_mask)

    @torch.no_grad()
    def predict_ids(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy decode to phrase IDs.

        Returns:
            (B, max_triples, 3) tensor of phrase IDs
        """
        latent, _raw_emb = self.encoder(input_embeds)
        latent = self.dynamics(latent, src_key_padding_mask=pad_mask)
        return self.decoder.decode_greedy(latent, memory_key_padding_mask=pad_mask)

    @torch.no_grad()
    def predict_triples(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> list[list[list[str]]]:
        """Greedy decode to phrase strings, stripping pad triples.

        Returns:
            list of B examples, each a list of [entity, attr, value] triples
        """
        ids = self.predict_ids(input_embeds, pad_mask)  # (B, M, 3)
        B, M, _ = ids.shape
        results = []
        for b in range(B):
            triples = []
            for m in range(M):
                e = self.vocab.decode_id(ids[b, m, 0].item(), "entity")
                a = self.vocab.decode_id(ids[b, m, 1].item(), "attr")
                v = self.vocab.decode_id(ids[b, m, 2].item(), "value")
                if e not in ("<pad>", "<unk>") or a not in ("<pad>", "<unk>"):
                    triples.append([e, a, v])
            results.append(triples)
        return results

    def load_dynamics_from_checkpoint(self, checkpoint_path: str, legacy: bool = True):
        """Load dynamics weights from a pretrained TWM checkpoint."""
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
        """Load encoder weights from a trained SentenceTripleWorldModel checkpoint."""
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
