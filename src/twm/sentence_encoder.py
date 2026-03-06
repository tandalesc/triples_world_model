"""Sentence-transformer encoder/decoder for Triple World Model.

Replaces TripleEncoder/TripleDecoder when working with free-text triples
(e.g., ATOMIC 2020) where each triple position contains a phrase rather
than a single token from a fixed vocabulary.

Architecture:
  SentenceEncoder: phrase -> sentence-transformer -> project to d_model -> + positional encoding
  SentenceDecoder: d_model -> project to ST dim -> nearest-neighbor lookup in phrase bank

The TransformerDynamics core is unchanged — it just sees (B, T, d_model) vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import ModelConfig


class SentenceEncoder(nn.Module):
    """Encodes free-text triple phrases into the TWM latent space.

    Each triple position (entity, attr, value) is a phrase encoded by a
    sentence-transformer, then projected to d_model and position-encoded.
    """

    def __init__(self, config: ModelConfig, st_dim: int):
        super().__init__()
        self.config = config
        self.st_dim = st_dim

        # Project sentence-transformer embeddings to model dim
        if st_dim != config.d_model:
            self.proj = nn.Linear(st_dim, config.d_model, bias=False)
        else:
            self.proj = nn.Identity()

        # Same positional encoding scheme as TripleEncoder
        self.triple_pos_emb = nn.Embedding(config.max_triples, config.d_model)
        self.role_emb = nn.Embedding(3, config.d_model)

    def _build_position_encoding(self, device: torch.device) -> torch.Tensor:
        triple_idx = torch.arange(self.config.max_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(self.config.max_triples)
        return (self.triple_pos_emb(triple_idx) + self.role_emb(role_idx)).unsqueeze(0)

    def forward(self, phrase_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            phrase_embeds: (B, max_triples * 3, st_dim) pre-computed ST embeddings

        Returns:
            latent: (B, T, d_model) — projected + position-encoded, ready for dynamics
            raw_emb: (B, T, d_model) — projected embeddings only, for skip connection
        """
        raw_emb = self.proj(phrase_embeds)
        pos_enc = self._build_position_encoding(phrase_embeds.device)
        return raw_emb + pos_enc, raw_emb


class SentenceDecoder(nn.Module):
    """Decodes TWM latent vectors back to phrases via nearest-neighbor lookup.

    Projects dynamics output back to sentence-transformer space, then finds
    the closest phrase in a pre-built phrase bank.
    """

    def __init__(self, config: ModelConfig, st_dim: int):
        super().__init__()
        self.config = config
        self.st_dim = st_dim
        self.ln_f = nn.LayerNorm(config.d_model)

        # Project back to sentence-transformer space
        if st_dim != config.d_model:
            self.proj = nn.Linear(config.d_model, st_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, latent: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            latent: (B, T, d_model) output from dynamics
            skip: (B, T, d_model) optional residual from encoder

        Returns:
            st_space: (B, T, st_dim) vectors in sentence-transformer space
        """
        x = self.ln_f(latent)
        if skip is not None:
            x = x + skip
        return self.proj(x)


class PhraseBank:
    """Indexed collection of phrases with their sentence-transformer embeddings.

    Used by SentenceDecoder for nearest-neighbor retrieval.
    Maintains separate banks per role (entity, attr, value) for constrained decoding.
    """

    def __init__(self):
        self.phrases: dict[str, list[str]] = {"entity": [], "attr": [], "value": []}
        self.embeddings: dict[str, torch.Tensor | None] = {"entity": None, "attr": None, "value": None}

    def build(self, examples: list[dict], encode_fn, roles=("entity", "attr", "value")):
        """Build phrase bank from training examples.

        Args:
            examples: list of {"state_t": [...], "state_t+1": [...]} dicts
            encode_fn: callable that takes list[str] -> (N, st_dim) tensor
            roles: role names matching triple positions
        """
        role_phrases: dict[str, set[str]] = {r: set() for r in roles}

        for ex in examples:
            for triples in (ex["state_t"], ex["state_t+1"]):
                for triple in triples:
                    for i, phrase in enumerate(triple):
                        role_phrases[roles[i]].add(phrase)

        for role in roles:
            phrases = sorted(role_phrases[role])
            self.phrases[role] = phrases
            if phrases:
                self.embeddings[role] = encode_fn(phrases)

    def lookup(self, vectors: torch.Tensor, role: str) -> list[str]:
        """Find nearest phrases for a batch of vectors.

        Args:
            vectors: (N, st_dim) query vectors
            role: which role bank to search

        Returns:
            list of N phrase strings
        """
        bank = self.embeddings[role]
        if bank is None or len(self.phrases[role]) == 0:
            return ["<unknown>"] * vectors.shape[0]

        bank = bank.to(vectors.device)
        # Cosine similarity
        v_norm = F.normalize(vectors, dim=-1)
        b_norm = F.normalize(bank, dim=-1)
        sims = v_norm @ b_norm.T  # (N, bank_size)
        indices = sims.argmax(dim=-1).cpu().tolist()
        return [self.phrases[role][i] for i in indices]

    def decode_triples(self, vectors: torch.Tensor, roles=("entity", "attr", "value")) -> list[list[str]]:
        """Decode a full (T,) sequence of ST-space vectors to triples.

        Args:
            vectors: (max_triples * 3, st_dim)

        Returns:
            list of [entity, attr, value] triples (pad triples stripped)
        """
        triples = []
        T = vectors.shape[0]
        for i in range(0, T, 3):
            if i + 3 > T:
                break
            triple = []
            is_pad = True
            for j, role in enumerate(roles):
                phrase = self.lookup(vectors[i + j].unsqueeze(0), role)[0]
                triple.append(phrase)
                if phrase != "<pad>" and phrase != "<unknown>":
                    is_pad = False
            if not is_pad:
                triples.append(triple)
        return triples

    def save(self, path):
        """Save phrase bank to disk."""
        data = {}
        for role in self.phrases:
            data[role] = {
                "phrases": self.phrases[role],
                "embeddings": self.embeddings[role].cpu().numpy().tolist() if self.embeddings[role] is not None else None,
            }
        import json
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path) -> "PhraseBank":
        """Load phrase bank from disk."""
        import json
        bank = cls()
        with open(path) as f:
            data = json.load(f)
        for role in data:
            bank.phrases[role] = data[role]["phrases"]
            if data[role]["embeddings"] is not None:
                bank.embeddings[role] = torch.tensor(data[role]["embeddings"], dtype=torch.float32)
        return bank
