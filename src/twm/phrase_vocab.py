"""Phrase vocabulary for seq2seq decoding.

Maps free-text phrases to integer IDs per role (entity/attr/value),
enabling cross-entropy training instead of cosine NN lookup.
"""

import json
from pathlib import Path

import torch

PAD_PHRASE = "<pad>"
UNK_PHRASE = "<unk>"
SPECIAL_PHRASES = [PAD_PHRASE, UNK_PHRASE]


class PhraseVocab:
    """Per-role phrase vocabularies with ID mapping.

    Unlike PhraseBank (which stores embeddings for NN lookup), this stores
    phrase -> integer ID mappings for cross-entropy training.
    """

    def __init__(self):
        self.roles = ("entity", "attr", "value")
        self.phrase_to_id: dict[str, dict[str, int]] = {r: {} for r in self.roles}
        self.id_to_phrase: dict[str, list[str]] = {r: [] for r in self.roles}

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {r: len(self.id_to_phrase[r]) for r in self.roles}

    def build(self, examples: list[dict]):
        """Build vocabulary from training examples.

        Args:
            examples: list of {"state_t": [...], "state_t+1": [...]} dicts
        """
        role_phrases: dict[str, set[str]] = {r: set() for r in self.roles}

        for ex in examples:
            for triples in (ex["state_t"], ex["state_t+1"]):
                for triple in triples:
                    for i, phrase in enumerate(triple):
                        role_phrases[self.roles[i]].add(phrase)

        for role in self.roles:
            phrases = SPECIAL_PHRASES + sorted(role_phrases[role] - set(SPECIAL_PHRASES))
            self.id_to_phrase[role] = phrases
            self.phrase_to_id[role] = {p: i for i, p in enumerate(phrases)}

    def encode_phrase(self, phrase: str, role: str) -> int:
        return self.phrase_to_id[role].get(phrase, self.phrase_to_id[role][UNK_PHRASE])

    def decode_id(self, idx: int, role: str) -> str:
        if 0 <= idx < len(self.id_to_phrase[role]):
            return self.id_to_phrase[role][idx]
        return UNK_PHRASE

    def encode_triples(self, triples: list[list[str]]) -> list[list[int]]:
        """Encode a list of triples to IDs."""
        return [
            [self.encode_phrase(p, self.roles[i]) for i, p in enumerate(triple)]
            for triple in triples
        ]

    def decode_triples(self, id_triples: list[list[int]]) -> list[list[str]]:
        """Decode a list of ID triples to phrases."""
        return [
            [self.decode_id(idx, self.roles[i]) for i, idx in enumerate(triple)]
            for triple in id_triples
        ]

    def build_embeddings(self, encode_fn) -> dict[str, torch.Tensor]:
        """Encode all phrases per role using a sentence-transformer.

        Returns:
            {"entity": (V_e, st_dim), "attr": (V_a, st_dim), "value": (V_v, st_dim)}
        """
        embeddings = {}
        for role in self.roles:
            phrases = self.id_to_phrase[role]
            if phrases:
                embeddings[role] = encode_fn(phrases)
            else:
                embeddings[role] = torch.zeros(0)
        return embeddings

    def save(self, path: str | Path):
        data = {role: self.id_to_phrase[role] for role in self.roles}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PhraseVocab":
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        for role in vocab.roles:
            vocab.id_to_phrase[role] = data[role]
            vocab.phrase_to_id[role] = {p: i for i, p in enumerate(data[role])}
        return vocab
