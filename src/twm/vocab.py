"""Vocabulary builder for Triple World Model."""

import json
from pathlib import Path

PAD_TOKEN = "<pad>"
PAD_ID = 0


class Vocabulary:
    def __init__(self):
        self.token2id: dict[str, int] = {PAD_TOKEN: PAD_ID}
        self.id2token: dict[int, str] = {PAD_ID: PAD_TOKEN}
        self._next_id = 1

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def __len__(self) -> int:
        return self._next_id

    def __getitem__(self, token: str) -> int:
        return self.token2id[token]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = self._next_id
            self.id2token[self._next_id] = token
            self._next_id += 1
        return self.token2id[token]

    def encode_triple(self, triple: list[str]) -> list[int]:
        return [self.token2id[t] for t in triple]

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.id2token[i] for i in ids]

    def decode_triples(self, ids: list[int]) -> list[list[str]]:
        """Decode flat token IDs to list of triples, stripping <pad> triples."""
        triples = []
        for i in range(0, len(ids), 3):
            chunk = ids[i : i + 3]
            if len(chunk) == 3 and all(c != PAD_ID for c in chunk):
                triples.append([self.id2token[c] for c in chunk])
        return triples

    @classmethod
    def from_files(cls, *paths: str | Path) -> "Vocabulary":
        vocab = cls()
        for path in paths:
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    for triple in data["state_t"] + data["state_t+1"]:
                        for token in triple:
                            vocab.add_token(token)
        return vocab

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id, "next_id": self._next_id}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab._next_id = data["next_id"]
        return vocab
