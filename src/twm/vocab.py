"""Vocabulary builder for Triple World Model.

Supports two modes:
1. Shared vocabulary: all tokens in one table (original behavior)
2. Split vocabulary: separate token→id mappings for entity/attr/value positions
"""

import json
from pathlib import Path

PAD_TOKEN = "<pad>"
PAD_ID = 0

ROLES = ("entity", "attr", "value")


class Vocabulary:
    def __init__(self):
        self.token2id: dict[str, int] = {PAD_TOKEN: PAD_ID}
        self.id2token: dict[int, str] = {PAD_ID: PAD_TOKEN}
        self._next_id = 1

        # Role-separated sub-vocabularies
        self.role_token2id: dict[str, dict[str, int]] = {
            r: {PAD_TOKEN: PAD_ID} for r in ROLES
        }
        self.role_id2token: dict[str, dict[int, str]] = {
            r: {PAD_ID: PAD_TOKEN} for r in ROLES
        }
        self._role_next_id: dict[str, int] = {r: 1 for r in ROLES}

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def __len__(self) -> int:
        return self._next_id

    def __getitem__(self, token: str) -> int:
        return self.token2id[token]

    def role_vocab_size(self, role: str) -> int:
        return self._role_next_id[role]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            self.token2id[token] = self._next_id
            self.id2token[self._next_id] = token
            self._next_id += 1
        return self.token2id[token]

    def add_role_token(self, token: str, role: str) -> int:
        """Add a token to a role-specific sub-vocabulary."""
        t2i = self.role_token2id[role]
        if token not in t2i:
            tid = self._role_next_id[role]
            t2i[token] = tid
            self.role_id2token[role][tid] = token
            self._role_next_id[role] = tid + 1
        return t2i[token]

    def encode_triple(self, triple: list[str]) -> list[int]:
        return [self.token2id[t] for t in triple]

    def encode_triple_split(self, triple: list[str]) -> list[int]:
        """Encode a triple using role-specific vocabularies."""
        return [
            self.role_token2id[role][token]
            for role, token in zip(ROLES, triple)
        ]

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

    def decode_triples_split(self, ids: list[int]) -> list[list[str]]:
        """Decode flat token IDs using role-specific vocabularies."""
        triples = []
        for i in range(0, len(ids), 3):
            chunk = ids[i : i + 3]
            if len(chunk) == 3 and all(c != PAD_ID for c in chunk):
                triple = [
                    self.role_id2token[role][cid]
                    for role, cid in zip(ROLES, chunk)
                ]
                triples.append(triple)
        return triples

    @classmethod
    def from_files(cls, *paths: str | Path) -> "Vocabulary":
        vocab = cls()
        for path in paths:
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    for triple in data["state_t"] + data["state_t+1"]:
                        for i, token in enumerate(triple):
                            vocab.add_token(token)
                            vocab.add_role_token(token, ROLES[i])
        return vocab

    def save(self, path: str | Path):
        data = {
            "token2id": self.token2id,
            "next_id": self._next_id,
            "role_token2id": self.role_token2id,
            "role_next_id": self._role_next_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab._next_id = data["next_id"]

        # Load role vocabs if present (backward compat)
        if "role_token2id" in data:
            vocab.role_token2id = data["role_token2id"]
            vocab._role_next_id = data["role_next_id"]
            vocab.role_id2token = {
                role: {v: k for k, v in t2i.items()}
                for role, t2i in vocab.role_token2id.items()
            }
        return vocab
