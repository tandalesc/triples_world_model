"""Dataset and collation for Triple World Model."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary


def _sort_triples(triples: list[list[str]]) -> list[list[str]]:
    """Canonical ordering: sort triples alphabetically by (entity, relation, value)."""
    return sorted(triples, key=lambda t: tuple(t))


def _pad_triples(triples: list[list[str]], max_triples: int, pad: str = "<pad>") -> list[list[str]]:
    """Pad triple list to max_triples with pad triples."""
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([pad, pad, pad])
    return padded[:max_triples]


def _flatten_triples(triples: list[list[str]], vocab: Vocabulary) -> list[int]:
    """Flatten list of triples to flat token ID list."""
    ids = []
    for triple in triples:
        ids.extend(vocab.encode_triple(triple))
    return ids


class TripleTransitionDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        vocab: Vocabulary,
        max_triples: int = 8,
    ):
        self.vocab = vocab
        self.max_triples = max_triples
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        input_triples, output_triples = self.examples[idx]

        # Canonical sort both sides — aligns positions for the input residual
        input_sorted = _sort_triples(input_triples)
        output_sorted = _sort_triples(output_triples)

        # Pad to max_triples
        input_padded = _pad_triples(input_sorted, self.max_triples)
        output_padded = _pad_triples(output_sorted, self.max_triples)

        input_ids = _flatten_triples(input_padded, self.vocab)
        target_ids = _flatten_triples(output_padded, self.vocab)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack pre-padded tensors into a batch."""
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "target_ids": torch.stack([ex["target_ids"] for ex in batch]),
    }
