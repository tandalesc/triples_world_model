"""Dataset for free-text triple transitions using sentence-transformer embeddings.

Unlike TripleTransitionDataset which maps tokens to integer IDs,
this dataset encodes each triple position as a sentence-transformer
embedding vector. Used with SentenceEncoder/SentenceDecoder.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


PAD_PHRASE = "<pad>"


def _sort_triples(triples: list[list[str]]) -> list[list[str]]:
    return sorted(triples, key=lambda t: tuple(t))


def _pad_triples(triples: list[list[str]], max_triples: int) -> list[list[str]]:
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([PAD_PHRASE, PAD_PHRASE, PAD_PHRASE])
    return padded[:max_triples]


class SentenceTripleDataset(Dataset):
    """Triple transition dataset that produces sentence-transformer embeddings.

    Pre-computes all embeddings at init time into contiguous tensors
    for fast batched training with zero per-item overhead.
    """

    def __init__(
        self,
        path: str | Path,
        encode_fn,
        max_triples: int = 8,
    ):
        self.max_triples = max_triples
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

        # Collect all unique phrases and encode once
        all_phrases = set()
        for inp, out in self.examples:
            for triple in inp + out:
                all_phrases.update(triple)
        all_phrases.add(PAD_PHRASE)

        phrase_list = sorted(all_phrases)
        embeddings = encode_fn(phrase_list)  # (N, st_dim)
        self.st_dim = embeddings.shape[1]

        # Build phrase -> index lookup
        phrase_to_idx = {p: i for i, p in enumerate(phrase_list)}
        pad_idx = phrase_to_idx[PAD_PHRASE]
        T = max_triples * 3

        # Build index arrays for all examples (pure Python, fast)
        n = len(self.examples)
        input_indices = torch.full((n, T), pad_idx, dtype=torch.long)
        target_indices = torch.full((n, T), pad_idx, dtype=torch.long)
        pad_masks = torch.ones((n, T), dtype=torch.bool)

        for i, (inp_triples, out_triples) in enumerate(self.examples):
            inp_padded = _pad_triples(_sort_triples(inp_triples), max_triples)
            out_padded = _pad_triples(_sort_triples(out_triples), max_triples)

            for j, triple in enumerate(inp_padded):
                for k, phrase in enumerate(triple):
                    pos = j * 3 + k
                    idx = phrase_to_idx[phrase]
                    input_indices[i, pos] = idx
                    if phrase != PAD_PHRASE:
                        pad_masks[i, pos] = False

            for j, triple in enumerate(out_padded):
                for k, phrase in enumerate(triple):
                    pos = j * 3 + k
                    target_indices[i, pos] = phrase_to_idx[phrase]

        # Single batched index lookup — all on GPU at once
        self._all_inputs = embeddings[input_indices]    # (n, T, st_dim)
        self._all_targets = embeddings[target_indices]  # (n, T, st_dim)
        self._all_pad_masks = pad_masks                 # (n, T)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_embeds": self._all_inputs[idx],
            "target_embeds": self._all_targets[idx],
            "pad_mask": self._all_pad_masks[idx],
        }


def collate_sentence_fn(batch: list[dict]) -> dict:
    return {
        "input_embeds": torch.stack([ex["input_embeds"] for ex in batch]),
        "target_embeds": torch.stack([ex["target_embeds"] for ex in batch]),
        "pad_mask": torch.stack([ex["pad_mask"] for ex in batch]),
    }
