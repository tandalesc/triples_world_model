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

    Pre-computes all embeddings at init time for fast training.
    """

    def __init__(
        self,
        path: str | Path,
        encode_fn,
        max_triples: int = 8,
    ):
        """
        Args:
            path: JSONL file with state_t / state_t+1 triples (free-text phrases)
            encode_fn: callable(list[str]) -> (N, st_dim) tensor
            max_triples: max triples per state
        """
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

        # Build lookup: phrase -> embedding vector
        self._phrase_to_embed = {p: embeddings[i] for i, p in enumerate(phrase_list)}

        # Pre-compute all examples
        self._inputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._pad_masks: list[torch.Tensor] = []

        for inp_triples, out_triples in self.examples:
            inp_sorted = _sort_triples(inp_triples)
            out_sorted = _sort_triples(out_triples)
            inp_padded = _pad_triples(inp_sorted, max_triples)
            out_padded = _pad_triples(out_sorted, max_triples)

            inp_embeds = self._flatten_to_embeds(inp_padded)
            out_embeds = self._flatten_to_embeds(out_padded)

            # Pad mask: True where input is <pad>
            pad_mask = torch.tensor([
                all(p == PAD_PHRASE for p in triple)
                for triple in inp_padded
                for p in triple  # one mask per position
            ], dtype=torch.bool)
            # Actually we want per-position: pad if the phrase is <pad>
            pad_mask = torch.tensor([
                p == PAD_PHRASE
                for triple in inp_padded
                for p in triple
            ], dtype=torch.bool)

            self._inputs.append(inp_embeds)
            self._targets.append(out_embeds)
            self._pad_masks.append(pad_mask)

    def _flatten_to_embeds(self, triples: list[list[str]]) -> torch.Tensor:
        """Convert padded triples to (max_triples * 3, st_dim) embedding tensor."""
        embeds = []
        for triple in triples:
            for phrase in triple:
                embeds.append(self._phrase_to_embed[phrase])
        return torch.stack(embeds)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_embeds": self._inputs[idx],    # (T, st_dim)
            "target_embeds": self._targets[idx],   # (T, st_dim)
            "pad_mask": self._pad_masks[idx],      # (T,)
        }


def collate_sentence_fn(batch: list[dict]) -> dict:
    return {
        "input_embeds": torch.stack([ex["input_embeds"] for ex in batch]),
        "target_embeds": torch.stack([ex["target_embeds"] for ex in batch]),
        "pad_mask": torch.stack([ex["pad_mask"] for ex in batch]),
    }
