"""Dataset for seq2seq triple prediction.

Extends SentenceTripleDataset with integer target IDs for cross-entropy
training. Inputs remain as sentence-transformer embeddings (the encoder
still needs ST vectors), but targets are phrase vocabulary indices.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .phrase_vocab import PhraseVocab, PAD_PHRASE


def _sort_triples(triples: list[list[str]]) -> list[list[str]]:
    return sorted(triples, key=lambda t: tuple(t))


def _pad_triples(triples: list[list[str]], max_triples: int) -> list[list[str]]:
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([PAD_PHRASE, PAD_PHRASE, PAD_PHRASE])
    return padded[:max_triples]


class Seq2SeqTripleDataset(Dataset):
    """Triple transition dataset for seq2seq training.

    Stores:
        - Input ST embeddings (for SentenceEncoder)
        - Target phrase IDs per role (for cross-entropy loss)
        - Per-triple pad masks
    """

    def __init__(
        self,
        path: str | Path,
        encode_fn,
        vocab: PhraseVocab,
        max_triples: int = 8,
    ):
        self.max_triples = max_triples
        self.vocab = vocab
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

        # Collect all unique phrases (input + output) and encode once
        all_phrases = set()
        for inp, out in self.examples:
            for triple in inp + out:
                all_phrases.update(triple)
        all_phrases.add(PAD_PHRASE)

        phrase_list = sorted(all_phrases)
        embeddings = encode_fn(phrase_list)  # (N, st_dim)
        self.st_dim = embeddings.shape[1]

        # Build input embeddings via index lookup (same as SentenceTripleDataset)
        phrase_to_idx = {p: i for i, p in enumerate(phrase_list)}
        pad_idx = phrase_to_idx[PAD_PHRASE]
        T = max_triples * 3
        M = max_triples

        n = len(self.examples)
        input_indices = torch.full((n, T), pad_idx, dtype=torch.long)
        input_pad_masks = torch.ones((n, T), dtype=torch.bool)

        # Target IDs per role (B, max_triples) — one ID per triple position
        target_entity_ids = torch.zeros((n, M), dtype=torch.long)
        target_attr_ids = torch.zeros((n, M), dtype=torch.long)
        target_value_ids = torch.zeros((n, M), dtype=torch.long)
        # Target ST embedding indices per role (for round-trip loss)
        target_entity_emb_idx = torch.full((n, M), pad_idx, dtype=torch.long)
        target_attr_emb_idx = torch.full((n, M), pad_idx, dtype=torch.long)
        target_value_emb_idx = torch.full((n, M), pad_idx, dtype=torch.long)
        target_pad_masks = torch.ones((n, M), dtype=torch.bool)  # True = pad triple

        for i, (inp_triples, out_triples) in enumerate(self.examples):
            inp_padded = _pad_triples(inp_triples, max_triples)
            out_padded = _pad_triples(out_triples, max_triples)

            # Input embeddings
            for j, triple in enumerate(inp_padded):
                for k, phrase in enumerate(triple):
                    pos = j * 3 + k
                    input_indices[i, pos] = phrase_to_idx[phrase]
                    if phrase != PAD_PHRASE:
                        input_pad_masks[i, pos] = False

            # Target phrase IDs + ST embedding indices
            for j, triple in enumerate(out_padded):
                e, a, v = triple
                target_entity_ids[i, j] = vocab.encode_phrase(e, "entity")
                target_attr_ids[i, j] = vocab.encode_phrase(a, "attr")
                target_value_ids[i, j] = vocab.encode_phrase(v, "value")
                target_entity_emb_idx[i, j] = phrase_to_idx.get(e, pad_idx)
                target_attr_emb_idx[i, j] = phrase_to_idx.get(a, pad_idx)
                target_value_emb_idx[i, j] = phrase_to_idx.get(v, pad_idx)
                if e != PAD_PHRASE:
                    target_pad_masks[i, j] = False

        # Batched gather for input embeddings
        self._all_inputs = embeddings[input_indices]  # (n, T, st_dim)
        self._all_input_pad_masks = input_pad_masks   # (n, T)
        self._all_target_entity = target_entity_ids   # (n, M)
        self._all_target_attr = target_attr_ids       # (n, M)
        self._all_target_value = target_value_ids     # (n, M)
        self._all_target_pad_masks = target_pad_masks # (n, M)
        # Target ST embeddings per role (for round-trip contrastive loss)
        self._all_target_entity_embeds = embeddings[target_entity_emb_idx]  # (n, M, st_dim)
        self._all_target_attr_embeds = embeddings[target_attr_emb_idx]      # (n, M, st_dim)
        self._all_target_value_embeds = embeddings[target_value_emb_idx]    # (n, M, st_dim)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_embeds": self._all_inputs[idx],
            "input_pad_mask": self._all_input_pad_masks[idx],
            "target_entity": self._all_target_entity[idx],
            "target_attr": self._all_target_attr[idx],
            "target_value": self._all_target_value[idx],
            "target_pad_mask": self._all_target_pad_masks[idx],
        }
