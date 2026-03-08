"""Dataset for hybrid T5 value decoder training.

Stores:
  - Input ST embeddings (for SentenceEncoder)
  - Target entity/attr phrase IDs (for discrete heads)
  - Target value token IDs (pre-tokenized by T5 tokenizer)
  - Per-triple pad masks
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from .phrase_vocab import PhraseVocab, PAD_PHRASE


def _sort_triples(triples: list[list[str]]) -> list[list[str]]:
    return sorted(triples, key=lambda t: tuple(t))


def _pad_triples(triples: list[list[str]], max_triples: int) -> list[list[str]]:
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([PAD_PHRASE, PAD_PHRASE, PAD_PHRASE])
    return padded[:max_triples]


class T5TripleDataset(Dataset):
    """Dataset for hybrid discrete + T5 training."""

    def __init__(
        self,
        path: str | Path,
        encode_fn,
        vocab: PhraseVocab,
        t5_tokenizer: T5Tokenizer,
        max_triples: int = 8,
        max_value_tokens: int = 32,
    ):
        self.max_triples = max_triples
        self.vocab = vocab
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

        # Collect all unique phrases and encode
        all_phrases = set()
        for inp, out in self.examples:
            for triple in inp + out:
                all_phrases.update(triple)
        all_phrases.add(PAD_PHRASE)

        phrase_list = sorted(all_phrases)
        embeddings = encode_fn(phrase_list)
        self.st_dim = embeddings.shape[1]

        phrase_to_idx = {p: i for i, p in enumerate(phrase_list)}
        pad_idx = phrase_to_idx[PAD_PHRASE]
        T = max_triples * 3
        M = max_triples
        n = len(self.examples)

        # Input embeddings
        input_indices = torch.full((n, T), pad_idx, dtype=torch.long)
        input_pad_masks = torch.ones((n, T), dtype=torch.bool)

        # Discrete targets (entity/attr)
        target_entity_ids = torch.zeros((n, M), dtype=torch.long)
        target_attr_ids = torch.zeros((n, M), dtype=torch.long)
        target_pad_masks = torch.ones((n, M), dtype=torch.bool)

        # Collect all target entity and value phrases for batch tokenization
        all_entity_phrases = []
        all_value_phrases = []
        phrase_map = []  # (example_idx, triple_idx)

        for i, (inp_triples, out_triples) in enumerate(self.examples):
            inp_padded = _pad_triples(inp_triples, max_triples)
            out_padded = _pad_triples(out_triples, max_triples)

            for j, triple in enumerate(inp_padded):
                for k, phrase in enumerate(triple):
                    pos = j * 3 + k
                    input_indices[i, pos] = phrase_to_idx[phrase]
                    if phrase != PAD_PHRASE:
                        input_pad_masks[i, pos] = False

            for j, triple in enumerate(out_padded):
                e, a, v = triple
                target_entity_ids[i, j] = vocab.encode_phrase(e, "entity")
                target_attr_ids[i, j] = vocab.encode_phrase(a, "attr")
                if e != PAD_PHRASE:
                    target_pad_masks[i, j] = False

                all_entity_phrases.append(e if e != PAD_PHRASE else "")
                all_value_phrases.append(v if v != PAD_PHRASE else "")
                phrase_map.append((i, j))

        self._all_inputs = embeddings[input_indices]
        self._all_input_pad_masks = input_pad_masks
        self._all_target_entity = target_entity_ids
        self._all_target_attr = target_attr_ids
        self._all_target_pad_masks = target_pad_masks

        # Batch-tokenize entity and value phrases
        print(f"  Tokenizing {len(all_entity_phrases)} entity phrases...", flush=True)
        entity_tokenized = t5_tokenizer(
            all_entity_phrases,
            padding="max_length",
            truncation=True,
            max_length=max_value_tokens,
            return_tensors="pt",
        )

        print(f"  Tokenizing {len(all_value_phrases)} value phrases...", flush=True)
        value_tokenized = t5_tokenizer(
            all_value_phrases,
            padding="max_length",
            truncation=True,
            max_length=max_value_tokens,
            return_tensors="pt",
        )

        # Reshape into (n, M, seq_len)
        seq_len = value_tokenized["input_ids"].shape[1]
        self._all_entity_token_ids = torch.zeros((n, M, seq_len), dtype=torch.long)
        self._all_value_token_ids = torch.zeros((n, M, seq_len), dtype=torch.long)
        self._all_value_attention_mask = torch.zeros((n, M, seq_len), dtype=torch.long)

        for flat_idx, (ex_idx, triple_idx) in enumerate(phrase_map):
            self._all_entity_token_ids[ex_idx, triple_idx] = entity_tokenized["input_ids"][flat_idx]
            self._all_value_token_ids[ex_idx, triple_idx] = value_tokenized["input_ids"][flat_idx]
            self._all_value_attention_mask[ex_idx, triple_idx] = value_tokenized["attention_mask"][flat_idx]

        self.value_seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_embeds": self._all_inputs[idx],
            "input_pad_mask": self._all_input_pad_masks[idx],
            "target_entity": self._all_target_entity[idx],
            "target_attr": self._all_target_attr[idx],
            "target_pad_mask": self._all_target_pad_masks[idx],
            "value_token_ids": self._all_value_token_ids[idx],
            "value_attention_mask": self._all_value_attention_mask[idx],
        }
