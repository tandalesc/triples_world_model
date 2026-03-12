"""Multi-task dataset for WebNLG multimodal training.

Provides four task types sharing the same examples:
  1. triple → triple (identity reconstruction)
  2. triple → text   (generation)
  3. text → triple   (extraction)
  4. text → text     (paraphrase / identity)

Each __getitem__ returns one task sample. A sampler or collate function
selects which task(s) to include per batch.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .domain_bpe import DomainBPETokenizer
from .phrase_vocab import PhraseVocab, PAD_PHRASE


TASK_TRIPLE_TRIPLE = 0
TASK_TRIPLE_TEXT = 1
TASK_TEXT_TRIPLE = 2
TASK_TEXT_TEXT = 3

TASK_NAMES = {
    TASK_TRIPLE_TRIPLE: "triple→triple",
    TASK_TRIPLE_TEXT: "triple→text",
    TASK_TEXT_TRIPLE: "text→triple",
    TASK_TEXT_TEXT: "text→text",
}


def _pad_triples(triples: list[list[str]], max_triples: int) -> list[list[str]]:
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([PAD_PHRASE, PAD_PHRASE, PAD_PHRASE])
    return padded[:max_triples]


class WebNLGMultimodalDataset(Dataset):
    """Multi-task dataset for triple ↔ text training.

    Pre-tokenizes all data on init. Returns dict with all fields needed
    for any of the 4 tasks; the training loop selects which to use.
    """

    def __init__(
        self,
        path: str | Path,
        vocab: PhraseVocab,
        tokenizer: DomainBPETokenizer,
        max_triples: int = 8,
        max_slot_tokens: int = 12,
        max_text_tokens: int = 64,
        task_weights: dict[int, float] | None = None,
        max_examples: int = 0,
    ):
        self.max_triples = max_triples
        self.max_slot_tokens = max_slot_tokens
        self.max_text_tokens = max_text_tokens
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.task_weights = task_weights or {
            TASK_TRIPLE_TRIPLE: 1.0,
            TASK_TRIPLE_TEXT: 1.0,
            TASK_TEXT_TRIPLE: 1.0,
            TASK_TEXT_TEXT: 0.5,
        }

        # Load raw examples
        self.examples: list[dict] = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))
                if max_examples > 0 and len(self.examples) >= max_examples:
                    break

        n = len(self.examples)
        M = max_triples
        S = max_slot_tokens
        T = max_text_tokens

        # === Triple-side data (compressor input / expander target) ===
        self._triple_token_ids = torch.zeros((n, M, 3, S), dtype=torch.long)
        self._triple_token_pad = torch.ones((n, M, 3, S), dtype=torch.bool)
        self._triple_pad = torch.ones((n, M), dtype=torch.bool)
        self._n_triples = torch.zeros(n, dtype=torch.long)

        # Target: discrete attr IDs + entity/value BPE for diffusion expander
        self._target_attr = torch.zeros((n, M), dtype=torch.long)
        self._target_pad = torch.ones((n, M), dtype=torch.bool)
        self._entity_token_ids = torch.zeros((n, M, S), dtype=torch.long)
        self._value_token_ids = torch.zeros((n, M, S), dtype=torch.long)

        # === Text-side data ===
        self._text_token_ids = torch.zeros((n, T), dtype=torch.long)
        self._text_pad_mask = torch.ones((n, T), dtype=torch.bool)
        self._text_lengths = torch.zeros(n, dtype=torch.long)

        pad_id = tokenizer.pad_token_id

        for i, ex in enumerate(self.examples):
            triples = ex["triples"]
            text = ex["text"]
            n_tri = min(len(triples), M)
            self._n_triples[i] = n_tri

            # Triple tokenization
            padded = _pad_triples(triples, M)
            for j, triple in enumerate(padded):
                e, a, v = triple
                if e == PAD_PHRASE:
                    continue

                self._triple_pad[i, j] = False
                self._target_pad[i, j] = False

                # Compressor input: BPE for all 3 roles
                e_ids = tokenizer.encode(e, max_length=S)
                a_ids = tokenizer.encode(a, max_length=S)
                v_ids = tokenizer.encode(v, max_length=S)

                self._triple_token_ids[i, j, 0] = torch.tensor(e_ids)
                self._triple_token_ids[i, j, 1] = torch.tensor(a_ids)
                self._triple_token_ids[i, j, 2] = torch.tensor(v_ids)

                self._triple_token_pad[i, j, 0] = torch.tensor(e_ids) == pad_id
                self._triple_token_pad[i, j, 1] = torch.tensor(a_ids) == pad_id
                self._triple_token_pad[i, j, 2] = torch.tensor(v_ids) == pad_id

                # Expander targets
                self._target_attr[i, j] = vocab.encode_phrase(a, "attr")
                self._entity_token_ids[i, j] = torch.tensor(
                    tokenizer.encode(e, max_length=S), dtype=torch.long,
                )
                self._value_token_ids[i, j] = torch.tensor(
                    tokenizer.encode(v, max_length=S), dtype=torch.long,
                )

            # Text tokenization
            text_ids = tokenizer.encode(text, max_length=T)
            self._text_token_ids[i] = torch.tensor(text_ids, dtype=torch.long)
            self._text_pad_mask[i] = torch.tensor(text_ids) == pad_id
            self._text_lengths[i] = sum(1 for t in text_ids if t != pad_id)

    def __len__(self) -> int:
        return len(self.examples)

    def sample_task(self) -> int:
        """Sample a task type according to weights."""
        tasks = list(self.task_weights.keys())
        weights = [self.task_weights[t] for t in tasks]
        return random.choices(tasks, weights=weights, k=1)[0]

    def __getitem__(self, idx: int) -> dict:
        """Return all data needed for any task. Task selection is external."""
        return {
            # Triple data
            "triple_token_ids": self._triple_token_ids[idx],
            "triple_token_pad": self._triple_token_pad[idx],
            "triple_pad": self._triple_pad[idx],
            "n_triples": self._n_triples[idx],
            # Triple targets (for expander)
            "target_attr": self._target_attr[idx],
            "target_pad": self._target_pad[idx],
            "entity_token_ids": self._entity_token_ids[idx],
            "value_token_ids": self._value_token_ids[idx],
            # Text data
            "text_token_ids": self._text_token_ids[idx],
            "text_pad_mask": self._text_pad_mask[idx],
            "text_length": self._text_lengths[idx],
        }
