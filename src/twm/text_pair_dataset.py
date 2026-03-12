"""Paired text dataset for text dynamics training.

Each example has input text, output text, and mode (identity/qa).
Pre-tokenizes both sides with the shared BPE tokenizer.

Format (JSONL):
    {"mode": "identity"|"qa", "input_text": "...", "output_text": "..."}
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .domain_bpe import DomainBPETokenizer

MODE_IDENTITY = 0
MODE_QA = 1
MODE_REVERSE = 2

MODE_MAP = {
    "identity": MODE_IDENTITY,
    "qa": MODE_QA,
    "reverse": MODE_REVERSE,
}


class TextPairDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 64,
        max_examples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
                if max_examples > 0 and len(examples) >= max_examples:
                    break

        n = len(examples)
        T = max_text_tokens
        pad_id = tokenizer.pad_token_id

        self._input_token_ids = torch.zeros((n, T), dtype=torch.long)
        self._input_pad_mask = torch.ones((n, T), dtype=torch.bool)
        self._output_token_ids = torch.zeros((n, T), dtype=torch.long)
        self._output_pad_mask = torch.ones((n, T), dtype=torch.bool)
        self._output_lengths = torch.zeros(n, dtype=torch.long)
        self._modes = torch.zeros(n, dtype=torch.long)

        for i, ex in enumerate(examples):
            # Input side
            in_ids = tokenizer.encode(ex["input_text"], max_length=T)
            self._input_token_ids[i] = torch.tensor(in_ids, dtype=torch.long)
            self._input_pad_mask[i] = torch.tensor(in_ids) == pad_id

            # Output side
            out_ids = tokenizer.encode(ex["output_text"], max_length=T)
            self._output_token_ids[i] = torch.tensor(out_ids, dtype=torch.long)
            self._output_pad_mask[i] = torch.tensor(out_ids) == pad_id
            self._output_lengths[i] = sum(1 for t in out_ids if t != pad_id)

            # Mode
            self._modes[i] = MODE_MAP.get(ex.get("mode", "identity"), MODE_IDENTITY)

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_token_ids": self._input_token_ids[idx],
            "input_pad_mask": self._input_pad_mask[idx],
            "output_token_ids": self._output_token_ids[idx],
            "output_pad_mask": self._output_pad_mask[idx],
            "output_length": self._output_lengths[idx],
            "mode": self._modes[idx],
        }
