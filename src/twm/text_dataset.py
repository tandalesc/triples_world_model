"""Simple text-only dataset for text compressor/expander training.

Reads JSONL with a "text" field (compatible with WebNLG multimodal format).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .domain_bpe import DomainBPETokenizer


class TextDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 64,
        max_examples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

        texts = []
        with open(path) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if max_examples > 0 and len(texts) >= max_examples:
                    break

        n = len(texts)
        T = max_text_tokens
        pad_id = tokenizer.pad_token_id

        self._text_token_ids = torch.zeros((n, T), dtype=torch.long)
        self._text_pad_mask = torch.ones((n, T), dtype=torch.bool)
        self._text_lengths = torch.zeros(n, dtype=torch.long)

        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, max_length=T)
            self._text_token_ids[i] = torch.tensor(ids, dtype=torch.long)
            self._text_pad_mask[i] = torch.tensor(ids) == pad_id
            self._text_lengths[i] = sum(1 for t in ids if t != pad_id)

        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text_token_ids": self._text_token_ids[idx],
            "text_pad_mask": self._text_pad_mask[idx],
            "text_length": self._text_lengths[idx],
        }
