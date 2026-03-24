"""Dataset for next-state prediction over narrative sequences.

Reads JSONL with {"states": [text_0, text_1, ...]} format. Each story
produces len(states)-1 training examples: predict state_{t+1} from
a context window of previous states.

The dynamics core receives concatenated bottleneck states as context
and predicts the next state's bottleneck.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .domain_bpe import DomainBPETokenizer


class NSPDataset(Dataset):
    """Next-state prediction dataset.

    Each example is a (context_window, target) pair extracted from
    a narrative sequence. Context is the previous K states; target
    is the next state.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 128,
        context_window: int = 2,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.context_window = context_window

        # Load stories and expand into (context, target) pairs
        self.pairs: list[tuple[list[str], str]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                states = data["states"]
                for t in range(len(states) - 1):
                    # Context: up to K previous states (including current)
                    ctx_start = max(0, t - context_window + 1)
                    context = states[ctx_start : t + 1]
                    target = states[t + 1]
                    self.pairs.append((context, target))

        # Pre-encode everything
        n = len(self.pairs)
        K = context_window
        T = max_text_tokens
        pad_id = tokenizer.pad_token_id

        # Context: (n, K, T) — padded context window
        self._ctx_ids = torch.zeros((n, K, T), dtype=torch.long)
        self._ctx_pad = torch.ones((n, K, T), dtype=torch.bool)
        self._ctx_lengths = torch.zeros((n, K), dtype=torch.long)
        self._ctx_count = torch.zeros(n, dtype=torch.long)  # actual context size

        # Target: (n, T)
        self._tgt_ids = torch.zeros((n, T), dtype=torch.long)
        self._tgt_pad = torch.ones((n, T), dtype=torch.bool)
        self._tgt_lengths = torch.zeros(n, dtype=torch.long)

        for i, (context, target) in enumerate(self.pairs):
            self._ctx_count[i] = len(context)
            for j, text in enumerate(context):
                ids = tokenizer.encode(text, max_length=T)
                self._ctx_ids[i, j] = torch.tensor(ids, dtype=torch.long)
                self._ctx_pad[i, j] = torch.tensor(ids) == pad_id
                self._ctx_lengths[i, j] = sum(1 for t in ids if t != pad_id)

            tgt_ids = tokenizer.encode(target, max_length=T)
            self._tgt_ids[i] = torch.tensor(tgt_ids, dtype=torch.long)
            self._tgt_pad[i] = torch.tensor(tgt_ids) == pad_id
            self._tgt_lengths[i] = sum(1 for t in tgt_ids if t != pad_id)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return {
            # Context window (previous states)
            "ctx_ids": self._ctx_ids[idx],          # (K, T)
            "ctx_pad": self._ctx_pad[idx],           # (K, T)
            "ctx_lengths": self._ctx_lengths[idx],   # (K,)
            "ctx_count": self._ctx_count[idx],       # scalar
            # Target (next state)
            "tgt_ids": self._tgt_ids[idx],           # (T,)
            "tgt_pad": self._tgt_pad[idx],           # (T,)
            "tgt_length": self._tgt_lengths[idx],    # scalar
        }
