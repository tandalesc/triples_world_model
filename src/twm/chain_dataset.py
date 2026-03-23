"""Dataset for multi-turn chain dynamics training.

Reads JSONL with {"chain": [text_0, text_1, ...], "mode": int} format.
The dynamics core is unrolled N-1 times for an N-step chain, with loss
at each intermediate step.

Modes: 0=advance, 1=query, 2=identity (matches TextDynamicsModel conventions).
If "mode" is absent, defaults to 0 (advance).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .domain_bpe import DomainBPETokenizer


class ChainDataset(Dataset):
    """Text chains for multi-turn dynamics with mode conditioning.

    Each example is a sequence of 2-3 text states plus a mode ID.
    The compressor encodes step 0; dynamics unrolls to predict steps 1, 2, ...
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 64,
        max_chain_len: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.max_chain_len = max_chain_len

        chains: list[list[str]] = []
        modes: list[int] = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                chain = data["chain"]
                if len(chain) >= 2:
                    chains.append(chain[:max_chain_len])
                    modes.append(data.get("mode", 0))

        n = len(chains)
        C = max_chain_len
        T = max_text_tokens
        pad_id = tokenizer.pad_token_id

        self._token_ids = torch.zeros((n, C, T), dtype=torch.long)
        self._pad_mask = torch.ones((n, C, T), dtype=torch.bool)
        self._text_lengths = torch.zeros((n, C), dtype=torch.long)
        self._chain_lengths = torch.zeros(n, dtype=torch.long)
        self._mode_ids = torch.tensor(modes, dtype=torch.long)

        for i, chain in enumerate(chains):
            self._chain_lengths[i] = len(chain)
            for step, text in enumerate(chain):
                ids = tokenizer.encode(text, max_length=T)
                self._token_ids[i, step] = torch.tensor(ids, dtype=torch.long)
                self._pad_mask[i, step] = torch.tensor(ids) == pad_id
                self._text_lengths[i, step] = sum(1 for t in ids if t != pad_id)

        self.chains = chains

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self._token_ids[idx, 0],
            "input_pad": self._pad_mask[idx, 0],
            "chain_ids": self._token_ids[idx],
            "chain_pad": self._pad_mask[idx],
            "chain_lengths": self._text_lengths[idx],
            "chain_len": self._chain_lengths[idx],
            "mode_id": self._mode_ids[idx],
        }
