"""JEPA chain dataset over GLUCOSE adjacent (state_t, state_{t+1}) pairs.

Spec reference: jepa_operator_v1_design.md §6 and T4 row in §12.

Cross-state pairing is mandatory: online encoder sees state_t; EMA target encoder
sees state_{t+1}. Same text to both encoders degenerates into self-reconstruction
with a moving target — no cross-state JEPA signal (spec §6 FIX, Judge 2 D2 flaw).

Storage: contiguous CPU tensors, direct index slicing (no DataLoader).
"""

import json
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from ..domain_bpe import DomainBPETokenizer


class JEPAChainDataset:
    """Adjacent cross-state pairs extracted from GLUCOSE chains.

    Each chain has 3 states [t0, t1, t2], yielding 2 pairs:
        (t0, t1) and (t1, t2).

    For N chains with chain_length=3, __len__() == 2·N.

    Tensors are stored contiguously on CPU (no DataLoader; the trainer does
    direct index slicing per the repo convention, matching chain_dataset.py).

    Args:
        path: path to chain_general_train.jsonl (or any GLUCOSE chain JSONL).
        tokenizer: a DomainBPETokenizer loaded from jepa_bpe_512.json.
        max_text_tokens: pad/truncate each state to this many tokens (spec: 64).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 64,
        append_eos: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.append_eos = append_eos
        pad_id = tokenizer.pad_token_id  # 0 per domain_bpe.py convention
        # <eos>=4 per the GLUCOSE BPE artifact (design v2 §7). Only used when
        # append_eos=True so the AR token decoder learns to stop.
        eos_id = 4

        # Collect adjacent (src_text, tgt_text) pairs from all chains.
        # A chain of length L yields L-1 adjacent pairs.
        src_texts: list[str] = []
        tgt_texts: list[str] = []
        # chain_ids (design §8.2): the originating chain index per pair. The two
        # adjacent pairs of a length-3 chain SHARE an id so the hard-negative MRR
        # (diagnostics) can build true same-chain negative pools instead of degenerating
        # to the easy pool (the integrator's flagged gap). Without this, every pair falls
        # into the `cid = idx` fallback and `easy_minus_hard_mrr` passes vacuously.
        chain_ids: list[int] = []

        with open(path) as f:
            for chain_no, line in enumerate(f):
                data = json.loads(line)
                chain: list[str] = data["chain"]
                for i in range(len(chain) - 1):
                    src_texts.append(chain[i])
                    tgt_texts.append(chain[i + 1])
                    chain_ids.append(chain_no)

        n = len(src_texts)
        T = max_text_tokens

        # Allocate contiguous tensors up-front (mirrors ChainDataset pattern).
        self._src_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._src_pad: Tensor = torch.ones((n, T), dtype=torch.bool)   # True = padding position
        self._tgt_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._tgt_pad: Tensor = torch.ones((n, T), dtype=torch.bool)

        def _insert_eos(ids: list[int]) -> list[int]:
            """Insert <eos> at the first pad slot (or overwrite T-1 if full) so the
            decoder learns to stop. <eos> stays a real (unmasked) target token;
            positions after it remain pad. List length is unchanged (== T)."""
            ids = list(ids)
            if pad_id in ids:
                ids[ids.index(pad_id)] = eos_id
            else:
                ids[-1] = eos_id  # sequence fills T — overwrite the last token
            return ids

        for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
            src_ids = tokenizer.encode(src, max_length=T)
            tgt_ids = tokenizer.encode(tgt, max_length=T)
            if append_eos:
                src_ids = _insert_eos(src_ids)
                tgt_ids = _insert_eos(tgt_ids)

            self._src_ids[i] = torch.tensor(src_ids, dtype=torch.long)
            self._tgt_ids[i] = torch.tensor(tgt_ids, dtype=torch.long)

            # Padding mask: True where position holds a pad token. With append_eos
            # the <eos> position is NOT pad (it is a real target token); only the
            # trailing post-eos pad positions are masked.
            self._src_pad[i] = torch.tensor(src_ids, dtype=torch.long) == pad_id
            self._tgt_pad[i] = torch.tensor(tgt_ids, dtype=torch.long) == pad_id

        # Keep the raw texts for iter_text_pairs() (operator-fit pass-2 §7).
        self._src_texts: list[str] = src_texts
        self._tgt_texts: list[str] = tgt_texts
        # Per-pair originating chain id (design §8.2). len == len(dataset).
        self._chain_ids: list[int] = chain_ids

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------

    @property
    def chain_ids(self) -> list[int]:
        """Per-pair originating chain id (design §8.2), len == len(self).

        Adjacent pairs of one chain share an id, so the diagnostics hard-negative MRR
        builds true same-chain negative pools. Stays aligned with the `max_chains` cap
        path (train_jepa_v2.py slices `_chain_ids` alongside the tensors).
        """
        return self._chain_ids

    def __len__(self) -> int:
        return self._src_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return a single adjacent (state_t, state_{t+1}) tensor pair.

        Returns:
            src_ids: (T_text,) long  — tokenized state_t, padded to max_text_tokens.
            src_pad: (T_text,) bool  — True at padding positions of src.
            tgt_ids: (T_text,) long  — tokenized state_{t+1}, padded.
            tgt_pad: (T_text,) bool  — True at padding positions of tgt.

        Cross-state pairing guarantee: src_ids and tgt_ids encode DIFFERENT states
        (state_t vs state_{t+1}). They are never identical in content by construction.
        """
        return {
            "src_ids": self._src_ids[idx],
            "src_pad": self._src_pad[idx],
            "tgt_ids": self._tgt_ids[idx],
            "tgt_pad": self._tgt_pad[idx],
        }

    def get_batch(self, indices) -> dict:
        """Return a batch of items for the given indices (list or Tensor).

        Convenience method for the no-DataLoader direct-slicing trainer convention.

        Returns tensors of shape (B, T_text) for each key.
        """
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return {
            "src_ids": self._src_ids[indices],
            "src_pad": self._src_pad[indices],
            "tgt_ids": self._tgt_ids[indices],
            "tgt_pad": self._tgt_pad[indices],
        }

    # ------------------------------------------------------------------
    # Operator-fit pass-2 interface (spec §7)
    # ------------------------------------------------------------------

    def iter_text_pairs(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield (src_ids, tgt_ids) token tensors for the operator-fit pass-2.

        Yields one (state_t, state_{t+1}) pair at a time as (T_text,) long tensors.
        Used by scripts/operator_group_fit.py pass-2 to re-verify the operator
        family choice in the trained noun space (spec §7).
        """
        for i in range(len(self)):
            yield self._src_ids[i], self._tgt_ids[i]
