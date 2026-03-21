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
        balance: bool = False,
        distributional_lookup: dict | None = None,
        max_triples: int = 16,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
                if max_examples > 0 and len(examples) >= max_examples:
                    break

        # Balance modes by downsampling majority class
        if balance:
            import random
            by_mode: dict[str, list] = {}
            for ex in examples:
                m = ex.get("mode", "identity")
                by_mode.setdefault(m, []).append(ex)
            min_count = min(len(v) for v in by_mode.values())
            examples = []
            for mode_examples in by_mode.values():
                random.shuffle(mode_examples)
                examples.extend(mode_examples[:min_count])
            random.shuffle(examples)

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

        # Distributional embeddings for CKA alignment (keyed on input text)
        self._dist_embs = None
        self._dist_mask = None
        if distributional_lookup is not None:
            self._build_dist_embs(examples, distributional_lookup, max_triples)

    def _build_dist_embs(self, examples, lookup, max_triples):
        """Build per-example distributional embedding tensors from lookup."""
        text_to_triples = lookup["text_to_triples"]
        span_embs = lookup["span_embeddings"]
        sample_role = next(iter(span_embs.values()))
        sample_emb = next(iter(sample_role.values()))
        d_dist = sample_emb.shape[0]

        n = len(examples)
        S = max_triples * 3
        self._dist_embs = torch.zeros((n, S, d_dist), dtype=torch.float16)
        self._dist_mask = torch.zeros((n, S), dtype=torch.bool)

        role_keys = ["entity", "attribute", "value"]
        matched = 0
        for i, ex in enumerate(examples):
            text = ex["input_text"]
            triples = text_to_triples.get(text)
            if triples is None:
                continue
            for t_idx, triple in enumerate(triples[:max_triples]):
                for r_idx, role in enumerate(role_keys):
                    span = triple[r_idx]
                    emb = span_embs[role].get(span)
                    if emb is not None:
                        slot = t_idx * 3 + r_idx
                        self._dist_embs[i, slot] = emb.half()
                        self._dist_mask[i, slot] = True
            matched += 1

        pct = matched / n * 100 if n > 0 else 0
        print(f"  Distributional: {matched}/{n} texts matched ({pct:.1f}%), "
              f"{self._dist_mask.sum().item()} valid slots, {d_dist}d")

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
