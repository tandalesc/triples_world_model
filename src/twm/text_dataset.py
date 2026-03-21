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
        distributional_lookup: dict | None = None,
        max_triples: int = 16,
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

        # Distributional embeddings for CKA alignment
        self._dist_embs = None
        self._dist_mask = None
        if distributional_lookup is not None:
            self._build_dist_embs(texts, distributional_lookup, max_triples)

    def _build_dist_embs(self, texts, lookup, max_triples):
        """Build per-example distributional embedding tensors from lookup."""
        text_to_triples = lookup["text_to_triples"]
        span_embs = lookup["span_embeddings"]
        # Infer embedding dim from first available span
        sample_role = next(iter(span_embs.values()))
        sample_emb = next(iter(sample_role.values()))
        d_dist = sample_emb.shape[0]

        n = len(texts)
        S = max_triples * 3  # total slots
        self._dist_embs = torch.zeros((n, S, d_dist), dtype=torch.float16)
        self._dist_mask = torch.zeros((n, S), dtype=torch.bool)

        role_keys = ["entity", "attribute", "value"]
        matched = 0
        for i, text in enumerate(texts):
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
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text_token_ids": self._text_token_ids[idx],
            "text_pad_mask": self._text_pad_mask[idx],
            "text_length": self._text_lengths[idx],
        }
