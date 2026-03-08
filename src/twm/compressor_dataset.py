"""Dataset for compressor/expander training.

Provides raw BPE token IDs for all three roles (entity, attr, value) per triple,
suitable for the TripleCompressor input. Also provides target token IDs for the
diffusion decoder (expander) and discrete attr IDs.

For identity advance: input triples == output triples.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .phrase_vocab import PhraseVocab, PAD_PHRASE
from .domain_bpe import DomainBPETokenizer


def _sort_triples(triples: list[list[str]]) -> list[list[str]]:
    return sorted(triples, key=lambda t: tuple(t))


def _pad_triples(triples: list[list[str]], max_triples: int) -> list[list[str]]:
    padded = list(triples)
    while len(padded) < max_triples:
        padded.append([PAD_PHRASE, PAD_PHRASE, PAD_PHRASE])
    return padded[:max_triples]


class CompressorDataset(Dataset):
    """Dataset providing BPE token IDs for compressor input and decoder targets."""

    def __init__(
        self,
        path: str | Path,
        vocab: PhraseVocab,
        domain_tokenizer: DomainBPETokenizer,
        max_triples: int = 8,
        max_value_tokens: int = 12,
    ):
        self.max_triples = max_triples
        self.vocab = vocab
        self.domain_tokenizer = domain_tokenizer
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

        M = max_triples
        S = max_value_tokens
        n = len(self.examples)

        # Input: BPE token IDs for all three roles per triple
        # Shape: (n, M, 3, S) — entity=0, attr=1, value=2
        self._all_input_token_ids = torch.zeros((n, M, 3, S), dtype=torch.long)
        self._all_input_token_pad = torch.ones((n, M, 3, S), dtype=torch.bool)
        self._all_triple_pad = torch.ones((n, M), dtype=torch.bool)

        # Targets (same structure as DomainTripleDataset)
        self._all_target_attr = torch.zeros((n, M), dtype=torch.long)
        self._all_target_pad_masks = torch.ones((n, M), dtype=torch.bool)
        self._all_entity_token_ids = torch.zeros((n, M, S), dtype=torch.long)
        self._all_value_token_ids = torch.zeros((n, M, S), dtype=torch.long)

        for i, (inp_triples, out_triples) in enumerate(self.examples):
            inp_padded = _pad_triples(inp_triples, M)
            out_padded = _pad_triples(out_triples, M)

            for j, triple in enumerate(inp_padded):
                e, a, v = triple
                if e == PAD_PHRASE:
                    continue

                self._all_triple_pad[i, j] = False

                # BPE-encode all three roles for compressor input
                e_ids = domain_tokenizer.encode(e, max_length=S)
                a_ids = domain_tokenizer.encode(a, max_length=S)
                v_ids = domain_tokenizer.encode(v, max_length=S)

                self._all_input_token_ids[i, j, 0] = torch.tensor(e_ids, dtype=torch.long)
                self._all_input_token_ids[i, j, 1] = torch.tensor(a_ids, dtype=torch.long)
                self._all_input_token_ids[i, j, 2] = torch.tensor(v_ids, dtype=torch.long)

                # Token-level pad masks (True where pad token)
                self._all_input_token_pad[i, j, 0] = torch.tensor(e_ids) == domain_tokenizer.pad_token_id
                self._all_input_token_pad[i, j, 1] = torch.tensor(a_ids) == domain_tokenizer.pad_token_id
                self._all_input_token_pad[i, j, 2] = torch.tensor(v_ids) == domain_tokenizer.pad_token_id

            for j, triple in enumerate(out_padded):
                e, a, v = triple
                self._all_target_attr[i, j] = vocab.encode_phrase(a, "attr")
                if e != PAD_PHRASE:
                    self._all_target_pad_masks[i, j] = False
                    self._all_entity_token_ids[i, j] = torch.tensor(
                        domain_tokenizer.encode(e, max_length=S), dtype=torch.long,
                    )
                    self._all_value_token_ids[i, j] = torch.tensor(
                        domain_tokenizer.encode(v, max_length=S), dtype=torch.long,
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_token_ids": self._all_input_token_ids[idx],
            "input_token_pad": self._all_input_token_pad[idx],
            "triple_pad": self._all_triple_pad[idx],
            "target_attr": self._all_target_attr[idx],
            "target_pad_mask": self._all_target_pad_masks[idx],
            "entity_token_ids": self._all_entity_token_ids[idx],
            "value_token_ids": self._all_value_token_ids[idx],
        }
