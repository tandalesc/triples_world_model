"""Token-level W-space dataset for Triple World Model.

Instead of sentence-transformer embeddings (one vector per phrase), this
dataset BPE-tokenizes each input phrase and produces per-token W-space
embeddings. The TWM sees richer, token-level input representations.

Input shape: (B, max_triples * 3 * max_tokens_per_slot, d_model)
  - Each of the 3 triple roles gets max_tokens_per_slot token positions
  - Token embeddings come from frozen W-space embeddings (same as decoder)

Output targets are unchanged from DomainTripleDataset:
  - attr: discrete classification IDs
  - entity/value: BPE token IDs for diffusion decoder
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


class TokenTripleDataset(Dataset):
    """Dataset with token-level W-space input embeddings."""

    def __init__(
        self,
        path: str | Path,
        token_emb_weight: torch.Tensor,
        vocab: PhraseVocab,
        domain_tokenizer: DomainBPETokenizer,
        max_triples: int = 8,
        max_tokens_per_slot: int = 12,
        max_value_tokens: int = 12,
    ):
        self.max_triples = max_triples
        self.max_tokens_per_slot = max_tokens_per_slot
        self.vocab = vocab
        self.domain_tokenizer = domain_tokenizer
        self.examples: list[tuple[list[list[str]], list[list[str]]]] = []

        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data["state_t"], data["state_t+1"]))

        n = len(self.examples)
        M = max_triples
        S = max_tokens_per_slot
        T = M * 3 * S  # total input sequence length

        # Input: per-token BPE IDs (will be embedded later)
        input_token_ids = torch.zeros((n, T), dtype=torch.long)
        input_pad_masks = torch.ones((n, T), dtype=torch.bool)

        # Discrete targets (attr)
        target_attr_ids = torch.zeros((n, M), dtype=torch.long)
        target_pad_masks = torch.ones((n, M), dtype=torch.bool)

        # Output entity/value token IDs
        all_entity_phrases = []
        all_value_phrases = []
        phrase_map = []

        for i, (inp_triples, out_triples) in enumerate(self.examples):
            inp_padded = _pad_triples(inp_triples, max_triples)
            out_padded = _pad_triples(out_triples, max_triples)

            # Tokenize input triple positions
            for j, triple in enumerate(inp_padded):
                for k, phrase in enumerate(triple):
                    slot_start = (j * 3 + k) * S
                    if phrase == PAD_PHRASE:
                        # Leave as zeros (pad), mask stays True
                        continue
                    ids = domain_tokenizer.encode(phrase, max_length=S)
                    for t, tid in enumerate(ids):
                        input_token_ids[i, slot_start + t] = tid
                        if tid != domain_tokenizer.pad_token_id:
                            input_pad_masks[i, slot_start + t] = False

            # Output targets (same as DomainTripleDataset)
            for j, triple in enumerate(out_padded):
                e, a, v = triple
                target_attr_ids[i, j] = vocab.encode_phrase(a, "attr")
                if e != PAD_PHRASE:
                    target_pad_masks[i, j] = False
                all_entity_phrases.append(e if e != PAD_PHRASE else "")
                all_value_phrases.append(v if v != PAD_PHRASE else "")
                phrase_map.append((i, j))

        # Embed input tokens using frozen W-space embeddings
        # token_emb_weight: (vocab_size, d_model)
        self._all_inputs = token_emb_weight[input_token_ids]  # (n, T, d_model)
        self._all_input_pad_masks = input_pad_masks

        self._all_target_attr = target_attr_ids
        self._all_target_pad_masks = target_pad_masks

        # Tokenize output entity/value phrases
        print(f"  Tokenizing {len(all_entity_phrases)} entity phrases (domain BPE)...", flush=True)
        entity_ids_list = domain_tokenizer.batch_encode(all_entity_phrases, max_length=max_value_tokens)
        print(f"  Tokenizing {len(all_value_phrases)} value phrases (domain BPE)...", flush=True)
        value_ids_list = domain_tokenizer.batch_encode(all_value_phrases, max_length=max_value_tokens)

        seq_len = max_value_tokens
        self._all_entity_token_ids = torch.zeros((n, M, seq_len), dtype=torch.long)
        self._all_value_token_ids = torch.zeros((n, M, seq_len), dtype=torch.long)

        for flat_idx, (ex_idx, triple_idx) in enumerate(phrase_map):
            self._all_entity_token_ids[ex_idx, triple_idx] = torch.tensor(entity_ids_list[flat_idx], dtype=torch.long)
            self._all_value_token_ids[ex_idx, triple_idx] = torch.tensor(value_ids_list[flat_idx], dtype=torch.long)

        self.value_seq_len = seq_len

        # Stats
        non_pad = ~input_pad_masks
        avg_tokens = non_pad.float().sum(dim=1).mean().item()
        print(f"  Token-level input: {T} positions/example, avg {avg_tokens:.1f} non-pad tokens")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_embeds": self._all_inputs[idx],
            "input_pad_mask": self._all_input_pad_masks[idx],
            "target_attr": self._all_target_attr[idx],
            "target_pad_mask": self._all_target_pad_masks[idx],
            "entity_token_ids": self._all_entity_token_ids[idx],
            "value_token_ids": self._all_value_token_ids[idx],
        }
