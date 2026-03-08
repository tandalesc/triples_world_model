"""Domain-specific word-level vocabulary for ATOMIC diffusion decoder.

Replaces the 32K T5 vocabulary with ~8K words from training data.
Phrases are split on underscores/spaces into word tokens.
"""

import json
import re
from pathlib import Path
from collections import Counter


PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
SPECIAL_TOKENS = ["<pad>", "<mask>", "<unk>"]


class DomainVocab:
    """Word-level vocabulary built from training phrases."""

    def __init__(self):
        self.word2id: dict[str, int] = {}
        self.id2word: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def build(self, phrases: list[str], min_count: int = 3):
        """Build vocabulary from a list of phrases."""
        counts = Counter()
        for phrase in phrases:
            for w in self._split(phrase):
                counts[w] += 1

        self.word2id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for word, count in counts.most_common():
            if count >= min_count:
                self.word2id[word] = len(self.word2id)
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, phrase: str, max_len: int) -> list[int]:
        """Encode phrase to word IDs, padded to max_len."""
        words = self._split(phrase)
        ids = [self.word2id.get(w, UNK_ID) for w in words[:max_len]]
        ids += [PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode word IDs back to phrase string."""
        words = []
        for i in ids:
            if i == PAD_ID:
                break
            if i in (MASK_ID, UNK_ID):
                continue
            word = self.id2word.get(i, "")
            if word:
                words.append(word)
        return "_".join(words)

    def batch_decode(self, id_tensor, **kwargs) -> list[str]:
        """Decode a batch of ID tensors. Drop-in replacement for T5Tokenizer.batch_decode."""
        results = []
        for row in id_tensor:
            results.append(self.decode(row.tolist()))
        return results

    def save(self, path: str | Path):
        data = {"word2id": self.word2id}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DomainVocab":
        with open(path) as f:
            data = json.load(f)
        v = cls()
        v.word2id = data["word2id"]
        # JSON keys are strings, but IDs should be int keys
        v.id2word = {int(i): w for w, i in v.word2id.items()}
        return v

    @staticmethod
    def _split(phrase: str) -> list[str]:
        return [w for w in re.split(r"[_ ]+", phrase.lower()) if w]

    @classmethod
    def from_training_data(cls, train_path: str | Path, min_count: int = 3) -> "DomainVocab":
        """Build vocab from a JSONL training file."""
        phrases = []
        with open(train_path) as f:
            for line in f:
                ex = json.loads(line)
                for t in ex.get("state_t", []) + ex.get("state_t+1", []):
                    phrases.append(t[0])  # entity
                    phrases.append(t[2])  # value
        v = cls()
        v.build(phrases, min_count=min_count)
        return v
