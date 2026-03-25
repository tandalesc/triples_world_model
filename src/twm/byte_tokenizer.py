"""Byte-level tokenizer — no BPE, no vocabulary assumptions.

Encodes text as raw UTF-8 bytes. Vocab size is 259:
  0 = <pad>, 1 = <mask>, 2 = <unk>, 3-258 = byte values 0-255.

Drop-in replacement for DomainBPETokenizer.
"""


PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
BYTE_OFFSET = 3  # byte value 0 maps to token ID 3


class ByteTokenizer:
    """Byte-level tokenizer with the same interface as DomainBPETokenizer."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.pad_token_id = PAD_ID
        self.mask_token_id = MASK_ID
        self.unk_token_id = UNK_ID
        self.vocab_size = 256 + BYTE_OFFSET  # 259

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ml = max_length or self.max_length
        byte_ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]
        byte_ids = byte_ids[:ml]
        byte_ids += [PAD_ID] * (ml - len(byte_ids))
        return byte_ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        byte_vals = []
        for i in ids:
            if skip_special_tokens and i in (PAD_ID, MASK_ID, UNK_ID):
                continue
            if i >= BYTE_OFFSET:
                byte_vals.append(i - BYTE_OFFSET)
        try:
            return bytes(byte_vals).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def batch_decode(self, id_tensor, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in id_tensor]

    def batch_encode(self, texts: list[str], max_length: int | None = None) -> list[list[int]]:
        return [self.encode(t, max_length=max_length) for t in texts]

    @classmethod
    def load(cls, path: str, max_length: int = 512) -> "ByteTokenizer":
        """Compatibility — bytes don't need a saved tokenizer."""
        return cls(max_length=max_length)

    def save(self, path: str) -> None:
        """No-op — nothing to save."""
        pass
