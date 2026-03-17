"""Domain-specific BPE tokenizer wrapper.

Wraps a HuggingFace `tokenizers` BPE tokenizer with a T5Tokenizer-compatible
interface so it can be used as a drop-in replacement in DiffusionDecoder.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer


PAD_ID = 0
MASK_ID = 1
UNK_ID = 2


class DomainBPETokenizer:
    """BPE tokenizer with T5Tokenizer-compatible encode/decode interface."""

    def __init__(self, tokenizer: Tokenizer, max_length: int = 12):
        self._tok = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.mask_token_id = tokenizer.token_to_id("<mask>")
        self.unk_token_id = tokenizer.token_to_id("<unk>")
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def load(cls, path: str | Path, max_length: int = 12) -> "DomainBPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer, max_length=max_length)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode text to token IDs, padded/truncated to max_length."""
        import re
        ml = max_length or self.max_length
        normalized = text.replace("_", " ").lower().strip()
        # Collapse spaces before punctuation so BPE doesn't create phantom
        # space tokens (e.g., "airport ?" → "airport?" = 2 tokens not 3)
        normalized = re.sub(r'\s+([?.!,;:])', r'\1', normalized)
        enc = self._tok.encode(normalized)
        ids = enc.ids[:ml]
        ids += [self.pad_token_id] * (ml - len(ids))
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to string."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.pad_token_id, self.mask_token_id)]
        return self._tok.decode(ids)

    def batch_decode(self, id_tensor, skip_special_tokens: bool = True, **kwargs) -> list[str]:
        """Decode a batch of ID tensors."""
        results = []
        for row in id_tensor:
            results.append(self.decode(row, skip_special_tokens=skip_special_tokens))
        return results

    def __call__(
        self,
        text_or_texts,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict:
        """T5Tokenizer-compatible __call__ for eval scripts."""
        import torch
        ml = max_length or self.max_length

        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = text_or_texts

        all_ids = []
        for text in texts:
            ids = self.encode(text, max_length=ml)
            all_ids.append(ids)

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_ids, dtype=torch.long)}
        return {"input_ids": all_ids}

    def batch_encode(self, texts: list[str], max_length: int | None = None) -> list[list[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, max_length=max_length) for t in texts]

    def build_t5_init_embeddings(self, target_dim: int = 128) -> torch.Tensor:
        """Build 128d embeddings initialized from T5's 768d embeddings via PCA.

        For each BPE token, tokenizes its string with T5, averages the T5
        embeddings for those subwords, then projects the full matrix from
        768d to target_dim via PCA. This gives the embedding space semantic
        structure from T5 while matching the denoiser's native dimension.

        Returns:
            (vocab_size, target_dim) tensor ready to assign to token_emb.weight
        """
        from transformers import T5EncoderModel, T5Tokenizer

        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5EncoderModel.from_pretrained("t5-small")
        t5_emb = t5_model.shared.weight.detach()  # (32100, 768)

        vocab = self._tok.get_vocab()  # {token_str: id}
        n = self.vocab_size
        t5_dim = t5_emb.shape[1]

        # For each BPE token, get averaged T5 embedding
        avg_embs = torch.zeros(n, t5_dim)
        for token_str, bpe_id in vocab.items():
            if token_str in ("<pad>", "<mask>", "<unk>", "<bos>", "<eos>"):
                # Special tokens: leave as zeros (will be near origin after PCA)
                continue
            # Tokenize the BPE token's string with T5
            t5_ids = t5_tokenizer.encode(token_str, add_special_tokens=False)
            if len(t5_ids) == 0:
                continue
            avg_embs[bpe_id] = t5_emb[t5_ids].mean(dim=0)

        # PCA: center, SVD, project to target_dim
        mean = avg_embs.mean(dim=0)
        centered = avg_embs - mean
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        projection = Vt[:target_dim].T  # (768, target_dim)
        projected = centered @ projection  # (n, target_dim)

        return projected

    def build_wspace_init_embeddings(
        self,
        encode_fn,
        encoder_proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Build embeddings in TWM W-space by encoding each BPE token through
        the same sentence-transformer + projection pipeline the TWM uses.

        Args:
            encode_fn: sentence-transformer encode function (list[str] -> Tensor)
            encoder_proj_weight: SentenceEncoder.proj weight (d_model, st_dim)

        Returns:
            (vocab_size, d_model) tensor in W-space
        """
        vocab = self._tok.get_vocab()
        token_strings = [""] * self.vocab_size
        for token_str, bpe_id in vocab.items():
            if token_str in ("<pad>", "<mask>", "<unk>", "<bos>", "<eos>"):
                token_strings[bpe_id] = "unknown"
            else:
                token_strings[bpe_id] = token_str

        # Encode all token strings with sentence-transformer
        st_embs = encode_fn(token_strings)  # (vocab_size, st_dim)

        # Project to W-space using the TWM encoder's projection
        # proj_weight is (d_model, st_dim), so emb @ proj_weight.T = (vocab, d_model)
        w_embs = st_embs.float().cpu() @ encoder_proj_weight.T.float().cpu()

        # Zero out special tokens
        for token_str, bpe_id in vocab.items():
            if token_str in ("<pad>", "<mask>", "<unk>", "<bos>", "<eos>"):
                w_embs[bpe_id] = 0.0

        return w_embs
