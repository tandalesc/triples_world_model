"""Text World Model: text compressor + text expander sharing a 256d bottleneck.

Trains text→text identity reconstruction. The model discovers its own
internal structure via learned extraction queries — no explicit triple
format is imposed. Query mode probing is an inference-time operation
on the bottleneck vectors.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .text_compressor import TextCompressor
from .text_expander import TextExpander


class TextWorldModel(nn.Module):
    """Text compressor/expander with shared frozen embeddings."""

    def __init__(
        self,
        config: ModelConfig,
        domain_tokenizer,
        text_compressor_layers: int = 4,
        text_expander_layers: int = 3,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
        alpha_min: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = domain_tokenizer
        self.max_text_tokens = max_text_tokens
        self._text_compressor_layers = text_compressor_layers
        self._text_expander_layers = text_expander_layers
        d = config.d_model

        # Shared frozen embedding table
        self.shared_token_emb = nn.Embedding(domain_tokenizer.vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        self.text_compressor = TextCompressor(
            token_emb=self.shared_token_emb,
            d_model=d,
            n_heads=config.n_heads,
            n_layers=text_compressor_layers,
            max_triples=config.max_triples,
            max_text_tokens=max_text_tokens,
            dropout=dropout,
        )

        self.text_expander = TextExpander(
            token_emb=self.shared_token_emb,
            d_model=d,
            n_heads=config.n_heads,
            n_layers=text_expander_layers,
            max_text_tokens=max_text_tokens,
            max_triples=config.max_triples,
            dropout=dropout,
            alpha_min=alpha_min,
            use_decode_proj=True,
        )

    def init_embeddings(self):
        with torch.no_grad():
            nn.init.normal_(self.shared_token_emb.weight, std=0.02)
            self.shared_token_emb.weight.data = F.normalize(
                self.shared_token_emb.weight.data, dim=-1
            )
            for sid in (self.tokenizer.pad_token_id,
                        self.tokenizer.mask_token_id,
                        self.tokenizer.unk_token_id):
                if sid is not None:
                    self.shared_token_emb.weight.data[sid] = 0.0

    def compress(self, text_token_ids, text_pad_mask):
        return self.text_compressor(text_token_ids, text_pad_mask, self.config.max_triples)

    def forward_expander(self, bottleneck, target_text_ids, target_text_pad_mask, timestep=None):
        return self.text_expander(bottleneck, target_text_ids, target_text_pad_mask, timestep=timestep)

    def forward_length(self, bottleneck):
        return self.text_expander.forward_length(bottleneck)

    @torch.no_grad()
    def generate(self, bottleneck, n_steps=10):
        return self.text_expander.generate(bottleneck, n_steps=n_steps)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, run_dir: str | Path, tokenizer_path: str | None = None):
        """Save model config, weights, and tokenizer reference."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(run_dir / "model_config.json")
        torch.save(self.state_dict(), run_dir / "weights.pt")
        meta = {
            "model_type": "io",
            "text_compressor_layers": self._text_compressor_layers,
            "text_expander_layers": self._text_expander_layers,
            "max_text_tokens": self.max_text_tokens,
            "tokenizer_path": tokenizer_path or str(getattr(self, '_tokenizer_path', '')),
        }
        with open(run_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, run_dir: str | Path, device: str = "cpu"):
        """Load a saved model from a run directory."""
        from .domain_bpe import DomainBPETokenizer

        run_dir = Path(run_dir)
        config = ModelConfig.load(run_dir / "model_config.json")
        with open(run_dir / "model_meta.json") as f:
            meta = json.load(f)
        tokenizer = DomainBPETokenizer.load(
            meta["tokenizer_path"], max_length=meta["max_text_tokens"]
        )
        model = cls(
            config=config, domain_tokenizer=tokenizer,
            text_compressor_layers=meta["text_compressor_layers"],
            text_expander_layers=meta["text_expander_layers"],
            max_text_tokens=meta["max_text_tokens"],
        )
        state = torch.load(run_dir / "weights.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)
