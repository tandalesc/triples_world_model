"""Text Dynamics Model: text compressor + dynamics core + text expander.

Mode triples live in W-space alongside data triples. A mode triple is
3 learned vectors with the same role encoding (E/A/V) as data triples.
The dynamics core processes them uniformly through self-attention —
there is nothing architecturally special about a mode triple.

Modes:
  - identity (0): passthrough — reconstruct input
  - qa (1): transform — question → answer
  - reverse (2): reverse triple order — forces mode-reading circuitry

Pipeline:
    input text → Compressor → [mode_triple | bottleneck] → Dynamics
                                                              ↓
                                  bottleneck' (+ input residual) → Expander → output text
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .modules import TransformerDynamics
from .text_compressor import TextCompressor
from .text_expander import TextExpander

NUM_MODES = 3  # identity=0, qa=1, reverse=2


class TextDynamicsModel(nn.Module):
    """Text compressor + dynamics core + text expander with mode conditioning.

    Mode triples are learned vectors in W-space with role encoding. The
    dynamics core sees them as regular triples. The gate is computed from
    the mode triple's output after dynamics processing.
    """

    def __init__(
        self,
        config: ModelConfig,
        domain_tokenizer,
        text_compressor_layers: int = 4,
        text_expander_layers: int = 3,
        dynamics_layers: int | None = None,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
        alpha_min: float = 0.01,
        num_modes: int = NUM_MODES,
        vae: bool = False,
        compressor_type: str = "standard",
        compressor_denoise_steps: int = 5,
        compressor_random_k: bool = False,
        compressor_k_min: int = 1,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = domain_tokenizer
        self.max_text_tokens = max_text_tokens
        self._text_compressor_layers = text_compressor_layers
        self._text_expander_layers = text_expander_layers
        self._vae = vae
        self._compressor_type = compressor_type
        self._compressor_denoise_steps = compressor_denoise_steps
        d = config.d_model
        dyn_layers = dynamics_layers if dynamics_layers is not None else config.n_layers
        self._dynamics_layers = dyn_layers

        # Shared frozen embedding table
        self.shared_token_emb = nn.Embedding(domain_tokenizer.vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        if compressor_type == "diffusion":
            from .diffusion_compressor import DiffusionCompressor
            self.text_compressor = DiffusionCompressor(
                token_emb=self.shared_token_emb,
                d_model=d,
                n_heads=config.n_heads,
                n_encoder_layers=text_compressor_layers,
                n_denoise_layers=text_expander_layers,
                n_denoise_steps=compressor_denoise_steps,
                max_triples=config.max_triples,
                max_text_tokens=max_text_tokens,
                dropout=dropout,
                random_k=compressor_random_k,
                k_min=compressor_k_min,
            )
        else:
            self.text_compressor = TextCompressor(
                token_emb=self.shared_token_emb,
                d_model=d,
                n_heads=config.n_heads,
                n_layers=text_compressor_layers,
                max_triples=config.max_triples,
                max_text_tokens=max_text_tokens,
                dropout=dropout,
                vae=vae,
            )

        # Dynamics core — zero_init so delta starts at 0 with input residual
        self.dynamics = TransformerDynamics(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=dyn_layers,
            d_ff=config.d_ff,
            dropout=dropout,
            zero_init=True,
        )

        # Mode triples: learned vectors in W-space, one triple per mode.
        # Each triple is 3 vectors (E/A/V slots). The content is whatever
        # the model learns — we label them for our benefit, the dynamics
        # core just sees triples.
        self.mode_emb = nn.Embedding(num_modes * 3, d)

        # Role encoding for mode triples — same E/A/V structure as data
        self.mode_role_emb = nn.Embedding(3, d)

        # Role centroids for bottleneck structure prior.
        # Entity/attribute/value slots are pulled toward learned centroids,
        # imposing role-conditioned geometry on the bottleneck space.
        self.role_centroids = nn.Embedding(3, d)

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

    def _build_mode_triple(self, mode_ids):
        """Build mode triples from learned embeddings + role encoding.

        Args:
            mode_ids: (B,) integer mode IDs

        Returns:
            (B, 3, d) mode triple in W-space
        """
        B = mode_ids.shape[0]
        device = mode_ids.device

        # Each mode has 3 embedding slots: mode_id*3 + {0,1,2}
        base = mode_ids * 3  # (B,)
        slot_ids = base.unsqueeze(1) + torch.arange(3, device=device)  # (B, 3)
        mode_triple = self.mode_emb(slot_ids)  # (B, 3, d)

        # Add role encoding
        role_idx = torch.arange(3, device=device)
        mode_triple = mode_triple + self.mode_role_emb(role_idx)

        return mode_triple

    def compress(self, text_token_ids, text_pad_mask):
        """Compress text to bottleneck. Returns (bottleneck, vae_info) if VAE, else bottleneck."""
        return self.text_compressor(text_token_ids, text_pad_mask, self.config.max_triples)

    def forward_dynamics(self, bottleneck, mode_ids):
        """Run dynamics core with mode triple conditioning and input residual.

        The input residual means the core learns a delta, not the full output.
        For identity mode, the optimal delta is zero — making identity the
        default behavior from initialization.

        Args:
            bottleneck: (B, N*3, d) from compressor
            mode_ids: (B,) integer mode IDs

        Returns:
            (B, N*3, d) transformed bottleneck (input + delta)
        """
        B = bottleneck.shape[0]

        # Build mode triple in W-space
        mode_triple = self._build_mode_triple(mode_ids)  # (B, 3, d)

        # Concatenate — dynamics core sees mode and data triples uniformly
        x = torch.cat([mode_triple, bottleneck], dim=1)  # (B, 3 + N*3, d)

        # Run dynamics transformer
        x = self.dynamics(x)

        # Strip mode triple, keep data positions
        delta = x[:, 3:]  # (B, N*3, d)

        # Input residual: core learns the delta, not the full output
        return bottleneck + delta

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
        """Save model config, weights, and metadata."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(run_dir / "model_config.json")
        torch.save(self.state_dict(), run_dir / "weights.pt")
        meta = {
            "model_type": "dynamics",
            "text_compressor_layers": self._text_compressor_layers,
            "text_expander_layers": self._text_expander_layers,
            "dynamics_layers": self._dynamics_layers,
            "max_text_tokens": self.max_text_tokens,
            "vae": self._vae,
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
            dynamics_layers=meta["dynamics_layers"],
            max_text_tokens=meta["max_text_tokens"],
            vae=meta.get("vae", False),
        )
        state = torch.load(run_dir / "weights.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)
