"""Multimodal model: shared 256d bottleneck with triple and text I/O.

Four components sharing a bottleneck:
  TripleCompressor → bottleneck → TripleExpander (DiffusionDecoder)
  TextCompressor   → bottleneck → TextExpander

All four combinations are valid:
  triple→triple, triple→text, text→triple, text→text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressor import TripleCompressor
from .text_compressor import TextCompressor
from .text_expander import TextExpander
from .diffusion_decoder import DiffusionDecoder
from .phrase_vocab import PhraseVocab
from .config import ModelConfig


class MultimodalWorldModel(nn.Module):
    """Shared-bottleneck model for triple ↔ text translation."""

    def __init__(
        self,
        config: ModelConfig,
        vocab: PhraseVocab,
        domain_tokenizer,
        # Compressor settings
        compressor_layers: int = 2,
        text_compressor_layers: int = 4,
        max_slot_tokens: int = 12,
        max_text_tokens: int = 64,
        # Triple expander settings
        denoiser_layers: int = 1,
        # Text expander settings
        text_expander_layers: int = 3,
        # Shared
        dropout: float = 0.1,
        alpha_min: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.tokenizer = domain_tokenizer
        self.max_text_tokens = max_text_tokens
        self.max_slot_tokens = max_slot_tokens
        d = config.d_model
        vocab_size = domain_tokenizer.vocab_size
        mask_id = domain_tokenizer.mask_token_id

        # Shared frozen embedding table
        self.shared_token_emb = nn.Embedding(vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        # === Triple Compressor (validated) ===
        self.triple_compressor = TripleCompressor(
            token_emb=self.shared_token_emb,
            d_model=d,
            n_heads=config.n_heads,
            n_layers=compressor_layers,
            n_roles=3,
            max_seq_len=max_slot_tokens,
            max_triples=config.max_triples,
            dropout=dropout,
        )

        # === Text Compressor (new) ===
        self.text_compressor = TextCompressor(
            token_emb=self.shared_token_emb,
            d_model=d,
            n_heads=config.n_heads,
            n_layers=text_compressor_layers,
            max_triples=config.max_triples,
            max_text_tokens=max_text_tokens,
            dropout=dropout,
        )

        # === Triple Expander (validated) ===
        # Uses unified decoder with role embeddings (entity=0, value=1)
        self.triple_expander = DiffusionDecoder(
            twm_dim=d,
            n_proj_tokens=3,
            max_seq_len=max_slot_tokens,
            vocab_size=vocab_size,
            d_model=d,  # Match dynamics dim
            n_heads=config.n_heads,
            n_layers=denoiser_layers,
            dropout=dropout,
            mask_token_id=mask_id,
            tokenizer=domain_tokenizer,
            use_cross_attention=True,
            use_adaln=True,
            use_continuous_noise=True,
            normalize_noise=True,
            alpha_min=alpha_min,
            n_roles=2,  # entity=0, value=1
            wspace=True,
            use_mse_prediction=True,
            use_decode_proj=True,
        )

        # === Text Expander (new) ===
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

        # === Discrete attr head ===
        self.ln_attr = nn.LayerNorm(d)
        self.attr_head = nn.Linear(d, vocab.vocab_sizes["attr"])

        # === Length heads ===
        self.triple_length_head = nn.Linear(d, 1)  # per-slot length
        # Text length head is inside text_expander

    def init_embeddings(self):
        """Initialize shared embeddings: random unit-norm, zero specials."""
        with torch.no_grad():
            nn.init.normal_(self.shared_token_emb.weight, std=0.02)
            self.shared_token_emb.weight.data = F.normalize(
                self.shared_token_emb.weight.data, dim=-1
            )
            for special_id in (self.tokenizer.pad_token_id,
                               self.tokenizer.mask_token_id,
                               self.tokenizer.unk_token_id):
                if special_id is not None:
                    self.shared_token_emb.weight.data[special_id] = 0.0

    # ── Compression ──────────────────────────────────────────────────

    def compress_triples(
        self,
        triple_token_ids: torch.Tensor,
        triple_token_pad: torch.Tensor,
        triple_pad: torch.Tensor,
    ) -> torch.Tensor:
        """(B, M, 3, S) → (B, M*3, d_model) bottleneck vectors."""
        return self.triple_compressor(triple_token_ids, triple_token_pad, triple_pad)

    def compress_text(
        self,
        text_token_ids: torch.Tensor,
        text_pad_mask: torch.Tensor,
        n_triples: int,
    ) -> torch.Tensor:
        """(B, T) → (B, M*3, d_model) bottleneck vectors."""
        return self.text_compressor(text_token_ids, text_pad_mask, n_triples)

    # ── Triple expansion ─────────────────────────────────────────────

    def _extract_triple_context(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """(B, M*3, d) → (B, M, 3*d) per-triple concatenated context."""
        M = self.config.max_triples
        device = bottleneck.device
        e_idx = torch.arange(0, M * 3, 3, device=device)
        a_idx = torch.arange(1, M * 3, 3, device=device)
        v_idx = torch.arange(2, M * 3, 3, device=device)
        return torch.cat([
            bottleneck[:, e_idx],
            bottleneck[:, a_idx],
            bottleneck[:, v_idx],
        ], dim=-1)

    def forward_triple_expander(
        self,
        role: str,
        bottleneck: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run diffusion forward for entity or value expansion."""
        role_id = 0 if role == "entity" else 1
        triple_ctx = self._extract_triple_context(bottleneck)
        B, M, D = triple_ctx.shape

        ctx_flat = triple_ctx.reshape(B * M, D)
        tgt_flat = target_ids.reshape(B * M, -1)
        pad_flat = target_pad_mask.reshape(B * M)

        valid = ~pad_flat
        if not valid.any():
            S = target_ids.shape[-1]
            return (torch.zeros(0, S, self.config.d_model, device=bottleneck.device),
                    torch.zeros(0, S, dtype=torch.bool, device=bottleneck.device))

        valid_t = None
        if timestep is not None:
            per_triple_t = timestep.unsqueeze(1).expand(B, M).reshape(B * M)
            valid_t = per_triple_t[valid]

        return self.triple_expander(
            ctx_flat[valid], tgt_flat[valid],
            timestep=valid_t, role_id=role_id,
        )

    def forward_attr(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Predict discrete attr from bottleneck. Returns (B, M, n_attrs) logits."""
        M = self.config.max_triples
        device = bottleneck.device
        a_idx = torch.arange(1, M * 3, 3, device=device)
        attr_vecs = self.ln_attr(bottleneck[:, a_idx])
        return self.attr_head(attr_vecs)

    def forward_triple_lengths(
        self, bottleneck: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entity and value token lengths from bottleneck."""
        M = self.config.max_triples
        device = bottleneck.device
        e_idx = torch.arange(0, M * 3, 3, device=device)
        v_idx = torch.arange(2, M * 3, 3, device=device)
        ent_pred = self.triple_length_head(bottleneck[:, e_idx]).squeeze(-1)
        val_pred = self.triple_length_head(bottleneck[:, v_idx]).squeeze(-1)
        return ent_pred, val_pred

    # ── Text expansion ───────────────────────────────────────────────

    def forward_text_expander(
        self,
        bottleneck: torch.Tensor,
        target_text_ids: torch.Tensor,
        target_text_pad_mask: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run text expander diffusion forward."""
        return self.text_expander(
            bottleneck, target_text_ids, target_text_pad_mask,
            triple_pad_mask=triple_pad_mask, timestep=timestep,
        )

    def forward_text_length(
        self,
        bottleneck: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict text length from bottleneck."""
        return self.text_expander.forward_length(bottleneck, triple_pad_mask)

    # ── Generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate_triple_ids(
        self,
        bottleneck: torch.Tensor,
        n_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate entity IDs, value IDs, and attr predictions.

        Returns:
            entity_ids: (B, M, S)
            value_ids: (B, M, S)
            attr_preds: (B, M) predicted attr class IDs
        """
        triple_ctx = self._extract_triple_context(bottleneck)
        B, M, D = triple_ctx.shape
        ctx_flat = triple_ctx.reshape(B * M, D)

        ent_ids = self.triple_expander.generate_ids(
            ctx_flat, n_steps=n_steps, role_id=0,
        ).view(B, M, -1)

        val_ids = self.triple_expander.generate_ids(
            ctx_flat, n_steps=n_steps, role_id=1,
        ).view(B, M, -1)

        attr_preds = self.forward_attr(bottleneck).argmax(-1)

        return ent_ids, val_ids, attr_preds

    @torch.no_grad()
    def generate_text_ids(
        self,
        bottleneck: torch.Tensor,
        triple_pad_mask: torch.Tensor | None = None,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Generate text token IDs from bottleneck.

        Returns:
            (B, T) text token IDs
        """
        return self.text_expander.generate(
            bottleneck, triple_pad_mask=triple_pad_mask, n_steps=n_steps,
        )

    # ── Alignment metric ─────────────────────────────────────────────

    @torch.no_grad()
    def bottleneck_alignment(
        self,
        triple_bottleneck: torch.Tensor,
        text_bottleneck: torch.Tensor,
        triple_pad: torch.Tensor,
    ) -> float:
        """Cosine similarity between triple and text compressor outputs.

        Higher = better alignment in the shared bottleneck space.
        """
        B, T, D = triple_bottleneck.shape
        M = self.config.max_triples
        pad_3x = triple_pad.unsqueeze(-1).expand(B, M, 3).reshape(B, M * 3)
        valid = ~pad_3x  # (B, M*3)

        if not valid.any():
            return 0.0

        tri_vecs = triple_bottleneck[valid]
        txt_vecs = text_bottleneck[valid]

        cos = F.cosine_similarity(tri_vecs, txt_vecs, dim=-1)
        return cos.mean().item()

    # ── Param counts ─────────────────────────────────────────────────

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def component_param_counts(self) -> dict[str, int]:
        return {
            "shared_emb": sum(p.numel() for p in self.shared_token_emb.parameters()),
            "triple_compressor": self.triple_compressor.param_count(),
            "text_compressor": self.text_compressor.param_count(),
            "triple_expander": self.triple_expander.param_count(),
            "text_expander": self.text_expander.param_count(),
            "attr_head": sum(p.numel() for p in self.attr_head.parameters())
                       + sum(p.numel() for p in self.ln_attr.parameters()),
            "triple_length_head": sum(p.numel() for p in self.triple_length_head.parameters()),
        }
