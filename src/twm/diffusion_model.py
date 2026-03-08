"""Hybrid TWM with diffusion decoders.

Entity: diffusion decoder (open vocabulary via T5 tokens)
Attr: discrete classification head (small closed vocab, works at ~75%)
Value: diffusion decoder — iterative denoising conditioned on TWM latent

Supports unified mode: single decoder with role embeddings (entity=0, value=1).
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .modules import TransformerDynamics
from .sentence_encoder import SentenceEncoder
from .token_encoder import TokenEncoder
from .diffusion_decoder import DiffusionDecoder
from .phrase_vocab import PhraseVocab


class DiffusionWorldModel(nn.Module):
    """TWM with diffusion entity/value decoders and discrete attr head."""

    ROLE_ENTITY = 0
    ROLE_VALUE = 1

    def __init__(
        self,
        config: ModelConfig,
        st_dim: int,
        vocab: PhraseVocab,
        max_value_tokens: int = 16,
        n_proj_tokens: int = 8,
        denoiser_layers: int = 4,
        denoiser_dim: int = 512,
        denoiser_heads: int = 8,
        dropout: float = 0.1,
        token_vocab_size: int = 32100,
        mask_token_id: int = 1,
        tokenizer=None,
        use_film: bool = False,
        use_cross_attention: bool = True,
        use_adaln: bool = False,
        use_continuous_noise: bool = False,
        normalize_noise: bool = True,
        alpha_min: float = 0.0,
        timestep_bias_power: float = 1.0,
        unified_decoder: bool = False,
        wspace: bool = False,
        use_structured_noise: bool = False,
        use_mse_prediction: bool = False,
        cond_drop_prob: float = 0.0,
        use_decode_proj: bool = False,
        token_level: bool = False,
        max_tokens_per_slot: int = 12,
    ):
        super().__init__()
        self.config = config
        self.st_dim = st_dim
        self.vocab = vocab
        self.unified_decoder = unified_decoder
        self.wspace = wspace
        self.token_level = token_level
        self.max_tokens_per_slot = max_tokens_per_slot

        sizes = vocab.vocab_sizes
        if token_level:
            self.encoder = TokenEncoder(config, max_tokens_per_slot=max_tokens_per_slot)
        else:
            self.encoder = SentenceEncoder(config, st_dim)
        self.dynamics = TransformerDynamics(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Discrete head for attr only (small closed vocab, works well)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.attr_head = nn.Linear(config.d_model, sizes["attr"])

        # Length predictor: how many real BPE tokens per slot
        # Applied per-role to 256d latent vectors (entity/value separately)
        self.length_head = nn.Linear(config.d_model, 1)

        decoder_kwargs = dict(
            twm_dim=config.d_model,
            n_proj_tokens=n_proj_tokens,
            max_seq_len=max_value_tokens,
            vocab_size=token_vocab_size,
            d_model=denoiser_dim,
            n_heads=denoiser_heads,
            n_layers=denoiser_layers,
            dropout=dropout,
            mask_token_id=mask_token_id,
            tokenizer=tokenizer,
            use_film=use_film,
            use_cross_attention=use_cross_attention,
            use_adaln=use_adaln,
            use_continuous_noise=use_continuous_noise,
            normalize_noise=normalize_noise,
            alpha_min=alpha_min,
            timestep_bias_power=timestep_bias_power,
            wspace=wspace,
            use_structured_noise=use_structured_noise,
            use_mse_prediction=use_mse_prediction,
            cond_drop_prob=cond_drop_prob,
            use_decode_proj=use_decode_proj,
        )

        if unified_decoder:
            # Single decoder with role embeddings (entity=0, value=1)
            self.triple_decoder = DiffusionDecoder(**decoder_kwargs, n_roles=2)
        else:
            # Separate decoders (legacy)
            self.entity_decoder = DiffusionDecoder(**decoder_kwargs)
            self.value_decoder = DiffusionDecoder(**decoder_kwargs)

    def _get_decoder(self, role: str) -> DiffusionDecoder:
        if self.unified_decoder:
            return self.triple_decoder
        return self.entity_decoder if role == "entity" else self.value_decoder

    def _get_role_id(self, role: str) -> int | None:
        if not self.unified_decoder:
            return None
        return self.ROLE_ENTITY if role == "entity" else self.ROLE_VALUE

    def encode_dynamics(
        self,
        input_embeds: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.token_level:
            latent, _raw = self.encoder(input_embeds, pad_mask)
        else:
            latent, _raw = self.encoder(input_embeds)
        # Cache pad_mask for token-level _extract_triple_context
        self._cached_pad_mask = pad_mask
        return self.dynamics(latent, src_key_padding_mask=pad_mask)

    def _extract_triple_context(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """(B, T, d_model) -> (B, M, 3*d_model) concatenated entity+attr+value.

        In token-level mode, each slot has S tokens. Mean-pool non-pad tokens
        per slot to produce one vector per role per triple.
        """
        M = self.config.max_triples
        device = latent.device

        if not self.token_level:
            e_idx = torch.arange(0, M * 3, 3, device=device)
            a_idx = torch.arange(1, M * 3, 3, device=device)
            v_idx = torch.arange(2, M * 3, 3, device=device)
            return torch.cat([latent[:, e_idx], latent[:, a_idx], latent[:, v_idx]], dim=-1)

        # Token-level: latent is (B, M*3*S, d_model)
        S = self.max_tokens_per_slot
        B, T, D = latent.shape
        pad_mask = getattr(self, '_cached_pad_mask', None)

        # Reshape to (B, M*3, S, D) — one group per slot
        grouped = latent.view(B, M * 3, S, D)

        if pad_mask is not None:
            # pad_mask: (B, M*3*S), True=pad
            mask = pad_mask.view(B, M * 3, S)  # (B, M*3, S)
            # Count non-pad tokens per slot
            non_pad_count = (~mask).float().sum(dim=-1, keepdim=True).clamp(min=1)  # (B, M*3, 1)
            # Zero out padded positions before summing
            grouped = grouped.masked_fill(mask.unsqueeze(-1), 0.0)
            pooled = grouped.sum(dim=2) / non_pad_count  # (B, M*3, D)
        else:
            pooled = grouped.mean(dim=2)  # (B, M*3, D)

        # Now pooled is (B, M*3, D) — same shape as sentence-level latent
        e_idx = torch.arange(0, M * 3, 3, device=device)
        a_idx = torch.arange(1, M * 3, 3, device=device)
        v_idx = torch.arange(2, M * 3, 3, device=device)
        return torch.cat([pooled[:, e_idx], pooled[:, a_idx], pooled[:, v_idx]], dim=-1)

    def forward_discrete(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.ln_f(latent)
        M = self.config.max_triples
        device = latent.device

        if not self.token_level:
            a_idx = torch.arange(1, M * 3, 3, device=device)
            return {"attr": self.attr_head(x[:, a_idx])}

        # Token-level: mean-pool attr slot tokens before classification
        S = self.max_tokens_per_slot
        B, T, D = x.shape
        pad_mask = getattr(self, '_cached_pad_mask', None)

        grouped = x.view(B, M * 3, S, D)
        # Attr slots are at indices 1, 4, 7, ... (role index 1 within each triple)
        a_idx = torch.arange(1, M * 3, 3, device=device)
        attr_grouped = grouped[:, a_idx]  # (B, M, S, D)

        if pad_mask is not None:
            mask = pad_mask.view(B, M * 3, S)[:, a_idx]  # (B, M, S)
            non_pad = (~mask).float().sum(dim=-1, keepdim=True).clamp(min=1)
            attr_grouped = attr_grouped.masked_fill(mask.unsqueeze(-1), 0.0)
            attr_pooled = attr_grouped.sum(dim=2) / non_pad  # (B, M, D)
        else:
            attr_pooled = attr_grouped.mean(dim=2)

        return {"attr": self.attr_head(attr_pooled)}

    def forward_lengths(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict entity/value token lengths from per-role latent vectors.

        Returns:
            entity_lengths: (B, M) raw regression values
            value_lengths: (B, M) raw regression values
        """
        M = self.config.max_triples
        device = latent.device
        e_idx = torch.arange(0, M * 3, 3, device=device)
        v_idx = torch.arange(2, M * 3, 3, device=device)
        ent_pred = self.length_head(latent[:, e_idx]).squeeze(-1)  # (B, M)
        val_pred = self.length_head(latent[:, v_idx]).squeeze(-1)  # (B, M)
        return ent_pred, val_pred

    @torch.no_grad()
    def predict_lengths(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict and round entity/value token lengths, clamped to [1, max_seq_len]."""
        max_len = self._get_decoder("entity").max_seq_len
        ent_raw, val_raw = self.forward_lengths(latent)
        return (
            ent_raw.round().long().clamp(1, max_len),
            val_raw.round().long().clamp(1, max_len),
        )

    def _forward_diffusion(
        self,
        role: str,
        latent: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run diffusion forward pass, filtering pad triples."""
        decoder = self._get_decoder(role)
        role_id = self._get_role_id(role)

        triple_ctx = self._extract_triple_context(latent)
        B, M, D = triple_ctx.shape

        ctx_flat = triple_ctx.reshape(B * M, D)
        tgt_flat = target_ids.reshape(B * M, -1)
        pad_flat = target_pad_mask.reshape(B * M)

        valid = ~pad_flat
        if not valid.any():
            S = target_ids.shape[-1]
            empty_logits = torch.zeros(0, S, decoder.vocab_size, device=latent.device)
            empty_mask = torch.zeros(0, S, dtype=torch.bool, device=latent.device)
            return empty_logits, empty_mask

        # Expand per-batch mask_ratio/timestep to per-triple
        valid_mask_ratio = None
        if mask_ratio is not None:
            per_triple = mask_ratio.unsqueeze(1).expand(B, M).reshape(B * M)
            valid_mask_ratio = per_triple[valid]

        valid_timestep = None
        if timestep is not None:
            per_triple_t = timestep.unsqueeze(1).expand(B, M).reshape(B * M)
            valid_timestep = per_triple_t[valid]

        return decoder(ctx_flat[valid], tgt_flat[valid],
                       mask_ratio=valid_mask_ratio, timestep=valid_timestep,
                       role_id=role_id)

    def forward_entity(
        self,
        latent: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_diffusion("entity", latent, target_ids, target_pad_mask,
                                       mask_ratio=mask_ratio, timestep=timestep)

    def forward_value(
        self,
        latent: torch.Tensor,
        target_ids: torch.Tensor,
        target_pad_mask: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_diffusion("value", latent, target_ids, target_pad_mask,
                                       mask_ratio=mask_ratio, timestep=timestep)

    def _generate_texts(
        self,
        role: str,
        latent: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        cosine_schedule: bool = True,
        soft: bool = False,
        guidance_scale: float = 1.0,
    ) -> list[list[str]]:
        decoder = self._get_decoder(role)
        role_id = self._get_role_id(role)

        triple_ctx = self._extract_triple_context(latent)
        B, M, D = triple_ctx.shape
        ctx_flat = triple_ctx.reshape(B * M, D)
        texts = decoder.generate(
            ctx_flat, n_steps=n_steps,
            temperature=temperature, cosine_schedule=cosine_schedule,
            role_id=role_id, soft=soft, guidance_scale=guidance_scale,
        )
        return [texts[i * M:(i + 1) * M] for i in range(B)]

    def _generate_ids(
        self,
        role: str,
        latent: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        soft: bool = False,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate raw token IDs (B, M, S) without string roundtrip."""
        decoder = self._get_decoder(role)
        role_id = self._get_role_id(role)

        triple_ctx = self._extract_triple_context(latent)
        B, M, D = triple_ctx.shape
        ctx_flat = triple_ctx.reshape(B * M, D)
        ids = decoder.generate_ids(
            ctx_flat, n_steps=n_steps,
            temperature=temperature,
            role_id=role_id, soft=soft, guidance_scale=guidance_scale,
        )  # (B*M, S)
        S = ids.shape[-1]
        return ids.view(B, M, S)

    @torch.no_grad()
    def generate_entities(
        self, latent: torch.Tensor, n_steps: int = 10,
        temperature: float = 0.0, cosine_schedule: bool = True,
        soft: bool = False, guidance_scale: float = 1.0,
    ) -> list[list[str]]:
        return self._generate_texts(
            "entity", latent, n_steps, temperature, cosine_schedule,
            soft=soft, guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def generate_values(
        self, latent: torch.Tensor, n_steps: int = 10,
        temperature: float = 0.0, cosine_schedule: bool = True,
        soft: bool = False, guidance_scale: float = 1.0,
    ) -> list[list[str]]:
        return self._generate_texts(
            "value", latent, n_steps, temperature, cosine_schedule,
            soft=soft, guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def generate_entity_ids(
        self, latent: torch.Tensor, n_steps: int = 10,
        temperature: float = 0.0,
        soft: bool = False, guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._generate_ids(
            "entity", latent, n_steps, temperature,
            soft=soft, guidance_scale=guidance_scale,
        )

    @torch.no_grad()
    def generate_value_ids(
        self, latent: torch.Tensor, n_steps: int = 10,
        temperature: float = 0.0,
        soft: bool = False, guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._generate_ids(
            "value", latent, n_steps, temperature,
            soft=soft, guidance_scale=guidance_scale,
        )

    def load_dynamics_from_checkpoint(self, checkpoint_path: str):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        dynamics_sd = {}
        for key, val in sd.items():
            if key.startswith("dynamics."):
                dynamics_sd[key.removeprefix("dynamics.")] = val
            elif key.startswith("encoder.encoder."):
                dynamics_sd[key.removeprefix("encoder.")] = val
        self.dynamics.load_state_dict(dynamics_sd, strict=True)

    def load_encoder_from_sentence_model(self, checkpoint_path: str):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_sd = {
            k.removeprefix("encoder."): v
            for k, v in sd.items()
            if k.startswith("encoder.") and not k.startswith("encoder.encoder.")
        }
        if encoder_sd:
            self.encoder.load_state_dict(encoder_sd, strict=True)

    def load_decoder_from_checkpoint(self, checkpoint_path: str, source_key: str = "value_decoder"):
        """Load unified triple_decoder weights from a v9 checkpoint's entity or value decoder.

        Args:
            checkpoint_path: path to v9 model checkpoint
            source_key: which decoder to load from ("entity_decoder" or "value_decoder")
        """
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        decoder_sd = {}
        prefix = f"{source_key}."
        current_sd = self.triple_decoder.state_dict()
        skipped = []
        for key, val in sd.items():
            if key.startswith(prefix):
                new_key = key.removeprefix(prefix)
                # Skip keys with shape mismatch (e.g. old time_embed dimensions)
                if new_key in current_sd and current_sd[new_key].shape != val.shape:
                    skipped.append(new_key)
                    continue
                decoder_sd[new_key] = val
        # Load with strict=False to allow missing role_emb and skipped keys
        missing, unexpected = self.triple_decoder.load_state_dict(decoder_sd, strict=False)
        print(f"  Loaded {len(decoder_sd)} keys from {source_key}")
        if skipped:
            print(f"  Skipped (shape mismatch, will train from scratch): {skipped}")
        if missing:
            print(f"  Missing (expected): {missing}")
        if unexpected:
            print(f"  Unexpected: {unexpected}")

    def freeze_dynamics(self):
        for p in self.dynamics.parameters():
            p.requires_grad = False

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
