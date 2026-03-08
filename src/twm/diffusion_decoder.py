"""Diffusion decoder for Triple World Model.

Supports two noise processes:
  - Discrete masking (LLaDA-style): binary mask/reveal, iterative unmasking
  - Continuous noise (DDPM-style): Gaussian corruption in embedding space,
    iterative denoising with timestep conditioning

Conditioning modes:
  - cross-attention: projected memory tokens from TWM latent
  - adaLN-Zero (DiT-style): adaptive layer normalization with zero-init gates
  - FiLM: per-position scale+shift (legacy)

Set-to-set: every denoising step predicts all positions simultaneously.
No autoregressive left-to-right bias, no EOS problem.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_noise_schedule(t: torch.Tensor, s: float = 0.008, alpha_min: float = 0.0) -> torch.Tensor:
    """Maps t in [0,1] to alpha_t in [~1, alpha_min].

    Standard DDPM cosine schedule, with optional alpha_min floor to avoid
    the singularity at t=1.0 where alpha=0 (zero signal). Setting alpha_min>0
    ensures the model always sees a faint trace of the original embedding,
    matching how image diffusion avoids pure noise.
    """
    f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f_t / f_0).clamp(min=max(alpha_min, 1e-6), max=1.0)


def importance_sample_timesteps(batch_size: int, device: torch.device, bias_power: float = 2.0) -> torch.Tensor:
    """Sample timesteps biased toward high-t (high noise) where the model struggles most.

    Uses u^(1/bias_power) where u ~ U(0,1). This concentrates samples toward 1.0.
    bias_power=1.0 recovers uniform sampling.
    bias_power=2.0 means ~50% of samples fall in [0.7, 1.0].
    """
    u = torch.rand(batch_size, device=device)
    return u.pow(1.0 / bias_power)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar timesteps.

    Args:
        t: (B,) timestep values in [0, 1]
        dim: embedding dimension
    Returns:
        (B, dim) embeddings
    """
    half_dim = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
    )
    emb = t.unsqueeze(-1) * freq.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Projects sinusoidal timestep encoding to conditioning dimension.

    Works in a small internal dimension (embed_dim) then projects up to
    out_dim. This avoids a massive MLP when out_dim is large.
    """

    def __init__(self, out_dim: int, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.proj_out = nn.Linear(embed_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps in [0, 1]
        Returns:
            (B, out_dim) timestep embeddings
        """
        emb = sinusoidal_embedding(t, self.embed_dim)
        return self.proj_out(self.mlp(emb))


class FiLMConditioner(nn.Module):
    """Projects TWM context into per-position scale+shift for FiLM conditioning."""

    def __init__(self, context_dim: int, d_model: int, n_positions: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(context_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, n_positions * d_model * 2),
        )
        self.n_positions = n_positions
        self.d_model = d_model

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.proj(context)
        out = out.view(-1, self.n_positions, self.d_model * 2)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma + 1.0, beta


class AdaLNZeroLayer(nn.Module):
    """Transformer layer with adaLN-Zero conditioning (DiT-style).

    Replaces standard LayerNorm with adaptive normalization where gamma/beta
    come from a conditioning projection. Output gates (alpha) are zero-initialized
    so the layer starts as identity.

    Supports optional cross-attention to projected memory tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_dim: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.use_cross_attention = use_cross_attention

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Cross-attention (optional)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        # adaLN projection: context -> (gamma, beta, gate) for each sub-layer
        # With cross-attention: 3 sub-layers x 3 params = 9 outputs
        # Without: 2 sub-layers x 3 params = 6 outputs
        n_params = 9 if use_cross_attention else 6
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, d_model * n_params),
        )
        # Zero-initialize so layer starts as identity
        nn.init.zeros_(self.adaln_proj[-1].weight)
        nn.init.zeros_(self.adaln_proj[-1].bias)

        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) token representations
            context: (B, context_dim) conditioning vector (per-triple)
            memory: (B, K, d_model) cross-attention keys/values (optional)
        """
        B, T, D = x.shape

        # Get all adaLN parameters at once
        params = self.adaln_proj(context)  # (B, d_model * n_params)
        if self.use_cross_attention:
            params = params.view(B, 9, D)
            gamma1, beta1, alpha1 = params[:, 0], params[:, 1], params[:, 2]
            gamma2, beta2, alpha2 = params[:, 3], params[:, 4], params[:, 5]
            gamma3, beta3, alpha3 = params[:, 6], params[:, 7], params[:, 8]
        else:
            params = params.view(B, 6, D)
            gamma1, beta1, alpha1 = params[:, 0], params[:, 1], params[:, 2]
            gamma3, beta3, alpha3 = params[:, 3], params[:, 4], params[:, 5]

        # Unsqueeze for broadcasting: (B, D) -> (B, 1, D)
        gamma1 = gamma1.unsqueeze(1)
        beta1 = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma3 = gamma3.unsqueeze(1)
        beta3 = beta3.unsqueeze(1)
        alpha3 = alpha3.unsqueeze(1)

        # Self-attention with adaLN-Zero
        x_norm = self.norm1(x) * (1 + gamma1) + beta1
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + alpha1 * sa_out

        # Cross-attention with adaLN-Zero (if enabled)
        if self.use_cross_attention and memory is not None:
            gamma2 = gamma2.unsqueeze(1)
            beta2 = beta2.unsqueeze(1)
            alpha2 = alpha2.unsqueeze(1)
            x_norm = self.norm2(x) * (1 + gamma2) + beta2
            ca_out, _ = self.cross_attn(x_norm, memory, memory)
            x = x + alpha2 * ca_out

        # FFN with adaLN-Zero
        x_norm = self.norm3(x) * (1 + gamma3) + beta3
        ff_out = self.ffn(x_norm)
        x = x + alpha3 * ff_out

        return x


class DiffusionDecoder(nn.Module):
    """Diffusion decoder with configurable noise process and conditioning.

    Supports two modes:
        - Discrete masking (default): binary mask tokens, iterative unmasking
        - Continuous noise: Gaussian corruption in embedding space with
          timestep conditioning, iterative DDPM-style denoising

    Structured noise (optional): concentrates noise along local ambiguity
    directions in embedding space, forcing the denoiser to rely on conditioning
    to disambiguate nearby tokens rather than exploiting proximity shortcuts.

    MSE x₀-prediction (optional): denoiser predicts clean embedding vectors
    directly instead of logits over vocabulary. Loss is MSE in embedding space.
    Inference uses nearest-neighbor lookup against token embeddings. Eliminates
    the output head and keeps the entire pipeline continuous.
    """

    def __init__(
        self,
        twm_dim: int = 256,
        n_proj_tokens: int = 8,
        max_seq_len: int = 16,
        vocab_size: int = 32100,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        mask_token_id: int = 1,
        tokenizer=None,
        use_film: bool = False,
        use_cross_attention: bool = True,
        use_adaln: bool = False,
        use_continuous_noise: bool = False,
        normalize_noise: bool = True,
        alpha_min: float = 0.0,
        timestep_bias_power: float = 1.0,
        n_roles: int = 0,
        wspace: bool = False,
        use_structured_noise: bool = False,
        use_mse_prediction: bool = False,
        cond_drop_prob: float = 0.0,
        use_decode_proj: bool = False,
    ):
        super().__init__()
        self.twm_dim = twm_dim
        self.n_proj_tokens = n_proj_tokens
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id
        self.use_film = use_film
        self.use_cross_attention = use_cross_attention
        self.use_adaln = use_adaln
        self.use_continuous_noise = use_continuous_noise
        self.normalize_noise = normalize_noise
        self.alpha_min = alpha_min
        self.timestep_bias_power = timestep_bias_power
        self.n_roles = n_roles
        self.wspace = wspace
        self.use_structured_noise = use_structured_noise
        self.use_mse_prediction = use_mse_prediction
        self.cond_drop_prob = cond_drop_prob
        self.use_decode_proj = use_decode_proj

        if d_ff is None:
            d_ff = d_model * 4  # Standard 4x expansion, not hardcoded 2048

        input_dim = twm_dim * 3

        # Role embedding for unified decoder (entity=0, value=1)
        if n_roles > 0:
            self.role_emb = nn.Embedding(n_roles, input_dim)

        # Cross-attention memory: W-space reshapes directly, legacy uses projection MLP
        if use_cross_attention and not wspace:
            output_dim = n_proj_tokens * d_model
            hidden_dim = max(input_dim, output_dim // 2)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        # Token + position embeddings (token_emb frozen — just a lookup table)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.token_emb.weight.requires_grad = False
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Timestep embedding for continuous noise mode
        if use_continuous_noise:
            self.time_embed = TimestepEmbedding(input_dim, embed_dim=d_model)

        # Denoising layers
        if use_adaln:
            # adaLN-Zero: custom layers with adaptive normalization
            self.layers = nn.ModuleList([
                AdaLNZeroLayer(
                    d_model=d_model, n_heads=n_heads,
                    context_dim=input_dim, d_ff=d_ff,
                    dropout=dropout,
                    use_cross_attention=use_cross_attention,
                )
                for _ in range(n_layers)
            ])
        elif use_cross_attention:
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    batch_first=True, norm_first=True,
                )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    batch_first=True, norm_first=True,
                )
                for _ in range(n_layers)
            ])

        # FiLM conditioner (legacy)
        if use_film:
            self.film = FiLMConditioner(input_dim, d_model, max_seq_len)

        # adaLN-Zero uses its own per-layer norms; still need final norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output head: logits for CE mode, omitted for MSE x₀-prediction
        if not use_mse_prediction:
            self.output_head = nn.Linear(d_model, vocab_size)

        # Learned projection for NN decode sharpening
        if use_decode_proj:
            self.decode_proj = nn.Linear(d_model, d_model)

        self.tokenizer = tokenizer

    def project_context(self, triple_context: torch.Tensor) -> torch.Tensor:
        B = triple_context.shape[0]
        if self.wspace:
            # Reshape 3*twm_dim → (3, d_model) — raw W-space memory tokens
            return triple_context.view(B, 3, self.d_model)
        return self.projection(triple_context).view(B, self.n_proj_tokens, self.d_model)

    def _run_denoiser(self, x: torch.Tensor, triple_context: torch.Tensor) -> torch.Tensor:
        memory = None
        if self.use_cross_attention:
            memory = self.project_context(triple_context)

        if self.use_adaln:
            for layer in self.layers:
                x = layer(x, triple_context, memory)
        else:
            for layer in self.layers:
                if self.use_film:
                    gamma, beta = self.film(triple_context)
                    x = gamma * x + beta

                if self.use_cross_attention:
                    x = layer(x, memory)
                else:
                    x = layer(x)

        return x

    def _make_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Generate noise, optionally normalized to match embedding magnitude."""
        noise = torch.randn_like(like)
        if self.normalize_noise:
            emb_norm = self.token_emb.weight.norm(dim=-1).mean()
            noise = noise * emb_norm / (noise.norm(dim=-1, keepdim=True) + 1e-8)
        return noise

    def _nn_decode(self, pred_emb: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbor decode: find closest token embedding for each position.

        If decode_proj exists, applies learned sharpening before cosine lookup.

        Args:
            pred_emb: (B, T, d_model) predicted clean embeddings
        Returns:
            (B, T) token IDs
        """
        emb = self.decode_proj(pred_emb) if self.use_decode_proj else pred_emb
        pred_norm = F.normalize(emb, dim=-1)
        emb_norm = F.normalize(self.token_emb.weight, dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.T)
        return sims.argmax(dim=-1)

    def decode_proj_logits(self, pred_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity logits through decode_proj for CE aux loss.

        Args:
            pred_emb: (B, T, d_model) predicted clean embeddings
        Returns:
            (B, T, vocab_size) similarity logits
        """
        emb = self.decode_proj(pred_emb) if self.use_decode_proj else pred_emb
        pred_norm = F.normalize(emb, dim=-1)
        emb_norm = F.normalize(self.token_emb.weight.detach(), dim=-1)
        return torch.matmul(pred_norm, emb_norm.T)

    def build_structured_noise_dirs(self, k: int = 10) -> None:
        """Precompute per-token neighbor difference vectors for structured noise.

        For each token embedding, finds k nearest neighbors and stores the
        difference vectors. Structured noise is sampled as a random linear
        combination of these directions, concentrating perturbation along
        the axes that connect nearby embeddings.

        Must be called after embedding initialization (e.g., W-space init).
        """
        with torch.no_grad():
            embs = self.token_emb.weight.detach()  # (V, D)
            V, D = embs.shape
            k = min(k, V - 1)

            # Cosine similarity to find neighbors
            norms = embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normed = embs / norms
            sims = normed @ normed.T  # (V, V)

            # Exclude self (set diagonal to -inf)
            sims.fill_diagonal_(-float('inf'))

            _, neighbor_idx = sims.topk(k, dim=-1)  # (V, k)
            diffs = embs[neighbor_idx] - embs.unsqueeze(1)  # (V, k, D)

            # Register as buffer so it moves with the model but isn't a parameter
            self.register_buffer('_neighbor_diffs', diffs)

    def _make_structured_noise(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Sample noise concentrated along local ambiguity directions.

        Args:
            token_ids: (B, T) ground truth token IDs
        Returns:
            (B, T, d_model) structured noise vectors
        """
        diffs = self._neighbor_diffs[token_ids]  # (B, T, k, D)
        B, T, k, D = diffs.shape
        weights = torch.randn(B, T, k, 1, device=diffs.device)
        noise = (weights * diffs).sum(dim=2)  # (B, T, D)
        # Normalize to match embedding magnitude (same as _make_noise)
        if self.normalize_noise:
            emb_norm = self.token_emb.weight.norm(dim=-1).mean()
            noise = noise * emb_norm / (noise.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            noise = F.normalize(noise, dim=-1) * math.sqrt(D)
        return noise

    # ── Discrete masking forward ──────────────────────────────────────

    def _forward_discrete(
        self,
        triple_context: torch.Tensor,
        target_ids: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        role_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = target_ids.shape
        device = target_ids.device

        # Add role embedding to context
        if role_id is not None and self.n_roles > 0:
            role = self.role_emb(torch.full((B,), role_id, device=device, dtype=torch.long))
            triple_context = triple_context + role

        if mask_ratio is None:
            mask_ratio = torch.rand(B, device=device)

        n_to_mask = (mask_ratio * S).clamp(min=1).long()

        rand_scores = torch.rand(B, S, device=device)
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        for i in range(B):
            _, indices = rand_scores[i].topk(n_to_mask[i].item(), largest=False)
            mask[i, indices] = True

        noisy_ids = target_ids.clone()
        noisy_ids[mask] = self.mask_token_id

        pos_idx = torch.arange(S, device=device).unsqueeze(0)
        x = self.token_emb(noisy_ids) + self.pos_emb(pos_idx)

        decoded = self._run_denoiser(x, triple_context)
        decoded = self.ln_f(decoded)
        logits = self.output_head(decoded)

        return logits, mask

    # ── Continuous noise forward ──────────────────────────────────────

    def _forward_continuous(
        self,
        triple_context: torch.Tensor,
        target_ids: torch.Tensor,
        timestep: torch.Tensor | None = None,
        role_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = target_ids.shape
        device = target_ids.device

        # Add role embedding to context
        if role_id is not None and self.n_roles > 0:
            role = self.role_emb(torch.full((B,), role_id, device=device, dtype=torch.long))
            triple_context = triple_context + role

        # Conditioning dropout for CFG training: zero out triple_context
        # on a fraction of examples so the model learns unconditioned behavior
        if self.training and self.cond_drop_prob > 0:
            drop_mask = torch.rand(B, 1, device=device) < self.cond_drop_prob
            triple_context = triple_context.masked_fill(drop_mask, 0.0)

        # Sample timestep if not provided
        if timestep is None:
            timestep = importance_sample_timesteps(B, device, self.timestep_bias_power)

        # Apply noise schedule with alpha_min floor
        alpha_t = cosine_noise_schedule(timestep, alpha_min=self.alpha_min)  # (B,)
        alpha_t = alpha_t.view(B, 1, 1)  # (B, 1, 1) for broadcasting

        # Get clean embeddings
        original_emb = self.token_emb(target_ids)  # (B, S, d_model)

        # Corrupt with noise (structured or isotropic)
        if self.use_structured_noise and hasattr(self, '_neighbor_diffs'):
            noise = self._make_structured_noise(target_ids)
        else:
            noise = self._make_noise(original_emb)
        corrupted = torch.sqrt(alpha_t) * original_emb + torch.sqrt(1 - alpha_t) * noise

        # Add positional embeddings
        pos_idx = torch.arange(S, device=device).unsqueeze(0)
        x = corrupted + self.pos_emb(pos_idx)

        # Add timestep to conditioning
        t_emb = self.time_embed(timestep)  # (B, input_dim)
        conditioned_context = triple_context + t_emb

        # Run denoiser
        decoded = self._run_denoiser(x, conditioned_context)
        decoded = self.ln_f(decoded)

        # All positions are targets in continuous mode
        all_mask = torch.ones(B, S, dtype=torch.bool, device=device)

        if self.use_mse_prediction:
            # Return predicted clean embeddings directly — no output head
            return decoded, all_mask
        else:
            logits = self.output_head(decoded)
            return logits, all_mask

    # ── Unified forward ───────────────────────────────────────────────

    def forward(
        self,
        triple_context: torch.Tensor,
        target_ids: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        role_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_continuous_noise:
            return self._forward_continuous(triple_context, target_ids, timestep, role_id=role_id)
        else:
            return self._forward_discrete(triple_context, target_ids, mask_ratio, role_id=role_id)

    # ── Discrete masking generation ───────────────────────────────────

    def _generate_discrete(
        self,
        triple_context: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        cosine_schedule: bool = True,
        role_id: int | None = None,
    ) -> list[str]:
        """Iterative unmasking: start fully masked, reveal tokens over N steps."""
        B = triple_context.shape[0]
        S = self.max_seq_len
        device = triple_context.device

        # Add role embedding to context
        if role_id is not None and self.n_roles > 0:
            role = self.role_emb(torch.full((B,), role_id, device=device, dtype=torch.long))
            triple_context = triple_context + role

        ids = torch.full((B, S), self.mask_token_id, dtype=torch.long, device=device)
        is_masked = torch.ones(B, S, dtype=torch.bool, device=device)

        pos_idx = torch.arange(S, device=device).unsqueeze(0)

        for step in range(n_steps):
            if cosine_schedule:
                t = (step + 1) / n_steps
                frac_unmasked = 1.0 - math.cos(t * math.pi / 2)
                n_target_unmasked = max(1, int(frac_unmasked * S))
            else:
                n_target_unmasked = max(1, int((step + 1) / n_steps * S))

            x = self.token_emb(ids) + self.pos_emb(pos_idx)
            decoded = self._run_denoiser(x, triple_context)
            decoded = self.ln_f(decoded)
            logits = self.output_head(decoded)

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                flat_probs = probs.reshape(-1, probs.shape[-1])
                sampled = torch.multinomial(flat_probs, 1).reshape(B, S)
                confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                pred_ids = sampled
            else:
                probs = F.softmax(logits, dim=-1)
                confidence, pred_ids = probs.max(dim=-1)

            confidence[~is_masked] = float('inf')

            for i in range(B):
                n_currently_unmasked = (~is_masked[i]).sum().item()
                n_to_reveal = max(0, n_target_unmasked - n_currently_unmasked)
                n_reveal = min(n_to_reveal, is_masked[i].sum().item())
                if n_reveal > 0:
                    masked_confidence = confidence[i].clone()
                    masked_confidence[~is_masked[i]] = -1.0
                    _, top_idx = masked_confidence.topk(n_reveal)
                    ids[i, top_idx] = pred_ids[i, top_idx]
                    is_masked[i, top_idx] = False

        # Final pass
        if is_masked.any():
            x = self.token_emb(ids) + self.pos_emb(pos_idx)
            decoded = self._run_denoiser(x, triple_context)
            decoded = self.ln_f(decoded)
            logits = self.output_head(decoded)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                flat_probs = probs.reshape(-1, probs.shape[-1])
                pred_ids = torch.multinomial(flat_probs, 1).reshape(B, S)
            else:
                pred_ids = logits.argmax(dim=-1)
            ids[is_masked] = pred_ids[is_masked]

        texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ── Continuous noise generation ───────────────────────────────────

    def _denoise_step(
        self,
        x: torch.Tensor,
        triple_context: torch.Tensor,
        t_emb: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step: returns predicted clean embedding (MSE) or decoded features (CE)."""
        conditioned_context = triple_context + t_emb
        x_input = x + pos_idx
        decoded = self._run_denoiser(x_input, conditioned_context)
        return self.ln_f(decoded)

    def _generate_continuous(
        self,
        triple_context: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        role_id: int | None = None,
        soft: bool = False,
        guidance_scale: float = 1.0,
    ) -> list[str]:
        """DDPM-style iterative denoising: start from near-pure noise, denoise over N steps.

        In MSE mode: fully continuous — denoiser output IS the predicted clean
        embedding. Intermediate steps re-noise the prediction directly. NN decode
        only at the very end.

        With guidance_scale > 1.0 and MSE mode: classifier-free guidance in
        embedding space. Runs conditioned and unconditioned passes, combines
        predictions directionally.

        In CE mode with soft=True: intermediate steps use probability-weighted
        soft embeddings instead of hard argmax.
        """
        B = triple_context.shape[0]
        S = self.max_seq_len
        device = triple_context.device
        use_cfg = self.use_mse_prediction and guidance_scale != 1.0

        # Add role embedding to context
        if role_id is not None and self.n_roles > 0:
            role = self.role_emb(torch.full((B,), role_id, device=device, dtype=torch.long))
            triple_context = triple_context + role

        # Unconditioned context for CFG (zero triple context, timestep added per step)
        uncond_context = torch.zeros_like(triple_context)

        # Start from maximum noise level (alpha_min, not zero)
        x = self._make_noise(torch.zeros(B, S, self.d_model, device=device))

        pos_emb = self.pos_emb(torch.arange(S, device=device).unsqueeze(0))

        # Step from t=1.0 to t=0.0
        schedule = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        soft_temp = max(temperature, 1e-6) if soft else temperature

        for i in range(n_steps):
            t_now = schedule[i]
            t_next = schedule[i + 1]
            alpha_next = cosine_noise_schedule(t_next.unsqueeze(0), alpha_min=self.alpha_min)

            t_batch = t_now.expand(B)
            t_emb = self.time_embed(t_batch)

            decoded = self._denoise_step(x, triple_context, t_emb, pos_emb)

            if use_cfg:
                # CFG: combine conditioned and unconditioned predictions
                decoded_uncond = self._denoise_step(x, uncond_context, t_emb, pos_emb)
                pred_emb = decoded_uncond + guidance_scale * (decoded - decoded_uncond)
            elif self.use_mse_prediction:
                pred_emb = decoded
            else:
                logits = self.output_head(decoded)
                if soft and i < n_steps - 1:
                    probs = F.softmax(logits / soft_temp, dim=-1)
                    pred_emb = probs @ self.token_emb.weight
                else:
                    if temperature > 0:
                        probs = F.softmax(logits / temperature, dim=-1)
                        flat_probs = probs.reshape(-1, probs.shape[-1])
                        pred_ids = torch.multinomial(flat_probs, 1).reshape(B, S)
                    else:
                        pred_ids = logits.argmax(dim=-1)
                    pred_emb = self.token_emb(pred_ids)

            # Re-noise to next (lower) noise level
            if i < n_steps - 1:
                noise = self._make_noise(pred_emb)
                alpha_n = alpha_next.view(1, 1, 1)
                x = torch.sqrt(alpha_n) * pred_emb + torch.sqrt(1 - alpha_n) * noise
            else:
                x = pred_emb

        # Final clean prediction
        t_batch = torch.zeros(B, device=device)
        t_emb = self.time_embed(t_batch)

        decoded = self._denoise_step(x, triple_context, t_emb, pos_emb)

        if use_cfg:
            decoded_uncond = self._denoise_step(x, uncond_context, t_emb, pos_emb)
            decoded = decoded_uncond + guidance_scale * (decoded - decoded_uncond)

        if self.use_mse_prediction:
            final_ids = self._nn_decode(decoded)
        else:
            logits = self.output_head(decoded)
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                flat_probs = probs.reshape(-1, probs.shape[-1])
                final_ids = torch.multinomial(flat_probs, 1).reshape(B, S)
            else:
                final_ids = logits.argmax(dim=-1)

        return final_ids

    # ── Unified generate ──────────────────────────────────────────────

    @torch.no_grad()
    def generate_ids(
        self,
        triple_context: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        cosine_schedule: bool = True,
        role_id: int | None = None,
        soft: bool = False,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate raw token IDs (B, S) without string roundtrip."""
        if self.use_continuous_noise:
            return self._generate_continuous(triple_context, n_steps, temperature,
                                            role_id=role_id, soft=soft,
                                            guidance_scale=guidance_scale)
        else:
            # Discrete path still returns strings — wrap for compatibility
            raise NotImplementedError("generate_ids only supported for continuous noise")

    @torch.no_grad()
    def generate(
        self,
        triple_context: torch.Tensor,
        n_steps: int = 10,
        temperature: float = 0.0,
        cosine_schedule: bool = True,
        role_id: int | None = None,
        soft: bool = False,
        guidance_scale: float = 1.0,
    ) -> list[str]:
        if self.use_continuous_noise:
            final_ids = self._generate_continuous(triple_context, n_steps, temperature,
                                                  role_id=role_id, soft=soft,
                                                  guidance_scale=guidance_scale)
            texts = self.tokenizer.batch_decode(final_ids, skip_special_tokens=True)
            return [t.strip() for t in texts]
        else:
            return self._generate_discrete(triple_context, n_steps, temperature,
                                          cosine_schedule, role_id=role_id)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
