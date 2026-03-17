"""Text Compressor: free text BPE tokens → triple-level bottleneck vectors.

Takes a variable-length BPE text sequence and produces the same bottleneck
format as TripleCompressor: (B, N_triples × 3, d_model).

The text has no explicit triple structure. Learned extraction queries with
role conditioning cross-attend to the contextualized text to discover and
extract triple-structured information.

Optional VAE mode: projects extracted vectors to μ/log_σ, reparameterizes,
and computes KL divergence against learned role-conditioned priors. This
forces entity, attribute, and value slots into distinct distributions,
preventing the bottleneck from collapsing onto a 1D manifold.
"""

import torch
import torch.nn as nn


class TextCompressor(nn.Module):
    """Compresses free text into triple-level bottleneck vectors.

    Architecture:
        frozen_token_emb → + text_pos_emb → self_attention (4L)
        → learned extraction queries (N*3) cross-attend to text
        → (B, N*3, d_model) bottleneck vectors

    With vae=True:
        ... → μ/log_σ heads → reparameterize → z
        KL(q(z|x) || p(z|role)) per slot, with learned role priors.
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_triples: int = 8,
        max_text_tokens: int = 64,
        dropout: float = 0.1,
        vae: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_triples = max_triples
        self.max_text_tokens = max_text_tokens
        self.vae = vae

        # Shared frozen embedding (same table as compressor/expander)
        self.token_emb = token_emb

        # Text position embeddings
        self.text_pos_emb = nn.Embedding(max_text_tokens, d_model)

        # Self-attention to contextualize the full text
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learned extraction queries — one per triple position (max_triples * 3)
        self.extract_queries = nn.Parameter(
            torch.randn(max_triples * 3, d_model) * 0.02
        )

        # Cross-attention: queries extract triple-structured info from text
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )

        # LayerNorm after cross-attention extraction
        self.cross_ln = nn.LayerNorm(d_model)

        # Query self-attention: slots coordinate after extraction to avoid
        # redundancy (two entity queries attending to the same text region)
        # and ensure coverage (some text going unattended)
        self.query_self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout,
        )
        self.query_self_ln = nn.LayerNorm(d_model)

        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.query_ffn_ln = nn.LayerNorm(d_model)

        # Role and triple position encoding for extraction queries
        self.role_emb = nn.Embedding(3, d_model)
        self.triple_pos_emb = nn.Embedding(max_triples, d_model)

        # Output normalization
        self.out_ln = nn.LayerNorm(d_model)

        # VAE heads: project to μ and log_σ, with learned role priors
        if vae:
            self.mu_head = nn.Linear(d_model, d_model)
            self.logvar_head = nn.Linear(d_model, d_model)
            # Initialize logvar head to output ~0 (σ ≈ 1) so early training
            # behaves like deterministic compressor
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)
            # Learned prior per role: μ and log_σ for N(μ_role, σ_role)
            self.prior_mu = nn.Embedding(3, d_model)
            self.prior_logvar = nn.Embedding(3, d_model)
            # Init priors spread apart: entity near -1, attr near 0, value near +1
            # (in a random direction — the actual separation is learned)
            nn.init.normal_(self.prior_mu.weight, std=0.5)
            nn.init.zeros_(self.prior_logvar.weight)

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_pad_mask: torch.Tensor,
        n_triples: int,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compress text to triple-level bottleneck vectors.

        Args:
            text_token_ids: (B, T) BPE token IDs
            text_pad_mask: (B, T) True where padding
            n_triples: number of triples to extract (max_triples during training)

        Returns:
            If vae=False: (B, max_triples * 3, d_model)
            If vae=True: ((B, max_triples * 3, d_model), vae_info dict)
                vae_info contains 'kl_loss' (scalar) and per-role KL values.
        """
        B = text_token_ids.shape[0]
        T = text_token_ids.shape[1]
        device = text_token_ids.device

        # Embed text tokens + position
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(text_token_ids) + self.text_pos_emb(pos_ids)

        # Contextualize with self-attention
        x = self.self_attn(x, src_key_padding_mask=text_pad_mask)

        # Build extraction queries with role + triple position encoding
        n_queries = n_triples * 3
        queries = self.extract_queries[:n_queries].unsqueeze(0).expand(B, -1, -1)

        # Add structural encoding to queries
        triple_idx = torch.arange(n_triples, device=device).repeat_interleave(3)
        role_idx = torch.arange(3, device=device).repeat(n_triples)
        struct_enc = self.role_emb(role_idx) + self.triple_pos_emb(triple_idx)
        queries = queries + struct_enc.unsqueeze(0)

        # Cross-attend: queries extract from text independently
        extracted, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=text_pad_mask,
        )
        extracted = self.cross_ln(extracted)

        # Query self-attention: slots coordinate to reduce redundancy
        # and improve coverage.
        sa_out, _ = self.query_self_attn(extracted, extracted, extracted)
        extracted = self.query_self_ln(extracted + sa_out)
        extracted = self.query_ffn_ln(extracted + self.query_ffn(extracted))

        # Pad to max_triples * 3 if n_triples < max_triples
        if n_triples < self.max_triples:
            pad_size = (self.max_triples - n_triples) * 3
            pad = torch.zeros(B, pad_size, self.d_model, device=device)
            extracted = torch.cat([extracted, pad], dim=1)

        extracted = self.out_ln(extracted)

        if not self.vae:
            return extracted

        # VAE: project to μ/log_σ, reparameterize, compute KL
        return self._vae_forward(extracted, n_triples)

    def _vae_forward(self, extracted, n_triples):
        """VAE reparameterization and KL computation."""
        B, T, d = extracted.shape
        device = extracted.device

        mu = self.mu_head(extracted)          # (B, T, d)
        logvar = self.logvar_head(extracted)   # (B, T, d)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # deterministic at eval

        # KL divergence against role-conditioned priors
        # Role pattern: [E, A, V, E, A, V, ...] for n_triples, then padding
        n_active = n_triples * 3
        role_idx = torch.arange(3, device=device).repeat(n_triples)  # (n_active,)

        prior_mu = self.prior_mu(role_idx)        # (n_active, d)
        prior_logvar = self.prior_logvar(role_idx)  # (n_active, d)

        # KL(N(mu, sigma) || N(mu_p, sigma_p)) per dimension:
        # = log(sigma_p/sigma) + (sigma^2 + (mu - mu_p)^2) / (2*sigma_p^2) - 0.5
        active_mu = mu[:, :n_active]         # (B, n_active, d)
        active_logvar = logvar[:, :n_active]  # (B, n_active, d)
        p_mu = prior_mu.unsqueeze(0)          # (1, n_active, d)
        p_logvar = prior_logvar.unsqueeze(0)  # (1, n_active, d)

        kl_per_dim = (
            0.5 * p_logvar - 0.5 * active_logvar
            + (torch.exp(active_logvar) + (active_mu - p_mu).pow(2))
            / (2 * torch.exp(p_logvar))
            - 0.5
        )

        # Per-role KL (mean over batch and dimensions, sum over slots of that role)
        kl_info = {}
        role_names = ["entity", "attribute", "value"]
        for r in range(3):
            role_mask = role_idx == r  # (n_active,)
            role_kl = kl_per_dim[:, role_mask].sum(dim=(1, 2)).mean()  # scalar
            kl_info[f"kl_{role_names[r]}"] = role_kl.item()

        # Total KL: mean over batch, sum over all active slots and dimensions
        kl_total = kl_per_dim.sum(dim=(1, 2)).mean()
        kl_info["kl_loss"] = kl_total
        kl_info["mu"] = mu  # expose for spectral loss (must measure structure, not noise)

        return z, kl_info

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
