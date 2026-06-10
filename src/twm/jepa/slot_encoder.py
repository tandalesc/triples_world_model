"""SlotEncoder: GLUCOSE BPE text -> M coordinated slots -> (k, verb_logits).

Spec: research/jepa_operator_v1_design.md §2 (forward pass + nano/mini param tables)
and §12 row T2. Contract: SlotEncoderProtocol in src/twm/jepa/__init__.py.

Pipeline (spec §2):

    text_ids (B,T_text)
      │ frozen token_emb + text_pos_emb
      │ L_text ALBERT-tied self-attn layers  -> context (B,T_text,d)
      │ M learned queries (per-slot mu/sigma init) cross-attend to context
      │ n_iters=3 slot self-attn coordination (slot-attention style, SHARED weights)
      ▼
    slots (B,M,d)
      ├── NounHead (d->dn) + batch standardize (NOT L2-norm) -> k (B,M,dn)
      └── VerbHead (d->V)                                    -> verb_logits (B,M,V)

Two load-bearing design constraints (do not "simplify" away):

  * ALBERT tying: when tie_text_layers=True the L_text self-attn layers share ONE
    weight block applied L_text times. Counted once in the param table (§2).

  * Slot coordination is MANDATORY (§2 FIX, Judge 1). Single-pass cross-attn gives
    zero slot-competition pressure -> stripe collapse with no recovery gradient. The
    n_iters=3 self-attn coordination block is the routing mechanism and shares weights
    across the iterations (slot-attention style recurrence).

  * NounHead standardizes (zero-mean / unit-var per dim over the B*M batch), it does
    NOT L2-normalize onto the sphere. L2-projection makes every 1D projection have
    std 1/sqrt(dn) and kills the SIGReg gradient (spec §3, the decisive shared flaw).
"""

import torch
import torch.nn as nn


class _SelfAttnBlock(nn.Module):
    """One pre-norm transformer encoder block (MHA + FFN), batch_first.

    Used both for the ALBERT-tied text self-attention and as the per-iteration
    slot coordination block. Plain nn primitives — mirrors text_compressor.py.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class _SlotCoordBlock(nn.Module):
    """Slot-attention style coordination update: MHA over slots + residual + LN.

    Deliberately attention-only (no FFN) — the §2 table counts the coordination
    block as 4*d^2 MHA weights. The FFN-bearing transformer block is reserved for
    the text self-attention path. Shared across the n_slot_iters iterations.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(slots, slots, slots)
        return self.ln(slots + a)


class NounHead(nn.Module):
    """slots (B,M,d) -> nouns k (B,M,dn), standardized over the batch.

    Linear d->dn then per-dimension batch standardization (zero-mean, unit-variance
    across the B*M samples). NO L2-normalization — see module docstring / spec §3.
    The standardize step has no parameters.
    """

    def __init__(self, d_model: int, d_noun: int, eps: float = 1e-5):
        super().__init__()
        self.proj = nn.Linear(d_model, d_noun)
        self.eps = eps

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        k = self.proj(slots)  # (B, M, dn)
        # Standardize per dim over the flattened (B*M) batch — NOT per-vector L2.
        flat = k.reshape(-1, k.shape[-1])  # (B*M, dn)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        k = (k - mean) / (std + self.eps)
        return k


class VerbHead(nn.Module):
    """slots (B,M,d) -> verb_logits (B,M,V). Pre-Gumbel-softmax logits."""

    def __init__(self, d_model: int, n_verbs: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_verbs)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.proj(slots)


class SlotEncoder(nn.Module):
    """ALBERT-tied text encoder -> M-query cross-attn -> 3-iter slot coordination.

    Satisfies SlotEncoderProtocol. Forward returns (slots, k, verb_logits).

    Args:
        token_emb: shared (optionally frozen) (vocab, d_model) embedding table.
            Frozen-able by the caller via requires_grad_(False); it is not counted
            in the non-embedding param budget either way.
        d_model:   d
        d_noun:    dn (NounHead output width)
        n_slots:   M learned queries
        n_verbs:   V (VerbHead output width)
        n_text_layers:   L_text self-attn applications
        tie_text_layers: ALBERT tying — one shared block applied L_text times.
        n_heads:   attention heads
        d_ff:      FFN hidden width (defaults to 2*d_model per nano/mini tables:
                   nano d=64->d_ff=128, mini d=128 uses d_ff=512 explicitly).
        n_slot_iters: coordination iterations (spec mandates 3; shared weights).
        max_text_tokens: T_text for the position table.
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        d_model: int = 64,
        d_noun: int = 32,
        n_slots: int = 8,
        n_verbs: int = 8,
        n_text_layers: int = 2,
        tie_text_layers: bool = True,
        n_heads: int = 4,
        d_ff: int | None = None,
        n_slot_iters: int = 3,
        max_text_tokens: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_noun = d_noun
        self.n_slots = n_slots
        self.n_verbs = n_verbs
        self.n_text_layers = n_text_layers
        self.tie_text_layers = tie_text_layers
        self.n_slot_iters = n_slot_iters
        self.max_text_tokens = max_text_tokens
        d_ff = d_ff if d_ff is not None else 2 * d_model

        # Shared (optionally frozen) token embedding; text position embedding.
        self.token_emb = token_emb
        self.text_pos_emb = nn.Embedding(max_text_tokens, d_model)

        # ALBERT-tied text self-attention: one block reused L_text times, or
        # n_text_layers independent blocks when tie_text_layers=False.
        if tie_text_layers:
            self.text_blocks = nn.ModuleList([_SelfAttnBlock(d_model, n_heads, d_ff, dropout)])
        else:
            self.text_blocks = nn.ModuleList(
                [_SelfAttnBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_text_layers)]
            )

        # M learned slot queries with per-slot mu/sigma init parameters.
        # Spec §2 counts these as M*d + 2*M*d = 3 tensors of shape (M, d):
        #   slot_query   — the learned base query (the slot identity)
        #   slot_mu      — per-slot init mean
        #   slot_log_sigma — per-slot init log-std
        # The init query for slot i is  slot_query_i + (mu_i + sigma_i * eps_i),
        # a learned slot-attention-style Gaussian init that keeps slots distinct.
        self.slot_query = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.slot_mu = nn.Parameter(torch.zeros(n_slots, d_model))
        self.slot_log_sigma = nn.Parameter(torch.zeros(n_slots, d_model))
        # Persistent random direction so each slot starts distinct (slot-attention
        # init): drawn once at construction, frozen (not a trainable parameter).
        self.register_buffer("slot_eps", torch.randn(n_slots, d_model))

        # Cross-attention: slot queries extract from text context (not tied).
        self.cross_ln_q = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=dropout
        )
        self.cross_ln = nn.LayerNorm(d_model)

        # Slot self-attention coordination block, shared across n_slot_iters.
        self.coord_block = _SlotCoordBlock(d_model, n_heads, dropout)

        # Heads.
        self.noun_head = NounHead(d_model, d_noun)
        self.verb_head = VerbHead(d_model, n_verbs)

    def _slot_queries(self, B: int, device: torch.device) -> torch.Tensor:
        """Build (B, M, d) initial slot queries from per-slot mu/sigma."""
        sigma = torch.exp(self.slot_log_sigma)
        q = self.slot_query + self.slot_mu + sigma * self.slot_eps  # (M, d)
        return q.unsqueeze(0).expand(B, -1, -1)

    def encode_text(
        self, text_ids: torch.Tensor, text_pad: torch.Tensor | None
    ) -> torch.Tensor:
        """text_ids (B,T) -> context (B,T,d) via embed + ALBERT-tied self-attn."""
        B, T = text_ids.shape
        device = text_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(text_ids) + self.text_pos_emb(pos)

        kpm = text_pad.bool() if text_pad is not None else None
        if self.tie_text_layers:
            block = self.text_blocks[0]
            for _ in range(self.n_text_layers):
                x = block(x, key_padding_mask=kpm)
        else:
            for block in self.text_blocks:
                x = block(x, key_padding_mask=kpm)
        return x

    def forward(
        self,
        text_ids: torch.Tensor,  # (B, T_text)
        text_pad: torch.Tensor,  # (B, T_text) padding mask (True/1 where pad)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (slots (B,M,d), k (B,M,dn), verb_logits (B,M,V))."""
        B = text_ids.shape[0]
        device = text_ids.device

        context = self.encode_text(text_ids, text_pad)  # (B, T, d)
        kpm = text_pad.bool() if text_pad is not None else None

        # Cross-attention: slot queries extract slot-structured info from context.
        q = self._slot_queries(B, device)  # (B, M, d)
        qn = self.cross_ln_q(q)
        extracted, _ = self.cross_attn(qn, context, context, key_padding_mask=kpm)
        slots = self.cross_ln(q + extracted)  # residual into LN

        # n_iters slot self-attn coordination (shared weights across iters).
        for _ in range(self.n_slot_iters):
            slots = self.coord_block(slots)

        k = self.noun_head(slots)            # (B, M, dn), standardized
        verb_logits = self.verb_head(slots)  # (B, M, V)
        return slots, k, verb_logits

    # ---- introspection helpers (used by the param-count test) ----------------

    def trainable_param_count(self, include_embedding: bool = False) -> int:
        total = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if not include_embedding and name.startswith("token_emb"):
                continue
            total += p.numel()
        return total

    def param_breakdown(self, include_embedding: bool = False) -> dict[str, int]:
        """Per-module trainable param counts (excludes frozen token_emb by default)."""
        groups = {
            "text_pos_emb": [self.text_pos_emb],
            "text_self_attn": list(self.text_blocks),
            "slot_queries+init": [],  # filled below (raw Parameters)
            "cross_attn": [self.cross_ln_q, self.cross_attn, self.cross_ln],
            "slot_coordination": [self.coord_block],
            "noun_head": [self.noun_head],
            "verb_head": [self.verb_head],
        }
        out: dict[str, int] = {}
        for name, mods in groups.items():
            out[name] = sum(
                p.numel() for m in mods for p in m.parameters() if p.requires_grad
            )
        # slot_query + slot_mu + slot_log_sigma are bare Parameters
        # (slot_eps is a buffer, not counted).
        out["slot_queries+init"] = (
            self.slot_query.numel()
            + self.slot_mu.numel()
            + self.slot_log_sigma.numel()
        )
        if include_embedding:
            out["token_emb (frozen)"] = self.token_emb.weight.numel()
        return out
