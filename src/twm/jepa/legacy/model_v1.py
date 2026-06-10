"""JEPAOperatorModel — composes SlotEncoder + Operator + Readout + Predictor.

Spec §2 (forward pass), §4 (EMA), §9 (pet engine API).

Forward (online path):
    text_ids -> SlotEncoder -> (slots, k, verb_logits)
             -> Gumbel-softmax verb path -> a* = B_v k  (soft mix in train, hard at eval)
             -> Readout (attn-pool over a*, query dim dn) -> zhat (B, dn)
             -> Predictor MLP (dn->dn) -> zhat

EMA target path (stop-grad):
    next_text -> EMA(SlotEncoder) -> k_next
              -> EMA(Readout) over RAW nouns (no operator) -> z (B, dn)

The operator is NEVER applied on the target side: z is the raw-noun pool of the
*next* state, a genuinely different object than the operator-transformed nouns of
the current state (spec §4 — why we keep EMA).

EMA module: deepcopy of (online encoder + readout) at step 0, requires_grad=False,
updated manually via `ema_update(tau)` AFTER each optimizer step. EMA params are
excluded from the trainable list and from grad clipping (the trainer enforces this
by iterating `online_parameters()`).

Pet engine API (spec §9): `step_latent(k, v)` = operator.apply; `undo_latent(a, v)`
= operator.inverse_apply (exact structural inverse).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):
    """Attention-pool over the M transformed nouns a* -> a single (B, dn) vector.

    A learned conditional query (dim dn) cross-attends to the M slots (dim dn).
    Query init is small (zero-ish) following the repo's zero-init out_gate spirit
    (§11) so the readout starts near a mean-pool and learns selectivity.
    """

    def __init__(self, d_noun: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_noun) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_noun, n_heads, batch_first=True
        )

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, M, dn) -> pooled (B, dn)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B,1,dn)
        key_pad = pad if pad is not None else None  # (B,M) True=ignore
        out, _ = self.attn(q, x, x, key_padding_mask=key_pad)
        return out.squeeze(1)  # (B, dn)


class Predictor(nn.Module):
    """Predictor MLP (dn -> dn). Spec §2: `2·dn²` params (one hidden of width dn)."""

    def __init__(self, d_noun: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_noun, d_noun),
            nn.GELU(),
            nn.Linear(d_noun, d_noun),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _EncoderReadout(nn.Module):
    """Bundle of (SlotEncoder + Readout) so the EMA copy is a single deepcopy.

    `encode_nouns` returns standardized nouns k for an input text. `pool_raw` runs
    those raw nouns through the readout (the target-side path: no operator).
    """

    def __init__(self, encoder: nn.Module, readout: Readout):
        super().__init__()
        self.encoder = encoder
        self.readout = readout

    def encode(self, text_ids, text_pad):
        # -> (slots, k, verb_logits)
        return self.encoder(text_ids, text_pad)

    def pool_raw(self, text_ids, text_pad) -> torch.Tensor:
        _, k, _ = self.encoder(text_ids, text_pad)
        return self.readout(k)  # (B, dn) raw-noun pool


class JEPAOperatorModel(nn.Module):
    """Composed JEPA operator world model (spec §2/§4/§9).

    Args:
        encoder:  a SlotEncoder (T2) with forward(text_ids, text_pad)
                  -> (slots (B,M,d), k (B,M,dn) standardized, verb_logits (B,M,V)).
        operator: an Operator (T1) with apply / inverse_apply over (B,M,dn),(B,M).
        d_noun:   dn.
        n_verbs:  V.
        n_heads:  attention heads for the readout pool.
    """

    def __init__(
        self,
        encoder: nn.Module,
        operator: nn.Module,
        d_noun: int,
        n_verbs: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.operator = operator
        self.d_noun = d_noun
        self.n_verbs = n_verbs

        self.readout = Readout(d_noun, n_heads)
        self.predictor = Predictor(d_noun)

        # EMA bundle (encoder + readout). deepcopy at construction == "step 0".
        # requires_grad=False; updated only via ema_update().
        self._online_bundle = _EncoderReadout(self.encoder, self.readout)
        self.ema = copy.deepcopy(self._online_bundle)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # The frozen token embedding is identical on both sides and never updated by
        # ema_update (online emb is frozen). Re-point the EMA copy at the shared
        # online table so we do not double-store (V×d) weights in the model / every
        # checkpoint. Safe because it is frozen and content-identical.
        if hasattr(self.encoder, "token_emb") and hasattr(self.ema.encoder, "token_emb"):
            self.ema.encoder.token_emb = self.encoder.token_emb

    # ------------------------------------------------------------------ verb path
    def _apply_all_verbs(self, k: torch.Tensor) -> torch.Tensor:
        """B_v k for every verb v. -> (B, M, V, dn).

        Calls operator.apply once per verb (V is small: 8/16). Used to build the
        soft Gumbel mix without materializing operator matrices.
        """
        B, M, dn = k.shape
        outs = []
        for v in range(self.n_verbs):
            v_idx = k.new_full((B, M), v, dtype=torch.long)
            outs.append(self.operator.apply(k, v_idx))  # (B,M,dn)
        return torch.stack(outs, dim=2)  # (B, M, V, dn)

    def _verb_transform(
        self,
        k: torch.Tensor,
        verb_logits: torch.Tensor,
        gumbel_tau: float,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a* and the discrete verb assignment.

        Train (hard=False): a* = Σ_v softmax_v · B_v k (Gumbel-softmax soft mix) so
        L_pred gradients flow into VerbHead logits (spec §3 VerbHead fix).
        Eval/export (hard=True): straight-through hard argmax.

        Returns (a* (B,M,dn), verb (B,M) long).
        """
        # Gumbel-softmax weights over verbs. (B,M,V)
        weights = F.gumbel_softmax(verb_logits, tau=gumbel_tau, hard=hard, dim=-1)
        all_a = self._apply_all_verbs(k)  # (B,M,V,dn)
        # weighted sum over V
        a = (weights.unsqueeze(-1) * all_a).sum(dim=2)  # (B,M,dn)
        verb = verb_logits.argmax(dim=-1)  # (B,M) discrete (reporting / export)
        return a, verb

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        src_ids: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_ids: torch.Tensor | None = None,
        tgt_pad: torch.Tensor | None = None,
        gumbel_tau: float = 1.0,
        hard: bool = False,
    ) -> dict:
        """Encode current state -> {k, verb, a, zhat, verb_logits, slots, z_target?}.

        If (tgt_ids, tgt_pad) are given, also computes the EMA target z_target
        (stop-grad, raw-noun pool of the next state) for L_pred. Inference-only pet
        encoding passes src alone.
        """
        slots, k, verb_logits = self.encoder(src_ids, src_pad)
        a, verb = self._verb_transform(k, verb_logits, gumbel_tau, hard)
        pooled = self.readout(a)            # (B, dn)
        zhat = self.predictor(pooled)       # (B, dn)

        out = {
            "k": k,
            "verb": verb,
            "a": a,
            "zhat": zhat,
            "verb_logits": verb_logits,
            "slots": slots,
        }

        if tgt_ids is not None:
            with torch.no_grad():
                z = self.ema.pool_raw(tgt_ids, tgt_pad)  # (B, dn)
            out["z_target"] = z.detach()
        return out

    # ------------------------------------------------------------------ pet API (§9)
    def step_latent(self, k: torch.Tensor, verb_idx) -> torch.Tensor:
        """One tick: a* = B_v k. verb_idx is (B,M) or a scalar int."""
        v = self._as_verb_index(k, verb_idx)
        return self.operator.apply(k, v)

    def undo_latent(self, a: torch.Tensor, verb_idx) -> torch.Tensor:
        """Exact undo: k = B_v^{-1} a (structural inverse)."""
        v = self._as_verb_index(a, verb_idx)
        return self.operator.inverse_apply(a, v)

    @staticmethod
    def _as_verb_index(x: torch.Tensor, verb_idx) -> torch.Tensor:
        B, M = x.shape[0], x.shape[1]
        if isinstance(verb_idx, int):
            return x.new_full((B, M), verb_idx, dtype=torch.long)
        v = torch.as_tensor(verb_idx, device=x.device, dtype=torch.long)
        if v.dim() == 0:
            return x.new_full((B, M), int(v), dtype=torch.long)
        return v

    # ------------------------------------------------------------------ EMA (§4)
    @torch.no_grad()
    def ema_update(self, tau: float = 0.995) -> None:
        """θ_ema ← τ·θ_ema + (1−τ)·θ_online. Call AFTER optimizer.step().

        Updates both params and buffers (e.g. LayerNorm running stats are buffers
        only if a layer uses them; we copy buffers outright to track the online side).
        """
        online_p = dict(self._online_bundle.named_parameters())
        for name, ema_param in self.ema.named_parameters():
            online = online_p.get(name)
            if online is None:
                continue
            ema_param.mul_(tau).add_(online.detach(), alpha=1.0 - tau)
        # Buffers: hard-copy (no momentum) so non-trainable state mirrors online.
        online_b = dict(self._online_bundle.named_buffers())
        for name, ema_buf in self.ema.named_buffers():
            online = online_b.get(name)
            if online is not None:
                ema_buf.copy_(online)

    # ------------------------------------------------------------------ param sets
    def online_parameters(self):
        """Trainable params only — excludes the EMA bundle (spec §4)."""
        ema_ids = {id(p) for p in self.ema.parameters()}
        for p in self.parameters():
            if id(p) not in ema_ids and p.requires_grad:
                yield p


def build_jepa_model(cfg, token_emb: nn.Module) -> JEPAOperatorModel:
    """Construct a JEPAOperatorModel from a JEPAConfig + a (frozen) token embedding.

    Composes the concurrently-developed SlotEncoder (T2) and RotationScaleOperator
    (T1), resolved lazily from the package so this import does not hard-fail while
    those modules are still landing. Constructor kwargs are filtered to what each
    class actually accepts (`_filtered_kwargs`) so minor naming drift between the
    sibling tasks does not break composition.
    """
    from twm.jepa import SlotEncoder, RotationScaleOperator

    m = cfg.model
    encoder = _construct(
        SlotEncoder,
        token_emb=token_emb,
        d_model=m.d_model,
        d_noun=m.d_noun,
        n_slots=m.n_slots,
        n_verbs=m.n_verbs,
        n_heads=m.n_heads,
        n_text_layers=m.n_text_layers,
        tie_text_layers=m.tie_text_layers,
        n_slot_iters=m.n_slot_iters,
        max_text_tokens=cfg.data.max_text_tokens,
    )
    operator = _construct(
        RotationScaleOperator,
        n_verbs=m.n_verbs,
        d_noun=m.d_noun,
        block=m.block,
    )
    return JEPAOperatorModel(
        encoder=encoder,
        operator=operator,
        d_noun=m.d_noun,
        n_verbs=m.n_verbs,
        n_heads=m.n_heads,
    )


def _construct(cls, **kwargs):
    """Instantiate `cls` passing only kwargs its __init__ accepts (drops the rest)."""
    import inspect

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return cls(**kwargs)
    accepted = {name for name in params if name != "self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})
