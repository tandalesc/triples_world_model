"""JEPAOperatorModelV2 — unsupervised latent actions + token-space grounding.

Authoritative spec: research/jepa_v2_latent_actions.md (§1 forward diagram, §5
losses, §6 leakage, §9 budget, §11 work breakdown row T-C).

This module is Task C: it COMPOSES the concurrently-developed sibling modules
behind their FROZEN interfaces (the only thing this file may assume about them):

  TransitionEncoder(encode_text_fn, d_model, n_verbs, mlp_hidden, use_delta)
      .forward(src_ids, src_pad, tgt_ids, tgt_pad, tau, hard)
          -> (v_onehot (B,V), v_logits (B,V), pool_t (B,d))   [training-only posterior]
  PriorHead(d_model, n_verbs, mlp_hidden)
      .forward(pool_t) -> p_logits (B,V)
  TokenDecoder(vocab_size, d_dec, n_layers, n_heads, d_ff, d_noun, max_text_tokens, pad_id)
      .forward(a_star (B,M,dn), tgt_ids, tgt_pad) -> logits (B,T,V)
      .generate(a_star, max_tokens, temperature) -> (B,T) ids
  SlotEncoder(...) -> (slots, k, verb_logits)   [v1; VerbHead output IGNORED in v2]
  RotationScaleOperator(...) with .apply(k, v) accepting v as soft (B,M,V) float
  Readout(d_noun, n_heads), Predictor(d_noun)  [reused from v1 model.py]

Forward diagram (design §1):

    text_t  -> SlotEncoder -> k (B,M,dn)
    text_t, text_t+1 -> TransitionEncoder (posterior) -> v_onehot (B,V), v_logits, pool_t
    v_onehot broadcast (B,V)->(B,M,V) -> Operator.apply(k, v) -> a* (B,M,dn)
    pool_t -> PriorHead -> p_logits (B,V)                     [L_prior = KL(stopgrad q ‖ p)]
    a* -> Readout -> Predictor -> zhat (B,dn)                 [L_pred aux, EMA target]
    a* -> TokenDecoder(memory=a*) -> logits (B,T,V)           [L_token PRIMARY, grounding]

LEAKAGE RULE (design §6, the load-bearing invariant), enforced STRUCTURALLY here:
  - The TokenDecoder is called with `a_star` as its ONLY memory. It never receives
    posterior features (pool_t, v_logits), raw text_t+1 encodings, or un-transformed k.
  - `a* = operator.apply(k, v)` and `k = f(text_t)` only; the ONLY path from text_t+1
    into the decoder conditioning is the discrete one-hot `v` (⌈log2 V⌉ bits).
  - The posterior sees text_t+1, but its sole output reaching the operator/decoder is
    the hard one-hot `v`. Hard ST (not a soft mix) bounds the channel (design §6 L3).

Inference (autonomous rollout, posterior GONE): sample v ~ PriorHead(pool_t) or set
v directly, then a* = B_v k, then TokenDecoder.generate(a*) -> text_t+1 (design §1).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v1 Readout / Predictor / encoder-readout EMA bundle (model.py is v1-owned;
# imported, never edited).
from .model import Readout, Predictor, _EncoderReadout


class JEPAOperatorModelV2(nn.Module):
    """Composed v2 latent-action world model (design §1/§5/§6).

    Args:
        encoder:     v1 SlotEncoder. forward(text_ids, text_pad) -> (slots, k, verb_logits).
                     The VerbHead output is IGNORED in v2 (the action comes from the
                     posterior, not per-slot verb heads — kills the v1 positional cheat).
        operator:    v1 RotationScaleOperator. apply(k, v) accepts v as soft (B,M,V) float.
        transition:  TransitionEncoder (Task A). Training-only posterior q(v|t,t+1).
        prior:       PriorHead (Task A). p(v|pool_t) for autonomous rollout.
        decoder:     TokenDecoder (Task B). Grounding AR decoder; memory = a* ONLY.
        d_noun:      dn.
        n_verbs:     V.
        n_heads:     readout attention heads.
        use_pred:    if True (default) build the L_pred aux branch (Readout+Predictor+EMA).
                     w_pred=0 is a supported ablation (design §5) but the branch is cheap
                     and kept built so the trainer can toggle the weight without rebuild.
    """

    def __init__(
        self,
        encoder: nn.Module,
        operator: nn.Module,
        transition: nn.Module,
        prior: nn.Module,
        decoder: nn.Module,
        d_noun: int,
        n_verbs: int,
        n_heads: int = 4,
        use_pred: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.operator = operator
        self.transition = transition
        self.prior = prior
        self.decoder = decoder
        self.d_noun = d_noun
        self.n_verbs = n_verbs
        self.use_pred = use_pred

        # L_pred aux branch (design §5): Readout pool over a* -> Predictor -> zhat, with
        # an EMA(SlotEncoder+Readout) target over the RAW nouns of text_t+1 (v1 pattern).
        self.readout = Readout(d_noun, n_heads)
        self.predictor = Predictor(d_noun)
        self._online_bundle = _EncoderReadout(self.encoder, self.readout)
        self.ema = copy.deepcopy(self._online_bundle)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # Re-point the EMA copy's frozen token_emb at the shared online table so we do
        # not double-store (V×d) frozen weights in every checkpoint (v1 model.py note).
        if hasattr(self.encoder, "token_emb") and hasattr(self.ema.encoder, "token_emb"):
            self.ema.encoder.token_emb = self.encoder.token_emb

    # ------------------------------------------------------------------ action path
    def _apply_action(self, k: torch.Tensor, v_onehot: torch.Tensor) -> torch.Tensor:
        """a* = B_v k for ONE sequence-level action per pair (design §2.3).

        v_onehot: (B, V) hard straight-through one-hot. Broadcast to (B, M, V) so the
        SAME action is applied to all M slots via the operator's soft-mix path; with a
        hard one-hot this is numerically the hard-index path (design §2.3). Using the
        soft path keeps the ST gradient flowing into v_logits/posterior.
        """
        B, M, _ = k.shape
        v_slots = v_onehot.unsqueeze(1).expand(B, M, -1)  # (B, M, V)
        return self.operator.apply(k, v_slots)            # (B, M, dn)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        src_ids: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_pad: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
    ) -> dict:
        """Training forward (design §1).

        Returns a dict with everything the v2 loss needs:
            k         (B, M, dn)  standardized nouns of text_t
            a         (B, M, dn)  a* = B_v k (operator-transformed nouns) — decoder memory
            v         (B,)        argmax posterior action (reporting/diagnostics)
            v_onehot  (B, V)      hard ST one-hot action
            v_logits  (B, V)      posterior logits
            p_logits  (B, V)      prior logits
            zhat      (B, dn)     L_pred prediction (None if use_pred=False)
            z_target  (B, dn)     EMA raw-noun pool of text_t+1, stop-grad (None if not use_pred)
            logits    (B, T, V)   token decoder logits over text_t+1 (PRIMARY grounding)

        Leakage (design §6): the decoder receives `a` (== operator output) as its ONLY
        memory; tgt_ids enter only as the standard teacher-forced AR target.
        """
        # --- noun path: text_t only (no t+1 info reaches k) ---
        _, k, _ = self.encoder(src_ids, src_pad)  # verb_logits IGNORED in v2

        # --- posterior: sees BOTH texts, emits ONE discrete action per pair ---
        v_onehot, v_logits, pool_t = self.transition(
            src_ids, src_pad, tgt_ids, tgt_pad, tau, hard
        )

        # --- prior p(v | text_t) for autonomous rollout (KL target is stopgrad q) ---
        p_logits = self.prior(pool_t)

        # --- operator-transformed nouns: the ONLY decoder conditioning channel ---
        a = self._apply_action(k, v_onehot)  # (B, M, dn)

        # --- token decoder: memory = a* ONLY (structural leakage block) ---
        logits = self.decoder(a, tgt_ids, tgt_pad)  # (B, T, V)

        out = {
            "k": k,
            "a": a,
            "v": v_onehot.argmax(dim=-1),  # (B,)
            "v_onehot": v_onehot,
            "v_logits": v_logits,
            "p_logits": p_logits,
            "logits": logits,
        }

        # --- L_pred aux branch (optional; design §5) ---
        if self.use_pred:
            pooled = self.readout(a)        # (B, dn)
            zhat = self.predictor(pooled)   # (B, dn)
            with torch.no_grad():
                z = self.ema.pool_raw(tgt_ids, tgt_pad)  # (B, dn) raw-noun pool of t+1
            out["zhat"] = zhat
            out["z_target"] = z.detach()
        else:
            out["zhat"] = None
            out["z_target"] = None

        return out

    # forward_v2 is the name diagnostics_v2.py duck-types against (it must be able to
    # tell a v2 model from a v1 model, whose `forward(src, src_pad)` has a different
    # signature). Alias it to `forward` so the diagnostics' v-ablation CE gap and
    # generated-sample paths resolve to the real v2 forward.
    forward_v2 = forward

    # ------------------------------------------------------------------ inference (§1)
    @torch.no_grad()
    def rollout(
        self,
        src_ids: torch.Tensor,
        src_pad: torch.Tensor,
        sample: bool = True,
        verb_idx: int | torch.Tensor | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Autonomous rollout: posterior GONE (design §1).

        Pick the action from the prior p(v|text_t) (sample or argmax) OR set it
        directly via `verb_idx` (a user/UI action), then a* = B_v k, then generate
        text_t+1 with the token decoder.

        Args:
            sample:   if True and verb_idx is None, sample v ~ Categorical(softmax(p_logits))
                      (multi-modal futures); else greedy argmax of p_logits.
            verb_idx: override action — scalar int or (B,) long tensor (demo/UI action).
            temperature: decoder sampling temperature (0 = greedy).

        Returns {"v": (B,), "a": (B,M,dn), "gen_ids": (B,T)}.
        """
        _, k, _ = self.encoder(src_ids, src_pad)
        B = k.shape[0]
        device = k.device

        # --- choose the action ---
        if verb_idx is not None:
            if isinstance(verb_idx, int):
                v = torch.full((B,), verb_idx, dtype=torch.long, device=device)
            else:
                v = torch.as_tensor(verb_idx, device=device, dtype=torch.long)
                if v.dim() == 0:
                    v = v.expand(B)
        else:
            pool_t = self._prior_pool(src_ids, src_pad)
            p_logits = self.prior(pool_t)  # (B, V)
            if sample:
                probs = F.softmax(p_logits, dim=-1)
                v = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            else:
                v = p_logits.argmax(dim=-1)  # (B,)

        v_onehot = F.one_hot(v, num_classes=self.n_verbs).to(k.dtype)  # (B, V)
        a = self._apply_action(k, v_onehot)  # (B, M, dn)
        gen_ids = self.decoder.generate(a, max_tokens=max_tokens, temperature=temperature)
        return {"v": v, "a": a, "gen_ids": gen_ids}

    def _prior_pool(self, src_ids, src_pad) -> torch.Tensor:
        """Masked-mean pool of text_t through the (shared) trunk, for the prior at
        rollout time (the posterior is unavailable). Mirrors the posterior's pool_t
        construction (design §2.1/§3) using the encoder's bound `encode_text`.
        """
        ctx = self.encoder.encode_text(src_ids, src_pad)  # (B, T, d)
        if src_pad is None:
            return ctx.mean(dim=1)
        mask = (~src_pad.bool()).unsqueeze(-1).to(ctx.dtype)  # (B, T, 1) 1 at real tokens
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (ctx * mask).sum(dim=1) / denom

    # ------------------------------------------------------------------ pet/demo API
    def step_latent(self, k: torch.Tensor, verb_idx) -> torch.Tensor:
        """One tick in latent space: a* = B_v k. verb_idx scalar or (B,)/(B,M)."""
        v_slots = self._verb_to_slots(k, verb_idx)
        return self.operator.apply(k, v_slots)

    def undo_latent(self, a: torch.Tensor, verb_idx) -> torch.Tensor:
        """Exact undo: k = B_v^{-1} a (structural inverse from the operator)."""
        v_slots = self._verb_to_slots(a, verb_idx)
        return self.operator.inverse_apply(a, v_slots)

    def _verb_to_slots(self, x: torch.Tensor, verb_idx) -> torch.Tensor:
        """Resolve a sequence-level / per-slot verb index to (B, M) long for the operator."""
        B, M = x.shape[0], x.shape[1]
        if isinstance(verb_idx, int):
            return x.new_full((B, M), verb_idx, dtype=torch.long)
        v = torch.as_tensor(verb_idx, device=x.device, dtype=torch.long)
        if v.dim() == 0:
            return x.new_full((B, M), int(v), dtype=torch.long)
        if v.dim() == 1:  # (B,) sequence-level -> broadcast to all slots
            return v.unsqueeze(1).expand(B, M)
        return v  # already (B, M)

    # ------------------------------------------------------------------ EMA (§5 aux)
    @torch.no_grad()
    def ema_update(self, tau: float = 0.995) -> None:
        """θ_ema ← τ·θ_ema + (1−τ)·θ_online. Call AFTER optimizer.step().

        No-op-safe if use_pred=False (the EMA bundle still exists but is unused;
        keeping the update cheap and harmless). Mirrors v1 model.ema_update.
        """
        online_p = dict(self._online_bundle.named_parameters())
        for name, ema_param in self.ema.named_parameters():
            online = online_p.get(name)
            if online is None:
                continue
            ema_param.mul_(tau).add_(online.detach(), alpha=1.0 - tau)
        online_b = dict(self._online_bundle.named_buffers())
        for name, ema_buf in self.ema.named_buffers():
            online = online_b.get(name)
            if online is not None:
                ema_buf.copy_(online)

    # ------------------------------------------------------------------ param sets
    def online_parameters(self):
        """Trainable params only — excludes the EMA bundle (design §5)."""
        ema_ids = {id(p) for p in self.ema.parameters()}
        for p in self.parameters():
            if id(p) not in ema_ids and p.requires_grad:
                yield p

    def trainable_param_count(self) -> int:
        """Total trainable, non-EMA params. Excludes frozen token_emb tables (they
        carry requires_grad=False) so the count is the §9 non-embedding budget."""
        return sum(p.numel() for p in self.online_parameters())


def build_jepa_model_v2(cfg, token_emb: nn.Module):
    """Construct a JEPAOperatorModelV2 from a JEPAConfig + a frozen token embedding.

    Resolves the concurrently-developed sibling classes lazily from the package so
    this import does not hard-fail while those modules are still landing (mirrors v1
    build_jepa_model). Constructor kwargs are filtered to what each class accepts so
    minor naming drift between sibling tasks does not break composition.

    Note: the TokenDecoder gets its OWN token embedding (vocab×d_dec), NOT the shared
    frozen encoder table — the decoder learns to GENERATE tokens (design §9 budget
    counts a separate decoder token_emb). The `token_emb` arg here is the frozen
    ENCODER embedding only.
    """
    from twm.jepa import (
        SlotEncoder,
        RotationScaleOperator,
        RotationOperator,
        SOnCayleyOperator,
        TransitionEncoder,
        PriorHead,
        TokenDecoder,
    )

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

    op_cls = {
        "rotation_scale": RotationScaleOperator,
        "rotation": RotationOperator,
        "son_cayley": SOnCayleyOperator,
    }.get(m.operator_group, RotationScaleOperator)
    operator = _construct(op_cls, n_verbs=m.n_verbs, d_noun=m.d_noun, block=m.block)

    # Posterior + prior action heads share the encoder's bound text trunk (design §2.1):
    # zero new attention params, posterior's view in the noun-building space.
    transition = _construct(
        TransitionEncoder,
        encode_text_fn=encoder.encode_text,
        d_model=m.d_model,
        n_verbs=m.n_verbs,
        mlp_hidden=m.transition.mlp_hidden,
        use_delta=m.transition.use_delta,
    )
    prior = _construct(
        PriorHead,
        d_model=m.d_model,
        n_verbs=m.n_verbs,
        mlp_hidden=m.prior.mlp_hidden,
    )

    decoder = _construct(
        TokenDecoder,
        vocab_size=cfg.data.vocab_size,
        d_dec=m.decoder.d_dec,
        n_layers=m.decoder.n_layers,
        n_heads=m.decoder.n_heads,
        d_ff=m.decoder.d_ff,
        d_noun=m.d_noun,
        max_text_tokens=cfg.data.max_text_tokens,
        pad_id=0,
    )

    return JEPAOperatorModelV2(
        encoder=encoder,
        operator=operator,
        transition=transition,
        prior=prior,
        decoder=decoder,
        d_noun=m.d_noun,
        n_verbs=m.n_verbs,
        n_heads=m.n_heads,
    )


def _construct(cls, **kwargs):
    """Instantiate `cls` passing only kwargs its __init__ accepts (sibling naming-drift
    safety; mirrors v1 model._construct)."""
    import inspect

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)
    accepted = {name for name in params if name != "self"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})
