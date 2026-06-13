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
  SlotEncoder(...) -> (slots, k, verb_logits)   [VerbHead output IGNORED in v2]
  RotationScaleOperator(...) with .apply(k, v) accepting v as soft (B,M,V) float
  Readout(d_noun, n_heads), Predictor(d_noun)  [defined in-file; the L_pred EMA branch]

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

class Readout(nn.Module):
    """Attention-pool over the M transformed nouns a* -> a single (B, dn) vector.

    A learned conditional query (dim dn) cross-attends to the M slots (dim dn).
    Query init is small (zero-ish) following the repo's zero-init out_gate spirit
    so the readout starts near a mean-pool and learns selectivity.
    """

    def __init__(self, d_noun: int, n_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_noun) * 0.02)
        self.attn = nn.MultiheadAttention(d_noun, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, M, dn) -> pooled (B, dn)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B,1,dn)
        key_pad = pad if pad is not None else None  # (B,M) True=ignore
        out, _ = self.attn(q, x, x, key_padding_mask=key_pad)
        return out.squeeze(1)  # (B, dn)


class Predictor(nn.Module):
    """Predictor MLP (dn -> dn): `2·dn²` params (one hidden of width dn)."""

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

    `pool_raw` runs the raw nouns through the readout (the target-side path: no
    operator).
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
        use_polar_conditioning: bool = False,
        use_kind_head: bool = False,
        kind_codebook_size: int = 16,
        use_norm_budget: bool = False,
        use_targeted_actions: bool = False,
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

        # v4 targeted latent actions (jepa_v4_design §1). When ON, the posterior emits a
        # per-slot target mask `g (B, M)`; `_apply_action` applies the operator B_v ONLY to
        # the targeted slots (convex combination, straight-through hard threshold at eval),
        # exact identity elsewhere. The mask heads live on the transition/prior modules
        # (built there when on). DEFAULT FALSE ⟹ g≡1 everywhere ⟹ v3 apply-all (bitwise).
        self.use_targeted_actions = use_targeted_actions

        # v2.1 polar conditioning (design §3): the H map owns the per-slot phase offset
        # θ_offset = H(|k|). ZERO-INIT ⟹ v2.1 == v2.0 at step 0 (the §11 gate). When
        # use_polar_conditioning=False, no conditioner is built and the model is exactly
        # v2.0 (the operator's theta_offset stays None).
        self.use_polar_conditioning = use_polar_conditioning
        n_blocks = d_noun // 2
        if use_polar_conditioning:
            from .conditioning import PolarConditioner
            self.conditioner = PolarConditioner(n_blocks)
        else:
            self.conditioner = None

        # v2.1 optional kind head (design §7): diagnostic/demo ONLY, never routes.
        self.use_kind_head = use_kind_head
        if use_kind_head:
            from .conditioning import KindHead
            self.kind_head = KindHead(n_blocks, codebook_size=kind_codebook_size)
        else:
            self.kind_head = None

        # L_pred aux branch (design §5): Readout pool over a* -> Predictor -> zhat, with
        # an EMA(SlotEncoder+Readout) target over the RAW nouns of text_t+1 (v1 pattern).
        self.readout = Readout(d_noun, n_heads)
        self.predictor = Predictor(d_noun)
        self._online_bundle = _EncoderReadout(self.encoder, self.readout)
        self.ema = copy.deepcopy(self._online_bundle)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # Re-point the EMA copy's frozen token_emb at the shared online table so we do
        # not double-store (V×d) frozen weights in every checkpoint.
        if hasattr(self.encoder, "token_emb") and hasattr(self.ema.encoder, "token_emb"):
            self.ema.encoder.token_emb = self.encoder.token_emb

        # entity-campaign norm budget (entity §1.0–§1.2). When ON, the operator
        # renormalizes each slot's modulus profile to its pre-step norm and returns the
        # extracted per-slot log-scale; the model accumulates it as `s_acc (B, M)` and
        # makes it visible to the anchor/readout (InfoNCE/decoder-conditioning) geometry —
        # NOT the decoder memory (the leakage invariant is unchanged; the decoder still
        # sees only `a*`). The augmented slot `concat[a, s_acc] (B, M, dn+1)` is projected
        # back to dn by `scale_readout_proj` (the ONLY new trainable param the budget adds:
        # (dn+1)*dn) before the existing Readout, so Readout is untouched and the pooled
        # InfoNCE vector now carries the irreversibility scalar. Built ONLY when on
        # (default off ⟹ bitwise v3: no extra param, no scale threading).
        self.use_norm_budget = use_norm_budget
        if use_norm_budget:
            self.scale_readout_proj = nn.Linear(d_noun + 1, d_noun)
        else:
            self.scale_readout_proj = None

    # ------------------------------------------------------------------ action path
    def _gate_from_logits(self, g_logits: torch.Tensor) -> torch.Tensor:
        """Resolve per-slot mask logits g_logits (B, M) to the gate g (B, M) (§1.2).

        Training: the SOFT sigmoid gate (full gradient). Eval: the HARD 0/1 threshold at
        0.5 with a straight-through estimator (hard forward, soft backward) — so targeted
        slots get the EXACT operator output and identity slots the EXACT input, making
        disjoint-support commutation a structural property (§1.0).
        """
        g_soft = torch.sigmoid(g_logits)  # (B, M) in (0, 1)
        if self.training:
            return g_soft
        g_hard = (g_soft > 0.5).to(g_soft.dtype)
        return g_hard + (g_soft - g_soft.detach())  # straight-through

    def _apply_action(self, k: torch.Tensor, v_onehot: torch.Tensor, g_logits=None):
        """a* = B_v k for ONE sequence-level action per pair (design §2.3).

        v_onehot: (B, V) hard straight-through one-hot. Broadcast to (B, M, V) so the
        SAME action is applied to all M slots via the operator's soft-mix path; with a
        hard one-hot this is numerically the hard-index path (design §2.3). Using the
        soft path keeps the ST gradient flowing into v_logits/posterior.

        v2.1 (design §3): when polar conditioning is on, the per-slot phase advance is
        `θ_eff = θ_v + H(|k|)`. The offset is computed from the PRE-step modulus (with
        gradient, design §3.1) and passed to the operator. Zero-init H ⟹ offset == 0 at
        step 0 ⟹ identical to v2.0.

        v4 targeted actions (jepa_v4_design §1.2): when `g_logits` is provided (the mask
        head output), the operator's effect is gated PER SLOT by the convex combination
            a_i = g_i · (B_v k_i) + (1 − g_i) · k_i
        with `g_i` the straight-through gate (hard 0/1 at eval). The polar conditioner
        reads `|k|` BEFORE gating (§1.2: H sees the pre-step modulus, unchanged), so the
        conditioning is untouched by the mask; on an identity slot the gated output is
        `k_i` exactly, so at the next hop H feeds on the EXACT unchanged modulus profile.
        The norm-budget `scale_delta` is ALSO gated by `g` (§1.2) so identity slots
        accumulate EXACTLY ZERO scale — the load-bearing correctness point that keeps the
        identity readout and the retraction inverse exact.

        Return contract:
          - targeted OFF (g_logits is None):
              * norm budget OFF: bare `a (B, M, dn)` — BITWISE v2/v3.
              * norm budget ON: `(a, scale_delta)` (entity §1.3) — BITWISE v3.
          - targeted ON (g_logits given): a trailing `g_hard (B, M)` (the eval hard mask,
            stored for the inverse §1.5) is appended:
              * norm budget OFF: `(a, g_hard)`.
              * norm budget ON: `(a, scale_delta, g_hard)`.
        """
        B, M, _ = k.shape
        v_slots = v_onehot.unsqueeze(1).expand(B, M, -1)  # (B, M, V)
        # H reads the PRE-step, PRE-gate modulus (§1.2) — gating is applied to the OUTPUT.
        theta_offset = self.conditioner(k) if self.conditioner is not None else None

        if self.use_norm_budget:
            a_op, scale_delta_op = self.operator.apply(
                k, v_slots, theta_offset=theta_offset, norm_budget=True
            )  # (B, M, dn), (B, M)
        else:
            if theta_offset is not None:
                a_op = self.operator.apply(k, v_slots, theta_offset=theta_offset)
            else:
                a_op = self.operator.apply(k, v_slots)  # (B, M, dn)
            scale_delta_op = None

        if g_logits is None:
            # Targeted OFF ⟹ v3 apply-all, bitwise (no convex combination).
            if self.use_norm_budget:
                return a_op, scale_delta_op
            return a_op

        # --- targeted ON: per-slot gated convex combination (§1.2) ---
        g = self._gate_from_logits(g_logits)               # (B, M) soft (train) / ST-hard (eval)
        a = g.unsqueeze(-1) * a_op + (1.0 - g).unsqueeze(-1) * k  # gated noun
        # g_hard is the eval threshold — stored for the EXACT inverse (§1.5). Computed
        # from the (detached) soft gate so it is a clean 0/1 partition regardless of mode.
        with torch.no_grad():
            g_hard = (torch.sigmoid(g_logits) > 0.5).to(k.dtype)  # (B, M)
        if self.use_norm_budget:
            # Gate the scale too: identity slots (g=0) accumulate ZERO scale (§1.2).
            scale_delta = g * scale_delta_op               # (B, M)
            return a, scale_delta, g_hard
        return a, g_hard

    # ------------------------------------------------------------------ targeted-mask input
    @torch.no_grad()
    def _target_slots(self, tgt_ids: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
        """Detached EMA raw-noun encode of s_{t+1} -> k_tgt (B, M, dn) (jepa_v4_design §1.1).

        The mask head reads `[k ; k_tgt ; |k_tgt − k|]`; `k_tgt` is the stop-grad EMA
        encode of the next state (the SAME `self.ema.encoder` trunk the InfoNCE key uses).
        Detaching here is what keeps the mask inferable from the pair WITHOUT opening a
        continuous future channel (§1.4) — the mask carries location bits, not content.
        """
        return self.ema.encoder(tgt_ids, tgt_pad)[1].detach()  # k slot nouns

    def _unpack_apply(self, ret):
        """Normalize `_apply_action`'s variable-length return to (a, scale_delta, g_hard).

        `_apply_action` returns one of {a, (a, scale_delta), (a, g_hard),
        (a, scale_delta, g_hard)} depending on (use_norm_budget, use_targeted_actions).
        This collapses all four shapes to a uniform triple with None for the absent
        elements, so the forward/unroll bodies stay branch-free. A bare 2-tuple is
        disambiguated by the instance flags: it is (a, scale_delta) when the budget is on
        and targeting is off, or (a, g_hard) when targeting is on and the budget is off
        (the two never coexist in a 2-tuple — both-on is a 3-tuple).
        """
        budget = self.use_norm_budget
        targeted = self.use_targeted_actions
        if budget and targeted:
            a, scale_delta, g_hard = ret
            return a, scale_delta, g_hard
        if budget:           # (a, scale_delta)
            a, scale_delta = ret
            return a, scale_delta, None
        if targeted:         # (a, g_hard)
            a, g_hard = ret
            return a, None, g_hard
        return ret, None, None  # bare a

    # ------------------------------------------------------------------ readout helper
    def _anchor_pool(self, a: torch.Tensor, s_acc: torch.Tensor | None) -> torch.Tensor:
        """Pooled readout over `a*` for the InfoNCE/L_pred anchor (entity §1.1).

        When the norm budget is on, the accumulated per-slot log-scale `s_acc (B, M)` is
        concatenated as an extra channel (`a_aug = concat[a, s_acc] (B, M, dn+1)`) and
        projected back to dn by `scale_readout_proj` BEFORE the existing Readout, so the
        irreversibility scalar enters the anchor/readout geometry. The decoder memory
        stays `a` (the leakage invariant is unchanged — `s_acc` never reaches the
        decoder). Budget off ⟹ this is exactly `self.readout(a)` (bitwise v3).
        """
        if self.scale_readout_proj is not None and s_acc is not None:
            a_aug = torch.cat([a, s_acc.unsqueeze(-1).to(a.dtype)], dim=-1)  # (B,M,dn+1)
            a = self.scale_readout_proj(a_aug)                              # (B,M,dn)
        return self.readout(a)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        src_ids: torch.Tensor,
        src_pad: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_pad: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
        posterior_inputs: tuple | None = None,
        decoder_target: tuple | None = None,
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
        memory; the teacher-forced CE target enters only as the AR target.

        v6 §B surface-augmentation invariance (research/jepa_v6_unsupervised_design.md):
        two OPTIONAL overrides decouple the surface frame the posterior/noun-path sees from
        the surface frame the decoder reconstructs (LABEL-FREE — both are rendered TEXT):
          - `posterior_inputs = (p_src_ids, p_src_pad, p_tgt_ids, p_tgt_pad)`: frame φ for
            BOTH posterior inputs AND the noun-path encoder of s_t. The surface variance
            common to the φ pair cannot route into v (it is held constant across the pair),
            so v is pushed onto the controllable semantic delta. Default None ⟹ use
            (src_ids, src_pad, tgt_ids, tgt_pad) — the bitwise v4 single-frame path.
          - `decoder_target = (d_tgt_ids, d_tgt_pad)`: frame φ' of s_{t+1} — the decoder's
            teacher-forced CE TARGET (the allowed φ'→decoder channel, §6). It does NOT enter
            the posterior or the noun path. Default None ⟹ the decoder target is tgt_ids
            (single-frame path).
        The leakage invariant is PRESERVED: the decoder memory is still ONLY a*; φ' enters
        only as the AR target; v is inferred only from the φ pair.
        """
        # v6 §B: resolve the φ pair (posterior + noun path) and the φ' decoder CE target.
        if posterior_inputs is not None:
            p_src_ids, p_src_pad, p_tgt_ids, p_tgt_pad = posterior_inputs
        else:
            p_src_ids, p_src_pad, p_tgt_ids, p_tgt_pad = src_ids, src_pad, tgt_ids, tgt_pad
        if decoder_target is not None:
            dec_tgt_ids, dec_tgt_pad = decoder_target
        else:
            dec_tgt_ids, dec_tgt_pad = tgt_ids, tgt_pad

        # --- noun path: φ(text_t) only (no t+1 info reaches k) ---
        _, k, _ = self.encoder(p_src_ids, p_src_pad)  # verb_logits IGNORED in v2

        # --- posterior: sees the φ pair, emits ONE discrete action per pair ---
        v_onehot, v_logits, pool_t = self.transition(
            p_src_ids, p_src_pad, p_tgt_ids, p_tgt_pad, tau, hard
        )

        # --- prior p(v | text_t) for autonomous rollout (KL target is stopgrad q) ---
        p_logits = self.prior(pool_t)

        # --- v4 targeted mask (jepa_v4_design §1.1/§1.3): posterior emits the per-slot
        # target mask from `[k ; k_tgt ; |k_tgt − k|]` (k_tgt = detached EMA encode of
        # s_{t+1}); the prior distills it from `k` alone. Both None when targeted off.
        # v6 §B: k_tgt reads the φ frame of s_{t+1} (the posterior's frame), NOT φ' — the
        # mask is a location signal inferred from the same φ pair the posterior conditions on. ---
        if self.use_targeted_actions:
            k_tgt = self._target_slots(p_tgt_ids, p_tgt_pad)     # (B, M, dn) detached
            g_logits = self.transition.forward_mask(k, k_tgt)    # (B, M)
            g_prior_logits = self.prior.forward_mask(k)          # (B, M)
        else:
            g_logits = None
            g_prior_logits = None

        # --- operator-transformed nouns: the ONLY decoder conditioning channel ---
        # entity §1.3: with the budget on, _apply_action returns (a, scale_delta); the
        # single-hop accumulator s_acc == scale_delta (s starts at 0). Off ⟹ bare a.
        # v4 §1.2: with targeting on, a trailing g_hard (the eval hard mask) is returned.
        a, scale_delta, g_hard = self._unpack_apply(
            self._apply_action(k, v_onehot, g_logits=g_logits)
        )
        s_acc = scale_delta  # single hop: s_acc == this step's (gated) scale_delta

        # --- token decoder: memory = a* ONLY (structural leakage block). v6 §B: the CE
        # target is the φ' frame (dec_tgt_*); a* is built from the φ pair, so the only
        # frame-invariant signal the decoder can rely on through v is the semantic delta. ---
        logits = self.decoder(a, dec_tgt_ids, dec_tgt_pad)  # (B, T, V)

        out = {
            "k": k,
            "a": a,
            "v": v_onehot.argmax(dim=-1),  # (B,)
            "v_onehot": v_onehot,
            "v_logits": v_logits,
            "p_logits": p_logits,
            "logits": logits,
            # entity §1.3: scale_delta / s_acc are None when the budget is off (back-compat;
            # downstream guards on `is not None`).
            "scale_delta": scale_delta,
            "s_acc": s_acc,
            # v4 §1.1/§1.3: per-slot mask logits + the stored hard mask (None when off).
            "g_logits": g_logits,
            "g_prior_logits": g_prior_logits,
            "g_hard": g_hard,
        }

        # v2.1 optional kind readout (design §7): diagnostic label only, never routes.
        if self.kind_head is not None:
            out["kind_ids"] = self.kind_head.assign(k)  # (B, M)

        # --- L_pred aux branch (optional; design §5). v6 §B: the EMA target reads the φ'
        # frame (dec_tgt_*) so the anchor zhat is contrasted against the SAME frame the
        # decoder reconstructs — invariance pressure flows into the readout geometry too. ---
        if self.use_pred:
            pooled = self._anchor_pool(a, s_acc)  # (B, dn) — carries s_acc when budget on
            zhat = self.predictor(pooled)   # (B, dn)
            with torch.no_grad():
                z = self.ema.pool_raw(dec_tgt_ids, dec_tgt_pad)  # (B, dn) raw-noun pool of t+1
            out["zhat"] = zhat
            out["z_target"] = z.detach()
        else:
            out["zhat"] = None
            out["z_target"] = None

        return out

    # forward_v2 is the name diagnostics.py duck-types against (it must be able to
    # tell a v2 model from the legacy v1 model, whose `forward(src, src_pad)` has a
    # different signature). Alias it to `forward` so the diagnostics' v-ablation CE
    # gap and generated-sample paths resolve to the real v2 forward.
    forward_v2 = forward

    # ------------------------------------------------------------------ unroll (v3 §2)
    def forward_unroll(
        self,
        s0_ids: torch.Tensor,
        s0_pad: torch.Tensor,
        s1_ids: torch.Tensor,
        s1_pad: torch.Tensor,
        s2_ids: torch.Tensor,
        s2_pad: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True,
    ) -> list[dict]:
        """Two-hop multi-step unroll over a chain triple (s0, s1, s2). v3 §2.3.

        Composes the operator twice from the SAME start nouns, threading the polar
        conditioning per hop:

            k0       = encoder(s0).k                       # start nouns
            v1       = posterior(s0, s1)                   # action s0 -> s1
            a1       = _apply_action(k0, v1)               # θoff_1 = H(|k0|)  (hop-1 modulus)
            v2       = posterior(s1, s2)                   # action s1 -> s2
            a2       = _apply_action(a1, v2)               # θoff_2 = H(|a1|)  (hop-2 modulus, §2.3)

        H reads the modulus of the operator's INPUT at each hop (`|k0|` for hop 1,
        `|a1|` for hop 2) — the design-mandated state-dependence (jepa_v21_polar §3.3):
        under a scaling hop-1 verb the modulus changes and the hop-2 offset must shift
        with it. `_apply_action` already computes `conditioner(<its k arg>)`, so passing
        `a1` at hop 2 needs no conditioner change — only this loop calling it on `a1`.

        Leakage (v3 §2.5): hop-2's only `s2` channel into the decoder memory `a2` is the
        discrete `v2` (the posterior's ⌈log₂V⌉ bits). `θoff_2 = H(|a1|)` is a function of
        `(k0, v1)` only — NO `s2`. So `a2 = f(k0, v1, v2)`; perturbing `s2` moves `a2`
        ONLY through `v2`. The hop-2 decoder's teacher-forced context is `s2_ids` (the
        standard AR target — identical to v2's single-hop teacher forcing).

        This method is LOSS-FREE: it returns the raw per-hop outputs and the trainer
        (Task B) applies the hop weights (1.0/0.5) and assembles the per-hop loss with
        the cross-hop InfoNCE hard negatives. Keeps model.py (C) and the train loop (B)
        disjoint.

        Returns a list of TWO per-hop dicts (hop 1, then hop 2), each with the same keys
        as `forward`:
            k         (B, M, dn)  the operator INPUT nouns at this hop (k0 / a1)
            a         (B, M, dn)  a* = operator output (decoder memory)  (a1 / a2)
            v         (B,)        argmax posterior action
            v_onehot  (B, V)      hard ST one-hot action
            v_logits  (B, V)      posterior logits
            p_logits  (B, V)      prior logits  (distilled from this hop's source pool)
            logits    (B, T, V)   token decoder logits over the hop target
            zhat      (B, dn)     L_pred/InfoNCE anchor over a* (None if use_pred=False)
            z_target  (B, dn)     EMA raw-noun pool of this hop's target, stop-grad (or None)
            scale_delta (B, M)    this hop's per-slot log_rho (entity §1.3; None if budget off)
            s_acc     (B, M)      accumulated log-scale AFTER this hop (None if budget off)

        Norm budget (entity §1.3): with `use_norm_budget` on, the loop additionally
        threads a per-slot log-scale accumulator `s_acc (B, M)` (zero at start) and stores
        each hop's `scale_delta` for the retraction probe's exact inverse. The anchor pool
        sees the accumulated `s_acc` through `scale_readout_proj` (irreversibility enters
        the InfoNCE geometry); the decoder memory stays `a` (leakage unchanged). Both keys
        are None when the budget is off (back-compat; downstream guards on `is not None`).
        """
        # --- start nouns: text_t0 only (no future info reaches k0) ---
        _, k0, _ = self.encoder(s0_ids, s0_pad)  # verb_logits IGNORED

        hops = [
            (s0_ids, s0_pad, s1_ids, s1_pad),  # hop 1: action s0 -> s1, target s1
            (s1_ids, s1_pad, s2_ids, s2_pad),  # hop 2: action s1 -> s2, target s2
        ]

        outs: list[dict] = []
        k_in = k0
        # entity §1.3: per-slot log-scale accumulator, scale=1.0 (log 0) at start. Only
        # threaded when the budget is on; otherwise stays None (bitwise v3).
        s_acc = torch.zeros(k0.shape[0], k0.shape[1], device=k0.device) if self.use_norm_budget else None
        for src_ids, src_pad, tgt_ids, tgt_pad in hops:
            # --- per-hop posterior q(v | src, tgt): own discrete action per hop (§2.2) ---
            v_onehot, v_logits, pool_src = self.transition(
                src_ids, src_pad, tgt_ids, tgt_pad, tau, hard
            )
            # --- per-hop prior p(v | src) distilled from this hop's source pool (§2.2) ---
            p_logits = self.prior(pool_src)

            # --- v4 targeted mask: per-hop posterior mask from k_in (operator INPUT) and
            # the detached EMA encode of THIS hop's target; prior mask from k_in alone. ---
            if self.use_targeted_actions:
                k_tgt = self._target_slots(tgt_ids, tgt_pad)         # (B, M, dn) detached
                g_logits = self.transition.forward_mask(k_in, k_tgt)  # (B, M)
                g_prior_logits = self.prior.forward_mask(k_in)        # (B, M)
            else:
                g_logits = None
                g_prior_logits = None

            # --- composed application: conditioning reads |k_in| at THIS hop (§2.3),
            # gated per-slot by the mask (§1.2) when targeting is on. ---
            a, scale_delta, g_hard = self._unpack_apply(
                self._apply_action(k_in, v_onehot, g_logits=g_logits)
            )
            if self.use_norm_budget:
                s_acc = s_acc + scale_delta  # log-domain accumulate (gated ⟹ identity=0)

            # --- token decoder: memory = a* ONLY (structural leakage block) ---
            logits = self.decoder(a, tgt_ids, tgt_pad)

            hop_out = {
                "k": k_in,
                "a": a,
                "v": v_onehot.argmax(dim=-1),
                "v_onehot": v_onehot,
                "v_logits": v_logits,
                "p_logits": p_logits,
                "logits": logits,
                "scale_delta": scale_delta,
                "s_acc": s_acc if self.use_norm_budget else None,
                "g_logits": g_logits,
                "g_prior_logits": g_prior_logits,
                "g_hard": g_hard,
            }
            if self.kind_head is not None:
                hop_out["kind_ids"] = self.kind_head.assign(k_in)

            # --- L_pred / InfoNCE anchor + EMA target for this hop (§2.4) ---
            if self.use_pred:
                pooled = self._anchor_pool(a, hop_out["s_acc"])  # carries s_acc when on
                zhat = self.predictor(pooled)    # (B, dn)
                with torch.no_grad():
                    z = self.ema.pool_raw(tgt_ids, tgt_pad)  # EMA pool of THIS hop's target
                hop_out["zhat"] = zhat
                hop_out["z_target"] = z.detach()
            else:
                hop_out["zhat"] = None
                hop_out["z_target"] = None

            outs.append(hop_out)
            k_in = a  # next hop composes on this hop's output (a2 = B_v2(B_v1 k0))

        return outs

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
        # v4 §1.3: at rollout the posterior is gone, so the mask comes from the PRIOR mask
        # head (reads state_t nouns k only — leakage-clean). None when targeting off.
        g_logits = self.prior.forward_mask(k) if self.use_targeted_actions else None
        # entity §1.3: _apply_action returns scale_delta when the budget is on; the decoder
        # generates from `a` only (leakage unchanged). scale_delta / g_hard surface in the
        # rollout dict so callers (rollout-fidelity eval, retraction probe) can thread them.
        a, scale_delta, g_hard = self._unpack_apply(
            self._apply_action(k, v_onehot, g_logits=g_logits)
        )
        gen_ids = self.decoder.generate(a, max_tokens=max_tokens, temperature=temperature)
        return {
            "v": v, "a": a, "gen_ids": gen_ids,
            "scale_delta": scale_delta, "g_hard": g_hard,
        }

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
    def step_latent(self, k: torch.Tensor, verb_idx, g_logits=None):
        """One tick in latent space: a* = B_v k. verb_idx scalar or (B,)/(B,M).

        v2.1 (design §3/§4.2): when polar conditioning is on, returns
        `(a, theta_offset)` so the caller can pass the SAME offset (computed from the
        PRE-step modulus) back to `undo_latent` for an exact inverse — the offset must
        never be recomputed from the post-step noun under a scaling verb (§3.3). When
        conditioning is off, returns just `a` (v2.0 signature preserved).

        entity §1.3: when the norm budget is on, `step_latent` ALSO returns the per-slot
        `scale_delta` (the operator's `log_rho`) so the caller threads it back to
        `undo_latent(scale_delta=...)` for an EXACT inverse (the budget renormalizes the
        radius, so the undo needs the stored scale). The return is then
        `(a, theta_offset, scale_delta)` (theta_offset is None if no conditioner) —
        threaded exactly like theta_offset is today.

        v4 targeted actions (jepa_v4_design §1.2/§1.5): pass `g_logits (B, M)` (e.g. from
        a mask head, or a constructed mask for the retraction probe) to gate the operator
        per slot. The return then APPENDS the stored hard mask `g_hard (B, M)` as the LAST
        element so the caller threads it to `undo_latent(g_hard=...)` for an exact inverse
        on a mixed mask. `g_hard` is appended in ALL conditioner/budget shapes:
            (a, g_hard) | (a, theta_offset, g_hard) | (a, theta_offset, scale_delta, g_hard)
        When `g_logits is None` the return is exactly the v2.1/entity shape (no g_hard) —
        the bitwise-preserved path.
        """
        v_slots = self._verb_to_slots(k, verb_idx)
        theta_offset = self.conditioner(k) if self.conditioner is not None else None

        if g_logits is None:
            # --- bitwise v2.1/entity path (no gating, no g_hard appended) ---
            if self.use_norm_budget:
                a, scale_delta = self.operator.apply(
                    k, v_slots, theta_offset=theta_offset, norm_budget=True
                )
                return a, theta_offset, scale_delta
            if theta_offset is not None:
                a = self.operator.apply(k, v_slots, theta_offset=theta_offset)
                return a, theta_offset
            return self.operator.apply(k, v_slots)

        # --- v4 gated path: per-slot convex combination (mirror _apply_action §1.2) ---
        g = self._gate_from_logits(g_logits)              # (B, M)
        with torch.no_grad():
            g_hard = (torch.sigmoid(g_logits) > 0.5).to(k.dtype)  # (B, M)
        if self.use_norm_budget:
            a_op, scale_delta_op = self.operator.apply(
                k, v_slots, theta_offset=theta_offset, norm_budget=True
            )
            a = g.unsqueeze(-1) * a_op + (1.0 - g).unsqueeze(-1) * k
            scale_delta = g * scale_delta_op              # identity slots ⟹ 0
            return a, theta_offset, scale_delta, g_hard
        if theta_offset is not None:
            a_op = self.operator.apply(k, v_slots, theta_offset=theta_offset)
        else:
            a_op = self.operator.apply(k, v_slots)
        a = g.unsqueeze(-1) * a_op + (1.0 - g).unsqueeze(-1) * k
        return a, theta_offset, g_hard

    def undo_latent(self, a: torch.Tensor, verb_idx, theta_offset=None, scale_delta=None,
                    g_hard=None) -> torch.Tensor:
        """Exact undo: k = B_v^{-1} a (structural inverse from the operator).

        v2.1 (design §4.2): pass the `theta_offset` returned by `step_latent` for an
        exact inverse. If `theta_offset` is None while conditioning is on, the offset is
        recomputed from `|a|` — EXACT for pure-rotation verbs (modulus preserved, so
        `H(|a|) == H(|k|)`, design §3.3) but only approximate under a scaling verb. The
        identity-persistence diagnostic (§8.1) asserts the pure-rotation case.

        entity §1.3: when the norm budget is on, pass the `scale_delta` returned by
        `step_latent` so the inverse re-applies the renormalized radius BEFORE inverting
        — the round-trip is then exact including the tracked scale. The budget undo
        REQUIRES `scale_delta` (the operator asserts it).

        v4 targeted actions (jepa_v4_design §1.5): pass the `g_hard (B, M)` stored by the
        forward gated step. The inverse is then gated by the SAME hard mask:
            k_i = g_i · inverse_apply(a, v, scale_delta=g·scale_delta_op)_i + (1 − g_i)·a_i
        On a targeted slot (g=1) `a_i = B_v k_i` so the operator inverse recovers `k_i`;
        on an identity slot (g=0) `a_i = k_i` so returning `a_i` recovers it exactly. The
        stored `scale_delta` (already gated by g in the forward) feeds the operator inverse
        unchanged — on identity slots it is 0, so the operator-inverse branch (which is
        masked out anyway) sees no radius change. When `g_hard is None` this is the exact
        v2.1/entity inverse (the bitwise path).
        """
        v_slots = self._verb_to_slots(a, verb_idx)
        if self.use_norm_budget:
            if self.conditioner is not None and theta_offset is None:
                theta_offset = self.conditioner(a)
            k_inv = self.operator.inverse_apply(
                a, v_slots, theta_offset=theta_offset,
                norm_budget=True, scale_delta=scale_delta,
            )
        elif self.conditioner is not None:
            if theta_offset is None:
                theta_offset = self.conditioner(a)
            k_inv = self.operator.inverse_apply(a, v_slots, theta_offset=theta_offset)
        else:
            k_inv = self.operator.inverse_apply(a, v_slots)

        if g_hard is None:
            return k_inv
        # v4 §1.5: gate the inverse with the SAME hard mask the forward used. Identity
        # slots (g=0) return `a` unchanged (which equals the original k_i on those slots).
        gh = g_hard.to(k_inv.dtype).unsqueeze(-1)  # (B, M, 1)
        return gh * k_inv + (1.0 - gh) * a

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
    # GatedMLPTransition lives in baseline_transition.py (this task owns it). Import it
    # directly from the owning module rather than through the package __getattr__
    # re-export so this factory does not depend on Task B's __init__.py edit landing first.
    from twm.jepa.baseline_transition import GatedMLPTransition

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
        "gated_mlp": GatedMLPTransition,
    }.get(m.operator_group, RotationScaleOperator)
    # GatedMLPTransition (v3 §4.2) reads d_e/d_h from the optional model.gated_mlp block;
    # _construct kwarg-filters them out for operator classes that don't accept them.
    gmlp = getattr(m, "gated_mlp", None)
    operator = _construct(
        op_cls,
        n_verbs=m.n_verbs,
        d_noun=m.d_noun,
        block=m.block,
        d_e=getattr(gmlp, "d_e", 4),
        d_h=getattr(gmlp, "d_h", 8),
    )

    # v4 targeted latent actions (jepa_v4_design §1): the mask heads are built ONLY when
    # use_targeted_actions=True. `mask_hidden` comes from the optional model.targeted block
    # (B-owned schema); default 2*dn for nano. getattr keeps this safe if B's schema field
    # has not landed yet (mirrors the use_norm_budget getattr pattern).
    use_targeted_actions = getattr(m, "use_targeted_actions", False)
    targeted_cfg = getattr(m, "targeted", None)
    mask_hidden = getattr(targeted_cfg, "mask_hidden", None)
    if mask_hidden is None:
        mask_hidden = 2 * m.d_noun

    # Posterior + prior action heads share the encoder's bound text trunk (design §2.1):
    # zero new attention params, posterior's view in the noun-building space.
    transition = _construct(
        TransitionEncoder,
        encode_text_fn=encoder.encode_text,
        d_model=m.d_model,
        n_verbs=m.n_verbs,
        mlp_hidden=m.transition.mlp_hidden,
        use_delta=m.transition.use_delta,
        use_targeted_actions=use_targeted_actions,
        d_noun=m.d_noun,
        mask_hidden=mask_hidden,
    )
    prior = _construct(
        PriorHead,
        d_model=m.d_model,
        n_verbs=m.n_verbs,
        mlp_hidden=m.prior.mlp_hidden,
        use_targeted_actions=use_targeted_actions,
        d_noun=m.d_noun,
        mask_hidden=mask_hidden,
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
        use_polar_conditioning=getattr(m, "use_polar_conditioning", False),
        use_kind_head=getattr(m, "use_kind_head", False),
        kind_codebook_size=getattr(m, "kind_codebook_size", 16),
        use_norm_budget=getattr(m, "use_norm_budget", False),
        use_targeted_actions=use_targeted_actions,
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
