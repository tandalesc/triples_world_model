"""JEPA v2 action heads — TransitionEncoder (posterior) + PriorHead.

Authoritative spec: research/jepa_v2_latent_actions.md §2 (TransitionEncoder),
§3 (PriorHead), §11 Task A (frozen interface). Training-only modules; the
posterior is GONE at inference (rollout samples from the prior).

Why these exist (v2 vs v1, spec §0): v1 predicted the next-state verb from
state_t alone — ill-posed on narrative data, so the encoder hardwired one verb
per slot position and gamed the diversity loss. v2 introduces a training-time
*posterior* q(v | text_t, text_{t+1}) that sees BOTH states and emits ONE
sequence-level discrete action `v` per pair. That single discrete `v` is the
ONLY path next-state information takes into the operator/decoder (the 3-bit
information bottleneck of spec §6). A *prior* p(v | text_t) is distilled from
the posterior so autonomous rollout needs no text_{t+1}.

Load-bearing constraints (do NOT "simplify" away):

  * ONE action per pair, sequence-level (NOT per-slot). Per-slot verbs were the
    v1 positional cheat (spec §0 pillar 2). v_logits is (B, V), v_onehot is
    (B, V) — broadcast to (B, M, V) by the operator's soft-path, which for a
    hard one-hot is numerically the hard-index path.

  * HARD straight-through Gumbel one-hot (`hard=True`), NOT a soft mix. The
    continuous mixing weights of a soft mix are themselves a high-bandwidth
    text_{t+1} channel (spec §6 L3); hard one-hot collapses the channel to
    ceil(log2 V) bits. The ST estimator preserves trainability — gradient flows
    through the soft sample into v_logits and the shared trunk.

  * Gumbel sample runs in fp32 under autocast(enabled=False), mirroring the
    operator/VQ gotcha (cos/sin/exp and the softmax are bf16-unstable; the
    -log(-log U) noise especially).

  * Shared trunk: the posterior and prior consume `encode_text` (the bound
    SlotEncoder method) — zero new attention params. The posterior's
    text_{t+1} pass is isolated here and its sole output is the discrete `v`;
    no text_{t+1} ACTIVATION ever reaches the noun path `k` (spec §6 L4/L5).

  * masked_mean over non-pad positions. Pad mask convention (data.py): True at
    pad positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import gumbel_softmax_sample


def _autocast_off(device_type: str):
    return torch.amp.autocast(device_type=device_type, enabled=False)


def masked_mean(context: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool (B, T, d) context over non-pad positions -> (B, d).

    Args:
        context: (B, T, d) encoded text.
        pad:     (B, T) padding mask, True where pad (data.py convention). None
                 means no padding (all positions valid).

    A row that is entirely pad would divide by zero; clamp the denominator to 1
    so an all-pad sequence pools to the zero vector rather than NaN.
    """
    if pad is None:
        return context.mean(dim=1)
    valid = (~pad.bool()).to(context.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (context * valid).sum(dim=1)                  # (B, d)
    denom = valid.sum(dim=1).clamp_min(1.0)                # (B, 1)
    return summed / denom


class TransitionEncoder(nn.Module):
    """Posterior q(v | text_t, text_{t+1}) -> ONE discrete action per pair.

    Reuses the SlotEncoder's text trunk (the bound `encode_text` callable) for
    both texts — no new attention params. Owns only a small MLP head over the
    pooled (and optionally delta-augmented) pair representation.

    Args:
        encode_text_fn: callable (ids (B,T), pad (B,T)) -> context (B,T,d). The
            SlotEncoder's bound `encode_text` method. NOT registered as a
            submodule (its weights are owned by the SlotEncoder); we hold the
            callable so gradients flow back into the shared trunk by design.
        d_model:    d (trunk width).
        n_verbs:    V (action codebook size).
        mlp_hidden: h (MLP hidden width).
        use_delta:  include the (pool_t1 - pool_t) delta channel as MLP input
            (spec §2.1). True -> 3d input; False -> 2d input.
    """

    def __init__(
        self,
        encode_text_fn,
        d_model: int = 64,
        n_verbs: int = 8,
        mlp_hidden: int = 128,
        use_delta: bool = True,
        use_targeted_actions: bool = False,
        d_noun: int = 32,
        mask_hidden: int = 64,
    ):
        super().__init__()
        # Stash the trunk callable WITHOUT registering it as a submodule. A plain
        # `self.encode_text = encode_text_fn` where the callable is a bound method
        # of an nn.Module would NOT auto-register (it's a method, not a Module),
        # but to be unambiguous and avoid any accidental param duplication in
        # `TransitionEncoder.parameters()`, bypass __setattr__.
        object.__setattr__(self, "_encode_text_ref", [encode_text_fn])

        self.d_model = d_model
        self.n_verbs = n_verbs
        self.use_delta = use_delta

        in_dim = 3 * d_model if use_delta else 2 * d_model
        # MLP: Linear(in -> h) -> GELU -> LayerNorm(h) -> Linear(h -> V). (§2.2)
        self.fc1 = nn.Linear(in_dim, mlp_hidden)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, n_verbs)

        # v4 targeted latent actions (jepa_v4_design §1.1): a SECOND, per-slot head that
        # emits a target-slot mask `g (B, M)` from the per-slot DELTA between the start
        # nouns `k` and the EMA target nouns `k_tgt`. Built ONLY when
        # use_targeted_actions=True (default off ⟹ NO new params, bitwise v3). Unlike the
        # verb head (which reads the POOLED pair vector), the mask head needs PER-SLOT
        # granularity, so it reads the slot nouns directly off the noun trunk — the masked-
        # mean text pool destroys the per-slot signal.
        self.use_targeted_actions = use_targeted_actions
        self.d_noun = d_noun
        self.mask_hidden = mask_hidden
        if use_targeted_actions:
            # feat = cat([k, k_tgt, |k_tgt - k|]) -> (B, M, 3*dn) -> per-slot logit.
            self.mask_fc1 = nn.Linear(3 * d_noun, mask_hidden)
            self.mask_act = nn.GELU()
            self.mask_fc2 = nn.Linear(mask_hidden, 1)
        else:
            self.mask_fc1 = None
            self.mask_act = None
            self.mask_fc2 = None

    @property
    def encode_text(self):
        return self._encode_text_ref[0]

    def pool(self, ids: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        """Shared-trunk masked-mean pool of one text -> (B, d)."""
        context = self.encode_text(ids, pad)  # (B, T, d) — shared trunk
        return masked_mean(context, pad)

    def forward(
        self,
        src_ids: torch.Tensor,   # (B, T_text) text_t token ids
        src_pad: torch.Tensor,   # (B, T_text) pad mask, True at pad
        tgt_ids: torch.Tensor,   # (B, T_text) text_{t+1} token ids
        tgt_pad: torch.Tensor,   # (B, T_text) pad mask, True at pad
        tau: float,              # current annealed Gumbel temperature
        hard: bool = True,       # straight-through hard one-hot (the v2 default)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (v_onehot (B,V), v_logits (B,V), pool_t (B,d)).

        pool_t is returned so the model can reuse it for the PriorHead WITHOUT
        recomputing the text_t trunk pass (spec §3: "reuse pool_t").

        v_onehot is the (straight-through) Gumbel sample over V codes — a hard
        one-hot in the forward value with soft gradient backward (hard=True), or
        the soft sample (hard=False). The operator broadcasts this (B,V) one-hot
        to (B,M,V); with a hard one-hot that is exactly the hard-index path.
        """
        pool_t = self.pool(src_ids, src_pad)    # (B, d)  shared trunk
        pool_t1 = self.pool(tgt_ids, tgt_pad)   # (B, d)  shared trunk

        if self.use_delta:
            # Delta channel makes the transition explicit (§2.1): biases the MLP
            # toward "what changed" rather than "what the next state is".
            pair = torch.cat([pool_t, pool_t1, pool_t1 - pool_t], dim=-1)  # (B, 3d)
        else:
            pair = torch.cat([pool_t, pool_t1], dim=-1)                    # (B, 2d)

        h = self.norm(self.act(self.fc1(pair)))
        v_logits = self.fc2(h)                  # (B, V)

        # v5 Step-1a: stash the pre-logits pair features `h` (mlp_hidden width) so the model
        # can hand them to the training-only verb-anchor aux head WITHOUT a second trunk pass.
        # Detached read happens in the model only when L_verb_anchor is active; storing here is
        # free and does not change the returned tuple (the frozen §11 forward signature).
        self._last_pair_features = h

        # Gumbel-softmax sample in fp32 under autocast-off (mirror operator gotcha).
        with _autocast_off(v_logits.device.type):
            v_onehot = gumbel_softmax_sample(v_logits.float(), tau=tau, hard=hard)
        v_onehot = v_onehot.to(v_logits.dtype)

        return v_onehot, v_logits, pool_t

    @property
    def pair_features_dim(self) -> int:
        """Width of the pre-logits pair features `h` (== mlp_hidden). The verb-anchor aux
        head's input dim (research/jepa_v5_discriminative_design.md Term 1)."""
        return self.fc2.in_features

    def forward_mask(
        self,
        k: torch.Tensor,      # (B, M, dn) start nouns (encode of s_t)
        k_tgt: torch.Tensor,  # (B, M, dn) EMA target nouns (detached encode of s_{t+1})
    ) -> torch.Tensor:
        """Per-slot target-mask logits g_logits (B, M) (jepa_v4_design §1.1).

        Scores each slot from `[k_i ; k_tgt_i ; |k_tgt_i - k_i|]` — the directly
        inferable "which slot's content changed" signal. `k_tgt` is the stop-grad EMA
        raw-noun encode of s_{t+1} (the model detaches it before passing here), which is
        what keeps the mask inferable from the pair WITHOUT leaking a continuous future
        channel (§1.4): the mask carries LOCATION bits, not CONTENT.

        Only valid when `use_targeted_actions=True` (the head exists then). Returns
        per-slot LOGITS; the model applies sigmoid + the straight-through hard threshold.
        """
        assert self.use_targeted_actions, (
            "forward_mask requires use_targeted_actions=True (no mask head built)"
        )
        feat = torch.cat([k, k_tgt, (k_tgt - k).abs()], dim=-1)  # (B, M, 3dn)
        g_logits = self.mask_fc2(self.mask_act(self.mask_fc1(feat)))  # (B, M, 1)
        return g_logits.squeeze(-1)  # (B, M)


class PriorHead(nn.Module):
    """Prior p(v | text_t) -> action logits from state_t alone (spec §3).

    This is what makes autonomous rollout possible: at inference the posterior is
    gone, so the action is sampled from the prior (or set directly by a user
    action). Trained to imitate the (stop-grad) posterior via L_prior (§3.1),
    which lives in losses.py — this module only emits p_logits.

    Reads the SAME pool_t the posterior already computed (passed in by the model;
    do NOT recompute the trunk pass). Deliberately minimal — no LayerNorm, since
    it reads the already-LN'd trunk pool (spec §3).

    Args:
        d_model:    d (trunk width; pool_t dimension).
        n_verbs:    V.
        mlp_hidden: h.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_verbs: int = 8,
        mlp_hidden: int = 64,
        use_targeted_actions: bool = False,
        d_noun: int = 32,
        mask_hidden: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_verbs = n_verbs
        # MLP_prior: Linear(d -> h) -> GELU -> Linear(h -> V). No LayerNorm (§3).
        self.fc1 = nn.Linear(d_model, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, n_verbs)

        # v4 targeted latent actions (jepa_v4_design §1.3): for autonomous rollout the
        # posterior is gone, so the prior must predict the mask too — from state_t slot
        # nouns ALONE (no s_{t+1}, so leakage-clean by the same argument as PriorHead
        # itself). Distilled from the (stop-grad) posterior mask via L_mask_prior
        # (losses.py, Task B). Built ONLY when use_targeted_actions=True (default off ⟹
        # no new params, bitwise v3).
        self.use_targeted_actions = use_targeted_actions
        self.d_noun = d_noun
        self.mask_hidden = mask_hidden
        if use_targeted_actions:
            self.mask_fc1 = nn.Linear(d_noun, mask_hidden)
            self.mask_act = nn.GELU()
            self.mask_fc2 = nn.Linear(mask_hidden, 1)
        else:
            self.mask_fc1 = None
            self.mask_act = None
            self.mask_fc2 = None

    def forward(self, pool_t: torch.Tensor) -> torch.Tensor:
        """pool_t (B, d) -> p_logits (B, V).

        pool_t MUST be the text_t pool (state_t only). The prior never sees
        text_{t+1} (spec §6 L7).
        """
        return self.fc2(self.act(self.fc1(pool_t)))

    def forward_mask(self, k: torch.Tensor) -> torch.Tensor:
        """Per-slot prior mask logits g_prior_logits (B, M) from start nouns k (B, M, dn).

        Reads ONLY state_t slot nouns (no s_{t+1}), so it is leakage-clean (§1.3). At
        rollout the model thresholds these at 0.5 and gates the operator the same way the
        posterior mask does. Only valid when use_targeted_actions=True.
        """
        assert self.use_targeted_actions, (
            "forward_mask requires use_targeted_actions=True (no prior mask head built)"
        )
        g = self.mask_fc2(self.mask_act(self.mask_fc1(k)))  # (B, M, 1)
        return g.squeeze(-1)  # (B, M)
