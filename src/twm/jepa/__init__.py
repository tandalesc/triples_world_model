"""JEPA Operator World Model — contracts + lazy re-exports.

Specs: research/jepa_v2_latent_actions.md (live v2 path),
research/jepa_operator_v1_design.md (demoted v1 baseline, src/twm/jepa/legacy/).

Two kinds of names are exported:

  1. Contract base classes defined *in this file* — the ABCs / Protocols / config
     dataclass skeletons. Concrete modules (operator.py, slot_encoder.py, ...)
     subclass / satisfy them.

  2. Concrete classes re-exported lazily (PEP 562 `__getattr__`) from their owning
     modules: the live v2 path (`model.JEPAOperatorModelV2`, `losses.JEPALoss` /
     its `JEPALossV2` alias, transition/decoder) and the legacy v1 baseline
     (`legacy.model_v1.JEPAOperatorModel`). The lazy import keeps `import twm.jepa`
     cheap and avoids pulling matplotlib/torch heads at package import.

Naming convention (refactor R.1): module files dropped the `_v2` suffix — the v2
path is THE path. The class export names `JEPAOperatorModelV2` / `JEPALossV2` are
kept as the frozen public interface of the in-flight GPU run; `JEPALoss` is an
alias of `JEPALossV2` on the live path.

Shapes (spec §2): B=batch, M=slots, dn=d_noun, V=verbs, T_text=text tokens, d=d_model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# torch is a hard dep of the project (pyproject); import for type hints / nn.Module base.
import torch
import torch.nn as nn


# =============================================================================
# T1 — Operator algebra (owned by src/twm/jepa/operator.py). Spec §1.6.
# =============================================================================

class Operator(nn.Module, ABC):
    """Block-diagonal 2×2 operator algebra over the noun space. Spec §1.

    1-to-N is the encoder's job, NOT the operator's. `B_v k` is deterministic;
    multiple outcomes from one state are routed into different slots by the slot
    encoder's competitive assignment. The operator never solves 1-to-N.

    Abelian commutativity is silent: `B_u B_v = B_v B_u` (diagonal-block scale +
    same-plane rotation commute). Harmless in v1 (no multiturn, no composition
    loss). Non-abelian expressiveness enters only via the v2 `velocity(k, v)`.

    All operator math runs in fp32 under `autocast(enabled=False)` (cos/sin/exp
    are bf16-unstable at large magnitude — mirrors the VQ gotcha).
    """

    @abstractmethod
    def apply(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """a* = B_v k.  (B,M,dn), (B,M) -> (B,M,dn).  RoPE-style, no matrix materialized."""
        raise NotImplementedError

    @abstractmethod
    def inverse_apply(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """k = B_v^{-1} a = diag_blocks(R(θ_b)ᵀ / r_b) a.  (B,M,dn), (B,M) -> (B,M,dn).

        STRUCTURAL inverse — exact, no consistency/invertibility loss.
        """
        raise NotImplementedError

    @abstractmethod
    def velocity(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Generator action for the T-step seam.  (B,M,dn), (B,M) -> (B,M,dn).

        v2 hook. v1: static block generator. v2: state-dependent MLP -> skew.
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self, k: torch.Tensor, v: torch.Tensor, T: int = 1) -> torch.Tensor:
        """T-step integration seam. T hard-set to 1 in v1 (single exp map == apply()).

        (B,M,dn), (B,M), int -> (B,M,dn). Endpoint-only supervision is structurally
        identical to a Neural-ODE endpoint loss.
        """
        raise NotImplementedError

    @abstractmethod
    def structural_sanity(self, v: int) -> dict:
        """Runtime invariant check for verb v -> {"bbT_err": float, "inv_err": float}.

        ≈ 0 when `apply` is correct. A diagnostic, NOT a loss.
        """
        raise NotImplementedError

    @abstractmethod
    def bake(self) -> dict:
        """Export-ready per-verb (cosθ, sinθ, r) for JS / INT8. Spec §8."""
        raise NotImplementedError

    @property
    @abstractmethod
    def n_verbs(self) -> int:
        """Number of verbs V in the operator codebook."""
        raise NotImplementedError


# Concrete operator names (frozen as contract symbols; bodies live in operator.py):
#   RotationScaleOperator  — v1 mandated family (C*^(d/2)): theta (V, dn//2), log_r (V, dn//2).
#   RotationOperator       — pure rotation (log_r=0 frozen) config ablation.
#   SOnCayleyOperator      — interface stub, raises NotImplementedError.


# =============================================================================
# T2 — Slot encoder (owned by src/twm/jepa/slot_encoder.py). Spec §2.
# =============================================================================

@runtime_checkable
class SlotEncoderProtocol(Protocol):
    """ALBERT-tied text self-attn → M-query cross-attn → n_iters=3 slot coordination.

    NounHead standardizes (NOT L2-norm) the nouns; VerbHead emits verb logits.
    Slot coordination (n_iters=3 self-attn) is MANDATORY — single-pass cross-attn
    gives zero slot-competition pressure → stripe collapse (spec §2 FIX).
    """

    def forward(
        self,
        text_ids: torch.Tensor,   # (B, T_text)
        text_pad: torch.Tensor,   # (B, T_text) bool/0-1 padding mask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (slots, k, verb_logits).

        slots:       (B, M, d)   coordinated slot reps
        k:           (B, M, dn)  nouns, standardized (zero-mean/unit-var per dim), NOT L2-normed
        verb_logits: (B, M, V)   pre-Gumbel-softmax verb logits
        """
        ...


# =============================================================================
# T3 — Losses (owned by src/twm/jepa/losses.py). Spec §3.
# =============================================================================

@runtime_checkable
class JEPALossProtocol(Protocol):
    """L = w_pred·L_pred + w_sigreg·L_sigreg + w_div·L_div.

    Defaults: w_pred=1.0, w_sigreg=0.05, w_div=0.1, w_scale_reg=0.0.

    - L_pred:   MSE(zhat, z.detach()), z = EMA-encoder(next_text) pooled in raw
                noun space (not rotated, not L2-normed).
    - L_sigreg: sliced isotropic-Gaussian GoF on STANDARDIZED nouns (center+scale by
                batch stats, NOT per-vector L2-norm — L2-projection kills the gradient).
                Epps-Pulley CF GoF, n_slices=256, n_knots=17 on [0,3]. Never on verbs.
    - L_div:    verb non-triviality over the Gumbel-softmax assignment: usage entropy
                + angle/scale spread (away from identity and from each other).

    VerbHead gradient: `a*` is the soft Gumbel-softmax mix Σ_v softmax_v · B_v k during
    training so L_pred flows into VerbHead; hard argmax (straight-through) at eval/export.
    Temperature anneals gumbel_tau: 2.0 -> 0.5 over the first `anneal_frac` of steps.
    """

    def forward(
        self,
        k: torch.Tensor,            # (B, M, dn) standardized nouns
        verb_logits: torch.Tensor,  # (B, M, V)
        zhat: torch.Tensor,         # (B, dn) predicted next-state pool from a*
        z_target: torch.Tensor,     # (B, dn) EMA-encoder(next_text) pool, stop-grad
        gumbel_tau: float,          # current annealed Gumbel temperature
        hard: bool = False,         # eval/export: hard argmax (straight-through)
    ) -> tuple[torch.Tensor, dict]:
        """-> (total_loss, components) where components has
        {"L_pred","L_sigreg","L_div", ...scalars for logging}.
        """
        ...


# =============================================================================
# T4 — Data (owned by src/twm/jepa/data.py). Spec §6.
# =============================================================================

@runtime_checkable
class JEPAChainDatasetProtocol(Protocol):
    """Adjacent cross-state pairs (state_t, state_{t+1}) over GLUCOSE chains.

    Online encoder sees state_t; EMA target encoder sees state_{t+1}. Cross-state
    pairing is mandatory — same-text-to-both degenerates into self-reconstruction.
    Tokenized with the domain BPE (vocab=512), padded to max_text_tokens, stored as
    contiguous CPU tensors (no DataLoader; direct index slicing).
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        """-> {"src_ids","src_pad","tgt_ids","tgt_pad"} for a single (t, t+1) pair.

        src_ids/tgt_ids: (T_text,) long;  src_pad/tgt_pad: (T_text,) padding mask.
        """
        ...

    def iter_text_pairs(self):
        """Yield (state_t, state_{t+1}) token tensors for the operator-fit pass-2 (§7)."""
        ...


# =============================================================================
# T5 — Model (owned by src/twm/jepa/model.py). Spec §9.
# =============================================================================

@runtime_checkable
class JEPAOperatorModelProtocol(Protocol):
    """Persistent-state pet engine: encode once -> k; tick via step_latent; undo exact.

    Owns the online SlotEncoder + Operator + Readout + Predictor, plus a deepcopied
    EMA encoder (online encoder + readout), requires_grad=False, updated manually
    after each optimizer step.
    """

    def forward(
        self,
        src_ids: torch.Tensor,  # (B, T_text)
        src_pad: torch.Tensor,  # (B, T_text)
    ) -> dict:
        """Encode current state -> {"k","verb","a","zhat", "verb_logits", "slots"}.

        k:    (B,M,dn) persistent points (stored in JS gameState)
        verb: (B,M)    discrete verb assignment
        a:    (B,M,dn) a* = B_verb k
        zhat: (B,dn)   readout pool over a* (the JEPA prediction)
        """
        ...

    def step_latent(self, k: torch.Tensor, verb_idx: torch.Tensor) -> torch.Tensor:
        """One tick: a* = B_v k.  (B,M,dn), (B,M) or scalar -> (B,M,dn)."""
        ...

    def undo_latent(self, a: torch.Tensor, verb_idx: torch.Tensor) -> torch.Tensor:
        """Exact undo: k = B_v^{-1} a.  (B,M,dn), (B,M) or scalar -> (B,M,dn)."""
        ...

    def ema_update(self, tau: float = 0.995) -> None:
        """θ_ema ← τ·θ_ema + (1−τ)·θ_online. Called manually after optimizer.step()."""
        ...


# =============================================================================
# T5 — Config (owned by src/twm/jepa/config.py). Spec §10.
# =============================================================================

# Profiles kept SEPARATE from config.PROFILES so build_model_config is untouched (§10).
JEPA_PROFILES: dict[str, dict] = {
    "jepa_nano": {
        "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
        "block": 2, "n_text_layers": 2, "tie_text_layers": True,
        "n_heads": 4, "n_slot_iters": 3,
        "operator_group": "rotation_scale", "n_steps_T": 1,
        "vocab_size": 512, "max_text_tokens": 64,
    },
    "jepa_mini": {
        "d_model": 128, "d_noun": 32, "n_slots": 12, "n_verbs": 16,
        "block": 2, "n_text_layers": 4, "tie_text_layers": False,
        "n_heads": 8, "n_slot_iters": 3,
        "operator_group": "rotation_scale", "n_steps_T": 1,
        "vocab_size": 512, "max_text_tokens": 64,
    },
    # v2 (latent actions + token decoder). Same encoder/operator shape as nano; the
    # nested transition/prior/decoder blocks come from the JSON `model` block (kept
    # OUT of the flat profile per design §10 — the profile stays minimal). Listed
    # here so `apply_profile`/from_dict resolve a known profile name.
    "jepa_nano_v2": {
        "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
        "block": 2, "n_text_layers": 2, "tie_text_layers": True,
        "n_heads": 4, "n_slot_iters": 3,
        "operator_group": "rotation_scale", "n_steps_T": 1,
        "vocab_size": 512, "max_text_tokens": 64,
    },
    # v3 (InfoNCE + multi-step unroll + d128/2L decoder). Same nano encoder/operator
    # shape; the d128/2L decoder, polar flag, mode, loss weights come from the JSON
    # model/loss/data blocks (design §3.1 keeps the profile minimal). The baseline
    # family reuses this profile and only flips operator_group="gated_mlp".
    "jepa_v3": {
        "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
        "block": 2, "n_text_layers": 2, "tie_text_layers": True,
        "n_heads": 4, "n_slot_iters": 3,
        "operator_group": "rotation_scale", "n_steps_T": 1,
        "vocab_size": 512, "max_text_tokens": 64,
    },
}


@dataclass
class SIGRegConfig:
    """Sliced isotropic-Gaussian GoF knobs (spec §3 L_sigreg)."""
    n_slices: int = 256
    n_knots: int = 17
    knot_max: float = 3.0
    standardize: bool = True  # MUST be True — L2-projection to sphere kills the gradient


@dataclass
class VerbConfig:
    """Gumbel-softmax verb-path anneal state (spec §3 VerbHead fix)."""
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    anneal_frac: float = 0.3  # fraction of total steps over which τ_g anneals


@dataclass
class NCEConfig:
    """InfoNCE next-state contrastive config (v3 design §1.3). loss.nce.

    Field spec owned by Task A (losses.py); declared here (the schema is B-owned,
    §6) so config.py can parse loss.nce.
    """
    temperature: float = 0.1  # τ_nce, fixed (SimCLR/MoCo default, NOT the Gumbel τ)


@dataclass
class UnrollConfig:
    """Multi-step unroll per-hop loss weights (v3 design §2.4). loss.unroll.

    Field spec owned by Task A; declared here (B-owned schema, §6) for config.py.
    hop_weights[h] scales the hop-h CE + InfoNCE; default [1.0, 0.5] downweights
    the harder, noisier hop-2 so it acts as composition pressure, not a destabilizer.
    """
    hop_weights: list[float] = field(default_factory=lambda: [1.0, 0.5])


@dataclass
class LossConfig:
    # v1 weights kept on the dataclass so v1 configs still parse; the live loss
    # (losses.JEPALoss / JEPALossV2 alias) ignores w_div/w_scale_reg (design §10).
    w_pred: float = 1.0
    w_sigreg: float = 0.05
    w_div: float = 0.1
    w_scale_reg: float = 0.0  # soft ‖log r‖₂ penalty; off by default (§1.5)
    # v2 additions (design §5 / §10):
    w_token: float = 1.0   # primary CE grounding loss
    w_prior: float = 0.1   # KL(stopgrad q ‖ p) for autonomous rollout
    # v3 additions (design §1.7 / §2.4):
    w_nce: float = 0.0     # InfoNCE next-state weight; v3 sets 0.25 (replaces w_pred)
    # v4 additions (jepa_v4_design §2/§3/§1.3). ALL default to the v3-bitwise neutral
    # value (w_diff=1.0 ⟹ uniform CE; w_margin=0.0 ⟹ no margin; w_mask_prior=0.0 ⟹ no
    # mask-prior KL), so every existing v3 config parses and trains IDENTICALLY.
    w_diff: float = 1.0       # diff-token CE up-weight (§2.2); 1.0 ⟹ uniform v3 CE
    w_margin: float = 0.0     # token-level hard-negative hinge (§3); 0.0 ⟹ off
    margin: float = 0.5       # hinge margin in nats (§3); read only when w_margin>0
    w_mask_prior: float = 0.0  # KL(stopgrad posterior-mask ‖ prior-mask) (§1.3); 0.0 ⟹ off
    # v4.2 masked-diff prediction (trad-JEPA masked reconstruction, focused by causality):
    # mask the CHANGED span in the decoder INPUT, CE only at masked positions (100%
    # discriminative). Default 0.0 ⟹ bitwise-neutral (term skipped, no extra decoder pass,
    # diff masks not even computed at load), so every existing config parses identically.
    w_masked_diff: float = 0.0  # masked-diff CE weight (v4.2); 0.0 ⟹ off
    # v4.1 escalation (jepa_v4_design §C5): pooled-space hard-negative InfoNCE that trains
    # DIRECTLY on the separation-AUC geometry. ALL default to the v4.0-bitwise neutral value
    # (w_pool_nce=0.0 ⟹ term off / not computed), so every existing v4 config parses and
    # trains IDENTICALLY. tau_pool/n_pool_negs are read only when w_pool_nce>0.
    w_pool_nce: float = 0.0    # pooled-space hard-neg InfoNCE weight (§C5); 0.0 ⟹ off
    tau_pool: float = 0.1      # cosine-sim temperature τ_pool for L_pool_nce
    n_pool_negs: int = 16      # in-batch NN hard negatives mined per anchor for L_pool_nce
    pool_nce_stop_grad_pos: bool = False  # detach the online positive pool (MoCo asymmetry)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    verb: VerbConfig = field(default_factory=VerbConfig)
    nce: NCEConfig = field(default_factory=NCEConfig)
    unroll: UnrollConfig = field(default_factory=UnrollConfig)


@dataclass
class TransitionConfig:
    """Posterior q(v | text_t, text_t+1) MLP head (design §2). model.transition."""
    mlp_hidden: int = 128
    use_delta: bool = True


@dataclass
class PriorConfig:
    """Prior p(v | text_t) MLP head (design §3). model.prior."""
    mlp_hidden: int = 64


@dataclass
class DecoderConfig:
    """Token AR decoder over a* memory (design §4). model.decoder."""
    d_dec: int = 64
    n_layers: int = 1
    n_heads: int = 4
    d_ff: int = 128


@dataclass
class GatedMLPConfig:
    """Black-box GatedMLPTransition baseline sizing (v3 design §4.2). model.gated_mlp.

    Field spec owned by Task C (operator.py); declared here (B-owned schema, §6) so
    config.py can parse model.gated_mlp and model.py's factory can read d_e/d_h.
    """
    d_e: int = 4  # verb embedding width
    d_h: int = 8  # MLP hidden width


@dataclass
class TargetedConfig:
    """Targeted latent-action mask head sizing (jepa_v4_design §1). model.targeted.

    Only read when ModelHParams.use_targeted_actions=True; ignored otherwise so
    every existing v3 config builds an identical (mask-off) model.
    """
    mask_hidden: int = 64  # hidden width of the per-slot posterior/prior mask MLPs


@dataclass
class DataConfig:
    path: str = "data/glucose/chain_general_train.jsonl"
    tokenizer: str = "data/glucose/jepa_bpe_512.json"
    vocab_size: int = 512
    max_text_tokens: int = 64
    pairing: str = "adjacent"
    # v3 unroll (design §2.1): "pairs" = exactly v2 behavior; "triples" emits one
    # (s0,s1,s2) example per chain for two-hop unroll. Default keeps v2.1 bitwise.
    mode: str = "pairs"
    max_chains: int | None = None  # smoke/debug cap on #chains (None = full dataset)
    append_eos: bool = False  # v2: append <eos>=4 to tokenized states (default = v1 behavior)


@dataclass
class ModelHParams:
    d_model: int = 64
    d_noun: int = 32
    n_slots: int = 8
    n_verbs: int = 8
    block: int = 2
    n_text_layers: int = 2
    tie_text_layers: bool = True
    n_heads: int = 4
    n_slot_iters: int = 3
    operator_group: str = "rotation_scale"  # "rotation_scale" | "rotation" | "son_cayley"
    n_steps_T: int = 1
    # v2.1 polar conditioning (design §10). All default-False ⟹ every existing v2.0
    # config still parses and builds an IDENTICAL model (the §11 behavior-preservation
    # gate depends on this).
    use_polar_conditioning: bool = False  # master switch for the H phase-offset path
    use_kind_head: bool = False           # §7 diagnostic kind readout; default off
    kind_codebook_size: int = 16          # K, only read when use_kind_head=True
    # entity-campaign norm budget (entity §1.5). DEFAULT FALSE ⟹ the operator returns a
    # bare tensor, _apply_action returns a bare tensor, no scale_readout_proj is built, no
    # s_acc is threaded — BITWISE v3 GLUCOSE behavior. Entity (structured) configs flip it
    # true; the model only constructs scale_readout_proj when it is on (entity §1.0/§1.5).
    use_norm_budget: bool = False
    # v4 targeted latent actions (jepa_v4_design §1). DEFAULT FALSE ⟹ the posterior/prior
    # build no mask heads, _apply_action ignores the mask, and the model is BITWISE v3.
    # Entity (structured) configs flip it true; model.py reads model.targeted.mask_hidden.
    use_targeted_actions: bool = False
    # v2 nested heads (design §10). default_factory so v1 configs without these blocks
    # still construct a valid ModelHParams.
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    # v4 targeted-action mask head sizing; only read when use_targeted_actions=True.
    targeted: TargetedConfig = field(default_factory=TargetedConfig)
    # v3 black-box baseline sizing (design §4.2). Only read when
    # operator_group=="gated_mlp"; ignored otherwise so existing configs are unchanged.
    gated_mlp: GatedMLPConfig = field(default_factory=GatedMLPConfig)


@dataclass
class EMAConfig:
    tau: float = 0.995
    schedule: str = "fixed"  # no cosine schedule on short GLUCOSE runs (§4)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0
    warmup_steps: int = 200


@dataclass
class EntityWorldEvalConfig:
    """Entity-world diagnostics block (campaign §3.0). eval.entity_world.

    Config-gated (default DISABLED) so a plain GLUCOSE config runs the v3 diagnostics
    exactly as today. Entity configs flip `enabled=true` and point at the labeled
    splits. Owned by Task B (the eval suite).
    """
    enabled: bool = False
    labeled_dir: str = "data/entity_world"  # holds {split}_labeled.jsonl + manifest.json
    splits: list[str] = field(
        default_factory=lambda: ["test_iid", "test_ood_near", "test_ood_far"]
    )
    subsample: int = 512                      # per-split cap for the ladder (§3b)
    n_rollout_chains: int = 128               # §3c rollout fidelity
    rollout_max_depth: int = 4                # depth 1..4 greedy decode vs gold
    action_recovery_split: str = "test_iid"   # §3a NMI split (uses its _labeled twin)


@dataclass
class EvalConfig:
    every_epochs: int = 5
    n_examples: int = 512
    out_dir: str = "results/jepa_nano"
    # v2 generated-text diagnostics (design §8 / §10):
    n_text_samples: int = 16
    temperatures: list[float] = field(default_factory=lambda: [0.7, 1.0])
    # entity-world campaign diagnostics (campaign §3); absent/disabled ⟹ no entity metrics.
    entity_world: EntityWorldEvalConfig = field(default_factory=EntityWorldEvalConfig)


@dataclass
class OperatorFitPass2Config:
    enabled: bool = True
    after_epoch: int = 5


@dataclass
class JEPAConfig:
    """Top-level JEPA experiment config (spec §10).

    Mirrors training_config.py conventions: `from_json`/`from_dict`. Profiles come
    from JEPA_PROFILES (kept separate from config.PROFILES).
    """
    profile: str = "jepa_nano"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelHParams = field(default_factory=ModelHParams)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    operator_fit_pass2: OperatorFitPass2Config = field(default_factory=OperatorFitPass2Config)

    @classmethod
    def from_json(cls, path) -> "JEPAConfig":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict) -> "JEPAConfig":
        raise NotImplementedError  # concrete parsing lives in config.py (T5)


# =============================================================================
# Lazy / deferred re-exports of concrete implementations.
# Guarded so `import twm.jepa` succeeds TODAY with no concrete modules present.
# Once a module lands, `from twm.jepa import RotationScaleOperator, ...` resolves
# to the concrete class; until then the contract base classes above stand in.
# =============================================================================

if TYPE_CHECKING:  # static-analysis-only imports; never executed at runtime
    from .operator import (
        RotationScaleOperator,
        RotationOperator,
        SOnCayleyOperator,
    )
    from .slot_encoder import SlotEncoder, NounHead, VerbHead
    from .losses import JEPALoss, JEPALossV2
    from .data import JEPAChainDataset
    from .model import JEPAOperatorModelV2
    from .transition import TransitionEncoder, PriorHead
    from .decoder import TokenDecoder
    from .conditioning import PolarConditioner, KindHead, block_modulus
    # v1 baseline (demoted to legacy/).
    from .legacy.model_v1 import JEPAOperatorModel


def __getattr__(name: str):
    """PEP 562 lazy attribute access for concrete classes.

    Resolves a re-exported concrete name (e.g. RotationScaleOperator) from its owning
    module the first time it is accessed. If the module is not yet written, raise a
    clear ImportError naming the still-pending owner file rather than a bare
    AttributeError — so `import twm.jepa` stays clean but a premature concrete access
    is diagnosable.
    """
    _MODULE_OF = {
        "RotationScaleOperator": ".operator",
        "RotationOperator": ".operator",
        "SOnCayleyOperator": ".operator",
        "SlotEncoder": ".slot_encoder",
        "NounHead": ".slot_encoder",
        "VerbHead": ".slot_encoder",
        "JEPALoss": ".losses",
        "JEPAChainDataset": ".data",
        # v1 baseline lives under legacy/ (demoted, not deleted).
        "JEPAOperatorModel": ".legacy.model_v1",
        # v2 concrete classes (the live path):
        "TransitionEncoder": ".transition",
        "PriorHead": ".transition",
        "TokenDecoder": ".decoder",
        "JEPALossV2": ".losses",
        "JEPAOperatorModelV2": ".model",
        # v2.1 polar conditioning (design §3 / §7):
        "PolarConditioner": ".conditioning",
        "KindHead": ".conditioning",
        "block_modulus": ".conditioning",
    }
    mod = _MODULE_OF.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        module = importlib.import_module(mod, __name__)
    except ImportError as e:
        raise ImportError(
            f"{name!r} is a frozen contract whose implementation lives in "
            f"twm.jepa{mod} — not written yet ({e}). The contract base classes "
            f"(Operator, *Protocol, JEPAConfig) are importable now."
        ) from e
    return getattr(module, name)


__all__ = [
    # Contract base classes / protocols (defined here, always available)
    "Operator",
    "SlotEncoderProtocol",
    "JEPALossProtocol",
    "JEPAChainDatasetProtocol",
    "JEPAOperatorModelProtocol",
    # Config (defined here)
    "JEPAConfig",
    "JEPA_PROFILES",
    "ModelHParams",
    "DataConfig",
    "LossConfig",
    "SIGRegConfig",
    "VerbConfig",
    "TransitionConfig",
    "PriorConfig",
    "DecoderConfig",
    "NCEConfig",
    "UnrollConfig",
    "GatedMLPConfig",
    "TargetedConfig",
    "EMAConfig",
    "OptimConfig",
    "EvalConfig",
    "EntityWorldEvalConfig",
    "OperatorFitPass2Config",
    # Concrete classes (lazily re-exported once their modules land)
    "RotationScaleOperator",
    "RotationOperator",
    "SOnCayleyOperator",
    "SlotEncoder",
    "NounHead",
    "VerbHead",
    "JEPALoss",
    "JEPAChainDataset",
    "JEPAOperatorModel",
    # v2 concrete classes (lazily re-exported once their modules land)
    "TransitionEncoder",
    "PriorHead",
    "TokenDecoder",
    "JEPALossV2",
    "JEPAOperatorModelV2",
    # v2.1 polar conditioning
    "PolarConditioner",
    "KindHead",
    "block_modulus",
]
