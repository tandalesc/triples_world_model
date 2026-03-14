"""Experiment configuration for config-driven training.

Defines the full experiment as a dataclass hierarchy:
  TrainingConfig → StageConfig → PhaseConfig

JSON serializable for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PhaseConfig:
    """Single training phase within a stage (e.g. high-noise warmup, full range)."""
    t_min: float = 0.0
    t_max: float = 1.0
    bias_power: float = 1.0
    epochs: int = 200
    patience: int = 50  # 0 = no early stopping
    lr: float | None = None  # None = inherit from stage
    metric: str = "tok_acc"  # metric for early stopping: "tok_acc" or "exact"


@dataclass
class StageConfig:
    """Training stage (io, dynamics, finetune)."""
    name: str  # "io", "dynamics", "finetune"
    dataset: str  # "identity" or "qa"
    phases: list[PhaseConfig] = field(default_factory=list)
    lr: float = 3e-4
    weight_decay: float = 0.01
    freeze: list[str] = field(default_factory=list)  # ["compressor", "expander"]
    unfreeze: list[str] | None = None  # None = auto (unfreeze length_head when expander frozen); [] = no overrides
    pretrained: str | None = None  # checkpoint path, None = auto-detect
    max_examples: int | None = None  # None = inherit from top-level


@dataclass
class TrainingConfig:
    """Complete experiment definition."""

    # Model
    model_type: str = "dynamics"  # "io" or "dynamics"
    profile: str = "base"
    d_model: int | None = None
    max_triples: int | None = None
    text_compressor_layers: int = 4
    text_expander_layers: int = 3
    dynamics_layers: int | None = None
    max_text_tokens: int = 64
    dropout: float = 0.1
    alpha_min: float = 0.01
    vae: bool = False  # enable VAE bottleneck with role-conditioned priors

    # Data
    data_dir: str = ""
    tokenizer_path: str = ""
    max_examples: int = 0

    # Training
    batch_size: int = 64
    denoise_steps: int = 10
    aux_ce_weight: float = 0.1
    length_weight: float = 0.1
    bottleneck_weight: float = 0.0  # direct bottleneck MSE (dynamics only)
    bn_role_weights: list[float] | None = None  # [entity_w, attr_w, value_w] for decomposed bn loss
    detach_dynamics_expander: bool = False  # cut token gradients to dynamics core
    role_prior_weight: float = 0.0  # role-conditioned centroid regularization (legacy, use vae instead)
    kl_weight: float = 0.0  # VAE KL weight (β). 0 = no KL. Annealed from 0 to this value.
    kl_anneal_epochs: int = 0  # linear anneal from 0 to kl_weight over this many epochs. 0 = constant.
    log_every: int = 10
    diagnostic_every: int = 50
    snapshot_every: int = 0  # 0 = disabled; >0 = save latent PCA frame every N epochs

    # Stages
    stages: list[StageConfig] = field(default_factory=list)

    # Output
    out_dir: str = ""
    device: str | None = None

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> TrainingConfig:
        stages_raw = data.pop("stages", [])
        stages = []
        for s in stages_raw:
            phases_raw = s.pop("phases", [])
            phases = [PhaseConfig(**p) for p in phases_raw]
            stages.append(StageConfig(phases=phases, **s))
        return cls(stages=stages, **data)

    def build_model_config(self):
        """Build a ModelConfig from this training config."""
        from .config import ModelConfig, PROFILES

        base = ModelConfig.from_profile(self.profile)
        d_model = self.d_model if self.d_model is not None else base.d_model
        max_triples = self.max_triples if self.max_triples is not None else base.max_triples
        if d_model != base.d_model or max_triples != base.max_triples:
            return ModelConfig(
                d_model=d_model, n_heads=base.n_heads,
                n_layers=base.n_layers, d_ff=d_model * 4,
                max_triples=max_triples,
            )
        return base
