"""JEPA experiment config — concrete from_dict parsing (spec §10).

Owns the runtime config dataclasses' construction logic. The dataclass *shapes*
and JEPA_PROFILES live in the frozen contract stub (twm/jepa/__init__.py); this
module imports them and supplies the concrete `from_dict` that the stub defers.

We re-export the dataclasses so callers can `from twm.jepa.config import JEPAConfig`
and get a JEPAConfig whose `from_dict` is fully implemented (the stub's
JEPAConfig.from_dict raises NotImplementedError on purpose — see §10).

Conventions mirror training_config.py: nested dataclasses, `from_json` reads a
file then defers to `from_dict`, unknown keys are tolerated only where the schema
explicitly allows (here we are strict: the §10 schema is fixed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict

# Reuse the frozen dataclass contracts and profile table from the stub so there is
# exactly one definition of each. config.py only adds the parsing behaviour.
from twm.jepa import (
    JEPA_PROFILES,
    SIGRegConfig,
    VerbConfig,
    LossConfig,
    DataConfig,
    ModelHParams,
    EMAConfig,
    OptimConfig,
    EvalConfig,
    OperatorFitPass2Config,
)


def _only_known(cls, data: dict) -> dict:
    """Drop keys not present on the dataclass `cls` (strictness w/ forward-compat)."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def _build_loss(data: dict) -> LossConfig:
    data = dict(data)
    sigreg = SIGRegConfig(**_only_known(SIGRegConfig, data.pop("sigreg", {})))
    verb = VerbConfig(**_only_known(VerbConfig, data.pop("verb", {})))
    return LossConfig(sigreg=sigreg, verb=verb, **_only_known(LossConfig, data))


@dataclass
class JEPAConfig:
    """Top-level JEPA experiment config with concrete parsing (spec §10).

    Shadows the stub's JEPAConfig (whose from_dict is a deliberate NotImplemented)
    with a fully-parsing version. The model trainer imports this one.

    `apply_profile()` overlays JEPA_PROFILES[profile] onto `model` for any field
    the JSON did not explicitly set, so configs can stay terse (profile name +
    overrides) exactly like the repo's profile convention.
    """
    profile: str = "jepa_nano"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelHParams = field(default_factory=ModelHParams)
    loss: LossConfig = field(default_factory=LossConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    operator_fit_pass2: OperatorFitPass2Config = field(
        default_factory=OperatorFitPass2Config
    )

    @classmethod
    def from_json(cls, path) -> "JEPAConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict) -> "JEPAConfig":
        data = dict(data)
        profile = data.get("profile", "jepa_nano")

        # model: start from profile defaults, overlay explicit JSON keys.
        model_raw = dict(JEPA_PROFILES.get(profile, {}))
        model_raw.update(data.get("model", {}))
        model = ModelHParams(**_only_known(ModelHParams, model_raw))

        return cls(
            profile=profile,
            seed=data.get("seed", 0),
            data=DataConfig(**_only_known(DataConfig, data.get("data", {}))),
            model=model,
            loss=_build_loss(data.get("loss", {})),
            ema=EMAConfig(**_only_known(EMAConfig, data.get("ema", {}))),
            optim=OptimConfig(**_only_known(OptimConfig, data.get("optim", {}))),
            eval=EvalConfig(**_only_known(EvalConfig, data.get("eval", {}))),
            operator_fit_pass2=OperatorFitPass2Config(
                **_only_known(
                    OperatorFitPass2Config, data.get("operator_fit_pass2", {})
                )
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


__all__ = [
    "JEPAConfig",
    "JEPA_PROFILES",
    "SIGRegConfig",
    "VerbConfig",
    "LossConfig",
    "DataConfig",
    "ModelHParams",
    "EMAConfig",
    "OptimConfig",
    "EvalConfig",
    "OperatorFitPass2Config",
]
