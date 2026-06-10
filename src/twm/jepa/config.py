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
    TransitionConfig,
    PriorConfig,
    DecoderConfig,
    NCEConfig,
    UnrollConfig,
    GatedMLPConfig,
    LossConfig,
    DataConfig,
    ModelHParams,
    EMAConfig,
    OptimConfig,
    EvalConfig,
    EntityWorldEvalConfig,
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
    # v3 nested blocks (design §1.7/§2.4): loss.nce, loss.unroll. Pop before the flat
    # overlay so the raw dicts never reach LossConfig(**...) as scalar fields.
    nce = NCEConfig(**_only_known(NCEConfig, data.pop("nce", {})))
    unroll = UnrollConfig(**_only_known(UnrollConfig, data.pop("unroll", {})))
    return LossConfig(
        sigreg=sigreg, verb=verb, nce=nce, unroll=unroll, **_only_known(LossConfig, data)
    )


def _build_eval(data: dict) -> EvalConfig:
    """Parse the eval block, including the nested eval.entity_world block (campaign §3.0).

    Mirrors `_build_loss`: pop the nested dict before the flat overlay so the raw dict
    never reaches EvalConfig(**...) as a scalar field. Absent ⟹ a disabled default
    EntityWorldEvalConfig (back-compat: a GLUCOSE config has no entity metrics)."""
    data = dict(data)
    entity_world = EntityWorldEvalConfig(
        **_only_known(EntityWorldEvalConfig, data.pop("entity_world", {}))
    )
    return EvalConfig(entity_world=entity_world, **_only_known(EvalConfig, data))


def _build_model(profile: str, model_json: dict) -> ModelHParams:
    """Overlay profile defaults with explicit JSON, parsing the nested v2 blocks
    (model.{transition,prior,decoder}) into their dataclasses (design §10).

    The nested blocks are popped before the flat field overlay so the raw dicts
    never reach ModelHParams(**...) as scalar fields.
    """
    raw = dict(JEPA_PROFILES.get(profile, {}))
    raw.update(model_json or {})
    transition = TransitionConfig(**_only_known(TransitionConfig, raw.pop("transition", {})))
    prior = PriorConfig(**_only_known(PriorConfig, raw.pop("prior", {})))
    decoder = DecoderConfig(**_only_known(DecoderConfig, raw.pop("decoder", {})))
    # v3 black-box baseline sizing (design §4.2): model.gated_mlp, read only when
    # operator_group=="gated_mlp" by model.py's factory; parsed always so it round-trips.
    gated_mlp = GatedMLPConfig(**_only_known(GatedMLPConfig, raw.pop("gated_mlp", {})))
    return ModelHParams(
        transition=transition,
        prior=prior,
        decoder=decoder,
        gated_mlp=gated_mlp,
        **_only_known(ModelHParams, raw),
    )


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

        # model: start from profile defaults, overlay explicit JSON keys, parse
        # the nested v2 transition/prior/decoder blocks.
        model = _build_model(profile, data.get("model", {}))

        return cls(
            profile=profile,
            seed=data.get("seed", 0),
            data=DataConfig(**_only_known(DataConfig, data.get("data", {}))),
            model=model,
            loss=_build_loss(data.get("loss", {})),
            ema=EMAConfig(**_only_known(EMAConfig, data.get("ema", {}))),
            optim=OptimConfig(**_only_known(OptimConfig, data.get("optim", {}))),
            eval=_build_eval(data.get("eval", {})),
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
    "TransitionConfig",
    "PriorConfig",
    "DecoderConfig",
    "NCEConfig",
    "UnrollConfig",
    "GatedMLPConfig",
    "LossConfig",
    "DataConfig",
    "ModelHParams",
    "EMAConfig",
    "OptimConfig",
    "EvalConfig",
    "EntityWorldEvalConfig",
    "OperatorFitPass2Config",
]
