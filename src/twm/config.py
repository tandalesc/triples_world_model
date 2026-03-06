"""Model configuration profiles for Triple World Model."""

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


PROFILES: dict[str, dict] = {
    "base": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "d_ff": 1024,
        "max_triples": 8,
        "dropout": 0.1,
    },
    "micro": {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 32,
        "max_triples": 8,
        "dropout": 0.05,
    },
    "atomic": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "d_ff": 1024,
        "max_triples": 12,
        "dropout": 0.1,
    },
}


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_triples: int = 8
    dropout: float = 0.1
    pretrained_embed_dim: int | None = None
    profile: str = "base"

    # Separate embedding vocab sizes (0 = use shared vocab_size)
    n_entities: int = 0
    n_attrs: int = 0
    n_values: int = 0

    @property
    def max_positions(self) -> int:
        return self.max_triples * 3

    @property
    def use_split_embeddings(self) -> bool:
        return self.n_entities > 0 and self.n_attrs > 0 and self.n_values > 0

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = json.load(f)
        # Handle configs saved before split embeddings existed
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_profile(cls, name: str, **overrides) -> "ModelConfig":
        if name not in PROFILES:
            raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILES.keys())}")
        params = {**PROFILES[name], "profile": name, **overrides}
        return cls(**params)
