"""Decision-distillation data: triples in, teacher distributions out.

Rows come from Hearthvale's ``scripts/distill-export.js``::

    {"key", "kind", "triples": [[scope, attr, value, num|null], ...],
     "options": [...], "target": [...], "n", "seeds", "aug", "model"}

``DecisionVocab`` maps the closed vocabularies (scopes, attributes, values,
options) to ids; ``encode_rows`` packs rows into padded tensors once, so a
training step is a gather on the device.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

PAD, UNK = "<pad>", "<unk>"
# Scopes whose triples describe the shared snapshot (cached per state at play
# time); the rest ('kind', 'q') describe the question.
STATE_SCOPES = ("town", "district", "proposal", "body")


def load_rows(paths: list[str | Path], kinds: list[str] | None = None) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if kinds and row["kind"] not in kinds:
                    continue
                rows.append(row)
    return rows


def _index(tokens) -> dict[str, int]:
    out = {PAD: 0, UNK: 1}
    for t in tokens:
        out.setdefault(t, len(out))
    return out


@dataclass
class DecisionVocab:
    scopes: dict[str, int]
    attrs: dict[str, int]
    values: dict[str, int]
    # Options appear twice: as "kind:option" (that kind's own) and as the bare
    # key, shared across kinds ('leave' in react_to_policy and elsewhere).
    kind_options: dict[str, int]
    options: dict[str, int]
    state_scope_ids: list[int] = field(default_factory=list)

    @classmethod
    def build(cls, rows: list[dict], min_value_count: int = 3) -> "DecisionVocab":
        scopes, attrs, values = Counter(), Counter(), Counter()
        kind_options, options = set(), set()
        for row in rows:
            for scope, attr, value, _ in row["triples"]:
                scopes[scope] += 1
                attrs[attr] += 1
                values[value] += 1
            for o in row["options"]:
                kind_options.add(f"{row['kind']}:{o}")
                options.add(o)
        vocab = cls(
            scopes=_index(sorted(scopes)),
            attrs=_index(sorted(attrs)),
            values=_index(sorted(v for v, c in values.items() if c >= min_value_count)),
            kind_options=_index(sorted(kind_options)),
            options=_index(sorted(options)),
        )
        vocab.state_scope_ids = [vocab.scopes[s] for s in STATE_SCOPES if s in vocab.scopes]
        return vocab

    @classmethod
    def extend(cls, old: "DecisionVocab", rows: list[dict], min_value_count: int = 3) -> "DecisionVocab":
        """The old vocabulary with any new tokens from `rows` appended: old ids keep their meaning."""
        new = cls.build(rows, min_value_count)
        def grow(a: dict, b: dict) -> dict:
            out = dict(a)
            for t in b:
                out.setdefault(t, len(out))
            return out
        vocab = cls(grow(old.scopes, new.scopes), grow(old.attrs, new.attrs), grow(old.values, new.values),
                    grow(old.kind_options, new.kind_options), grow(old.options, new.options))
        vocab.state_scope_ids = [vocab.scopes[s] for s in STATE_SCOPES if s in vocab.scopes]
        return vocab

    def to_json(self) -> dict:
        return {"scopes": self.scopes, "attrs": self.attrs, "values": self.values,
                "kind_options": self.kind_options, "options": self.options, "state_scopes": list(STATE_SCOPES)}

    @classmethod
    def from_json(cls, data: dict) -> "DecisionVocab":
        vocab = cls(data["scopes"], data["attrs"], data["values"], data["kind_options"], data["options"])
        vocab.state_scope_ids = [vocab.scopes[s] for s in STATE_SCOPES if s in vocab.scopes]
        return vocab

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path: str | Path) -> "DecisionVocab":
        return cls.from_json(json.loads(Path(path).read_text()))


@dataclass
class Encoded:
    """Padded tensors for N rows (T triples, K options at most)."""

    scope: torch.Tensor      # (N, T) long
    attr: torch.Tensor       # (N, T) long
    value: torch.Tensor      # (N, T) long
    num: torch.Tensor        # (N, T) float, 0 where absent
    has_num: torch.Tensor    # (N, T) float 0/1
    tok_mask: torch.Tensor   # (N, T) bool, True = real triple
    kind_opt: torch.Tensor   # (N, K) long
    opt: torch.Tensor        # (N, K) long
    opt_mask: torch.Tensor   # (N, K) bool, True = offered
    target: torch.Tensor     # (N, K) float, teacher distribution (0 on padding)
    weight: torch.Tensor     # (N,) float, sqrt(times asked), mean 1
    kind: list[str]

    def __len__(self) -> int:
        return self.scope.shape[0]

    def to(self, device) -> "Encoded":
        return Encoded(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()})

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)}


def encode_rows(rows: list[dict], vocab: DecisionVocab) -> Encoded:
    n = len(rows)
    T = max(len(r["triples"]) for r in rows)
    K = max(len(r["options"]) for r in rows)
    scope = np.zeros((n, T), dtype=np.int64)
    attr = np.zeros((n, T), dtype=np.int64)
    value = np.zeros((n, T), dtype=np.int64)
    num = np.zeros((n, T), dtype=np.float32)
    has_num = np.zeros((n, T), dtype=np.float32)
    kind_opt = np.zeros((n, K), dtype=np.int64)
    opt = np.zeros((n, K), dtype=np.int64)
    target = np.zeros((n, K), dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    S, A, V, KO, O = vocab.scopes, vocab.attrs, vocab.values, vocab.kind_options, vocab.options
    for i, r in enumerate(rows):
        t = r["triples"]
        m = len(t)
        scope[i, :m] = [S.get(x[0], 1) for x in t]
        attr[i, :m] = [A.get(x[1], 1) for x in t]
        value[i, :m] = [V.get(x[2], 1) for x in t]
        for j, x in enumerate(t):
            if x[3] is not None:
                num[i, j] = x[3]
                has_num[i, j] = 1.0
        k = len(r["options"])
        kind_opt[i, :k] = [KO.get(f"{r['kind']}:{o}", 1) for o in r["options"]]
        opt[i, :k] = [O.get(o, 1) for o in r["options"]]
        p = np.asarray(r["target"], dtype=np.float32)
        target[i, :k] = p / max(float(p.sum()), 1e-9)
        weight[i] = math.sqrt(max(1, r.get("n", 1)))
    weight /= weight.mean()
    t = torch.from_numpy
    scope_t = t(scope)
    kind_opt_t = t(kind_opt)
    return Encoded(scope_t, t(attr), t(value), t(num), t(has_num), scope_t > 0, kind_opt_t, t(opt), kind_opt_t > 0,
                   t(target), t(weight), [r["kind"] for r in rows])


def split_by_seed(rows: list[dict], test_fraction: float = 0.2, salt: int = 0) -> tuple[list[dict], list[dict]]:
    """Hold out whole towns: a row is test only if every seed that asked it is held out.

    Rows also asked in a training town stay in train, so nothing leaks.
    """
    seeds = sorted({s for r in rows for s in r.get("seeds", [0])})
    held = {s for s in seeds if (((s * 2654435761) ^ salt) & 0xFFFFFFFF) / 2**32 < test_fraction}
    # At least one town each side, when there are two.
    if len(seeds) > 1 and not held:
        held = {seeds[-1]}
    if len(seeds) > 1 and len(held) == len(seeds):
        held.discard(seeds[0])
    train, test = [], []
    for r in rows:
        (test if all(s in held for s in r.get("seeds", [0])) else train).append(r)
    return train, test
