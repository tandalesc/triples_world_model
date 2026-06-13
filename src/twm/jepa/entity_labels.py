"""Oracle entity-world labels for the v5 discriminative objectives (Step-1a).

This module attaches two PER-TRANSITION supervision signals to an entity-world chain,
computed ONCE at dataset load from the `*_labeled.jsonl` twin:

  - oracle_verb_id (int): the ground-truth verb of the transition, parsed from the
    labeled record's "actions" ("<verb>@<idx>") and mapped through a STABLE verb->id
    table (the generator's ACTIONS list — 11 named verbs). -1 when no label exists.
    Feeds L_verb_anchor (sparse oracle-verb supervision of the bottleneck).

  - canonical_next_state_id (int): a stable integer id of the oracle CANONICAL
    next-state STRING, computed by replaying the chain from "initial_states" +
    "actions" through the oracle (apply_action) and canonicalizing the resulting
    multi-entity state to a sorted string. Same canonical string ⟹ same id (so
    paraphrase renderings of the same underlying state collapse to ONE id — the
    invariance L_sep exploits). Feeds L_sep (sibling-contrastive separation).

Why a separate module: the oracle dynamics live in scripts/generate_entity_world.py
(not importable as a package). We load it by file path the same way diagnostics /
run_geometry_probes do, and KEEP the import lazy + guarded so a GLUCOSE config (no
oracle, no labeled twin) pays nothing and never imports it.

Cardinality note (design): the model's verb codebook has V=8 codes; the oracle has 11
named verbs. The aux head predicts the 11-way oracle verb — it does NOT force
codebook==oracle (it is a TRAINING-ONLY bias on the bottleneck, dropped at inference).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# Stable verb -> id map. Mirrors generate_entity_world.ACTIONS order (the canonical,
# exhaustive verb vocabulary). Frozen here so the id mapping is independent of import
# success — the generator's ACTIONS is asserted to match at load when available.
ORACLE_VERBS = [
    "feed", "play", "wash", "rest", "water",
    "switch on", "switch off", "fill", "open", "close", "wait",
]
N_ORACLE_VERBS = len(ORACLE_VERBS)
_VERB2ID = {v: i for i, v in enumerate(ORACLE_VERBS)}


def verb_to_id(verb: str) -> int:
    """Map a verb string to its stable oracle id, or -1 if unknown (ignore_index)."""
    return _VERB2ID.get(verb, -1)


def parse_action_label(label: str) -> tuple[str, int]:
    """Parse a '<verb>@<idx>' action label -> (verb, actor_idx). ('', -1) on malformed."""
    try:
        verb, idx = label.rsplit("@", 1)
        return verb, int(idx)
    except (ValueError, IndexError, AttributeError):
        return "", -1


_GEN_MOD = None


def _load_oracle_module():
    """Lazily import scripts/generate_entity_world.py by file path (cached)."""
    global _GEN_MOD
    if _GEN_MOD is not None:
        return _GEN_MOD
    repo = Path(__file__).resolve().parents[3]  # .../triples_world_model
    path = repo / "scripts" / "generate_entity_world.py"
    spec = importlib.util.spec_from_file_location("generate_entity_world", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _GEN_MOD = mod
    return mod


def _canonical_state_str(entities) -> str:
    """Canonicalize a multi-entity oracle state to a stable string.

    `entities` is a list of (type_name, {attr: value}). We sort each entity's attr
    items and sort the entities by (type_name, attrs) so the SAME underlying state
    always renders to the SAME string regardless of entity order or dict ordering —
    paraphrase surface forms collapse to one canonical id (the L_sep invariance).
    """
    parts = []
    for type_name, state in entities:
        items = ";".join(f"{a}={state[a]}" for a in sorted(state))
        parts.append(f"{type_name}|{items}")
    return " || ".join(sorted(parts))


def replay_canonical_states(types, initial_states, actions):
    """Replay a chain through the oracle -> list of canonical next-state strings.

    Returns one canonical string PER STATE in the chain (len == len(actions)+1):
    index 0 is the initial state, index i+1 is the state AFTER actions[i]. The caller
    aligns transition i (state_i -> state_{i+1}) with the canonical string at i+1.

    Returns None if the oracle is unavailable or the labeled fields are missing
    (the caller then leaves canonical_next_state_id = -1).
    """
    if not types or not initial_states:
        return None
    try:
        gen = _load_oracle_module()
    except Exception:
        return None
    apply_action = gen.apply_action
    entities = [(tn, dict(st)) for tn, st in zip(types, initial_states)]
    canon = [_canonical_state_str(entities)]
    for label in actions:
        verb, actor_idx = parse_action_label(label)
        if actor_idx < 0 or actor_idx >= len(entities):
            # Malformed actor — record current canonical (no transition) and continue.
            canon.append(_canonical_state_str(entities))
            continue
        type_name = entities[actor_idx][0]
        new_state = apply_action(type_name, entities[actor_idx][1], verb)
        entities[actor_idx] = (type_name, new_state)
        canon.append(_canonical_state_str(entities))
    return canon


class CanonicalStateRegistry:
    """Global (per-dataset) canonical-state-string -> int id registry.

    Stable within a dataset build: the first time a canonical string is seen it gets the
    next integer id, so positives (same canonical next-state) share an id and the L_sep
    SupCon labels are consistent across the whole dataset. Ids are NOT meaningful across
    datasets (they are only used for in-batch equality), so a simple monotone counter is
    sufficient and reproducible (insertion order is deterministic for a fixed file).
    """

    def __init__(self) -> None:
        self._map: dict[str, int] = {}

    def get(self, canon: str | None) -> int:
        if canon is None:
            return -1
        i = self._map.get(canon)
        if i is None:
            i = len(self._map)
            self._map[canon] = i
        return i

    def __len__(self) -> int:
        return len(self._map)


def load_labeled_records(path: str | Path) -> list[dict] | None:
    """Read a *_labeled.jsonl twin -> list of records, or None if it does not exist.

    Each record carries "chain", "actions", "types", "initial_states" (see
    generate_entity_world.build_split). Returns None (not an error) when the labeled
    twin is absent, so a GLUCOSE/unlabeled config silently skips the labeled path.
    """
    p = Path(path)
    if not p.exists():
        return None
    records = []
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def labeled_path_for(path: str | Path) -> Path:
    """Return the '<stem>_labeled.jsonl' twin path for a chain jsonl path.

    e.g. data/entity_world/train.jsonl -> data/entity_world/train_labeled.jsonl.
    """
    p = Path(path)
    return p.with_name(p.stem + "_labeled" + p.suffix)
