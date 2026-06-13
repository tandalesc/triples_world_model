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
    states = _replay_entity_states(types, initial_states, actions)
    if states is None:
        return None
    return [_canonical_state_str(entities) for entities in states]


def _replay_entity_states(types, initial_states, actions):
    """Replay a chain through the oracle -> list of STRUCTURED entity states.

    Like `replay_canonical_states` but returns, per state in the chain, the structured
    entity list `[(type_name, {attr: value}), ...]` (len == len(actions)+1) instead of a
    canonical string. Used by the changed-attribute delta label, which needs the per-attr
    values (not just the rendered string) to diff a transition. Returns None when the
    oracle is unavailable or the labeled fields are missing.
    """
    if not types or not initial_states:
        return None
    try:
        gen = _load_oracle_module()
    except Exception:
        return None
    apply_action = gen.apply_action
    entities = [(tn, dict(st)) for tn, st in zip(types, initial_states)]
    states = [[(tn, dict(st)) for tn, st in entities]]
    for label in actions:
        verb, actor_idx = parse_action_label(label)
        if actor_idx < 0 or actor_idx >= len(entities):
            # Malformed actor — record current state (no transition) and continue.
            states.append([(tn, dict(st)) for tn, st in entities])
            continue
        type_name = entities[actor_idx][0]
        new_state = apply_action(type_name, entities[actor_idx][1], verb)
        entities[actor_idx] = (type_name, new_state)
        states.append([(tn, dict(st)) for tn, st in entities])
    return states


_ATTR_LADDERS = None


def _attr_ladders() -> dict | None:
    """Return the oracle ATTRIBUTE_POOL ordinal ladders (cached), or None if unavailable.

    Each ladder is an ordered list of values, index 0 = best, last = worst. Used to assign
    a DIRECTION ("up" = toward index 0 / better, "down" = toward the end / worse) to a
    changed attribute by comparing the before/after ordinal positions.
    """
    global _ATTR_LADDERS
    if _ATTR_LADDERS is not None:
        return _ATTR_LADDERS
    try:
        gen = _load_oracle_module()
    except Exception:
        return None
    _ATTR_LADDERS = dict(gen.ATTRIBUTE_POOL)
    return _ATTR_LADDERS


def _attr_direction(attr: str, old_val, new_val) -> str:
    """Direction of an attribute change via its ordinal ladder.

    "up" = moved toward index 0 (better), "down" = moved toward the end (worse). Falls back
    to a stable string-comparison label ("set" prefix) when the ladder or value is unknown,
    so an off-ladder value still yields a deterministic, entity-agnostic token rather than
    crashing.
    """
    ladders = _attr_ladders()
    ladder = ladders.get(attr) if ladders else None
    if ladder and old_val in ladder and new_val in ladder:
        oi, ni = ladder.index(old_val), ladder.index(new_val)
        if ni < oi:
            return "up"
        if ni > oi:
            return "down"
        return "same"  # unreachable (values differ ⟹ indices differ), defensive
    # Unknown ladder/value: fall back to the destination value so the label is still stable.
    return f"set:{new_val}"


def changed_attr_label(prev_entities, next_entities) -> str:
    """Entity-agnostic canonical label of the CHANGED-ATTRIBUTE DELTA of one transition.

    Diffs the previous vs next structured entity states (`[(type_name, {attr: value}), ...]`)
    and produces a stable canonical string of the MULTISET of changed `(attribute, direction)`
    pairs — ENTITY-AGNOSTIC: which entity changed is dropped, so "hunger decreased" is the
    SAME class no matter which entity it happened to (and regardless of entity order). This is
    deliberately COARSER than the joint next-state string: it collapses the 33K-class joint
    label (78% singletons) down to the tens-of-classes space of "what kind of change happened",
    so most in-batch anchors find a positive (the verb-anchor coverage that already works).

    Why (attribute, direction) and not (entity, attribute, value): the oracle changes 1-2
    attributes per action by a single ordinal step, so the *kind* of change ("hunger up",
    "cleanliness down") recurs across thousands of transitions while the full joint state is
    near-unique. Keeping the direction (not just the attribute) preserves the semantic content
    L_sep needs — "fed" (hunger up) and "starved" (hunger down) stay distinct sub-manifolds —
    while staying frequent. Using a MULTISET (sorted, with counts) keeps two-attribute actions
    (e.g. "feed" raising hunger AND mood) as their own class, distinct from either single change.

    No-op (nothing changed — e.g. an action with no in-schema effect, or a clamped value at
    a ladder boundary) maps to the dedicated "no_change" token, its own class.

    Args:
        prev_entities: structured state BEFORE the transition (list of (type, {attr:val})).
        next_entities: structured state AFTER  the transition (same shape, same entity order).

    Returns:
        a stable canonical string, e.g. "hunger:up" or "hunger:up,mood:up" or "no_change".
    """
    changed: list[str] = []
    n = min(len(prev_entities), len(next_entities))
    for i in range(n):
        _, prev_state = prev_entities[i]
        _, next_state = next_entities[i]
        for attr in sorted(set(prev_state) | set(next_state)):
            ov = prev_state.get(attr)
            nv = next_state.get(attr)
            if ov != nv:
                changed.append(f"{attr}:{_attr_direction(attr, ov, nv)}")
    if not changed:
        return "no_change"
    # Sorted multiset (counts preserved by the sort over the per-pair tokens) ⟹ stable,
    # entity-agnostic, order-independent. Comma-join the sorted pair tokens.
    return ",".join(sorted(changed))


def replay_changed_attr_labels(types, initial_states, actions):
    """Replay a chain -> list of changed-attribute delta labels, one PER TRANSITION.

    Returns a list of length len(actions): entry i is the changed_attr_label of transition i
    (state_i -> state_{i+1}). Returns None when the oracle/labeled fields are unavailable (the
    caller then leaves the changed-attr id = -1, mirroring replay_canonical_states).
    """
    states = _replay_entity_states(types, initial_states, actions)
    if states is None:
        return None
    return [
        changed_attr_label(states[i], states[i + 1]) for i in range(len(states) - 1)
    ]


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
