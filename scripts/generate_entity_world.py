#!/usr/bin/env python3
"""Synthetic entity-world trace generator for the engram-wm OOD-entity experiment.

Spirit reference: scripts/generate_pet_sim.py (the original closed-vocab pet generator).
Output format: the JEPA chain JSONL the pipeline consumes (src/twm/jepa/data.py):

    {"chain": [text_0, text_1, ...]}                       # main splits
    {"chain": [...], "actions": [action_0, action_1, ...]} # *_labeled variants

Why this exists
---------------
The engram-wm program (substrate objective `engram-wm`, task 2) measures OOD-ENTITY
generalization: a *continuous modulus-identity* memory should address entity types it
never saw in training, whereas a lookup table structurally cannot. This generator builds
a world where entity TYPES respond to the SAME action DIFFERENTLY (the differential
response profile is the thing continuous identity must capture and a lookup memorizes).
Disjoint train_types / ood_types with a graded-similarity knob (near vs far OOD) let the
experiment discriminate interpolation in identity space from pure memorization.

World spec
----------
- Entity TYPES, each with a typed ATTRIBUTE SCHEMA (subset of the shared attribute pool,
  each attribute an ordinal ladder of values) and a RESPONSE PROFILE: per-(action) the
  ordinal effect on each of its attributes ("up" toward index 0 = better, "down" toward
  the worst index). Some effects are CONDITIONAL on the current state.
- A fixed ACTION vocabulary shared across all types. Types ignore actions outside their
  schema (no-op) -> the same action sequence yields type-specific state trajectories.
- ORACLE dynamics: deterministic ordinal shifts; optional mild stochastic tie-breaks are
  gated off by default (STOCHASTIC=False) so traces are exactly replayable for the oracle
  consistency test. (When on, a per-chain rng-stream is recorded so replay still matches.)

Splits
------
- train            : chains over train_types only
- test_iid         : held-out chains over train_types (same vocab, fresh states/actions)
- test_ood_near    : chains over near-OOD types (seen schema + SMALL perturbation of a seen
                     type's response profile) -> identity space should interpolate
- test_ood_far     : chains over far-OOD types (novel recombination of seen attribute
                     schemas + a structurally novel response profile) -> hardest

Every split also gets a `*_labeled.jsonl` twin carrying the ground-truth action per step,
enabling action-recovery evaluation of the unsupervised latent actions.

Run (from repo root):
    uv run python scripts/generate_entity_world.py
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

# =====================================================================================
# CONFIG  (edit here; everything below is driven by this dict)
# =====================================================================================

CONFIG = {
    "seed": 7,
    "out_dir": "data/entity_world",
    "n_train_chains": 40_000,
    "n_test_chains": 2_000,          # per test split
    "chain_len_min": 4,              # states per chain (inclusive)
    "chain_len_max": 8,
    "entities_per_chain": (1, 2),    # sampled uniformly from this inclusive range
    "wait_weight": 0.15,            # relative sampling weight of the no-op 'wait' action
    "stochastic": False,            # deterministic oracle (replayable). See module docstring.
    # The graded-similarity knob. near-OOD perturbs ONE action's effect on a base type;
    # far-OOD rebuilds the response profile from a novel recombination. Documented in the
    # TYPE_LIBRARY below and surfaced in the manifest.
    "bpe_path": "data/glucose/jepa_bpe_512.json",   # existing 512 domain BPE
    "max_text_tokens": 64,
}

# =====================================================================================
# SHARED PRIMITIVES
# =====================================================================================

# Attribute pool: name -> ordinal ladder (index 0 = best, last = worst).
# Kept in GLUCOSE-friendly common-English register so the existing BPE mostly transfers.
ATTRIBUTE_POOL = {
    "hunger":   ["full", "fed", "hungry", "starving"],
    "energy":   ["lively", "rested", "tired", "worn out"],
    "mood":     ["cheerful", "happy", "calm", "sad"],
    "cleanliness": ["spotless", "clean", "messy", "dirty"],
    "thirst":   ["watered", "fresh", "thirsty", "parched"],
    "power":    ["on", "warming", "cooling", "off"],
    "fill":     ["full", "filling", "draining", "empty"],
    "open":     ["wide open", "open", "ajar", "shut"],
}

# Actions: a single shared vocabulary. Effects are defined PER TYPE in the response
# profile; an action not in a type's profile is a no-op for that type.
ACTIONS = ["feed", "play", "wash", "rest", "water", "switch on", "switch off", "fill", "open", "close", "wait"]

# Human-readable surface forms for rendering. {ent} is the entity's display noun phrase.
ACTION_TEMPLATES = {
    "feed":       "Someone_A feeds {ent}.",
    "play":       "Someone_A plays with {ent}.",
    "wash":       "Someone_A washes {ent}.",
    "rest":       "{ent} rests for a while.",
    "water":      "Someone_A waters {ent}.",
    "switch on":  "Someone_A switches on {ent}.",
    "switch off": "Someone_A switches off {ent}.",
    "fill":       "Someone_A fills {ent}.",
    "open":       "Someone_A opens {ent}.",
    "close":      "Someone_A closes {ent}.",
    "wait":       "Time passes.",
}

# Attribute -> how its current value reads in a state sentence: "{ent} is/feels <val>".
# "feel" reads natural for living things; "is" for objects/devices. Chosen per attribute.
ATTR_COPULA = {
    "hunger": "feel(s)", "energy": "feel(s)", "mood": "feel(s)", "thirst": "feel(s)",
    "cleanliness": "is", "power": "is", "fill": "is", "open": "is",
}

# =====================================================================================
# TYPE LIBRARY
# =====================================================================================
# Each type:
#   display : noun phrase for rendering, e.g. "the dog"
#   schema  : ordered list of attributes (subset of ATTRIBUTE_POOL keys)
#   profile : {action: {attr: "up"/"down"}}  base ordinal effects
#   cond    : {action: [(condition, override_effects), ...]}  first match wins
#             condition: {attr: set_of_values}; override replaces base for that action
#   split_role : "train" | "near_ood" | "far_ood"
#   derived_from / similarity : documentation of the graded-similarity knob
#
# CONTRACT: near_ood types share a base type's schema and differ by a SMALL perturbation
# (one action's effect flipped/changed). far_ood types use a novel recombination of seen
# schemas with a structurally different profile.

def _living_schema():
    return ["hunger", "energy", "mood", "cleanliness"]

def _plant_schema():
    return ["thirst", "mood", "cleanliness"]

def _device_schema():
    return ["power", "fill", "cleanliness"]

def _container_schema():
    return ["fill", "open", "cleanliness"]


TYPE_LIBRARY = {
    # ----------------------------- TRAIN TYPES -----------------------------
    "dog": {
        "display": "the dog",
        "schema": _living_schema(),
        "profile": {
            "feed": {"hunger": "up", "mood": "up"},
            "play": {"mood": "up", "energy": "down"},
            "wash": {"cleanliness": "up", "mood": "down"},
            "rest": {"energy": "up"},
        },
        "cond": {
            "play": [({"energy": {"worn out"}}, {"mood": "down", "energy": "down"})],
        },
        "split_role": "train",
    },
    "cat": {
        "display": "the cat",
        "schema": _living_schema(),
        "profile": {
            "feed": {"hunger": "up"},
            "play": {"mood": "up", "energy": "down"},
            "wash": {"cleanliness": "up", "mood": "down"},   # cats hate baths
            "rest": {"energy": "up", "mood": "up"},
        },
        "cond": {
            "wash": [({"mood": {"sad"}}, {"cleanliness": "up", "mood": "down", "energy": "down"})],
        },
        "split_role": "train",
    },
    "horse": {
        "display": "the horse",
        "schema": _living_schema(),
        "profile": {
            "feed": {"hunger": "up"},
            "play": {"mood": "up", "energy": "down", "cleanliness": "down"},
            "wash": {"cleanliness": "up"},
            "rest": {"energy": "up"},
        },
        "cond": {},
        "split_role": "train",
    },
    "fern": {
        "display": "the fern",
        "schema": _plant_schema(),
        "profile": {
            "water": {"thirst": "up", "mood": "up"},
            "wash": {"cleanliness": "up"},
            "wait": {"thirst": "down"},
        },
        "cond": {
            "water": [({"thirst": {"watered"}}, {"thirst": "up", "mood": "down"})],  # overwatering
        },
        "split_role": "train",
    },
    "lamp": {
        "display": "the lamp",
        "schema": _device_schema(),
        "profile": {
            "switch on": {"power": "up"},
            "switch off": {"power": "down"},
            "wash": {"cleanliness": "up"},
            "wait": {"fill": "down"},     # battery/oil drains over time
        },
        "cond": {},
        "split_role": "train",
    },
    "kettle": {
        "display": "the kettle",
        "schema": _container_schema(),
        "profile": {
            "fill": {"fill": "up"},
            "open": {"open": "up"},
            "close": {"open": "down"},
            "wash": {"cleanliness": "up"},
            "wait": {"fill": "down"},
        },
        "cond": {},
        "split_role": "train",
    },
    "box": {
        "display": "the box",
        "schema": _container_schema(),
        "profile": {
            "fill": {"fill": "up", "cleanliness": "down"},
            "open": {"open": "up"},
            "close": {"open": "down"},
            "wash": {"cleanliness": "up"},
        },
        "cond": {},
        "split_role": "train",
    },

    # --------------------------- NEAR-OOD TYPES ----------------------------
    # Seen schema + SMALL perturbation of a seen type's response profile.
    "puppy": {
        "display": "the puppy",
        "schema": _living_schema(),
        # derived from dog: feed gives MORE mood; wash is enjoyed (mood up not down).
        "profile": {
            "feed": {"hunger": "up", "mood": "up"},
            "play": {"mood": "up", "energy": "down"},
            "wash": {"cleanliness": "up", "mood": "up"},     # <- flipped vs dog
            "rest": {"energy": "up"},
        },
        "cond": {
            "play": [({"energy": {"worn out"}}, {"mood": "down", "energy": "down"})],
        },
        "split_role": "near_ood",
        "derived_from": "dog",
        "similarity": "near: wash mood effect flipped (down->up)",
    },
    "pony": {
        "display": "the pony",
        "schema": _living_schema(),
        # derived from horse: play does not dirty it (drop the cleanliness-down effect).
        "profile": {
            "feed": {"hunger": "up"},
            "play": {"mood": "up", "energy": "down"},        # <- dropped cleanliness:down
            "wash": {"cleanliness": "up"},
            "rest": {"energy": "up", "mood": "up"},           # <- added mood:up on rest
        },
        "cond": {},
        "split_role": "near_ood",
        "derived_from": "horse",
        "similarity": "near: play no longer dirties; rest now lifts mood",
    },
    "sprout": {
        "display": "the sprout",
        "schema": _plant_schema(),
        # derived from fern: no overwatering penalty; wait drains thirst faster (also mood).
        "profile": {
            "water": {"thirst": "up", "mood": "up"},
            "wash": {"cleanliness": "up"},
            "wait": {"thirst": "down", "mood": "down"},       # <- wait now also lowers mood
        },
        "cond": {},
        "split_role": "near_ood",
        "derived_from": "fern",
        "similarity": "near: overwatering penalty removed; wait also lowers mood",
    },

    # ---------------------------- FAR-OOD TYPES ----------------------------
    # Novel RECOMBINATION of seen attribute schemas + structurally novel profile.
    "terrarium": {
        "display": "the terrarium",
        # novel schema: container (fill/open) + plant-ish thirst + cleanliness.
        "schema": ["fill", "open", "thirst", "cleanliness"],
        "profile": {
            "water": {"thirst": "up", "fill": "up"},          # watering also fills it
            "fill": {"fill": "up"},
            "open": {"open": "up", "thirst": "down"},          # opening dries it out
            "close": {"open": "down"},
            "wash": {"cleanliness": "up"},
            "wait": {"thirst": "down", "fill": "down"},
        },
        "cond": {
            "open": [({"thirst": {"parched"}}, {"open": "up"})],  # already dry: no extra dry
        },
        "split_role": "far_ood",
        "derived_from": "kettle+fern",
        "similarity": "far: novel schema combo (container+plant); cross-attribute coupling water->fill, open->thirst",
    },
    "robot pet": {
        "display": "the robot pet",
        # novel schema: device (power/fill) + living (mood/energy). A device that ALSO has
        # mood — no training type couples these. Charge couples to mood and energy.
        "schema": ["power", "fill", "mood", "energy"],
        "profile": {
            "switch on": {"power": "up", "energy": "up"},
            "switch off": {"power": "down", "energy": "down"},
            "play": {"mood": "up", "fill": "down"},            # play drains battery
            "fill": {"fill": "up", "mood": "up"},              # charging makes it happy
            "wait": {"fill": "down"},
        },
        "cond": {
            "play": [({"fill": {"empty"}}, {"mood": "down"})],  # can't play when dead
        },
        "split_role": "far_ood",
        "derived_from": "lamp+dog",
        "similarity": "far: device+living combo; charge<->mood<->energy coupling unseen in training",
    },
}

# =====================================================================================
# ORACLE DYNAMICS
# =====================================================================================

def _ladder(attr):
    return ATTRIBUTE_POOL[attr]


def _shift(value, attr, direction):
    vals = _ladder(attr)
    idx = vals.index(value)
    if direction == "up":
        idx = max(0, idx - 1)
    else:
        idx = min(len(vals) - 1, idx + 1)
    return vals[idx]


def _matches(state, condition):
    return all(state.get(attr) in allowed for attr, allowed in condition.items())


def _effects_for(type_def, state, action):
    """Resolve the ordinal effects of `action` on `state` for this type.

    Returns {} (no-op) if the action is not in the type's profile. Conditional overrides
    (first match wins) replace the base effects for that action.
    """
    for condition, override in type_def.get("cond", {}).get(action, []):
        if _matches(state, condition):
            return dict(override)
    return dict(type_def["profile"].get(action, {}))


def apply_action(type_name, state, action, rng=None):
    """Oracle transition. Deterministic unless CONFIG['stochastic'] and a tie occurs.

    `state` is {attr: value} restricted to the type's schema. Returns a NEW state dict.
    Effects on attributes outside the schema are ignored (defensive; profiles are authored
    to stay in-schema).
    """
    type_def = TYPE_LIBRARY[type_name]
    schema = set(type_def["schema"])
    effects = _effects_for(type_def, state, action)
    new = dict(state)
    for attr, direction in effects.items():
        if attr in schema:
            new[attr] = _shift(state[attr], attr, direction)
    return new


# =====================================================================================
# RENDERING  (state dict -> GLUCOSE-like text)
# =====================================================================================

def _capitalize_sentence(s):
    """Capitalize the first letter of a rendered clause (display nouns start with 'the')."""
    return s[:1].upper() + s[1:] if s else s


def _value_clause(ent_display, attr, value):
    cop = ATTR_COPULA[attr]
    return f"{ent_display} {cop} {value}"


def render_state(entities):
    """Render a multi-entity world state as 2-4 short sentences.

    `entities` is a list of (type_name, state_dict). We surface up to two salient
    attributes per entity (the first two in its schema) to keep states at 2-4 sentences.
    Every sentence starts capitalized.
    """
    sentences = []
    for type_name, state in entities:
        disp = TYPE_LIBRARY[type_name]["display"]
        schema = TYPE_LIBRARY[type_name]["schema"]
        salient = schema[:2]
        clauses = [_value_clause(disp, attr, state[attr]) for attr in salient]
        # "The dog feel(s) hungry. The dog is messy."
        sentences.extend(_capitalize_sentence(c) + "." for c in clauses)
    return " ".join(sentences)


def render_action(entities, action, actor_idx):
    """Render the action sentence (applied to entity `actor_idx`)."""
    type_name, _ = entities[actor_idx]
    disp = TYPE_LIBRARY[type_name]["display"]
    template = ACTION_TEMPLATES[action]
    return template.format(ent=disp)


# =====================================================================================
# CHAIN GENERATION
# =====================================================================================

def _random_state(rng, type_name):
    schema = TYPE_LIBRARY[type_name]["schema"]
    return {attr: rng.choice(_ladder(attr)) for attr in schema}


def _applicable_actions(type_name):
    """Actions that have a (possibly conditional) effect for this type, plus 'wait'."""
    type_def = TYPE_LIBRARY[type_name]
    acts = set(type_def["profile"].keys()) | set(type_def.get("cond", {}).keys())
    acts.add("wait")
    return sorted(acts)


def _sample_action(rng, type_name, wait_weight):
    """Sample an applicable action, down-weighting the no-op 'wait' so chains carry more
    state-changing cross-state signal. 'wait' is kept (it is a real dynamics rule for some
    types: lamp drains, fern dries) but rarer."""
    acts = _applicable_actions(type_name)
    weights = [wait_weight if a == "wait" else 1.0 for a in acts]
    return rng.choices(acts, weights=weights, k=1)[0]


def generate_chain(rng, type_names, chain_len, wait_weight=0.15):
    """Generate one chain over the given entity types.

    Returns (chain_texts, action_labels) where action_labels[i] is the action applied to
    go from state i to state i+1 (so len(action_labels) == len(chain_texts) - 1). Each
    label is "<action>@<entity_index>" so action-recovery eval can align actor + verb.
    """
    entities = [(tn, _random_state(rng, tn)) for tn in type_names]

    chain_texts = []
    action_labels = []

    # state_0
    chain_texts.append(render_state(entities))

    for _ in range(chain_len - 1):
        actor_idx = rng.randrange(len(entities))
        type_name = entities[actor_idx][0]
        action = _sample_action(rng, type_name, wait_weight)

        # apply oracle
        new_state = apply_action(type_name, entities[actor_idx][1], action, rng)
        entities[actor_idx] = (type_name, new_state)

        action_sentence = render_action(entities, action, actor_idx)
        state_sentence = render_state(entities)
        # Each chain step text = the action that happened + the resulting state.
        chain_texts.append(f"{action_sentence} {state_sentence}")
        action_labels.append(f"{action}@{actor_idx}")

    return chain_texts, action_labels


def replay_chain(type_names, initial_states, action_labels):
    """Oracle replay: given initial states + action labels, reproduce the state sequence.

    Returns list of entity-states snapshots (one per chain step, len == n_actions + 1).
    Used by the oracle-consistency test.
    """
    entities = [(tn, dict(st)) for tn, st in zip(type_names, initial_states)]
    snapshots = [[dict(st) for _, st in entities]]
    for label in action_labels:
        action, actor_idx = label.rsplit("@", 1)
        actor_idx = int(actor_idx)
        type_name = entities[actor_idx][0]
        new_state = apply_action(type_name, entities[actor_idx][1], action)
        entities[actor_idx] = (type_name, new_state)
        snapshots.append([dict(st) for _, st in entities])
    return snapshots


# =====================================================================================
# SPLIT BUILD
# =====================================================================================

def _types_for_role(role):
    return [name for name, d in TYPE_LIBRARY.items() if d["split_role"] == role]


def _sample_type_names(rng, allowed_types, k_range):
    k = rng.randint(*k_range)
    # sample with replacement allowed only across DIFFERENT type slots is fine; but two of
    # the same display in one state reads oddly, so sample distinct when possible.
    if k <= len(allowed_types):
        return rng.sample(allowed_types, k)
    return [rng.choice(allowed_types) for _ in range(k)]


def _initial_states_for_chain(type_names, texts, rng_seed=None):
    """Reconstruct the initial state dicts by parsing the first state text against the
    type schemas.  Used by the retraction probe (§4.2) to obtain parse-free, exact initial
    states without a re-seed.

    Returns a list of state dicts (one per entity in type_names order).  Note: this is
    only needed when building labeled records; ``build_split`` now emits the states
    directly, so the field is deterministic and parse-free.
    """
    # We can't parse text->state reliably in the general case, so this helper is NOT
    # used at build time.  The caller (build_split) captures states directly during
    # generation and stores them.  This function is kept as a reference; callers should
    # read ``initial_states`` from the labeled record directly.
    raise NotImplementedError("Use the 'initial_states' field from the labeled record instead.")


def build_split(rng, allowed_types, n_chains, cfg):
    """Build n_chains chains over allowed_types. Returns (plain_records, labeled_records).

    Each labeled record now includes an ``initial_states`` field: a list of state dicts
    (one per entity, in type_names order) representing the entity states BEFORE any action
    has been applied.  This lets the retraction probe (jepa_entity_campaign.md §4.2) replay
    chains from the oracle without fragile text parsing.  The field is deterministic (seeded
    generator) and regenerated by the GPU job on every run.
    """
    plain, labeled = [], []
    for _ in range(n_chains):
        chain_len = rng.randint(cfg["chain_len_min"], cfg["chain_len_max"])
        type_names = _sample_type_names(rng, allowed_types, cfg["entities_per_chain"])
        # generate_chain builds entities = [(type_name, state_dict), ...] internally.
        # We call it and also capture the initial states for the labeled record.
        entities_init = [(tn, _random_state(rng, tn)) for tn in type_names]
        # Capture initial states BEFORE mutation by the chain generator.
        initial_states = [dict(st) for _, st in entities_init]
        texts, actions = _generate_chain_from_entities(rng, entities_init, chain_len,
                                                        cfg.get("wait_weight", 0.15))
        plain.append({"chain": texts})
        labeled.append({
            "chain": texts,
            "actions": actions,
            "types": type_names,
            "initial_states": initial_states,
        })
    return plain, labeled


def _generate_chain_from_entities(rng, entities, chain_len, wait_weight=0.15):
    """Generate a chain starting from the given (mutable) entities list.

    Mirrors the body of ``generate_chain`` but accepts pre-constructed entities so
    ``build_split`` can capture the initial states before generation begins.

    Mutates ``entities`` in place (same semantics as ``generate_chain``).
    """
    chain_texts = [render_state(entities)]
    action_labels = []
    for _ in range(chain_len - 1):
        actor_idx = rng.randrange(len(entities))
        type_name = entities[actor_idx][0]
        action = _sample_action(rng, type_name, wait_weight)
        new_state = apply_action(type_name, entities[actor_idx][1], action, rng)
        entities[actor_idx] = (type_name, new_state)
        action_sentence = render_action(entities, action, actor_idx)
        state_sentence = render_state(entities)
        chain_texts.append(f"{action_sentence} {state_sentence}")
        action_labels.append(f"{action}@{actor_idx}")
    return chain_texts, action_labels


# =====================================================================================
# I/O + STATS
# =====================================================================================

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_manifest(cfg):
    """Describe every type's schema and response profile + the split design."""
    types = {}
    for name, d in TYPE_LIBRARY.items():
        types[name] = {
            "display": d["display"],
            "schema": d["schema"],
            "split_role": d["split_role"],
            "profile": d["profile"],
            "conditional": {
                a: [{"when": {k: sorted(v) for k, v in cond.items()}, "effects": eff}
                    for cond, eff in lst]
                for a, lst in d.get("cond", {}).items()
            },
            **({"derived_from": d["derived_from"]} if "derived_from" in d else {}),
            **({"similarity": d["similarity"]} if "similarity" in d else {}),
        }
    return {
        "config": cfg,
        "attribute_pool": ATTRIBUTE_POOL,
        "actions": ACTIONS,
        "attr_copula": ATTR_COPULA,
        "splits": {
            "train":         {"types": _types_for_role("train"),    "discriminates": "in-distribution dynamics"},
            "test_iid":      {"types": _types_for_role("train"),    "discriminates": "generalization to fresh states/actions, seen types"},
            "test_ood_near": {"types": _types_for_role("near_ood"), "discriminates": "identity-space interpolation (small profile perturbation)"},
            "test_ood_far":  {"types": _types_for_role("far_ood"),  "discriminates": "novel schema recombination + structurally novel profile"},
        },
        "types": types,
    }


def coverage_report(records_by_split, bpe_path, max_tokens):
    """Report tokens-per-state under the existing 512 BPE. Returns a verdict dict."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return {"available": False, "reason": "tokenizers not importable"}
    if not Path(bpe_path).exists():
        return {"available": False, "reason": f"{bpe_path} missing"}

    tok = Tokenizer.from_file(bpe_path)
    report = {"available": True, "bpe_path": bpe_path, "vocab_size": tok.get_vocab_size(),
              "max_text_tokens": max_tokens, "per_split": {}}
    all_lengths = []
    all_frag = []
    for split, records in records_by_split.items():
        lengths, frags = [], []
        for r in records:
            for text in r["chain"]:
                enc = tok.encode(text)
                lengths.append(len(enc.ids))
                # byte-fallback heuristic: single alpha chars left as their own token
                frags.append(sum(1 for t in enc.tokens
                                 if len(t.replace("Ġ", "")) == 1 and t.replace("Ġ", "").isalpha()))
        all_lengths += lengths
        all_frag += frags
        report["per_split"][split] = {
            "states": len(lengths),
            "mean_tokens": round(statistics.mean(lengths), 2),
            "p95_tokens": sorted(lengths)[int(len(lengths) * 0.95)],
            "max_tokens": max(lengths),
            "pct_over_max": round(100 * sum(1 for l in lengths if l > max_tokens) / len(lengths), 2),
            "mean_byte_frags": round(statistics.mean(frags), 2),
        }
    mean_len = statistics.mean(all_lengths)
    mean_frag = statistics.mean(all_frag)
    pct_over = 100 * sum(1 for l in all_lengths if l > max_tokens) / len(all_lengths)
    # Verdict: poor if many states overflow OR heavy byte fragmentation.
    poor = pct_over > 1.0 or mean_frag > 4.0
    report["overall"] = {
        "mean_tokens": round(mean_len, 2),
        "mean_byte_frags": round(mean_frag, 2),
        "pct_over_max": round(pct_over, 2),
        "verdict": "POOR -> build dedicated BPE" if poor else "GOOD -> existing 512 BPE transfers",
        "poor": poor,
    }
    return report


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    cfg = CONFIG
    rng = random.Random(cfg["seed"])
    out_dir = Path(cfg["out_dir"])

    train_types = _types_for_role("train")
    near_types = _types_for_role("near_ood")
    far_types = _types_for_role("far_ood")

    print("Entity-World generator")
    print(f"  train types : {train_types}")
    print(f"  near-OOD    : {near_types}")
    print(f"  far-OOD     : {far_types}")
    print(f"  seed={cfg['seed']} stochastic={cfg['stochastic']}")
    print()

    # Build splits. Distinct rng streams keep splits reproducible & disjoint in states.
    train_plain, train_labeled = build_split(
        random.Random(cfg["seed"] + 1), train_types, cfg["n_train_chains"], cfg)
    iid_plain, iid_labeled = build_split(
        random.Random(cfg["seed"] + 2), train_types, cfg["n_test_chains"], cfg)
    near_plain, near_labeled = build_split(
        random.Random(cfg["seed"] + 3), near_types, cfg["n_test_chains"], cfg)
    far_plain, far_labeled = build_split(
        random.Random(cfg["seed"] + 4), far_types, cfg["n_test_chains"], cfg)

    splits = {
        "train": (train_plain, train_labeled),
        "test_iid": (iid_plain, iid_labeled),
        "test_ood_near": (near_plain, near_labeled),
        "test_ood_far": (far_plain, far_labeled),
    }

    for name, (plain, labeled) in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", plain)
        write_jsonl(out_dir / f"{name}_labeled.jsonl", labeled)

    # Manifest
    manifest = build_manifest(cfg)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats table
    print("Split stats")
    print(f"  {'split':<16}{'chains':>8}{'states':>9}{'mean_len':>10}{'actions':>9}")
    for name, (plain, labeled) in splits.items():
        n_states = sum(len(r["chain"]) for r in plain)
        n_actions = sum(len(r["actions"]) for r in labeled)
        mean_len = round(n_states / len(plain), 2)
        print(f"  {name:<16}{len(plain):>8}{n_states:>9}{mean_len:>10}{n_actions:>9}")
    print()

    # Tokenizer coverage
    report = coverage_report({k: v[0] for k, v in splits.items()},
                             cfg["bpe_path"], cfg["max_text_tokens"])
    with open(out_dir / "coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if report.get("available"):
        print("Tokenizer coverage (existing 512 BPE)")
        for split, s in report["per_split"].items():
            print(f"  {split:<16} mean={s['mean_tokens']:>6} p95={s['p95_tokens']:>4} "
                  f"max={s['max_tokens']:>4} frags={s['mean_byte_frags']:>5} over_max={s['pct_over_max']}%")
        ov = report["overall"]
        print(f"  OVERALL mean={ov['mean_tokens']} frags={ov['mean_byte_frags']} "
              f"over_max={ov['pct_over_max']}%  ->  {ov['verdict']}")
        if ov["poor"]:
            print()
            print("  Coverage is POOR. Build a dedicated BPE by adapting build_jepa_bpe.py:")
            print("    - point collect_texts at data/entity_world/train.jsonl")
            print("    - OUT_PATH = data/entity_world/bpe_512.json")
            print("    Then: uv run python scripts/build_entity_world_bpe.py")
    else:
        print(f"Tokenizer coverage unavailable: {report.get('reason')}")

    print()
    print(f"Wrote {out_dir}/ : "
          f"{', '.join(sorted(p.name for p in out_dir.glob('*.jsonl')))}, "
          f"manifest.json, coverage_report.json")


if __name__ == "__main__":
    main()
