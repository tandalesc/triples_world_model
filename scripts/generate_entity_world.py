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
    # ---------------------------------------------------------------------------
    # v4 entity-world-v2 extension (jepa_v4_design.md §4). All keys below default to
    # v1 values so the default CONFIG is byte-identical to the campaign (world_version=1).
    # ---------------------------------------------------------------------------
    # Master switch: 1 = campaign byte-identical (ignore all *_v2 keys); 2 = expanded.
    "world_version": 1,
    # v2 entity counts (used only when world_version==2)
    "entities_per_chain_v2": (3, 5),
    # v2 chain lengths (used only when world_version==2)
    "chain_len_min_v2": 6,
    "chain_len_max_v2": 12,
    # v2 training chains (used only when world_version==2)
    "n_train_chains_v2": 120_000,
    # Optional stochastic mode (used only when world_version==2 and stochastic_v2=True).
    # When on, per-type "stochastic" tables are sampled; oracle_dist is emitted in labeled.
    "stochastic_v2": False,
    "stochastic_p": 0.15,
    # ---------------------------------------------------------------------------
    # v4.3 surface-variety (paraphrase mode). See SURFACE-VARIETY block below.
    # Default OFF -> rendering is byte-identical to the v4.2 campaign output.
    # When ON, each underlying state renders through one of K>=4 paraphrase
    # templates per sentence family + a clause-order shuffle. The underlying
    # state and oracle are UNTOUCHED; paraphrase affects RENDERING only.
    # ---------------------------------------------------------------------------
    "surface_variety": False,
    # ---------------------------------------------------------------------------
    # v6 §B LAM surface-augmentation invariance (research/jepa_v6_unsupervised_design.md,
    # arXiv:2506.15691). Default OFF -> output is unchanged. When ON, each chain ALSO emits
    # a `chain_aug` field: a SECOND independent surface frame φ' of the SAME state sequence,
    # rendered with a FRESH chain_template_seed (and surface_variety forced on for the φ'
    # frame). The posterior/noun path sees the primary `chain` frame (φ); the decoder CE
    # target is the `chain_aug` frame (φ'). The two frames render the SAME underlying states
    # (the renderer is an INPUT TRANSFORM — augmentation, not a label). The trainer/loss only
    # ever see rendered text. The augmentation interface is PLUGGABLE: a text-level
    # paraphraser could supply `chain_aug` instead of the renderer (see the design doc).
    # ---------------------------------------------------------------------------
    "emit_lam_aug": False,
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
# SURFACE-VARIETY (paraphrase mode, v4.3)
# =====================================================================================
# HYPOTHESIS under test: single-template data never DEMANDS metric pooled geometry.
# Many surface forms per underlying state force invariance, and invariance is exactly
# semantic geometry. So we render the SAME state/oracle/dynamics through many surface
# forms (variety ACROSS chains/samples) while holding template choice STABLE WITHIN a
# chain so the masked-diff alignment (data._diff_mask, SequenceMatcher) still isolates
# the single changed clause rather than diffing the whole text.
#
# DESIGN DECISION (critical invariant): the template chosen for a given clause is a
# deterministic function of (chain_template_seed, entity_index, attribute_key) for state
# clauses, and (chain_template_seed, entity_index) for the entity's action-verb style /
# clause order. It is NOT a function of the attribute VALUE. Consequence:
#   - ACROSS chains/samples: the per-chain seed varies, so the SAME underlying state can
#     render many different ways (surface entropy gain).
#   - WITHIN a chain: when an attribute value changes (cat full -> fed) the surrounding
#     template words are byte-identical between consecutive states, so SequenceMatcher's
#     LCS alignment marks ONLY the changed value word as the diff span. If template choice
#     depended on the value, every step would re-template and the diff mask would cover the
#     whole text -- defeating masked-diff. The clause-order shuffle is likewise fixed per
#     (chain, entity), so entity order within a state text is stable through the chain.
#
# All paraphrase vocabulary REUSES words already in reach (common-English register); a
# fresh BPE is rebuilt on the paraphrase renderings (build_entity_world_bpe.py) because
# the campaign's single-template BPE never saw "seems"/"looks"/"now"/etc.

# K>=4 attribute-statement templates. {ent}=display noun phrase, {cop}=copula
# ("feel(s)"/"is"), {val}=ordinal value word. Each renders ONE attribute clause.
# Variant 0 is the canonical single-template form ("{ent} {cop} {val}.") so the family
# always includes the campaign surface form.
ATTR_TEMPLATES = [
    "{ent} {cop} {val}.",                  # The cat feel(s) full.
    "{ent} {cop} {val} now.",              # The cat feel(s) full now.
    "{ent} {cop} {val} today.",            # The cat feel(s) full today.
    "{ent} {cop} rather {val}.",           # The cat feel(s) rather full.
    "{ent} still {cop} {val}.",            # The cat still feel(s) full.
]

# K>=4 action-sentence templates per action. Variant 0 is the canonical ACTION_TEMPLATES
# form. We keep one shared schema of "extra" framings and instantiate per action so the
# verb morphology stays correct. {ent}=display, {a}=action key (used to pick the row).
# Each list value is a list of >=4 surface strings for that action.
ACTION_TEMPLATES_VAR = {
    "feed": [
        "Someone_A feeds {ent}.",
        "Someone_A gives {ent} some food.",
        "{ent} is fed by Someone_A.",
        "Someone_A feeds {ent} now.",
    ],
    "play": [
        "Someone_A plays with {ent}.",
        "Someone_A has a play with {ent}.",
        "{ent} plays with Someone_A.",
        "Someone_A plays with {ent} now.",
    ],
    "wash": [
        "Someone_A washes {ent}.",
        "Someone_A gives {ent} a wash.",
        "{ent} is washed by Someone_A.",
        "Someone_A washes {ent} now.",
    ],
    "rest": [
        "{ent} rests for a while.",
        "{ent} takes a rest.",
        "{ent} rests now.",
        "{ent} has a rest for a while.",
    ],
    "water": [
        "Someone_A waters {ent}.",
        "Someone_A gives {ent} some water.",
        "{ent} is watered by Someone_A.",
        "Someone_A waters {ent} now.",
    ],
    "switch on": [
        "Someone_A switches on {ent}.",
        "Someone_A turns on {ent}.",
        "{ent} is switched on by Someone_A.",
        "Someone_A switches on {ent} now.",
    ],
    "switch off": [
        "Someone_A switches off {ent}.",
        "Someone_A turns off {ent}.",
        "{ent} is switched off by Someone_A.",
        "Someone_A switches off {ent} now.",
    ],
    "fill": [
        "Someone_A fills {ent}.",
        "Someone_A tops up {ent}.",
        "{ent} is filled by Someone_A.",
        "Someone_A fills {ent} now.",
    ],
    "open": [
        "Someone_A opens {ent}.",
        "Someone_A pulls open {ent}.",
        "{ent} is opened by Someone_A.",
        "Someone_A opens {ent} now.",
    ],
    "close": [
        "Someone_A closes {ent}.",
        "Someone_A shuts {ent}.",
        "{ent} is closed by Someone_A.",
        "Someone_A closes {ent} now.",
    ],
    "wait": [
        "Time passes.",
        "Some time passes.",
        "Time goes by.",
        "A while passes.",
    ],
}


def _stable_choice(options, *keys):
    """Deterministic index into `options` from a tuple of hashable keys.

    Pure function of `keys` (no RNG state): the same (chain_template_seed, entity_idx,
    attr) always selects the same template. Uses a small FNV-1a-style hash over the
    repr of the key tuple so results are stable across Python runs (Python's builtin
    hash() is salted per-process and must NOT be used here)."""
    h = 1469598103934665603  # FNV offset basis (64-bit)
    for part in keys:
        for b in repr(part).encode("utf-8"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF  # FNV prime, 64-bit wrap
    return options[h % len(options)]

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

    # ---- ADDITIONAL TRAIN TYPES (v2: world_version=2 only) ----
    "rabbit": {
        "display": "the rabbit",
        "schema": _living_schema(),
        "profile": {
            "feed": {"hunger": "up", "mood": "up"},
            "play": {"mood": "up", "energy": "down"},
            "rest": {"energy": "up"},
            "wash": {"cleanliness": "up"},
        },
        "cond": {
            "feed": [({"hunger": {"full"}}, {"mood": "down"})],  # overfeeding upsets it
        },
        "split_role": "train",
        "world_version_min": 2,
    },
    "goat": {
        "display": "the goat",
        "schema": _living_schema(),
        "profile": {
            "feed": {"hunger": "up"},
            "play": {"mood": "up", "energy": "down", "cleanliness": "down"},
            "wash": {"cleanliness": "up"},
            "rest": {"energy": "up", "mood": "up"},
        },
        "cond": {},
        "split_role": "train",
        "world_version_min": 2,
    },
    "cactus": {
        "display": "the cactus",
        "schema": _plant_schema(),
        "profile": {
            "water": {"thirst": "up"},
            "wash": {"cleanliness": "up"},
            "wait": {"thirst": "down"},
        },
        "cond": {
            "water": [({"thirst": {"watered"}}, {"mood": "down"})],  # overwatering harms it
        },
        "split_role": "train",
        "world_version_min": 2,
    },
    "vine": {
        "display": "the vine",
        "schema": _plant_schema(),
        "profile": {
            "water": {"thirst": "up", "mood": "up"},
            "wash": {"cleanliness": "up"},
            "wait": {"thirst": "down", "mood": "down"},
        },
        "cond": {},
        "split_role": "train",
        "world_version_min": 2,
    },
    "heater": {
        "display": "the heater",
        "schema": _device_schema(),
        "profile": {
            "switch on": {"power": "up", "fill": "down"},   # burns fuel when on
            "switch off": {"power": "down"},
            "fill": {"fill": "up"},
            "wash": {"cleanliness": "up"},
        },
        "cond": {
            "switch on": [({"fill": {"empty"}}, {"power": "up"})],  # runs but no fuel
        },
        "split_role": "train",
        "world_version_min": 2,
    },
    "fan": {
        "display": "the fan",
        "schema": _device_schema(),
        "profile": {
            "switch on": {"power": "up"},
            "switch off": {"power": "down"},
            "wash": {"cleanliness": "up"},
            "wait": {"cleanliness": "down"},   # dust accumulates
        },
        "cond": {},
        "split_role": "train",
        "world_version_min": 2,
    },
    "jar": {
        "display": "the jar",
        "schema": _container_schema(),
        "profile": {
            "fill": {"fill": "up"},
            "open": {"open": "up"},
            "close": {"open": "down"},
            "wash": {"cleanliness": "up"},
        },
        "cond": {},
        "split_role": "train",
        "world_version_min": 2,
    },
    "crate": {
        "display": "the crate",
        "schema": _container_schema(),
        "profile": {
            "fill": {"fill": "up", "cleanliness": "down"},
            "open": {"open": "up"},
            "close": {"open": "down"},
            "wash": {"cleanliness": "up"},
            "wait": {"cleanliness": "down"},
        },
        "cond": {},
        "split_role": "train",
        "world_version_min": 2,
    },

    # ---- ADDITIONAL NEAR-OOD TYPE (v2, 4th near-OOD) ----
    "kitten": {
        "display": "the kitten",
        "schema": _living_schema(),
        # derived from cat: rest also boosts energy (cats sleep a lot); play drains less.
        "profile": {
            "feed": {"hunger": "up"},
            "play": {"mood": "up"},                         # <- no energy drain (tiny)
            "wash": {"cleanliness": "up", "mood": "down"},
            "rest": {"energy": "up", "mood": "up"},
        },
        "cond": {},
        "split_role": "near_ood",
        "derived_from": "cat",
        "similarity": "near: play no longer drains energy; rest already strong",
        "world_version_min": 2,
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
    # ---- ADDITIONAL FAR-OOD TYPES (v2) ----
    "aquarium": {
        "display": "the aquarium",
        # novel schema: container (fill/open) + living-mood (fish well-being) + thirst (water quality).
        # No training type couples container mechanics with a living creature's mood.
        "schema": ["fill", "open", "mood", "thirst"],
        "profile": {
            "fill": {"fill": "up", "thirst": "up"},        # more water = cleaner water
            "open": {"open": "up", "mood": "up"},           # fish enjoy open lid briefly
            "close": {"open": "down"},
            "water": {"thirst": "up"},
            "feed": {"mood": "up"},                         # feeding cheers the fish
            "wait": {"thirst": "down", "fill": "down"},     # water evaporates + quality drops
        },
        "cond": {
            "open": [({"fill": {"empty"}}, {"mood": "down"})],  # empty tank = sad fish
        },
        "split_role": "far_ood",
        "derived_from": "kettle+fern+dog",
        "similarity": "far: container+living-mood+water-quality; cross-schema coupling unseen in training",
        "world_version_min": 2,
    },
    "greenhouse": {
        "display": "the greenhouse",
        # novel schema: device (power/fill=fuel) + plant (thirst/mood=plant health).
        # Power drives the irrigation pump; fill is fuel/water reservoir.
        "schema": ["power", "fill", "thirst", "mood"],
        "profile": {
            "switch on": {"power": "up", "thirst": "up"},   # pump runs; waters plants
            "switch off": {"power": "down"},
            "fill": {"fill": "up"},
            "water": {"thirst": "up", "mood": "up"},         # manual watering
            "wait": {"thirst": "down", "fill": "down", "mood": "down"},
        },
        "cond": {
            "switch on": [({"fill": {"empty"}}, {"power": "up"})],  # runs dry: no water
        },
        "split_role": "far_ood",
        "derived_from": "lamp+fern",
        "similarity": "far: device+plant combo; power-driven irrigation coupling unseen in training",
        "world_version_min": 2,
    },
}


# ---------------------------------------------------------------------------
# world_version helpers: filter TYPE_LIBRARY by minimum world_version
# ---------------------------------------------------------------------------

def _type_min_version(name: str) -> int:
    """Return the minimum world_version a type requires (default 1 = always present)."""
    return TYPE_LIBRARY[name].get("world_version_min", 1)


def _types_for_world(world_version: int = 1) -> dict:
    """Return the subset of TYPE_LIBRARY active under `world_version`."""
    return {n: d for n, d in TYPE_LIBRARY.items()
            if _type_min_version(n) <= world_version}


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


def apply_action(type_name, state, action, rng=None, stochastic_v2: bool = False):
    """Oracle transition. Deterministic unless stochastic_v2=True and the type has a
    stochastic table for this action.

    `state` is {attr: value} restricted to the type's schema. Returns a NEW state dict.
    Effects on attributes outside the schema are ignored (defensive; profiles are authored
    to stay in-schema).

    When stochastic_v2=True and the type has `"stochastic": {action: [(p, effects), ...]}`
    the effects are sampled from that distribution. If rng is None we fall back to the
    deterministic effects (safe for replay when the branch was pre-recorded).
    """
    type_def = TYPE_LIBRARY[type_name]
    schema = set(type_def["schema"])

    # Stochastic branch: check for a stochastic table for this action.
    if stochastic_v2 and rng is not None:
        stoch_table = type_def.get("stochastic", {}).get(action)
        if stoch_table:
            probs = [p for p, _ in stoch_table]
            effects_list = [eff for _, eff in stoch_table]
            # Sample a branch
            total = sum(probs)
            r = rng.random() * total
            cum = 0.0
            effects = effects_list[-1]  # fallback to last
            for p, eff in zip(probs, effects_list):
                cum += p
                if r < cum:
                    effects = dict(eff)
                    break
        else:
            effects = _effects_for(type_def, state, action)
    else:
        effects = _effects_for(type_def, state, action)

    new = dict(state)
    for attr, direction in effects.items():
        if attr in schema:
            new[attr] = _shift(state[attr], attr, direction)
    return new


def _oracle_dist_for(type_name, state, action, stochastic_v2: bool = False):
    """Return the oracle next-state distribution as a list of {"text": rendered, "prob": p}.

    For deterministic transitions or when stochastic_v2=False, emits a single entry
    with prob=1.0.  When stochastic_v2=True and the type has a stochastic table,
    emits the full distribution (with each branch's rendered state approximated by
    just the entity's own state — the full rendering needs the other entities too,
    so we record the per-entity effects distribution instead).

    Note: full multi-entity state rendering would require the entire entities list;
    this function records the raw effects distribution so callers can construct
    oracle_dist at the chain level.
    """
    type_def = TYPE_LIBRARY[type_name]
    if stochastic_v2:
        stoch_table = type_def.get("stochastic", {}).get(action)
        if stoch_table:
            return [{"effects": dict(eff), "prob": float(p)}
                    for p, eff in stoch_table]
    # Deterministic: single branch, prob 1.0.
    effects = _effects_for(type_def, state, action)
    return [{"effects": dict(effects), "prob": 1.0}]


# =====================================================================================
# RENDERING  (state dict -> GLUCOSE-like text)
# =====================================================================================

def _capitalize_sentence(s):
    """Capitalize the first letter of a rendered clause (display nouns start with 'the')."""
    return s[:1].upper() + s[1:] if s else s


def _value_clause(ent_display, attr, value):
    cop = ATTR_COPULA[attr]
    return f"{ent_display} {cop} {value}"


def _value_clause_para(ent_display, attr, value, chain_template_seed, entity_idx):
    """Paraphrase one attribute clause (already capitalized, with trailing period).

    Template choice is a deterministic function of (chain_template_seed, entity_idx,
    attr) -- NOT of `value`. So within a chain the surrounding words are byte-stable as
    the value changes (masked-diff isolates the value word); across chains the seed
    varies so the same (attr, value) renders many ways. Variant 0 reproduces the
    canonical "{ent} {cop} {val}." form."""
    cop = ATTR_COPULA[attr]
    tmpl = _stable_choice(ATTR_TEMPLATES, chain_template_seed, entity_idx, attr)
    clause = tmpl.format(ent=ent_display, cop=cop, val=value)
    return _capitalize_sentence(clause)


def render_state(entities, surface_variety=False, chain_template_seed=0):
    """Render a multi-entity world state as 2-4 short sentences.

    `entities` is a list of (type_name, state_dict). We surface up to two salient
    attributes per entity (the first two in its schema) to keep states at 2-4 sentences.
    Every sentence starts capitalized.

    surface_variety=False reproduces the v4.2 campaign byte-for-byte. surface_variety=True
    paraphrases each clause and shuffles the ENTITY (clause-group) order via a per-(chain,
    entity) stable permutation key -- entity order stays fixed within the chain but varies
    across chains.
    """
    if not surface_variety:
        sentences = []
        for type_name, state in entities:
            disp = TYPE_LIBRARY[type_name]["display"]
            schema = TYPE_LIBRARY[type_name]["schema"]
            salient = schema[:2]
            clauses = [_value_clause(disp, attr, state[attr]) for attr in salient]
            # "The dog feel(s) hungry. The dog is messy."
            sentences.extend(_capitalize_sentence(c) + "." for c in clauses)
        return " ".join(sentences)

    # Paraphrase path. Render each entity's clause group, then order the GROUPS by a
    # stable per-entity sort key (clause-order shuffle, fixed within a chain).
    groups = []  # (sort_key, [sentence, ...])
    for entity_idx, (type_name, state) in enumerate(entities):
        disp = TYPE_LIBRARY[type_name]["display"]
        schema = TYPE_LIBRARY[type_name]["schema"]
        salient = schema[:2]
        clauses = [
            _value_clause_para(disp, attr, state[attr], chain_template_seed, entity_idx)
            for attr in salient
        ]
        # Stable order key: hash of (seed, "order", entity_idx) -> reproducible across
        # the chain's states (independent of value) so the entity order is held fixed.
        order_key = _stable_choice(list(range(10_000)), chain_template_seed, "order", entity_idx)
        groups.append(((order_key, entity_idx), clauses))
    groups.sort(key=lambda g: g[0])
    sentences = [s for _key, clauses in groups for s in clauses]
    return " ".join(sentences)


def render_action(entities, action, actor_idx,
                  surface_variety=False, chain_template_seed=0):
    """Render the action sentence (applied to entity `actor_idx`).

    surface_variety=False reproduces ACTION_TEMPLATES[action] exactly. surface_variety=True
    picks one of the >=4 action-sentence variants via a stable (chain, actor) key, so the
    SAME entity's action verbs share a framing style through the chain but vary across
    chains. Variant 0 is the canonical ACTION_TEMPLATES form."""
    type_name, _ = entities[actor_idx]
    disp = TYPE_LIBRARY[type_name]["display"]
    if not surface_variety:
        template = ACTION_TEMPLATES[action]
        return template.format(ent=disp)
    variants = ACTION_TEMPLATES_VAR[action]
    template = _stable_choice(variants, chain_template_seed, "action", actor_idx)
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


def generate_chain(rng, type_names, chain_len, wait_weight=0.15,
                   surface_variety=False):
    """Generate one chain over the given entity types.

    Returns (chain_texts, action_labels) where action_labels[i] is the action applied to
    go from state i to state i+1 (so len(action_labels) == len(chain_texts) - 1). Each
    label is "<action>@<entity_index>" so action-recovery eval can align actor + verb.
    """
    entities = [(tn, _random_state(rng, tn)) for tn in type_names]
    texts, actions, _, _ = _generate_chain_from_entities(
        rng, entities, chain_len, wait_weight, surface_variety=surface_variety)
    return texts, actions


def replay_chain(type_names, initial_states, action_labels,
                 surface_variety=False, chain_template_seed=0):
    """Oracle replay: given initial states + action labels, reproduce the state sequence.

    Returns list of entity-states snapshots (one per chain step, len == n_actions + 1).
    Used by the oracle-consistency test. The (surface_variety, chain_template_seed) args
    do NOT affect the returned STATE snapshots (the oracle is render-independent); they
    are accepted so a caller can re-render the snapshots with matching templates.
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

def _types_for_role(role, world_version: int = 1):
    """Return type names for `role` active under `world_version`."""
    return [name for name, d in TYPE_LIBRARY.items()
            if d["split_role"] == role and _type_min_version(name) <= world_version]


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

    v4 extension: when world_version==2, uses *_v2 chain_len and entities_per_chain from
    cfg.  When stochastic_v2=True, oracle_dist is added to each labeled step.
    """
    world_version = cfg.get("world_version", 1)
    stochastic_v2 = cfg.get("stochastic_v2", False) and world_version == 2

    if world_version == 2:
        len_min = cfg.get("chain_len_min_v2", cfg["chain_len_min"])
        len_max = cfg.get("chain_len_max_v2", cfg["chain_len_max"])
        ent_range = cfg.get("entities_per_chain_v2", cfg["entities_per_chain"])
    else:
        len_min = cfg["chain_len_min"]
        len_max = cfg["chain_len_max"]
        ent_range = cfg["entities_per_chain"]

    surface_variety = cfg.get("surface_variety", False)
    emit_lam_aug = cfg.get("emit_lam_aug", False)

    plain, labeled = [], []
    for _ in range(n_chains):
        chain_len = rng.randint(len_min, len_max)
        type_names = _sample_type_names(rng, allowed_types, ent_range)
        entities_init = [(tn, _random_state(rng, tn)) for tn in type_names]
        initial_states = [dict(st) for _, st in entities_init]
        texts, actions, oracle_dists, chain_aug = _generate_chain_from_entities(
            rng, entities_init, chain_len,
            cfg.get("wait_weight", 0.15),
            stochastic_v2=stochastic_v2,
            surface_variety=surface_variety,
            emit_lam_aug=emit_lam_aug,
        )
        plain_rec = {"chain": texts}
        if chain_aug is not None:
            plain_rec["chain_aug"] = chain_aug  # v6 §B second surface frame φ' (label-free text)
        plain.append(plain_rec)
        rec = {
            "chain": texts,
            "actions": actions,
            "types": type_names,
            "initial_states": initial_states,
        }
        # oracle_dist: present always when world_version==2 (degenerate when deterministic).
        if world_version == 2:
            rec["oracle_dist"] = oracle_dists
        labeled.append(rec)
    return plain, labeled


def _generate_chain_from_entities(rng, entities, chain_len, wait_weight=0.15,
                                   stochastic_v2: bool = False,
                                   surface_variety: bool = False,
                                   emit_lam_aug: bool = False):
    """Generate a chain starting from the given (mutable) entities list.

    Mirrors the body of ``generate_chain`` but accepts pre-constructed entities so
    ``build_split`` can capture the initial states before generation begins.

    Returns (chain_texts, action_labels, oracle_dists, chain_aug) where oracle_dists is a
    list of per-step oracle-distribution dicts (one per action, same length as
    action_labels). `chain_aug` is None unless emit_lam_aug is set (v6 §B), in which case it
    is a SECOND independent surface frame φ' of the SAME state sequence (a parallel list of
    the same length as chain_texts), rendered with a FRESH template seed and surface_variety
    forced on — the LAM surface-augmentation invariance second view.

    Mutates ``entities`` in place (same semantics as ``generate_chain``).

    surface_variety=False renders byte-identically to the v4.2 campaign. surface_variety=True
    draws a per-chain template seed (held fixed for ALL states in this chain) so paraphrase
    template choices are stable within the chain (masked-diff isolates the changed clause)
    but vary across chains. The seed draw is GUARDED behind surface_variety so the False-path
    RNG stream -- and thus the campaign output -- is byte-identical.
    """
    # Per-chain template seed: stable within the chain, varies across chains. Drawn ONLY
    # in the paraphrase path so the deterministic campaign RNG stream is untouched.
    chain_template_seed = rng.getrandbits(63) if surface_variety else 0
    # v6 §B: a FRESH, INDEPENDENT template seed for the φ' frame (always surface_variety=True).
    # Drawn only when emit_lam_aug so the non-aug RNG stream / output is byte-identical.
    aug_seed = rng.getrandbits(63) if emit_lam_aug else 0

    chain_texts = [render_state(entities, surface_variety, chain_template_seed)]
    chain_aug = [render_state(entities, True, aug_seed)] if emit_lam_aug else None
    action_labels = []
    oracle_dists = []
    for _ in range(chain_len - 1):
        actor_idx = rng.randrange(len(entities))
        type_name = entities[actor_idx][0]
        action = _sample_action(rng, type_name, wait_weight)
        # Collect oracle distribution BEFORE applying (uses current state).
        dist = _oracle_dist_for(type_name, entities[actor_idx][1], action, stochastic_v2)
        new_state = apply_action(type_name, entities[actor_idx][1], action, rng,
                                  stochastic_v2=stochastic_v2)
        entities[actor_idx] = (type_name, new_state)
        action_sentence = render_action(entities, action, actor_idx,
                                         surface_variety, chain_template_seed)
        state_sentence = render_state(entities, surface_variety, chain_template_seed)
        chain_texts.append(f"{action_sentence} {state_sentence}")
        if emit_lam_aug:
            aug_action = render_action(entities, action, actor_idx, True, aug_seed)
            aug_state = render_state(entities, True, aug_seed)
            chain_aug.append(f"{aug_action} {aug_state}")
        action_labels.append(f"{action}@{actor_idx}")
        oracle_dists.append({"type": type_name, "action": action, "dist": dist})
    return chain_texts, action_labels, oracle_dists, chain_aug


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
    world_version = cfg.get("world_version", 1)
    active_types = _types_for_world(world_version)
    types = {}
    for name, d in active_types.items():
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
            "train":         {"types": _types_for_role("train", world_version),    "discriminates": "in-distribution dynamics"},
            "test_iid":      {"types": _types_for_role("train", world_version),    "discriminates": "generalization to fresh states/actions, seen types"},
            "test_ood_near": {"types": _types_for_role("near_ood", world_version), "discriminates": "identity-space interpolation (small profile perturbation)"},
            "test_ood_far":  {"types": _types_for_role("far_ood", world_version),  "discriminates": "novel schema recombination + structurally novel profile"},
        },
        "types": types,
        "world_version": world_version,
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
# SURFACE-ENTROPY STATS
# =====================================================================================

def surface_entropy_stats(cfg, allowed_types, n_states=4000):
    """Measure distinct renderings per underlying state, OFF vs ON.

    Draws `n_states` random single-entity underlying states, then renders each one many
    times -- ONCE per off-path (template-free) and across many independent chain seeds for
    the on-path -- and reports the mean number of DISTINCT surface forms per underlying
    state. OFF is 1.0 by construction (one template); the ratio ON/OFF is the surface
    entropy gain the paraphrase mode buys.

    Returns a dict with the off/on distinct-rendering counts and the gain ratio.
    """
    import math

    rng = random.Random(cfg["seed"] + 777)
    n_seeds = 32  # independent chain-template seeds per underlying state
    off_distinct, on_distinct, on_entropy = [], [], []
    for _ in range(n_states):
        tn = rng.choice(allowed_types)
        state = _random_state(rng, tn)
        entities = [(tn, state)]
        off = {render_state(entities, surface_variety=False)}
        renders = []
        for _ in range(n_seeds):
            seed = rng.getrandbits(63)
            renders.append(render_state(entities, surface_variety=True,
                                        chain_template_seed=seed))
        uniq = set(renders)
        off_distinct.append(len(off))
        on_distinct.append(len(uniq))
        # Shannon entropy (bits) of the empirical rendering distribution at this state.
        counts = Counter(renders)
        tot = len(renders)
        ent = -sum((c / tot) * math.log2(c / tot) for c in counts.values())
        on_entropy.append(ent)
    off_mean = statistics.mean(off_distinct)
    on_mean = statistics.mean(on_distinct)
    return {
        "n_states_sampled": n_states,
        "n_seeds_per_state": n_seeds,
        "off_distinct_per_state": round(off_mean, 3),
        "on_distinct_per_state": round(on_mean, 3),
        "gain_ratio": round(on_mean / off_mean, 2) if off_mean else None,
        "on_mean_entropy_bits": round(statistics.mean(on_entropy), 3),
    }


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    cfg = CONFIG
    rng = random.Random(cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    world_version = cfg.get("world_version", 1)

    train_types = _types_for_role("train", world_version)
    near_types = _types_for_role("near_ood", world_version)
    far_types = _types_for_role("far_ood", world_version)

    print("Entity-World generator")
    print(f"  world_version={world_version}")
    print(f"  train types ({len(train_types)}) : {train_types}")
    print(f"  near-OOD    ({len(near_types)}) : {near_types}")
    print(f"  far-OOD     ({len(far_types)}) : {far_types}")
    print(f"  seed={cfg['seed']} stochastic={cfg.get('stochastic', False)} "
          f"stochastic_v2={cfg.get('stochastic_v2', False)} "
          f"surface_variety={cfg.get('surface_variety', False)}")
    print()

    # Determine train chain count (v2 may use n_train_chains_v2).
    if world_version == 2:
        n_train = cfg.get("n_train_chains_v2", cfg["n_train_chains"])
    else:
        n_train = cfg["n_train_chains"]

    # Build splits. Distinct rng streams keep splits reproducible & disjoint in states.
    train_plain, train_labeled = build_split(
        random.Random(cfg["seed"] + 1), train_types, n_train, cfg)
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

    # Surface-entropy stats (only meaningful when surface_variety is on; reported always
    # so the OFF baseline of 1.0 distinct rendering/state is visible).
    se = surface_entropy_stats(cfg, train_types)
    with open(out_dir / "surface_entropy.json", "w") as f:
        json.dump(se, f, indent=2)
    print("Surface entropy (distinct renderings per underlying state)")
    print(f"  off={se['off_distinct_per_state']}  on={se['on_distinct_per_state']}  "
          f"gain={se['gain_ratio']}x  mean_entropy={se['on_mean_entropy_bits']} bits "
          f"(n={se['n_states_sampled']} states x {se['n_seeds_per_state']} seeds)")
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
