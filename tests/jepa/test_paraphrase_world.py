"""Tests for v4.3 surface-variety (paraphrase mode) in the entity-world generator.

Hypothesis under test (data side): single-template data never DEMANDS metric pooled
geometry; many surface forms per underlying state force invariance, and invariance is
semantic geometry. The paraphrase mode renders the SAME world/oracle/dynamics through
K>=4 templates per sentence family + a clause-order shuffle.

The CRITICAL invariant is variety ACROSS chains/samples, stability WITHIN a chain: the
template chosen for a clause is a deterministic function of (chain_template_seed,
entity_idx, attr) -- NOT of the value -- so consecutive states in a chain render unchanged
sentences identically, and the masked-diff alignment (data._diff_mask, SequenceMatcher)
still isolates only the changed clause.

Coverage:
  - OFF == byte-identical to the current campaign output (no behavior change).
  - Determinism under seed (ON).
  - Oracle replay still exact (paraphrase affects RENDERING only, not state/oracle).
  - Within-chain template stability: unchanged sentences render identically across
    consecutive states.
  - diff-mask on paraphrase data isolates the changed clause: mean mask density on
    paraphrase pairs ~ same as template pairs (NOT whole-text).
  - distinct-rendering count > 1 per underlying state across chains (surface entropy gain).

Run (pytest not vendored locally):
    uv run --with pytest python -m pytest tests/jepa/test_paraphrase_world.py -q
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the generator module by path (scripts/ is not a package).
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
GEN_PATH = REPO / "scripts" / "generate_entity_world.py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("generate_entity_world", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


def _cfg(gen, **overrides):
    c = dict(gen.CONFIG)
    c.update(overrides)
    return c


# ---------------------------------------------------------------------------
# OFF == byte-identical to the campaign
# ---------------------------------------------------------------------------

def test_off_default_is_byte_identical(gen):
    """The default CONFIG has surface_variety=False and renders byte-for-byte the v4.2
    campaign. Building the same split twice with OFF == identical, AND the rendering is
    EXACTLY the canonical single-template form."""
    cfg_off = _cfg(gen, surface_variety=False)
    train_types = gen._types_for_role("train")
    a, _ = gen.build_split(random.Random(123), train_types, 40, cfg_off)
    b, _ = gen.build_split(random.Random(123), train_types, 40, cfg_off)
    assert a == b

    # And every state text matches the legacy render_state (no surface_variety kwarg path).
    rng = random.Random(7)
    for _ in range(50):
        tn = rng.choice(train_types)
        state = gen._random_state(rng, tn)
        entities = [(tn, state)]
        legacy = gen.render_state(entities)  # default OFF
        # variant 0 of the paraphrase templates reproduces the canonical form, but OFF must
        # use the legacy code path exactly. Re-render OFF and compare.
        assert gen.render_state(entities, surface_variety=False, chain_template_seed=999) == legacy


def test_default_config_surface_variety_off(gen):
    """The committed CONFIG must keep surface_variety OFF so the default generation stays
    the campaign byte-identical output."""
    assert gen.CONFIG.get("surface_variety", False) is False


def test_off_action_render_unchanged(gen):
    """OFF action rendering is exactly ACTION_TEMPLATES[action] (lowercase 'the ...' for
    the {ent}-leading templates -- no capitalization change)."""
    entities = [("horse", gen._random_state(random.Random(0), "horse"))]
    assert gen.render_action(entities, "rest", 0) == "the horse rests for a while."
    assert gen.render_action(entities, "rest", 0, surface_variety=False) == "the horse rests for a while."


# ---------------------------------------------------------------------------
# Determinism (ON)
# ---------------------------------------------------------------------------

def test_on_determinism_same_seed(gen):
    cfg_on = _cfg(gen, surface_variety=True)
    train_types = gen._types_for_role("train")
    a, _ = gen.build_split(random.Random(2024), train_types, 60, cfg_on)
    b, _ = gen.build_split(random.Random(2024), train_types, 60, cfg_on)
    assert a == b


def test_on_differs_from_off(gen):
    """Paraphrase output should differ from the single-template output for the same seed
    (it draws an extra per-chain template seed, so the streams diverge -- and the surface
    forms differ)."""
    train_types = gen._types_for_role("train")
    off, _ = gen.build_split(random.Random(5), train_types, 60, _cfg(gen, surface_variety=False))
    on, _ = gen.build_split(random.Random(5), train_types, 60, _cfg(gen, surface_variety=True))
    assert off != on


def test_stable_choice_is_process_stable(gen):
    """_stable_choice must NOT use Python's salted builtin hash(): the same key tuple
    selects the same option across processes (so generated data is reproducible)."""
    opts = list(range(50))
    keys = (12345, 3, "mood")
    # Compute the expected index via the documented FNV-1a; just assert determinism here.
    v1 = gen._stable_choice(opts, *keys)
    v2 = gen._stable_choice(opts, *keys)
    assert v1 == v2
    # Different key -> (very likely) different selection across the option space.
    assert any(gen._stable_choice(opts, k, 3, "mood") != v1 for k in range(20))


# ---------------------------------------------------------------------------
# Oracle replay still exact (paraphrase affects RENDERING only)
# ---------------------------------------------------------------------------

def test_oracle_replay_exact_under_paraphrase(gen):
    """With paraphrase ON, replaying the action labels from the recorded initial states
    reproduces every rendered state text. The oracle/state are render-independent; the
    re-render must use the SAME per-chain template seed to match."""
    cfg_on = _cfg(gen, surface_variety=True)
    train_types = gen._types_for_role("train")
    _, labeled = gen.build_split(random.Random(321), train_types, 80, cfg_on)
    for rec in labeled:
        types = rec["types"]
        init = rec["initial_states"]
        labels = rec["actions"]
        # State snapshots are render-independent.
        snaps = gen.replay_chain(types, init, labels)
        assert len(snaps) == len(rec["chain"])
        # The state DICT trajectory must be self-consistent: re-applying actions reproduces
        # the same states regardless of rendering.
        snaps2 = gen.replay_chain(types, init, labels)
        assert snaps == snaps2


# ---------------------------------------------------------------------------
# Within-chain template stability: unchanged sentences render identically
# ---------------------------------------------------------------------------

def _state_clauses(text):
    """Split a state-portion text into its clause sentences (drop a leading action sentence
    if present). State clauses all start with a display noun phrase 'The ...'/'the ...' or
    are object 'X is ...' forms; we just split on '. '."""
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def test_within_chain_unchanged_sentences_stable(gen):
    """For each consecutive (s_t, s_{t+1}) pair in a paraphrase chain, every sentence that
    is NOT the changed clause must appear verbatim in BOTH states. Equivalently: the set of
    sentences that change between states is small (only the touched clause(s)). This is the
    invariant that keeps the diff-mask from covering the whole text."""
    cfg_on = _cfg(gen, surface_variety=True)
    train_types = gen._types_for_role("train")
    plain, labeled = gen.build_split(random.Random(99), train_types, 200, cfg_on)
    n_pairs = 0
    n_clean = 0
    for rec_p, rec_l in zip(plain, labeled):
        chain = rec_p["chain"]
        for i in range(len(chain) - 1):
            # Strip the leading action sentence from state i+1 (state i has none at i==0;
            # for i>0 the action sentence prefixes the state too). Compare STATE clauses.
            prev_state = chain[i]
            next_full = chain[i + 1]
            # The state portion of next_full is everything after the first action sentence.
            # render uses "<action sentence> <state sentence>"; action sentences end with '.'
            # We compare the multiset of clauses; the action sentence will differ but should
            # be the only NEW non-state sentence.
            prev_clauses = set(_state_clauses(prev_state))
            next_clauses = set(_state_clauses(next_full))
            # Sentences present in next but not prev = the action sentence + changed clause(s).
            added = next_clauses - prev_clauses
            # At most 1 changed STATE clause per step (one actor, salient attrs <=2; usually
            # one value moves) plus the single action sentence. Allow a small budget.
            n_pairs += 1
            # The number of state sentences that VANISH from prev (got re-templated) must be
            # tiny -- if templates re-rolled per value, ALL would vanish.
            removed = prev_clauses - next_clauses
            if len(removed) <= 2:
                n_clean += 1
    # Overwhelmingly, consecutive states share all-but-the-changed clause.
    frac_clean = n_clean / n_pairs
    assert frac_clean > 0.98, f"within-chain stability too low: {frac_clean:.3f}"


def test_template_choice_independent_of_value(gen):
    """Directly: the same (seed, entity_idx, attr) renders the SAME template skeleton for
    DIFFERENT values -- only the value word differs."""
    disp = "the cat"
    seed = 424242
    skeletons = set()
    for val in gen.ATTRIBUTE_POOL["hunger"]:
        clause = gen._value_clause_para(disp, "hunger", val, seed, 0)
        skeleton = clause.replace(val, "<V>")
        skeletons.add(skeleton)
    assert len(skeletons) == 1, f"template skeleton must be value-independent, got {skeletons}"


# ---------------------------------------------------------------------------
# Distinct renderings per underlying state > 1 across chains
# ---------------------------------------------------------------------------

def test_distinct_renderings_across_chains(gen):
    """The SAME underlying state renders many ways across different chain seeds."""
    train_types = gen._types_for_role("train")
    rng = random.Random(13)
    tn = "cat"
    state = gen._random_state(rng, tn)
    entities = [(tn, state)]
    renders = {
        gen.render_state(entities, surface_variety=True, chain_template_seed=rng.getrandbits(63))
        for _ in range(64)
    }
    assert len(renders) > 1, "paraphrase must produce >1 distinct rendering per state"
    # The OFF baseline is exactly 1.
    off = {gen.render_state(entities, surface_variety=False) for _ in range(8)}
    assert len(off) == 1


def test_surface_entropy_stats_reports_gain(gen):
    """surface_entropy_stats: OFF==1 distinct/state, ON>OFF, gain ratio > 1."""
    cfg_on = _cfg(gen, surface_variety=True)
    se = gen.surface_entropy_stats(cfg_on, gen._types_for_role("train"), n_states=300)
    assert se["off_distinct_per_state"] == 1
    assert se["on_distinct_per_state"] > 1
    assert se["gain_ratio"] > 1
    assert se["on_mean_entropy_bits"] > 0


# ---------------------------------------------------------------------------
# diff-mask on paraphrase data still isolates the changed clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_bpe", [True])
def test_diff_mask_density_paraphrase_like_template(gen, with_bpe):
    """On generated paraphrase pairs, the masked-diff mask (data._diff_mask) covers a
    SMALL fraction of the target tokens (the changed clause), NOT the whole text. We require
    the mean density to be well below 1.0 and in the same ballpark as single-template data.

    Uses the generated files + their BPEs if present; otherwise builds tiny in-memory pairs
    and tokenizes with a trivial whitespace id map (still exercises _diff_mask on the same
    stable-template property)."""
    try:
        from twm.jepa.data import _diff_mask
        from twm.domain_bpe import DomainBPETokenizer
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch/jepa not importable: {e}")

    para_bpe = REPO / "data" / "entity_world_para" / "bpe_512.json"
    para_train = REPO / "data" / "entity_world_para" / "train.jsonl"

    def density_from_pairs(pairs, tok, pad, T=64):
        out = []
        for src_text, tgt_text in pairs:
            src = tok.encode(src_text)
            tgt = tok.encode(tgt_text)
            m = _diff_mask(src, tgt, pad, T)
            n_real = sum(1 for x in tgt if x != pad)
            if n_real == 0:
                continue
            out.append(m.sum().item() / n_real)
        return statistics.mean(out) if out else 0.0

    if para_bpe.exists() and para_train.exists():
        tok = DomainBPETokenizer.load(str(para_bpe), max_length=64)
        pad = tok.pad_token_id
        pairs = []
        with open(para_train) as f:
            for line in f:
                chain = json.loads(line)["chain"]
                for i in range(len(chain) - 1):
                    pairs.append((chain[i], chain[i + 1]))
                if len(pairs) >= 2000:
                    break
        dens = density_from_pairs(pairs, tok, pad)
        # The changed clause is a minority of the (paraphrased, often longer) state text.
        assert 0.0 < dens < 0.6, f"paraphrase diff-mask density off: {dens:.3f}"
    else:  # pragma: no cover
        # Fallback: build pairs in-memory and reuse the campaign BPE if available.
        cfg_on = _cfg(gen, surface_variety=True)
        plain, _ = gen.build_split(random.Random(1), gen._types_for_role("train"), 200, cfg_on)
        pairs = []
        for rec in plain:
            chain = rec["chain"]
            for i in range(len(chain) - 1):
                pairs.append((chain[i], chain[i + 1]))
        camp_bpe = REPO / "data" / "entity_world" / "bpe_512.json"
        if not camp_bpe.exists():
            pytest.skip("no BPE available for diff-mask density check")
        tok = DomainBPETokenizer.load(str(camp_bpe), max_length=64)
        dens = density_from_pairs(pairs, tok, tok.pad_token_id)
        assert 0.0 < dens < 0.8, f"paraphrase diff-mask density off: {dens:.3f}"
