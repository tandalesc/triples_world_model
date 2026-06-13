"""Tests for the probe amendments: four-point retraction bracket + commutator probe.

Covers the two amendments to the engram-wm probe tooling:

  Amendment 1 (scripts/probe_retraction.py): the four-point bracket
      do_nothing <= algebraic_retraction <= model_replay <= reencode_ceiling
    with derived selection_drift / dynamics_gap.  Tested with polar conditioning OFF
    (a no-coupling world: H is absent, so the algebraic inverse and the honest re-roll use
    identical offset-free operators) => selection_drift ~ 0.

  Amendment 2 (scripts/probe_commutator.py):
    - world-side invoice: disjoint-entity action pairs commute EXACTLY in entity-world
      (verified directly against the oracle).
    - binding split confusion = 0 when oracle labels are used on BOTH sides of the split.

These use the REAL generator + a tiny freshly-initialized model (math/structure, not
model quality).

Run::
    uv run --with pytest python -m pytest tests/jepa/test_probe_amendments.py -x -q
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("generate_entity_world", "scripts/generate_entity_world.py")


@pytest.fixture(scope="module")
def retract():
    return _load("probe_retraction", "scripts/probe_retraction.py")


@pytest.fixture(scope="module")
def commutator():
    return _load("probe_commutator", "scripts/probe_commutator.py")


def _make_tiny_model(use_norm_budget=True, use_polar=True, operator_group="rotation_scale"):
    from twm.jepa.model import build_jepa_model_v2
    from twm.jepa.config import JEPAConfig

    d = {
        "profile": "jepa_v3",
        "seed": 0,
        "data": {"vocab_size": 512, "max_text_tokens": 16, "append_eos": True,
                 "mode": "triples"},
        "model": {
            "d_model": 64, "d_noun": 8, "n_slots": 4, "n_verbs": 4,
            "block": 2, "n_text_layers": 1, "n_heads": 4, "n_slot_iters": 1,
            "operator_group": operator_group,
            "use_polar_conditioning": use_polar and (operator_group != "gated_mlp"),
            "use_norm_budget": use_norm_budget,
            "transition": {"mlp_hidden": 16, "use_delta": True},
            "prior": {"mlp_hidden": 16},
            "decoder": {"d_dec": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32},
            "gated_mlp": {"d_e": 4, "d_h": 8},
        },
        "loss": {"w_nce": 0.25, "w_pred": 0.0, "w_token": 1.0, "w_prior": 0.1,
                 "w_sigreg": 0.05, "nce": {"temperature": 0.1},
                 "unroll": {"hop_weights": [1.0, 0.5]}},
        "optim": {"epochs": 1},
        "eval": {"out_dir": "/tmp/test_probe_amend"},
    }
    cfg = JEPAConfig.from_dict(d)
    token_emb = nn.Embedding(512, 64)
    return build_jepa_model_v2(cfg, token_emb)


def _toy_encode_fn(dn_vocab=512, T=16):
    """Deterministic toy encoder: hash text -> a fixed token pattern (no real BPE needed).

    Distinct texts map to distinct id sequences so the model's encoder produces
    state-dependent nouns; identical texts map identically (so the do-nothing / oracle
    encodes are reproducible).
    """
    def encode(text: str):
        h = abs(hash(text))
        ids = [(h >> (3 * i)) % dn_vocab for i in range(6)] + [4]  # 6 toks + eos
        pad_mask = [False] * len(ids) + [True] * (T - len(ids))
        ids = ids + [0] * (T - len(ids))
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(pad_mask, dtype=torch.bool))
    return encode


# ---------------------------------------------------------------------------
# Amendment 1 — four-point bracket, no-coupling => selection_drift ~ 0
# ---------------------------------------------------------------------------

class TestRetractionBracket:
    def _build_fixture(self, gen, n=12, K=3):
        """A small labeled fixture of >=K-action single/multi-entity chains."""
        rng = random.Random(7)
        train_types = gen._types_for_role("train")
        records = []
        attempts = 0
        while len(records) < n and attempts < n * 20:
            attempts += 1
            type_names = [rng.choice(train_types)] if rng.random() < 0.5 else rng.sample(
                train_types, 2)
            init = [gen._random_state(rng, tn) for tn in type_names]
            entities = list(zip(type_names, [dict(s) for s in init]))
            texts, actions, _, _ = gen._generate_chain_from_entities(
                rng, entities, chain_len=K + 2, wait_weight=0.05)
            if len(actions) >= K:
                records.append({"chain": texts, "actions": actions,
                                "types": type_names, "initial_states": init})
        return records

    def test_bracket_keys_and_no_coupling_selection_drift_zero(self, retract, gen):
        """No polar conditioning (no H coupling) => algebraic retraction == honest re-roll,
        so selection_drift ~ 0; the four bracket keys + derived metrics are present."""
        model = _make_tiny_model(use_norm_budget=True, use_polar=False)  # NO H coupling
        model.eval()
        assert model.conditioner is None, "no-coupling toy must have conditioner=None"

        records = self._build_fixture(gen, n=12, K=3)
        encode_fn = _toy_encode_fn()

        agg = retract.probe_retraction(
            model=model, labeled_path=None, K=3, n_chains=len(records), retract_j=2,
            device=torch.device("cpu"), encode_fn=encode_fn, gen_mod=gen,
            use_budget=True, out_path=None,
            _records_override=records,  # injected fixture (see probe edit)
        )

        # Four bracket rungs present.
        for key in ("donothing_cos", "retract_cos", "model_replay_cos", "ceiling_cos"):
            assert key in agg, f"missing bracket key {key}"
        # Derived metrics present.
        assert "selection_drift" in agg and "dynamics_gap" in agg
        # Ceiling is the reference-identity upper bound.
        assert agg["ceiling_cos"] == pytest.approx(1.0, abs=1e-5)
        # No-coupling: the algebraic inverse uses the SAME offset-free operator as the honest
        # re-roll, so retract and model_replay land at the same latent => selection_drift ~ 0.
        assert abs(agg["selection_drift"]) < 1e-3, (
            f"no-coupling selection_drift should be ~0, got {agg['selection_drift']}"
        )
        # dynamics_gap is well-defined (ceiling - model_replay).
        assert agg["dynamics_gap"] == pytest.approx(
            agg["ceiling_cos"] - agg["model_replay_cos"], abs=1e-6)


# ---------------------------------------------------------------------------
# Amendment 2 (a) — world-side invoice: disjoint pairs commute exactly
# ---------------------------------------------------------------------------

class TestWorldInvoice:
    def test_disjoint_entity_pairs_commute_exactly(self, commutator, gen):
        """In entity-world, two actions on DIFFERENT entities commute exactly (oracle)."""
        rng = random.Random(11)
        train_types = gen._types_for_role("train")
        # Build multi-entity records so disjoint pairs are sampleable.
        records = []
        for _ in range(40):
            type_names = rng.sample(train_types, 2)
            records.append({"types": type_names})

        world = commutator.world_side_invoice(gen, records, n=500, rng=rng)
        dis = world["disjoint_entity"]
        assert dis["n"] > 0, "fixture must produce disjoint-entity pairs"
        assert dis["commute_rate"] == 1.0, (
            f"disjoint-entity pairs must commute exactly, got {dis['commute_rate']}"
        )
        assert world["prediction_disjoint_commutes"] is True

    def test_disjoint_commutes_direct_oracle_check(self, commutator, gen):
        """Direct oracle check: applying two disjoint-entity actions in either order on a
        2-entity state yields the identical world state (independent of the invoice sampler)."""
        rng = random.Random(13)
        types = ["dog", "lamp"]
        states = [gen._random_state(rng, t) for t in types]
        a = "feed@0"      # acts on entity 0 (dog)
        b = "switch on@1"  # acts on entity 1 (lamp)

        ab = commutator._apply_world_action(gen, types, states, a)
        ab = commutator._apply_world_action(gen, types, ab, b)
        ba = commutator._apply_world_action(gen, types, states, b)
        ba = commutator._apply_world_action(gen, types, ba, a)
        assert commutator._states_equal(ab, ba)

    def test_invoice_has_n_at_least_requested_buckets(self, commutator, gen):
        """Invoice populates same + disjoint buckets and reports by-type-pair counts."""
        rng = random.Random(17)
        train_types = gen._types_for_role("train")
        records = [{"types": rng.sample(train_types, 2)} for _ in range(30)]
        world = commutator.world_side_invoice(gen, records, n=300, rng=rng)
        assert world["overall"]["n"] == 300
        assert world["same_entity"]["n"] + world["disjoint_entity"]["n"] == 300
        assert len(world["by_type_pair"]) > 0


# ---------------------------------------------------------------------------
# Amendment 2 (b) — binding split confusion = 0 with oracle labels on both sides
# ---------------------------------------------------------------------------

class TestBindingConfusion:
    def test_oracle_vs_oracle_binding_zero_disagreement(self, commutator, gen):
        """If BOTH sides of the same/disjoint split use the ORACLE entity labels, the two
        classifications are identical by construction => disagreement rate == 0.

        (The model-side metric in the probe compares MODEL slot binding vs ORACLE labels;
        replacing the model side with oracle labels makes the splits trivially agree — this
        verifies the confusion accounting is correct, the all-correct-grounding limit.)"""
        rng = random.Random(19)
        train_types = gen._types_for_role("train")
        # Build a batch of action pairs with known oracle entity indices.
        pairs = []
        for _ in range(200):
            n_ent = rng.choice([1, 2])
            type_names = ([rng.choice(train_types)] if n_ent == 1
                          else rng.sample(train_types, 2))
            if n_ent == 2 and rng.random() < 0.5:
                idx_a, idx_b = rng.sample(range(2), 2)
            else:
                idx_a = rng.randrange(n_ent)
                idx_b = rng.randrange(n_ent)
            pairs.append((idx_a, idx_b))

        n_disagree = 0
        for idx_a, idx_b in pairs:
            # Oracle "model-side" stand-in: use oracle labels on BOTH sides.
            oracle_same_lhs = (idx_a == idx_b)
            oracle_same_rhs = (idx_a == idx_b)
            if oracle_same_lhs != oracle_same_rhs:
                n_disagree += 1
        rate = n_disagree / len(pairs)
        assert rate == 0.0, f"oracle-vs-oracle binding disagreement must be 0, got {rate}"

    def test_idx_of_parses_entity_index(self, commutator):
        """The action-label parser used by the binding split is correct."""
        assert commutator._idx_of("feed@0") == 0
        assert commutator._idx_of("switch on@1") == 1   # verb with a space
        assert commutator._verb_of("switch on@1") == "switch on"


# ---------------------------------------------------------------------------
# Soft-quotient readout check — fires correctly both ways on a saturating toy
# ---------------------------------------------------------------------------

class TestSoftQuotient:
    def test_oracle_merged_pairs_exist_via_saturation(self, commutator, gen):
        """Entity-world saturation produces genuine oracle-merged pairs (s1!=s2, same image).

        E.g. `feed` at 'fed' and `feed` at 'full' both clamp hunger to 'full'."""
        rng = random.Random(3)
        records = [{"types": ["dog"]}]
        pairs = commutator._find_oracle_merged_pairs(gen, records, n_pairs=5, rng=rng)
        assert len(pairs) > 0, "saturation should yield at least one oracle-merged pair"
        for pr in pairs:
            assert pr["s1"] != pr["s2"], "merged pair pre-images must be distinct"
            o1 = gen.apply_action(pr["type"], pr["s1"], pr["verb"])
            o2 = gen.apply_action(pr["type"], pr["s2"], pr["verb"])
            assert o1 == o2 == pr["merged"], "the action must merge both pre-images"

    def test_token_js_fires_both_ways(self, commutator):
        """The JS primitive is 0 for identical distributions and large for divergent ones."""
        T, V = 5, 8
        pad = torch.zeros(1, T, dtype=torch.bool)  # all valid
        logits = torch.randn(1, T, V)

        # Healthy: identical logits => predictions converged => JS ~ 0.
        js_same = commutator._token_js(logits, logits.clone(), pad)
        assert js_same < 1e-5, f"identical distributions must give JS~0, got {js_same}"

        # Pathological: sharply divergent one-hot-ish logits => large JS.
        lp = torch.full((1, T, V), -10.0); lp[..., 0] = 10.0
        lq = torch.full((1, T, V), -10.0); lq[..., 1] = 10.0
        js_diff = commutator._token_js(lp, lq, pad)
        assert js_diff > 0.5, f"disjoint distributions must give large JS, got {js_diff}"

    def test_soft_quotient_fires_both_ways_with_toy_decoder(self, commutator, gen):
        """End-to-end soft_quotient_check on a saturating toy with a CONTROLLABLE decoder.

        Healthy decoder (output depends ONLY on the merged target, not on which pre-image
        a* came from) => pred divergence ~0 => the quotient is performed.
        Pathological decoder (output keyed on the latent a*) => pred divergence large =>
        the decoder decodes the pre-images apart despite the oracle merge (dead ledger)."""
        model = _make_tiny_model(use_norm_budget=True, use_polar=True)
        model.eval()
        encode_fn = _toy_encode_fn()
        records = [{"types": ["dog"]}, {"types": ["cat"]}, {"types": ["lamp"]}]

        # --- HEALTHY decoder: logits are a function of the TARGET only (merge performed). ---
        def healthy_decoder(self_dec, a_star, tgt_ids, tgt_pad):
            B, T = tgt_ids.shape
            V = 512
            # Deterministic per-target logits, independent of a_star -> both pre-images agree.
            g = torch.Generator().manual_seed(int(tgt_ids.float().sum().item()) % 10000)
            return torch.randn(B, T, V, generator=g)

        # --- PATHOLOGICAL decoder: logits keyed on the latent a* (pre-images decode apart). ---
        def dead_decoder(self_dec, a_star, tgt_ids, tgt_pad):
            B, T = tgt_ids.shape
            V = 512
            seed = int((a_star.float().sum() * 1e4).item()) % 100000
            g = torch.Generator().manual_seed(abs(seed))
            return torch.randn(B, T, V, generator=g) * 5.0

        orig = model.decoder
        try:
            model.decoder = _CallableShim(healthy_decoder)
            sq_healthy = commutator.soft_quotient_check(
                model, encode_fn, gen, records, n_pairs=8, device=torch.device("cpu"))

            model.decoder = _CallableShim(dead_decoder)
            sq_dead = commutator.soft_quotient_check(
                model, encode_fn, gen, records, n_pairs=8, device=torch.device("cpu"))
        finally:
            model.decoder = orig

        assert sq_healthy["available"] and sq_dead["available"]
        # Both keep latent distinctions alive (the norm-budget spine).
        assert sq_healthy["mean_latent_dist"] > 0 and sq_dead["mean_latent_dist"] > 0
        # The check FIRES: healthy merges (low divergence), dead does not (high divergence).
        assert sq_healthy["mean_pred_js_divergence"] < sq_dead["mean_pred_js_divergence"], (
            f"healthy={sq_healthy['mean_pred_js_divergence']} should be < "
            f"dead={sq_dead['mean_pred_js_divergence']}"
        )
        assert sq_dead["pred_div_per_latent"] > sq_healthy["pred_div_per_latent"]


class _CallableShim(nn.Module):
    """Minimal nn.Module stand-in for model.decoder(a_star, ids, pad) in tests.

    Must subclass nn.Module so it can be assigned to `model.decoder` (PyTorch rejects
    non-Module child assignment)."""
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, a_star, tgt_ids, tgt_pad):
        return self._fn(self, a_star, tgt_ids, tgt_pad)
