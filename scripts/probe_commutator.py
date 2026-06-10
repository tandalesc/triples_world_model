#!/usr/bin/env python3
"""Commutator probe — the NON-COMMUTATIVITY invoice (world-side ground truth + model-side).

Engram-wm companion to the retraction probe (jepa_entity_campaign.md §4).  Two halves:

(a) WORLD-SIDE INVOICE (oracle only, NO model)
    The ground-truth non-commutativity of the entity world.  Sample N (state, action_a,
    action_b) triples and compare oracle(a, b, s) vs oracle(b, a, s) — does applying the two
    actions in the opposite order land in the same world state?  Reported as an exact-match
    COMMUTATION RATE plus the mean state-distance when the orderings differ, SPLIT by
        - same-entity vs disjoint-entity action pairs (the two actions target the same /
          different entity index), and
        - action-TYPE pair (which (verb_a, verb_b) unordered pair).

    PRE-REGISTERED PREDICTION (the reason this split exists):
        * DISJOINT-entity pairs commute EXACTLY (~100%): two actions on different entities
          touch disjoint slices of the world state, so order cannot matter.
        * SAME-entity pairs are PARTIALLY non-commuting: ordinal-ladder SATURATION (a value
          clamped at the best/worst index) and state-dependent CONDITIONAL effects make the
          composition order-sensitive.
    This is the pre-registered prediction check — the disjoint/same split IS the experiment.

(b) MODEL-SIDE
    For a trained checkpoint, the operator matrices B_a, B_b are abelian (they commute by
    construction — §operator.py).  But the polar conditioner H recomputes a per-slot phase
    offset from the CURRENT modulus at each hop, so the SELECTION (which effective angle the
    operator applies) can be path-dependent even though the matrices commute.  We measure:
        * pure-matrix defect  : ||B_a(B_b(z)) - B_b(B_a(z))|| with theta_offset DISABLED —
                                the abelian sanity, should be ~0.
        * selection defect    : the SAME with H offsets RECOMPUTED PER ORDERING — the
                                selection-inclusive defect (non-zero iff H is path-dependent).
    Each action's target is classified same/disjoint TWICE:
        * by the MODEL's slot-attention assignment (which slot's noun the action moves most),
        * by the ORACLE entity labels (the ground-truth entity_idx of the action).
    The confusion between the two splits is reported as `binding_disagreement_rate` — a FREE
    grounding-quality metric (how often the model binds an action to the wrong entity slot).

GRACEFUL DEGRADATION: the world-side invoice always runs (no model needed).  If no trained
entity checkpoint is found, the model-side falls back to results/jepa_ent_smoke/model_latest.pt
with `undertrained_caveat: true` flagged in the output JSON (the smoke model's geometry is
collapsed, so the model-side numbers are a smoke test of the TOOLING, not a real measurement).

Usage::

    uv run python scripts/probe_commutator.py \\
        --labeled data/entity_world/test_iid_labeled.jsonl \\
        --n_world 2000 --n_model 512 \\
        --ckpt results/jepa_ent_s0/model_latest.pt \\
        --config configs/jepa/jepa_ent_s0.json \\
        --out results/jepa_ent_s0/commutator.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------------------
# Generator + tokenizer loaders (shared shape with probe_retraction.py)
# ---------------------------------------------------------------------------

def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "generate_entity_world", REPO / "scripts" / "generate_entity_world.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tokenizer(tokenizer_path: str, max_text_tokens: int, append_eos: bool = True):
    """BPE encode(text) -> (ids_tensor, pad_tensor), matching the dataset's _encode."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    T = max_text_tokens
    eos_id = 4

    def encode(text: str):
        ids = tok.encode(text).ids
        if append_eos:
            ids = ids + [eos_id]
        if len(ids) > T:
            ids = ids[:T]
        pad_mask = [False] * len(ids) + [True] * (T - len(ids))
        ids_padded = ids + [0] * (T - len(ids))
        return (torch.tensor(ids_padded, dtype=torch.long),
                torch.tensor(pad_mask, dtype=torch.bool))

    return encode


# ===========================================================================
# (a) WORLD-SIDE INVOICE — oracle-only ground-truth non-commutativity
# ===========================================================================

def _verb_of(action_label: str) -> str:
    """'<verb>@<idx>' -> '<verb>'."""
    return action_label.rsplit("@", 1)[0]


def _idx_of(action_label: str) -> int:
    """'<verb>@<idx>' -> <idx> (entity index within the chain's entity list)."""
    return int(action_label.rsplit("@", 1)[1])


def _apply_world_action(gen_mod, types, states, action_label):
    """Apply one '<verb>@<idx>' action to a multi-entity world state (list of dicts).

    Returns a NEW list of per-entity state dicts (oracle deterministic transition).
    """
    verb = _verb_of(action_label)
    idx = _idx_of(action_label)
    new_states = [dict(s) for s in states]
    new_states[idx] = gen_mod.apply_action(types[idx], states[idx], verb)
    return new_states


def _states_equal(a, b) -> bool:
    """Exact equality of two multi-entity world states (list of dicts)."""
    if len(a) != len(b):
        return False
    return all(a[i] == b[i] for i in range(len(a)))


def _state_distance(gen_mod, types, a, b) -> int:
    """Ordinal L1 distance between two multi-entity states (sum over attrs of |Δidx|)."""
    dist = 0
    for i in range(len(a)):
        ladders = gen_mod.ATTRIBUTE_POOL
        for attr in a[i]:
            va, vb = a[i][attr], b[i].get(attr)
            if vb is None:
                continue
            ladder = ladders[attr]
            dist += abs(ladder.index(va) - ladder.index(vb))
    return dist


def _sample_world_triples(gen_mod, records, n, rng):
    """Sample n (types, state, action_a, action_b) triples from the labeled split.

    Prefers MULTI-entity chains so disjoint-entity action pairs are well represented; for
    single-entity chains both actions necessarily target entity 0 (same-entity).  Each
    action is drawn from the world's applicable-action vocabulary for its target entity.
    """
    triples = []
    multi = [r for r in records if len(r.get("types", [])) >= 2]
    # Mix: half from multi-entity chains (gives disjoint pairs), half from any chain.
    pools = [multi, records] if multi else [records]
    attempts = 0
    while len(triples) < n and attempts < n * 50:
        attempts += 1
        pool = pools[len(triples) % len(pools)] if multi else records
        if not pool:
            break
        r = rng.choice(pool)
        types = r["types"]
        n_ent = len(types)
        # A fresh random world state (not tied to a chain step — we want broad coverage).
        states = [gen_mod._random_state(rng, tn) for tn in types]
        # Sample two actions.  Bias toward distinct entities when possible so both
        # same-entity and disjoint-entity classes are populated.
        if n_ent >= 2 and rng.random() < 0.5:
            idx_a, idx_b = rng.sample(range(n_ent), 2)
        else:
            idx_a = rng.randrange(n_ent)
            idx_b = rng.randrange(n_ent)
        verb_a = gen_mod._sample_action(rng, types[idx_a], wait_weight=0.0)
        verb_b = gen_mod._sample_action(rng, types[idx_b], wait_weight=0.0)
        triples.append({
            "types": types,
            "states": states,
            "action_a": f"{verb_a}@{idx_a}",
            "action_b": f"{verb_b}@{idx_b}",
        })
    return triples


def world_side_invoice(gen_mod, records, n, rng):
    """Ground-truth non-commutativity invoice (oracle only)."""
    triples = _sample_world_triples(gen_mod, records, n, rng)

    overall = {"n": 0, "commute": 0, "dist_when_differ": 0, "n_differ": 0}
    by_binding = {"same": dict(overall), "disjoint": dict(overall)}
    by_type_pair = defaultdict(lambda: {"n": 0, "commute": 0})

    for t in triples:
        types, states = t["types"], t["states"]
        a, b = t["action_a"], t["action_b"]

        # oracle(a, b, s): apply a then b.
        ab = _apply_world_action(gen_mod, types, states, a)
        ab = _apply_world_action(gen_mod, types, ab, b)
        # oracle(b, a, s): apply b then a.
        ba = _apply_world_action(gen_mod, types, states, b)
        ba = _apply_world_action(gen_mod, types, ba, a)

        commutes = _states_equal(ab, ba)
        binding = "same" if _idx_of(a) == _idx_of(b) else "disjoint"
        type_pair = tuple(sorted((_verb_of(a), _verb_of(b))))

        for bucket in (overall, by_binding[binding]):
            bucket["n"] += 1
            if commutes:
                bucket["commute"] += 1
            else:
                bucket["n_differ"] += 1
                bucket["dist_when_differ"] += _state_distance(gen_mod, types, ab, ba)

        tp = by_type_pair[type_pair]
        tp["n"] += 1
        if commutes:
            tp["commute"] += 1

    def _finalize(bucket):
        n_ = bucket["n"]
        return {
            "n": n_,
            "commute_rate": (bucket["commute"] / n_) if n_ else None,
            "mean_dist_when_differ": (
                bucket["dist_when_differ"] / bucket["n_differ"]
                if bucket["n_differ"] else 0.0
            ),
            "n_differ": bucket["n_differ"],
        }

    type_pair_out = {
        f"{tp[0]}|{tp[1]}": {
            "n": d["n"],
            "commute_rate": d["commute"] / d["n"] if d["n"] else None,
        }
        for tp, d in sorted(by_type_pair.items())
    }

    return {
        "overall": _finalize(overall),
        "same_entity": _finalize(by_binding["same"]),
        "disjoint_entity": _finalize(by_binding["disjoint"]),
        "by_type_pair": type_pair_out,
        # The pre-registered prediction: disjoint should be ~1.0, same < 1.0.
        "prediction_disjoint_commutes": (
            by_binding["disjoint"]["n"] > 0
            and by_binding["disjoint"]["commute"] == by_binding["disjoint"]["n"]
        ),
        "prediction_same_partial_noncommute": (
            by_binding["same"]["n"] > 0
            and by_binding["same"]["commute"] < by_binding["same"]["n"]
        ),
    }


# ===========================================================================
# (b) MODEL-SIDE — operator commutator + binding-disagreement grounding metric
# ===========================================================================

def _encode_state(model, encode_fn, gen_mod, types, states, device):
    """Render a multi-entity world state to text, encode -> nouns k (1, M, dn)."""
    text = gen_mod.render_state(list(zip(types, states)))
    ids, pad = encode_fn(text)
    with torch.no_grad():
        _, k, _ = model.encoder(ids.unsqueeze(0).to(device), pad.unsqueeze(0).to(device))
    return k


def _posterior_verb_idx(model, encode_fn, gen_mod, types, states, action_label, device):
    """Infer the MODEL's latent verb index for an oracle action via the posterior.

    Build the (state, oracle-next-state) pair the action induces and read the hard-argmax
    latent action q(v | s, s') — this grounds the abstract model verb code in the action's
    actual world semantics.
    """
    next_states = _apply_world_action(gen_mod, types, states, action_label)
    src_text = gen_mod.render_state(list(zip(types, states)))
    tgt_text = gen_mod.render_state(list(zip(types, next_states)))
    s_ids, s_pad = encode_fn(src_text)
    t_ids, t_pad = encode_fn(tgt_text)
    with torch.no_grad():
        v_onehot, _, _ = model.transition(
            s_ids.unsqueeze(0).to(device), s_pad.unsqueeze(0).to(device),
            t_ids.unsqueeze(0).to(device), t_pad.unsqueeze(0).to(device),
            tau=1.0, hard=True,
        )
    return int(v_onehot.argmax(dim=-1).item())


def _apply_verb(model, k, verb_idx, device, use_offset):
    """Apply operator verb `verb_idx` to all M slots of k.

    use_offset=True recomputes the polar conditioner H on k (selection-inclusive path);
    use_offset=False passes theta_offset=None (pure abelian-matrix path).  norm_budget is
    OFF here (we measure the geometric commutator of the noun map; the scale accumulator is
    a separate per-slot scalar that commutes trivially under addition).
    """
    B, M, _ = k.shape
    v_slots = k.new_full((B, M), int(verb_idx), dtype=torch.long)
    theta_offset = None
    if use_offset and getattr(model, "conditioner", None) is not None:
        theta_offset = model.conditioner(k)
    with torch.no_grad():
        out = model.operator.apply(k, v_slots, theta_offset=theta_offset)
    # operator.apply returns a bare tensor when norm_budget is off (the path we take here).
    return out


def _model_binding_slot(model, k, verb_idx, device):
    """Which slot does this action MOVE most?  The model's slot-attention assignment of the
    action's target = argmax over slots of the per-slot noun displacement norm.

    The operator applies the verb to all M slots, but the polar conditioner H gives each slot
    a state-dependent effective angle, so the slot the model has BOUND to the acting entity
    moves most.  Pure model-internal — no oracle labels.
    """
    a = _apply_verb(model, k, verb_idx, device, use_offset=True)
    delta = (a.float() - k.float()).norm(dim=-1)  # (1, M) per-slot displacement
    return int(delta.argmax(dim=-1).item())


def model_side_commutator(model, encode_fn, gen_mod, records, n, device):
    """Operator commutator defects + binding-disagreement grounding metric."""
    rng = random.Random(1234)
    triples = _sample_world_triples(gen_mod, records, n, rng)

    pure_defects = []        # ||B_a B_b z - B_b B_a z|| with offsets OFF (abelian sanity ~0)
    sel_defects = []         # same with H recomputed per ordering (selection-inclusive)
    sel_rel_defects = []     # selection defect normalized by ||z||

    # Binding splits: model slot-attention assignment vs oracle entity labels.
    model_same = []          # bool per triple: model binds both actions to the same slot
    oracle_same = []         # bool per triple: oracle entity indices equal
    n_disagree = 0

    model.eval()
    for t in triples:
        types, states = t["types"], t["states"]
        a_lbl, b_lbl = t["action_a"], t["action_b"]

        k = _encode_state(model, encode_fn, gen_mod, types, states, device)  # (1,M,dn)

        v_a = _posterior_verb_idx(model, encode_fn, gen_mod, types, states, a_lbl, device)
        v_b = _posterior_verb_idx(model, encode_fn, gen_mod, types, states, b_lbl, device)

        # --- pure-matrix defect (offsets OFF): abelian operators commute -> ~0 ---
        ab_pure = _apply_verb(model, _apply_verb(model, k, v_b, device, False),
                              v_a, device, False)
        ba_pure = _apply_verb(model, _apply_verb(model, k, v_a, device, False),
                              v_b, device, False)
        pure_defects.append(float((ab_pure - ba_pure).norm().item()))

        # --- selection-inclusive defect (H recomputed per ordering) ---
        ab_sel = _apply_verb(model, _apply_verb(model, k, v_b, device, True),
                             v_a, device, True)
        ba_sel = _apply_verb(model, _apply_verb(model, k, v_a, device, True),
                             v_b, device, True)
        d = float((ab_sel - ba_sel).norm().item())
        sel_defects.append(d)
        zn = float(k.float().norm().item()) + 1e-8
        sel_rel_defects.append(d / zn)

        # --- binding splits ---
        slot_a = _model_binding_slot(model, k, v_a, device)
        slot_b = _model_binding_slot(model, k, v_b, device)
        m_same = (slot_a == slot_b)
        o_same = (_idx_of(a_lbl) == _idx_of(b_lbl))
        model_same.append(m_same)
        oracle_same.append(o_same)
        if m_same != o_same:
            n_disagree += 1

    n_ = len(triples)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n": n_,
        "pure_matrix_defect_mean": _mean(pure_defects),   # sanity: should be ~0
        "pure_matrix_defect_max": max(pure_defects) if pure_defects else None,
        "selection_defect_mean": _mean(sel_defects),       # selections may not commute
        "selection_defect_rel_mean": _mean(sel_rel_defects),
        "selection_defect_max": max(sel_defects) if sel_defects else None,
        "binding_disagreement_rate": (n_disagree / n_) if n_ else None,
        "model_same_rate": _mean([1.0 if x else 0.0 for x in model_same]),
        "oracle_same_rate": _mean([1.0 if x else 0.0 for x in oracle_same]),
    }


# ===========================================================================
# (c) SOFT-QUOTIENT READOUT CHECK — does the model actually PERFORM the merge?
# ===========================================================================
#
# The norm budget MARKS destruction without performing it: the renormalized unit-norm spine
# keeps every distinction alive (no two distinct states ever collapse in the noun geometry),
# and the per-slot scale scalar merely ANNOTATES the lost radius.  The failure mode invisible
# to every latent-side instrument: the readout/decoder never actually applies the quotient —
# it logs irreversibility in the ledger but still decodes the two pre-images apart.
#
# ORACLE-MERGED PAIRS make this testable.  The entity world has genuine many-to-one dynamics:
# ordinal-ladder SATURATION clamps distinct pre-images onto the same image (e.g. `feed` at
# "fed" and `feed` at "full" both land on "full").  Sample state pairs (s1 != s2) with an
# action `a` such that oracle(a, s1) == oracle(a, s2).  Encode both, apply the model action,
# then measure:
#   (i)  LATENT distance of the transformed pairs ‖a*(s1) - a*(s2)‖ — EXPECTED nonzero (the
#        spine preserves the distinction; the budget only annotates it).
#   (ii) DECODER-PREDICTION divergence (mean token-level JS between the two next-state token
#        distributions, teacher-forced on the shared merged target) — EXPECTED → 0 if the
#        model HEALTHILY performs the quotient (both pre-images predict the same merged future).
# Report the RATIO pred_div / latent_dist.  If the decoder predictions do NOT converge while
# the oracle says merged, the scale ledger is dead bookkeeping: irreversibility is
# logged-but-unmodeled.


def _find_oracle_merged_pairs(gen_mod, records, n_pairs, rng, max_attempts=200000):
    """Find (types, s1, s2, action) with s1 != s2 but oracle(action, s1) == oracle(action, s2).

    Single-entity saturation cases: pick a type + action, then two distinct states that the
    action clamps onto the same image (ordinal-ladder saturation / conditional overrides).
    """
    train_types = gen_mod._types_for_role("train")
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        tn = rng.choice(train_types)
        verb = gen_mod._sample_action(rng, tn, wait_weight=0.0)
        s1 = gen_mod._random_state(rng, tn)
        s2 = gen_mod._random_state(rng, tn)
        if s1 == s2:
            continue
        o1 = gen_mod.apply_action(tn, s1, verb)
        o2 = gen_mod.apply_action(tn, s2, verb)
        if o1 == o2:  # the action MERGED two distinct pre-images (saturation)
            pairs.append({"type": tn, "s1": s1, "s2": s2, "verb": verb, "merged": o1})
    return pairs


def _token_js(logits_p, logits_q, tgt_pad):
    """Mean per-token Jensen-Shannon divergence between two (1,T,V) logit tensors.

    Averaged over non-pad target positions.  JS in [0, log 2]; 0 ⟺ identical distributions.
    """
    lp = logits_p.float().log_softmax(dim=-1)  # (1,T,V)
    lq = logits_q.float().log_softmax(dim=-1)
    p = lp.exp()
    q = lq.exp()
    m = (0.5 * (p + q)).clamp_min(1e-12)
    lm = m.log()
    kl_pm = (p * (lp - lm)).sum(-1)  # (1,T)
    kl_qm = (q * (lq - lm)).sum(-1)
    js = 0.5 * (kl_pm + kl_qm)        # (1,T)
    valid = (~tgt_pad.bool()).float()  # (1,T)
    denom = valid.sum().clamp_min(1.0)
    return float((js * valid).sum() / denom)


def soft_quotient_check(model, encode_fn, gen_mod, records, n_pairs, device):
    """Latent-distance vs decoder-prediction-divergence on oracle-merged pairs."""
    rng = random.Random(4242)
    pairs = _find_oracle_merged_pairs(gen_mod, records, n_pairs, rng)
    if not pairs:
        return {"available": False, "reason": "no oracle-merged pairs found",
                "n_pairs": 0}

    latent_dists = []
    pred_divs = []
    model.eval()
    use_budget = getattr(model, "use_norm_budget", False)

    for pr in pairs:
        tn, s1, s2, verb, merged = pr["type"], pr["s1"], pr["s2"], pr["verb"], pr["merged"]
        types = [tn]

        # Encode both pre-images.
        k1 = _encode_state(model, encode_fn, gen_mod, types, [s1], device)  # (1,M,dn)
        k2 = _encode_state(model, encode_fn, gen_mod, types, [s2], device)

        # The model's latent verb for this action (grounded via the posterior on s->merged).
        v1 = _posterior_verb_idx(model, encode_fn, gen_mod, types, [s1], f"{verb}@0", device)
        v2 = _posterior_verb_idx(model, encode_fn, gen_mod, types, [s2], f"{verb}@0", device)

        # Apply the action (selection-inclusive: H on).  a* is the decoder memory.
        a1 = _apply_verb(model, k1, v1, device, use_offset=True)  # (1,M,dn)
        a2 = _apply_verb(model, k2, v2, device, use_offset=True)

        # (i) latent distance of the transformed pair (spine should keep them apart).
        latent_dists.append(float((a1.float() - a2.float()).norm().item()))

        # (ii) decoder-prediction divergence, teacher-forced on the SHARED merged target.
        merged_text = gen_mod.render_state([(tn, merged)])
        m_ids, m_pad = encode_fn(merged_text)
        m_ids = m_ids.unsqueeze(0).to(device)
        m_pad = m_pad.unsqueeze(0).to(device)
        with torch.no_grad():
            logits1 = model.decoder(a1, m_ids, m_pad)  # (1,T,V)
            logits2 = model.decoder(a2, m_ids, m_pad)
        pred_divs.append(_token_js(logits1, logits2, m_pad))

    n_ = len(pairs)
    mean_latent = sum(latent_dists) / n_
    mean_pred = sum(pred_divs) / n_
    return {
        "available": True,
        "n_pairs": n_,
        "mean_latent_dist": mean_latent,          # expected NONZERO (spine preserves)
        "mean_pred_js_divergence": mean_pred,     # expected ~0 if model performs the quotient
        # ratio: pred-divergence per unit latent distance.  Small ⇒ decoder DOES merge the
        # pre-images (healthy: latent kept apart but predictions converged).  Large ⇒ the
        # decoder decodes the two pre-images apart despite the oracle merge ⇒ the scale ledger
        # is dead bookkeeping (irreversibility logged-but-unmodeled).
        "pred_div_per_latent": (mean_pred / (mean_latent + 1e-8)),
        "uses_norm_budget": use_budget,
    }


# ===========================================================================
# Checkpoint discovery (graceful degradation)
# ===========================================================================

def _discover_checkpoint(explicit_ckpt, explicit_config):
    """Return (ckpt_path, config_path, undertrained_caveat) or (None, None, _).

    Prefers an explicit --ckpt/--config.  Otherwise tries the trained entity seeds, then
    falls back to the smoke checkpoint with the undertrained caveat flagged.
    """
    if explicit_ckpt and Path(explicit_ckpt).exists():
        return explicit_ckpt, explicit_config, False

    candidates = [
        ("results/jepa_ent_s0/model_latest.pt", "configs/jepa/jepa_ent_s0.json", False),
        ("results/jepa_ent_s1/model_latest.pt", "configs/jepa/jepa_ent_s1.json", False),
        ("results/jepa_ent_s2/model_latest.pt", "configs/jepa/jepa_ent_s2.json", False),
        ("results/jepa_ent_smoke/model_latest.pt", "configs/jepa/jepa_ent_smoke.json", True),
    ]
    for ckpt, config, caveat in candidates:
        if (REPO / ckpt).exists():
            return str(REPO / ckpt), str(REPO / config), caveat
    return None, None, False


# ===========================================================================
# World-invoice banking (frozen pre-registration artifact)
# ===========================================================================

def _bank_world_invoice(gen_mod, world, args, labeled_path):
    """Write the oracle commutation measurement to a frozen versioned artifact.

    Embeds the generator seed + a content hash of the generator CONFIG + TYPE_LIBRARY so the
    baseline is reproducible and clearly tied to the world that produced it.  Marked as the
    PRE-targeted-actions baseline.  Refuses to silently overwrite an existing banked file
    (the whole point is that it is frozen); writes a `.latest.json` sidecar instead so a
    re-run is recorded without clobbering the canonical artifact.
    """
    import hashlib

    # Content hash of the world spec (config + type library) so the artifact is pinned to the
    # exact generative world.  Deterministic JSON dump for a stable hash.
    spec = {
        "config": gen_mod.CONFIG,
        "type_library": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in gen_mod.TYPE_LIBRARY.items()},
        "actions": gen_mod.ACTIONS,
        "attribute_pool": gen_mod.ATTRIBUTE_POOL,
    }
    spec_blob = json.dumps(spec, sort_keys=True, default=str).encode()
    world_hash = hashlib.sha256(spec_blob).hexdigest()[:16]

    artifact = {
        "artifact": "world_commutator_invoice",
        "version": "v1",
        "baseline_marker": "PRE-targeted-actions",
        "note": (
            "Oracle ground-truth non-commutativity of entity-world, measured BEFORE any "
            "targeted-actions architecture change.  After targeted actions, disjoint-support "
            "commutation is 0 by construction; pre/post comparisons are only valid against "
            "this frozen baseline."
        ),
        "generator_seed": gen_mod.CONFIG.get("seed"),
        "world_spec_sha256_16": world_hash,
        "labeled_split": str(labeled_path),
        "n_world": args.n_world,
        "sample_seed": args.seed,
        "world_side": world,
    }

    out_dir = REPO / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = out_dir / "world_commutator_invoice_v1.json"
    if canonical.exists():
        # Frozen: never overwrite the canonical baseline.  Record the re-run separately.
        sidecar = out_dir / "world_commutator_invoice_v1.latest.json"
        with open(sidecar, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nBanked invoice: canonical {canonical.name} EXISTS (frozen) — "
              f"wrote re-run to {sidecar.name} (world_hash={world_hash}).")
    else:
        with open(canonical, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nBanked FROZEN baseline: {canonical} "
              f"(seed={artifact['generator_seed']}, world_hash={world_hash}).")


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Commutator probe: world-side non-commutativity invoice + model-side defect."
    )
    p.add_argument("--labeled", default="data/entity_world/test_iid_labeled.jsonl",
                   help="Labeled split JSONL (carries 'types').")
    p.add_argument("--n_world", type=int, default=2000,
                   help="World-side triples to sample (>=2000 for the invoice). Default 2000.")
    p.add_argument("--n_model", type=int, default=512,
                   help="Model-side triples (smaller; each needs encodes). Default 512.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint. If absent, auto-discovers (smoke = undertrained caveat).")
    p.add_argument("--config", default=None, help="Model config JSON for --ckpt.")
    p.add_argument("--out", default=None, help="Output JSON path.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--world_only", action="store_true",
                   help="Run only the world-side invoice (no model).")
    return p


def main():
    args = _build_arg_parser().parse_args()
    gen_mod = _load_gen()

    labeled_path = args.labeled
    if not Path(labeled_path).is_absolute():
        cand = REPO / labeled_path
        labeled_path = str(cand) if cand.exists() else labeled_path
    records = []
    with open(labeled_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} labeled chains from {labeled_path}")

    # ---- (a) world-side invoice (always) ----
    rng = random.Random(args.seed)
    print(f"World-side invoice: sampling {args.n_world} (state, a, b) triples...")
    world = world_side_invoice(gen_mod, records, args.n_world, rng)

    w = world
    print(
        "\nWORLD-SIDE NON-COMMUTATIVITY INVOICE\n"
        f"  overall          commute_rate={w['overall']['commute_rate']:.4f}  "
        f"(n={w['overall']['n']}, mean_dist_when_differ={w['overall']['mean_dist_when_differ']:.2f})\n"
        f"  same-entity      commute_rate={w['same_entity']['commute_rate']:.4f}  "
        f"(n={w['same_entity']['n']}, mean_dist_when_differ={w['same_entity']['mean_dist_when_differ']:.2f})\n"
        f"  disjoint-entity  commute_rate={w['disjoint_entity']['commute_rate']:.4f}  "
        f"(n={w['disjoint_entity']['n']})\n"
        f"  PREDICTION  disjoint~100%={w['prediction_disjoint_commutes']}  "
        f"same partial non-commute={w['prediction_same_partial_noncommute']}"
    )

    result = {"world_side": world, "labeled": labeled_path, "n_world": args.n_world}

    # ---- BANK THE WORLD-SIDE INVOICE AS A FROZEN PRE-REGISTRATION ARTIFACT ----
    # Pre-registration ordering: the upcoming targeted-actions change makes disjoint-support
    # commutation ZERO BY CONSTRUCTION (an action targeting disjoint slots cannot reorder),
    # collapsing the model-side instrument to the same-slot cell.  Pre/post comparisons are
    # only valid against THIS banked baseline — frozen BEFORE any architecture change exists.
    _bank_world_invoice(gen_mod, world, args, labeled_path)

    # ---- (b) model-side (graceful degradation) ----
    if not args.world_only:
        ckpt, config, caveat = _discover_checkpoint(args.ckpt, args.config)
        if ckpt is None or config is None:
            print("\nModel-side: no checkpoint found — skipping (world-side invoice stands).")
            result["model_side"] = {"available": False,
                                    "reason": "no checkpoint/config found"}
        else:
            if args.device:
                device = torch.device(args.device)
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            from twm.jepa.config import JEPAConfig
            from twm.jepa.model import build_jepa_model_v2
            import torch.nn as nn

            cfg = JEPAConfig.from_json(config)
            token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
            model = build_jepa_model_v2(cfg, token_emb).to(device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            state = state.get("model_state_dict", state.get("model", state))
            model.load_state_dict(state, strict=False)
            print(f"\nModel-side: loaded {ckpt}"
                  + ("  [UNDERTRAINED smoke checkpoint — tooling smoke test only]" if caveat else ""))

            encode_fn = _load_tokenizer(
                cfg.data.tokenizer, cfg.data.max_text_tokens,
                append_eos=getattr(cfg.data, "append_eos", True),
            )
            ms = model_side_commutator(model, encode_fn, gen_mod, records, args.n_model, device)
            ms["available"] = True
            ms["undertrained_caveat"] = caveat
            ms["ckpt"] = ckpt
            result["model_side"] = ms

            print(
                "\nMODEL-SIDE COMMUTATOR\n"
                f"  pure_matrix_defect (offsets OFF, abelian sanity ~0): "
                f"mean={ms['pure_matrix_defect_mean']:.2e} max={ms['pure_matrix_defect_max']:.2e}\n"
                f"  selection_defect   (H per ordering): "
                f"mean={ms['selection_defect_mean']:.4f} rel_mean={ms['selection_defect_rel_mean']:.4f}\n"
                f"  binding_disagreement_rate = {ms['binding_disagreement_rate']:.4f}  "
                f"(model_same={ms['model_same_rate']:.3f}, oracle_same={ms['oracle_same_rate']:.3f})"
                + ("\n  [undertrained_caveat=True]" if caveat else "")
            )

            # ---- (c) soft-quotient readout check (oracle-merged pairs) ----
            sq = soft_quotient_check(model, encode_fn, gen_mod, records, args.n_model, device)
            sq["undertrained_caveat"] = caveat
            result["soft_quotient"] = sq
            if sq.get("available"):
                print(
                    "\nSOFT-QUOTIENT READOUT CHECK (oracle-merged pairs)\n"
                    f"  n_pairs={sq['n_pairs']}\n"
                    f"  mean_latent_dist        = {sq['mean_latent_dist']:.4f}  "
                    f"(expect NONZERO — spine preserves distinctions)\n"
                    f"  mean_pred_js_divergence = {sq['mean_pred_js_divergence']:.4f}  "
                    f"(expect ~0 if model performs the quotient)\n"
                    f"  pred_div_per_latent     = {sq['pred_div_per_latent']:.4f}  "
                    f"(small ⇒ decoder merges pre-images; large ⇒ dead scale ledger)"
                    + ("\n  [undertrained_caveat=True]" if caveat else "")
                )
            else:
                print(f"\nSoft-quotient check skipped: {sq.get('reason')}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {out_p}")

    return result


if __name__ == "__main__":
    main()
