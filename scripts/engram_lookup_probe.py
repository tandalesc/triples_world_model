#!/usr/bin/env python3
"""Engram lookup-table vs H-conditioner pre-registered experiment.

Question (pre-registered): can an Engram-style DETERMINISTIC lookup table replace the
H conditioning network (PolarConditioner) for operator selection in the JEPA world model?

The model (frozen checkpoint, results/jepa_ent_s0/model_latest.pt):
  - slots M=8, d_noun=32 ⟹ nb=16 complex blocks per slot (the modulus profile dimension)
  - verbs V=8 (global per step), selected by the posterior q(v | s, s')
  - H = PolarConditioner: a zero-init-at-start, trained Linear(nb→nb, no bias) mapping a
    slot's MODULUS profile |k_i| to a per-block phase offset θ_offset_i = H(|k_i|).
  - operator.apply(k, v, theta_offset=θoff) applies B(θ_v + θoff_i) per slot; the SAME
    offset is replayed by inverse_apply (the ledger ⟹ exact inversion).
  - use_norm_budget=True: apply also renormalizes each slot to its pre-step modulus and
    returns a per-slot log_rho (scale_delta) that the inverse re-applies.

This script is read-only on the model: NO training, NO retraining. It builds ONE variable
at a time and reports three pre-registered numbers, then stops.

BUILD (one variable at a time)
------------------------------
1. LOOKUP TABLE
   key   = (cluster_id(|k_i|), verb_id)   — per-SLOT, since H is shared & acts per-slot.
   value = the θ-offset vector H(|k_i|) the frozen H produced for that slot.
   Quantization: k-means (k=64), Lloyd, fit on TRAINING-trajectory per-slot modulus
   profiles (one nb-vector per (step, slot)). Shared codebook across slots (documented).
   Population = LEDGER REPLAY of the frozen encoder+posterior+H over training chains:
   the logged (modulus_profile, verb, offset) triples. No gradient, no training.
   Per-key stats: count, mean offset, per-key offset VARIANCE (the static-pattern
   diagnostic: a deterministic table is only faithful where the variance is ~0).

2. COVERAGE PROBE (held-out test_iid)
   For each step+slot, lookup HIT = the (cluster_id, verb) key exists in the populated
   table. Hit rate overall AND split by the oracle commuting/interaction classification.
   SPLITTING RULE (documented exactly): a chain step's action targets entity index e.
   The step is INTERACTION-SECTOR iff the adjacent step (previous OR next action in the
   same chain) targets the SAME entity index e (same-entity ⟹ partially non-commuting per
   the world invoice). Otherwise the step is COMMUTING-SECTOR (disjoint-entity adjacency ⟹
   commutes exactly). Single-action chains / boundary steps with no adjacent action are
   classified by their only neighbor; a step with NO neighbors is excluded from the split
   (counted in overall only). Slots inherit their step's sector.

3. SWAP-IN EVAL (same frozen checkpoint)
   Replace H's per-slot output with the table value on HIT, keep H on MISS. Compare,
   teacher-forced on the held-out step targets, vs the pure-H baseline (offsets entirely
   from H):
     (a) per-hop token CE (decoder NLL of the next-state text given a*),
     (b) identity-persistence: ‖|k| − |a*|‖ (rotation preserves modulus; budget restores
         radius ⟹ identity profile should persist),
     (c) inverse round-trip exactness: ‖undo(apply(k)) − k‖ using the SAME (offset,
         scale_delta) ledger that produced a* (the ledger code path).
   Ledger check: retrieved offsets must LOG and INVERT identically to H's — same
   inverse_apply code path, same per-slot scale_delta, exact to fp32 eps.

REPORT (then STOP)
------------------
  (1) hit rate: commuting vs interaction sector
  (2) lookup-vs-H accuracy on hits: CE delta (swap-in CE − pure-H CE, on hit-heavy steps)
  (3) ledger-properties verdict: inverse round-trip exact yes/no (swap-in vs baseline)
  + the per-key offset-variance distribution.
JSON + short markdown to results/jepa_engram_lookup/.
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
# Loaders (shared shape with probe_commutator.py)
# ---------------------------------------------------------------------------

def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "generate_entity_world", REPO / "scripts" / "generate_entity_world.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tokenizer(tokenizer_path: str, max_text_tokens: int, append_eos: bool = True):
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


def _read_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# Chain replay helpers (oracle-grounded verb via the posterior, ledger via H)
# ---------------------------------------------------------------------------

def _verb_of(label):
    return label.rsplit("@", 1)[0]


def _idx_of(label):
    return int(label.rsplit("@", 1)[1])


@torch.no_grad()
def _encode_text(model, encode_fn, text, device):
    ids, pad = encode_fn(text)
    _, k, _ = model.encoder(ids.unsqueeze(0).to(device), pad.unsqueeze(0).to(device))
    return k  # (1, M, dn)


@torch.no_grad()
def _posterior_verb(model, encode_fn, src_text, tgt_text, device):
    s_ids, s_pad = encode_fn(src_text)
    t_ids, t_pad = encode_fn(tgt_text)
    v_onehot, _, _ = model.transition(
        s_ids.unsqueeze(0).to(device), s_pad.unsqueeze(0).to(device),
        t_ids.unsqueeze(0).to(device), t_pad.unsqueeze(0).to(device),
        tau=1.0, hard=True,
    )
    return int(v_onehot.argmax(dim=-1).item())


def _iter_chain_steps(records, max_chains=None, rng=None, shuffle=False):
    """Yield per-step replay units from labeled chain records.

    A labeled record: {"chain": [text_0, text_1, ...], "actions": ["v@idx", ...]}.
    chain[i] is the state text BEFORE action i; chain[i+1] carries action i's result. The
    encoder is fed the STATE text (the same render the posterior's src uses). For step i:
      src_text = chain[i]   (state before the action — operator input k)
      tgt_text = chain[i+1] (state after — decoder/posterior target)
      action   = actions[i] ("verb@entity_idx")
    Adjacency for the sector split is over the `actions` list within the chain.
    """
    idxs = list(range(len(records)))
    if shuffle and rng is not None:
        rng.shuffle(idxs)
    if max_chains is not None:
        idxs = idxs[:max_chains]
    for ci in idxs:
        r = records[ci]
        chain = r["chain"]
        actions = r.get("actions", [])
        n_steps = min(len(actions), len(chain) - 1)
        ent_idxs = [_idx_of(a) for a in actions[:n_steps]]
        for i in range(n_steps):
            # Sector: interaction iff a neighbor action targets the same entity idx.
            e = ent_idxs[i]
            neigh = []
            if i - 1 >= 0:
                neigh.append(ent_idxs[i - 1])
            if i + 1 < n_steps:
                neigh.append(ent_idxs[i + 1])
            if not neigh:
                sector = "none"          # isolated step: no adjacency ⟹ excluded from split
            elif any(n == e for n in neigh):
                sector = "interaction"   # same-entity neighbor ⟹ partially non-commuting
            else:
                sector = "commuting"     # all neighbors disjoint ⟹ commutes exactly
            yield {
                "src_text": chain[i],
                "tgt_text": chain[i + 1],
                "verb_label": actions[i],
                "sector": sector,
            }


# ---------------------------------------------------------------------------
# (1) k-means quantizer (Lloyd, self-contained, fp32) over per-slot modulus profiles
# ---------------------------------------------------------------------------

def _kmeans_fit(X, k, n_iter=50, seed=0, device="cpu"):
    """Lloyd k-means on X (N, nb). Returns centroids (k, nb). Empty clusters re-seeded."""
    g = torch.Generator(device=device).manual_seed(seed)
    N = X.shape[0]
    # k-means++-ish init: random distinct points.
    perm = torch.randperm(N, generator=g, device=device)[:k]
    C = X[perm].clone()
    for _ in range(n_iter):
        # assign
        d = torch.cdist(X, C)              # (N, k)
        assign = d.argmin(dim=1)           # (N,)
        newC = C.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                newC[j] = X[mask].mean(dim=0)
            else:
                # re-seed empty cluster at a random point
                ri = torch.randint(0, N, (1,), generator=g, device=device).item()
                newC[j] = X[ri]
        shift = (newC - C).norm().item()
        C = newC
        if shift < 1e-6:
            break
    return C


def _kmeans_assign(X, C):
    """Assign each row of X (N, nb) to nearest centroid in C (k, nb). Returns (N,) long."""
    return torch.cdist(X, C).argmin(dim=1)


# ---------------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/jepa_ent_s0/model_latest.pt")
    ap.add_argument("--config", default="configs/jepa/jepa_ent_s0.json")
    ap.add_argument("--train_labeled", default="data/entity_world/train_labeled.jsonl")
    ap.add_argument("--test_labeled", default="data/entity_world/test_iid_labeled.jsonl")
    ap.add_argument("--k", type=int, default=64, help="k-means codebook size")
    ap.add_argument("--n_train_chains", type=int, default=4000,
                    help="training chains for table population (ledger replay)")
    ap.add_argument("--n_kmeans_chains", type=int, default=2000,
                    help="training chains to fit the k-means codebook")
    ap.add_argument("--n_test_chains", type=int, default=2000)
    ap.add_argument("--out", default="results/jepa_engram_lookup/engram_lookup.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU thread cap")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Be a polite CPU citizen under contention (training jobs share the box).
    try:
        torch.set_num_threads(max(1, args.threads))
    except Exception:
        pass
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2
    from twm.jepa.conditioning import block_modulus
    import torch.nn as nn

    cfg = JEPAConfig.from_json(str(REPO / args.config) if not Path(args.config).is_absolute() else args.config)
    token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, token_emb).to(device)
    ckpt_path = str(REPO / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("model_state_dict", state.get("model", state))
    model.load_state_dict(state, strict=False)
    model.eval()

    assert model.conditioner is not None, "checkpoint has no PolarConditioner (H)"
    use_budget = model.use_norm_budget
    M = cfg.model.n_slots
    dn = cfg.model.d_noun
    nb = dn // 2
    print(f"Loaded {ckpt_path}  M={M} dn={dn} nb={nb} V={cfg.model.n_verbs} "
          f"use_norm_budget={use_budget} use_polar={model.use_polar_conditioning}")

    encode_fn = _load_tokenizer(
        str(REPO / cfg.data.tokenizer) if not Path(cfg.data.tokenizer).is_absolute() else cfg.data.tokenizer,
        cfg.data.max_text_tokens, append_eos=getattr(cfg.data, "append_eos", True),
    )

    train_path = str(REPO / args.train_labeled) if not Path(args.train_labeled).is_absolute() else args.train_labeled
    test_path = str(REPO / args.test_labeled) if not Path(args.test_labeled).is_absolute() else args.test_labeled
    train_recs = _read_jsonl(train_path)
    test_recs = _read_jsonl(test_path)
    print(f"train chains={len(train_recs)} test chains={len(test_recs)}")

    gen_mod = _load_gen()  # not strictly needed (labels carry entity idx) but kept for parity

    # =====================================================================
    # STEP A: fit k-means codebook on TRAINING per-slot modulus profiles
    # =====================================================================
    print("\n[A] Collecting training per-slot modulus profiles for k-means...", flush=True)
    prof_samples = []
    km_rng = random.Random(args.seed + 1)
    _a_seen = 0
    for unit in _iter_chain_steps(train_recs, max_chains=args.n_kmeans_chains,
                                  rng=km_rng, shuffle=True):
        k = _encode_text(model, encode_fn, unit["src_text"], device)  # (1, M, dn)
        m = block_modulus(k.float())[0]                               # (M, nb)
        prof_samples.append(m.cpu())
        _a_seen += 1
        if _a_seen % 2000 == 0:
            print(f"    [A] {_a_seen} steps encoded", flush=True)
    X = torch.cat(prof_samples, dim=0)  # (N, nb), one row per (step, slot)
    print(f"    k-means fit data: {X.shape[0]} per-slot profiles (nb={X.shape[1]})")
    C = _kmeans_fit(X, args.k, n_iter=50, seed=args.seed, device="cpu")  # (k, nb)
    # codebook usage on fit data
    fit_assign = _kmeans_assign(X, C)
    used = int(fit_assign.unique().numel())
    print(f"    codebook clusters used on fit data: {used}/{args.k}")

    # =====================================================================
    # STEP B: POPULATE THE LOOKUP TABLE via ledger replay over training chains
    #   key = (cluster_id(|k_i|), verb_id) ; value = H(|k_i|) offset (nb-vector)
    # =====================================================================
    print("\n[B] Populating lookup table (ledger replay over training chains)...")
    # accumulate per-key: sum offset, sumsq offset, count
    key_sum = defaultdict(lambda: torch.zeros(nb))
    key_sumsq = defaultdict(lambda: torch.zeros(nb))
    key_count = defaultdict(int)
    pop_rng = random.Random(args.seed + 2)
    n_pop_steps = 0
    for unit in _iter_chain_steps(train_recs, max_chains=args.n_train_chains,
                                  rng=pop_rng, shuffle=True):
        k = _encode_text(model, encode_fn, unit["src_text"], device)  # (1, M, dn)
        v = _posterior_verb(model, encode_fn, unit["src_text"], unit["tgt_text"], device)
        with torch.no_grad():
            offs = model.conditioner(k)[0].float().cpu()              # (M, nb) = H(|k_i|)
        m = block_modulus(k.float())[0].cpu()                         # (M, nb)
        clusters = _kmeans_assign(m, C)                               # (M,)
        for i in range(M):
            key = (int(clusters[i].item()), int(v))
            o = offs[i]
            key_sum[key] += o
            key_sumsq[key] += o * o
            key_count[key] += 1
        n_pop_steps += 1
        if n_pop_steps % 2000 == 0:
            print(f"    [B] {n_pop_steps} steps populated, keys={len(key_count)}", flush=True)
    print(f"    populated from {n_pop_steps} steps; distinct keys = {len(key_count)}")

    # finalize table: mean offset + per-key scalar variance (mean over nb of per-dim var)
    table_mean = {}
    key_var = {}
    for key, cnt in key_count.items():
        mean = key_sum[key] / cnt
        table_mean[key] = mean
        # population variance per dim, then mean over dims -> scalar diagnostic
        ex2 = key_sumsq[key] / cnt
        var = (ex2 - mean * mean).clamp_min(0.0)
        key_var[key] = float(var.mean().item())

    var_values = sorted(key_var.values())
    # weight variance distribution by key count (how much of REPLAYED MASS is low-variance)
    total_cnt = sum(key_count.values())
    weighted_var = sum(key_var[k] * key_count[k] for k in key_count) / max(total_cnt, 1)

    def _quantile(vals, q):
        if not vals:
            return None
        idx = min(len(vals) - 1, int(q * (len(vals) - 1)))
        return vals[idx]

    var_dist = {
        "n_keys": len(key_var),
        "mean_per_key_var": (sum(var_values) / len(var_values)) if var_values else None,
        "count_weighted_var": weighted_var,
        "p50": _quantile(var_values, 0.50),
        "p90": _quantile(var_values, 0.90),
        "p99": _quantile(var_values, 0.99),
        "max": var_values[-1] if var_values else None,
        "frac_keys_var_lt_1e-4": (sum(1 for v in var_values if v < 1e-4) / len(var_values)) if var_values else None,
        "frac_keys_var_lt_1e-3": (sum(1 for v in var_values if v < 1e-3) / len(var_values)) if var_values else None,
    }
    print(f"    per-key var: mean={var_dist['mean_per_key_var']:.3e} "
          f"count-weighted={weighted_var:.3e} p90={var_dist['p90']:.3e} max={var_dist['max']:.3e}")

    # =====================================================================
    # STEP C: COVERAGE PROBE on held-out test_iid (hit rate by sector)
    # =====================================================================
    print("\n[C] Coverage probe on test_iid (hit rate by sector)...")
    # also drive (E) swap-in eval in the same pass to avoid re-encoding.
    cov = {s: {"slots": 0, "hits": 0} for s in ("overall", "commuting", "interaction")}

    # swap-in accumulators (token CE, identity persistence, inverse exactness)
    ce_H = []          # baseline pure-H per-step token CE
    ce_swap = []       # swap-in per-step token CE
    ce_H_hitheavy = []     # steps where >=50% slots hit (lookup is load-bearing)
    ce_swap_hitheavy = []
    idperp_H = []
    idperp_swap = []
    inv_err_H = []
    inv_err_swap = []
    n_test_steps = 0

    test_rng = random.Random(args.seed + 3)
    for unit in _iter_chain_steps(test_recs, max_chains=args.n_test_chains,
                                  rng=test_rng, shuffle=True):
        src_text, tgt_text, sector = unit["src_text"], unit["tgt_text"], unit["sector"]
        k = _encode_text(model, encode_fn, src_text, device)             # (1, M, dn)
        v = _posterior_verb(model, encode_fn, src_text, tgt_text, device)
        m = block_modulus(k.float())[0].cpu()                            # (M, nb)
        clusters = _kmeans_assign(m, C)                                  # (M,)

        with torch.no_grad():
            off_H = model.conditioner(k)[0].float()                      # (M, nb) on device

        # build swap-in offset: table value on hit, H on miss
        off_swap = off_H.clone().cpu()
        hit_mask = torch.zeros(M, dtype=torch.bool)
        for i in range(M):
            key = (int(clusters[i].item()), int(v))
            if key in table_mean:
                hit_mask[i] = True
                off_swap[i] = table_mean[key]
        off_swap = off_swap.to(device)

        n_hit = int(hit_mask.sum().item())
        cov["overall"]["slots"] += M
        cov["overall"]["hits"] += n_hit
        if sector in ("commuting", "interaction"):
            cov[sector]["slots"] += M
            cov[sector]["hits"] += n_hit

        # --- swap-in eval (token CE / identity persistence / inverse exactness) ---
        tgt_ids, tgt_pad = encode_fn(tgt_text)
        tgt_ids = tgt_ids.unsqueeze(0).to(device)
        tgt_pad = tgt_pad.unsqueeze(0).to(device)
        v_slots = k.new_full((1, M), int(v), dtype=torch.long)

        ce_b = _run_branch(model, k, v_slots, off_H.unsqueeze(0), tgt_ids, tgt_pad,
                           use_budget, idperp_H, inv_err_H)
        ce_s = _run_branch(model, k, v_slots, off_swap.unsqueeze(0), tgt_ids, tgt_pad,
                           use_budget, idperp_swap, inv_err_swap)
        ce_H.append(ce_b)
        ce_swap.append(ce_s)
        if n_hit >= (M + 1) // 2:
            ce_H_hitheavy.append(ce_b)
            ce_swap_hitheavy.append(ce_s)
        n_test_steps += 1
        if n_test_steps % 2000 == 0:
            print(f"    [C] {n_test_steps} test steps evaluated", flush=True)

    def _rate(d):
        return (d["hits"] / d["slots"]) if d["slots"] else None

    hit_rates = {
        "overall": {"slots": cov["overall"]["slots"], "hit_rate": _rate(cov["overall"])},
        "commuting": {"slots": cov["commuting"]["slots"], "hit_rate": _rate(cov["commuting"])},
        "interaction": {"slots": cov["interaction"]["slots"], "hit_rate": _rate(cov["interaction"])},
    }
    print(f"    test steps={n_test_steps}")
    print(f"    HIT RATE  overall={hit_rates['overall']['hit_rate']:.4f}  "
          f"commuting={hit_rates['commuting']['hit_rate']:.4f}  "
          f"interaction={hit_rates['interaction']['hit_rate']:.4f}")

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    ce_delta_all = (_mean(ce_swap) - _mean(ce_H)) if ce_H else None
    ce_delta_hitheavy = (
        (_mean(ce_swap_hitheavy) - _mean(ce_H_hitheavy))
        if ce_H_hitheavy else None
    )
    print(f"    CE  pure-H={_mean(ce_H):.4f}  swap-in={_mean(ce_swap):.4f}  "
          f"delta={ce_delta_all:+.4f}  (hit-heavy delta={ce_delta_hitheavy})")

    inv_err_H_mean = _mean(inv_err_H)
    inv_err_swap_mean = _mean(inv_err_swap)
    inv_err_swap_max = max(inv_err_swap) if inv_err_swap else None
    inv_exact = (inv_err_swap_max is not None and inv_err_swap_max < 1e-4)
    print(f"    inverse round-trip  pure-H mean={inv_err_H_mean:.2e}  "
          f"swap-in mean={inv_err_swap_mean:.2e} max={inv_err_swap_max:.2e}  "
          f"EXACT={inv_exact}")
    print(f"    identity-persistence ‖|k|-|a|‖  pure-H={_mean(idperp_H):.4f}  "
          f"swap-in={_mean(idperp_swap):.4f}")

    # =====================================================================
    # ASSEMBLE REPORT
    # =====================================================================
    result = {
        "experiment": "engram_lookup_vs_H",
        "ckpt": ckpt_path,
        "config": args.config,
        "model": {
            "M": M, "d_noun": dn, "nb": nb, "n_verbs": cfg.model.n_verbs,
            "use_norm_budget": use_budget,
            "use_polar_conditioning": model.use_polar_conditioning,
        },
        "lookup_table": {
            "key": "(kmeans_cluster_id(per_slot_modulus_profile), verb_id)",
            "value": "H(|k_i|) per-slot phase-offset vector (nb)",
            "kmeans_k": args.k,
            "kmeans_clusters_used_on_fit": used,
            "kmeans_fit_profiles": int(X.shape[0]),
            "shared_codebook_across_slots": True,
            "n_population_steps": n_pop_steps,
            "n_distinct_keys": len(key_count),
            "total_replay_mass": total_cnt,
        },
        # ----- PRE-REGISTERED NUMBER (1): hit rate commuting vs interaction -----
        "number_1_hit_rate": hit_rates,
        # ----- PRE-REGISTERED NUMBER (2): lookup-vs-H accuracy on hits (CE delta) -----
        "number_2_ce_delta": {
            "ce_pure_H_mean": _mean(ce_H),
            "ce_swap_in_mean": _mean(ce_swap),
            "ce_delta_swap_minus_H": ce_delta_all,
            "ce_delta_hit_heavy_steps": ce_delta_hitheavy,
            "n_hit_heavy_steps": len(ce_H_hitheavy),
            "note": "CE delta > 0 means swap-in is WORSE (higher NLL) than pure H.",
        },
        # ----- PRE-REGISTERED NUMBER (3): ledger-properties verdict -----
        "number_3_ledger_verdict": {
            "inverse_roundtrip_exact": inv_exact,
            "inv_err_pure_H_mean": inv_err_H_mean,
            "inv_err_swap_in_mean": inv_err_swap_mean,
            "inv_err_swap_in_max": inv_err_swap_max,
            "identity_persistence_pure_H": _mean(idperp_H),
            "identity_persistence_swap_in": _mean(idperp_swap),
            "note": (
                "Swap-in offsets routed through the SAME operator.inverse_apply ledger "
                "path (with the SAME per-slot scale_delta). Exact iff max round-trip "
                "error < 1e-4 (fp32 eps regime)."
            ),
        },
        # ----- the registered static-pattern diagnostic -----
        "per_key_variance_distribution": var_dist,
        "splitting_rule": (
            "A chain step's action targets entity idx e. INTERACTION-SECTOR iff an "
            "adjacent action (prev or next in the same chain) targets the SAME e; "
            "COMMUTING-SECTOR iff all adjacent actions target disjoint entities; "
            "isolated steps (no adjacent action) excluded from the split, counted in "
            "overall only. Slots inherit their step's sector."
        ),
        "n_test_steps": n_test_steps,
    }

    out_p = Path(args.out)
    if not out_p.is_absolute():
        out_p = REPO / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_p}")

    # also dump the per-key variance values (for a histogram) as a sidecar
    var_sidecar = out_p.parent / "per_key_variance.json"
    with open(var_sidecar, "w") as f:
        json.dump({
            "per_key_variance_sorted": var_values,
            "per_key_count_sorted_same_order": [key_count[k] for k in
                                                sorted(key_var, key=lambda kk: key_var[kk])],
        }, f)
    print(f"Wrote {var_sidecar}")

    return result


@torch.no_grad()
def _run_branch(model, k, v_slots, theta_offset, tgt_ids, tgt_pad,
                use_budget, idperp_acc, inv_err_acc):
    """Apply operator with the given per-slot theta_offset, decode, measure CE +
    identity-persistence + exact inverse round-trip through the SAME ledger path.

    theta_offset: (1, M, nb). Returns the token CE (scalar) and appends identity-
    persistence and inverse-error to the provided accumulators.
    """
    from twm.jepa.conditioning import block_modulus

    if use_budget:
        a, scale_delta = model.operator.apply(
            k, v_slots, theta_offset=theta_offset, norm_budget=True
        )
        # exact inverse through the SAME ledger (same offset + stored scale_delta)
        k_inv = model.operator.inverse_apply(
            a, v_slots, theta_offset=theta_offset,
            norm_budget=True, scale_delta=scale_delta,
        )
    else:
        a = model.operator.apply(k, v_slots, theta_offset=theta_offset)
        k_inv = model.operator.inverse_apply(a, v_slots, theta_offset=theta_offset)

    # token CE (teacher-forced); pad_id ignored
    logits = model.decoder(a, tgt_ids, tgt_pad)  # (1, T, V)
    V = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, V).float(), tgt_ids.reshape(-1),
        ignore_index=model.decoder.pad_id,
    ).item()

    # identity-persistence: modulus profile drift ‖|k| - |a|‖ (per-state, averaged over slots)
    m_k = block_modulus(k.float())
    m_a = block_modulus(a.float())
    idperp_acc.append(float((m_k - m_a).norm(dim=-1).mean().item()))

    # inverse round-trip exactness
    inv_err_acc.append(float((k_inv.float() - k.float()).norm().item()))
    return ce


if __name__ == "__main__":
    main()
