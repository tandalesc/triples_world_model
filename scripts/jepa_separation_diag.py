"""Latent-space separation diagnostic for the entity-world encoder.

Gates the density-ranking retrieval arm: if hard-pool candidates are coincident in
latent space, no density scorer can rank them.

Three measurements on 512 hard pools (seed 0, pool ~41):

  (i)  Separation index: mean within-pool pairwise L2 / global mean pairwise L2.
         near 1 = pools as spread as random = good; near 0 = coincident = fatal.
  (ii) NN margin: for each distractor, dist to its nearest within-pool neighbour vs
         dist to gold. Fraction of distractors closer to gold than to any other candidate
         ("gold-hugging fraction"). Also: noise floor via token-level pad-shift re-encode.
 (iii) Linear-probe AUC: LR on (z_candidate, z_query_a*) pairs -> gold/distractor label.
         AUC >> 0.5 = information exists for ranking; AUC ~ 0.5 = encoder-fix-first.

Three encoder representation variants:
  ema   : model.ema.pool_raw(t+1)  — the EMA-key used in training
  online: model._online_bundle.pool_raw(t+1)  — the online (trained) encoder
  slot_mean: k.mean(dim=1) where k = online encoder's slot nouns of t+1

CPU-only. Usage:
  CUDA_VISIBLE_DEVICES= uv run python scripts/jepa_separation_diag.py \\
      --ckpt results/jepa_ent_s0/model_ep50.pt \\
      --out  results/jepa_separation/ep50 \\
      --repo .
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

SEED = 0
N_POOLS = 512       # number of queries / pools
HARD_NN = 40        # NN distractors per pool (matches diagnostics default)
PAD_ID = 0


# ───────────────────────────────── loading (mirrors mrr_diagnosis.py) ─────────
def load(ckpt_path, repo):
    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2
    from twm.domain_bpe import DomainBPETokenizer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = JEPAConfig.from_dict(ckpt["config"])
    m = cfg.model
    meta = {
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "n_slots": m.n_slots,
        "n_verbs": m.n_verbs,
        "d_noun": m.d_noun,
        "d_model": m.d_model,
        "operator_group": getattr(m, "operator_group", None),
        "use_norm_budget": getattr(m, "use_norm_budget", False),
        "vocab_size": cfg.data.vocab_size,
        "max_text_tokens": cfg.data.max_text_tokens,
    }
    emb = nn.Embedding(cfg.data.vocab_size, m.d_model)
    emb.weight.requires_grad_(False)
    model = build_jepa_model_v2(cfg, emb)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    meta["missing_keys"] = list(missing)
    meta["unexpected_keys"] = list(unexpected)
    model.eval()
    tok = DomainBPETokenizer.load(
        str(Path(repo) / "data" / "entity_world" / "bpe_512.json"),
        max_length=cfg.data.max_text_tokens,
    )
    return model, tok, cfg, meta


def read_chains(path):
    chains = []
    with open(path) as f:
        for line in f:
            chains.append(json.loads(line)["chain"])
    return chains


def build_pairs(chains):
    src, tgt, cid = [], [], []
    for ci, ch in enumerate(chains):
        for i in range(len(ch) - 1):
            src.append(ch[i])
            tgt.append(ch[i + 1])
            cid.append(ci)
    return src, tgt, np.array(cid)


def encode_batch(tok, texts, T):
    ids = torch.zeros((len(texts), T), dtype=torch.long)
    for i, t in enumerate(texts):
        ids[i] = torch.tensor(tok.encode(t, max_length=T), dtype=torch.long)
    pad = ids == PAD_ID
    return ids, pad


# ────────────────────────────────── encoding helpers ──────────────────────────
@torch.no_grad()
def encode_all_variants(model, tgt_ids, tgt_pad, bs=64):
    """Return EMA-pool, online-pool, and slot-mean representations for all targets."""
    N, T = tgt_ids.shape
    d_noun = model.d_noun
    ema_vecs   = torch.zeros(N, d_noun)
    online_vecs = torch.zeros(N, d_noun)
    slot_vecs  = torch.zeros(N, d_noun)
    for s in range(0, N, bs):
        ti = tgt_ids[s:s + bs]
        tp = tgt_pad[s:s + bs]
        # EMA pool (the target in L_pred / MRR)
        ema_vecs[s:s + bs] = model.ema.pool_raw(ti, tp)
        # Online pool (same architecture, but online weights)
        online_vecs[s:s + bs] = model._online_bundle.pool_raw(ti, tp)
        # Slot-set mean: k.mean(M)
        _, k, _ = model.encoder(ti, tp)
        slot_vecs[s:s + bs] = k.mean(dim=1)
    return ema_vecs, online_vecs, slot_vecs


@torch.no_grad()
def get_query_zhat(model, src_ids, src_pad, tgt_ids, tgt_pad, bs=64):
    """zhat = predictor(readout(a*)) for queries — the 'query representation' under
    the EMA-cosine head used in training. (Matches MRR variant (a) exactly.)"""
    N = src_ids.shape[0]
    d_noun = model.d_noun
    zhat_vecs = torch.zeros(N, d_noun)
    for s in range(0, N, bs):
        out = model.forward_v2(
            src_ids[s:s + bs], src_pad[s:s + bs],
            tgt_ids[s:s + bs], tgt_pad[s:s + bs],
            tau=1.0, hard=True,
        )
        zhat_vecs[s:s + bs] = out["zhat"]
    return zhat_vecs


# ────────────────────────────────── pool construction ─────────────────────────
def build_pools(z_target, chain_ids):
    """Hard pools: gold at [0] + same-chain + 40 NN of EMA-key.
    Mirrors diagnostics._compute_retrieval_mrr exactly."""
    N = z_target.shape[0]
    same_chain = defaultdict(list)
    for i, cid in enumerate(chain_ids):
        for j, cid2 in enumerate(chain_ids):
            if j != i and cid2 == cid:
                same_chain[i].append(j)
    zt_n = F.normalize(z_target.float(), dim=-1)
    sims_mat = (zt_n @ zt_n.T).cpu().numpy()
    hard_pools = []
    for i in range(N):
        base = [i] + same_chain[i]
        baseset = set(base)
        order = np.argsort(-sims_mat[i])
        nn = [int(j) for j in order if j not in baseset][:HARD_NN]
        hard_pools.append(base + nn)
    return hard_pools


# ────────────────────────── noise floor via pad-shift re-encode ───────────────
@torch.no_grad()
def noise_floor(model, tgt_ids, tgt_pad, n_samples=512, bs=64):
    """Estimate encoder noise floor by re-encoding with a random 1-position pad-shift.

    For each of `n_samples` states, the canonical encoding z and a perturbed encoding
    z' (shift all tokens right by 1, drop the last, pad first column) are computed;
    noise floor = mean L2(z - z') over the sample. This is the minimum inter-example
    distance we could plausibly resolve; distances below this are indistinguishable.

    Returns (mean_noise, p90_noise) for each variant (ema, online, slot_mean).
    """
    sel = min(n_samples, tgt_ids.shape[0])
    ids = tgt_ids[:sel]
    pad = tgt_pad[:sel]

    # Shift: roll tokens right by 1, set col 0 = PAD_ID
    ids_shift = torch.roll(ids, 1, dims=1)
    ids_shift[:, 0] = PAD_ID
    pad_shift = ids_shift == PAD_ID

    ema_a,   online_a,   slot_a   = encode_all_variants(model, ids,       pad,       bs=bs)
    ema_b,   online_b,   slot_b   = encode_all_variants(model, ids_shift,  pad_shift, bs=bs)

    def _stats(a, b):
        diff = (a - b).norm(dim=-1).numpy()
        return float(diff.mean()), float(np.percentile(diff, 90))

    return {
        "ema":       _stats(ema_a,    ema_b),
        "online":    _stats(online_a, online_b),
        "slot_mean": _stats(slot_a,   slot_b),
    }


# ────────────────────────────────── global pairwise distance ─────────────────
def global_mean_pairwise_l2(vecs, n_sample=1000, seed=0):
    """Estimate global mean pairwise L2 by sampling n_sample random pairs."""
    rng = np.random.default_rng(seed)
    N = vecs.shape[0]
    idx_a = rng.integers(0, N, size=n_sample)
    idx_b = rng.integers(0, N, size=n_sample)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    dists = (vecs[idx_a].float() - vecs[idx_b].float()).norm(dim=-1).numpy()
    return float(dists.mean())


# ─────────────────────────── per-pool pairwise distances ─────────────────────
def within_pool_pairwise_l2(vecs, pools, subsample_pools=512):
    """Mean within-pool pairwise L2 over a random subsample of pools."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pools), size=min(subsample_pools, len(pools)), replace=False)
    all_within = []
    for i in idx:
        pool = pools[i]
        if len(pool) < 2:
            continue
        z = vecs[pool].float()
        # Pairwise L2 over pool members (including gold at 0)
        diff = z.unsqueeze(0) - z.unsqueeze(1)   # (P, P, d)
        dists = diff.norm(dim=-1)                 # (P, P)
        triu = dists[torch.triu(torch.ones(len(pool), len(pool), dtype=torch.bool), diagonal=1)]
        all_within.extend(triu.numpy().tolist())
    return float(np.mean(all_within)) if all_within else 0.0


# ─────────────────────────────── NN margin ────────────────────────────────────
def nn_margin_stats(vecs, pools, subsample_pools=512):
    """For each distractor in each pool, compute:
      d_to_gold   = L2(distractor, gold)
      d_to_nn     = L2(distractor, nearest OTHER distractor in pool)
    Report:
      mean/median d_to_gold, mean/median d_to_nn,
      gold_hugging_frac = fraction of distractors with d_to_gold < d_to_nn
        (distractor is closer to gold than to its own nearest neighbour)
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pools), size=min(subsample_pools, len(pools)), replace=False)
    d_gold_all, d_nn_all, gold_hug_all = [], [], []
    for i in idx:
        pool = pools[i]
        P = len(pool)
        if P < 3:
            continue
        z = vecs[pool].float()     # (P, d); z[0] = gold
        gold = z[0]
        distractors = z[1:]        # (P-1, d)
        # dist to gold
        d_to_gold = (distractors - gold.unsqueeze(0)).norm(dim=-1)  # (P-1,)
        # dist to nearest distractor (within distractors only, excluding self)
        diff = distractors.unsqueeze(0) - distractors.unsqueeze(1)  # (P-1, P-1, d)
        dd = diff.norm(dim=-1)                                        # (P-1, P-1)
        # zero diagonal to find nearest neighbour
        dd.fill_diagonal_(float("inf"))
        d_to_nn = dd.min(dim=1).values                               # (P-1,)
        d_gold_all.extend(d_to_gold.numpy().tolist())
        d_nn_all.extend(d_to_nn.numpy().tolist())
        hugging = (d_to_gold < d_to_nn).numpy()
        gold_hug_all.extend(hugging.tolist())

    if not d_gold_all:
        return {"mean_d_to_gold": float("nan"), "median_d_to_gold": float("nan"),
                "mean_d_to_nn": float("nan"), "median_d_to_nn": float("nan"),
                "gold_hugging_frac": float("nan")}
    return {
        "mean_d_to_gold":    float(np.mean(d_gold_all)),
        "median_d_to_gold":  float(np.median(d_gold_all)),
        "mean_d_to_nn":      float(np.mean(d_nn_all)),
        "median_d_to_nn":    float(np.median(d_nn_all)),
        "gold_hugging_frac": float(np.mean(gold_hug_all)),
    }


# ─────────────────────────────── linear probe AUC ────────────────────────────
def linear_probe_auc(cand_vecs, query_vecs, pools, max_neg_per_pool=10, seed=0):
    """Binary LR probe: can (z_candidate, z_query) predict gold vs distractor?

    Feature vector: [z_cand; z_query; |z_cand - z_query|; z_cand * z_query]
    (cosine-style outer-product features, dim = 4*d).
    Label: 1 = gold, 0 = distractor.
    Training: 80% pools, evaluation: 20% pools. Report AUC-ROC.

    `cand_vecs`: the candidate representation (one of ema, online, slot_mean).
    `query_vecs`: zhat (the query-side representation, always the predictor output
                  matching training).
    """
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []
    for qi, pool in enumerate(pools):
        qv = query_vecs[qi].float().numpy()     # (d,)
        gold_idx = pool[0]
        distractors = pool[1:]
        # Gold feature
        cv = cand_vecs[gold_idx].float().numpy()
        feat = np.concatenate([cv, qv, np.abs(cv - qv), cv * qv])
        X_list.append(feat); y_list.append(1)
        # Sample up to max_neg_per_pool distractors
        neg_sel = rng.choice(len(distractors),
                             size=min(max_neg_per_pool, len(distractors)), replace=False)
        for ni in neg_sel:
            cv = cand_vecs[distractors[ni]].float().numpy()
            feat = np.concatenate([cv, qv, np.abs(cv - qv), cv * qv])
            X_list.append(feat); y_list.append(0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    n = len(X)
    perm = rng.permutation(n)
    X = X[perm]; y = y[perm]
    split = int(0.8 * n)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Standardize
    mu = X_tr.mean(0, keepdims=True); std = X_tr.std(0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / std; X_te = (X_te - mu) / std

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
    except Exception as e:
        # Fallback: dot-product scoring (cosine of [cand * query] block)
        d = cand_vecs.shape[1]
        # scores = dot product of first d and second d blocks
        scores_tr = (X_tr[:, :d] * X_tr[:, d:2*d]).sum(1)
        scores_te = (X_te[:, :d] * X_te[:, d:2*d]).sum(1)
        # simple threshold AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_te, scores_te))
        except Exception:
            # pure-numpy AUC by sorting
            order = np.argsort(-scores_te)
            y_sorted = y_te[order]
            n_pos = y_sorted.sum()
            n_neg = len(y_sorted) - n_pos
            if n_pos == 0 or n_neg == 0:
                return float("nan"), str(e)
            tp = np.cumsum(y_sorted)
            fp = np.cumsum(1 - y_sorted)
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc = float(np.trapz(tpr, fpr))
        return auc, f"sklearn_unavailable_fallback: {e}"
    return auc, "sklearn_lr"


# ─────────────────────────────────── main ────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=os.path.expanduser("~/triples_world_model_Glucose"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] model + tokenizer ...", flush=True)
    model, tok, cfg, meta = load(args.ckpt, args.repo)
    T = cfg.data.max_text_tokens
    print(f"[load] epoch={meta['epoch']} M={meta['n_slots']} dn={meta['d_noun']} "
          f"V={meta['n_verbs']} missing={len(meta['missing_keys'])}", flush=True)

    # ── data ──────────────────────────────────────────────────────────────────
    chains = read_chains(Path(args.repo) / "data" / "entity_world" / "test_iid.jsonl")
    src_texts, tgt_texts, chain_id = build_pairs(chains)
    n_all = len(src_texts)
    rng = np.random.default_rng(SEED)
    if n_all > N_POOLS:
        sel = rng.choice(n_all, size=N_POOLS, replace=False)
        sel.sort()
        src_texts = [src_texts[i] for i in sel]
        tgt_texts = [tgt_texts[i] for i in sel]
        chain_id  = chain_id[sel]
    N = len(src_texts)
    print(f"[data] {N} pools from {len(chains)} chains ({n_all} total pairs)", flush=True)

    src_ids, src_pad = encode_batch(tok, src_texts, T)
    tgt_ids, tgt_pad = encode_batch(tok, tgt_texts, T)

    # ── encode ─────────────────────────────────────────────────────────────────
    print("[enc] encoding all target states with 3 variants ...", flush=True)
    ema_vecs, online_vecs, slot_vecs = encode_all_variants(model, tgt_ids, tgt_pad)
    print(f"[enc] shapes: ema={tuple(ema_vecs.shape)} online={tuple(online_vecs.shape)} "
          f"slot_mean={tuple(slot_vecs.shape)}", flush=True)

    print("[enc] encoding query zhat (predictor output) ...", flush=True)
    zhat_vecs = get_query_zhat(model, src_ids, src_pad, tgt_ids, tgt_pad)
    print(f"[enc] zhat shape={tuple(zhat_vecs.shape)}", flush=True)

    # ── pools ──────────────────────────────────────────────────────────────────
    # Hard pools are built from EMA keys (mirrors the MRR head exactly)
    print("[pool] building hard pools ...", flush=True)
    pools = build_pools(ema_vecs, chain_id)
    pool_sizes = [len(p) for p in pools]
    print(f"[pool] mean pool size = {np.mean(pool_sizes):.1f} "
          f"min={min(pool_sizes)} max={max(pool_sizes)}", flush=True)

    # ── noise floor ────────────────────────────────────────────────────────────
    print("[noise] estimating noise floor via pad-shift re-encode ...", flush=True)
    nf = noise_floor(model, tgt_ids, tgt_pad, n_samples=min(512, N))
    for vname, (mn, p90) in nf.items():
        print(f"  noise_floor[{vname}] mean={mn:.4f} p90={p90:.4f}", flush=True)

    # ── global mean pairwise L2 (reference) ───────────────────────────────────
    print("[sep] global mean pairwise L2 (random pairs) ...", flush=True)
    global_l2 = {
        "ema":       global_mean_pairwise_l2(ema_vecs),
        "online":    global_mean_pairwise_l2(online_vecs),
        "slot_mean": global_mean_pairwise_l2(slot_vecs),
    }
    for vname, gl in global_l2.items():
        print(f"  global_pairwise_l2[{vname}] = {gl:.4f}", flush=True)

    # ── per-variant analysis ───────────────────────────────────────────────────
    variant_vecs = {"ema": ema_vecs, "online": online_vecs, "slot_mean": slot_vecs}
    results_per_variant = {}

    for vname, vecs in variant_vecs.items():
        print(f"\n[variant={vname}] ─────────────────────────────────", flush=True)

        # (i) Separation index
        wp_l2 = within_pool_pairwise_l2(vecs, pools)
        sep_idx = wp_l2 / (global_l2[vname] + 1e-12)
        print(f"  within_pool_l2={wp_l2:.4f}  global_l2={global_l2[vname]:.4f}  "
              f"separation_index={sep_idx:.4f}  (1=perfect, 0=fatal)", flush=True)

        # (ii) NN margin
        margin = nn_margin_stats(vecs, pools)
        print(f"  gold_hugging_frac={margin['gold_hugging_frac']:.4f}  "
              f"(fraction of distractors closer to gold than to each other)", flush=True)
        print(f"  mean_d_to_gold={margin['mean_d_to_gold']:.4f}  "
              f"mean_d_to_nn={margin['mean_d_to_nn']:.4f}", flush=True)

        # (iii) Linear probe AUC
        print(f"  [auc] fitting linear probe ...", flush=True)
        auc, method = linear_probe_auc(vecs, zhat_vecs, pools)
        print(f"  AUC={auc:.4f}  method={method}", flush=True)

        nf_mean, nf_p90 = nf[vname]
        results_per_variant[vname] = {
            "separation_index":     sep_idx,
            "within_pool_l2_mean":  wp_l2,
            "global_pairwise_l2":   global_l2[vname],
            "noise_floor_mean":     nf_mean,
            "noise_floor_p90":      nf_p90,
            "nn_margin":            margin,
            "linear_probe_auc":     auc,
            "linear_probe_method":  method,
        }

    # ── verdict ────────────────────────────────────────────────────────────────
    # Primary signal: best AUC across variants
    best_vname = max(results_per_variant, key=lambda v: results_per_variant[v]["linear_probe_auc"])
    best_auc   = results_per_variant[best_vname]["linear_probe_auc"]
    best_sep   = results_per_variant[best_vname]["separation_index"]
    best_ghf   = results_per_variant[best_vname]["nn_margin"]["gold_hugging_frac"]

    # Thresholds (conservative):
    #   AUC > 0.65  -> discriminative signal exists; density head is viable
    #   sep_idx > 0.5 -> candidates spread enough for scoring
    #   gold_hugging_frac < 0.5 -> distractors not massed on gold
    if best_auc >= 0.65 and best_sep >= 0.5:
        verdict = (f"DENSITY-HEAD VIABLE — best AUC={best_auc:.3f} ({best_vname}), "
                   f"sep_idx={best_sep:.3f}, gold_hugging_frac={best_ghf:.3f}. "
                   f"Discriminative signal exists in latent space; a Gaussian density head "
                   f"can learn to rank hard-pool candidates.")
    elif best_auc >= 0.55 and best_sep >= 0.3:
        verdict = (f"MARGINAL — best AUC={best_auc:.3f} ({best_vname}), "
                   f"sep_idx={best_sep:.3f}, gold_hugging_frac={best_ghf:.3f}. "
                   f"Weak discriminative signal. Density head may work with strong "
                   f"regularization but encoder geometry improvement recommended first.")
    else:
        verdict = (f"ENCODER-FIX-FIRST — best AUC={best_auc:.3f} ({best_vname}), "
                   f"sep_idx={best_sep:.3f}, gold_hugging_frac={best_ghf:.3f}. "
                   f"Hard-pool candidates too coincident for a density scorer to rank. "
                   f"Fix encoder geometry before adding the retrieval head.")

    print(f"\n[VERDICT] {verdict}", flush=True)

    # ── save ───────────────────────────────────────────────────────────────────
    out = {
        "meta": meta,
        "n_pools": N,
        "seed": SEED,
        "hard_nn_per_query": HARD_NN,
        "mean_pool_size": float(np.mean(pool_sizes)),
        "noise_floor": nf,
        "global_pairwise_l2": global_l2,
        "variants": results_per_variant,
        "best_variant": best_vname,
        "verdict": verdict,
    }
    out_path = out_dir / "separation_diag.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
