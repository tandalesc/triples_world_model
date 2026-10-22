#!/usr/bin/env python3
"""Gradient-boosted trees (XGBoost) as a baseline student of Hearthvale's decision model.

    python scripts/train_decision_gbdt.py DATA.jsonl [--kinds a,b] [--depth 6] [--rounds 400] [--out metrics.json]

Needs only numpy, scipy and xgboost (not torch), so it can run from a small
separate environment. On macOS xgboost needs libomp: scikit-learn's wheel
bundles one (DYLD_LIBRARY_PATH=.../site-packages/sklearn/.dylibs).

Same data, same split by town and same metrics as train_decision_student.py
(``split_by_seed`` below must stay identical to twm/decision/data.py's), so
the numbers compare directly. One model per kind: one-hot (scope, attr, value)
columns, one numeric column per (scope, attr) that carries numbers (missing
elsewhere), and multi:softprob over the kind's options trained on the
teacher's distribution (each row expanded into one weighted row per option).
Early stopping uses towns held out from the training towns, never the test towns.

``--holdout-pairs FILE.json`` holds out rows with both triples of a pair, as
train_decision_student.py does, and scores them separately.

Also reports what the trees would cost in the browser: trees, nodes, an
estimated packed size, and node visits per question.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
import xgboost as xgb


def split_by_seed(rows, test_fraction=0.2, salt=0):
    seeds = sorted({s for r in rows for s in r.get("seeds", [0])})
    held = {s for s in seeds if (((s * 2654435761) ^ salt) & 0xFFFFFFFF) / 2**32 < test_fraction}
    if len(seeds) > 1 and not held:
        held = {seeds[-1]}
    if len(seeds) > 1 and len(held) == len(seeds):
        held.discard(seeds[0])
    train, test = [], []
    for r in rows:
        (test if all(s in held for s in r.get("seeds", [0])) else train).append(r)
    return train, test


def holdout_split(rows, pairs):
    keep, held = [], []
    for r in rows:
        keys = None
        for p in pairs:
            if p["kind"] != r["kind"]:
                continue
            keys = keys or {f"{t[0]}.{t[1]}={t[2]}" for t in r["triples"]}
            if p["a"] in keys and p["b"] in keys:
                held.append({**r, "pair": f"{p['kind']}: {p['a']} × {p['b']}"})
                break
        else:
            keep.append(r)
    return keep, held


class KindFeatures:
    def __init__(self, rows, min_count=3):
        cats = Counter()
        nums = set()
        for r in rows:
            for s, a, v, x in r["triples"]:
                if x is None:
                    cats[(s, a, v)] += 1
                else:
                    nums.add((s, a, v))
        self.cat = {k: i for i, k in enumerate(sorted(k for k, c in cats.items() if c >= min_count))}
        self.num = {k: len(self.cat) + i for i, k in enumerate(sorted(nums))}
        self.width = len(self.cat) + len(self.num)
        options = []
        for r in rows:
            for o in r["options"]:
                if o not in options:
                    options.append(o)
        self.options = options

    def matrix(self, rows):
        data, ind, ptr = [], [], [0]
        for r in rows:
            seen = {}
            for s, a, v, x in r["triples"]:
                if x is None:
                    j = self.cat.get((s, a, v))
                    if j is not None:
                        seen[j] = 1.0
                else:
                    j = self.num.get((s, a, v))
                    if j is not None:
                        seen[j] = float(x)
            for j, val in sorted(seen.items()):
                ind.append(j)
                data.append(val)
            ptr.append(len(ind))
        # Absent entries of a CSR matrix are "missing" to xgboost: for a one-hot
        # column that sends them down the learned default branch, as 0 would.
        return sp.csr_matrix((np.array(data, dtype=np.float32), np.array(ind), np.array(ptr)), shape=(len(rows), self.width))

    def targets(self, rows):
        k = len(self.options)
        t = np.zeros((len(rows), k), dtype=np.float32)
        offered = np.zeros((len(rows), k), dtype=bool)
        for i, r in enumerate(rows):
            for o, p in zip(r["options"], r["target"]):
                j = self.options.index(o)
                t[i, j] = p
                offered[i, j] = True
        t /= np.maximum(t.sum(1, keepdims=True), 1e-9)
        return t, offered


def expand(X, T, w):
    """One row per (row, option) with weight p · w: cross-entropy on soft labels."""
    rows, labels, weights = [], [], []
    for i in range(T.shape[0]):
        for j in np.nonzero(T[i] > 1e-4)[0]:
            rows.append(i)
            labels.append(j)
            weights.append(T[i, j] * w[i])
    return X[rows], np.array(labels), np.array(weights, dtype=np.float32)


def kl_rows(T, P, offered):
    P = np.where(offered, P, 0.0)
    P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(T > 0, T * (np.log(np.maximum(T, 1e-12)) - np.log(np.maximum(P, 1e-12))), 0.0)
    return terms.sum(1), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--kinds", default=None)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--min-value-count", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--holdout-pairs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kinds = args.kinds.split(",") if args.kinds else None
    rows = []
    for path in args.data:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not kinds or r["kind"] in kinds:
                    rows.append(r)
    pairs = json.load(open(args.holdout_pairs)) if args.holdout_pairs else []
    rows, held_rows = holdout_split(rows, pairs)
    train_rows, test_rows = split_by_seed(rows, args.test_fraction)
    fit_rows, val_rows = split_by_seed(train_rows, 0.15, salt=0x5BD1E995)
    print(f"{len(rows)} rows: {len(train_rows)} train ({len(fit_rows)} fit, {len(val_rows)} early-stopping), {len(test_rows)} test (held-out towns)")

    by_kind = defaultdict(lambda: ([], [], [], []))
    for i, part in enumerate((fit_rows, val_rows, test_rows, held_rows)):
        for r in part:
            by_kind[r["kind"]][i].append(r)

    results, all_kl, all_top = {}, [], []
    trees_total = nodes_total = visits_total = 0
    t0 = time.time()
    held_results = {}
    for kind, (fit, val, test, held) in sorted(by_kind.items(), key=lambda kv: -len(kv[1][2])):
        if not fit or not test:
            continue
        feats = KindFeatures(fit, args.min_value_count)
        k = len(feats.options)
        w_fit = np.array([math.sqrt(max(1, r.get("n", 1))) for r in fit], dtype=np.float32)
        Xf, Tf = feats.matrix(fit), feats.targets(fit)[0]
        Xe, ye, we = expand(Xf, Tf, w_fit / w_fit.mean())
        dfit = xgb.DMatrix(Xe, label=ye, weight=we)
        evals = []
        if val:
            Xv, Tv = feats.matrix(val), feats.targets(val)[0]
            Xve, yve, wve = expand(Xv, Tv, np.ones(len(val), dtype=np.float32))
            evals = [(xgb.DMatrix(Xve, label=yve, weight=wve), "val")]
        params = {"objective": "multi:softprob", "num_class": k, "max_depth": args.depth, "eta": args.eta,
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "lambda": 1.0,
                  "eval_metric": "mlogloss", "tree_method": "hist", "nthread": args.threads}
        booster = xgb.train(params, dfit, args.rounds, evals=evals, early_stopping_rounds=30 if evals else None, verbose_eval=False)
        best = (booster.best_iteration + 1) if evals else args.rounds
        Tt, offered = feats.targets(test)
        P = booster.predict(xgb.DMatrix(feats.matrix(test)), iteration_range=(0, best))
        kl, P = kl_rows(Tt, P, offered)
        top = P.argmax(1) == Tt.argmax(1)
        trees = booster[:best].get_dump(dump_format="json")
        n_nodes = sum(t.count('"nodeid"') for t in trees)
        depth_visits = len(trees) * args.depth
        trees_total += len(trees)
        nodes_total += n_nodes
        visits_total += depth_visits * len(test)
        results[kind] = {"rows": len(test), "kl": float(kl.mean()), "top1": float(top.mean()), "rounds": best, "trees": len(trees), "nodes": n_nodes}
        all_kl.append(kl)
        all_top.append(top)
        if held:
            Th, offered_h = feats.targets(held)
            Ph = booster.predict(xgb.DMatrix(feats.matrix(held)), iteration_range=(0, best))
            hkl, Ph = kl_rows(Th, Ph, offered_h)
            htop = Ph.argmax(1) == Th.argmax(1)
            for name in sorted({r["pair"] for r in held}):
                ix = np.array([r["pair"] == name for r in held])
                held_results[name] = {"rows": int(ix.sum()), "kl": float(hkl[ix].mean()), "top1": float(htop[ix].mean())}
                print(f"held-out pair {name}: {int(ix.sum())} rows, KL {hkl[ix].mean():.4f}, top-1 {htop[ix].mean():.1%}")
        print(f"{kind:26s} {len(test):7d} {kl.mean():8.4f} {top.mean():6.1%}   {best:4d} rounds, {len(trees):5d} trees, {n_nodes:7d} nodes  ({time.time() - t0:.0f} s)")

    kl = np.concatenate(all_kl)
    top = np.concatenate(all_top)
    # Packed for the browser: per node a feature index (u16), threshold (f32),
    # two child offsets (u16 each) or a leaf value (f32): about 10 bytes a node.
    size_kb = nodes_total * 10 / 1024
    summary = {"all": {"rows": int(len(kl)), "kl": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)), "top1": float(top.mean())},
               "browser": {"trees": trees_total, "nodes": nodes_total, "packed_kb_estimate": size_kb,
                           "node_visits_per_question": visits_total / max(1, len(kl))},
               "kinds": results, "held-out pairs": held_results}
    print(f"\nall {len(kl)} rows: KL {kl.mean():.4f}, worst 1% above {np.quantile(kl, 0.99):.4f}, top-1 {top.mean():.1%}")
    print(f"browser: {trees_total} trees, {nodes_total:,} nodes, ~{size_kb:.0f} KB packed, ~{summary['browser']['node_visits_per_question']:.0f} node visits a question")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "test": summary}, f, indent=2)


if __name__ == "__main__":
    main()
