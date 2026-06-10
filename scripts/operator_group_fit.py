"""
operator_group_fit.py — Decide which group a temporal transition operator should live in.

We model a state transition as a single per-element linear operator B acting on a
sentence embedding: B @ z_t ~= z_t+1. We fit four candidate operator families of
increasing expressivity and measure held-out residual, split by reversibility class:

  1. U(1)^(d/2)  : block-diagonal 2x2 rotations (RotatE-style, orthogonal, norm-preserving)
  2. rot+scale   : block-diagonal 2x2 rotation with a per-block scalar scale (can contract/forget)
  3. orthogonal  : full orthogonal Procrustes (SVD with det correction)
  4. general     : unconstrained ridge least squares

Baselines: identity (B=I) and mean-shift (z_t + global mean delta).

Data: GLUCOSE causal chains. We extract (cause, effect) text pairs from the 10
GLUCOSE causal dimensions in GLUCOSE_training_data_final.csv. Each dimension carries
a relation label (Causes/Enables, Motivates, Enables, Causes, Results in). We embed
both sides with a sentence encoder, L2-normalize, and treat (z_cause, z_effect) as
(state_t, state_t+1).

Reversibility labels (two independent strategies, agreement reported):
  A) GLUCOSE-dimension proxy: dims 1-5 are enabling pre-conditions / motivations
     (typically reversible standing states); dims 8-10 "Results in" are changes of
     state/location/possession (typically irreversible). dims 6,7 dropped as ambiguous.
  B) Verb/keyword heuristic on the effect text: consumption/destruction/creation/
     terminal verbs => irreversible; locomotion/manipulation/toggle verbs => reversible.

Run end-to-end:
  uv run --with sentence-transformers --with scikit-learn python scripts/operator_group_fit.py

Falls back to TF-IDF + TruncatedSVD(d=128) if the sentence encoder cannot be loaded.
"""

import os
import csv
import re
import json
import sys
import random

import numpy as np

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "glucose", "GLUCOSE_training_data_final.csv")
OUT_DIR = os.path.join(ROOT, "results", "operator_fit")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_PAIRS = 12000          # cap on extracted pairs before embedding
RIDGE_LAMBDA = 1e-2        # ridge for general-linear fit (on normalized embeddings)
N_CLUSTERS = 16            # verb-codebook clusters on transition direction
MIN_CLUSTER = 40           # min held-in pairs to fit a per-cluster operator


# --------------------------------------------------------------------------------------
# Data extraction
# --------------------------------------------------------------------------------------

# Verb-keyword reversibility heuristic (strategy B), applied to the EFFECT text.
IRREVERSIBLE_KW = [
    r"\beat", r"\bate\b", r"\beaten\b", r"\bdrink", r"\bdrank\b", r"\bdrunk\b",
    r"\bbreak", r"\bbroke", r"\bbroken\b", r"\bshatter", r"\bsmash", r"\bdestroy",
    r"\bdie\b", r"\bdied\b", r"\bdead\b", r"\bkill", r"\bburn", r"\bburnt\b",
    r"\bburned\b", r"\bspill", r"\bmelt", r"\btear\b", r"\btore\b", r"\bripped\b",
    r"\bcrash", r"\bfell\b", r"\bfall", r"\bdrop", r"\bdropped\b", r"\bcut\b",
    r"\bfinish", r"\bcrumble", r"\brot", r"\bdecay", r"\bdissolve", r"\bexplode",
    r"\bwasted\b", r"\blose\b", r"\blost\b", r"\bgone\b", r"\bwon\b", r"\bwin\b",
    r"\bbuilt\b", r"\bcreat", r"\bborn\b", r"\bmade\b",
]
REVERSIBLE_KW = [
    r"\bmove", r"\bmoved\b", r"\bwalk", r"\bran\b", r"\brun\b", r"\bwent\b",
    r"\bgo\b", r"\bopen", r"\bclose", r"\bpick", r"\bput\b", r"\bplace",
    r"\bsit\b", r"\bsat\b", r"\bstand", r"\bstood\b", r"\blook", r"\bhold",
    r"\bheld\b", r"\bturn", r"\bpush", r"\bpull", r"\bcarry", r"\bcarried\b",
    r"\bswim", r"\bdrive", r"\bdrove\b", r"\bvisit", r"\bgrab", r"\bwait",
    r"\bat (home|the)", r"\bin (the|bed|her|his)",
]
_IRR_RE = re.compile("|".join(IRREVERSIBLE_KW), re.I)
_REV_RE = re.compile("|".join(REVERSIBLE_KW), re.I)

# GLUCOSE-dimension proxy (strategy A): map dim index -> reversibility label.
# dims 1-5 = antecedent enabling states / motivations -> reversible (standing states)
# dims 8-10 = "Results in" changes of state/location/possession -> irreversible
# dims 6,7  = ambiguous causal/emotional consequents -> drop from the proxy-labeled set
DIM_PROXY = {1: "rev", 2: "rev", 3: "rev", 4: "rev", 5: "rev",
             6: None, 7: None, 8: "irr", 9: "irr", 10: "irr"}


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def extract_pairs():
    """Yield dicts with cause text, effect text, dim, relation."""
    pairs = []
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for dim in range(1, 11):
                nl = row.get(f"{dim}_specificNL", "")
                if not nl or nl == "escaped":
                    continue
                m = re.split(r">\s*([^>]+?)\s*>", nl)
                if len(m) != 3:
                    continue
                cause, rel, effect = clean(m[0]), clean(m[1]), clean(m[2])
                if len(cause) < 4 or len(effect) < 4:
                    continue
                pairs.append({"cause": cause, "effect": effect, "dim": dim, "rel": rel})
    return pairs


def kw_label(effect):
    """Strategy B: keyword vote on effect text. Returns 'rev', 'irr', or None."""
    irr = bool(_IRR_RE.search(effect))
    rev = bool(_REV_RE.search(effect))
    if irr and not rev:
        return "irr"
    if rev and not irr:
        return "rev"
    return None  # ambiguous / no signal


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------

def embed_texts(texts):
    """Return (embeddings[N,d], backend_name). L2-normalized."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, normalize_embeddings=True,
                           batch_size=256, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float64), "all-MiniLM-L6-v2"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sentence-transformers failed ({e!r}); falling back to TF-IDF+SVD",
              file=sys.stderr)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        d = min(128, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=d, random_state=SEED)
        emb = svd.fit_transform(X)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return emb.astype(np.float64), f"tfidf-svd-{d}"


# --------------------------------------------------------------------------------------
# Operator families
# --------------------------------------------------------------------------------------

def fit_orthogonal(Zt, Zt1):
    """Full orthogonal Procrustes: min ||B Zt - Zt1||, B orthogonal (det free)."""
    # B = U V^T from SVD of cross-covariance M = Zt1^T Zt  (columns are samples? use rows)
    # rows are samples: want B z_t ~= z_t1  =>  Zt1 ~= Zt B^T  (B applied to each row)
    M = Zt1.T @ Zt                      # (d,d)
    U, _, Vt = np.linalg.svd(M)
    B = U @ Vt
    return B


def fit_general(Zt, Zt1, lam=RIDGE_LAMBDA):
    """Ridge least squares: B = Zt1^T Zt (Zt^T Zt + lam I)^-1 (rows are samples)."""
    d = Zt.shape[1]
    A = Zt.T @ Zt + lam * np.eye(d)
    B = (Zt1.T @ Zt) @ np.linalg.inv(A)
    return B


def _pair_blocks(d):
    return [(2 * k, 2 * k + 1) for k in range(d // 2)]


def fit_block_rotation(Zt, Zt1, with_scale=False):
    """
    Block-diagonal 2x2 operator. Per block we fit either a pure rotation (U(1)) or a
    rotation with a single positive scale (rot+scale). Closed-form per 2D block via the
    2x2 orthogonal Procrustes solution.

    For samples (a=Zt block, b=Zt1 block), each row 2D:
      cross-cov H = sum_i a_i b_i^T  (2x2). SVD H = U S V^T, rotation R = V U^T
      (maps a -> b). Scale s = trace(S)/||a||^2 (least-squares optimal scalar).
    """
    d = Zt.shape[1]
    blocks = _pair_blocks(d)
    Bs = []
    for (i, j) in blocks:
        a = Zt[:, [i, j]]      # (N,2)
        b = Zt1[:, [i, j]]     # (N,2)
        H = a.T @ b            # (2,2)
        U, S, Vt = np.linalg.svd(H)
        # rotation mapping a -> b : R = V @ diag(1,det) @ U^T (proper rotation, det=+1)
        D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        if with_scale:
            denom = (a * a).sum() + 1e-12
            s = (S * np.diag(D)).sum() / denom  # optimal positive-ish scalar (Umeyama)
            R = s * R
        Bs.append(R)
    # assemble block-diagonal B; handle odd d by leaving last dim as identity 1x1
    B = np.zeros((d, d))
    for k, (i, j) in enumerate(blocks):
        B[np.ix_([i, j], [i, j])] = Bs[k]
    if d % 2 == 1:
        B[d - 1, d - 1] = 1.0
    return B


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def residual_metrics(B, Zt, Zt1):
    pred = Zt @ B.T
    err = pred - Zt1
    mse = float((err * err).sum(axis=1).mean())          # mean squared L2 residual
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    cos = float((pn * Zt1).sum(axis=1).mean())           # Zt1 already unit norm
    return {"residual_mse": mse, "cosine": cos}


def mean_shift_metrics(delta, Zt, Zt1):
    pred = Zt + delta
    err = pred - Zt1
    mse = float((err * err).sum(axis=1).mean())
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    cos = float((pn * Zt1).sum(axis=1).mean())
    return {"residual_mse": mse, "cosine": cos}


def identity_metrics(Zt, Zt1):
    err = Zt - Zt1
    mse = float((err * err).sum(axis=1).mean())
    cos = float((Zt * Zt1).sum(axis=1).mean())
    return {"residual_mse": mse, "cosine": cos}


FAMILIES = ["u1_rotation", "rot_scale", "orthogonal", "general"]


def fit_family(name, Zt, Zt1):
    if name == "u1_rotation":
        return fit_block_rotation(Zt, Zt1, with_scale=False)
    if name == "rot_scale":
        return fit_block_rotation(Zt, Zt1, with_scale=True)
    if name == "orthogonal":
        return fit_orthogonal(Zt, Zt1)
    if name == "general":
        return fit_general(Zt, Zt1)
    raise ValueError(name)


def eval_split(name, Zt_tr, Zt1_tr, Zt_te, Zt1_te):
    B = fit_family(name, Zt_tr, Zt1_tr)
    return residual_metrics(B, Zt_te, Zt1_te), B


def global_and_clustered(Zt_tr, Zt1_tr, Zt_te, Zt1_te, clusters_tr=None, clusters_te=None):
    """Return dict family -> {global:{...}, clustered:{...}} plus baselines."""
    out = {}
    # baselines
    out["identity"] = {"global": identity_metrics(Zt_te, Zt1_te)}
    delta = (Zt1_tr - Zt_tr).mean(axis=0)
    out["mean_shift"] = {"global": mean_shift_metrics(delta, Zt_te, Zt1_te)}

    for fam in FAMILIES:
        rec = {}
        rec["global"], _ = eval_split(fam, Zt_tr, Zt1_tr, Zt_te, Zt1_te)
        if clusters_tr is not None:
            # per-cluster operator; eval each test point with its cluster's B
            pred = np.empty_like(Zt1_te)
            valid = np.zeros(len(Zt_te), dtype=bool)
            for c in np.unique(clusters_tr):
                tr_mask = clusters_tr == c
                te_mask = clusters_te == c
                if tr_mask.sum() < MIN_CLUSTER or te_mask.sum() == 0:
                    continue
                B = fit_family(fam, Zt_tr[tr_mask], Zt1_tr[tr_mask])
                pred[te_mask] = Zt_te[te_mask] @ B.T
                valid[te_mask] = True
            if valid.any():
                p, t = pred[valid], Zt1_te[valid]
                err = p - t
                pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
                rec["clustered"] = {
                    "residual_mse": float((err * err).sum(axis=1).mean()),
                    "cosine": float((pn * t).sum(axis=1).mean()),
                    "coverage": float(valid.mean()),
                }
        out[fam] = rec
    return out


def spectrum(B):
    s = np.linalg.svd(B, compute_uv=False)
    return s


def effective_rank(M):
    """Effective rank of a set of row vectors (transition directions) via entropy of
    normalized singular values of the covariance."""
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("[1/6] extracting GLUCOSE cause->effect pairs ...")
    pairs = extract_pairs()
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    print(f"  extracted {len(pairs)} pairs (capped at {MAX_PAIRS})")

    # Two reversibility labels
    for p in pairs:
        p["lab_dim"] = DIM_PROXY[p["dim"]]
        p["lab_kw"] = kw_label(p["effect"])

    # agreement on pairs where both fire
    both = [p for p in pairs if p["lab_dim"] and p["lab_kw"]]
    agree = sum(p["lab_dim"] == p["lab_kw"] for p in both)
    agreement = agree / len(both) if both else float("nan")
    print(f"  label agreement (both strategies fired, n={len(both)}): {agreement:.3f}")

    # Embed
    print("[2/6] embedding ...")
    causes = [p["cause"] for p in pairs]
    effects = [p["effect"] for p in pairs]
    Zc, backend = embed_texts(causes)
    Ze, _ = embed_texts(effects)
    print(f"  backend={backend} d={Zc.shape[1]}")

    Zt = Zc
    Zt1 = Ze

    # Clustering on transition direction (verb codebook proxy)
    print("[3/6] clustering transition directions ...")
    from sklearn.cluster import KMeans
    direction = Zt1 - Zt
    dn = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=4).fit(dn)
    clusters = km.labels_

    # train/test split
    n = len(pairs)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    def subset(mask_label=None, strategy=None):
        """Return train/test index arrays restricted to a reversibility class (or all)."""
        if mask_label is None:
            return tr, te
        key = "lab_dim" if strategy == "dim" else "lab_kw"
        keep = np.array([pairs[i][key] == mask_label for i in range(n)])
        return tr[keep[tr]], te[keep[te]]

    print("[4/6] fitting operator families (global + clustered) per class ...")
    results = {"backend": backend, "n_pairs": n, "d": int(Zt.shape[1]),
               "label_agreement": agreement, "n_clusters": N_CLUSTERS,
               "ridge_lambda": RIDGE_LAMBDA, "seed": SEED, "splits": {}}

    def run_class(name, tr_i, te_i):
        if len(tr_i) < 50 or len(te_i) < 20:
            return None
        res = global_and_clustered(
            Zt[tr_i], Zt1[tr_i], Zt[te_i], Zt1[te_i],
            clusters_tr=clusters[tr_i], clusters_te=clusters[te_i],
        )
        # general-linear spectrum on this class (global fit)
        B_gen = fit_general(Zt[tr_i], Zt1[tr_i])
        s = spectrum(B_gen)
        res["_meta"] = {
            "n_train": int(len(tr_i)), "n_test": int(len(te_i)),
            "gen_singvals": s.tolist(),
            "gen_sv_lt1_frac": float((s < 1.0).mean()),
            "gen_sv_mean": float(s.mean()),
            "gen_sv_median": float(np.median(s)),
            "transition_effective_rank": effective_rank((Zt1 - Zt)[tr_i]),
        }
        return res

    # all pairs
    results["splits"]["all"] = run_class("all", tr, te)

    # by dimension proxy
    for lab in ("rev", "irr"):
        a, b = subset(lab, "dim")
        results["splits"][f"dim_{lab}"] = run_class(f"dim_{lab}", a, b)
    # by keyword
    for lab in ("rev", "irr"):
        a, b = subset(lab, "kw")
        results["splits"][f"kw_{lab}"] = run_class(f"kw_{lab}", a, b)

    # class counts
    results["class_counts"] = {
        "dim_rev": int(sum(p["lab_dim"] == "rev" for p in pairs)),
        "dim_irr": int(sum(p["lab_dim"] == "irr" for p in pairs)),
        "kw_rev": int(sum(p["lab_kw"] == "rev" for p in pairs)),
        "kw_irr": int(sum(p["lab_kw"] == "irr" for p in pairs)),
    }

    print("[5/6] writing results json ...")
    with open(os.path.join(OUT_DIR, "operator_fit_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    print("[6/6] plotting ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (1) residual bars: family x class (global)
        classes = ["all", "dim_rev", "dim_irr", "kw_rev", "kw_irr"]
        fams = ["identity", "mean_shift"] + FAMILIES
        fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.15
        x = np.arange(len(classes))
        for fi, fam in enumerate(fams):
            vals = []
            for c in classes:
                r = results["splits"].get(c)
                v = (r.get(fam, {}).get("global", {}) or {}).get("residual_mse", np.nan) if r else np.nan
                vals.append(v)
            ax.bar(x + fi * width, vals, width, label=fam)
        ax.set_xticks(x + width * (len(fams) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.set_ylabel("held-out residual MSE")
        ax.set_title("Operator family residual by reversibility class (global fit)")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "residual_by_family_class.png"), dpi=130)
        plt.close(fig)

        # (2) singular value spectra of general-linear map, rev vs irr (dim proxy)
        fig, ax = plt.subplots(figsize=(9, 5))
        for c, color in [("dim_rev", "tab:blue"), ("dim_irr", "tab:red"),
                         ("kw_rev", "tab:cyan"), ("kw_irr", "tab:orange")]:
            r = results["splits"].get(c)
            if r and "_meta" in r:
                s = np.array(r["_meta"]["gen_singvals"])
                ax.plot(np.sort(s)[::-1], label=f"{c} (frac<1={r['_meta']['gen_sv_lt1_frac']:.2f})",
                        color=color, lw=1.5)
        ax.axhline(1.0, color="k", ls="--", lw=0.8, label="s=1 (norm preserving)")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("singular value")
        ax.set_title("General-linear operator spectrum: rev vs irr\n(s<1 = contraction = scale DOF needed)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "singular_spectra.png"), dpi=130)
        plt.close(fig)
        print("  plots written")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plotting failed: {e!r}", file=sys.stderr)

    # console summary
    print("\n=== SUMMARY (held-out residual MSE, global fit) ===")
    hdr = ["class"] + ["identity", "mean_shift"] + FAMILIES
    print("  " + " | ".join(f"{h:>11}" for h in hdr))
    for c in ["all", "dim_rev", "dim_irr", "kw_rev", "kw_irr"]:
        r = results["splits"].get(c)
        if not r:
            continue
        row = [c]
        for fam in ["identity", "mean_shift"] + FAMILIES:
            v = (r.get(fam, {}).get("global", {}) or {}).get("residual_mse", float("nan"))
            row.append(f"{v:.4f}")
        print("  " + " | ".join(f"{x:>11}" for x in row))

    print("\n=== general-linear spectrum (s<1 fraction = contraction evidence) ===")
    for c in ["all", "dim_rev", "dim_irr", "kw_rev", "kw_irr"]:
        r = results["splits"].get(c)
        if r and "_meta" in r:
            m = r["_meta"]
            print(f"  {c:>8}: frac(s<1)={m['gen_sv_lt1_frac']:.3f}  mean_s={m['gen_sv_mean']:.3f}  "
                  f"median_s={m['gen_sv_median']:.3f}  eff_rank(dir)={m['transition_effective_rank']:.1f}")

    print(f"\nartifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
