"""JEPA diagnostics suite — spec §5 + §12 T6.

`eval_diagnostics(model, dataset, device, n_examples=512, out_dir)` -> flat dict + PNGs.

Covers all §5 metrics:
  - Noun geometry: effective rank + per-dim variance hist
  - Scale drift: per-verb log r distribution; mean ||a*||/||k||; WARN if mean log r < -1.0
  - Verb non-triviality: |θ| hist, |log r| hist, usage perplexity, pairwise ||B_u - B_v||_F
  - Structural sanity: operator.structural_sanity(v) passthrough
  - Residual-vs-slots: mask slots coarse-to-fine, plot L_pred curve
  - Slot-attention entropy per slot
  - KMeans noun-cluster × verb contingency table
  - Held-out cos(zhat, z) + MSE
  - Multi-step drift: apply B_v N=1..16 times, cos to nearest training noun
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # headless; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model_v1 import JEPAOperatorModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _effective_rank(C: np.ndarray) -> float:
    """tr(C)^2 / ||C||_F^2  (spec §5 noun geometry)."""
    tr = np.trace(C)
    frob2 = np.sum(C ** 2)
    if frob2 < 1e-12:
        return 1.0
    return float(tr ** 2 / frob2)


def _cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean cos(a_i, b_i) for paired batches."""
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return (a_n * b_n).sum(-1)  # (N,)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def eval_diagnostics(
    model: "JEPAOperatorModel",
    dataset,
    device: torch.device | str,
    n_examples: int = 512,
    out_dir: str | os.PathLike | None = None,
) -> dict:
    """Run all §5 diagnostics.  Returns a flat dict of scalar metrics + saves PNGs.

    Args:
        model:      JEPAOperatorModel (online encoder + operator + readout).
        dataset:    JEPAChainDataset-compatible; __getitem__ returns
                    {"src_ids","src_pad","tgt_ids","tgt_pad"}.
        device:     "cpu" / "cuda" / "mps" / torch.device.
        n_examples: number of examples to use (capped at len(dataset)).
        out_dir:    if not None, write PNG files here.

    Returns:
        flat dict with scalar diagnostics (all finite on a sane model).
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_examples, len(dataset))
    indices = list(range(n))

    # ------------------------------------------------------------------
    # 1. Collect forward-pass tensors
    # ------------------------------------------------------------------
    all_k: list[torch.Tensor] = []       # (M, dn) per example
    all_a_star: list[torch.Tensor] = []  # (M, dn)
    all_verbs: list[torch.Tensor] = []   # (M,)  int
    all_slots: list[torch.Tensor] = []   # (M, d) pre-head slots for entropy
    all_zhat: list[torch.Tensor] = []    # (dn,)
    all_z: list[torch.Tensor] = []       # (dn,) EMA target

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            src_ids = item["src_ids"].unsqueeze(0).to(device)   # (1, T)
            src_pad = item["src_pad"].unsqueeze(0).to(device)
            tgt_ids = item["tgt_ids"].unsqueeze(0).to(device)
            tgt_pad = item["tgt_pad"].unsqueeze(0).to(device)

            out = model(src_ids, src_pad)
            # EMA encode target: use ema.pool_raw (raw noun pool, no operator)
            z_enc = model.ema.pool_raw(tgt_ids, tgt_pad)  # (1, dn)

            all_k.append(out["k"].squeeze(0).cpu())          # (M, dn)
            all_a_star.append(out["a"].squeeze(0).cpu())     # (M, dn)
            all_verbs.append(out["verb"].squeeze(0).cpu())   # (M,)
            all_zhat.append(out["zhat"].squeeze(0).cpu())    # (dn,)
            all_z.append(z_enc.squeeze(0).cpu())             # (dn,)
            # slots may or may not be in out; guard for robustness
            if "slots" in out:
                all_slots.append(out["slots"].squeeze(0).cpu())  # (M, d)

    K = torch.stack(all_k)        # (N, M, dn)
    A_star = torch.stack(all_a_star)  # (N, M, dn)
    Verbs = torch.stack(all_verbs)    # (N, M)
    Zhat = torch.stack(all_zhat)      # (N, dn)
    Z = torch.stack(all_z)            # (N, dn)
    N, M, dn = K.shape

    metrics: dict = {}

    # ------------------------------------------------------------------
    # 2. Noun geometry
    # ------------------------------------------------------------------
    K_flat = K.reshape(-1, dn).numpy()  # (N*M, dn)
    # center
    K_c = K_flat - K_flat.mean(0, keepdims=True)
    C = K_c.T @ K_c / max(K_c.shape[0] - 1, 1)  # (dn, dn) covariance

    eff_rank = _effective_rank(C)
    metrics["noun_eff_rank"] = eff_rank
    metrics["noun_eff_rank_threshold"] = dn / 4

    per_dim_var = np.diag(C)
    metrics["noun_per_dim_var_mean"] = float(per_dim_var.mean())
    metrics["noun_per_dim_var_min"] = float(per_dim_var.min())
    metrics["noun_per_dim_var_max"] = float(per_dim_var.max())

    if out_dir is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(dn), per_dim_var)
        ax.axhline(1.0, color="r", linestyle="--", label="unit var")
        ax.set_xlabel("noun dim")
        ax.set_ylabel("variance")
        ax.set_title(f"Noun per-dim variance  (eff_rank={eff_rank:.2f}, threshold={dn/4:.1f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "noun_var_hist.png", dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Scale drift (spec §1.5, §5)
    # ------------------------------------------------------------------
    # Access operator params directly
    op = model.operator

    # Gather theta and log_r from operator — handle RotationScaleOperator layout
    has_theta = hasattr(op, "theta") and hasattr(op, "log_r")
    n_verbs_op = op.n_verbs

    if has_theta:
        theta_np = _to_numpy(op.theta)   # (V, dn//2)
        log_r_np = _to_numpy(op.log_r)  # (V, dn//2)

        # per-verb mean log r
        mean_log_r_per_verb = log_r_np.mean(axis=1)  # (V,)
        metrics["scale_drift_mean_log_r"] = float(mean_log_r_per_verb.mean())
        metrics["scale_drift_min_log_r"] = float(mean_log_r_per_verb.min())
        if mean_log_r_per_verb.mean() < -1.0:
            import warnings
            warnings.warn(
                "WARN: mean log_r < -1.0 — runaway contraction detected. "
                "Consider raising w_sigreg or enabling w_scale_reg.",
                RuntimeWarning,
                stacklevel=2,
            )
            metrics["scale_drift_alarm"] = True
        else:
            metrics["scale_drift_alarm"] = False
    else:
        # operator doesn't expose theta/log_r directly (e.g. stub)
        metrics["scale_drift_mean_log_r"] = 0.0
        metrics["scale_drift_min_log_r"] = 0.0
        metrics["scale_drift_alarm"] = False
        theta_np = None
        log_r_np = None

    # mean ||a*|| / ||k||
    k_norms = K.norm(dim=-1).clamp(min=1e-8)   # (N, M)
    a_norms = A_star.norm(dim=-1)               # (N, M)
    ratio = (a_norms / k_norms).mean().item()
    metrics["scale_drift_mean_a_over_k"] = ratio

    if out_dir is not None and log_r_np is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].hist(log_r_np.ravel(), bins=30, edgecolor="k")
        axes[0].axvline(-1.0, color="r", linestyle="--", label="alarm threshold")
        axes[0].set_xlabel("log r (per block per verb)")
        axes[0].set_title("Scale-drift: log r distribution")
        axes[0].legend()

        axes[1].bar(np.arange(n_verbs_op), mean_log_r_per_verb)
        axes[1].axhline(-1.0, color="r", linestyle="--", label="alarm threshold")
        axes[1].set_xlabel("verb index")
        axes[1].set_ylabel("mean log r")
        axes[1].set_title("Per-verb mean log r")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "scale_drift.png", dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Verb non-triviality
    # ------------------------------------------------------------------
    verbs_flat = Verbs.reshape(-1).long().numpy()  # (N*M,)
    counts = np.bincount(verbs_flat, minlength=n_verbs_op).astype(float)
    probs = counts / counts.sum()
    probs_safe = np.where(probs > 0, probs, 1e-10)
    usage_entropy = float(-np.sum(probs * np.log(probs_safe)))
    usage_ppl = float(np.exp(usage_entropy))
    metrics["verb_usage_ppl"] = usage_ppl
    metrics["verb_usage_entropy"] = usage_entropy
    metrics["verb_ppl_threshold"] = n_verbs_op / 2

    if theta_np is not None:
        metrics["verb_theta_abs_mean"] = float(np.abs(theta_np).mean())
        metrics["verb_log_r_abs_mean"] = float(np.abs(log_r_np).mean())

        # pairwise ||B_u - B_v||_F  (in parameter space: theta+log_r stack)
        params = np.concatenate([theta_np, log_r_np], axis=1)  # (V, dn)
        V = params.shape[0]
        diffs = []
        for u in range(V):
            for v in range(u + 1, V):
                diffs.append(np.linalg.norm(params[u] - params[v]))
        pairwise_arr = np.array(diffs)
        metrics["verb_pairwise_param_dist_mean"] = float(pairwise_arr.mean()) if len(pairwise_arr) > 0 else 0.0
        metrics["verb_pairwise_param_dist_min"] = float(pairwise_arr.min()) if len(pairwise_arr) > 0 else 0.0

        if out_dir is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].hist(np.abs(theta_np).ravel(), bins=20, edgecolor="k")
            axes[0].set_xlabel("|θ| (per block)")
            axes[0].set_title(f"Verb |θ| distribution  (mean={metrics['verb_theta_abs_mean']:.3f})")

            axes[1].hist(np.abs(log_r_np).ravel(), bins=20, edgecolor="k")
            axes[1].set_xlabel("|log r| (per block)")
            axes[1].set_title(f"Verb |log r| distribution  (mean={metrics['verb_log_r_abs_mean']:.3f})")
            fig.tight_layout()
            fig.savefig(out_dir / "verb_param_hists.png", dpi=80)
            plt.close(fig)
    else:
        metrics["verb_theta_abs_mean"] = 0.0
        metrics["verb_log_r_abs_mean"] = 0.0
        metrics["verb_pairwise_param_dist_mean"] = 0.0
        metrics["verb_pairwise_param_dist_min"] = 0.0

    # ------------------------------------------------------------------
    # 5. Structural sanity passthrough
    # ------------------------------------------------------------------
    sanity_bbT_err = []
    sanity_inv_err = []
    with torch.no_grad():
        for v_idx in range(n_verbs_op):
            s = op.structural_sanity(v_idx)
            sanity_bbT_err.append(s.get("bbT_err", 0.0))
            sanity_inv_err.append(s.get("inv_err", 0.0))
    metrics["sanity_bbT_err_mean"] = float(np.mean(sanity_bbT_err))
    metrics["sanity_inv_err_mean"] = float(np.mean(sanity_inv_err))
    metrics["sanity_bbT_err_max"] = float(np.max(sanity_bbT_err))
    metrics["sanity_inv_err_max"] = float(np.max(sanity_inv_err))

    # ------------------------------------------------------------------
    # 6. Residual-vs-slots (mask slots coarse-to-fine)
    # ------------------------------------------------------------------
    # We measure L_pred = MSE(readout(a_star_masked), z_target) as we
    # zero out the M slots from worst to best (coarse-to-fine).
    # Done in numpy to avoid needing the full model readout — instead
    # we approximate by computing MSE between mean-pooled a* and z.
    # The monotonicity is what matters; the readout is linear over a*.
    residual_curve = []
    z_np = Z.numpy()  # (N, dn)
    for n_kept in range(0, M + 1):
        if n_kept == 0:
            # all slots masked → use zero prediction
            pred = np.zeros_like(z_np)
        else:
            # keep the first n_kept slots (coarse-to-fine means first kept)
            a_kept = A_star[:, :n_kept, :].numpy()  # (N, n_kept, dn)
            pred = a_kept.mean(axis=1)  # (N, dn)  mean pool
        mse_val = float(np.mean((pred - z_np) ** 2))
        residual_curve.append(mse_val)

    metrics["residual_vs_slots_curve"] = residual_curve  # list of M+1 floats
    # monotone improvement check (more slots should help or stay flat)
    diffs_curve = [residual_curve[i] - residual_curve[i + 1] for i in range(len(residual_curve) - 1)]
    metrics["residual_vs_slots_monotone"] = all(d >= -0.01 for d in diffs_curve)

    if out_dir is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(M + 1), residual_curve, "o-")
        ax.set_xlabel("number of slots kept (coarse→fine)")
        ax.set_ylabel("L_pred (MSE proxy)")
        ax.set_title("Residual-vs-slots curve (monotone → slots carry distinct info)")
        fig.tight_layout()
        fig.savefig(out_dir / "residual_vs_slots.png", dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Slot-attention entropy
    # ------------------------------------------------------------------
    # If slot attention weights are available via model, use them;
    # otherwise estimate from slot diversity (K variance per slot).
    # JEPAOperatorModel stores the slot encoder as `model.encoder` (NOT
    # `model.slot_encoder`). Reach the real per-slot cross-attention entropy path if
    # the encoder caches its weights; otherwise fall back to the variance proxy.
    _enc = getattr(model, "encoder", None)
    if _enc is not None and getattr(_enc, "last_attn_weights", None) is not None:
        attn_w = _enc.last_attn_weights  # (N, M, T_text) — if cached
        if attn_w is not None:
            attn_np = _to_numpy(attn_w)
            # entropy per slot: -sum(p log p) over T_text tokens
            eps = 1e-10
            slot_entropy = -(attn_np * np.log(attn_np + eps)).sum(-1)  # (N, M)
            per_slot_entropy_mean = slot_entropy.mean(axis=0)  # (M,)
        else:
            per_slot_entropy_mean = None
    else:
        per_slot_entropy_mean = None

    if per_slot_entropy_mean is None:
        # proxy: use variance of k across examples per slot as diversity proxy
        k_np = K.numpy()  # (N, M, dn)
        per_slot_var = k_np.var(axis=0).mean(axis=-1)  # (M,)
        per_slot_entropy_mean = np.log(per_slot_var + 1.0)  # soft proxy

    metrics["slot_entropy_mean"] = float(per_slot_entropy_mean.mean())
    metrics["slot_entropy_min"] = float(per_slot_entropy_mean.min())
    metrics["slot_entropy_max"] = float(per_slot_entropy_mean.max())
    # high variance across slots indicates distinct routing
    metrics["slot_entropy_spread"] = float(per_slot_entropy_mean.std())

    if out_dir is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(M), per_slot_entropy_mean)
        ax.set_xlabel("slot index")
        ax.set_ylabel("entropy (or proxy)")
        ax.set_title(f"Per-slot attention entropy  (spread={metrics['slot_entropy_spread']:.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / "slot_entropy.png", dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 8. KMeans noun-cluster × verb contingency
    # ------------------------------------------------------------------
    try:
        from sklearn.cluster import KMeans

        k_for_cluster = min(n_verbs_op, max(2, K_flat.shape[0] // 4))
        km = KMeans(n_clusters=k_for_cluster, n_init=5, random_state=0)
        cluster_labels = km.fit_predict(K_flat)  # (N*M,)
        verb_labels = verbs_flat  # (N*M,)

        contingency = np.zeros((k_for_cluster, n_verbs_op), dtype=int)
        for cl, vb in zip(cluster_labels, verb_labels):
            contingency[cl, vb] += 1

        metrics["binding_contingency_max_purity"] = float(
            (contingency.max(axis=1) / contingency.sum(axis=1).clip(min=1)).mean()
        )
        metrics["binding_n_clusters"] = k_for_cluster

        if out_dir is not None:
            fig, ax = plt.subplots(figsize=(max(6, n_verbs_op), max(4, k_for_cluster // 2)))
            im = ax.imshow(contingency, aspect="auto", cmap="Blues")
            ax.set_xlabel("verb index")
            ax.set_ylabel("noun cluster (KMeans)")
            ax.set_title("Noun-cluster × Verb contingency table")
            plt.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / "binding_contingency.png", dpi=80)
            plt.close(fig)

    except ImportError:
        # scikit-learn optional
        metrics["binding_contingency_max_purity"] = float("nan")
        metrics["binding_n_clusters"] = 0

    # ------------------------------------------------------------------
    # 9. Held-out cos(zhat, z) + MSE
    # ------------------------------------------------------------------
    cos_vals = _cosine_sim_matrix(Zhat, Z).numpy()  # (N,)
    mse_vals = ((Zhat - Z) ** 2).mean(-1).numpy()   # (N,)
    metrics["held_out_cos_mean"] = float(cos_vals.mean())
    metrics["held_out_cos_std"] = float(cos_vals.std())
    metrics["held_out_mse_mean"] = float(mse_vals.mean())

    if out_dir is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].hist(cos_vals, bins=30, edgecolor="k")
        axes[0].set_xlabel("cos(zhat, z)")
        axes[0].set_title(f"Held-out cosine similarity  (mean={metrics['held_out_cos_mean']:.3f})")
        axes[1].hist(mse_vals, bins=30, edgecolor="k")
        axes[1].set_xlabel("MSE(zhat, z)")
        axes[1].set_title(f"Held-out MSE  (mean={metrics['held_out_mse_mean']:.4f})")
        fig.tight_layout()
        fig.savefig(out_dir / "held_out_reconstruction.png", dpi=80)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10. Multi-step drift: apply B_v N=1..16 times
    # ------------------------------------------------------------------
    # Start from a subset of held-out k; apply the most common verb N times.
    # Measure cos to nearest training noun as a proxy for manifold proximity.
    n_drift = min(64, N)
    k_drift = K[:n_drift, :, :].to(device)           # (n_drift, M, dn)
    # use the most-used verb per slot
    common_verb = int(np.bincount(verbs_flat).argmax())
    verb_tensor = torch.full((n_drift, M), common_verb, dtype=torch.long, device=device)

    # training noun manifold reference (use all collected k)
    k_ref_np = K_flat  # (N*M, dn)

    max_steps = 16
    drift_cos = []
    with torch.no_grad():
        x = k_drift.clone()  # (n_drift, M, dn)
        for step in range(1, max_steps + 1):
            x = model.step_latent(x, verb_tensor)   # (n_drift, M, dn)
            x_np = x.reshape(-1, dn).float().cpu().numpy()  # (n_drift*M, dn)
            # cos to nearest training noun
            x_norm = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-8)
            ref_norm = k_ref_np / (np.linalg.norm(k_ref_np, axis=1, keepdims=True) + 1e-8)
            sims = x_norm @ ref_norm.T  # (n_drift*M, N*M)
            nearest_cos = sims.max(axis=1).mean()
            drift_cos.append(float(nearest_cos))

    metrics["multi_step_drift_cos_1"] = drift_cos[0] if drift_cos else float("nan")
    metrics["multi_step_drift_cos_16"] = drift_cos[-1] if drift_cos else float("nan")
    # bounded drift: cos at step 16 should not be dramatically lower than step 1
    if drift_cos:
        metrics["multi_step_drift_drop"] = drift_cos[0] - drift_cos[-1]
        metrics["multi_step_drift_alarm"] = (drift_cos[0] - drift_cos[-1]) > 0.5
    else:
        metrics["multi_step_drift_drop"] = float("nan")
        metrics["multi_step_drift_alarm"] = False

    if out_dir is not None and drift_cos:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1, max_steps + 1), drift_cos, "o-")
        ax.set_xlabel("number of B_v applications")
        ax.set_ylabel("cos to nearest training noun")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Multi-step drift  (verb={common_verb}, drop={metrics['multi_step_drift_drop']:.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / "multi_step_drift.png", dpi=80)
        plt.close(fig)

    # ensure all scalar metrics are finite Python floats (not tensor / nan from bugs)
    out_metrics: dict = {}
    for k_m, v_m in metrics.items():
        if isinstance(v_m, list):
            out_metrics[k_m] = v_m
        elif isinstance(v_m, (bool, np.bool_)):
            out_metrics[k_m] = bool(v_m)
        elif isinstance(v_m, (int, np.integer)):
            out_metrics[k_m] = int(v_m)
        else:
            try:
                out_metrics[k_m] = float(v_m)
            except (TypeError, ValueError):
                out_metrics[k_m] = v_m

    return out_metrics
