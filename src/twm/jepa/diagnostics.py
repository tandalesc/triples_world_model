"""JEPA v2 diagnostics suite (T-task D). Design doc §8.

`eval_diagnostics_v2(model, dataset, device, ...)` -> flat dict + JSON/PNG artifacts.

New metrics vs v1:
  - Generated text samples (THE quality artifact): greedy + temperature, logged as a
    table {text_t, gold_t+1, v_posterior, v_prior, gen_greedy, gen_temp_...} and
    saved to out_dir/samples_epoch{epoch}.json. chrF + exact-match vs gold.
  - v-ablation CE gap (headline): L_token with true posterior v vs L_token with v
    forced to a constant code. Gap > 0.1 nats => v carries causal info.
  - Latent-action usage perplexity (diagnostic only): posterior argmax v usage ppl;
    prior argmax v usage ppl; posterior↔prior agreement rate. High ppl is NOT
    sufficient (v1 had 6.2/8 with empty semantics) — interpret jointly with CE gap.
  - Emergent action semantics probe: cluster held-out pairs by argmax v, emit per-v
    example table (3-5 (text_t -> text_t+1) pairs per code) to
    out_dir/action_semantics_epoch{epoch}.json.
  - Hard-negative MRR (regression metric): port of probe1_retrieval.py logic.
    easy_minus_hard_mrr must become >= 0 (v1 was -0.041, below chance).

Retained from v1 (imported, not reimplemented):
  - Noun geometry: effective rank, per-dim variance.
  - Scale drift: log r distribution, mean ||a*||/||k||.
  - Slot-attention entropy proxy.
  - Structural sanity passthrough.

v2 model interface expected by this module (from model.py / design doc §11 Task C):
  model.forward_v2(src_ids, src_pad, tgt_ids, tgt_pad, tau=1.0, hard=True)
    -> dict with keys: k, a, v, v_logits, p_logits, zhat, z_target, logits
  model.operator            — RotationScaleOperator (for noun geometry / scale drift)
  model.decoder.generate(a_star, max_tokens, temperature) -> (B, T) ids

Tokenizer interface: tokenizer.decode(ids) -> str, tokenizer.encode(text) -> list[int].

Both interfaces are duck-typed; if a key is missing the relevant metric is skipped
gracefully so partial implementations (other concurrent task builders) can import and
call this module without a TypeError.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # headless; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Import the small numeric helpers from the demoted v1 diagnostics — no
# reimplementation. Guard the import so this module stays importable even if the
# legacy file is ever pruned (the fallback defs below are equivalent).
try:
    from .legacy.diagnostics_v1 import (
        _effective_rank,
        _cosine_sim_matrix,
        _to_numpy,
    )
except ImportError:
    def _effective_rank(C):
        tr = np.trace(C); frob2 = np.sum(C ** 2)
        return float(tr ** 2 / frob2) if frob2 > 1e-12 else 1.0

    def _cosine_sim_matrix(a, b):
        a_n = F.normalize(a.float(), dim=-1); b_n = F.normalize(b.float(), dim=-1)
        return (a_n * b_n).sum(-1)

    def _to_numpy(t):
        return t.detach().float().cpu().numpy()

from .losses import token_ce

if TYPE_CHECKING:
    pass  # model.JEPAOperatorModelV2 — avoid circular import


# ---------------------------------------------------------------------------
# v2.1 polar per-factor diagnostics (design §5.2, §8.1). NOT losses — eval only.
# ---------------------------------------------------------------------------

def _block_modulus_np(k: np.ndarray) -> np.ndarray:
    """Per-block complex modulus |z_b| for k (..., dn) -> (..., nb). numpy mirror of
    conditioning.block_modulus (design §5.2)."""
    dn = k.shape[-1]
    pair = k.reshape(*k.shape[:-1], dn // 2, 2)
    return np.sqrt((pair ** 2).sum(axis=-1))


def _block_phase_np(k: np.ndarray) -> np.ndarray:
    """Per-block phase arg(z_b) = atan2(y_b, x_b) for k (..., dn) -> (..., nb)."""
    dn = k.shape[-1]
    pair = k.reshape(*k.shape[:-1], dn // 2, 2)
    return np.arctan2(pair[..., 1], pair[..., 0])


def _phase_uniformity(K_flat: np.ndarray, n: int) -> dict:
    """Phase-uniformity via mean resultant length (design §5.2.1).

    For each block b collect phases φ_b over the eval set and compute the mean
    resultant length R_b = |(1/N) Σ e^{iφ}| ∈ [0,1]. R_b → 0 for uniform (ideal),
    R_b → 1 for collapsed. Report phase_uniformity = 1 − mean_b R_b. Also run the
    Rayleigh test z_b = N·R_b² per block and count blocks above the 99% χ²₂ critical
    value (≈ 9.21) as phase-collapsed.

    K_flat: (N, dn) standardized nouns; n: N (number of samples, for the z-stat).
    """
    phases = _block_phase_np(K_flat)            # (N, nb)
    C = np.cos(phases).mean(axis=0)             # (nb,)
    S = np.sin(phases).mean(axis=0)             # (nb,)
    R = np.sqrt(C ** 2 + S ** 2)                # (nb,) mean resultant length per block
    z = n * R ** 2                              # Rayleigh test statistic per block
    chi2_2_99 = 9.21034                         # 99% critical value of χ²₂
    n_collapsed = int((z > chi2_2_99).sum())
    return {
        "phase_uniformity": float(1.0 - R.mean()),
        "phase_collapsed_blocks": n_collapsed,
        "phase_R_mean": float(R.mean()),
        "phase_R_max": float(R.max()),
    }


@torch.no_grad()
def _identity_persistence(model, k_probe: torch.Tensor) -> dict:
    """Identity-persistence check (design §8.1): rotation must preserve modulus.

    For each verb v, take a probe batch, compute the conditioned offset θ_off = H(|k|)
    (if polar conditioning is on; else None), apply the operator, and measure relative
    modulus drift `|| |a| − |k| || / (|| |k| || + eps)`.

    For a PURE-ROTATION verb (log_r row exactly 0) the drift must be < 1e-5 — a
    structural identity (rotation preserves complex modulus; the phase offset cannot
    perturb modulus). This is the load-bearing test of the polar claim. The diagnostic
    REPORTS per-verb drift and ASSERTS the pure-rotation case.

    Note: the diagnostic detaches the modulus used for the offset (design §3.1: stop_grad
    only inside the diagnostic, never in the forward). We are in no_grad anyway.
    """
    op = getattr(model, "operator", None)
    if op is None or not hasattr(op, "theta") or not hasattr(op, "log_r"):
        return {}

    device = op.theta.device
    k_probe = k_probe.to(device)
    n_verbs = op.theta.shape[0]
    log_r = op.log_r.detach()
    eps = 1e-8

    conditioner = getattr(model, "conditioner", None)

    drifts = []
    max_rot_drift = 0.0
    rot_verbs = []
    for v in range(n_verbs):
        v_slots = torch.full(k_probe.shape[:2], v, dtype=torch.long, device=device)
        theta_offset = conditioner(k_probe) if conditioner is not None else None
        a = op.apply(k_probe, v_slots, theta_offset=theta_offset)
        m_k = _block_modulus_torch(k_probe)
        m_a = _block_modulus_torch(a)
        drift = ((m_a - m_k).norm() / (m_k.norm() + eps)).item()
        drifts.append(drift)
        # A verb is "pure rotation" iff its log_r row is exactly 0 (RotationOperator, or
        # a verb whose scale was never moved). Those must preserve modulus to 1e-5.
        if torch.all(log_r[v].abs() < 1e-12):
            rot_verbs.append(v)
            max_rot_drift = max(max_rot_drift, drift)

    return {
        "identity_persistence_err": max_rot_drift,  # max drift over pure-rotation verbs
        "modulus_drift_per_verb": [float(d) for d in drifts],
        "n_pure_rotation_verbs": len(rot_verbs),
        # assertion surface: pure-rotation verbs must preserve modulus to machine eps.
        "identity_persistence_pass": bool(max_rot_drift < 1e-5) if rot_verbs else True,
    }


def _block_modulus_torch(k: torch.Tensor) -> torch.Tensor:
    """torch per-block modulus |z_b| for k (..., dn) -> (..., nb)."""
    dn = k.shape[-1]
    pair = k.reshape(*k.shape[:-1], dn // 2, 2)
    return pair.pow(2).sum(dim=-1).sqrt()


# ---------------------------------------------------------------------------
# chrF helper (reference-free at word level; no sacrebleu dep)
# ---------------------------------------------------------------------------

def _chrf(hyp: str, ref: str, n: int = 6) -> float:
    """Character n-gram F-score (ChrF) for a single hypothesis/reference pair.

    Precision = matched_ngrams / hyp_ngrams,
    Recall    = matched_ngrams / ref_ngrams,
    F1        = harmonic mean.

    No normalization beyond lowercasing — the BPE tokenizer already handles OOV.
    """
    def ngrams(s, n):
        return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]

    hyp = hyp.lower().strip()
    ref = ref.lower().strip()

    total_p = total_r = total_f = 0.0
    for ng in range(1, n + 1):
        h_ng = ngrams(hyp, ng)
        r_ng = ngrams(ref, ng)
        if not h_ng and not r_ng:
            total_f += 1.0
            continue
        h_count: dict[str, int] = {}
        for g in h_ng:
            h_count[g] = h_count.get(g, 0) + 1
        r_count: dict[str, int] = {}
        for g in r_ng:
            r_count[g] = r_count.get(g, 0) + 1
        match = sum(min(h_count.get(g, 0), r_count[g]) for g in r_count)
        prec = match / len(h_ng) if h_ng else 0.0
        rec = match / len(r_ng) if r_ng else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        total_p += prec; total_r += rec; total_f += f
    return total_f / n


# ---------------------------------------------------------------------------
# Decode token IDs to text via the dataset's tokenizer (duck-typed)
# ---------------------------------------------------------------------------

def _decode_ids(tokenizer, ids: list[int]) -> str:
    """Decode a list of token ids to a string, stripping special tokens."""
    PAD, MASK, UNK, BOS, EOS = 0, 1, 2, 3, 4
    clean = [i for i in ids if i not in (PAD, MASK, UNK, BOS, EOS)]
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(clean)
        except Exception:
            pass
    if hasattr(tokenizer, "id_to_token"):
        tokens = [tokenizer.id_to_token(i) for i in clean]
        return " ".join(t for t in tokens if t)
    return " ".join(str(i) for i in clean)


def _get_text(item: dict, key: str, tokenizer) -> str:
    """Recover the raw text for src or tgt from a dataset item."""
    ids_key = f"{key}_ids"
    pad_key = f"{key}_pad"
    if ids_key not in item:
        return ""
    ids = item[ids_key].tolist()
    if pad_key in item:
        mask = item[pad_key].tolist()
        ids = [i for i, m in zip(ids, mask) if not m]
    return _decode_ids(tokenizer, ids)


# ---------------------------------------------------------------------------
# Retrieval hard-negative MRR (ported from probe1_retrieval.py)
# ---------------------------------------------------------------------------

def _ranks_for_pools(
    zhat: torch.Tensor,              # (N, d) predicted latents
    zpool: torch.Tensor,             # (N, d) target latents (gold at index i for query i)
    pool_index_lists: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """For each query i, rank its pool by cosine(zhat_i, z_cand). Return 1-based ranks."""
    zhat_n = F.normalize(zhat.float(), dim=-1)
    zpool_n = F.normalize(zpool.float(), dim=-1)
    ranks = []
    pool_sizes = []
    for i, pool in enumerate(pool_index_lists):
        cand = zpool_n[pool]           # (P, d)
        sims = cand @ zhat_n[i]        # (P,)
        gold_sim = sims[0]
        rank = 1 + int((sims > gold_sim).sum().item())
        ranks.append(rank)
        pool_sizes.append(len(pool))
    return np.array(ranks, dtype=float), np.array(pool_sizes, dtype=float)


def _mrr(ranks: np.ndarray) -> float:
    return float((1.0 / ranks).mean())


def _compute_retrieval_mrr(
    zhat: torch.Tensor,
    z_target: torch.Tensor,
    chain_ids: list[int],
    hard_nn_per_query: int = 40,
) -> dict:
    """Compute easy-pool and hard-pool MRR; return {easy_mrr, hard_mrr, easy_minus_hard_mrr}.

    The easy pool is the full global pool (can be beaten by topic/vocabulary similarity).
    The hard pool adds same-chain negatives + cosine nearest-neighbors of the target
    encoding — only genuine transition dynamics can rank correctly in the hard pool.
    This is the v1 regression check: easy_minus_hard_mrr must become >= 0 (v1 = -0.041).
    """
    N = zhat.shape[0]
    # Easy pools: gold at position 0, all others as distractors.
    easy_pools = [[i] + [j for j in range(N) if j != i] for i in range(N)]

    # Hard pools: same-chain distractors + NN distractors.
    same_chain: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(chain_ids):
        for j, cid2 in enumerate(chain_ids):
            if j != i and cid2 == cid:
                same_chain[i].append(j)

    zt_n = F.normalize(z_target.float(), dim=-1)
    sims_mat = (zt_n @ zt_n.T).cpu().numpy()    # (N, N) target-target cosine

    hard_pools = []
    for i in range(N):
        base = [i] + same_chain[i]
        baseset = set(base)
        order = np.argsort(-sims_mat[i])
        nn = [int(j) for j in order if j not in baseset][:hard_nn_per_query]
        hard_pools.append(base + nn)

    r_easy, ps_easy = _ranks_for_pools(zhat, z_target, easy_pools)
    r_hard, ps_hard = _ranks_for_pools(zhat, z_target, hard_pools)

    easy_mrr = _mrr(r_easy)
    hard_mrr = _mrr(r_hard)
    chance_easy = float((1.0 / ps_easy).mean())
    chance_hard = float((1.0 / ps_hard).mean())
    return {
        "easy_mrr": easy_mrr,
        "hard_mrr": hard_mrr,
        "easy_minus_hard_mrr": easy_mrr - hard_mrr,
        "chance_easy_mrr": chance_easy,
        "chance_hard_mrr": chance_hard,
        "n": N,
        # regression guard: v1 was -0.041 (below chance on hard pool)
        "regression_pass": bool(easy_mrr - hard_mrr >= 0.0),
    }


# ---------------------------------------------------------------------------
# v-ablation CE gap (headline success metric)
# ---------------------------------------------------------------------------

def _v_ablation_ce_gap(
    model,
    src_ids: torch.Tensor,
    src_pad: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_pad: torch.Tensor,
    device: torch.device,
    tau: float = 1.0,
    pad_id: int = 0,
    ablation_verb_idx: int = 0,
) -> dict:
    """Compute L_token with true posterior v vs v forced to a constant code.

    Gap > 0.1 nats (sustained) => v carries causal information.
    Gap ≈ 0 => regression to v1 (verb decorative / decoder ignores it).

    The ablation replaces v with a one-hot on code `ablation_verb_idx` (constant
    for all examples); the decoder still sees a* = B_{v_ablated}(k), so the operator
    is active but the discrete information content is zero bits.
    """
    with torch.no_grad():
        out = model.forward_v2(src_ids, src_pad, tgt_ids, tgt_pad, tau=tau, hard=True)

    logits_true = out.get("logits")
    a_true = out.get("a")
    k = out.get("k")

    if logits_true is None or a_true is None or k is None:
        return {"ce_true": float("nan"), "ce_ablated": float("nan"),
                "ce_gap_nats": float("nan"), "gap_passes_threshold": False}

    with torch.no_grad():
        ce_true = token_ce(logits_true, tgt_ids, pad_id=pad_id).item()

    # Build ablated a*: apply constant verb to every slot.
    try:
        B, M, dn = k.shape
        ablated_v = torch.full((B, M), ablation_verb_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            a_ablated = model.operator.apply(k, ablated_v)
            # Re-run the decoder with the ablated a*.
            if hasattr(model, "decoder"):
                logits_ablated = model.decoder.forward(a_ablated, tgt_ids, tgt_pad)
                ce_ablated = token_ce(logits_ablated, tgt_ids, pad_id=pad_id).item()
            else:
                ce_ablated = float("nan")
    except Exception:
        ce_ablated = float("nan")

    gap = ce_true - ce_ablated  # negative = true is BETTER (lower CE), as expected
    # A well-functioning v2 has ce_true < ce_ablated (true v helps); gap is negative.
    # Report ce_gap = ce_ablated - ce_true (positive = helpful).
    ce_gap = ce_ablated - ce_true
    return {
        "ce_true_nats": ce_true,
        "ce_ablated_nats": ce_ablated,
        "ce_gap_nats": ce_gap,                       # > 0 means v helps
        "gap_passes_threshold": bool(ce_gap > 0.1),  # headline pass criterion
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def eval_diagnostics_v2(
    model,
    dataset,
    device,
    n_examples: int = 512,
    out_dir=None,
    n_text_samples: int = 16,
    temperatures: list[float] | None = None,
    epoch: int | None = None,
    tokenizer=None,
    pad_id: int = 0,
    tau: float = 1.0,
    hard_nn_per_query: int = 40,
) -> dict:
    """Run all v2 diagnostics. Returns a flat dict of scalar metrics + saves artifacts.

    Args:
        model:          JEPAOperatorModelV2 (or duck-typed compatible).
        dataset:        JEPAChainDataset-compatible; __getitem__ returns
                        {"src_ids","src_pad","tgt_ids","tgt_pad"}.
        device:         "cpu" / "cuda" / "mps" / torch.device.
        n_examples:     number of examples to use (capped at len(dataset)).
        out_dir:        if not None, write artifact JSON/PNG files here.
        n_text_samples: number of examples to include in the text-samples table.
        temperatures:   list of sampling temperatures for text generation [0.7, 1.0].
        epoch:          current training epoch (used in artifact filenames).
        tokenizer:      BPE tokenizer for text decode/encode. If None, uses
                        dataset.tokenizer if available.
        pad_id:         padding token id (0 for jepa_bpe_512.json).
        tau:            Gumbel temperature to use for diagnostic forward passes.
        hard_nn_per_query: NN distractors per query for the hard MRR pool.

    Returns:
        flat dict with all scalar metrics; non-scalar artifacts are written to out_dir.
    """
    if temperatures is None:
        temperatures = [0.7, 1.0]

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve tokenizer: from arg, then dataset.tokenizer, else None.
    if tokenizer is None:
        tokenizer = getattr(dataset, "tokenizer", None)

    n = min(n_examples, len(dataset))
    indices = list(range(n))

    epoch_str = f"epoch{epoch}" if epoch is not None else "latest"

    # ------------------------------------------------------------------
    # 1. Collect forward-pass tensors
    # ------------------------------------------------------------------
    all_k: list[torch.Tensor] = []
    all_a_star: list[torch.Tensor] = []
    all_v_post: list[int] = []         # argmax posterior v (B=1 per item)
    all_v_prior: list[int] = []        # argmax prior v
    all_v_logits: list[torch.Tensor] = []   # (V,) posterior logits
    all_p_logits: list[torch.Tensor] = []   # (V,) prior logits
    all_zhat: list[torch.Tensor] = []
    all_z: list[torch.Tensor] = []
    all_chain_ids: list[int] = []
    all_src_texts: list[str] = []
    all_tgt_texts: list[str] = []
    all_tgt_ids: list[torch.Tensor] = []
    all_tgt_pad: list[torch.Tensor] = []

    _has_chain_id = hasattr(dataset, "chain_ids") or hasattr(dataset, "pairs")

    with torch.no_grad():
        for idx in indices:
            item = dataset[idx]
            src_ids = item["src_ids"].unsqueeze(0).to(device)
            src_pad = item["src_pad"].unsqueeze(0).to(device)
            tgt_ids_ = item["tgt_ids"].unsqueeze(0).to(device)
            tgt_pad_ = item["tgt_pad"].unsqueeze(0).to(device)

            # Try v2 forward; fall back to v1-style forward with best-effort keys.
            if hasattr(model, "forward_v2"):
                out = model.forward_v2(src_ids, src_pad, tgt_ids_, tgt_pad_,
                                       tau=tau, hard=True)
            elif hasattr(model, "forward"):
                # v1-compatible fallback — may lack v2 keys; collect what's available
                out = model(src_ids, src_pad)
            else:
                continue

            k = out.get("k")
            a = out.get("a")
            zhat = out.get("zhat")
            z_tgt = out.get("z_target")
            v_logits = out.get("v_logits")
            p_logits_b = out.get("p_logits")

            if k is None or a is None:
                continue

            all_k.append(k.squeeze(0).cpu())
            all_a_star.append(a.squeeze(0).cpu())

            if v_logits is not None:
                all_v_logits.append(v_logits.squeeze(0).cpu())
                all_v_post.append(int(v_logits.squeeze(0).argmax(-1).item()))
            if p_logits_b is not None:
                all_p_logits.append(p_logits_b.squeeze(0).cpu())
                all_v_prior.append(int(p_logits_b.squeeze(0).argmax(-1).item()))

            if zhat is not None:
                all_zhat.append(zhat.squeeze(0).cpu())
            if z_tgt is not None:
                all_z.append(z_tgt.squeeze(0).cpu())

            # Chain id (for hard-MRR pool construction)
            cid = idx  # fallback: each example is its own chain
            if hasattr(dataset, "chain_ids") and idx < len(dataset.chain_ids):
                cid = dataset.chain_ids[idx]
            elif hasattr(dataset, "pairs") and idx < len(dataset.pairs):
                cid = getattr(dataset.pairs[idx], "chain_id", idx)
            all_chain_ids.append(cid)

            # Texts (for samples table; best-effort)
            if tokenizer is not None:
                all_src_texts.append(_get_text(item, "src", tokenizer))
                all_tgt_texts.append(_get_text(item, "tgt", tokenizer))
            all_tgt_ids.append(tgt_ids_.squeeze(0).cpu())
            all_tgt_pad.append(tgt_pad_.squeeze(0).cpu())

    metrics: dict = {}

    # ------------------------------------------------------------------
    # 2. Noun geometry (reuses v1 _effective_rank helper)
    # ------------------------------------------------------------------
    if all_k:
        K = torch.stack(all_k)        # (N, M, dn)
        N, M, dn = K.shape
        K_flat = K.reshape(-1, dn).numpy()
        K_c = K_flat - K_flat.mean(0, keepdims=True)
        C = K_c.T @ K_c / max(K_c.shape[0] - 1, 1)

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
            ax.set_xlabel("noun dim"); ax.set_ylabel("variance")
            ax.set_title(f"Noun per-dim variance (eff_rank={eff_rank:.2f})")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / f"noun_var_hist_{epoch_str}.png", dpi=80)
            plt.close(fig)

    # ------------------------------------------------------------------
    # 2b. v2.1 polar per-factor diagnostics (design §5.2, §8.1)
    # ------------------------------------------------------------------
    if all_k:
        K = torch.stack(all_k)              # (N, M, dn)
        N, M, dn = K.shape
        K_flat = K.reshape(-1, dn).numpy()  # (N·M, dn)
        nb = dn // 2

        # §5.2.1 phase-uniformity (mean resultant length per block).
        metrics.update(_phase_uniformity(K_flat, n=K_flat.shape[0]))

        # §5.2.2 modulus-profile effective rank (identity-space dimensionality).
        M_prof = _block_modulus_np(K_flat)                  # (N·M, nb)
        M_c = M_prof - M_prof.mean(0, keepdims=True)
        Cm = M_c.T @ M_c / max(M_c.shape[0] - 1, 1)         # (nb, nb)
        modulus_eff_rank = _effective_rank(Cm)
        metrics["modulus_eff_rank"] = float(modulus_eff_rank)
        metrics["modulus_eff_rank_threshold"] = nb / 4
        # warn flags (report-only; mirror the existing noun_eff_rank threshold spirit).
        metrics["modulus_eff_rank_warn"] = bool(modulus_eff_rank < nb / 4)
        metrics["phase_uniformity_warn"] = bool(metrics.get("phase_uniformity", 1.0) < 0.8)

        # §8.1 identity-persistence assertion (rotation preserves modulus to 1e-5).
        try:
            id_probe = K[: min(64, N)].to(device)
            metrics.update(_identity_persistence(model, id_probe))
        except Exception as exc:
            metrics["_identity_persistence_error"] = str(exc)

    # ------------------------------------------------------------------
    # 2c. v2.1 optional kind-cluster table (design §7; artifact only)
    # ------------------------------------------------------------------
    kind_head = getattr(model, "kind_head", None)
    if kind_head is not None and all_k and all_src_texts:
        try:
            with torch.no_grad():
                K_dev = torch.stack(all_k).to(device)       # (N, M, dn)
                kind_ids = kind_head.assign(K_dev).cpu().numpy()  # (N, M)
            # Slot-level kind -> example src texts (use the dominant slot kind per item).
            from collections import defaultdict as _dd
            kind_examples: dict[str, list[str]] = _dd(list)
            for i in range(min(len(all_src_texts), kind_ids.shape[0])):
                kid = int(np.bincount(kind_ids[i]).argmax())  # dominant slot kind
                if len(kind_examples[str(kid)]) < 5:
                    kind_examples[str(kid)].append(all_src_texts[i])
            metrics["n_kind_clusters_used"] = len(kind_examples)
            if out_dir is not None:
                with open(out_dir / f"kind_clusters_{epoch_str}.json", "w") as f:
                    json.dump(dict(kind_examples), f, indent=2)
        except Exception as exc:
            metrics["_kind_cluster_error"] = str(exc)

    # ------------------------------------------------------------------
    # 3. Scale drift
    # ------------------------------------------------------------------
    if all_k and all_a_star:
        A_star = torch.stack(all_a_star)
        op = getattr(model, "operator", None)
        if op is not None and hasattr(op, "theta") and hasattr(op, "log_r"):
            theta_np = _to_numpy(op.theta)
            log_r_np = _to_numpy(op.log_r)
            mean_log_r = float(log_r_np.mean())
            metrics["scale_drift_mean_log_r"] = mean_log_r
            metrics["scale_drift_alarm"] = bool(mean_log_r < -1.0)
        k_norms = K.norm(dim=-1).clamp(min=1e-8)
        a_norms = A_star.norm(dim=-1)
        metrics["scale_drift_mean_a_over_k"] = float((a_norms / k_norms).mean().item())

    # ------------------------------------------------------------------
    # 4. Latent-action usage perplexity (diagnostic only; not a loss)
    # ------------------------------------------------------------------
    if all_v_post:
        n_verbs = len(all_v_logits[0]) if all_v_logits else 8
        post_counts = np.bincount(all_v_post, minlength=n_verbs).astype(float)
        post_probs = post_counts / post_counts.sum()
        post_safe = np.where(post_probs > 0, post_probs, 1e-12)
        post_ppl = float(np.exp(-(post_probs * np.log(post_safe)).sum()))
        metrics["v_usage_ppl_posterior"] = post_ppl
        metrics["v_usage_counts_posterior"] = post_counts.astype(int).tolist()

        if all_v_prior:
            prior_counts = np.bincount(all_v_prior, minlength=n_verbs).astype(float)
            prior_probs = prior_counts / prior_counts.sum()
            prior_safe = np.where(prior_probs > 0, prior_probs, 1e-12)
            prior_ppl = float(np.exp(-(prior_probs * np.log(prior_safe)).sum()))
            metrics["v_usage_ppl_prior"] = prior_ppl
            # Posterior↔prior agreement rate
            agree = sum(int(vp == vq) for vp, vq in zip(all_v_post, all_v_prior))
            metrics["v_post_prior_agreement"] = float(agree) / max(len(all_v_post), 1)

        # WARN: high ppl is NOT sufficient — v1 had 6.2/8 with semantically empty codes.
        # Interpret jointly with the CE gap.
        metrics["v_usage_ppl_healthy_threshold"] = n_verbs / 2
        if out_dir is not None:
            fig, ax = plt.subplots(figsize=(6, 3))
            bar_w = 0.35; x = np.arange(n_verbs)
            ax.bar(x - bar_w / 2, post_counts / post_counts.sum(),
                   bar_w, label=f"posterior (ppl={post_ppl:.1f})")
            if all_v_prior:
                ax.bar(x + bar_w / 2, prior_probs,
                       bar_w, label=f"prior (ppl={prior_ppl:.1f})")
            ax.set_xlabel("action code v"); ax.set_ylabel("usage fraction")
            ax.set_title("Latent-action usage distribution (diagnostic only)")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / f"action_usage_{epoch_str}.png", dpi=80)
            plt.close(fig)

    # ------------------------------------------------------------------
    # 5. v-ablation CE gap (headline metric)
    # ------------------------------------------------------------------
    # Run on the first min(64, n) examples (batched) for speed.
    n_abl = min(64, n)
    try:
        abl_src_ids = torch.stack([
            dataset[i]["src_ids"] for i in range(n_abl)
        ]).to(device)
        abl_src_pad = torch.stack([
            dataset[i]["src_pad"] for i in range(n_abl)
        ]).to(device)
        abl_tgt_ids = torch.stack([
            dataset[i]["tgt_ids"] for i in range(n_abl)
        ]).to(device)
        abl_tgt_pad = torch.stack([
            dataset[i]["tgt_pad"] for i in range(n_abl)
        ]).to(device)

        abl_metrics = _v_ablation_ce_gap(
            model, abl_src_ids, abl_src_pad, abl_tgt_ids, abl_tgt_pad,
            device=device, tau=tau, pad_id=pad_id,
        )
        metrics.update(abl_metrics)
    except Exception as exc:
        metrics["ce_gap_nats"] = float("nan")
        metrics["gap_passes_threshold"] = False
        metrics["_ce_gap_error"] = str(exc)

    # ------------------------------------------------------------------
    # 6. Generated text samples (THE quality artifact)
    # ------------------------------------------------------------------
    samples_table = []
    chrf_greedy_list: list[float] = []
    exact_greedy_list: list[float] = []

    if tokenizer is not None and hasattr(model, "decoder") and all_a_star:
        A_star_dev = torch.stack(all_a_star[:n_text_samples]).to(device)
        with torch.no_grad():
            # Greedy generation
            try:
                gen_greedy_ids = model.decoder.generate(
                    A_star_dev, max_tokens=64, temperature=0.0
                )  # (n_text_samples, T)
            except Exception:
                gen_greedy_ids = None

            # Temperature sampling
            gen_temp_ids: dict[float, torch.Tensor | None] = {}
            for temp in temperatures:
                try:
                    gen_temp_ids[temp] = model.decoder.generate(
                        A_star_dev, max_tokens=64, temperature=temp
                    )
                except Exception:
                    gen_temp_ids[temp] = None

        for i in range(min(n_text_samples, len(all_src_texts))):
            src_text = all_src_texts[i] if i < len(all_src_texts) else ""
            tgt_text = all_tgt_texts[i] if i < len(all_tgt_texts) else ""

            entry: dict = {
                "text_t": src_text,
                "gold_t1": tgt_text,
                "v_posterior": all_v_post[i] if i < len(all_v_post) else None,
                "v_prior": all_v_prior[i] if i < len(all_v_prior) else None,
            }

            if gen_greedy_ids is not None and i < gen_greedy_ids.shape[0]:
                gen_text = _decode_ids(tokenizer, gen_greedy_ids[i].tolist())
                entry["gen_greedy"] = gen_text
                chrf_greedy_list.append(_chrf(gen_text, tgt_text))
                exact_greedy_list.append(float(gen_text.strip() == tgt_text.strip()))

            for temp in temperatures:
                tid = gen_temp_ids.get(temp)
                if tid is not None and i < tid.shape[0]:
                    entry[f"gen_temp_{temp}"] = _decode_ids(tokenizer, tid[i].tolist())

            samples_table.append(entry)

        if out_dir is not None and samples_table:
            with open(out_dir / f"samples_{epoch_str}.json", "w") as f:
                json.dump(samples_table, f, indent=2)

    metrics["gen_chrf_greedy"] = float(np.mean(chrf_greedy_list)) if chrf_greedy_list else float("nan")
    metrics["gen_exact_greedy"] = float(np.mean(exact_greedy_list)) if exact_greedy_list else float("nan")
    metrics["n_text_samples"] = len(samples_table)

    # ------------------------------------------------------------------
    # 7. Emergent action semantics probe
    # ------------------------------------------------------------------
    if all_v_post and all_src_texts and all_tgt_texts:
        n_verbs_sem = len(all_v_logits[0]) if all_v_logits else 8
        v_to_pairs: dict[int, list[dict]] = defaultdict(list)
        for i, (src, tgt, v) in enumerate(zip(all_src_texts, all_tgt_texts, all_v_post)):
            v_to_pairs[v].append({"text_t": src, "text_t1": tgt, "pair_idx": i})

        action_semantics: dict[str, list[dict]] = {}
        for v_code in range(n_verbs_sem):
            pairs = v_to_pairs[v_code]
            # Sample up to 5 examples per code.
            action_semantics[str(v_code)] = pairs[:5]

        metrics["n_action_codes_used"] = int(len([v for v, ps in v_to_pairs.items() if ps]))

        if out_dir is not None:
            with open(out_dir / f"action_semantics_{epoch_str}.json", "w") as f:
                json.dump(action_semantics, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Hard-negative retrieval MRR (regression guard)
    # ------------------------------------------------------------------
    if all_zhat and all_z and len(all_zhat) == len(all_z):
        Zhat = torch.stack(all_zhat)
        Z = torch.stack(all_z)
        mrr_results = _compute_retrieval_mrr(
            Zhat, Z, all_chain_ids, hard_nn_per_query=hard_nn_per_query
        )
        metrics.update(mrr_results)

        # Held-out cos/MSE (v1 style)
        cos_vals = _cosine_sim_matrix(Zhat, Z).numpy()
        mse_vals = ((Zhat - Z) ** 2).mean(-1).numpy()
        metrics["held_out_cos_mean"] = float(cos_vals.mean())
        metrics["held_out_mse_mean"] = float(mse_vals.mean())

    # ------------------------------------------------------------------
    # 9. Slot-attention entropy proxy (v1 pass-through, simplified)
    # ------------------------------------------------------------------
    if all_k:
        K = torch.stack(all_k)
        k_np = K.numpy()
        per_slot_var = k_np.var(axis=0).mean(axis=-1)  # (M,)
        per_slot_entropy_mean = np.log(per_slot_var + 1.0)
        metrics["slot_entropy_mean"] = float(per_slot_entropy_mean.mean())
        metrics["slot_entropy_spread"] = float(per_slot_entropy_mean.std())

    # ------------------------------------------------------------------
    # 10. Normalise all scalars to Python primitives
    # ------------------------------------------------------------------
    out_metrics: dict = {}
    for km, vm in metrics.items():
        if isinstance(vm, list):
            out_metrics[km] = vm
        elif isinstance(vm, (bool, np.bool_)):
            out_metrics[km] = bool(vm)
        elif isinstance(vm, (int, np.integer)):
            out_metrics[km] = int(vm)
        else:
            try:
                out_metrics[km] = float(vm)
            except (TypeError, ValueError):
                out_metrics[km] = vm

    return out_metrics
