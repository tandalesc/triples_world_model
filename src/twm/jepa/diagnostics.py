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

# ===========================================================================
# Entity-world diagnostics (campaign §3): action-recovery NMI, OOD ladder,
# rollout fidelity. Config-gated under eval.entity_world (Task B).
# ===========================================================================

# Special token ids (mirrors _decode_ids; data.py / domain_bpe convention).
_SPECIAL_IDS = (0, 1, 2, 3, 4)  # PAD, MASK, UNK, BOS, EOS
_PAD_ID = 0
_EOS_ID = 4


def _encode_state(tokenizer, text: str, max_text_tokens: int, append_eos: bool = True):
    """Tokenize one state -> (ids (T,) long, pad (T,) bool), matching JEPAChainDataset._encode.

    Replicates data.py's 4-line BPE + <eos>-at-first-pad + pad-mask logic so the labeled
    loader produces tensors bitwise-compatible with the training dataset (campaign §3.1)."""
    T = max_text_tokens
    pad_id = getattr(tokenizer, "pad_token_id", _PAD_ID)
    ids = tokenizer.encode(text, max_length=T)
    if append_eos:
        ids = list(ids)
        if pad_id in ids:
            ids[ids.index(pad_id)] = _EOS_ID
        else:
            ids[-1] = _EOS_ID
    ids_t = torch.tensor(ids, dtype=torch.long)
    return ids_t, (ids_t == pad_id)


def _load_labeled_split(
    labeled_dir, split: str, tokenizer, max_text_tokens: int, append_eos: bool = True,
    max_chains: int | None = None,
) -> list[dict]:
    """Load `{split}_labeled.jsonl` -> per-chain dicts (campaign §3.1).

    Each record `{"chain":[...], "actions":["<verb>@<idx>",...], "types":[...]}` becomes
    `{"chain": [str], "actions": [str], "types": [str], "ids": [(T,) long], "pad": [(T,) bool]}`
    where ids/pad are the tokenized states (one per chain state) using the SAME encode
    logic the dataset uses. Used by §3a (NMI) and §3c (rollout fidelity)."""
    path = Path(labeled_dir) / f"{split}_labeled.jsonl"
    chains: list[dict] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            chain = rec["chain"]
            ids_list, pad_list = [], []
            for st in chain:
                ids, pad = _encode_state(tokenizer, st, max_text_tokens, append_eos)
                ids_list.append(ids)
                pad_list.append(pad)
            chains.append({
                "chain": chain,
                "actions": rec.get("actions", []),
                "types": rec.get("types", []),
                "ids": ids_list,
                "pad": pad_list,
            })
            if max_chains is not None and len(chains) >= max_chains:
                break
    return chains


def _load_manifest(labeled_dir) -> dict:
    """Load manifest.json once (schema/profiles) — best-effort, {} if absent."""
    path = Path(labeled_dir) / "manifest.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _nmi(labels_a, labels_b) -> float:
    """Normalized mutual info between two label sequences.

    Uses sklearn if available; else a self-contained NMI from joint counts
    (campaign §3a: implement the fallback so the GPU env without sklearn still reports it).
    NMI = I(A;B) / sqrt(H(A)·H(B)), 0 when either has a single cluster (NMI undefined → 0)."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) == 0:
        return 0.0
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(a, b))
    except Exception:
        pass
    # Fallback: factorize labels to dense ints, build the joint histogram.
    ua = {v: i for i, v in enumerate(sorted(set(a.tolist())))}
    ub = {v: i for i, v in enumerate(sorted(set(b.tolist())))}
    na, nb = len(ua), len(ub)
    if na <= 1 or nb <= 1:
        return 0.0
    n = len(a)
    joint = np.zeros((na, nb), dtype=float)
    for x, y in zip(a.tolist(), b.tolist()):
        joint[ua[x], ub[y]] += 1.0
    joint /= n
    pa = joint.sum(axis=1)  # (na,)
    pb = joint.sum(axis=0)  # (nb,)
    # Mutual information.
    mi = 0.0
    for i in range(na):
        for j in range(nb):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (pa[i] * pb[j]))
    # Entropies; normalize by the arithmetic mean of H(A), H(B) to match sklearn's
    # default `average_method="arithmetic"`.
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    denom = 0.5 * (ha + hb)
    return float(mi / denom) if denom > 1e-12 else 0.0


@torch.no_grad()
def _infer_posterior_action(model, src_ids, src_pad, tgt_ids, tgt_pad, device) -> int:
    """Hard argmax latent action v_hat from the pair posterior q(v|s_t,s_{t+1}) (campaign §3a).

    Returns the argmax verb index (the same `model.transition` call the forward uses with
    hard=True; we read v_logits argmax for a deterministic cluster id)."""
    v_onehot, v_logits, _ = model.transition(
        src_ids.to(device), src_pad.to(device),
        tgt_ids.to(device), tgt_pad.to(device), tau=1.0, hard=True,
    )
    return int(v_logits.squeeze(0).argmax(-1).item())


def _op_apply(model, k, v_onehot):
    """Apply the operator+conditioning for ONE action, robust to the norm-budget contract.

    Reuses model._apply_action (Task A may make it return (a, scale_delta) when the norm
    budget is on; v3 returns a bare tensor). Returns just the transformed nouns `a` — the
    entity eval suite does not consume the scale delta (that is the retraction probe, §4)."""
    out = model._apply_action(k, v_onehot)
    if isinstance(out, tuple):
        return out[0]
    return out


def _strip_decode(tokenizer, ids: list[int]) -> str:
    """Decode ids to text with the v2 special-token strip (shared with _decode_ids)."""
    return _decode_ids(tokenizer, ids)


@torch.no_grad()
def _action_recovery_nmi(model, chains, device, out_dir, epoch_str) -> dict:
    """(§3a) Action-recovery NMI vs oracle labels + shuffle baseline.

    For each adjacent (s_t, s_{t+1}) pair in every chain, get the hard argmax posterior
    latent action v_hat, and align it against the oracle label `actions[i]`:
      - verb_only: strip @<idx> (which action?)
      - verb_entity: full "<verb>@<idx>" (which action AND which entity moved?)
    Report NMI(v_hat; verb_only), NMI(v_hat; verb_entity), and a shuffle baseline."""
    v_hats: list[int] = []
    verb_only: list[str] = []
    verb_entity: list[str] = []

    for ch in chains:
        ids, pad, actions = ch["ids"], ch["pad"], ch["actions"]
        n_pairs = min(len(ids) - 1, len(actions))
        for i in range(n_pairs):
            v_hat = _infer_posterior_action(
                model,
                ids[i].unsqueeze(0), pad[i].unsqueeze(0),
                ids[i + 1].unsqueeze(0), pad[i + 1].unsqueeze(0),
                device,
            )
            v_hats.append(v_hat)
            lab = actions[i]
            verb_entity.append(lab)
            verb_only.append(lab.split("@", 1)[0])

    metrics: dict = {}
    if not v_hats:
        return {"ent_action_nmi_verb": float("nan"),
                "ent_action_nmi_verb_entity": float("nan"),
                "ent_action_nmi_shuffle": float("nan"),
                "ent_action_nmi_verb_pass": False}

    nmi_verb = _nmi(v_hats, verb_only)
    nmi_ve = _nmi(v_hats, verb_entity)

    # Shuffle baseline: permute v_hat (destroys alignment) and recompute vs verb_only.
    rng = np.random.RandomState(0)
    shuffled = list(v_hats)
    rng.shuffle(shuffled)
    nmi_shuffle = _nmi(shuffled, verb_only)

    metrics["ent_action_nmi_verb"] = nmi_verb
    metrics["ent_action_nmi_verb_entity"] = nmi_ve
    metrics["ent_action_nmi_shuffle"] = nmi_shuffle
    # Pre-registered bar: >= 0.2 AND comfortably above shuffle.
    metrics["ent_action_nmi_verb_pass"] = bool(nmi_verb >= 0.2 and nmi_verb > nmi_shuffle)
    metrics["ent_action_n_pairs"] = len(v_hats)

    # Cluster -> verb contingency artifact.
    if out_dir is not None:
        contingency: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for vh, verb in zip(v_hats, verb_only):
            contingency[str(vh)][verb] += 1
        with open(out_dir / f"action_nmi_contingency_{epoch_str}.json", "w") as f:
            json.dump({k: dict(v) for k, v in contingency.items()}, f, indent=2)

    return metrics


@torch.no_grad()
def _ood_ladder(model, tokenizer, labeled_dir, splits, subsample, device,
                max_text_tokens, append_eos, out_dir, epoch_str,
                hard_nn_per_query: int = 40) -> dict:
    """(§3b) OOD ladder: CE + hard-MRR + chrF per split on subsampled adjacent pairs.

    Reuses the existing v2 machinery (token_ce, _compute_retrieval_mrr, _chrf, decoder.generate)
    on the FIRST `subsample` adjacent pairs flattened from each split's chains (deterministic)."""
    metrics: dict = {}
    mrr_by_split: dict[str, float] = {}

    for split in splits:
        chains = _load_labeled_split(labeled_dir, split, tokenizer, max_text_tokens, append_eos)
        # Flatten to adjacent pairs (deterministic order), with originating chain id.
        pairs = []  # (src_ids, src_pad, tgt_ids, tgt_pad, chain_idx)
        for ci, ch in enumerate(chains):
            ids, pad = ch["ids"], ch["pad"]
            for i in range(len(ids) - 1):
                pairs.append((ids[i], pad[i], ids[i + 1], pad[i + 1], ci))
                if len(pairs) >= subsample:
                    break
            if len(pairs) >= subsample:
                break

        if not pairs:
            continue

        ce_vals: list[float] = []
        chrf_vals: list[float] = []
        all_zhat, all_z, chain_ids = [], [], []
        samples = []

        for (si, sp, ti, tp, cid) in pairs:
            si_b = si.unsqueeze(0).to(device)
            sp_b = sp.unsqueeze(0).to(device)
            ti_b = ti.unsqueeze(0).to(device)
            tp_b = tp.unsqueeze(0).to(device)
            out = model.forward_v2(si_b, sp_b, ti_b, tp_b, tau=1.0, hard=True)

            logits = out.get("logits")
            if logits is not None:
                ce_vals.append(token_ce(logits, ti_b, pad_id=_PAD_ID).item())

            zhat, z_tgt = out.get("zhat"), out.get("z_target")
            if zhat is not None and z_tgt is not None:
                all_zhat.append(zhat.squeeze(0).cpu())
                all_z.append(z_tgt.squeeze(0).cpu())
                chain_ids.append(cid)

            # chrF on greedy-decoded a*.
            a = out.get("a")
            if a is not None and hasattr(model, "decoder"):
                gen_ids = model.decoder.generate(a, max_tokens=max_text_tokens, temperature=0.0)
                gen_text = _strip_decode(tokenizer, gen_ids[0].tolist())
                gold_text = _strip_decode(tokenizer, ti.tolist())
                chrf_vals.append(_chrf(gen_text, gold_text))
                if len(samples) < 16:
                    samples.append({
                        "text_t": _strip_decode(tokenizer, si.tolist()),
                        "gold_t1": gold_text, "gen_greedy": gen_text,
                    })

        if ce_vals:
            metrics[f"ent_{split}_ce"] = float(np.mean(ce_vals))
        if chrf_vals:
            metrics[f"ent_{split}_chrf"] = float(np.mean(chrf_vals))
        if len(all_zhat) >= 2:
            Zhat = torch.stack(all_zhat)
            Z = torch.stack(all_z)
            mrr = _compute_retrieval_mrr(Zhat, Z, chain_ids, hard_nn_per_query=hard_nn_per_query)
            metrics[f"ent_{split}_hard_mrr"] = mrr["hard_mrr"]
            mrr_by_split[split] = mrr["hard_mrr"]

        if out_dir is not None and samples:
            with open(out_dir / f"entity_samples_{split}_{epoch_str}.json", "w") as f:
                json.dump(samples, f, indent=2)

    # Ladder monotonicity (iid >= near > far) on hard-MRR.
    ladder = ["test_iid", "test_ood_near", "test_ood_far"]
    seq = [mrr_by_split[s] for s in ladder if s in mrr_by_split]
    if len(seq) >= 2:
        metrics["ent_ladder_monotone_mrr"] = bool(
            all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
        )
    return metrics


@torch.no_grad()
def _rollout_fidelity(model, tokenizer, chains, max_depth, device,
                      max_text_tokens, out_dir, epoch_str) -> dict:
    """(§3c) Rollout fidelity depth 1..D, teacher-forced AND prior-sampled actions.

    For each chain (length >= max_depth+1 so depth D has a gold target):
      1. encode s0 -> k0.
      2. TF: action from pair posterior q(v|s_{h-1},s_h) on GOLD states.
         PR: action from prior p(v|pooled current latent), current latent = ROLLED state.
      3. apply operators stepwise from k0 (threading conditioning + norm budget via _op_apply).
      4. at each depth d greedy-decode decoder.generate(a_d) -> exact-match + chrF vs gold chain[d].

    Depth-1 teacher-forced reduces to the standard forward (single apply on k0 with the
    posterior action) — the test asserts this equivalence."""
    usable = [ch for ch in chains if len(ch["ids"]) >= max_depth + 1 and len(ch["actions"]) >= max_depth]
    n_skipped = len(chains) - len(usable)

    # Accumulators: depth -> list of exact/chrf for each source.
    acc = {src: {d: {"exact": [], "chrf": []} for d in range(1, max_depth + 1)}
           for src in ("tf", "pr")}
    transcripts = []

    for ch in usable:
        ids, pad, gold = ch["ids"], ch["pad"], ch["chain"]
        # Encode s0 -> k0 (start nouns).
        _, k0, _ = model.encoder(ids[0].unsqueeze(0).to(device), pad[0].unsqueeze(0).to(device))

        row = {"types": ch["types"], "tf": {}, "pr": {}}
        for src in ("tf", "pr"):
            k = k0
            for d in range(1, max_depth + 1):
                if src == "tf":
                    # Posterior on gold (s_{d-1}, s_d).
                    v_onehot, v_logits, _ = model.transition(
                        ids[d - 1].unsqueeze(0).to(device), pad[d - 1].unsqueeze(0).to(device),
                        ids[d].unsqueeze(0).to(device), pad[d].unsqueeze(0).to(device),
                        tau=1.0, hard=True,
                    )
                    v = int(v_logits.squeeze(0).argmax(-1).item())
                else:
                    # Prior on the ROLLED current latent's source pool. The prior reads the
                    # encoder pool of the current *text*; in autonomous rollout the canonical
                    # source is the decoded current state. We approximate with the prior over
                    # the rolled state's decode (greedy), matching model.rollout's prior path.
                    if d == 1:
                        pool = model._prior_pool(
                            ids[0].unsqueeze(0).to(device), pad[0].unsqueeze(0).to(device)
                        )
                    else:
                        pool = model._prior_pool(prev_gen_ids, prev_gen_pad)
                    p_logits = model.prior(pool)
                    v = int(p_logits.squeeze(0).argmax(-1).item())

                v_oh = F.one_hot(
                    torch.tensor([v], device=device), num_classes=model.n_verbs
                ).to(k.dtype)
                a = _op_apply(model, k, v_oh)
                gen_ids = model.decoder.generate(a, max_tokens=max_text_tokens, temperature=0.0)
                gen_text = _strip_decode(tokenizer, gen_ids[0].tolist())
                gold_text = gold[d]
                acc[src][d]["exact"].append(float(gen_text.strip() == gold_text.strip()))
                acc[src][d]["chrf"].append(_chrf(gen_text, gold_text))
                row[src][f"d{d}"] = gen_text
                # next hop composes on this hop's rolled output.
                k = a
                if src == "pr":
                    # Re-encode the generated text for the next prior pool (autonomous).
                    gen_full = gen_ids[0].tolist()
                    gi, gp = _pad_gen_ids(gen_full, max_text_tokens)
                    prev_gen_ids = gi.unsqueeze(0).to(device)
                    prev_gen_pad = gp.unsqueeze(0).to(device)
        if len(transcripts) < 16:
            for d in range(1, max_depth + 1):
                row.setdefault("gold", {})[f"d{d}"] = gold[d]
            transcripts.append(row)

    metrics: dict = {}
    for src in ("tf", "pr"):
        for d in range(1, max_depth + 1):
            ex = acc[src][d]["exact"]
            cf = acc[src][d]["chrf"]
            metrics[f"ent_rollout_{src}_exact_d{d}"] = float(np.mean(ex)) if ex else float("nan")
            metrics[f"ent_rollout_{src}_chrf_d{d}"] = float(np.mean(cf)) if cf else float("nan")
    metrics["ent_rollout_n_chains"] = len(usable)
    metrics["ent_rollout_n_skipped"] = n_skipped

    if out_dir is not None and transcripts:
        with open(out_dir / f"entity_rollout_{epoch_str}.json", "w") as f:
            json.dump(transcripts, f, indent=2)
    return metrics


def _pad_gen_ids(gen_ids: list[int], max_text_tokens: int):
    """Pad/truncate a generated id list to (T,) long + (T,) bool pad mask.

    Trims everything from the first <eos> onward (the decoded current state) and pads with
    PAD so the prior's encoder pool sees a clean re-encode (campaign §3c PR path)."""
    ids = list(gen_ids)
    if _EOS_ID in ids:
        ids = ids[: ids.index(_EOS_ID)]
    ids = ids[:max_text_tokens]
    ids = ids + [_PAD_ID] * (max_text_tokens - len(ids))
    t = torch.tensor(ids, dtype=torch.long)
    return t, (t == _PAD_ID)


@torch.no_grad()
def eval_entity_world(model, ew_cfg, device, tokenizer, max_text_tokens: int = 64,
                      out_dir=None, epoch: int | None = None,
                      append_eos: bool = True) -> dict:
    """Entity-world diagnostics entry point (campaign §3). Returns a flat dict of `ent_*`
    scalars; saves per-family artifact JSON to out_dir.

    Three metric families:
      (a) action-recovery NMI vs oracle labels + shuffle  [§3a]
      (b) OOD ladder CE + hard-MRR + chrF per split        [§3b]
      (c) rollout fidelity depth 1..D, TF + PR             [§3c]

    Robust to a black-box model (gated MLP operator): rollout still runs via _op_apply,
    which unpacks the norm-budget tuple contract if present."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    epoch_str = f"epoch{epoch}" if epoch is not None else "latest"

    labeled_dir = ew_cfg.labeled_dir
    splits = list(ew_cfg.splits)
    subsample = ew_cfg.subsample
    n_rollout = ew_cfg.n_rollout_chains
    max_depth = ew_cfg.rollout_max_depth
    nmi_split = ew_cfg.action_recovery_split

    _load_manifest(labeled_dir)  # warm/validate (schema unused by this suite directly)
    metrics: dict = {}

    # (a) action-recovery NMI — uses the nmi split's labeled twin.
    try:
        nmi_chains = _load_labeled_split(
            labeled_dir, nmi_split, tokenizer, max_text_tokens, append_eos,
            max_chains=max(subsample, 64),
        )
        metrics.update(_action_recovery_nmi(model, nmi_chains, device, out_dir, epoch_str))
    except Exception as exc:
        metrics["_ent_action_nmi_error"] = str(exc)

    # (b) OOD ladder.
    try:
        metrics.update(_ood_ladder(
            model, tokenizer, labeled_dir, splits, subsample, device,
            max_text_tokens, append_eos, out_dir, epoch_str,
        ))
    except Exception as exc:
        metrics["_ent_ladder_error"] = str(exc)

    # (c) rollout fidelity — from test_iid chains long enough for depth D.
    try:
        roll_chains = _load_labeled_split(
            labeled_dir, "test_iid", tokenizer, max_text_tokens, append_eos,
            max_chains=n_rollout * 4,  # over-read; filter to length >= D+1, cap below
        )
        # Keep only chains long enough, then cap to n_rollout.
        roll_chains = [c for c in roll_chains if len(c["ids"]) >= max_depth + 1][:n_rollout]
        metrics.update(_rollout_fidelity(
            model, tokenizer, roll_chains, max_depth, device,
            max_text_tokens, out_dir, epoch_str,
        ))
    except Exception as exc:
        metrics["_ent_rollout_error"] = str(exc)

    # (d) target-recovery metrics (v4 §1.6) — only when use_targeted_actions=True.
    if getattr(model, "use_targeted_actions", False):
        try:
            nmi_chains_for_target = _load_labeled_split(
                labeled_dir, nmi_split, tokenizer, max_text_tokens, append_eos,
                max_chains=max(subsample, 64),
            )
            metrics.update(_target_recovery(model, nmi_chains_for_target, device, out_dir, epoch_str))
        except Exception as exc:
            metrics["_ent_target_recovery_error"] = str(exc)

    # (e) separation AUC (v4 §C5) — every-N-epochs metric, the v4 success criterion.
    try:
        sep_chains = _load_labeled_split(
            labeled_dir, nmi_split, tokenizer, max_text_tokens, append_eos,
            max_chains=max(subsample, 64),
        )
        metrics.update(_separation_auc(model, sep_chains, device))
    except Exception as exc:
        metrics["_ent_separation_auc_error"] = str(exc)

    # Normalise scalars to Python primitives (mirror eval_diagnostics_v2 tail).
    out: dict = {}
    for km, vm in metrics.items():
        if isinstance(vm, (bool, np.bool_)):
            out[km] = bool(vm)
        elif isinstance(vm, (int, np.integer)):
            out[km] = int(vm)
        elif isinstance(vm, str):
            out[km] = vm
        else:
            try:
                out[km] = float(vm)
            except (TypeError, ValueError):
                out[km] = vm
    return out


# ===========================================================================
# v4 diagnostics: target-recovery (§1.6) and separation-AUC (§C5)
# ===========================================================================

@torch.no_grad()
def _target_recovery(model, chains, device, out_dir, epoch_str) -> dict:
    """(v4 §1.6) Target-recovery: inferred mask g_hard vs oracle moved-entity.

    For each adjacent (s_t, s_{t+1}) pair with oracle label "<verb>@<entity_idx>":
      - Run the posterior TransitionEncoder.forward_mask(k, k_tgt) to get g_logits (B,M).
      - Hard-threshold at 0.5 to get g_hard (B,M) ∈ {0,1}.
      - The oracle "moved slots" are those belonging to entity `entity_idx`. Since
        slot↔entity is latent (unlabeled), we do a HUNGARIAN best-match assignment
        over the dataset to find the best slot-to-entity-index mapping, then compute F1.
    Also reports mask-sparsity (mean fraction of slots with g_hard=1) and a shuffle
    baseline.

    Returns ent_target_recovery_f1, ent_target_recovery_nmi, ent_target_recovery_shuffle,
    ent_target_mask_density, ent_target_recovery_pass.
    """
    # Guard: requires use_targeted_actions.
    if not getattr(model, "use_targeted_actions", False):
        return {}

    device = torch.device(device) if not isinstance(device, torch.device) else device

    # Collect per-pair (g_hard, oracle_entity_idx, n_entities)
    pair_masks = []   # list of (M,) int arrays (g_hard)
    pair_actors = []  # list of actor entity indices (int)
    pair_n_ent = []   # list of n_entities per chain step

    for ch in chains:
        ids, pad, actions = ch["ids"], ch["pad"], ch["actions"]
        n_pairs = min(len(ids) - 1, len(actions))
        # Determine n_entities per chain from types (if available).
        n_ent = len(ch.get("types", [])) or 1
        for i in range(n_pairs):
            try:
                actor_idx = int(actions[i].rsplit("@", 1)[1])
            except (IndexError, ValueError):
                continue
            src_ids = ids[i].unsqueeze(0).to(device)
            src_pad_t = pad[i].unsqueeze(0).to(device)
            tgt_ids = ids[i + 1].unsqueeze(0).to(device)
            tgt_pad_t = pad[i + 1].unsqueeze(0).to(device)
            # Get k (start nouns) from the encoder.
            try:
                _, k, _ = model.encoder(src_ids, src_pad_t)
                # Get k_tgt via the EMA target encoder (the same path as TransitionEncoder.forward_mask).
                k_tgt = model._target_slots(tgt_ids, tgt_pad_t)   # (1, M, dn) detached
                g_logits = model.transition.forward_mask(k, k_tgt)  # (1, M)
                g_hard = (torch.sigmoid(g_logits) > 0.5).squeeze(0).cpu().numpy().astype(int)  # (M,)
            except Exception:
                continue
            pair_masks.append(g_hard)
            pair_actors.append(actor_idx)
            pair_n_ent.append(n_ent)

    if not pair_masks:
        return {
            "ent_target_recovery_f1": float("nan"),
            "ent_target_recovery_nmi": float("nan"),
            "ent_target_recovery_shuffle": float("nan"),
            "ent_target_mask_density": float("nan"),
            "ent_target_recovery_pass": False,
        }

    M = pair_masks[0].shape[0]
    masks_arr = np.stack(pair_masks)       # (N, M)
    actors_arr = np.array(pair_actors)     # (N,)

    # --- Mask density ---
    mask_density = float(masks_arr.mean())

    # --- NMI between mask-as-label and oracle actor ---
    # For NMI, represent each pair's mask as a tuple (which slots fired) -> integer label.
    # This is too sparse; instead use the dot-product "actor score" approach:
    # For each slot s and entity e, count how often g_hard[s]=1 when oracle actor=e.
    # The slot assignment is the Hungarian match maximizing this co-occurrence.
    n_actors = int(actors_arr.max()) + 1
    slot_entity_cooccur = np.zeros((M, n_actors), dtype=float)
    for g, a in zip(pair_masks, pair_actors):
        for s in range(M):
            slot_entity_cooccur[s, a] += g[s]

    # Hungarian assignment: assign slots to entities to maximize co-occurrence.
    try:
        from scipy.optimize import linear_sum_assignment
        # We want to maximize co-occurrence -> minimize negative.
        row_ind, col_ind = linear_sum_assignment(-slot_entity_cooccur)
        slot_to_entity = {s: e for s, e in zip(row_ind, col_ind)}
        # For slots not assigned, assign them to the entity with max co-occurrence.
        for s in range(M):
            if s not in slot_to_entity:
                slot_to_entity[s] = int(slot_entity_cooccur[s].argmax())
    except ImportError:
        # Fallback: greedy assignment by max co-occurrence.
        slot_to_entity = {s: int(slot_entity_cooccur[s].argmax()) for s in range(M)}

    # Compute F1: for each pair, predicted moved slots = g_hard=1 slots;
    # oracle moved slots = slots assigned to the actor entity.
    tp_total = fp_total = fn_total = 0
    for g, actor in zip(pair_masks, pair_actors):
        predicted = set(s for s in range(M) if g[s] == 1)
        oracle_slots = set(s for s in range(M) if slot_to_entity[s] == actor)
        tp = len(predicted & oracle_slots)
        fp = len(predicted - oracle_slots)
        fn = len(oracle_slots - predicted)
        tp_total += tp; fp_total += fp; fn_total += fn
    prec = tp_total / max(tp_total + fp_total, 1)
    rec = tp_total / max(tp_total + fn_total, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # NMI between slot-assignment-score (per pair: majority entity of fired slots) and oracle.
    pred_entities = []
    for g in pair_masks:
        fired = [s for s in range(M) if g[s] == 1]
        if fired:
            entity_votes = [slot_to_entity[s] for s in fired]
            pred_e = max(set(entity_votes), key=entity_votes.count)
        else:
            pred_e = 0
        pred_entities.append(pred_e)
    nmi_val = _nmi(pred_entities, actors_arr.tolist())

    # Shuffle baseline: permute predicted entities.
    rng = np.random.RandomState(0)
    shuffled_pred = list(pred_entities)
    rng.shuffle(shuffled_pred)
    nmi_shuffle = _nmi(shuffled_pred, actors_arr.tolist())

    metrics = {
        "ent_target_recovery_f1": float(f1),
        "ent_target_recovery_nmi": float(nmi_val),
        "ent_target_recovery_shuffle": float(nmi_shuffle),
        "ent_target_mask_density": float(mask_density),
        "ent_target_recovery_pass": bool(f1 > nmi_shuffle + 0.05),
    }

    # Save per-pair contingency.
    if out_dir is not None:
        contingency = [
            {"actor": int(a), "mask": g.tolist(), "pred_entity": int(pe)}
            for a, g, pe in zip(actors_arr, pair_masks, pred_entities)
        ]
        with open(Path(out_dir) / f"target_recovery_{epoch_str}.json", "w") as f:
            json.dump({"metrics": metrics, "slot_to_entity": {str(k): int(v) for k, v in slot_to_entity.items()},
                       "n_pairs": len(pair_masks), "contingency": contingency[:200]}, f, indent=2)
    return metrics


@torch.no_grad()
def _separation_auc(model, chains, device) -> dict:
    """(v4 §C5) Separation AUC: linear-probe AUC on hard pools.

    Ports the core of scripts/jepa_separation_diag.py (bb01bfd) into a per-epoch
    metric. Measures whether the encoder's latent space has enough discriminative
    structure to rank the correct next-state above same-chain distractors.

    Three variants: ema, online, slot_mean (mirrors the script).
    Reports ent_separation_auc, ent_separation_auc_ema, ent_separation_auc_online,
    ent_separation_auc_slot_mean. The v4 success criterion is ent_separation_auc > 0.7.

    No-ops cleanly on models without the required attributes (returns NaN scalars).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    # Collect (src_ids, src_pad, tgt_ids, tgt_pad, chain_id) from chains.
    src_ids_list, src_pad_list, tgt_ids_list, tgt_pad_list, cid_list = [], [], [], [], []
    for ci, ch in enumerate(chains):
        ids, pad = ch["ids"], ch["pad"]
        for i in range(len(ids) - 1):
            src_ids_list.append(ids[i])
            src_pad_list.append(pad[i])
            tgt_ids_list.append(ids[i + 1])
            tgt_pad_list.append(pad[i + 1])
            cid_list.append(ci)

    if len(src_ids_list) < 4:
        return {
            "ent_separation_auc": float("nan"),
            "ent_separation_auc_ema": float("nan"),
            "ent_separation_auc_online": float("nan"),
            "ent_separation_auc_slot_mean": float("nan"),
        }

    N = len(src_ids_list)
    T = src_ids_list[0].shape[0]
    src_ids_t = torch.stack(src_ids_list)     # (N, T)
    src_pad_t = torch.stack(src_pad_list)     # (N, T)
    tgt_ids_t = torch.stack(tgt_ids_list)     # (N, T)
    tgt_pad_t = torch.stack(tgt_pad_list)     # (N, T)
    chain_ids = np.array(cid_list)

    # Encode variants in batches.
    BS = 64
    d_noun = getattr(model, "d_noun", None)
    if d_noun is None:
        return {"ent_separation_auc": float("nan")}

    ema_vecs    = torch.zeros(N, d_noun)
    online_vecs = torch.zeros(N, d_noun)
    slot_vecs   = torch.zeros(N, d_noun)
    zhat_vecs   = torch.zeros(N, d_noun)

    has_ema = hasattr(model, "ema") and hasattr(model.ema, "pool_raw")
    has_online = hasattr(model, "_online_bundle") and hasattr(model._online_bundle, "pool_raw")
    has_encoder = hasattr(model, "encoder")
    has_fwd = hasattr(model, "forward_v2")

    for s in range(0, N, BS):
        ti = tgt_ids_t[s:s + BS].to(device)
        tp = tgt_pad_t[s:s + BS].to(device)
        si = src_ids_t[s:s + BS].to(device)
        sp = src_pad_t[s:s + BS].to(device)
        b = ti.shape[0]
        if has_ema:
            try:
                ema_vecs[s:s + b] = model.ema.pool_raw(ti, tp).cpu()
            except Exception:
                pass
        if has_online:
            try:
                online_vecs[s:s + b] = model._online_bundle.pool_raw(ti, tp).cpu()
            except Exception:
                pass
        if has_encoder:
            try:
                _, k, _ = model.encoder(ti, tp)
                slot_vecs[s:s + b] = k.mean(dim=1).cpu()
            except Exception:
                pass
        if has_fwd:
            try:
                out = model.forward_v2(si, sp, ti, tp, tau=1.0, hard=True)
                zh = out.get("zhat")
                if zh is not None:
                    zhat_vecs[s:s + b] = zh.cpu()
            except Exception:
                pass

    # Build hard pools (mirrors jepa_separation_diag.build_pools exactly).
    same_chain = defaultdict(list)
    for i, cid in enumerate(chain_ids):
        for j, cid2 in enumerate(chain_ids):
            if j != i and cid2 == cid:
                same_chain[i].append(j)

    HARD_NN = 40
    zt_n = F.normalize(ema_vecs.float(), dim=-1)
    sims_mat = (zt_n @ zt_n.T).numpy()
    hard_pools = []
    for i in range(N):
        base = [i] + same_chain[i]
        baseset = set(base)
        order = np.argsort(-sims_mat[i])
        nn = [int(j) for j in order if j not in baseset][:HARD_NN]
        hard_pools.append(base + nn)

    # Run the linear-probe AUC for each variant.
    def _probe_auc(cand_vecs):
        """Compute LR-probe AUC (mirrors jepa_separation_diag.linear_probe_auc)."""
        rng2 = np.random.default_rng(0)
        X_list, y_list = [], []
        MAX_NEG = 10
        for qi, pool in enumerate(hard_pools):
            qv = zhat_vecs[qi].float().numpy()
            gold_idx = pool[0]
            cv = cand_vecs[gold_idx].float().numpy()
            feat = np.concatenate([cv, qv, np.abs(cv - qv), cv * qv])
            X_list.append(feat); y_list.append(1)
            distractors = pool[1:]
            neg_sel = rng2.choice(len(distractors),
                                   size=min(MAX_NEG, len(distractors)), replace=False)
            for ni in neg_sel:
                cv2 = cand_vecs[distractors[ni]].float().numpy()
                feat2 = np.concatenate([cv2, qv, np.abs(cv2 - qv), cv2 * qv])
                X_list.append(feat2); y_list.append(0)
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        n = len(X)
        if n < 10:
            return float("nan")
        perm = rng2.permutation(n)
        X = X[perm]; y = y[perm]
        split = int(0.8 * n)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        mu = X_tr.mean(0, keepdims=True); std = X_tr.std(0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / std; X_te = (X_te - mu) / std
        if len(np.unique(y_te)) < 2:
            return float("nan")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            return float(roc_auc_score(y_te, proba))
        except Exception:
            # Fallback: cosine-score AUC
            d = cand_vecs.shape[1]
            scores_te = (X_te[:, :d] * X_te[:, d:2 * d]).sum(1)
            try:
                from sklearn.metrics import roc_auc_score
                return float(roc_auc_score(y_te, scores_te))
            except Exception:
                # Pure-numpy trapz AUC
                try:
                    order = np.argsort(-scores_te)
                    y_sorted = y_te[order]
                    n_pos = y_sorted.sum(); n_neg = len(y_sorted) - n_pos
                    if n_pos == 0 or n_neg == 0:
                        return float("nan")
                    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
                    tpr = tp / n_pos; fpr = fp / n_neg
                    return float(np.trapz(tpr, fpr))
                except Exception:
                    return float("nan")

    auc_ema    = _probe_auc(ema_vecs)    if has_ema else float("nan")
    auc_online = _probe_auc(online_vecs) if has_online else float("nan")
    auc_slot   = _probe_auc(slot_vecs)   if has_encoder else float("nan")
    best_auc   = max(v for v in [auc_ema, auc_online, auc_slot] if not np.isnan(v)) \
                 if any(not np.isnan(v) for v in [auc_ema, auc_online, auc_slot]) \
                 else float("nan")

    return {
        "ent_separation_auc":           float(best_auc),
        "ent_separation_auc_ema":       float(auc_ema),
        "ent_separation_auc_online":    float(auc_online),
        "ent_separation_auc_slot_mean": float(auc_slot),
        # v4 success criterion: AUC moving from 0.53 toward > 0.7
        "ent_separation_auc_pass":      bool(best_auc > 0.7),
    }


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
