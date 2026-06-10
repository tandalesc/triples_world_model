#!/usr/bin/env python3
"""Retraction probe — remove a past event from a rolled latent state.

Engram-wm headline demonstration (jepa_entity_campaign.md §4): retract event j from a
K-event rolled latent state by applying the STRUCTURAL INVERSE of that event (using its
stored phase offset + scale delta), and show the result lands near the oracle replay
*without* that event.

For the structured operator (rotation_scale) with norm budget on, ``inverse_apply`` is
exact (round-trip == identity to fp32 eps).  For the black-box (gated_mlp), the probe
prints the asymmetry message, records ``{"backend":"blackbox","inverse_supported":false}``,
and exits 0 — the negative result IS the data point.

THREE-POINT → FOUR-POINT BRACKET (amendment)
--------------------------------------------
Every reconstruction is scored by cosine to the SAME reference: the fresh encoding of the
oracle's true *j-deleted* final state (``z_oracle``).  We report a four-rung bracket whose
rungs are EXPECTED to be monotone non-decreasing:

    do_nothing  <=  algebraic_retraction  <=  model_replay  <=  reencode_ceiling

  - do_nothing            : the rolled state WITH event j still present.  Worst — j was
                            never removed.
  - algebraic_retraction  : apply the structured INVERSE of event j to the *contaminated*
                            rolled state (the H offsets used by the later events were
                            computed on the WITH-j path).  The abelian operator commutes the
                            matrices exactly, but the SELECTION (H = θ_offset, a function of
                            the per-hop modulus) was keyed on the wrong, j-contaminated path.
  - model_replay          : the model's HONEST counterfactual — re-roll its own latent
                            trajectory on the j-DELETED action history (encode s_0, apply the
                            teacher-forced posterior actions skipping j, with H offsets
                            RECOMPUTED FRESH on the counterfactual path).  No path
                            contamination; the only error left is the model's own
                            dynamics/encode fidelity.
  - reencode_ceiling      : cosine of ``z_oracle`` to itself = 1.0 — the literal upper bound
                            (the best any reconstruction could match the reference encoding).

DERIVED METRICS — opposite remedies (documented because they are easy to conflate):

  - selection_drift = model_replay_cos - retract_cos   (MODEL / SELECTION side)
      How much the algebraic inverse loses RELATIVE to the honest fresh re-roll.  It is a
      pure SELECTION defect: both paths use the exact same operator matrices, but the
      algebraic retraction inherits H offsets computed on the j-contaminated rolled path,
      whereas model_replay recomputes H on the clean counterfactual path.  Large
      selection_drift ⇒ the polar conditioner H is path-dependent (non-commuting SELECTIONS).
      REMEDY: make the retraction recompute / store the counterfactual H offsets (a
      selection-side fix — recompute the conditioner on the post-retraction path, or store
      per-hop offsets keyed to the deleted history), NOT a better world model.

  - dynamics_gap = ceiling_cos - model_replay_cos      (WORLD / DYNAMICS side)
      How far the model's honest counterfactual rollout sits below the perfect encoding of
      the true state.  Even with zero path contamination, the rolled latent differs from a
      fresh encode by the model's own dynamics + decoder error.  Large dynamics_gap ⇒ the
      world model's rollout is not faithful.  REMEDY: train a better dynamics/operator/encoder
      (a world-side fix — more capacity, more data, lower rollout drift), NOT a smarter
      inverse.  A selection-side fix does NOTHING for dynamics_gap and vice-versa — they are
      orthogonal failure channels, which is the whole point of reporting both.

Usage::

    uv run python scripts/probe_retraction.py \\
        --ckpt results/jepa_ent_s0/model_latest.pt \\
        --config configs/jepa/jepa_ent_s0.json \\
        --labeled data/entity_world/test_iid_labeled.jsonl \\
        --K 4 --n_chains 256 --retract_j 2 \\
        --out results/jepa_ent_s0/retraction.json

Note: this script can also be used in TEST MODE (no checkpoint) to verify the
retraction math on a synthetic toy model.  Pass ``--test_mode`` to use a freshly
initialized model instead of a checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------------------
# Generator module loader (scripts/ is not a package)
# ---------------------------------------------------------------------------

def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "generate_entity_world", REPO / "scripts" / "generate_entity_world.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def _load_tokenizer(tokenizer_path: str, max_text_tokens: int, append_eos: bool = True):
    """Load a HuggingFace tokenizers BPE and return an encode function.

    Returns a callable ``encode(text) -> (ids_tensor, pad_tensor)`` that matches the
    dataset's ``_encode`` / ``_insert_eos`` logic (BPE + optional EOS(=4) + zero-pad to
    ``max_text_tokens``).
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError(
            "The 'tokenizers' package is required: uv add tokenizers"
        ) from e

    tok = Tokenizer.from_file(tokenizer_path)
    T = max_text_tokens
    eos_id = 4

    def encode(text: str):
        enc = tok.encode(text)
        ids = enc.ids
        if append_eos:
            ids = ids + [eos_id]
        # Truncate or pad to T.
        if len(ids) > T:
            ids = ids[:T]
        pad_mask = [False] * len(ids) + [True] * (T - len(ids))
        ids_padded = ids + [0] * (T - len(ids))
        ids_t = torch.tensor(ids_padded, dtype=torch.long)
        pad_t = torch.tensor(pad_mask, dtype=torch.bool)
        return ids_t, pad_t

    return encode


# ---------------------------------------------------------------------------
# Model + readout helpers
# ---------------------------------------------------------------------------

def _readout_pool(model, a: torch.Tensor, s_acc: torch.Tensor | None) -> torch.Tensor:
    """Pool ``a`` (B,M,dn) [+ optional scale accumulator] through the readout head.

    When ``use_norm_budget`` is on, the model has a ``scale_readout_proj`` that maps
    concat([a, s_acc.unsqueeze(-1)]) from (B,M,dn+1) -> (B,M,dn) before the readout.
    When it's off, pool directly.  Followed by the predictor for the anchor geometry
    (matching the training anchor path).
    """
    use_budget = getattr(model, "use_norm_budget", False)
    scale_proj = getattr(model, "scale_readout_proj", None)

    if use_budget and scale_proj is not None and s_acc is not None:
        # Augmented slot: concat [a, s_acc.unsqueeze(-1)] -> (B,M,dn+1)
        a_aug = torch.cat([a, s_acc.unsqueeze(-1)], dim=-1)  # (B,M,dn+1)
        a_for_readout = scale_proj(a_aug)                     # (B,M,dn)
    else:
        a_for_readout = a

    pooled = model.readout(a_for_readout)      # (B,dn)
    zhat = model.predictor(pooled)             # (B,dn)
    return zhat


def _encode_fresh(model, ids: torch.Tensor, pad: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    """Encode a state to nouns k (B,M,dn)."""
    ids = ids.unsqueeze(0).to(device)
    pad = pad.unsqueeze(0).to(device)
    with torch.no_grad():
        _, k, _ = model.encoder(ids, pad)
    return k


def _readout_z(model, a: torch.Tensor, s_acc: torch.Tensor | None,
               device: torch.device) -> torch.Tensor:
    """Compute the pooled anchor vector z from nouns ``a`` and optional scale ``s_acc``."""
    with torch.no_grad():
        z = _readout_pool(model, a, s_acc)
    return z


def _posterior_action(model, src_ids, src_pad, tgt_ids, tgt_pad,
                       device: torch.device):
    """Compute posterior v_onehot (B,V) for a pair."""
    with torch.no_grad():
        v_onehot, v_logits, _ = model.transition(
            src_ids.unsqueeze(0).to(device),
            src_pad.unsqueeze(0).to(device),
            tgt_ids.unsqueeze(0).to(device),
            tgt_pad.unsqueeze(0).to(device),
            tau=1.0,
            hard=True,
        )
    return v_onehot  # (1,V)


def _apply_one_hop(model, k_in: torch.Tensor, v_onehot: torch.Tensor,
                   device: torch.device, use_budget: bool):
    """Apply one hop.  Returns (a_out, theta_offset, scale_delta).

    ``theta_offset`` is None when polar conditioning is off.
    ``scale_delta`` is None when norm budget is off.
    """
    B, M, dn = k_in.shape
    v_slots = v_onehot.unsqueeze(1).expand(B, M, -1)  # (B,M,V)

    conditioner = getattr(model, "conditioner", None)
    operator = model.operator

    with torch.no_grad():
        if conditioner is not None:
            theta_offset = conditioner(k_in)  # (B,M,nb)
        else:
            theta_offset = None

        if use_budget:
            result = operator.apply(k_in, v_slots, theta_offset=theta_offset,
                                    norm_budget=True)
            if isinstance(result, tuple):
                a_out, scale_delta = result
            else:
                # Norm budget not yet implemented on this operator; degrade gracefully.
                a_out = result
                scale_delta = torch.zeros(B, M, device=device)
        else:
            a_out = operator.apply(k_in, v_slots, theta_offset=theta_offset)
            scale_delta = None

    return a_out, theta_offset, scale_delta


def _inverse_one_hop(model, a: torch.Tensor, v_onehot: torch.Tensor,
                     theta_offset, scale_delta,
                     device: torch.device, use_budget: bool):
    """Apply inverse of one hop.  Returns k_retract.

    Raises ``NotImplementedError`` for black-box operators (GatedMLPTransition).
    """
    B, M, dn = a.shape
    v_slots = v_onehot.unsqueeze(1).expand(B, M, -1)  # (B,M,V)
    operator = model.operator

    with torch.no_grad():
        if use_budget:
            k_retract = operator.inverse_apply(
                a, v_slots,
                theta_offset=theta_offset,
                norm_budget=True,
                scale_delta=scale_delta,
            )
        else:
            k_retract = operator.inverse_apply(a, v_slots, theta_offset=theta_offset)

    return k_retract


# ---------------------------------------------------------------------------
# Oracle replay helper
# ---------------------------------------------------------------------------

def _oracle_replay_minus_j(gen_mod, chain_record: dict, j: int) -> list[dict]:
    """Replay the chain omitting action j (1-based).

    Uses ``ew.replay_chain`` with ``actions_minus_j = actions[:j-1] + actions[j:]``.
    Returns snapshots (list of per-entity state dicts per step).
    """
    types = chain_record["types"]
    actions = chain_record["actions"]
    initial_states = chain_record["initial_states"]

    j0 = j - 1  # 0-based
    actions_minus_j = actions[:j0] + actions[j0 + 1:]
    snapshots = gen_mod.replay_chain(types, initial_states, actions_minus_j)
    return snapshots


# ---------------------------------------------------------------------------
# Main probe function
# ---------------------------------------------------------------------------

def probe_retraction(
    model,
    labeled_path: str,
    K: int,
    n_chains: int,
    retract_j: int,
    device: torch.device,
    encode_fn,
    gen_mod,
    use_budget: bool,
    out_path: str | None = None,
    _records_override: list[dict] | None = None,
) -> dict:
    """Run the retraction probe on ``n_chains`` chains from ``labeled_path``.

    Returns an aggregate result dict.  Writes JSON to ``out_path`` if given.

    The probe measures whether the structured inverse of event ``retract_j`` moves the
    rolled latent state toward the oracle-without-j target.

    ``_records_override`` (test hook): if given, use these labeled records directly instead
    of reading ``labeled_path`` (lets unit tests inject an in-memory fixture).
    """
    # ---- detect backend ----
    from twm.jepa.baseline_transition import GatedMLPTransition
    is_blackbox = isinstance(model.operator, GatedMLPTransition)

    if is_blackbox:
        print(
            "Black-box operator detected (GatedMLPTransition).\n"
            "  inverse_apply RAISES by design — that asymmetry IS the experiment.\n"
            "  Attempting inverse_apply to capture the NotImplementedError..."
        )
        # Attempt to demonstrate the raise.
        try:
            dummy = torch.zeros(1, 1, model.d_noun, device=device)
            model.operator.inverse_apply(dummy, dummy[..., 0].long())
        except NotImplementedError as e:
            print(f"  Confirmed: NotImplementedError: {e}")
        result = {
            "backend": "blackbox",
            "inverse_supported": False,
            "n_chains": 0,
            "K": K,
            "retract_j": retract_j,
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        return result

    # ---- structured operator path ----
    model.eval()

    # Load labeled data (or use an injected in-memory fixture for tests).
    if _records_override is not None:
        records = _records_override
    else:
        records = []
        with open(labeled_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    # Filter chains with enough events.
    eligible = [r for r in records if len(r.get("actions", [])) >= K
                and "initial_states" in r]
    if len(eligible) == 0:
        raise ValueError(
            f"No chains with >= {K} actions and 'initial_states' in {labeled_path}. "
            "Regenerate data with the updated generate_entity_world.py."
        )

    chains_used = eligible[:n_chains]
    print(f"Running retraction probe: {len(chains_used)} chains, K={K}, j={retract_j}")

    per_chain: list[dict] = []

    for chain_record in chains_used:
        types = chain_record["types"]
        chain = chain_record["chain"]
        actions = chain_record["actions"]
        initial_states = chain_record["initial_states"]

        # Tokenize all states.
        states_enc = [encode_fn(s) for s in chain]  # list of (ids, pad)

        # 1. Reference: direct encode of s_K (fresh encode, s_acc=0).
        s_K_ids, s_K_pad = states_enc[K]
        k_K_ref = _encode_fresh(model, s_K_ids, s_K_pad, device)
        z_ref = _readout_z(model, k_K_ref,
                           torch.zeros(1, k_K_ref.shape[1], device=device) if use_budget else None,
                           device)

        # 2. Rolled state: encode s0, apply K teacher-forced posterior actions.
        s0_ids, s0_pad = states_enc[0]
        k_rolled = _encode_fresh(model, s0_ids, s0_pad, device)
        s_acc = torch.zeros(1, k_rolled.shape[1], device=device) if use_budget else None

        # Store per-hop state for retraction.
        hop_data = []  # list of (v_onehot, theta_offset, scale_delta) per hop 1..K

        for h in range(1, K + 1):
            src_ids, src_pad = states_enc[h - 1]
            tgt_ids, tgt_pad = states_enc[h]

            v_onehot = _posterior_action(
                model, src_ids, src_pad, tgt_ids, tgt_pad, device
            )  # (1,V)

            a_out, theta_offset, scale_delta = _apply_one_hop(
                model, k_rolled, v_onehot, device, use_budget
            )

            hop_data.append((v_onehot, theta_offset, scale_delta))

            k_rolled = a_out
            if use_budget and scale_delta is not None:
                s_acc = s_acc + scale_delta  # accumulate log-scale

        k_K_roll = k_rolled  # rolled nouns after K hops
        s_K_roll = s_acc     # accumulated scale after K hops

        # Compute z of the rolled state (with event j present).
        z_donothing = _readout_z(model, k_K_roll, s_K_roll, device)

        # 3. Retract event j (1-based).
        j0 = retract_j - 1   # 0-based hop index
        v_j, theta_j, scale_j = hop_data[j0]

        k_retract = _inverse_one_hop(
            model, k_K_roll, v_j, theta_j, scale_j, device, use_budget
        )

        if use_budget and scale_j is not None:
            s_retract = s_K_roll - scale_j  # undo the accumulated log-scale of event j
        else:
            s_retract = None

        z_retract = _readout_z(model, k_retract, s_retract, device)

        # 4. Oracle-without-j target: replay chain omitting action j.
        snapshots = _oracle_replay_minus_j(gen_mod, chain_record, retract_j)
        # The oracle final state is after K-1 actions.
        # Render and encode it.
        final_entities = list(zip(types, snapshots[-1]))
        oracle_text = gen_mod.render_state(final_entities)
        oracle_ids, oracle_pad = encode_fn(oracle_text)
        k_oracle = _encode_fresh(model, oracle_ids, oracle_pad, device)
        z_oracle = _readout_z(
            model, k_oracle,
            torch.zeros(1, k_oracle.shape[1], device=device) if use_budget else None,
            device,
        )

        # 5. MODEL-REPLAY-WITHOUT-J (the model's HONEST counterfactual): re-roll the
        #    model's OWN latent trajectory on the j-deleted action history.  Encode s0,
        #    apply the teacher-forced posterior actions for `actions_minus_j` stepwise, with
        #    the H phase offsets RECOMPUTED FRESH on this counterfactual path (each
        #    `_apply_one_hop` re-runs the conditioner on its own input — so the offsets are
        #    keyed to the clean j-deleted path, NOT the contaminated with-j roll).  This is
        #    the model-side honest baseline: no path contamination, only dynamics/encode error.
        actions_minus_j = actions[:j0] + actions[j0 + 1:]
        k_replay = _encode_fresh(model, s0_ids, s0_pad, device)
        s_acc_r = torch.zeros(1, k_replay.shape[1], device=device) if use_budget else None

        # Build a tokenized state sequence for the minus-j chain using oracle snapshots.
        # snapshots[h] is the state after h actions from actions_minus_j.
        for h in range(len(actions_minus_j)):
            ent_states = list(zip(types, snapshots[h]))
            src_text = gen_mod.render_state(ent_states)
            ent_states_next = list(zip(types, snapshots[h + 1]))
            tgt_text = gen_mod.render_state(ent_states_next)
            src_ids_r, src_pad_r = encode_fn(src_text)
            tgt_ids_r, tgt_pad_r = encode_fn(tgt_text)

            v_r = _posterior_action(model, src_ids_r, src_pad_r,
                                    tgt_ids_r, tgt_pad_r, device)
            a_r, _, sd_r = _apply_one_hop(model, k_replay, v_r, device, use_budget)
            k_replay = a_r
            if use_budget and sd_r is not None:
                s_acc_r = s_acc_r + sd_r

        z_replay = _readout_z(model, k_replay, s_acc_r, device)

        # ---- Metrics ----
        # All reconstructions are scored by cosine to the SAME reference z_oracle (the fresh
        # encode of the oracle's true j-deleted final state).  The reencode_ceiling is the
        # cosine of that reference to ITSELF (= 1.0) — the literal upper bound a model-side
        # reconstruction could attain against this reference encoding.
        def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
            a_n = F.normalize(a.float(), dim=-1)
            b_n = F.normalize(b.float(), dim=-1)
            return float((a_n * b_n).sum())

        def mse(a: torch.Tensor, b: torch.Tensor) -> float:
            return float(F.mse_loss(a.float(), b.float()))

        retract_cos = cos_sim(z_retract, z_oracle)
        donothing_cos = cos_sim(z_donothing, z_oracle)
        replay_cos = cos_sim(z_replay, z_oracle)
        ceiling_cos = cos_sim(z_oracle, z_oracle)  # ≡ 1.0 — reference-identity ceiling

        per_chain.append({
            "donothing_cos": donothing_cos,
            "donothing_mse": mse(z_donothing, z_oracle),
            "retract_cos": retract_cos,
            "retract_mse": mse(z_retract, z_oracle),
            "model_replay_cos": replay_cos,
            "model_replay_mse": mse(z_replay, z_oracle),
            "ceiling_cos": ceiling_cos,
            "ceiling_mse": mse(z_oracle, z_oracle),
            # Derived per-chain (opposite remedies; see module docstring).
            "selection_drift": replay_cos - retract_cos,   # model / selection side
            "dynamics_gap": ceiling_cos - replay_cos,       # world / dynamics side
        })

    if not per_chain:
        raise RuntimeError("No chains processed.")

    n = len(per_chain)

    def _mean(key: str) -> float:
        return sum(c[key] for c in per_chain) / n

    donothing_cos = _mean("donothing_cos")
    retract_cos = _mean("retract_cos")
    model_replay_cos = _mean("model_replay_cos")
    ceiling_cos = _mean("ceiling_cos")

    agg = {
        "backend": "structured",
        "inverse_supported": True,
        "n_chains": n,
        "K": K,
        "retract_j": retract_j,
        # --- four-point bracket (expected monotone non-decreasing) ---
        "donothing_cos": donothing_cos,
        "donothing_mse": _mean("donothing_mse"),
        "retract_cos": retract_cos,
        "retract_mse": _mean("retract_mse"),
        "model_replay_cos": model_replay_cos,
        "model_replay_mse": _mean("model_replay_mse"),
        "ceiling_cos": ceiling_cos,
        "ceiling_mse": _mean("ceiling_mse"),
        # --- derived metrics (opposite remedies; see module docstring) ---
        "selection_drift": _mean("selection_drift"),  # model / selection side (H contamination)
        "dynamics_gap": _mean("dynamics_gap"),         # world / dynamics side (rollout fidelity)
        # --- bracket ordering checks ---
        "retract_beats_donothing": retract_cos > donothing_cos,
        "bracket_monotone": (
            donothing_cos <= retract_cos + 1e-6
            and retract_cos <= model_replay_cos + 1e-6
            and model_replay_cos <= ceiling_cos + 1e-6
        ),
        "per_chain": per_chain,
    }

    print(
        f"\nRetraction probe results (n={n}, K={K}, j={retract_j}):\n"
        f"  BRACKET (cos to oracle-without-j, expect non-decreasing):\n"
        f"    do_nothing            = {agg['donothing_cos']:.4f}  (rolled WITH event j — worst)\n"
        f"    algebraic_retraction  = {agg['retract_cos']:.4f}  (inverse of j on contaminated roll)\n"
        f"    model_replay          = {agg['model_replay_cos']:.4f}  (honest j-deleted re-roll)\n"
        f"    reencode_ceiling      = {agg['ceiling_cos']:.4f}  (reference-identity upper bound)\n"
        f"  DERIVED (opposite remedies):\n"
        f"    selection_drift       = {agg['selection_drift']:+.4f}  (model/selection: H path contamination)\n"
        f"    dynamics_gap          = {agg['dynamics_gap']:+.4f}  (world/dynamics: rollout fidelity)\n"
        f"  retract_beats_donothing = {agg['retract_beats_donothing']}   bracket_monotone = {agg['bracket_monotone']}"
    )

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nWrote {out_p}")

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Retraction probe for JEPA entity-world models (jepa_entity_campaign.md §4)."
    )
    p.add_argument("--ckpt", default=None,
                   help="Path to model checkpoint (model_latest.pt).  "
                        "Required unless --test_mode is set.")
    p.add_argument("--config", required=True,
                   help="Path to model config JSON (e.g. configs/jepa/jepa_ent_s0.json).")
    p.add_argument("--labeled", required=True,
                   help="Labeled split JSONL (e.g. data/entity_world/test_iid_labeled.jsonl).")
    p.add_argument("--K", type=int, default=4,
                   help="Number of events to roll before retracting.  Default: 4.")
    p.add_argument("--n_chains", type=int, default=256,
                   help="Number of chains to probe.  Default: 256.")
    p.add_argument("--retract_j", type=int, default=2,
                   help="1-based index of the event to retract (1 <= j <= K).  Default: 2.")
    p.add_argument("--out", default=None,
                   help="Output JSON path.  If not given, prints only to stdout.")
    p.add_argument("--test_mode", action="store_true",
                   help="Use a freshly initialized model (no checkpoint) for unit testing.")
    p.add_argument("--device", default=None,
                   help="Device: 'cpu', 'cuda', 'mps'.  Auto-detected if not given.")
    return p


def main():
    args = _build_arg_parser().parse_args()

    # Validate j.
    if not (1 <= args.retract_j <= args.K):
        sys.exit(f"ERROR: --retract_j must be in [1, K={args.K}], got {args.retract_j}")

    # Device.
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Config.
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(args.config)
    use_budget = getattr(cfg.model, "use_norm_budget", False)

    # Build model.
    from twm.jepa.model import build_jepa_model_v2
    import torch.nn as nn
    token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, token_emb)
    model = model.to(device)

    if not args.test_mode:
        if args.ckpt is None:
            sys.exit("ERROR: --ckpt is required unless --test_mode is set.")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("Test mode: using freshly initialized model (no checkpoint).")

    model.eval()

    # Tokenizer.
    encode_fn = _load_tokenizer(
        cfg.data.tokenizer, cfg.data.max_text_tokens,
        append_eos=getattr(cfg.data, "append_eos", True)
    )

    # Generator module.
    gen_mod = _load_gen()

    # Run probe.
    probe_retraction(
        model=model,
        labeled_path=args.labeled,
        K=args.K,
        n_chains=args.n_chains,
        retract_j=args.retract_j,
        device=device,
        encode_fn=encode_fn,
        gen_mod=gen_mod,
        use_budget=use_budget,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
