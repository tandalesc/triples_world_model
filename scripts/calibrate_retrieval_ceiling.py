#!/usr/bin/env python3
"""Retrieval ceiling calibration (jepa_entity_campaign.md §5).

Establishes the best achievable next-state retrieval under two backends:

  --backend oracle     (entity-world; free, deterministic)
      Uses the oracle dynamics (ew.apply_action) to deterministically compute the
      true next state, then checks whether the gold candidate is uniquely identifiable
      in the hard pool.  Expected oracle_hard_mrr ≈ 1.0 — this CONFIRMS the hard pool
      is solvable (model gap, not pool defect).  Runs free, no API.

  --backend anthropic  (LLM judge; graceful skip when no API key)
      Uses ``claude-haiku-4-5`` to rank next-state candidates.  Gracefully skips if
      ANTHROPIC_API_KEY is not set.  Costs money; gate it.

Usage::

    # Oracle ceiling (free, runs locally on the committed data)
    uv run python scripts/calibrate_retrieval_ceiling.py \\
        --backend oracle \\
        --labeled data/entity_world/test_iid_labeled.jsonl \\
        --n 512 \\
        --out results/calib_oracle.json

    # Anthropic ceiling (requires ANTHROPIC_API_KEY)
    uv run python scripts/calibrate_retrieval_ceiling.py \\
        --backend anthropic \\
        --glucose data/glucose/chain_general_test.jsonl \\
        --entity data/entity_world/test_iid_labeled.jsonl \\
        --n 200 \\
        --out results/calib_haiku.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

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
# Hard pool builder
# ---------------------------------------------------------------------------

def _build_hard_pool(records: list[dict], n_queries: int, rng_seed: int = 42):
    """Build a flat list of (state_t, state_t1, pool_candidates, gold_idx) tuples.

    ``pool_candidates`` is a list of candidate next-state texts (shuffled); ``gold_idx``
    is the index of the true next-state (state_t1) in the shuffled candidate list.

    The hard pool is assembled as: the true next state + up to ``pool_size - 1``
    distractors drawn from *other chains* adjacent pairs in the dataset.  Pool size
    mirrors the diagnostic convention (at least 5 candidates total, up to 10).
    """
    pool_size = 8   # same-order as diagnostics hard-pool
    rng = random.Random(rng_seed)

    # Flatten all adjacent (state_t, state_t1) pairs.
    all_pairs: list[tuple[str, str]] = []
    for r in records:
        chain = r["chain"]
        for i in range(len(chain) - 1):
            all_pairs.append((chain[i], chain[i + 1]))

    rng.shuffle(all_pairs)
    queries = all_pairs[:n_queries]

    results = []
    for (s_t, s_t1) in queries:
        # Build a pool: the gold + random distractors from the dataset.
        distractors = [s for _, s in all_pairs if s != s_t1]
        n_dis = min(pool_size - 1, len(distractors))
        pool_dis = rng.sample(distractors, n_dis)
        pool = [s_t1] + pool_dis
        rng.shuffle(pool)
        gold_idx = pool.index(s_t1)
        results.append((s_t, s_t1, pool, gold_idx))
    return results


# ---------------------------------------------------------------------------
# Oracle backend
# ---------------------------------------------------------------------------

def run_oracle_backend(
    labeled_path: str,
    n: int,
    out_path: str | None,
    gen_mod,
) -> dict:
    """Oracle ceiling: exact-match retrieval via oracle replay.

    For each query pair (s_t, s_{t+1}) with its gold action, the oracle ranks candidates
    by whether ``ew.apply_action(type, state_t, action) == candidate``.  Because oracle
    dynamics is deterministic, the gold sits at rank 1 by construction (MRR = 1.0) when
    the pool contains the gold.  Any non-1.0 indicates a pool bug.

    Returns a result dict and writes JSON to ``out_path`` if given.
    """
    # Load labeled records.
    records = []
    with open(labeled_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Oracle backend: {len(records)} chains in {labeled_path}")

    # Build query set: adjacent pairs with their oracle action + type info.
    # We need (s_t_text, types, initial_states, action_idx) for each pair so we can
    # call apply_action without parsing the text.

    # Collect query tuples: (rendered_s_t, rendered_s_t1, type_names, state_t_dicts, action_label)
    query_tuples: list[tuple] = []
    for r in records:
        if "initial_states" not in r or "actions" not in r:
            continue
        types = r["types"]
        chain = r["chain"]
        actions = r["actions"]
        initial_states = r["initial_states"]

        # Replay the chain to get per-step entity states.
        snapshots = gen_mod.replay_chain(types, initial_states, actions)
        for h in range(len(actions)):
            s_t_text = chain[h]
            s_t1_text = chain[h + 1]
            action_label = actions[h]
            entity_states_at_t = snapshots[h]  # list of state dicts
            query_tuples.append((s_t_text, s_t1_text, types, entity_states_at_t, action_label))

    if not query_tuples:
        # Fallback: labeled data without initial_states field.
        print("WARNING: no 'initial_states' in labeled records.  "
              "Regenerate data with the updated generate_entity_world.py.  "
              "Oracle backend will use text-only pool ranking (approximate).")
        # Degrade to a pool-solvability check: can the gold be identified by exact text match?
        records_plain = []
        with open(labeled_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records_plain.append(json.loads(line))
        query_pool = _build_hard_pool(records_plain, min(n, sum(len(r["chain"]) - 1
                                                                for r in records_plain)))
        query_pool = query_pool[:n]
        n_solvable = sum(1 for _, s_t1, pool, gold_idx in query_pool if s_t1 in pool)
        result = {
            "backend": "oracle",
            "oracle_hard_mrr": n_solvable / len(query_pool) if query_pool else 0.0,
            "pool_solvable_frac": n_solvable / len(query_pool) if query_pool else 0.0,
            "n": len(query_pool),
            "mode": "text_only_fallback",
            "note": "initial_states missing; using text-based pool check only",
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        return result

    # Use up to n queries.
    rng = random.Random(42)
    rng.shuffle(query_tuples)
    query_tuples = query_tuples[:n]

    # Build the pool and check oracle rank.
    # Pool: gold (oracle-computed next state text) + distractors.
    all_texts = [s_t1 for _, s_t1, _, _, _ in query_tuples]

    n_solvable = 0
    rr_list: list[float] = []

    for (s_t_text, s_t1_text, types, entity_states_at_t, action_label) in query_tuples:
        # Oracle next state for this pair.
        action, actor_idx_s = action_label.rsplit("@", 1)
        actor_idx = int(actor_idx_s)
        type_name = types[actor_idx]
        state_t = entity_states_at_t[actor_idx]
        oracle_next_state = gen_mod.apply_action(type_name, state_t, action)

        # Reconstruct the FULL chain text for the next state.
        # The dataset stores chain[h+1] = "{action_sentence} {state_sentence}".
        # We must match the full format the generator emits, not just the state part.
        updated_entities = list(zip(types, [dict(s) for s in entity_states_at_t]))
        updated_entities[actor_idx] = (type_name, oracle_next_state)
        # Render the full next-step text: action sentence + state sentence.
        action_sentence = gen_mod.render_action(updated_entities, action, actor_idx)
        state_sentence = gen_mod.render_state(updated_entities)
        oracle_rendered = f"{action_sentence} {state_sentence}"

        # Build hard pool: gold (s_t1_text) + distractors.
        distractors = [t for t in all_texts if t != s_t1_text]
        n_dis = min(7, len(distractors))
        pool_dis = rng.sample(distractors, n_dis)
        pool = [s_t1_text] + pool_dis
        rng.shuffle(pool)

        gold_in_pool = s_t1_text in pool
        if not gold_in_pool:
            # Structural error: pool doesn't contain gold.  This shouldn't happen.
            rr_list.append(0.0)
            continue

        gold_pool_idx = pool.index(s_t1_text)

        # Oracle score: if oracle_rendered == s_t1_text (the gold), the oracle uniquely
        # identifies the correct next state and would rank it first.  RR = 1.0.
        # If they don't match (oracle render diverged), RR = 0.0.
        oracle_correct = (oracle_rendered == s_t1_text)
        if oracle_correct:
            rr = 1.0
            n_solvable += 1
        else:
            # Oracle didn't reproduce the gold text — render inconsistency.
            rr = 0.0

        rr_list.append(rr)

    mrr = sum(rr_list) / len(rr_list) if rr_list else 0.0
    solvable_frac = n_solvable / len(rr_list) if rr_list else 0.0

    print(f"Oracle backend results (n={len(rr_list)}):")
    print(f"  oracle_hard_mrr    = {mrr:.4f}  (expected ~1.0)")
    print(f"  pool_solvable_frac = {solvable_frac:.4f}  (fraction uniquely identifiable)")
    if mrr < 0.99:
        print("  WARNING: oracle_hard_mrr < 0.99 — check generator/pool consistency.")

    result = {
        "backend": "oracle",
        "oracle_hard_mrr": mrr,
        "pool_solvable_frac": solvable_frac,
        "n": len(rr_list),
        "mode": "oracle_replay",
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")

    return result


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def run_anthropic_backend(
    glucose_path: str | None,
    entity_path: str | None,
    n: int,
    out_path: str | None,
) -> dict:
    """LLM-judge ceiling using claude-haiku-4-5.

    Gracefully skips if ANTHROPIC_API_KEY is not set (exits 0, prints a message).
    The Haiku model does NOT support the ``effort`` param — never pass it.

    Returns a result dict.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        msg = (
            "ANTHROPIC_API_KEY not set — skipping anthropic backend "
            "(oracle backend is the free ceiling)"
        )
        print(msg)
        result = {
            "backend": "anthropic",
            "skipped": True,
            "reason": "ANTHROPIC_API_KEY not set",
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        return result

    try:
        import anthropic
    except ImportError as e:
        print(f"anthropic SDK not installed — skipping anthropic backend: {e}")
        result = {
            "backend": "anthropic",
            "skipped": True,
            "reason": f"anthropic package not installed: {e}",
        }
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        return result

    MODEL_ID = "claude-haiku-4-5"
    client = anthropic.Anthropic(max_retries=4)

    def rank_pool(state_t: str, candidates: list[str]) -> int:
        """Ask Haiku to pick the most likely next state.  Returns 0-based index or -1."""
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
        try:
            msg = client.messages.create(
                model=MODEL_ID,
                max_tokens=16,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Given the current state:\n{state_t}\n\n"
                        f"Which numbered option is the most likely NEXT state? "
                        f"Answer with only the number.\n{numbered}"
                    ),
                }],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "").strip()
            digits = "".join(ch for ch in text if ch.isdigit())
            return int(digits) if digits else -1
        except Exception as e:
            print(f"  API error (skipping query): {e}")
            return -1

    def _load_plain_records(path: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _evaluate_dataset(records: list[dict], n_queries: int, tag: str) -> dict:
        """Evaluate Haiku accuracy on a hard pool from this dataset."""
        queries = _build_hard_pool(records, n_queries)
        n_correct = 0
        n_attempted = 0
        for (s_t, s_t1, pool, gold_idx) in queries:
            pick = rank_pool(s_t, pool)
            if pick == -1:
                continue  # skipped
            n_attempted += 1
            if pick == gold_idx:
                n_correct += 1
        acc = n_correct / n_attempted if n_attempted > 0 else 0.0
        print(f"  {tag}: n={n_attempted} correct={n_correct} accuracy={acc:.3f}")
        return {"accuracy": acc, "n_correct": n_correct, "n_attempted": n_attempted}

    print(f"Anthropic backend (model={MODEL_ID}, n={n})")
    result: dict = {"backend": "anthropic", "model": MODEL_ID, "n": n}

    if glucose_path and Path(glucose_path).exists():
        glucose_records = _load_plain_records(glucose_path)
        print(f"  GLUCOSE: {len(glucose_records)} chains from {glucose_path}")
        g = _evaluate_dataset(glucose_records, n, "GLUCOSE")
        result["glucose_judge_acc"] = g["accuracy"]
        result["glucose_n_correct"] = g["n_correct"]
        result["glucose_n_attempted"] = g["n_attempted"]

    if entity_path and Path(entity_path).exists():
        entity_records = _load_plain_records(entity_path)
        print(f"  Entity:  {len(entity_records)} chains from {entity_path}")
        e = _evaluate_dataset(entity_records, n, "entity")
        result["entity_judge_acc"] = e["accuracy"]
        result["entity_n_correct"] = e["n_correct"]
        result["entity_n_attempted"] = e["n_attempted"]

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Retrieval ceiling calibration (jepa_entity_campaign.md §5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["oracle", "anthropic"], required=True,
                   help="Backend: 'oracle' (free, deterministic) or 'anthropic' (LLM judge).")
    p.add_argument("--labeled",
                   help="(oracle backend) Labeled split JSONL with initial_states.")
    p.add_argument("--glucose",
                   help="(anthropic backend) GLUCOSE chain JSONL for LLM judge eval.")
    p.add_argument("--entity",
                   help="(anthropic backend) Entity-world JSONL for LLM judge eval.")
    p.add_argument("--n", type=int, default=512,
                   help="Number of queries to evaluate.  Default: 512.")
    p.add_argument("--out", default=None,
                   help="Output JSON path.  If not given, prints to stdout only.")
    return p


def main():
    args = _build_arg_parser().parse_args()

    gen_mod = _load_gen()

    if args.backend == "oracle":
        if not args.labeled:
            sys.exit("ERROR: --labeled is required for --backend oracle")
        if not Path(args.labeled).exists():
            sys.exit(f"ERROR: labeled file not found: {args.labeled}")
        run_oracle_backend(
            labeled_path=args.labeled,
            n=args.n,
            out_path=args.out,
            gen_mod=gen_mod,
        )

    elif args.backend == "anthropic":
        run_anthropic_backend(
            glucose_path=args.glucose,
            entity_path=args.entity,
            n=args.n,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
