"""Benchmark a local LLM on the triple transition task.

Two evaluation modes:
  1. Exact match (strict vocabulary)
  2. Semantic match (embedding cosine similarity via Gemma)

Supports 0-shot and few-shot prompting.

Usage:
    python scripts/benchmark_llm.py --split test_comp
    python scripts/benchmark_llm.py --split test_comp --few-shot 5
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests


LLM_URL = "http://192.168.1.194:8001/v1/chat/completions"
LLM_MODEL = "ramblerun/Multimodal-AI"
EMBED_URL = "http://192.168.1.194:8002/v1/embeddings"
EMBED_MODEL = "ramblerun/TextEmbedding-AI"
PARALLEL = 4
TIMEOUT = 60


SYSTEM_PROMPT = """You predict the next state of a world described by triples.

Each triple is [entity, attribute, value]. Given the current state, predict what the state will be at the next timestep.

Rules:
- Output ONLY a JSON array of triples: [["entity", "attr", "value"], ...]
- Some triples may stay the same (persist), some may change, some may appear or disappear
- Think about physical causality: hot things cool, full things can empty, unsupported things fall, etc.
- Use short, single-word or underscore_separated values (e.g. "satisfied" not "not thirsty")
- Output valid JSON only, no explanation."""


def build_few_shot_messages(train_path: Path, n_shots: int) -> list[dict]:
    """Build few-shot examples from training data as conversation turns."""
    examples = []
    with open(train_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Pick diverse examples: different triple counts and domains
    random.seed(42)
    selected = random.sample(examples, min(n_shots, len(examples)))

    messages = []
    for ex in selected:
        messages.append({
            "role": "user",
            "content": f"Current state:\n{json.dumps(ex['state_t'])}\n\nPredict the next state (JSON array of triples):"
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["state_t+1"])
        })
    return messages


def predict_one(example: dict, idx: int, few_shot_msgs: list[dict]) -> tuple[int, list[list[str]], str]:
    """Send one example to the LLM and parse the response."""
    state_t = example["state_t"]
    user_msg = f"Current state:\n{json.dumps(state_t)}\n\nPredict the next state (JSON array of triples):"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(few_shot_msgs)
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = requests.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }, timeout=TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Parse JSON from response
        content_clean = content.strip()
        match = re.search(r'\[.*\]', content_clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            triples = []
            for item in parsed:
                if isinstance(item, list) and len(item) == 3:
                    triples.append([str(x).strip().lower() for x in item])
            return idx, triples, content
        return idx, [], content
    except Exception as e:
        return idx, [], f"ERROR: {e}"


# --- Embedding-based semantic matching ---

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Get embeddings from Gemma embedding model."""
    resp = requests.post(EMBED_URL, json={
        "model": EMBED_MODEL,
        "input": texts,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    return np.array([d["embedding"] for d in data])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def triple_to_text(triple: list[str]) -> str:
    """Convert triple to natural text for embedding."""
    return f"{triple[0]} {triple[1]} is {triple[2]}"


def semantic_set_match(
    pred_triples: list[list[str]],
    tgt_triples: list[list[str]],
    threshold: float = 0.85,
) -> dict:
    """Match predicted triples to target triples using embedding similarity.

    For each target triple, find the best-matching predicted triple.
    A match counts if cosine similarity exceeds threshold.
    """
    if not tgt_triples:
        return {"sem_precision": 1.0, "sem_recall": 1.0, "sem_f1": 1.0, "sem_exact": 1}
    if not pred_triples:
        return {"sem_precision": 0.0, "sem_recall": 0.0, "sem_f1": 0.0, "sem_exact": 0}

    # Get embeddings for all triples
    pred_texts = [triple_to_text(t) for t in pred_triples]
    tgt_texts = [triple_to_text(t) for t in tgt_triples]
    all_texts = pred_texts + tgt_texts

    embeddings = get_embeddings(all_texts)
    pred_embs = embeddings[:len(pred_texts)]
    tgt_embs = embeddings[len(pred_texts):]

    # Build similarity matrix
    sim_matrix = np.zeros((len(tgt_triples), len(pred_triples)))
    for i in range(len(tgt_triples)):
        for j in range(len(pred_triples)):
            sim_matrix[i, j] = cosine_sim(tgt_embs[i], pred_embs[j])

    # Greedy matching: for each target, find best unmatched prediction
    matched_pred = set()
    matched_tgt = 0
    for _ in range(min(len(tgt_triples), len(pred_triples))):
        best_sim = -1
        best_i, best_j = -1, -1
        for i in range(len(tgt_triples)):
            for j in range(len(pred_triples)):
                if j not in matched_pred and sim_matrix[i, j] > best_sim:
                    best_sim = sim_matrix[i, j]
                    best_i, best_j = i, j
        if best_sim >= threshold:
            matched_pred.add(best_j)
            matched_tgt += 1
            # Zero out this target row so it's not matched again
            sim_matrix[best_i, :] = -1
        else:
            break

    prec = matched_tgt / max(len(pred_triples), 1)
    rec = matched_tgt / max(len(tgt_triples), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    exact = int(matched_tgt == len(tgt_triples) == len(pred_triples))

    return {"sem_precision": prec, "sem_recall": rec, "sem_f1": f1, "sem_exact": exact}


# --- Exact matching (original) ---

def exact_set_match(pred: list[list[str]], tgt: list[list[str]]) -> dict:
    pred_set = set(tuple(t) for t in pred)
    tgt_set = set(tuple(t) for t in tgt)
    correct = len(pred_set & tgt_set)
    prec = correct / max(len(pred_set), 1)
    rec = correct / max(len(tgt_set), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"exact_f1": f1, "exact_match": int(pred_set == tgt_set)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test_comp")
    parser.add_argument("--data-dir", type=str, default="data/combined")
    parser.add_argument("--few-shot", type=int, default=5)
    parser.add_argument("--sem-threshold", type=float, default=0.85)
    parser.add_argument("--show-predictions", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / f"{args.split}.jsonl"
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Build few-shot messages
    few_shot_msgs = []
    if args.few_shot > 0:
        few_shot_msgs = build_few_shot_messages(data_dir / "train.jsonl", args.few_shot)

    print(f"Benchmarking {LLM_MODEL} on {args.split} ({len(examples)} examples)")
    print(f"  Few-shot: {args.few_shot}, Semantic threshold: {args.sem_threshold}")

    # Check connectivity
    for name, url in [("LLM", "http://192.168.1.194:8001/v1/models"),
                      ("Embeddings", "http://192.168.1.194:8002/v1/models")]:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            print(f"  {name}: reachable ✓")
        except requests.RequestException as e:
            print(f"  {name}: NOT reachable — {e}")
            if name == "LLM":
                return

    print()

    # Run LLM predictions in parallel
    predictions = [None] * len(examples)
    raw_outputs = [""] * len(examples)
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {
            pool.submit(predict_one, ex, i, few_shot_msgs): i
            for i, ex in enumerate(examples)
        }
        done = 0
        for future in as_completed(futures):
            idx, pred, raw = future.result()
            predictions[idx] = pred
            raw_outputs[idx] = raw
            done += 1
            if done % 10 == 0 or done == len(examples):
                print(f"  LLM predictions: {done}/{len(examples)}")

    # Compute metrics (exact + semantic)
    print("\nComputing semantic similarities...")
    results = []
    for i, (ex, pred) in enumerate(zip(examples, predictions)):
        tgt = ex["state_t+1"]
        inp = ex["state_t"]

        em = exact_set_match(pred, tgt)
        try:
            sm = semantic_set_match(pred, tgt, threshold=args.sem_threshold)
        except Exception as e:
            sm = {"sem_precision": 0, "sem_recall": 0, "sem_f1": 0, "sem_exact": 0}

        results.append({**em, **sm})

        if args.show_predictions:
            ok_exact = "✓" if em["exact_match"] else "✗"
            ok_sem = "✓" if sm["sem_exact"] else "✗"
            note = ex.get("note", "")
            print(f"  [{i+1}] exact={ok_exact} sem={ok_sem} | {note}")
            print(f"    IN:   {inp}")
            print(f"    TGT:  {tgt}")
            print(f"    PRED: {pred}")
            if not em["exact_match"]:
                pred_set = set(tuple(t) for t in pred)
                tgt_set = set(tuple(t) for t in tgt)
                miss = tgt_set - pred_set
                extra = pred_set - tgt_set
                if miss: print(f"    MISS: {[list(t) for t in miss]}")
                if extra: print(f"    XTRA: {[list(t) for t in extra]}")
            print()

    # Aggregate
    n = len(results)
    print(f"\n{'='*55}")
    print(f"  {args.split} ({n} examples), {args.few_shot}-shot")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'Exact':>8} {'Semantic':>10}")
    print(f"  {'-'*45}")
    for key_e, key_s, label in [
        ("exact_f1", "sem_f1", "F1"),
        ("exact_match", "sem_exact", "Exact Match"),
    ]:
        ve = sum(r[key_e] for r in results) / n
        vs = sum(r[key_s] for r in results) / n
        print(f"  {label:<25} {ve:>8.3f} {vs:>10.3f}")

    # Save raw results for later comparison
    out_path = Path("results") / f"llm_bench_{args.split}_{args.few_shot}shot.json"
    with open(out_path, "w") as f:
        json.dump({
            "split": args.split,
            "few_shot": args.few_shot,
            "model": LLM_MODEL,
            "n_examples": n,
            "exact_f1": sum(r["exact_f1"] for r in results) / n,
            "exact_match": sum(r["exact_match"] for r in results) / n,
            "sem_f1": sum(r["sem_f1"] for r in results) / n,
            "sem_exact": sum(r["sem_exact"] for r in results) / n,
            "per_example": results,
        }, f, indent=2)
    print(f"\nSaved details to {out_path}")


if __name__ == "__main__":
    main()
