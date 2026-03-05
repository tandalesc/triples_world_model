"""Evaluate TWM and MLP with semantic similarity matching.

Uses the same embedding Gemma metric as the LLM benchmark for fair comparison.
"""

import json
from pathlib import Path

import numpy as np
import requests
import torch

from twm.vocab import Vocabulary
from twm.model import ModelConfig, TripleWorldModel
from twm.mlp_baseline import MLPWorldModel
from twm.dataset import TripleTransitionDataset


EMBED_URL = "http://192.168.1.194:8002/v1/embeddings"
EMBED_MODEL = "ramblerun/TextEmbedding-AI"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_embeddings(texts: list[str]) -> np.ndarray:
    resp = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=30)
    resp.raise_for_status()
    return np.array([d["embedding"] for d in resp.json()["data"]])


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def triple_to_text(t):
    return f"{t[0]} {t[1]} is {t[2]}"


def semantic_set_match(pred, tgt, threshold=0.85):
    if not tgt:
        return {"sem_f1": 1.0, "sem_exact": 1}
    if not pred:
        return {"sem_f1": 0.0, "sem_exact": 0}

    all_texts = [triple_to_text(t) for t in pred] + [triple_to_text(t) for t in tgt]
    embs = get_embeddings(all_texts)
    pred_embs = embs[:len(pred)]
    tgt_embs = embs[len(pred):]

    sim = np.zeros((len(tgt), len(pred)))
    for i in range(len(tgt)):
        for j in range(len(pred)):
            sim[i, j] = cosine_sim(tgt_embs[i], pred_embs[j])

    matched_pred = set()
    matched = 0
    for _ in range(min(len(tgt), len(pred))):
        best_s, best_i, best_j = -1, -1, -1
        for i in range(len(tgt)):
            for j in range(len(pred)):
                if j not in matched_pred and sim[i, j] > best_s:
                    best_s, best_i, best_j = sim[i, j], i, j
        if best_s >= threshold:
            matched_pred.add(best_j)
            matched += 1
            sim[best_i, :] = -1
        else:
            break

    prec = matched / max(len(pred), 1)
    rec = matched / max(len(tgt), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"sem_f1": f1, "sem_exact": int(matched == len(tgt) == len(pred))}


def exact_set_match(pred, tgt):
    ps = set(tuple(t) for t in pred)
    ts = set(tuple(t) for t in tgt)
    c = len(ps & ts)
    p = c / max(len(ps), 1)
    r = c / max(len(ts), 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"exact_f1": f1, "exact_match": int(ps == ts)}


def evaluate_model(model, dataset, vocab, device, name):
    model.train(False)
    exact_f1s, sem_f1s, exact_ems, sem_ems = [], [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            pred_ids = model.predict(ex["input_ids"].unsqueeze(0).to(device))[0].cpu().tolist()
            pred = vocab.decode_triples(pred_ids)
            tgt = vocab.decode_triples(ex["target_ids"].tolist())

            em = exact_set_match(pred, tgt)
            sm = semantic_set_match(pred, tgt)
            exact_f1s.append(em["exact_f1"])
            exact_ems.append(em["exact_match"])
            sem_f1s.append(sm["sem_f1"])
            sem_ems.append(sm["sem_exact"])

    return {
        "exact_f1": np.mean(exact_f1s),
        "exact_match": np.mean(exact_ems),
        "sem_f1": np.mean(sem_f1s),
        "sem_exact": np.mean(sem_ems),
    }


def main():
    device = get_device()
    data_dir = Path("data/combined")
    vocab = Vocabulary.load(data_dir / "vocab.json")

    # Load TWM
    run_dir = Path("results/06_expanded_tests")
    t_config = ModelConfig.load(run_dir / "config.json")
    twm = TripleWorldModel(t_config).to(device)
    ckpt = run_dir / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    twm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    # Load LLM results
    llm_results = {}
    for split in ["test_comp", "test_seen", "openpi_dev"]:
        p = Path(f"results/comparisons/llm_bench_{split}_5shot.json")
        if p.exists():
            with open(p) as f:
                llm_results[split] = json.load(f)

    print(f"{'Split':<15} {'Model':<20} {'Exact F1':>10} {'Sem F1':>10} {'Exact EM':>10} {'Sem EM':>10}")
    print("=" * 75)

    for split in ["test_comp", "test_seen", "openpi_dev"]:
        ds_path = data_dir / f"{split}.jsonl"
        if not ds_path.exists():
            continue
        ds = TripleTransitionDataset(ds_path, vocab, max_triples=8)

        # TWM
        m = evaluate_model(twm, ds, vocab, device, "TWM")
        print(f"{split:<15} {'TWM':<20} {m['exact_f1']:>10.3f} {m['sem_f1']:>10.3f} {m['exact_match']:>10.3f} {m['sem_exact']:>10.3f}")

        # LLM
        if split in llm_results:
            lr = llm_results[split]
            print(f"{'':<15} {'Qwen3-VL 8B 5-shot':<20} {lr['exact_f1']:>10.3f} {lr['sem_f1']:>10.3f} {lr['exact_match']:>10.3f} {lr['sem_exact']:>10.3f}")

        print()

    # Save combined results
    all_results = {}
    for split in ["test_comp", "test_seen", "openpi_dev"]:
        ds_path = data_dir / f"{split}.jsonl"
        if not ds_path.exists():
            continue
        ds = TripleTransitionDataset(ds_path, vocab, max_triples=8)
        all_results[split] = {
            "twm": evaluate_model(twm, ds, vocab, device, "TWM"),
        }
        if split in llm_results:
            all_results[split]["llm"] = {
                "exact_f1": llm_results[split]["exact_f1"],
                "sem_f1": llm_results[split]["sem_f1"],
                "exact_match": llm_results[split]["exact_match"],
                "sem_exact": llm_results[split]["sem_exact"],
            }

    with open("results/comparisons/semantic_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved results/comparisons/semantic_comparison.json")


if __name__ == "__main__":
    main()
