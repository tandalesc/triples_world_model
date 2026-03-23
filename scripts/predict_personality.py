#!/usr/bin/env python3
"""Generate Kaggle submission for Playground Series S5E7 (Personality).

Usage:
    uv run python scripts/predict_personality.py \
        --checkpoint results/personality \
        --test-csv data/playground-series-s5e7/test.csv \
        --output submission_personality.csv
"""

import argparse
import csv
from pathlib import Path

import torch

from twm.vocab import Vocabulary
from twm.config import ModelConfig
from twm.model import TripleWorldModel

from convert_personality import row_to_attrs, MODE_ADVANCE, ATTR_KEYS


def predict_personality(model, vocab, attrs, device, max_triples):
    from twm.dataset import _sort_triples, _pad_triples, _flatten_triples

    triples = [MODE_ADVANCE] + [["person", k, attrs[k]] for k in ATTR_KEYS if k in attrs]
    sorted_t = _sort_triples(triples)
    padded = _pad_triples(sorted_t, max_triples)
    ids = _flatten_triples(padded, vocab)
    input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    pred_ids = model.predict(input_tensor)[0].cpu().tolist()
    pred_triples = vocab.decode_triples(pred_ids)

    for t in pred_triples:
        if len(t) == 3 and t[1] == "personality":
            return t[2].capitalize()
    return "Extrovert"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default="data/playground-series-s5e7/test.csv")
    parser.add_argument("--output", type=str, default="submission_personality.csv")
    args = parser.parse_args()

    run_dir = Path(args.checkpoint)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(run_dir / "vocab.json")
    config = ModelConfig.load(run_dir / "config.json")
    model = TripleWorldModel(config).to(device)

    ckpt = run_dir / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.train(False)
    print(f"Loaded {ckpt} ({model.param_count():,} params) on {device}")

    with open(args.test_csv) as f:
        rows = list(csv.DictReader(f))

    results = []
    with torch.no_grad():
        for row in rows:
            attrs = row_to_attrs(row)
            personality = predict_personality(
                model, vocab, attrs, device, config.max_triples
            )
            results.append((row["id"], personality))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Personality"])
        for rid, personality in results:
            writer.writerow([rid, personality])

    counts = {}
    for _, p in results:
        counts[p] = counts.get(p, 0) + 1
    print(f"\nPredictions: {len(results)} total")
    for p, c in sorted(counts.items()):
        print(f"  {p}: {c} ({100*c/len(results):.1f}%)")
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
