#!/usr/bin/env python3
"""Generate Kaggle submission for Spaceship Titanic using trained TWM model.

Usage:
    uv run python scripts/predict_spaceship_titanic.py \
        --checkpoint results/spaceship_titanic \
        --test-csv data/spaceship-titanic/test.csv \
        --output submission_spaceship.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

from twm.vocab import Vocabulary
from twm.config import ModelConfig
from twm.model import TripleWorldModel

from convert_spaceship_titanic import row_to_attrs, MODE_ADVANCE, ATTR_ORDER


def predict_transported(model, vocab, attrs, device, max_triples):
    """Run model on input triples and extract transported prediction."""
    from twm.dataset import _sort_triples, _pad_triples, _flatten_triples

    triples = [MODE_ADVANCE] + [["passenger", k, attrs[k]] for k in ATTR_ORDER if k in attrs]
    sorted_t = _sort_triples(triples)
    padded = _pad_triples(sorted_t, max_triples)
    ids = _flatten_triples(padded, vocab)
    input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    pred_ids = model.predict(input_tensor)[0].cpu().tolist()
    pred_triples = vocab.decode_triples(pred_ids)

    for t in pred_triples:
        if len(t) == 3 and t[1] == "transported":
            return t[2] == "true"
    return False  # default if not found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default="data/spaceship-titanic/test.csv")
    parser.add_argument("--output", type=str, default="submission_spaceship.csv")
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

    # Read test CSV
    with open(args.test_csv) as f:
        rows = list(csv.DictReader(f))

    # Compute group sizes
    group_counts: Counter[str] = Counter()
    for row in rows:
        group_counts[row["PassengerId"].split("_")[0]] += 1

    # Predict
    results = []
    with torch.no_grad():
        for row in rows:
            attrs = row_to_attrs(row, group_counts)
            transported = predict_transported(
                model, vocab, attrs, device, config.max_triples
            )
            results.append((row["PassengerId"], transported))

    # Write submission
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Transported"])
        for pid, transported in results:
            writer.writerow([pid, str(transported)])

    n_true = sum(1 for _, t in results if t)
    print(f"\nPredictions: {len(results)} total")
    print(f"Transported: {n_true}/{len(results)} ({100*n_true/len(results):.1f}%)")
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
