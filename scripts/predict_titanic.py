#!/usr/bin/env python3
"""Generate Kaggle submission for classic Titanic using trained TWM model.

Usage:
    uv run python scripts/predict_titanic.py \
        --checkpoint results/titanic \
        --test-csv data/titanic/test.csv \
        --output submission_titanic.csv
"""

import argparse
import csv
from pathlib import Path

import torch

from twm.vocab import Vocabulary
from twm.config import ModelConfig
from twm.model import TripleWorldModel

from convert_titanic import row_to_attrs, MODE_ADVANCE, ATTR_KEYS


def predict_survived(model, vocab, attrs, device, max_triples):
    """Run model on input triples and extract survived prediction."""
    from twm.dataset import _sort_triples, _pad_triples, _flatten_triples

    triples = [MODE_ADVANCE] + [["passenger", k, attrs[k]] for k in ATTR_KEYS if k in attrs]
    sorted_t = _sort_triples(triples)
    padded = _pad_triples(sorted_t, max_triples)
    ids = _flatten_triples(padded, vocab)
    input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    pred_ids = model.predict(input_tensor)[0].cpu().tolist()
    pred_triples = vocab.decode_triples(pred_ids)

    for t in pred_triples:
        if len(t) == 3 and t[1] == "survived":
            return 1 if t[2] == "yes" else 0
    return 0  # default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default="data/titanic/test.csv")
    parser.add_argument("--output", type=str, default="submission_titanic.csv")
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

    # Predict
    results = []
    skipped = 0
    with torch.no_grad():
        for row in rows:
            attrs = row_to_attrs(row)
            if attrs is None:
                # Fallback for missing data — use majority class (died)
                results.append((int(row["PassengerId"]), 0))
                skipped += 1
            else:
                survived = predict_survived(
                    model, vocab, attrs, device, config.max_triples
                )
                results.append((int(row["PassengerId"]), survived))

    # Write submission
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in results:
            writer.writerow([pid, survived])

    n_survived = sum(s for _, s in results)
    print(f"\nPredictions: {len(results)} total, {skipped} skipped (fallback=0)")
    print(f"Survived: {n_survived}/{len(results)} ({100*n_survived/len(results):.1f}%)")
    print(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
