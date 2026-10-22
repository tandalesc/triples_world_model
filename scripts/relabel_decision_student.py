#!/usr/bin/env python3
"""Label questions with a trained student, for a second round of distillation.

    uv run python scripts/relabel_decision_student.py results/decision_student/RUN UNLABELLED.jsonl OUT.jsonl

UNLABELLED.jsonl is Hearthvale's ``distill-export.js --unlabelled`` output:
questions the decision model never answered. The student (usually the set
model) writes its distribution as each row's ``target`` and marks the row
``model: "student:<run>"``; train_decision_student.py treats such rows as
training-only (never scored, never in the baseline). Rows from the held-out
towns (the same split_by_seed as training) are dropped, so nothing about the
test towns reaches the next student.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.decision.data import DecisionVocab, encode_rows, load_rows, split_by_seed  # noqa: E402
from twm.decision.model import StudentConfig, build_student  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(Path(args.run) / "model.pt", map_location="cpu")
    vocab = DecisionVocab.from_json(ckpt["vocab"])
    model = build_student(vocab, StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tag = f"student:{Path(args.run).name}"

    rows = load_rows([args.data])
    train_rows, test_rows = split_by_seed(rows, args.test_fraction)
    print(f"{len(rows)} rows; {len(test_rows)} from held-out towns dropped; labelling {len(train_rows)} with {tag} on {device}")
    t0 = time.time()
    with open(args.out, "w") as out, torch.no_grad():
        for start in range(0, len(train_rows), args.chunk):
            part = train_rows[start:start + args.chunk]
            for r in part:
                r["target"] = [1.0 / len(r["options"])] * len(r["options"])  # placeholder for encoding
            data = encode_rows(part, vocab)
            probs = []
            for b0 in range(0, len(part), args.batch):
                idx = torch.arange(b0, min(b0 + args.batch, len(part)))
                b = {k: v.to(device) for k, v in data.batch(idx).items()}
                probs.append(torch.softmax(model(b), dim=-1).cpu())
            probs = torch.cat(probs)
            for i, r in enumerate(part):
                r["target"] = [round(p, 6) for p in probs[i, : len(r["options"])].tolist()]
                r["model"] = tag
                out.write(json.dumps(r, separators=(",", ":")) + "\n")
            print(f"  {start + len(part)}/{len(train_rows)} ({time.time() - t0:.0f} s)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
