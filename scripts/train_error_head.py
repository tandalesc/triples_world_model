#!/usr/bin/env python3
"""Train and check a set model's confidence head (twm/decision/confidence.py).

    uv run python scripts/train_error_head.py results/decision_student/RUN \
        --data TRAIN.jsonl [TRAIN2.jsonl ...] --eval REALPLAY.jsonl

The head learns the student's KL to the teacher on the rows the student never
trained on: the held-out towns of the same split train_decision_student.py used
(same data files, same --test-fraction). It is then checked on --eval, a set
neither saw (Hearthvale's real-play questions from d1-run towns):
  - rank correlation between predicted and actual KL;
  - capture: flagging the top 1/5/10% by predicted KL, what share of all error
    and of the worst 5% of questions it catches (random flagging catches 1/5/10%);
  - calibration: mean predicted against mean actual KL in five bins.
The head and a calibration scale (fitted on the held-out rows, so its output
reads as nats) are saved into RUN/model.pt (``err_head``, ``err_head_report``),
so export_decision_student.py ships them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.decision.confidence import EPS, ErrorHead, features  # noqa: E402
from twm.decision.data import DecisionVocab, encode_rows, load_rows, split_by_seed  # noqa: E402
from twm.decision.model import StudentConfig, build_student, kl_teacher_student  # noqa: E402


def collect(model, rows, vocab, device, batch=4096):
    feats, kls = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 20000):
            data = encode_rows(rows[start:start + 20000], vocab)
            for b0 in range(0, len(data), batch):
                idx = torch.arange(b0, min(b0 + batch, len(data)))
                b = {k: v.to(device) for k, v in data.batch(idx).items()}
                logits, f = features(model, b)
                feats.append(f.cpu())
                kls.append(kl_teacher_student(logits, b["target"], b["opt_mask"]).cpu())
    return torch.cat(feats), torch.cat(kls)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    path = Path(args.run) / "model.pt"
    ckpt = torch.load(path, map_location="cpu")
    vocab = DecisionVocab.from_json(ckpt["vocab"])
    cfg = StudentConfig(**ckpt["config"])
    if cfg.arch != "set":
        raise SystemExit("the confidence head is for set models")
    model = build_student(vocab, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    rows = load_rows(args.data)
    is_aux = lambda r: str(r.get("model") or "").startswith("student:")  # noqa: E731
    _, held = split_by_seed(rows, args.test_fraction)
    held = [r for r in held if not is_aux(r)]
    fh, kh = collect(model, held, vocab, device)
    print(f"training the head on {len(held)} held-out-town rows (mean KL {kh.mean():.4f})")

    head = ErrorHead(cfg.d_model).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    fh_d, yh = fh.to(device), torch.log(kh + EPS).to(device)
    gen = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        idx = torch.randint(0, len(fh_d), (2048,), generator=gen).to(device)
        loss = torch.nn.functional.mse_loss(head(fh_d[idx]), yh[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    # Trained on log KL, the head predicts a typical error, not the mean: one
    # scale, fitted on the held-out rows, turns it into nats.
    with torch.no_grad():
        scale = float(kh.mean() / ErrorHead.expected_kl(head(fh_d)).cpu().mean())
    print(f"calibration scale {scale:.2f}")

    ev = load_rows([args.eval])
    fe, ke = collect(model, ev, vocab, device)
    with torch.no_grad():
        pred = scale * ErrorHead.expected_kl(head(fe.to(device))).cpu().numpy()
    actual = ke.numpy()
    order = np.argsort(-pred)
    total = actual.sum()
    worst = actual >= np.quantile(actual, 0.95)
    report = {"eval_rows": len(ev), "scale": scale, "rank_corr": rank_corr(pred, actual), "capture": {}, "calibration": []}
    print(f"\non {len(ev)} unseen rows ({args.eval}): rank correlation {report['rank_corr']:.3f}")
    for frac in (0.01, 0.05, 0.10):
        top = order[: int(frac * len(order))]
        share = actual[top].sum() / total
        hit = worst[top].sum() / worst.sum()
        report["capture"][f"{frac:.0%}"] = {"error_share": float(share), "worst5_caught": float(hit)}
        print(f"  flag top {frac:4.0%}: catches {share:5.1%} of all error, {hit:5.1%} of the worst 5% (random: {frac:.0%})")
    bins = np.array_split(np.argsort(pred), 5)
    print("  calibration (predicted vs actual mean KL, five bins by prediction):")
    for i, ix in enumerate(bins):
        report["calibration"].append([float(pred[ix].mean()), float(actual[ix].mean())])
        print(f"    bin {i + 1}: predicted {pred[ix].mean():.4f}  actual {actual[ix].mean():.4f}")
    for q in (0.9, 0.95, 0.99):
        report[f"threshold_p{int(q * 100)}"] = float(np.quantile(pred, q))
    # Expected KL at a ladder of quantiles, so the game can send the top x% of
    # its questions to the teacher for any x (decision-local.js).
    report["quantiles"] = [[q, float(np.quantile(pred, q))] for q in (0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)]

    ckpt["err_head"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    ckpt["err_head_report"] = report
    torch.save(ckpt, path)
    (Path(args.run) / "confidence.json").write_text(json.dumps(report, indent=2))
    print(f"saved the head into {path}")


if __name__ == "__main__":
    main()
