#!/usr/bin/env python3
"""Score trained decision students on a labelled set, beyond mean KL.

    uv run python scripts/eval_decision_student.py DATA.jsonl --runs results/decision_student/cmp/set-p0 ... \
        [--questions QUESTIONS.jsonl] [--baseline TRAIN.jsonl] [--out report.json]

DATA.jsonl: labelled rows (Hearthvale's distill-export.js), scored whole (no split).
--questions: the collection's questions.jsonl, whose `n` (times asked) gives
    each kind's share of real traffic; without it every kind counts by its rows.
--baseline: training rows, for each kind's average distribution (the slot prior).

Per run it reports
  - KL averaged over distinct questions, and weighted by traffic: each kind by
    its share of questions asked, each question within a kind by how often it
    was asked;
  - a per-kind table (traffic share, KL, slot-prior KL, top-1);
  - signed bias per option, traffic-weighted: the student's mean probability
    minus the teacher's. A small consistent bias, applied at every sweep for
    150 years, can move a town more than larger random errors;
  - tail ratios for options the teacher picks rarely (mean below 10%): the
    student's mean over the teacher's, which mean KL hides;
  - for kinds whose options are free text, how often the student's top choice is
    simply the slot the prior favours, next to how often the teacher's is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.decision.data import DecisionVocab, encode_rows, load_rows  # noqa: E402
from twm.decision.model import StudentConfig, build_student  # noqa: E402

FREE_TEXT_KINDS = ("where_to_settle", "choose_site", "institution_draft", "event_response")


def predict(run: Path, rows: list[dict], device) -> list[np.ndarray]:
    ckpt = torch.load(run / "model.pt", map_location="cpu")
    vocab = DecisionVocab.from_json(ckpt["vocab"])
    model = build_student(vocab, StudentConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 8192):
            part = rows[start:start + 8192]
            data = encode_rows(part, vocab)
            b = {k: v.to(device) for k, v in data.__dict__.items() if isinstance(v, torch.Tensor)}
            p = torch.softmax(model(b), dim=-1).cpu().numpy()
            out.extend(p[i, : len(r["options"])] for i, r in enumerate(part))
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-12)))).sum())


def slot_priors(rows: list[dict]) -> dict[tuple, np.ndarray]:
    acc, n = {}, defaultdict(int)
    for r in rows:
        key = (r["kind"], tuple(r["options"]))
        acc[key] = acc.get(key, 0) + np.asarray(r["target"], dtype=np.float64)
        n[key] += 1
    return {k: v / n[k] for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))

    rows = load_rows([args.data])
    for r in rows:
        t = np.asarray(r["target"], dtype=np.float64)
        r["_p"] = t / t.sum()
    kinds = sorted({r["kind"] for r in rows})

    traffic = defaultdict(float)
    if args.questions:
        with open(args.questions) as f:
            for line in f:
                q = json.loads(line)
                traffic[q["kind"]] += q.get("n", 1)
    else:
        for r in rows:
            traffic[r["kind"]] += 1
    total = sum(traffic[k] for k in kinds)
    share = {k: traffic[k] / total for k in kinds}

    priors = slot_priors(load_rows([args.baseline])) if args.baseline else {}

    report = {"data": args.data, "rows": len(rows), "traffic_share": share, "runs": {}}
    print(f"{len(rows)} labelled rows, {len(kinds)} kinds; traffic from {'questions asked' if args.questions else 'rows'}")
    for run in args.runs:
        run = Path(run)
        preds = predict(run, rows, device)
        by_kind = defaultdict(list)
        for r, q in zip(rows, preds):
            by_kind[r["kind"]].append((r, q))
        per_kind, bias, tails, free = {}, [], [], {}
        for k, items in by_kind.items():
            w = np.array([max(1, r.get("n", 1)) for r, _ in items], dtype=np.float64)
            kls = np.array([kl(r["_p"], q) for r, q in items])
            top = np.array([np.argmax(q) == np.argmax(r["_p"]) for r, q in items])
            pri = []
            for r, _ in items:
                prior = priors.get((k, tuple(r["options"])))
                pri.append(kl(r["_p"], prior) if prior is not None else np.nan)
            prior = float(np.nanmean(pri)) if pri and not all(np.isnan(pri)) else None
            per_kind[k] = {"rows": len(items), "share": share[k], "kl": float(kls.mean()), "kl_traffic": float((kls * w).sum() / w.sum()),
                           "prior_kl": prior, "top1": float(top.mean())}
            # Signed bias and tails over rows offering the kind's full option set.
            full = max(len(r["options"]) for r, _ in items)
            sub = [(r, q, wi) for (r, q), wi in zip(items, w) if len(r["options"]) == full]
            if sub:
                ww = np.array([s[2] for s in sub])
                P = np.stack([s[0]["_p"] for s in sub])
                Q = np.stack([s[1] for s in sub])
                pt, qt = (P * ww[:, None]).sum(0) / ww.sum(), (Q * ww[:, None]).sum(0) / ww.sum()
                for j, o in enumerate(sub[0][0]["options"]):
                    bias.append({"kind": k, "option": o, "teacher": float(pt[j]), "student": float(qt[j]), "bias": float(qt[j] - pt[j]), "share": share[k]})
                    if pt[j] < 0.10 and len(sub) >= 50:
                        tails.append({"kind": k, "option": o, "teacher": float(pt[j]), "student": float(qt[j]), "ratio": float(qt[j] / max(pt[j], 1e-9))})
            if k in FREE_TEXT_KINDS:
                hits_s = hits_t = n = 0
                for r, q in items:
                    prior = priors.get((k, tuple(r["options"])))
                    if prior is None:
                        continue
                    n += 1
                    hits_s += int(np.argmax(q) == np.argmax(prior))
                    hits_t += int(np.argmax(r["_p"]) == np.argmax(prior))
                if n:
                    free[k] = {"rows": n, "student_picks_prior_slot": hits_s / n, "teacher_picks_prior_slot": hits_t / n}
        kl_uniform = float(np.mean([kl(r["_p"], q) for r, q in zip(rows, preds)]))
        kl_traffic = float(sum(share[k] * per_kind[k]["kl_traffic"] for k in per_kind))
        report["runs"][run.name] = {"kl": kl_uniform, "kl_traffic": kl_traffic, "per_kind": per_kind,
                                    "bias": sorted(bias, key=lambda b: -abs(b["bias"]) * (b["share"] ** 0.5)), "tails": tails, "free_text": free}
        print(f"\n== {run.name}: KL {kl_uniform:.4f} over distinct questions, {kl_traffic:.4f} weighted by traffic")
        print(f"{'kind':26s} {'share':>6s} {'KL':>7s} {'KLtraf':>7s} {'prior':>7s} {'top-1':>6s}")
        for k in sorted(per_kind, key=lambda k: -share[k]):
            s = per_kind[k]
            pk = f"{s['prior_kl']:.4f}" if s["prior_kl"] is not None else "   -  "
            print(f"{k:26s} {s['share']:6.1%} {s['kl']:7.4f} {s['kl_traffic']:7.4f} {pk:>7s} {s['top1']:6.1%}")
        print("largest signed biases (student − teacher, traffic-weighted mean probability):")
        for b in report["runs"][run.name]["bias"][:10]:
            print(f"  {b['kind']:24s} {b['option']:22s} teacher {b['teacher']:.3f} student {b['student']:.3f}  bias {b['bias']:+.3f}")
        if tails:
            print("rare options (teacher mean < 10%): student / teacher")
            for t in sorted(tails, key=lambda t: abs(np.log(max(t["ratio"], 1e-6))), reverse=True)[:10]:
                print(f"  {t['kind']:24s} {t['option']:22s} teacher {t['teacher']:.4f} student {t['student']:.4f}  ratio {t['ratio']:.2f}")
        for k, f in free.items():
            print(f"free text {k}: top choice is the prior's slot for the student {f['student_picks_prior_slot']:.0%}, for the teacher {f['teacher_picks_prior_slot']:.0%} ({f['rows']} rows)")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
