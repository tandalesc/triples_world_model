#!/usr/bin/env python3
"""Train a student of Hearthvale's decision model on exported triples.

    uv run python scripts/train_decision_student.py DATA.jsonl [DATA2.jsonl ...] \
        --arch set|twm|mlp|fm [--kinds resident_routine,crime_charity] [--epochs 30] [--device mps] \
        [--out results/decision_student/NAME]

Data: Hearthvale's ``scripts/distill-export.js`` output. Train and test are
split by town (seed), so the test towns are ones the student never saw.

Reports on the held-out towns, overall and per kind: mean KL(teacher || student)
in nats, the worst 1% of rows, and how often the student's top choice matches
the teacher's, next to a baseline that answers every question of a kind with
that kind's average distribution. It also reports the held-out rows whose
snapshot (the town and district triples) never appeared in training: new
combinations of known values, where structure should generalize best.

Rows labelled by another student (``model: "student:..."``, from
relabel_decision_student.py) are training-only: never scored, never in the
baseline, weighted by ``--aux-weight``.

``--init-from RUN`` warm-starts from a trained student instead of training
from scratch: its config and weights carry over, its vocabulary keeps every
id, new tokens are appended with embeddings started from its ``<unk>`` row,
and training then fine-tunes everything (usually with fewer epochs and a lower
learning rate). This is how a new version learns new values without a rebuild.

``--holdout-pairs FILE.json`` ([{"kind", "a": "scope.attr=value", "b": ...}])
is the sharper compositional test: every row of that kind holding both triples
is taken out of training (rows with either alone stay in) and scored
separately, from whichever town it came. Saves ``model.pt`` (weights, config, vocab)
and ``metrics.json`` in --out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.decision.data import DecisionVocab, encode_rows, load_rows, split_by_seed  # noqa: E402
from twm.decision.model import StudentConfig, build_student, kl_teacher_student  # noqa: E402


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def kind_mean_baseline(train_rows: list[dict], test_rows: list[dict]) -> list[float]:
    """KL of answering each question with the training mean for its kind and options."""
    sums: dict[tuple, list[float]] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for r in train_rows:
        key = (r["kind"], tuple(r["options"]))
        acc = sums.setdefault(key, [0.0] * len(r["options"]))
        for i, p in enumerate(r["target"]):
            acc[i] += p
        counts[key] += 1
    out = []
    for r in test_rows:
        key = (r["kind"], tuple(r["options"]))
        k = len(r["options"])
        q = [s / counts[key] for s in sums[key]] if key in sums else [1.0 / k] * k
        out.append(sum(p * (math.log(max(p, 1e-12)) - math.log(max(qi, 1e-6))) for p, qi in zip(r["target"], q) if p > 0))
    return out


@torch.no_grad()
def evaluate(model, data, batch_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    kls, agree = [], []
    for start in range(0, len(data), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(data)), device=data.scope.device)
        b = data.batch(idx)
        logits = model(b)
        kls.append(kl_teacher_student(logits, b["target"], b["opt_mask"]).cpu())
        agree.append((logits.argmax(-1) == b["target"].argmax(-1)).cpu())
    model.train()
    return torch.cat(kls), torch.cat(agree)


STATE_SCOPES = {"town", "district", "proposal", "body"}


def snapshot_key(row: dict) -> str:
    return json.dumps([t for t in row["triples"] if t[0] in STATE_SCOPES])


def triple_key(t) -> str:
    return f"{t[0]}.{t[1]}={t[2]}"


def holdout_split(rows: list[dict], pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """-> (rows without any held-out pair, rows with one), tagging the latter with its pair."""
    keep, held = [], []
    for r in rows:
        keys = None
        for p in pairs:
            if p["kind"] != r["kind"]:
                continue
            keys = keys or {triple_key(t) for t in r["triples"]}
            if p["a"] in keys and p["b"] in keys:
                held.append({**r, "pair": f"{p['kind']}: {p['a']} × {p['b']}"})
                break
        else:
            keep.append(r)
    return keep, held


def summarize(kinds: list[str], kl: torch.Tensor, agree: torch.Tensor, base: list[float], unseen: list[bool]) -> dict:
    base_t = torch.tensor(base)
    out = {"all": {"rows": len(kinds), "kl": kl.mean().item(), "kl_p99": kl.quantile(0.99).item() if len(kl) > 1 else kl.max().item(),
                   "top1": agree.float().mean().item(), "baseline_kl": base_t.mean().item()}}
    ix = torch.tensor(unseen)
    if ix.any():
        out["unseen snapshots"] = {"rows": int(ix.sum()), "kl": kl[ix].mean().item(), "top1": agree[ix].float().mean().item(),
                                   "baseline_kl": base_t[ix].mean().item()}
    by: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(kinds):
        by[k].append(i)
    for k, idx in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ix = torch.tensor(idx)
        out[k] = {"rows": len(idx), "kl": kl[ix].mean().item(), "top1": agree[ix].float().mean().item(), "baseline_kl": base_t[ix].mean().item()}
    return out


def print_summary(summary: dict) -> None:
    print(f"{'kind':26s} {'rows':>7s} {'KL':>8s} {'base KL':>8s} {'top-1':>6s}")
    for k, s in summary.items():
        print(f"{k:26s} {s['rows']:7d} {s['kl']:8.4f} {s['baseline_kl']:8.4f} {s['top1']:6.1%}")
    print(f"worst 1% of rows: KL above {summary['all']['kl_p99']:.4f}")


def warm_start(model, old_state: dict) -> None:
    """Copy a trained student's weights into a model whose embedding tables may have grown.

    Rows the old tables had keep their values; new rows start from the old
    ``<unk>`` row (index 1), which is what the old model used for those tokens.
    """
    state = model.state_dict()
    for name, old in old_state.items():
        if name not in state:
            continue
        new = state[name]
        if new.shape == old.shape:
            state[name] = old.clone()
        elif new.dim() == 2 and new.shape[1] == old.shape[1] and new.shape[0] > old.shape[0]:
            grown = new.clone()
            grown[: old.shape[0]] = old
            grown[old.shape[0]:] = old[1]
            state[name] = grown
        else:
            raise ValueError(f"cannot warm-start {name}: {tuple(old.shape)} -> {tuple(new.shape)}")
    model.load_state_dict(state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--arch", default="set", choices=["set", "twm", "mlp", "fm"])
    ap.add_argument("--kinds", default=None, help="comma-separated kinds (default: all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--state-layers", type=int, default=2)
    ap.add_argument("--question-layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=64)
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fm-order", type=int, default=2, choices=[2, 3])
    ap.add_argument("--min-value-count", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--no-aug", action="store_true", help="drop need-band variants of walker questions")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-weight", type=float, default=1.0, help="weight of rows labelled by another student")
    ap.add_argument("--holdout-pairs", default=None, help="JSON list of {kind, a, b} triple pairs to hold out")
    ap.add_argument("--init-from", default=None, help="warm-start from this run's model.pt (config, weights, vocabulary)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    kinds = args.kinds.split(",") if args.kinds else None
    rows = load_rows(args.data, kinds)
    if args.no_aug:
        rows = [r for r in rows if not r.get("aug")]
    pairs = json.loads(Path(args.holdout_pairs).read_text()) if args.holdout_pairs else []
    rows, held_rows = holdout_split(rows, pairs)
    train_rows, test_rows = split_by_seed(rows, args.test_fraction)
    is_aux = lambda r: str(r.get("model") or "").startswith("student:")  # noqa: E731
    test_rows = [r for r in test_rows if not is_aux(r)]
    held_rows = [r for r in held_rows if not is_aux(r)]
    n_aux = sum(map(is_aux, train_rows))
    if n_aux:
        print(f"{n_aux} training rows labelled by another student (weight {args.aux_weight})")
    print(f"{len(rows)} rows: {len(train_rows)} train, {len(test_rows)} test (held-out towns); device {device}")

    if args.init_from:
        base = torch.load(Path(args.init_from) / "model.pt", map_location="cpu")
        old_vocab = DecisionVocab.from_json(base["vocab"])
        vocab = DecisionVocab.extend(old_vocab, train_rows, args.min_value_count)
        cfg = StudentConfig(**base["config"])
        model = build_student(vocab, cfg)
        warm_start(model, base["state_dict"])
        grown = {name: len(getattr(vocab, name)) - len(getattr(old_vocab, name)) for name in ("attrs", "values", "kind_options", "options")}
        print(f"warm start from {args.init_from}: new tokens {grown}")
        model = model.to(device)
    else:
        vocab = DecisionVocab.build(train_rows, args.min_value_count)
        cfg = StudentConfig(arch=args.arch, d_model=args.d_model, n_heads=args.heads, state_layers=args.state_layers,
                            question_layers=args.question_layers, d_ff=args.d_ff, mlp_hidden=args.mlp_hidden,
                            dropout=args.dropout, fm_order=args.fm_order)
        model = build_student(vocab, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_params:,} parameters; vocab {len(vocab.attrs)} attrs, {len(vocab.values)} values, {len(vocab.kind_options)} options")

    train = encode_rows(train_rows, vocab)
    if n_aux:
        train.weight *= torch.tensor([args.aux_weight if is_aux(r) else 1.0 for r in train_rows])
    train = train.to(device)
    test = encode_rows(test_rows, vocab).to(device)
    teacher_rows = [r for r in train_rows if not is_aux(r)]
    base = kind_mean_baseline(teacher_rows, test_rows)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    best, best_state = float("inf"), None
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train), device=device)
        total = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            b = train.batch(idx)
            loss = (kl_teacher_student(model(b), b["target"], b["opt_mask"]) * b["weight"]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        kl, agree = evaluate(model, test)
        test_kl = kl.mean().item()
        if test_kl < best:
            best, best_state = test_kl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_kl": total / steps_per_epoch, "test_kl": test_kl, "top1": agree.float().mean().item(),
                        "seconds": time.time() - t0})
        print(f"epoch {epoch + 1:3d}  train KL {total / steps_per_epoch:.4f}  test KL {test_kl:.4f}  top-1 {agree.float().mean():.1%}  ({time.time() - t0:.0f} s)")

    model.load_state_dict(best_state)
    kl, agree = evaluate(model, test)
    seen = {snapshot_key(r) for r in teacher_rows}
    summary = summarize(test.kind, kl, agree, base, [snapshot_key(r) not in seen for r in test_rows])
    if held_rows:
        held = encode_rows(held_rows, vocab).to(device)
        hkl, hagree = evaluate(model, held)
        hbase = kind_mean_baseline(teacher_rows, held_rows)
        by_pair: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(held_rows):
            by_pair[r["pair"]].append(i)
        summary["held-out pairs"] = {}
        for name, idx in by_pair.items():
            ix = torch.tensor(idx)
            summary["held-out pairs"][name] = {"rows": len(idx), "kl": hkl[ix].mean().item(), "top1": hagree[ix].float().mean().item(),
                                               "baseline_kl": torch.tensor(hbase)[ix].mean().item()}
        summary["held-out pairs"]["all"] = {"rows": len(held_rows), "kl": hkl.mean().item(), "top1": hagree.float().mean().item(),
                                            "baseline_kl": sum(hbase) / len(hbase)}
    print()
    print_summary({k: v for k, v in summary.items() if k != "held-out pairs"})
    for name, s in summary.get("held-out pairs", {}).items():
        print(f"held-out pair {name}: {s['rows']} rows, KL {s['kl']:.4f} (baseline {s['baseline_kl']:.4f}), top-1 {s['top1']:.1%}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"config": cfg.to_dict(), "vocab": vocab.to_json(), "state_dict": best_state}, out / "model.pt")
        (out / "metrics.json").write_text(json.dumps({"params": n_params, "args": vars(args), "test": summary, "history": history,
                                                      "rows": {"train": len(train_rows), "test": len(test_rows), "held_out_pairs": len(held_rows)}},
                                                     indent=2))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
