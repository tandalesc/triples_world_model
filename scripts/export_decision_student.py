#!/usr/bin/env python3
"""Export a trained decision student to the JSON bundle Hearthvale loads.

    uv run python scripts/export_decision_student.py results/decision_student/NAME OUT.json \
        --data DATA.jsonl [--fixture 64] [--kinds resident_routine]

The bundle holds the config, the vocabularies, every weight and a fixture:
rows from --data with the probabilities PyTorch gives them, so Hearthvale's JS
forward pass can be checked against it
(``node scripts/distill-check-student.js OUT.json``). ``kinds`` lists the
kinds the student was trained on (less --exclude-kinds); the game only asks
it about those and sends the rest to the hosted model.

Weights follow scripts/export_jepa_weights.py: 2-D matrices (embeddings,
projections) as per-row symmetric INT8 with an fp32 scale per row, 1-D
parameters as fp16; both base64-encoded. ``--float`` keeps float32 instead.
The fixture is computed with the weights as exported (dequantized), so the
JS check is exact either way; the KL the quantization costs is printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.decision.data import DecisionVocab, encode_rows, load_rows  # noqa: E402
from twm.decision.confidence import ErrorHead, features  # noqa: E402
from twm.decision.model import StudentConfig, build_student, kl_teacher_student  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from export_jepa_weights import _int8_quantize_per_channel  # noqa: E402


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def export_weight(t: torch.Tensor, as_float: bool) -> tuple[dict, torch.Tensor]:
    """-> (bundle entry, the tensor as the game will see it)."""
    t = t.detach().float().cpu()
    if as_float:
        return {"shape": list(t.shape), "f32": b64(t.numpy().astype("<f4"))}, t
    if t.dim() == 2:
        q, scale = _int8_quantize_per_channel(t)
        seen = torch.from_numpy(q.astype(np.float32) * scale[:, None])
        return {"shape": list(t.shape), "i8": b64(q), "scale": b64(scale.astype("<f4"))}, seen
    half = t.half()
    return {"shape": list(t.shape), "f16": b64(half.numpy().astype("<f2"))}, half.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--fixture", type=int, default=64)
    ap.add_argument("--kinds", default=None)
    ap.add_argument("--exclude-kinds", default=None, help="kinds to leave to the hosted model (not answered locally)")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--float", action="store_true", help="float32 weights instead of INT8/fp16")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.run) / "model.pt", map_location="cpu")
    cfg = StudentConfig(**ckpt["config"])
    vocab = DecisionVocab.from_json(ckpt["vocab"])
    model = build_student(vocab, cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    metrics_path = Path(args.run) / "metrics.json"
    trained_kinds = None
    if metrics_path.exists():
        trained_kinds = json.loads(metrics_path.read_text())["args"].get("kinds")
    kinds = (args.kinds or trained_kinds)
    kinds = kinds.split(",") if kinds else sorted({k.split(":")[0] for k in vocab.kind_options if ":" in k})
    if args.exclude_kinds:
        kinds = [k for k in kinds if k not in args.exclude_kinds.split(",")]

    weights, seen = {}, {}
    for name, t in ckpt["state_dict"].items():
        weights[name], seen[name] = export_weight(t, args.float)
    # The confidence head (train_error_head.py), if trained: float32, it is tiny.
    confidence = None
    if "err_head" in ckpt:
        for name, t in ckpt["err_head"].items():
            weights[f"err_head.{name}"] = export_weight(t, True)[0]
        rep = ckpt["err_head_report"]
        confidence = {"scale": rep["scale"], "eps": 1e-3, "thresholds": {k: v for k, v in rep.items() if k.startswith("threshold_")},
                      "quantiles": rep.get("quantiles"), "rank_corr": rep["rank_corr"]}

    rows = load_rows(args.data, kinds)
    random.Random(0).shuffle(rows)
    rows = rows[: args.fixture]
    b = encode_rows(rows, vocab).batch(torch.arange(len(rows)))
    with torch.no_grad():
        full = torch.softmax(model(b), dim=-1)
        model.load_state_dict(seen)
        logits = model(b)
        probs = torch.softmax(logits, dim=-1)
        cost = kl_teacher_student(logits, full.nan_to_num(0.0), b["opt_mask"]).mean().item()
    expected = None
    if confidence:
        head = ErrorHead(cfg.d_model)
        head.load_state_dict(ckpt["err_head"])
        with torch.no_grad():
            _, f = features(model, b)
            expected = (confidence["scale"] * ErrorHead.expected_kl(head(f))).tolist()
    fixture = [{"kind": r["kind"], "triples": r["triples"], "options": r["options"],
                "probabilities": [round(p, 7) for p in probs[i, : len(r["options"])].tolist()],
                **({"expected_kl": round(expected[i], 7)} if expected else {})} for i, r in enumerate(rows)]

    bundle = {
        "format": "hearthvale-decision-student",
        "version": args.version,
        "features": rows[0].get("features", 1) if rows else 1,
        "teacher": rows[0].get("model") if rows else None,
        "kinds": kinds,
        "config": cfg.to_dict(),
        "vocab": vocab.to_json(),
        "weights": weights,
        "confidence": confidence,
        "fixture": fixture,
    }
    Path(args.out).write_text(json.dumps(bundle, separators=(",", ":")))
    n = sum(p.numel() for p in model.parameters())
    print(f"{args.out}: {cfg.arch}, {n:,} parameters, {len(kinds)} kinds, {Path(args.out).stat().st_size / 1024:.0f} KB; "
          f"{'float32' if args.float else 'int8'} weights cost KL {cost:.2e} nats against float32 on the fixture")


if __name__ == "__main__":
    main()
