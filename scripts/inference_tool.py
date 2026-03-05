#!/usr/bin/env python3
"""Train-if-needed + inference CLI for Triple World Model.

Examples:
  # 1) Run inference using an existing checkpoint
  uv run python scripts/inference_tool.py \
    --checkpoint results/08_context_dependent \
    --input '[["glass","state","full"],["alice","state","thirsty"]]'

  # 2) If checkpoint weights are missing, train first then run inference
  uv run python scripts/inference_tool.py \
    --checkpoint results/inference_ready \
    --train-if-missing \
    --data-dir data/combined \
    --pretrained-embeds data/combined/pretrained_embeds.pt \
    --epochs 200 \
    --input '[["glass","state","full"],["alice","state","thirsty"]]'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from twm.serve import WorldModel


def _has_weights(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "model_best.pt").exists() or (checkpoint_dir / "model_final.pt").exists()


def _ensure_checkpoint(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.checkpoint)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if _has_weights(ckpt_dir):
        return

    if not args.train_if_missing:
        raise FileNotFoundError(
            f"No model weights found in {ckpt_dir}. Expected model_best.pt or model_final.pt. "
            "Pass --train-if-missing to pretrain first."
        )

    cmd = [
        sys.executable,
        "-m",
        "twm.train",
        "--data-dir",
        args.data_dir,
        "--out-dir",
        str(ckpt_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]

    if args.pretrained_embeds:
        pe = Path(args.pretrained_embeds)
        if pe.exists():
            cmd.extend(["--pretrained-embeds", args.pretrained_embeds])
        else:
            print(f"[inference_tool] pretrained embeddings not found at {pe}; training with random init")

    print("[inference_tool] No weights found; running pretraining:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_input(input_json: str) -> list[list[str]]:
    state = json.loads(input_json)
    if not isinstance(state, list):
        raise ValueError("Input must be a JSON list of triples")

    for t in state:
        if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t)):
            raise ValueError("Each triple must be [entity, relation, value] with strings")
    return state


def main() -> None:
    p = argparse.ArgumentParser(description="TWM inference tool (pretrain + serve)")
    p.add_argument("--checkpoint", required=True, help="Run directory containing config/vocab/(model_*.pt)")
    p.add_argument("--input", required=True, help="JSON list of input triples")
    p.add_argument("--steps", type=int, default=1, help="Number of forward simulation steps")
    p.add_argument("--device", default=None)

    p.add_argument("--train-if-missing", action="store_true", help="Pretrain if no weights are found")
    p.add_argument("--data-dir", default="data/combined")
    p.add_argument("--pretrained-embeds", default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()

    _ensure_checkpoint(args)
    state = _parse_input(args.input)

    wm = WorldModel(args.checkpoint, device=args.device)

    try:
        if args.steps == 1:
            pred = wm.advance(state)
            print(json.dumps(pred, indent=2))
        else:
            traj = wm.advance_n(state, args.steps)
            print(json.dumps(traj, indent=2))
    except KeyError as e:
        token = str(e).strip("'\"")
        print(
            json.dumps(
                {
                    "error": "unknown_token",
                    "token": token,
                    "hint": "Token not in training vocab. Retrain with data that includes this token or normalize entities/values.",
                },
                indent=2,
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
