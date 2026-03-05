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
import hashlib
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


def _pick_entity_anchor(token: str, vocab_tokens: set[str]) -> str | None:
    anchors = ["alice", "bob", "carol", "person_a", "person_b", "person_c", "agent", "object"]
    available = [a for a in anchors if a in vocab_tokens]
    if not available:
        return None
    # deterministic mapping for stable behavior across calls
    idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % len(available)
    return available[idx]


def _canonicalize_token(token: str, role: str, vocab_tokens: set[str]) -> str:
    t = token.strip().lower()
    if t in vocab_tokens:
        return t

    if role == "relation":
        for candidate in ("state", "location", "status"):
            if candidate in vocab_tokens:
                return candidate
        return t

    if role == "entity":
        anchor = _pick_entity_anchor(t, vocab_tokens)
        return anchor if anchor else t

    # value role
    synonym_map = {
        "on": "lit",
        "off": "dead",
        "hot": "warm",
        "cool": "cold",
        "tired": "resting",
        "ok": "resting",
    }
    if t in synonym_map and synonym_map[t] in vocab_tokens:
        return synonym_map[t]

    for candidate in ("resting", "neutral", "unknown"):
        if candidate in vocab_tokens:
            return candidate
    return t


def _canonicalize_state(state: list[list[str]], vocab_tokens: set[str]) -> tuple[list[list[str]], list[dict[str, str]]]:
    out: list[list[str]] = []
    changes: list[dict[str, str]] = []
    roles = ("entity", "relation", "value")

    for triple in state:
        fixed = []
        for role, tok in zip(roles, triple):
            new_tok = _canonicalize_token(tok, role, vocab_tokens)
            fixed.append(new_tok)
            if tok != new_tok:
                changes.append({"role": role, "from": tok, "to": new_tok})
        out.append(fixed)

    return out, changes


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
    p.add_argument(
        "--canonicalize-oov",
        action="store_true",
        help="Map OOV tokens to stable in-vocab anchors/synonyms before inference",
    )
    p.add_argument(
        "--show-canonicalization",
        action="store_true",
        help="Print token remapping decisions to stderr",
    )

    args = p.parse_args()

    _ensure_checkpoint(args)
    state = _parse_input(args.input)

    wm = WorldModel(args.checkpoint, device=args.device)

    if args.canonicalize_oov:
        vocab_tokens = set(wm.vocab.token2id.keys())
        state, changes = _canonicalize_state(state, vocab_tokens)
        if args.show_canonicalization and changes:
            print(json.dumps({"canonicalization": changes}, indent=2), file=sys.stderr)

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
