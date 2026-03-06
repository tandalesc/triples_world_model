#!/usr/bin/env python3
"""Benchmark TWM model family: micro vs base vs MLP baseline.

Trains each variant and compares param count, memory, and accuracy.

Usage:
  uv run python scripts/benchmark_family.py --data-dir data/combined --epochs 300
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from twm.vocab import Vocabulary
from twm.config import ModelConfig
from twm.model import TripleWorldModel
from twm.mlp_baseline import MLPWorldModel
from twm.dataset import TripleTransitionDataset
from twm.metrics import compute_metrics, copy_baseline


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def train_variant(name: str, out_dir: str, data_dir: str, epochs: int, extra_args: list[str]):
    cmd = [
        sys.executable, "-m", "twm.train",
        "--data-dir", data_dir,
        "--out-dir", out_dir,
        "--epochs", str(epochs),
        "--log-every", "50",
    ] + extra_args
    print(f"\n{'='*60}")
    print(f"Training {name}: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def eval_variant(name: str, run_dir: str, data_dir: str, device: torch.device) -> dict:
    run_path = Path(run_dir)
    data_path = Path(data_dir)

    vocab = Vocabulary.load(run_path / "vocab.json")
    config = ModelConfig.load(run_path / "config.json")

    split_vocab = config.use_split_embeddings
    model = TripleWorldModel(config).to(device)

    ckpt = run_path / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_path / "model_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.train(False)

    results = {
        "name": name,
        "params": model.param_count(),
        "memory_bytes": memory_bytes(model),
        "memory_kb": memory_bytes(model) / 1024,
        "profile": config.profile,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "split_embeddings": config.use_split_embeddings,
    }

    for split in ["train", "test_comp", "test_seen", "test_context"]:
        p = data_path / f"{split}.jsonl"
        if not p.exists():
            continue
        ds = TripleTransitionDataset(p, vocab, max_triples=config.max_triples,
                                      split_vocab=split_vocab)
        m = compute_metrics(model, ds, vocab, device, split_vocab=split_vocab)
        for k, v in m.items():
            results[f"{split}/{k}"] = v

    # Accuracy per parameter (F1 per 1K params)
    comp_f1 = results.get("test_comp/f1", 0)
    results["efficiency"] = comp_f1 / (results["params"] / 1000) if results["params"] > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark TWM model family")
    parser.add_argument("--data-dir", default="data/combined")
    parser.add_argument("--results-dir", default="results/family_benchmark")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--pretrained-embeds", default=None)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just eval")
    args = parser.parse_args()

    device = get_device()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "twm-base": {
            "out": str(results_dir / "twm_base"),
            "args": ["--config", "base"],
        },
        "twm-base-split": {
            "out": str(results_dir / "twm_base_split"),
            "args": ["--config", "base", "--split-embeddings"],
        },
        "twm-micro": {
            "out": str(results_dir / "twm_micro"),
            "args": ["--config", "micro"],
        },
        "twm-micro-split": {
            "out": str(results_dir / "twm_micro_split"),
            "args": ["--config", "micro", "--split-embeddings"],
        },
        "twm-micro-qat": {
            "out": str(results_dir / "twm_micro_qat"),
            "args": ["--config", "micro", "--split-embeddings", "--quantize-aware"],
        },
    }

    if args.pretrained_embeds:
        variants["twm-base"]["args"] += ["--pretrained-embeds", args.pretrained_embeds]

    # Train all variants
    if not args.skip_train:
        for name, v in variants.items():
            train_variant(name, v["out"], args.data_dir, args.epochs, v["args"])

    # Evaluate all variants
    all_results = []
    for name, v in variants.items():
        run_dir = Path(v["out"])
        if not (run_dir / "vocab.json").exists():
            print(f"Skipping {name}: no checkpoint found")
            continue
        r = eval_variant(name, v["out"], args.data_dir, device)
        all_results.append(r)
        print(f"\n{name}:")
        print(f"  params: {r['params']:,} | memory: {r['memory_kb']:.1f} KB")
        for split in ["train", "test_comp", "test_seen", "test_context"]:
            f1_key = f"{split}/f1"
            if f1_key in r:
                print(f"  {split} F1: {r[f1_key]:.3f}")

    # Save results
    out_path = results_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model':<20} {'Params':>8} {'Mem KB':>8} {'Comp F1':>8} {'Seen F1':>8} {'Ctx F1':>8} {'Eff':>8}")
    print("-"*80)
    for r in all_results:
        print(f"{r['name']:<20} {r['params']:>8,} {r['memory_kb']:>8.1f} "
              f"{r.get('test_comp/f1', 0):>8.3f} {r.get('test_seen/f1', 0):>8.3f} "
              f"{r.get('test_context/f1', 0):>8.3f} {r.get('efficiency', 0):>8.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
