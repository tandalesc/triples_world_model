#!/usr/bin/env python3
"""Evaluate chain dynamics model on held-out and out-of-distribution examples.

Tests:
  1. GLUCOSE test set (in-distribution) — per-mode breakdown
  2. Hand-crafted causal chains (out-of-distribution)
  3. Longer chains (3+ dynamics steps) via chain stitching

Usage:
  uv run python scripts/eval_chain.py --run-dir results/glucose_chain_v2 \
    --test-data data/glucose/augmented_chain_general_test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset


def decode_from_bottleneck(model, bottleneck, device):
    """Generate text from a bottleneck via expander."""
    texts = model.generate(bottleneck, n_steps=10)
    return texts


def eval_chain(model, text_chain, mode_id, tokenizer, device, max_text_tokens=64):
    """Run a single chain through the model and compare predictions to targets.

    Returns dict with per-step predictions and metrics.
    """
    pad_id = tokenizer.pad_token_id

    # Encode step 0
    ids_0 = tokenizer.encode(text_chain[0], max_length=max_text_tokens)
    input_ids = torch.tensor([ids_0], dtype=torch.long, device=device)
    input_pad = (input_ids == pad_id)

    bottleneck = model.compress(input_ids, input_pad)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]

    mode = torch.tensor([mode_id], dtype=torch.long, device=device)
    token_emb = model.shared_token_emb
    emb_norm = F.normalize(token_emb.weight, dim=-1)

    results = {"input": text_chain[0], "targets": [], "predictions": [], "tok_accs": []}

    for step in range(1, len(text_chain)):
        bottleneck = model.forward_dynamics(bottleneck, mode)

        # Generate prediction
        pred_ids = model.generate(bottleneck, n_steps=10)  # (1, T)
        pred_text = tokenizer.decode(pred_ids[0].tolist())
        results["predictions"].append(pred_text)
        results["targets"].append(text_chain[step])

        # Compute tok_acc against target
        target_ids_list = tokenizer.encode(text_chain[step], max_length=max_text_tokens)
        target_ids = torch.tensor([target_ids_list], dtype=torch.long, device=device)
        target_pad = (target_ids == pad_id)

        # Run expander in eval mode to get pred_emb
        pred_emb, _ = model.forward_expander(bottleneck, target_ids, target_pad)
        non_pad = ~target_pad
        if non_pad.any():
            pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
            nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
            tok_acc = (nn_ids == target_ids[non_pad]).float().mean().item()
            results["tok_accs"].append(tok_acc)
        else:
            results["tok_accs"].append(0.0)

    return results


def make_ood_chains():
    """Hand-crafted out-of-distribution causal chains."""
    return [
        # Physical causation
        {
            "name": "ice melting",
            "chain": [
                "Someone_A left Something_A (that is ice) in the sun.",
                "Something_A melted.",
                "Someone_A had a puddle of water.",
            ],
            "mode": 0,
        },
        # Emotional chain
        {
            "name": "surprise party",
            "chain": [
                "Some People_A planned a surprise party for Someone_A.",
                "Someone_A walked into the room and saw the decorations.",
                "Someone_A feel(s) surprised and happy.",
            ],
            "mode": 0,
        },
        # Reverse causation (query mode)
        {
            "name": "broken window (query)",
            "chain": [
                "Something_A (that is a window) is broken.",
                "Someone_A threw Something_B at Something_A.",
                "Someone_A was angry.",
            ],
            "mode": 1,
        },
        # Multi-entity
        {
            "name": "trade",
            "chain": [
                "Someone_A possess(es) Something_A. Someone_B possess(es) Something_B.",
                "Someone_A and Someone_B traded.",
                "Someone_A possess(es) Something_B. Someone_B possess(es) Something_A.",
            ],
            "mode": 0,
        },
        # Simple identity
        {
            "name": "identity",
            "chain": [
                "Someone_A is at Somewhere_A. Someone_A feel(s) happy.",
                "Someone_A is at Somewhere_A. Someone_A feel(s) happy.",
            ],
            "mode": 2,
        },
        # Novel domain: cooking
        {
            "name": "cooking (novel)",
            "chain": [
                "Someone_A possess(es) flour and eggs. Someone_A want(s) to bake.",
                "Someone_A mixed the ingredients together.",
                "Someone_A possess(es) cake batter. Someone_A feel(s) satisfied.",
            ],
            "mode": 0,
        },
        # Longer chain: 4 steps (3 dynamics calls)
        {
            "name": "4-step journey",
            "chain": [
                "Someone_A is at Somewhere_A. Someone_A want(s) to go to Somewhere_B.",
                "Someone_A left Somewhere_A.",
                "Someone_A is traveling to Somewhere_B.",
                "Someone_A arrived at Somewhere_B. Someone_A feel(s) relieved.",
            ],
            "mode": 0,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TextDynamicsModel.load(args.run_dir, device=str(device))
    model.eval()
    tokenizer = model.tokenizer

    print(f"Model loaded from {args.run_dir}")
    print(f"Device: {device}")
    print()

    # 1. GLUCOSE test set evaluation
    if args.test_data:
        print("=" * 60)
        print("GLUCOSE Test Set (in-distribution)")
        print("=" * 60)

        ds = ChainDataset(args.test_data, tokenizer, max_text_tokens=model.max_text_tokens)
        loader = DataLoader(ds, batch_size=64)

        mode_names = {0: "advance", 1: "query", 2: "identity"}
        mode_correct = {m: 0 for m in mode_names}
        mode_total = {m: 0 for m in mode_names}

        token_emb = model.shared_token_emb
        emb_norm = F.normalize(token_emb.weight, dim=-1)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                input_pad = batch["input_pad"].to(device)
                chain_ids = batch["chain_ids"].to(device)
                chain_pad = batch["chain_pad"].to(device)
                mode_ids = batch["mode_id"].to(device)
                chain_len = batch["chain_len"].to(device)

                bottleneck = model.compress(input_ids, input_pad)
                if isinstance(bottleneck, tuple):
                    bottleneck = bottleneck[0]

                max_c = chain_len.max().item()
                for step in range(1, max_c):
                    bottleneck = model.forward_dynamics(bottleneck, mode_ids)
                    active = chain_len > step
                    if not active.any():
                        break

                    target_ids = chain_ids[active, step]
                    target_pad = chain_pad[active, step]
                    active_modes = mode_ids[active]

                    pred_emb, _ = model.forward_expander(
                        bottleneck[active], target_ids, target_pad
                    )
                    non_pad = ~target_pad

                    for m in mode_names:
                        mask = active_modes == m
                        if not mask.any():
                            continue
                        m_non_pad = non_pad[mask]
                        if not m_non_pad.any():
                            continue
                        m_pred = F.normalize(pred_emb[mask][m_non_pad], dim=-1)
                        m_tgt = target_ids[mask][m_non_pad]
                        m_nn = torch.matmul(m_pred, emb_norm.T).argmax(-1)
                        mode_correct[m] += (m_nn == m_tgt).sum().item()
                        mode_total[m] += m_tgt.numel()

        print(f"\n{'Mode':<12} {'Tok Acc':>8} {'Tokens':>10}")
        print("-" * 32)
        total_c = total_t = 0
        for m, name in mode_names.items():
            if mode_total[m] > 0:
                acc = mode_correct[m] / mode_total[m]
                print(f"{name:<12} {acc:>8.1%} {mode_total[m]:>10,}")
                total_c += mode_correct[m]
                total_t += mode_total[m]
        if total_t > 0:
            print("-" * 32)
            print(f"{'overall':<12} {total_c/total_t:>8.1%} {total_t:>10,}")
        print()

    # 2. Out-of-distribution chains
    print("=" * 60)
    print("Out-of-Distribution Chains")
    print("=" * 60)

    ood_chains = make_ood_chains()
    mode_labels = {0: "advance", 1: "query", 2: "identity"}

    with torch.no_grad():
        for ex in ood_chains:
            r = eval_chain(model, ex["chain"], ex["mode"], tokenizer, device,
                          max_text_tokens=model.max_text_tokens)
            avg_tok = sum(r["tok_accs"]) / len(r["tok_accs"]) if r["tok_accs"] else 0
            print(f"\n[{ex['name']}] mode={mode_labels[ex['mode']]} | avg_tok={avg_tok:.1%}")
            print(f"  Input:  {r['input']}")
            for i, (tgt, pred, acc) in enumerate(zip(r["targets"], r["predictions"], r["tok_accs"])):
                print(f"  Step {i+1}: tok={acc:.1%}")
                print(f"    Target: {tgt}")
                print(f"    Pred:   {pred}")

    print()


if __name__ == "__main__":
    main()
