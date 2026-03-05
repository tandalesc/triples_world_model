"""Training loop for Triple World Model."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .vocab import Vocabulary
from .dataset import TripleTransitionDataset, collate_fn
from .model import ModelConfig, TripleWorldModel
from .metrics import compute_metrics, copy_baseline


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
    pad_weight: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy with reduced weight for <pad> target positions."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Weight: 1.0 for real tokens, pad_weight for <pad>
    weights = torch.where(targets_flat == pad_id, pad_weight, 1.0)
    return (loss_per_token * weights).sum() / weights.sum()


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # --- output directory ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- vocab ---
    data_dir = Path(args.data_dir)
    data_files = [data_dir / "train.jsonl"]
    for f in ["test_comp.jsonl", "test_seen.jsonl"]:
        p = data_dir / f
        if p.exists():
            data_files.append(p)
    vocab = Vocabulary.from_files(*data_files)
    vocab.save(out_dir / "vocab.json")
    print(f"Vocabulary: {len(vocab)} tokens")

    # --- model ---
    config = ModelConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_triples=args.max_triples,
        dropout=args.dropout,
    )
    config.save(out_dir / "config.json")
    model = TripleWorldModel(config).to(device)
    print(f"Parameters: {model.param_count():,}")

    # --- data ---
    train_ds = TripleTransitionDataset(
        data_dir / "train.jsonl", vocab, max_triples=args.max_triples
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_datasets = {}
    for name in ["train", "test_comp", "test_seen"]:
        p = data_dir / f"{name}.jsonl"
        if p.exists():
            test_datasets[name] = TripleTransitionDataset(
                p, vocab, max_triples=args.max_triples
            )

    # --- optimizer + scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # --- training loop ---
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")
    global_step = 0
    best_train_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)")
    print(f"  batch_size={args.batch_size}, lr={args.lr}, warmup={warmup_steps}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            logits = model(input_ids)
            loss = compute_loss(logits, target_ids, pad_id=vocab.pad_id, pad_weight=args.pad_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.shape[0]
            epoch_tokens += input_ids.shape[0]
            global_step += 1

        avg_loss = epoch_loss / epoch_tokens

        # --- periodic assessment ---
        if epoch % args.log_every == 0 or epoch == args.epochs:
            model.train(False)
            row = {"epoch": epoch, "step": global_step, "train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}

            for name, ds in test_datasets.items():
                m = compute_metrics(model, ds, vocab, device)
                for k, v in m.items():
                    row[f"{name}/{k}"] = v

            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            train_f1 = row.get("train/f1", 0.0)
            status = (
                f"  epoch {epoch:4d} | loss {avg_loss:.4f} | "
                f"train_f1 {row.get('train/f1', 0):.3f} | "
                f"comp_f1 {row.get('test_comp/f1', 0):.3f} | "
                f"seen_f1 {row.get('test_seen/f1', 0):.3f}"
            )
            print(status)

            # Save best model
            if train_f1 > best_train_acc:
                best_train_acc = train_f1
                torch.save(model.state_dict(), out_dir / "model_best.pt")

        elif epoch % 50 == 0:
            print(f"  epoch {epoch:4d} | loss {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    log_f.close()

    # --- final assessment ---
    print("\n--- Final Results ---")
    model.train(False)
    for name, ds in test_datasets.items():
        m = compute_metrics(model, ds, vocab, device)
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    if "train" in test_datasets:
        cb = copy_baseline(test_datasets["train"])
        print(f"\nCopy baseline (train): f1={cb['f1']:.4f}, exact_match={cb['exact_match']:.4f}")

    print(f"\nCheckpoints saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Triple World Model")
    # data
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="results/run")
    # model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--max-triples", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pad-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
