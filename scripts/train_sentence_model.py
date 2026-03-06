#!/usr/bin/env python3
"""Train the sentence-level Triple World Model on ATOMIC or other free-text data.

Usage:
    uv run python scripts/train_sentence_model.py \
        --data-dir data/atomic \
        --out-dir results/atomic_sentence \
        --config atomic \
        --epochs 200

    # With a specific sentence-transformer model
    uv run python scripts/train_sentence_model.py \
        --data-dir data/atomic \
        --out-dir results/atomic_sentence \
        --st-model all-MiniLM-L6-v2 \
        --epochs 200
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from twm.config import ModelConfig, PROFILES
from twm.sentence_model import SentenceTripleWorldModel, cosine_embedding_loss
from twm.sentence_dataset import SentenceTripleDataset, collate_sentence_fn
from twm.sentence_encoder import PhraseBank


def make_encode_fn(model_name: str, device: torch.device, batch_size: int = 256):
    """Create an encoding function using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name, device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    def encode(phrases: list[str]) -> torch.Tensor:
        embeddings = st_model.encode(
            phrases,
            batch_size=batch_size,
            show_progress_bar=len(phrases) > 1000,
            convert_to_tensor=True,
            device=str(device),
        )
        return embeddings.cpu()

    return encode, st_dim


def build_eval(
    model: SentenceTripleWorldModel,
    dataset: SentenceTripleDataset,
    phrase_bank: PhraseBank,
    device: torch.device,
    max_examples: int = 200,
) -> dict[str, float]:
    """Compute cosine loss and nearest-neighbor triple accuracy."""
    model.train(False)

    total_loss = 0.0
    total_correct_triples = 0
    total_triples = 0
    n = 0

    with torch.no_grad():
        for i in range(min(len(dataset), max_examples)):
            ex = dataset[i]
            inp = ex["input_embeds"].unsqueeze(0).to(device)
            tgt = ex["target_embeds"].unsqueeze(0).to(device)
            pad = ex["pad_mask"].unsqueeze(0).to(device)

            pred = model.predict(inp, pad)
            loss = cosine_embedding_loss(pred, tgt, pad)
            total_loss += loss.item()

            # Decode predicted and target triples via phrase bank
            pred_triples = phrase_bank.decode_triples(pred[0].cpu())
            tgt_triples = phrase_bank.decode_triples(tgt[0].cpu())

            pred_set = set(tuple(t) for t in pred_triples)
            tgt_set = set(tuple(t) for t in tgt_triples)
            total_correct_triples += len(pred_set & tgt_set)
            total_triples += len(tgt_set)
            n += 1

    return {
        "loss": total_loss / max(n, 1),
        "triple_recall": total_correct_triples / max(total_triples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train sentence-level TWM")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="atomic",
                        choices=list(PROFILES.keys()))
    parser.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model name")
    parser.add_argument("--pretrained-dynamics", type=str, default=None,
                        help="Path to pretrained TWM checkpoint (.pt) to load dynamics from")
    parser.add_argument("--freeze-dynamics", action="store_true",
                        help="Freeze dynamics weights, only train projection layers")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Sentence-transformer: {args.st_model}")

    # Build encoding function
    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(args.st_model, device)
    print(f"  Embedding dim: {st_dim}")

    # Load config
    config = ModelConfig.from_profile(args.config)
    print(f"  Config: {args.config} (d_model={config.d_model}, {config.n_layers}L, {config.n_heads}H)")

    # Build datasets
    print("Building datasets (encoding all phrases)...")
    train_path = data_dir / "train.jsonl"
    train_ds = SentenceTripleDataset(train_path, encode_fn, max_triples=config.max_triples)
    print(f"  Train: {len(train_ds)} examples, st_dim={train_ds.st_dim}")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = SentenceTripleDataset(test_path, encode_fn, max_triples=config.max_triples)
        print(f"  Test: {len(test_ds)} examples")

    # Build phrase bank from training data
    print("Building phrase bank...")
    train_examples = []
    with open(train_path) as f:
        for line in f:
            train_examples.append(json.loads(line))
    phrase_bank = PhraseBank()
    phrase_bank.build(train_examples, encode_fn)
    for role in ("entity", "attr", "value"):
        print(f"  {role}: {len(phrase_bank.phrases[role])} phrases")
    phrase_bank.save(out_dir / "phrase_bank.pt")
    print("  Phrase bank saved.", flush=True)

    # Build model
    model = SentenceTripleWorldModel(config, st_dim).to(device)

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...", flush=True)
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)

    if args.freeze_dynamics:
        model.freeze_dynamics()
        print(f"Model: {model.param_count():,} total params, {model.trainable_param_count():,} trainable (dynamics frozen)", flush=True)
    else:
        print(f"Model: {model.param_count():,} params", flush=True)

    # Save config
    config.save(out_dir / "config.json")
    with open(out_dir / "st_config.json", "w") as f:
        json.dump({"st_model": args.st_model, "st_dim": st_dim}, f, indent=2)

    # Training — use differential LR if dynamics are pretrained but not frozen
    if args.pretrained_dynamics and not args.freeze_dynamics:
        proj_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        dyn_params = list(model.dynamics.parameters())
        optimizer = torch.optim.AdamW([
            {"params": proj_params, "lr": args.lr},
            {"params": dyn_params, "lr": args.lr * 0.1},
        ], weight_decay=0.01)
        print(f"  Differential LR: projections={args.lr}, dynamics={args.lr * 0.1}", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_sentence_fn, drop_last=True,
    )

    best_test_loss = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            inp = batch["input_embeds"].to(device)
            tgt = batch["target_embeds"].to(device)
            pad = batch["pad_mask"].to(device)

            pred = model(inp, pad)
            loss = cosine_embedding_loss(pred, tgt, pad)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == 1:
            train_metrics = build_eval(model, train_ds, phrase_bank, device)
            log = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | train_recall {train_metrics['triple_recall']:.3f}"

            if test_ds is not None:
                test_metrics = build_eval(model, test_ds, phrase_bank, device)
                log += f" | test_recall {test_metrics['triple_recall']:.3f}"

                if test_metrics["loss"] < best_test_loss:
                    best_test_loss = test_metrics["loss"]
                    torch.save(model.state_dict(), out_dir / "model_best.pt")
            else:
                test_metrics = None

            print(log, flush=True)
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_recall": train_metrics["triple_recall"],
                **({"test_loss": test_metrics["loss"], "test_recall": test_metrics["triple_recall"]} if test_metrics else {}),
            })

    # Save final
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    if best_test_loss < float("inf"):
        print(f"Best test loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    main()
