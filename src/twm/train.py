"""Training loop for Triple World Model."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .vocab import Vocabulary
from .dataset import TripleTransitionDataset, collate_fn
from .config import ModelConfig
from .model import TripleWorldModel
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

    # Clamp targets to valid range for split embeddings (different vocab sizes per role)
    targets_flat = targets_flat.clamp(0, V - 1)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Weight: 1.0 for real tokens, pad_weight for <pad>
    weights = torch.where(targets_flat == pad_id, pad_weight, 1.0)
    return (loss_per_token * weights).sum() / weights.sum()


def _fake_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate int8 quantization: quantize then dequantize."""
    scale = tensor.abs().max() / 127.0
    if scale == 0:
        return tensor
    quantized = (tensor / scale).round().clamp(-128, 127)
    return quantized * scale


def _apply_qat_noise(model: TripleWorldModel):
    """Replace weight data with fake-quantized version for QAT forward pass."""
    saved = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            saved[name] = param.data.clone()
            param.data = _fake_quantize(param.data)
    return saved


def _restore_weights(model: TripleWorldModel, saved: dict):
    """Restore original float32 weights after QAT forward pass."""
    for name, param in model.named_parameters():
        if name in saved:
            param.data = saved[name]


def save_int8_checkpoint(model: TripleWorldModel, path: Path):
    """Save an int8-quantized version of the model state dict."""
    state = {}
    for name, param in model.state_dict().items():
        if param.ndim >= 2:
            scale = param.abs().max() / 127.0
            if scale > 0:
                state[name] = (param / scale).round().clamp(-128, 127).to(torch.int8)
                state[name + "._scale"] = scale
            else:
                state[name] = param.to(torch.int8)
                state[name + "._scale"] = torch.tensor(0.0)
        else:
            state[name] = param
    torch.save(state, path)


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # --- output directory ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- vocab: scan ALL data files the model will see ---
    data_dir = Path(args.data_dir)
    data_files = []
    for f in data_dir.glob("*.jsonl"):
        data_files.append(f)
    if not data_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    vocab = Vocabulary.from_files(*data_files)
    vocab.save(out_dir / "vocab.json")
    print(f"Vocabulary: {len(vocab)} tokens")
    if args.split_embeddings:
        for role in ("entity", "attr", "value"):
            print(f"  {role}: {vocab.role_vocab_size(role)} tokens")

    # --- model config ---
    if args.config:
        config = ModelConfig.from_profile(args.config, vocab_size=len(vocab))
    else:
        config = ModelConfig(vocab_size=len(vocab))

    # Override with explicit CLI args (only if user passed them)
    for attr in ("d_model", "n_heads", "n_layers", "d_ff", "max_triples", "dropout"):
        cli_val = getattr(args, attr.replace("-", "_"), None)
        if cli_val is not None:
            setattr(config, attr, cli_val)

    if args.split_embeddings:
        config.n_entities = vocab.role_vocab_size("entity")
        config.n_attrs = vocab.role_vocab_size("attr")
        config.n_values = vocab.role_vocab_size("value")

    pretrained_embeds = None
    if args.pretrained_embeds:
        if args.split_embeddings:
            print("Warning: pretrained embeddings not supported with split embeddings, ignoring")
        else:
            pretrained_embeds = torch.load(args.pretrained_embeds, weights_only=True)
            print(f"Pretrained embeddings: {pretrained_embeds.shape}")

    model = TripleWorldModel(config, pretrained_embeds=pretrained_embeds).to(device)
    config.save(out_dir / "config.json")
    print(f"Parameters: {model.param_count():,}")
    if args.quantize_aware:
        print("Quantization-aware training: ENABLED")

    # --- data ---
    split_vocab = args.split_embeddings
    train_ds = TripleTransitionDataset(
        data_dir / "train.jsonl", vocab, max_triples=config.max_triples,
        split_vocab=split_vocab,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_datasets = {}
    for name in ["train", "test_comp", "test_seen", "test_context", "propara_dev", "openpi_dev"]:
        p = data_dir / f"{name}.jsonl"
        if p.exists():
            test_datasets[name] = TripleTransitionDataset(
                p, vocab, max_triples=config.max_triples,
                split_vocab=split_vocab,
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

            if args.quantize_aware:
                saved = _apply_qat_noise(model)
                logits = model(input_ids)
                loss = compute_loss(logits, target_ids, pad_id=vocab.pad_id, pad_weight=args.pad_weight)
                optimizer.zero_grad()
                loss.backward()
                _restore_weights(model, saved)
            else:
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
                m = compute_metrics(model, ds, vocab, device, split_vocab=split_vocab)
                for k, v in m.items():
                    row[f"{name}/{k}"] = v

            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            train_f1 = row.get("train/f1", 0.0)
            parts = [f"  epoch {epoch:4d} | loss {avg_loss:.4f}"]
            for label in ["train", "test_comp", "test_seen", "test_context", "propara_dev", "openpi_dev"]:
                key = f"{label}/f1"
                if key in row:
                    short = label.replace("test_", "").replace("propara_", "pp_")
                    parts.append(f"{short}_f1 {row[key]:.3f}")
            print(" | ".join(parts))

            # Save best model
            if train_f1 > best_train_acc:
                best_train_acc = train_f1
                torch.save(model.state_dict(), out_dir / "model_best.pt")

        elif epoch % 50 == 0:
            print(f"  epoch {epoch:4d} | loss {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    if args.quantize_aware:
        save_int8_checkpoint(model, out_dir / "model_int8.pt")
        print(f"Int8 checkpoint saved to {out_dir / 'model_int8.pt'}")
    log_f.close()

    # --- final assessment ---
    print("\n--- Final Results ---")
    model.train(False)
    for name, ds in test_datasets.items():
        m = compute_metrics(model, ds, vocab, device, split_vocab=split_vocab)
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
    # config profile
    parser.add_argument("--config", type=str, default=None,
                        help="Config profile: base, micro, atomic (overrides model defaults)")
    # model
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--split-embeddings", action="store_true",
                        help="Use separate entity/attr/value embedding tables")
    # training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pad-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--pretrained-embeds", type=str, default=None,
                        help="Path to pretrained embedding matrix (.pt)")
    parser.add_argument("--quantize-aware", action="store_true",
                        help="Enable quantization-aware training (simulated int8)")

    args = parser.parse_args()

    # Apply profile defaults for unset args
    if args.config:
        profile = ModelConfig.from_profile(args.config)
        for attr in ("d_model", "n_heads", "n_layers", "d_ff", "max_triples", "dropout"):
            if getattr(args, attr) is None:
                setattr(args, attr, getattr(profile, attr))
    else:
        # Legacy defaults matching original behavior
        defaults = {"d_model": 256, "n_heads": 4, "n_layers": 4, "d_ff": 1024,
                     "max_triples": 8, "dropout": 0.1}
        for attr, val in defaults.items():
            if getattr(args, attr) is None:
                setattr(args, attr, val)

    train(args)


if __name__ == "__main__":
    main()
