"""Train MLP baseline and compare with transformer.

Uses same data, same GloVe embeddings, same training setup.
The only difference: MLP has no attention (no cross-position interaction).
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from twm.vocab import Vocabulary
from twm.dataset import TripleTransitionDataset, collate_fn
from twm.model import ModelConfig, TripleWorldModel
from twm.mlp_baseline import MLPWorldModel
from twm.metrics import compute_metrics, compute_delta_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_loss(logits, targets, pad_id=0, pad_weight=0.1):
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    weights = torch.where(targets_flat == pad_id, pad_weight, 1.0)
    return (loss_per_token * weights).sum() / weights.sum()


def train_model(model, train_loader, vocab, device, epochs=500, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup = min(100, total_steps // 5)

    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(input_ids)
            loss = compute_loss(logits, target_ids, pad_id=vocab.pad_id)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

        if epoch % 100 == 0 or epoch == epochs:
            print(f"  epoch {epoch}, loss={loss.item():.4f}")

    return model


def main():
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path("data/combined")
    vocab = Vocabulary.load(data_dir / "vocab.json")
    pretrained = torch.load(data_dir / "pretrained_embeds.pt", weights_only=True)

    config = ModelConfig(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_triples=8,
        dropout=0.1,
    )

    train_ds = TripleTransitionDataset(data_dir / "train.jsonl", vocab, max_triples=8)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_sets = {}
    for name in ["test_comp", "test_seen", "propara_dev", "openpi_dev"]:
        p = data_dir / f"{name}.jsonl"
        if p.exists():
            test_sets[name] = TripleTransitionDataset(p, vocab, max_triples=8)

    # --- Train MLP baseline ---
    print("\n=== Training MLP Baseline ===")
    mlp = MLPWorldModel(config, pretrained_embeds=pretrained.clone()).to(device)
    print(f"MLP params: {mlp.param_count():,}")
    mlp = train_model(mlp, train_loader, vocab, device)

    # --- Load transformer (already trained) ---
    print("\n=== Loading Transformer ===")
    run_dir = Path("results/run_v3_expanded")
    if not run_dir.exists():
        run_dir = Path("results/run_v3_pretrained")
    t_config = ModelConfig.load(run_dir / "config.json")
    transformer = TripleWorldModel(t_config).to(device)
    ckpt = run_dir / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    transformer.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    transformer.train(False)
    print(f"Transformer params: {transformer.param_count():,}")

    # --- Evaluate both ---
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'MLP':>10} {'Transformer':>12}")
    print("=" * 70)

    for name, ds in test_sets.items():
        m_mlp = compute_metrics(mlp, ds, vocab, device)
        m_tf = compute_metrics(transformer, ds, vocab, device)
        d_mlp = compute_delta_metrics(mlp, ds, vocab, device)
        d_tf = compute_delta_metrics(transformer, ds, vocab, device)

        print(f"\n--- {name} ({len(ds)} examples) ---")
        for key in ["f1", "exact_match"]:
            print(f"  {key:<28} {m_mlp[key]:>10.3f} {m_tf[key]:>12.3f}")
        for key in ["delta_f1", "delta_exact"]:
            print(f"  {key:<28} {d_mlp[key]:>10.3f} {d_tf[key]:>12.3f}")

    # Train set delta metrics
    print(f"\n--- train ({len(train_ds)} examples) ---")
    m_mlp = compute_metrics(mlp, train_ds, vocab, device)
    m_tf = compute_metrics(transformer, train_ds, vocab, device)
    d_mlp = compute_delta_metrics(mlp, train_ds, vocab, device)
    d_tf = compute_delta_metrics(transformer, train_ds, vocab, device)
    for key in ["f1", "exact_match"]:
        print(f"  {key:<28} {m_mlp[key]:>10.3f} {m_tf[key]:>12.3f}")
    for key in ["delta_f1", "delta_exact"]:
        print(f"  {key:<28} {d_mlp[key]:>10.3f} {d_tf[key]:>12.3f}")


if __name__ == "__main__":
    main()
