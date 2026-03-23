#!/usr/bin/env python3
"""Train multi-turn chain dynamics.

Unrolls the dynamics core N-1 times for an N-step chain, computing
reconstruction loss at each intermediate step. No spectral loss, no VAE,
no staging — just compressor → unrolled dynamics → expander.

Usage:
  uv run python scripts/train_chain.py configs/glucose_chain.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset


def compute_chain_loss(model, batch, device):
    """Compute loss over an unrolled chain.

    For each step 1..chain_len-1:
      1. Run dynamics on current bottleneck
      2. Compute expander reconstruction loss against target text
      3. Accumulate

    Returns total loss and metrics dict.
    """
    input_ids = batch["input_ids"].to(device)       # (B, T)
    input_pad = batch["input_pad"].to(device)        # (B, T)
    chain_ids = batch["chain_ids"].to(device)        # (B, C, T)
    chain_pad = batch["chain_pad"].to(device)        # (B, C, T)
    chain_lengths = batch["chain_lengths"].to(device)  # (B, C)
    chain_len = batch["chain_len"].to(device)        # (B,)

    B, C, T = chain_ids.shape
    token_emb = model.shared_token_emb

    # Compress step 0
    bottleneck = model.compress(input_ids, input_pad)  # (B, N*3, d)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]  # drop VAE info if present

    # Mode 0 = identity-like "advance" — we use a single mode for chain dynamics
    # since the chain structure itself provides the transformation signal
    mode_ids = torch.zeros(B, dtype=torch.long, device=device)

    total_loss = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_tok_acc = 0.0
    n_steps = 0

    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        # Advance dynamics
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)

        # Only compute loss for examples that have this step
        active = chain_len > step  # (B,)
        if not active.any():
            break

        target_ids = chain_ids[active, step]    # (B', T)
        target_pad = chain_pad[active, step]    # (B', T)
        active_bn = bottleneck[active]          # (B', N*3, d)

        # Expander: predict clean embeddings from bottleneck
        pred_emb, _ = model.forward_expander(
            active_bn, target_ids, target_pad
        )  # (B', T, d)

        # MSE loss in embedding space
        non_pad = ~target_pad
        if not non_pad.any():
            continue

        target_clean = token_emb(target_ids)
        step_mse = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])
        total_loss = total_loss + step_mse

        # Aux CE loss via decode projection
        if model.text_expander.use_decode_proj:
            logits = model.text_expander.decode_proj_logits(pred_emb)
            ce = F.cross_entropy(
                logits[non_pad] / 0.1, target_ids[non_pad], ignore_index=0
            )
            total_loss = total_loss + 0.1 * ce

        # Length loss
        len_pred = model.forward_length(active_bn)  # (B',)
        target_len = chain_lengths[active, step].float()
        total_loss = total_loss + 0.1 * F.mse_loss(len_pred, target_len)

        # Metrics (no grad)
        with torch.no_grad():
            pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
            emb_norm = F.normalize(token_emb.weight, dim=-1)
            nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
            tok_acc = (nn_ids == target_ids[non_pad]).float().mean().item()
            total_tok_acc += tok_acc
            total_mse += step_mse.item()
            n_steps += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "loss": total_loss.item(),
        "mse": total_mse / max(n_steps, 1),
        "tok_acc": total_tok_acc / max(n_steps, 1),
        "steps": n_steps,
    }
    return total_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device(cfg.get("device", "cpu"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Tokenizer
    tokenizer = DomainBPETokenizer.load(
        cfg["tokenizer_path"], max_length=cfg.get("max_text_tokens", 64)
    )

    # Dataset
    max_text_tokens = cfg.get("max_text_tokens", 64)
    train_ds = ChainDataset(
        cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens
    )
    test_ds = ChainDataset(
        cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens
    )

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model
    model_config = ModelConfig(
        d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("dynamics_layers", 2),
        d_ff=cfg.get("d_ff", 256),
        max_triples=cfg.get("max_triples", 8),
        dropout=cfg.get("dropout", 0.1),
    )

    model = TextDynamicsModel(
        config=model_config,
        domain_tokenizer=tokenizer,
        text_compressor_layers=cfg.get("compressor_layers", 3),
        text_expander_layers=cfg.get("expander_layers", 3),
        dynamics_layers=cfg.get("dynamics_layers", 2),
        max_text_tokens=max_text_tokens,
        dropout=cfg.get("dropout", 0.1),
        alpha_min=cfg.get("alpha_min", 0.01),
        vae=False,
    )
    model.init_embeddings()
    model.to(device)

    print(f"Model: {model.param_count():,} params ({model.trainable_param_count():,} trainable)")
    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")
    print(f"Dynamics layers: {cfg.get('dynamics_layers', 2)}, d_model: {model_config.d_model}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01
    )

    epochs = cfg.get("epochs", 200)
    log_every = cfg.get("log_every", 10)
    eval_every = cfg.get("eval_every", 5)
    best_tok_acc = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tok = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            loss, metrics = compute_chain_loss(model, batch, device)
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        epoch_tok /= max(n_batches, 1)
        dt = time.time() - t0

        if epoch % log_every == 0 or epoch == 1:
            msg = f"ep {epoch:4d} | loss {epoch_loss:.4f} | tok {epoch_tok:.3f} | {dt:.1f}s"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Eval
        if epoch % eval_every == 0:
            model.eval()
            eval_loss = 0.0
            eval_tok = 0.0
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_chain_loss(model, batch, device)
                    eval_loss += metrics["loss"]
                    eval_tok += metrics["tok_acc"]
                    n_eval += 1
            eval_loss /= max(n_eval, 1)
            eval_tok /= max(n_eval, 1)

            msg = f"  eval | loss {eval_loss:.4f} | tok {eval_tok:.3f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if eval_tok > best_tok_acc:
                best_tok_acc = eval_tok
                model.save(out_dir, tokenizer_path=cfg["tokenizer_path"])
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")


if __name__ == "__main__":
    main()
