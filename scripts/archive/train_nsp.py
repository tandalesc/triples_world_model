#!/usr/bin/env python3
"""Train next-state prediction: predict state_{t+1} from context window.

The dynamics core receives concatenated bottleneck states as context
(replacing mode conditioning) and predicts the next state.

Usage:
  uv run python scripts/train_nsp.py configs/glucose_nsp.json
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
from twm.nsp_dataset import NSPDataset


def compute_nsp_loss(model, batch, device):
    """Compute next-state prediction loss.

    Compresses each context state, concatenates their bottlenecks,
    feeds through dynamics, and predicts the target state.
    """
    ctx_ids = batch["ctx_ids"].to(device)      # (B, K, T)
    ctx_pad = batch["ctx_pad"].to(device)       # (B, K, T)
    ctx_count = batch["ctx_count"].to(device)   # (B,)
    tgt_ids = batch["tgt_ids"].to(device)       # (B, T)
    tgt_pad = batch["tgt_pad"].to(device)       # (B, T)
    tgt_length = batch["tgt_length"].to(device)  # (B,)

    B, K, T = ctx_ids.shape
    token_emb = model.shared_token_emb
    N3 = model.config.max_triples * 3  # bottleneck positions per state

    # Compress each context state, skipping all-pad slots
    d = model.config.d_model

    # Identify which context slots are valid (data is left-aligned in context window)
    ctx_valid = torch.arange(K, device=device).unsqueeze(0) < ctx_count.unsqueeze(1)  # (B, K)
    valid_mask = ctx_valid.reshape(B * K)  # (B*K,)

    ctx_flat_ids = ctx_ids.reshape(B * K, T)
    ctx_flat_pad = ctx_pad.reshape(B * K, T)

    # Allocate flat buffer, compress only valid slots
    all_bn = torch.zeros(B * K, N3, d, device=device)
    if valid_mask.any():
        valid_bn = model.compress(ctx_flat_ids[valid_mask], ctx_flat_pad[valid_mask])
        if isinstance(valid_bn, tuple):
            valid_bn = valid_bn[0]
        # Scatter back into buffer
        indices = valid_mask.nonzero(as_tuple=True)[0]
        all_bn[indices] = valid_bn

    # Concatenate context bottlenecks: (B, K*N*3, d)
    ctx_bn_flat = all_bn.reshape(B, K * N3, d)

    # Run dynamics: context bottlenecks → predicted next state
    # The dynamics core processes all context positions via self-attention
    # We extract the last *valid* state's positions for the prediction
    dynamics_out = model.dynamics(ctx_bn_flat)  # (B, K*N*3, d)

    # Extract prediction from the last valid context state + residual
    # For each example, last valid slot is at index (ctx_count-1)
    pred_bn = torch.zeros(B, N3, d, device=device)
    for b in range(B):
        last_slot = ctx_count[b].item() - 1
        start = last_slot * N3
        end = start + N3
        pred_bn[b] = ctx_bn_flat[b, start:end] + dynamics_out[b, start:end]

    # Expander loss: reconstruct target text from predicted bottleneck
    pred_emb, _ = model.forward_expander(pred_bn, tgt_ids, tgt_pad)  # (B, T, d)

    non_pad = ~tgt_pad
    if not non_pad.any():
        return torch.tensor(0.0, device=device), {"loss": 0.0, "tok_acc": 0.0}

    target_clean = token_emb(tgt_ids)
    mse_loss = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])

    total_loss = mse_loss

    # Aux CE
    if model.text_expander.use_decode_proj:
        logits = model.text_expander.decode_proj_logits(pred_emb)
        ce = F.cross_entropy(logits[non_pad] / 0.1, tgt_ids[non_pad], ignore_index=0)
        total_loss = total_loss + 0.1 * ce

    # Length loss
    len_pred = model.forward_length(pred_bn)
    total_loss = total_loss + 0.1 * F.mse_loss(len_pred, tgt_length.float())

    # Metrics
    with torch.no_grad():
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
        nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
        tok_acc = (nn_ids == tgt_ids[non_pad]).float().mean().item()

    metrics = {
        "loss": total_loss.item(),
        "mse": mse_loss.item(),
        "tok_acc": tok_acc,
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

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Tokenizer
    max_text_tokens = cfg.get("max_text_tokens", 128)
    if cfg.get("tokenizer_pretrained"):
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"], max_length=max_text_tokens,
            save_path=out_dir / "tokenizer.json",
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            cfg["tokenizer_path"], max_length=max_text_tokens
        )

    # Dataset
    context_window = cfg.get("context_window", 2)
    train_ds = NSPDataset(
        cfg["train_data"], tokenizer,
        max_text_tokens=max_text_tokens, context_window=context_window,
    )
    test_ds = NSPDataset(
        cfg["test_data"], tokenizer,
        max_text_tokens=max_text_tokens, context_window=context_window,
    )

    batch_size = cfg.get("batch_size", 32)
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
    print(f"Train: {len(train_ds)} pairs, Test: {len(test_ds)} pairs")
    print(f"Context window: {context_window}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01
    )

    epochs = cfg.get("epochs", 200)
    log_every = cfg.get("log_every", 5)
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
            loss, metrics = compute_nsp_loss(model, batch, device)
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

        if epoch % eval_every == 0:
            model.eval()
            eval_loss = 0.0
            eval_tok = 0.0
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_nsp_loss(model, batch, device)
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
                tok_path = cfg.get("tokenizer_path") or str(out_dir / "tokenizer.json")
                model.save(out_dir, tokenizer_path=tok_path)
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")


if __name__ == "__main__":
    main()
