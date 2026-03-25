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


def compute_chain_loss(model, batch, device, cfg=None):
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
    mode_ids = batch["mode_id"].to(device)           # (B,)

    B, C, T = chain_ids.shape
    token_emb = model.shared_token_emb

    # Compress step 0
    bottleneck = model.compress(input_ids, input_pad)  # (B, N*3, d)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]  # drop VAE info if present

    total_loss = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_tok_acc = 0.0
    n_steps = 0

    # Per-mode accumulators
    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_tok_sum = {m: 0.0 for m in mode_names}
    mode_tok_n = {m: 0 for m in mode_names}

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
        active_modes = mode_ids[active]         # (B',)
        active_bn = bottleneck[active]          # (B', N*3, d)

        # Expander: predict clean embeddings from bottleneck
        pred_emb, _ = model.forward_expander(
            active_bn, target_ids, target_pad
        )  # (B', T, d)

        non_pad = ~target_pad
        if not non_pad.any():
            continue

        target_clean = token_emb(target_ids)
        step_mse = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])
        total_loss = total_loss + step_mse

        # CE loss via decode projection (cheap at small vocab, e.g. 259 bytes)
        if model.text_expander.use_decode_proj:
            logits = model.text_expander.decode_proj_logits(pred_emb)
            ce = F.cross_entropy(
                logits[non_pad] / 0.1, target_ids[non_pad], ignore_index=0
            )
            total_loss = total_loss + 0.1 * ce

        len_pred = model.forward_length(active_bn)
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

            for m in mode_names:
                mask = active_modes == m
                if mask.any():
                    mode_non_pad = ~target_pad[mask]
                    if mode_non_pad.any():
                        mode_pred = pred_emb[mask][mode_non_pad]
                        mode_tgt = target_ids[mask][mode_non_pad]
                        mode_pred_n = F.normalize(mode_pred, dim=-1)
                        mode_nn = torch.matmul(mode_pred_n, emb_norm.T).argmax(-1)
                        mode_tok_sum[m] += (mode_nn == mode_tgt).float().mean().item()
                        mode_tok_n[m] += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "loss": total_loss.item(),
        "mse": total_mse / max(n_steps, 1),
        "tok_acc": total_tok_acc / max(n_steps, 1),
        "steps": n_steps,
    }

    for m, name in mode_names.items():
        if mode_tok_n[m] > 0:
            metrics[f"tok_{name}"] = mode_tok_sum[m] / mode_tok_n[m]

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
    max_text_tokens = cfg.get("max_text_tokens", 64)
    if cfg.get("tokenizer") == "bytes":
        from twm.byte_tokenizer import ByteTokenizer
        tokenizer = ByteTokenizer(max_length=max_text_tokens)
    elif cfg.get("tokenizer_pretrained"):
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"], max_length=max_text_tokens,
            save_path=out_dir / "tokenizer.json",
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            cfg["tokenizer_path"], max_length=max_text_tokens
        )

    # Dataset
    max_chain_len = cfg.get("max_chain_len", 3)
    train_ds = ChainDataset(
        cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens,
        max_chain_len=max_chain_len,
    )
    test_ds = ChainDataset(
        cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens,
        max_chain_len=max_chain_len,
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
        bottleneck_dim=cfg.get("bottleneck_dim"),
    )
    model.init_embeddings()
    model.to(device)

    print(f"Model: {model.param_count():,} params ({model.trainable_param_count():,} trainable)")
    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")
    print(f"Dynamics layers: {cfg.get('dynamics_layers', 2)}, d_model: {model_config.d_model}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01
    )

    # Mixed precision
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = cfg.get("epochs", 200)
    log_every = cfg.get("log_every", 10)
    eval_every = cfg.get("eval_every", 5)
    best_tok_acc = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tok = 0.0
        epoch_mode_tok: dict[str, float] = {}
        epoch_mode_n: dict[str, int] = {}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, metrics = compute_chain_loss(model, batch, device, cfg)
            if loss.requires_grad:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode_tok[k] = epoch_mode_tok.get(k, 0.0) + metrics[k]
                    epoch_mode_n[k] = epoch_mode_n.get(k, 0) + 1
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        epoch_tok /= max(n_batches, 1)
        dt = time.time() - t0

        if epoch % log_every == 0 or epoch == 1:
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if epoch_mode_n.get(k, 0) > 0:
                    mode_str += f" | {k.replace('tok_', '')}: {epoch_mode_tok[k]/epoch_mode_n[k]:.3f}"
            msg = f"ep {epoch:4d} | loss {epoch_loss:.4f} | tok {epoch_tok:.3f}{mode_str} | {dt:.1f}s"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Eval
        if epoch % eval_every == 0:
            model.eval()
            eval_loss = 0.0
            eval_tok = 0.0
            eval_mode_tok: dict[str, float] = {}
            eval_mode_n: dict[str, int] = {}
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_chain_loss(model, batch, device, cfg)
                    eval_loss += metrics["loss"]
                    eval_tok += metrics["tok_acc"]
                    for k in ("tok_adv", "tok_qry", "tok_id"):
                        if k in metrics:
                            eval_mode_tok[k] = eval_mode_tok.get(k, 0.0) + metrics[k]
                            eval_mode_n[k] = eval_mode_n.get(k, 0) + 1
                    n_eval += 1
            eval_loss /= max(n_eval, 1)
            eval_tok /= max(n_eval, 1)

            eval_mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if eval_mode_n.get(k, 0) > 0:
                    eval_mode_str += f" | {k.replace('tok_', '')}: {eval_mode_tok[k]/eval_mode_n[k]:.3f}"
            msg = f"  eval | loss {eval_loss:.4f} | tok {eval_tok:.3f}{eval_mode_str}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            # BLEU: generate from bottleneck and compare to target (subset)
            bleu_samples = cfg.get("bleu_samples", 0)
            if bleu_samples > 0:
                from sacrebleu.metrics import BLEU
                bleu_metric = BLEU()
                refs = []
                hyps = []
                bleu_n = 0
                with torch.no_grad():
                    for batch in test_loader:
                        if bleu_n >= bleu_samples:
                            break
                        input_ids = batch["input_ids"].to(device)
                        input_pad = batch["input_pad"].to(device)
                        chain_ids = batch["chain_ids"].to(device)
                        chain_pad = batch["chain_pad"].to(device)
                        chain_len_b = batch["chain_len"].to(device)
                        mode_ids_b = batch["mode_id"].to(device)

                        bn = model.compress(input_ids, input_pad)
                        if isinstance(bn, tuple):
                            bn = bn[0]

                        # Unroll to final step, generate
                        max_c = chain_len_b.max().item()
                        for s in range(1, max_c):
                            bn = model.forward_dynamics(bn, mode_ids_b)

                        # Generate in small batches to avoid memory spikes
                        gen_bs = 8
                        for gi in range(0, len(input_ids), gen_bs):
                            if bleu_n >= bleu_samples:
                                break
                            ge = min(gi + gen_bs, len(input_ids))
                            gen_ids = model.generate(bn[gi:ge], n_steps=10)
                            for j in range(gen_ids.shape[0]):
                                if bleu_n >= bleu_samples:
                                    break
                                idx = gi + j
                                last = chain_len_b[idx].item() - 1
                                tgt_ids = chain_ids[idx, last]
                                tgt_pad = chain_pad[idx, last]
                                ref = tokenizer.decode(tgt_ids[~tgt_pad].tolist())
                                hyp = tokenizer.decode(gen_ids[j].tolist())
                                refs.append(ref)
                                hyps.append(hyp)
                                bleu_n += 1

                bleu_score = bleu_metric.corpus_score(hyps, [refs]).score
                bleu_str = f" | bleu: {bleu_score:.1f}"
                msg = msg + bleu_str
                # Reprint with BLEU
                print(f"  bleu: {bleu_score:.1f} ({bleu_n} samples)")
                log_file.write(f"  bleu: {bleu_score:.1f}\n")
                log_file.flush()

            # PCA snapshot
            if cfg.get("pca_snapshots", False):
                import numpy as np
                from sklearn.decomposition import PCA
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                snap_pts = []
                snap_modes = []
                snap_n = 0
                with torch.no_grad():
                    for batch in test_loader:
                        if snap_n >= 200:
                            break
                        ids_b = batch["input_ids"].to(device)
                        pad_b = batch["input_pad"].to(device)
                        mode_b = batch["mode_id"].to(device)
                        bn = model.compress(ids_b, pad_b)
                        if isinstance(bn, tuple):
                            bn = bn[0]
                        snap_pts.append(bn.mean(dim=1).cpu().numpy())
                        snap_modes.append(mode_b.cpu().numpy())
                        snap_n += len(ids_b)

                pts = np.concatenate(snap_pts)[:200]
                mds = np.concatenate(snap_modes)[:200]
                pca = PCA(n_components=2).fit(pts)
                pts_2d = pca.transform(pts)

                fig, ax = plt.subplots(figsize=(6, 5))
                colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
                for m, name in [(0, "adv"), (1, "qry"), (2, "id")]:
                    mask = mds == m
                    if mask.any():
                        ax.scatter(pts_2d[mask, 0], pts_2d[mask, 1], c=colors[m], s=8, alpha=0.5, label=name)
                ax.legend()
                ax.set_title(f"ep {epoch} | tok {eval_tok:.3f}")
                frames_dir = out_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                fig.savefig(frames_dir / f"pca_{epoch:04d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

            if eval_tok > best_tok_acc:
                best_tok_acc = eval_tok
                tok_path = cfg.get("tokenizer_path") or str(out_dir / "tokenizer.json")
                model.save(out_dir, tokenizer_path=tok_path)
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")


if __name__ == "__main__":
    main()
