#!/usr/bin/env python3
"""Train multi-turn chain dynamics with factored consistency losses.

Separates dynamics learning from generation:
  - Dynamics loss: latent consistency (predicted vs target-encoded)
  - Decoder loss: diffusion reconstruction (only at endpoint)

The dynamics core trains in latent space via EMA target encoder.
The diffusion expander trains as a conditional generator.
Multi-step unrolling happens purely in latent space — no decode
at intermediate steps.

Usage:
  uv run python scripts/train_factored_chain.py configs/factored_v1.json
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, decay: float):
    """Exponential moving average update for target encoder."""
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


def compute_jepa_chain_loss(model, target_compressor, batch, device, cfg=None):
    """Compute JEPA-style loss over an unrolled chain.

    For each step 1..chain_len-1:
      1. Run dynamics on current bottleneck (latent space)
      2. Encode ground truth with target encoder
      3. Consistency loss: predicted latent vs target latent
      4. Only decode at final step for generation loss

    Returns total loss and metrics dict.
    """
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_lengths = batch["chain_lengths"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T = chain_ids.shape
    token_emb = model.shared_token_emb

    lambda_consist = cfg.get("consistency_weight", 1.0) if cfg else 1.0
    lambda_decode = cfg.get("decode_weight", 1.0) if cfg else 1.0
    decode_every = cfg.get("decode_every_step", False) if cfg else False

    # Compress step 0 with online encoder
    bottleneck = model.compress(input_ids, input_pad)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]

    total_loss = torch.tensor(0.0, device=device)
    consist_total = 0.0
    decode_total = 0.0
    total_tok_acc = 0.0
    n_steps = 0

    # Per-mode accumulators
    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_tok_sum = {m: 0.0 for m in mode_names}
    mode_tok_n = {m: 0 for m in mode_names}

    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        # Save pre-dynamics for multi-level conditioning
        pre_dynamics = bottleneck

        # Advance dynamics in latent space
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)

        active = chain_len > step
        if not active.any():
            break

        target_ids = chain_ids[active, step]
        target_pad = chain_pad[active, step]
        active_modes = mode_ids[active]
        active_bn = bottleneck[active]
        active_pre = pre_dynamics[active]

        # --- Consistency loss: predicted latent vs target-encoded latent ---
        with torch.no_grad():
            target_bn = target_compressor(target_ids, target_pad, model.config.max_triples)
            if isinstance(target_bn, tuple):
                target_bn = target_bn[0]

        # Cosine consistency loss (scale-invariant)
        pred_flat = active_bn.reshape(active_bn.shape[0], -1)
        tgt_flat = target_bn.reshape(target_bn.shape[0], -1)
        pred_norm = F.normalize(pred_flat, dim=-1)
        tgt_norm = F.normalize(tgt_flat, dim=-1)
        consist_loss = (1 - (pred_norm * tgt_norm).sum(-1)).mean()
        total_loss = total_loss + lambda_consist * consist_loss
        consist_total += consist_loss.item()

        # --- Decode loss: diffusion reconstruction ---
        is_final = (chain_len[active] == step + 1)
        should_decode = decode_every or is_final.any()

        if should_decode:
            if decode_every:
                decode_mask = torch.ones(active.sum(), dtype=torch.bool, device=device)
            else:
                decode_mask = is_final

            if decode_mask.any():
                d_bn = active_bn[decode_mask]
                d_pre = active_pre[decode_mask]
                d_tgt_ids = target_ids[decode_mask]
                d_tgt_pad = target_pad[decode_mask]

                pred_emb, _ = model.forward_expander(
                    d_bn, d_tgt_ids, d_tgt_pad, pre_dynamics=d_pre
                )
                non_pad = ~d_tgt_pad
                if non_pad.any():
                    target_clean = token_emb(d_tgt_ids)
                    decode_loss = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])
                    total_loss = total_loss + lambda_decode * decode_loss
                    decode_total += decode_loss.item()

                    # CE loss for decode_proj
                    if model.text_expander.use_decode_proj:
                        logits = model.text_expander.decode_proj_logits(pred_emb)
                        ce = F.cross_entropy(
                            logits[non_pad] / 0.1, d_tgt_ids[non_pad], ignore_index=0
                        )
                        total_loss = total_loss + 0.1 * ce

        # Length loss
        len_pred = model.forward_length(active_bn)
        target_len_val = chain_lengths[active, step].float()
        total_loss = total_loss + 0.1 * F.mse_loss(len_pred, target_len_val)

        # Metrics (no grad) — use online encoder's reconstruction quality
        with torch.no_grad():
            # Teacher-forced tok_acc via expander
            pred_emb_metric, _ = model.forward_expander(
                active_bn, target_ids, target_pad, pre_dynamics=active_pre
            )
            non_pad = ~target_pad
            if non_pad.any():
                pred_norm_m = F.normalize(pred_emb_metric[non_pad], dim=-1)
                emb_norm = F.normalize(token_emb.weight, dim=-1)
                nn_ids = torch.matmul(pred_norm_m, emb_norm.T).argmax(-1)
                tok_acc = (nn_ids == target_ids[non_pad]).float().mean().item()
                total_tok_acc += tok_acc

                for m in mode_names:
                    mask = active_modes == m
                    if mask.any():
                        mode_non_pad = ~target_pad[mask]
                        if mode_non_pad.any():
                            mode_pred = pred_emb_metric[mask][mode_non_pad]
                            mode_tgt = target_ids[mask][mode_non_pad]
                            mode_pred_n = F.normalize(mode_pred, dim=-1)
                            mode_nn = torch.matmul(mode_pred_n, emb_norm.T).argmax(-1)
                            mode_tok_sum[m] += (mode_nn == mode_tgt).float().mean().item()
                            mode_tok_n[m] += 1

            n_steps += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "loss": total_loss.item(),
        "consist": consist_total / max(n_steps, 1),
        "decode": decode_total / max(n_steps, 1),
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

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg.get("device", "cuda"))

    # Tokenizer
    if "tokenizer_pretrained" in cfg:
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"],
            max_length=cfg.get("max_text_tokens", 128),
        )
    elif cfg.get("tokenizer") == "bytes":
        tokenizer = DomainBPETokenizer.bytes_tokenizer(
            max_length=cfg.get("max_text_tokens", 128),
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            cfg["tokenizer"], max_length=cfg.get("max_text_tokens", 128),
        )

    max_text_tokens = cfg.get("max_text_tokens", 128)
    train_ds = ChainDataset(cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens)

    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")

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

    # EMA target encoder — deep copy of compressor, no gradients
    target_compressor = copy.deepcopy(model.text_compressor)
    target_compressor.requires_grad_(False)
    target_compressor.to(device)

    print(f"Model: {model.param_count():,} params ({model.trainable_param_count():,} trainable)")
    print(f"Dynamics layers: {cfg.get('dynamics_layers', 2)}, d_model: {cfg.get('d_model', 64)}")

    ema_decay = cfg.get("ema_decay", 0.996)
    print(f"EMA decay: {ema_decay}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = cfg.get("epochs", 200)
    log_every = cfg.get("log_every", 10)
    eval_every = cfg.get("eval_every", 5)
    best_tok_acc = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tok = 0.0
        epoch_consist = 0.0
        epoch_decode = 0.0
        epoch_mode_tok = {}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = compute_jepa_chain_loss(
                    model, target_compressor, batch, device, cfg
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update target encoder
            update_ema(model.text_compressor, target_compressor, ema_decay)

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            epoch_consist += metrics.get("consist", 0)
            epoch_decode += metrics.get("decode", 0)
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode_tok[k] = epoch_mode_tok.get(k, 0.0) + metrics[k]
                    epoch_mode_tok[k + "_n"] = epoch_mode_tok.get(k + "_n", 0) + 1
            n_batches += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches
        epoch_consist /= n_batches
        epoch_decode /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = epoch_mode_tok.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {epoch_mode_tok[k]/n:.3f}"

            msg = (f"ep {epoch:4d} | loss {epoch_loss:.4f} | consist {epoch_consist:.4f} "
                   f"| decode {epoch_decode:.4f} | tok {epoch_tok:.3f}{mode_str} "
                   f"| {time.time():.0f}")
            # Replace absolute time with elapsed
            msg = (f"ep {epoch:4d} | loss {epoch_loss:.4f} | consist {epoch_consist:.4f} "
                   f"| decode {epoch_decode:.4f} | tok {epoch_tok:.3f}{mode_str}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            model.eval()
            eval_loss = eval_tok = eval_consist = 0.0
            eval_mode_tok = {}
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_jepa_chain_loss(
                        model, target_compressor, batch, device, cfg
                    )
                    eval_loss += metrics["loss"]
                    eval_tok += metrics["tok_acc"]
                    eval_consist += metrics.get("consist", 0)
                    for k in ("tok_adv", "tok_qry", "tok_id"):
                        if k in metrics:
                            eval_mode_tok[k] = eval_mode_tok.get(k, 0.0) + metrics[k]
                            eval_mode_tok[k + "_n"] = eval_mode_tok.get(k + "_n", 0) + 1
                    n_eval += 1

            eval_loss /= n_eval
            eval_tok /= n_eval
            eval_consist /= n_eval
            eval_mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = eval_mode_tok.get(k + "_n", 0)
                if n > 0:
                    eval_mode_str += f" | {k.split('_')[1]}: {eval_mode_tok[k]/n:.3f}"

            msg = (f"  eval | loss {eval_loss:.4f} | consist {eval_consist:.4f} "
                   f"| tok {eval_tok:.3f}{eval_mode_str}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            # chrF generation eval
            bleu_samples = cfg.get("bleu_samples", 0)
            if bleu_samples > 0:
                from sacrebleu.metrics import CHRF
                chrf_metric = CHRF()
                refs, hyps = [], []
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

                        max_c = chain_len_b.max().item()
                        pre_bn = bn
                        for s in range(1, max_c):
                            pre_bn = bn
                            bn = model.forward_dynamics(bn, mode_ids_b)

                        gen_bs = 8
                        for gi in range(0, len(input_ids), gen_bs):
                            if bleu_n >= bleu_samples:
                                break
                            ge = min(gi + gen_bs, len(input_ids))
                            gen_ids = model.generate(
                                bn[gi:ge], n_steps=10,
                                pre_dynamics=pre_bn[gi:ge]
                            )
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

                chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
                print(f"  chrF: {chrf_score:.1f} ({bleu_n} samples)")
                log_file.write(f"  chrF: {chrf_score:.1f}\n")
                log_file.flush()

            if eval_tok > best_tok_acc:
                best_tok_acc = eval_tok
                tok_path = str(out_dir / "tokenizer.json")
                tokenizer.save(tok_path)
                model.save(out_dir, tokenizer_path=tok_path)
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")


if __name__ == "__main__":
    main()
