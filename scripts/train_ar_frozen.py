#!/usr/bin/env python3
"""Train AR decoder on frozen factored_v5 latents.

Dynamics core + compressor frozen (oracle parity confirmed).
Only the AR decoder trains. Teacher forcing + free-running generation eval.

Usage:
  uv run python scripts/train_ar_frozen.py configs/ar_frozen_v1.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.text_dynamics_model import TextDynamicsModel
from twm.ar_decoder import ARDecoder
from twm.chain_dataset import ChainDataset


def compute_loss(frozen_model, decoder, batch, device):
    """Teacher-forced CE loss on frozen latents."""
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T = chain_ids.shape

    with torch.no_grad():
        bottleneck = frozen_model.compress(input_ids, input_pad)
        if isinstance(bottleneck, tuple):
            bottleneck = bottleneck[0]

    total_loss = torch.tensor(0.0, device=device)
    total_tok_acc = 0.0
    n_steps = 0

    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_tok_sum = {m: 0.0 for m in mode_names}
    mode_tok_n = {m: 0 for m in mode_names}

    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        with torch.no_grad():
            pre_dynamics = bottleneck
            bottleneck = frozen_model.forward_dynamics(bottleneck, mode_ids)

        active = chain_len > step
        if not active.any():
            break

        target_ids = chain_ids[active, step]
        target_pad = chain_pad[active, step]
        active_modes = mode_ids[active]
        active_bn = bottleneck[active]
        active_pre = pre_dynamics[active]

        logits = decoder(active_bn, target_ids, target_pad, pre_dynamics=active_pre)
        non_pad = ~target_pad
        if non_pad.any():
            ce_loss = F.cross_entropy(logits[non_pad], target_ids[non_pad], ignore_index=0)
            total_loss = total_loss + ce_loss

        with torch.no_grad():
            if non_pad.any():
                preds = logits[non_pad].argmax(-1)
                tok_acc = (preds == target_ids[non_pad]).float().mean().item()
                total_tok_acc += tok_acc

                for m in mode_names:
                    mask = active_modes == m
                    if mask.any():
                        m_non_pad = ~target_pad[mask]
                        if m_non_pad.any():
                            m_preds = logits[mask][m_non_pad].argmax(-1)
                            m_tgt = target_ids[mask][m_non_pad]
                            mode_tok_sum[m] += (m_preds == m_tgt).float().mean().item()
                            mode_tok_n[m] += 1

            n_steps += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "loss": total_loss.item(),
        "tok_acc": total_tok_acc / max(n_steps, 1),
    }
    for m, name in mode_names.items():
        if mode_tok_n[m] > 0:
            metrics[f"tok_{name}"] = mode_tok_sum[m] / mode_tok_n[m]

    return total_loss, metrics


def generation_eval(frozen_model, decoder, test_loader, device, tok, n_samples=200):
    """Free-running generation: greedy AR decode, compute tok_acc + chrF."""
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()

    refs, hyps = [], []
    gen_correct = gen_total = 0
    mode_gen_correct = {0: 0, 1: 0, 2: 0}
    mode_gen_total = {0: 0, 1: 0, 2: 0}
    n = 0

    with torch.no_grad():
        for batch in test_loader:
            if n >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            input_pad = batch["input_pad"].to(device)
            chain_ids = batch["chain_ids"].to(device)
            chain_pad = batch["chain_pad"].to(device)
            chain_len_b = batch["chain_len"].to(device)
            mode_ids_b = batch["mode_id"].to(device)

            bn = frozen_model.compress(input_ids, input_pad)
            if isinstance(bn, tuple):
                bn = bn[0]
            pre_bn = bn
            max_c = chain_len_b.max().item()
            for s in range(1, max_c):
                pre_bn = bn
                bn = frozen_model.forward_dynamics(bn, mode_ids_b)

            gen_bs = 8
            for gi in range(0, len(input_ids), gen_bs):
                if n >= n_samples:
                    break
                ge = min(gi + gen_bs, len(input_ids))
                gen_ids = decoder.generate(
                    bn[gi:ge], pre_dynamics=pre_bn[gi:ge],
                    max_tokens=frozen_model.max_text_tokens,
                )

                for j in range(gen_ids.shape[0]):
                    if n >= n_samples:
                        break
                    idx = gi + j
                    last = chain_len_b[idx].item() - 1
                    tgt_ids = chain_ids[idx, last]
                    tgt_pad = chain_pad[idx, last]
                    tgt_non_pad = ~tgt_pad

                    ref = tok.decode(tgt_ids[tgt_non_pad].tolist())
                    hyp = tok.decode(gen_ids[j].tolist())
                    refs.append(ref)
                    hyps.append(hyp)

                    # Per-token generation accuracy
                    gen_len = min(gen_ids.shape[1], tgt_ids.shape[0])
                    tgt_trimmed = tgt_ids[:gen_len]
                    gen_trimmed = gen_ids[j, :gen_len]
                    mask = tgt_trimmed != tok.pad_token_id
                    if mask.any():
                        correct = (gen_trimmed[mask] == tgt_trimmed[mask]).sum().item()
                        gen_correct += correct
                        gen_total += mask.sum().item()

                        m = mode_ids_b[idx].item()
                        mode_gen_correct[m] += correct
                        mode_gen_total[m] += mask.sum().item()

                    n += 1

    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
    gen_tok_acc = gen_correct / max(gen_total, 1)

    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_str = ""
    for m, name in mode_names.items():
        if mode_gen_total[m] > 0:
            mode_str += f" | gen_{name}: {mode_gen_correct[m]/mode_gen_total[m]:.3f}"

    return chrf_score, gen_tok_acc, mode_str, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg.get("device", "cuda"))

    # Frozen model
    frozen_model = TextDynamicsModel.load(cfg["frozen_model_dir"], device=str(device))
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False
    tok = frozen_model.tokenizer
    d = frozen_model.config.d_model
    bn_d = frozen_model._bottleneck_dim
    print(f"Frozen model: {frozen_model.param_count():,} params (all frozen)")

    # AR decoder
    decoder = ARDecoder(
        vocab_size=tok.vocab_size,
        d_model=d,
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("decoder_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_text_tokens=frozen_model.max_text_tokens,
        dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=bn_d,
        pad_id=tok.pad_token_id,
    ).to(device)
    print(f"AR decoder: {decoder.trainable_param_count():,} trainable params")

    train_ds = ChainDataset(cfg["train_data"], tok, max_text_tokens=frozen_model.max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tok, max_text_tokens=frozen_model.max_text_tokens)
    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01)

    # Cosine schedule with warmup
    warmup_steps = cfg.get("warmup_steps", 1000)
    total_steps = len(train_loader) * cfg.get("epochs", 40)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = cfg.get("epochs", 40)
    log_every = cfg.get("log_every", 2)
    eval_every = cfg.get("eval_every", 5)
    best_chrf = 0.0
    log_file = open(out_dir / "train.log", "w")
    global_step = 0

    for epoch in range(1, epochs + 1):
        decoder.train()
        epoch_loss = epoch_tok = 0.0
        epoch_mode = {}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = compute_loss(frozen_model, decoder, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode[k] = epoch_mode.get(k, 0) + metrics[k]
                    epoch_mode[k + "_n"] = epoch_mode.get(k + "_n", 0) + 1
            n_batches += 1
            global_step += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches
        lr_now = scheduler.get_last_lr()[0]

        if epoch % log_every == 0 or epoch == 1:
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = epoch_mode.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {epoch_mode[k]/n:.3f}"
            msg = f"ep {epoch:4d} | loss {epoch_loss:.4f} | tok {epoch_tok:.3f}{mode_str} | lr {lr_now:.2e}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            decoder.eval()

            # Teacher-forced eval
            eval_loss = eval_tok = 0.0
            eval_mode = {}
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_loss(frozen_model, decoder, batch, device)
                    eval_loss += metrics["loss"]
                    eval_tok += metrics["tok_acc"]
                    for k in ("tok_adv", "tok_qry", "tok_id"):
                        if k in metrics:
                            eval_mode[k] = eval_mode.get(k, 0) + metrics[k]
                            eval_mode[k + "_n"] = eval_mode.get(k + "_n", 0) + 1
                    n_eval += 1

            eval_loss /= n_eval
            eval_tok /= n_eval
            eval_mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = eval_mode.get(k + "_n", 0)
                if n > 0:
                    eval_mode_str += f" | {k.split('_')[1]}: {eval_mode[k]/n:.3f}"

            msg = f"  tf_eval | loss {eval_loss:.4f} | tok {eval_tok:.3f}{eval_mode_str}"
            print(msg)
            log_file.write(msg + "\n")

            # Free-running generation eval
            chrf_score, gen_tok, gen_mode_str, n_gen = generation_eval(
                frozen_model, decoder, test_loader, device, tok,
                n_samples=cfg.get("gen_samples", 200),
            )
            msg = f"  gen     | chrF {chrf_score:.1f} | tok {gen_tok:.3f}{gen_mode_str} ({n_gen} samples)"
            print(msg)
            log_file.write(msg + "\n")

            gap = eval_tok - gen_tok
            msg = f"  gap     | tf-gen = {gap:.3f} ({gap*100:.1f} points)"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if chrf_score > best_chrf:
                best_chrf = chrf_score
                torch.save(decoder.state_dict(), out_dir / "decoder_weights.pt")
                print(f"  saved (best chrF: {best_chrf:.1f})")
                log_file.write(f"  saved (best chrF: {best_chrf:.1f})\n")
                log_file.flush()

    log_file.close()
    print(f"\nDone. Best chrF: {best_chrf:.1f}")
    print(f"Comparison targets (same frozen latents):")
    print(f"  Diffusion expander: 83% tok (tf), generation garbage")
    print(f"  BPE parallel CE:   23% tok, 12.1 chrF")


if __name__ == "__main__":
    main()
