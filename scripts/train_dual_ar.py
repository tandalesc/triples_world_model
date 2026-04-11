#!/usr/bin/env python3
"""Train dual-memory AR decoder on TextWorld dynamics.

Dense (compressor) + sparse (dynamics) cross-attention channels.
Dense channel dropped 30% during training to prevent copy shortcut.

Usage:
  uv run python scripts/train_dual_ar.py configs/dual_ar_v1.json
"""

import argparse
import copy
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

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_compressor import TextCompressor
from twm.modules import TransformerDynamics
from twm.dual_ar_decoder import DualARDecoder
from twm.chain_dataset import ChainDataset

NUM_MODES = 3


@torch.no_grad()
def update_ema(online, target, decay):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(decay).add_(p_o.data, alpha=1 - decay)


class DualARModel(nn.Module):
    def __init__(self, config, tokenizer, compressor_layers=3, dynamics_layers=2,
                 decoder_layers=3, d_ff=512, max_text_tokens=128, dropout=0.1,
                 bottleneck_dim=None, dense_dropout=0.3):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        d = config.d_model
        bn_d = bottleneck_dim or d

        self.shared_token_emb = nn.Embedding(tokenizer.vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        self.compressor = TextCompressor(
            token_emb=self.shared_token_emb, d_model=d, n_heads=config.n_heads,
            n_layers=compressor_layers, max_triples=config.max_triples,
            max_text_tokens=max_text_tokens, dropout=dropout, vae=False,
            bottleneck_dim=bn_d,
        )

        dyn_heads = max(1, bn_d // 16)
        self.dynamics = TransformerDynamics(
            d_model=bn_d, n_heads=dyn_heads, n_layers=dynamics_layers,
            d_ff=bn_d * 4, dropout=dropout, zero_init=True,
        )

        self.mode_emb = nn.Embedding(NUM_MODES * 3, bn_d)
        self.mode_role_emb = nn.Embedding(3, bn_d)

        self.decoder = DualARDecoder(
            vocab_size=tokenizer.vocab_size, d_model=d, n_heads=config.n_heads,
            n_layers=decoder_layers, d_ff=d_ff, max_text_tokens=max_text_tokens,
            dropout=dropout, bottleneck_dim=bn_d, pad_id=tokenizer.pad_token_id,
            dense_dropout=dense_dropout,
        )

        # Dense projection: dynamics bottleneck → compressor-space
        # Re-expands the sparse prediction into entity-grounded representation
        # Trained to match compressor(target_text) via MSE
        self.dense_proj = nn.Sequential(
            nn.Linear(bn_d, bn_d),
            nn.GELU(),
            nn.Linear(bn_d, bn_d),
            nn.LayerNorm(bn_d),
        )

    def init_embeddings(self):
        with torch.no_grad():
            nn.init.normal_(self.shared_token_emb.weight, std=0.02)
            self.shared_token_emb.weight.data = F.normalize(
                self.shared_token_emb.weight.data, dim=-1)

    def compress(self, text_ids, text_pad):
        return self.compressor(text_ids, text_pad, self.config.max_triples)

    def _build_mode_triple(self, mode_ids):
        device = mode_ids.device
        base = mode_ids * 3
        slot_ids = base.unsqueeze(1) + torch.arange(3, device=device)
        return self.mode_emb(slot_ids) + self.mode_role_emb(torch.arange(3, device=device))

    def forward_dynamics(self, bottleneck, mode_ids):
        mode_triple = self._build_mode_triple(mode_ids)
        x = torch.cat([mode_triple, bottleneck], dim=1)
        x = self.dynamics(x)
        return bottleneck + x[:, 3:]


def compute_loss(model, target_compressor, batch, device, cfg):
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T = chain_ids.shape
    lambda_consist = cfg.get("consistency_weight", 1.0)
    lambda_dense = cfg.get("dense_proj_weight", 0.5)

    # Compress input
    compressor_out = model.compress(input_ids, input_pad)
    if isinstance(compressor_out, tuple):
        compressor_out = compressor_out[0]

    bottleneck = compressor_out

    total_loss = torch.tensor(0.0, device=device)
    consist_total = dense_proj_total = 0.0
    total_tok_acc = 0.0
    n_steps = 0

    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_tok_sum = {m: 0.0 for m in mode_names}
    mode_tok_n = {m: 0 for m in mode_names}

    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        # Dynamics: sparse channel
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)

        active = chain_len > step
        if not active.any():
            break

        target_ids = chain_ids[active, step]
        target_pad = chain_pad[active, step]
        active_modes = mode_ids[active]
        active_dynamics = bottleneck[active]

        # Dense channel: project dynamics output → compressor-space
        # This is the PREDICTED next-state in dense representation
        active_dense = model.dense_proj(active_dynamics)

        # Consistency loss (sparse)
        with torch.no_grad():
            target_bn = target_compressor(target_ids, target_pad, model.config.max_triples)
            if isinstance(target_bn, tuple):
                target_bn = target_bn[0]

        pred_flat = active_dynamics.reshape(active_dynamics.shape[0], -1)
        tgt_flat = target_bn.reshape(target_bn.shape[0], -1)
        consist_loss = (1 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1)).mean()
        total_loss = total_loss + lambda_consist * consist_loss
        consist_total += consist_loss.item()

        # Dense projection loss: predicted dense should match target's compressor output
        dense_proj_loss = F.mse_loss(active_dense, target_bn.detach())
        total_loss = total_loss + lambda_dense * dense_proj_loss
        dense_proj_total += dense_proj_loss.item()

        # Dual-memory AR decode
        logits = model.decoder(
            dynamics_out=active_dynamics,   # sparse channel
            compressor_out=active_dense,     # dense channel
            target_ids=target_ids,
            target_pad_mask=target_pad,
        )

        # EOS-aware CE loss
        eos_mask = ~target_pad.clone()
        B_active = target_pad.shape[0]
        for b in range(B_active):
            pad_pos = target_pad[b].nonzero(as_tuple=True)[0]
            if len(pad_pos) > 0:
                eos_mask[b, pad_pos[0]] = True
        if eos_mask.any():
            ce_loss = F.cross_entropy(logits[eos_mask], target_ids[eos_mask])
            total_loss = total_loss + ce_loss

        # Metrics
        with torch.no_grad():
            non_pad = ~target_pad
            if non_pad.any():
                preds = logits[non_pad].argmax(-1)
                tok_acc = (preds == target_ids[non_pad]).float().mean().item()
                total_tok_acc += tok_acc
                for m in mode_names:
                    mask = active_modes == m
                    if mask.any():
                        m_np = ~target_pad[mask]
                        if m_np.any():
                            mp = logits[mask][m_np].argmax(-1)
                            mt = target_ids[mask][m_np]
                            mode_tok_sum[m] += (mp == mt).float().mean().item()
                            mode_tok_n[m] += 1
            n_steps += 1

    if n_steps > 0:
        total_loss = total_loss / n_steps

    metrics = {
        "loss": total_loss.item(),
        "consist": consist_total / max(n_steps, 1),
        "dense_mse": dense_proj_total / max(n_steps, 1),
        "tok_acc": total_tok_acc / max(n_steps, 1),
    }
    for m, name in mode_names.items():
        if mode_tok_n[m] > 0:
            metrics[f"tok_{name}"] = mode_tok_sum[m] / mode_tok_n[m]

    return total_loss, metrics


def generation_eval(model, test_loader, device, tok, n_samples=200):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()
    refs, hyps, examples = [], [], []
    gen_correct = gen_total = 0
    mode_gen = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
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

            compressor_out = model.compress(input_ids, input_pad)
            if isinstance(compressor_out, tuple):
                compressor_out = compressor_out[0]

            bn = compressor_out
            max_c = chain_len_b.max().item()
            for s in range(1, max_c):
                bn = model.forward_dynamics(bn, mode_ids_b)

            # Dense channel: project dynamics output → compressor-space
            dense_pred = model.dense_proj(bn)

            for i in range(len(input_ids)):
                if n >= n_samples:
                    break
                last = chain_len_b[i].item() - 1
                tgt_ids = chain_ids[i, last]
                tgt_pad = chain_pad[i, last]
                non_pad = ~tgt_pad

                gen_ids = model.decoder.generate(
                    dynamics_out=bn[i:i+1],
                    compressor_out=dense_pred[i:i+1],
                    max_tokens=model.max_text_tokens,
                )

                ref = tok.decode(tgt_ids[non_pad].tolist())
                hyp = tok.decode(gen_ids[0].tolist())
                refs.append(ref)
                hyps.append(hyp)

                if len(examples) < 5:
                    examples.append({
                        "mode": mode_ids_b[i].item(),
                        "ref": ref[:100],
                        "gen": hyp[:100],
                    })

                gen_len = min(gen_ids.shape[1], tgt_ids.shape[0])
                mask = tgt_ids[:gen_len] != tok.pad_token_id
                if mask.any():
                    c = (gen_ids[0, :gen_len][mask] == tgt_ids[:gen_len][mask]).sum().item()
                    gen_correct += c
                    gen_total += mask.sum().item()
                    m = mode_ids_b[i].item()
                    mode_gen[m][0] += c
                    mode_gen[m][1] += mask.sum().item()
                n += 1

    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
    gen_tok = gen_correct / max(gen_total, 1)
    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_str = ""
    for m, name in mode_names.items():
        if mode_gen[m][1] > 0:
            mode_str += f" | gen_{name}: {mode_gen[m][0]/mode_gen[m][1]:.3f}"

    return chrf_score, gen_tok, mode_str, n, examples


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
    tokenizer = DomainBPETokenizer.from_pretrained(
        cfg.get("tokenizer_pretrained", "gpt2"),
        max_length=cfg.get("max_text_tokens", 128),
    )

    max_text_tokens = cfg.get("max_text_tokens", 128)
    train_ds = ChainDataset(cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    model_config = ModelConfig(
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("dynamics_layers", 2), d_ff=cfg.get("d_ff", 512),
        max_triples=cfg.get("max_triples", 16), dropout=cfg.get("dropout", 0.1),
    )

    model = DualARModel(
        config=model_config, tokenizer=tokenizer,
        compressor_layers=cfg.get("compressor_layers", 3),
        dynamics_layers=cfg.get("dynamics_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 3),
        d_ff=cfg.get("d_ff", 512),
        max_text_tokens=max_text_tokens, dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=cfg.get("bottleneck_dim"),
        dense_dropout=cfg.get("dense_dropout", 0.3),
    )
    model.init_embeddings()
    model.to(device)

    target_compressor = copy.deepcopy(model.compressor)
    target_compressor.requires_grad_(False)
    target_compressor.to(device)

    ema_decay = cfg.get("ema_decay", 0.999)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params ({trainable:,} trainable)")
    print(f"Dense dropout: {cfg.get('dense_dropout', 0.3)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 3e-4), weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg.get("epochs", 40)
    warmup = cfg.get("warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = cfg.get("epochs", 40)
    log_every = cfg.get("log_every", 2)
    eval_every = cfg.get("eval_every", 5)
    best_chrf = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        ep = {"loss": 0, "consist": 0, "dense_mse": 0, "tok_acc": 0}
        ep_mode = {}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = compute_loss(model, target_compressor, batch, device, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(model.compressor, target_compressor, ema_decay)

            for k in ep:
                ep[k] += metrics.get(k, 0)
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    ep_mode[k] = ep_mode.get(k, 0) + metrics[k]
                    ep_mode[k + "_n"] = ep_mode.get(k + "_n", 0) + 1
            n_batches += 1

        for k in ep:
            ep[k] /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = ep_mode.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {ep_mode[k]/n:.3f}"
            msg = (f"ep {epoch:4d} | loss {ep['loss']:.4f} | consist {ep['consist']:.4f} "
                   f"| dense {ep['dense_mse']:.4f} | tok {ep['tok_acc']:.3f}{mode_str} | lr {lr_now:.2e}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            model.eval()
            chrf, gen_tok, gen_mode_str, n_gen, examples = generation_eval(
                model, test_loader, device, tokenizer,
                n_samples=cfg.get("gen_samples", 200),
            )
            msg = f"  gen | chrF {chrf:.1f} | tok {gen_tok:.3f}{gen_mode_str} ({n_gen} samples)"
            print(msg)
            log_file.write(msg + "\n")

            for ex in examples[:3]:
                mode_name = {0: "adv", 1: "qry", 2: "id"}[ex["mode"]]
                print(f"  [{mode_name}] ref: {ex['ref']}")
                print(f"          gen: {ex['gen']}")
                log_file.write(f"  [{mode_name}] ref: {ex['ref']}\n")
                log_file.write(f"          gen: {ex['gen']}\n")

            log_file.flush()

            if chrf > best_chrf:
                best_chrf = chrf
                torch.save(model.state_dict(), out_dir / "weights.pt")
                tokenizer.save(str(out_dir / "tokenizer.json"))
                print(f"  saved (best chrF: {best_chrf:.1f})")
                log_file.write(f"  saved (best chrF: {best_chrf:.1f})\n")
                log_file.flush()

    log_file.close()
    print(f"\nDone. Best chrF: {best_chrf:.1f}")
    print(f"AR baseline (single memory): chrF 49.0")
    print(f"Flow matching: chrF 21.1 (failed)")


if __name__ == "__main__":
    main()
