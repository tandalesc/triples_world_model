#!/usr/bin/env python3
"""Train factored chain dynamics with parallel decoder.

Same factored consistency loss as train_factored_chain.py, but replaces
the diffusion expander with a parallel (DETR-style) decoder.
CE loss per position, no diffusion, train=gen by construction.

Usage:
  uv run python scripts/train_factored_parallel.py configs/parallel_v1.json
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
from twm.text_compressor import TextCompressor
from twm.modules import TransformerDynamics
from twm.parallel_decoder import ParallelDecoder
from twm.chain_dataset import ChainDataset


NUM_MODES = 3


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, decay: float):
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


class FactoredParallelModel(nn.Module):
    """Compressor + dynamics + parallel decoder with factored losses."""

    def __init__(self, config, tokenizer, compressor_layers=3, dynamics_layers=2,
                 decoder_layers=3, max_text_tokens=128, dropout=0.1,
                 bottleneck_dim=None, num_modes=NUM_MODES):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        d = config.d_model
        bn_d = bottleneck_dim or d

        # Shared frozen embedding
        self.shared_token_emb = nn.Embedding(tokenizer.vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        # Compressor
        self.text_compressor = TextCompressor(
            token_emb=self.shared_token_emb,
            d_model=d, n_heads=config.n_heads,
            n_layers=compressor_layers,
            max_triples=config.max_triples,
            max_text_tokens=max_text_tokens,
            dropout=dropout, vae=False, bottleneck_dim=bn_d,
        )

        # Dynamics
        dyn_heads = max(1, bn_d // 16)
        self.dynamics = TransformerDynamics(
            d_model=bn_d, n_heads=dyn_heads,
            n_layers=dynamics_layers,
            d_ff=bn_d * 4, dropout=dropout, zero_init=True,
        )

        # Mode embeddings
        self.mode_emb = nn.Embedding(num_modes * 3, bn_d)
        self.mode_role_emb = nn.Embedding(3, bn_d)

        # Parallel decoder
        self.decoder = ParallelDecoder(
            token_emb=self.shared_token_emb,
            d_model=d, n_heads=config.n_heads,
            n_layers=decoder_layers,
            max_text_tokens=max_text_tokens,
            dropout=dropout, bottleneck_dim=bn_d,
        )

    def init_embeddings(self):
        with torch.no_grad():
            nn.init.normal_(self.shared_token_emb.weight, std=0.02)
            self.shared_token_emb.weight.data = F.normalize(
                self.shared_token_emb.weight.data, dim=-1
            )

    def compress(self, text_ids, text_pad):
        return self.text_compressor(text_ids, text_pad, self.config.max_triples)

    def _build_mode_triple(self, mode_ids):
        B = mode_ids.shape[0]
        device = mode_ids.device
        base = mode_ids * 3
        slot_ids = base.unsqueeze(1) + torch.arange(3, device=device)
        mode_triple = self.mode_emb(slot_ids)
        role_idx = torch.arange(3, device=device)
        mode_triple = mode_triple + self.mode_role_emb(role_idx)
        return mode_triple

    def forward_dynamics(self, bottleneck, mode_ids):
        mode_triple = self._build_mode_triple(mode_ids)
        x = torch.cat([mode_triple, bottleneck], dim=1)
        x = self.dynamics(x)
        delta = x[:, 3:]
        return bottleneck + delta

    def forward_decoder(self, bottleneck, target_ids, target_pad, pre_dynamics=None):
        return self.decoder(bottleneck, target_ids, target_pad, pre_dynamics=pre_dynamics)

    def generate(self, bottleneck, max_tokens=None, pre_dynamics=None):
        return self.decoder.generate(bottleneck, max_tokens=max_tokens, pre_dynamics=pre_dynamics)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def compute_loss(model, target_compressor, batch, device, cfg):
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T = chain_ids.shape
    lambda_consist = cfg.get("consistency_weight", 1.0)
    lambda_decode = cfg.get("decode_weight", 1.0)

    bottleneck = model.compress(input_ids, input_pad)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]

    total_loss = torch.tensor(0.0, device=device)
    consist_total = 0.0
    decode_total = 0.0
    total_tok_acc = 0.0
    n_steps = 0

    mode_names = {0: "adv", 1: "qry", 2: "id"}
    mode_tok_sum = {m: 0.0 for m in mode_names}
    mode_tok_n = {m: 0 for m in mode_names}

    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        pre_dynamics = bottleneck
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)

        active = chain_len > step
        if not active.any():
            break

        target_ids = chain_ids[active, step]
        target_pad = chain_pad[active, step]
        active_modes = mode_ids[active]
        active_bn = bottleneck[active]
        active_pre = pre_dynamics[active]

        # Consistency loss
        with torch.no_grad():
            target_bn = target_compressor(target_ids, target_pad, model.config.max_triples)
            if isinstance(target_bn, tuple):
                target_bn = target_bn[0]

        pred_flat = active_bn.reshape(active_bn.shape[0], -1)
        tgt_flat = target_bn.reshape(target_bn.shape[0], -1)
        consist_loss = (1 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1)).mean()
        total_loss = total_loss + lambda_consist * consist_loss
        consist_total += consist_loss.item()

        # Decode loss (CE) — at every step
        logits = model.forward_decoder(active_bn, target_ids, target_pad, pre_dynamics=active_pre)
        non_pad = ~target_pad
        if non_pad.any():
            ce_loss = F.cross_entropy(logits[non_pad], target_ids[non_pad], ignore_index=0)
            total_loss = total_loss + lambda_decode * ce_loss
            decode_total += ce_loss.item()

        # Length loss
        len_pred = model.decoder.forward_length(active_bn)
        from twm.chain_dataset import ChainDataset
        target_len = (~target_pad).sum(dim=-1).float()
        total_loss = total_loss + 0.1 * F.mse_loss(len_pred, target_len)

        # Metrics
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
        "consist": consist_total / max(n_steps, 1),
        "decode": decode_total / max(n_steps, 1),
        "tok_acc": total_tok_acc / max(n_steps, 1),
    }
    for m, name in mode_names.items():
        if mode_tok_n[m] > 0:
            metrics[f"tok_{name}"] = mode_tok_sum[m] / mode_tok_n[m]

    return total_loss, metrics


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

    if "tokenizer_pretrained" in cfg:
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"], max_length=cfg.get("max_text_tokens", 128))
    elif cfg.get("tokenizer") == "bytes":
        tokenizer = DomainBPETokenizer.bytes_tokenizer(max_length=cfg.get("max_text_tokens", 128))
    else:
        tokenizer = DomainBPETokenizer.load(cfg["tokenizer"], max_length=cfg.get("max_text_tokens", 128))

    max_text_tokens = cfg.get("max_text_tokens", 128)
    train_ds = ChainDataset(cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens)
    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    model_config = ModelConfig(
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("dynamics_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_triples=cfg.get("max_triples", 16),
        dropout=cfg.get("dropout", 0.1),
    )

    model = FactoredParallelModel(
        config=model_config, tokenizer=tokenizer,
        compressor_layers=cfg.get("compressor_layers", 3),
        dynamics_layers=cfg.get("dynamics_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 3),
        max_text_tokens=max_text_tokens,
        dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=cfg.get("bottleneck_dim"),
    )
    model.init_embeddings()
    model.to(device)

    target_compressor = copy.deepcopy(model.text_compressor)
    target_compressor.requires_grad_(False)
    target_compressor.to(device)

    print(f"Model: {model.param_count():,} params ({model.trainable_param_count():,} trainable)")
    print(f"Dynamics layers: {cfg.get('dynamics_layers', 2)}, d_model: {cfg.get('d_model', 128)}")

    ema_decay = cfg.get("ema_decay", 0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = cfg.get("epochs", 40)
    log_every = cfg.get("log_every", 2)
    eval_every = cfg.get("eval_every", 2)
    best_tok_acc = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_tok = epoch_consist = epoch_decode = 0.0
        epoch_mode = {}
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

            update_ema(model.text_compressor, target_compressor, ema_decay)

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            epoch_consist += metrics.get("consist", 0)
            epoch_decode += metrics.get("decode", 0)
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode[k] = epoch_mode.get(k, 0) + metrics[k]
                    epoch_mode[k + "_n"] = epoch_mode.get(k + "_n", 0) + 1
            n_batches += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches
        epoch_consist /= n_batches
        epoch_decode /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = epoch_mode.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {epoch_mode[k]/n:.3f}"
            msg = (f"ep {epoch:4d} | loss {epoch_loss:.4f} | consist {epoch_consist:.4f} "
                   f"| decode {epoch_decode:.4f} | tok {epoch_tok:.3f}{mode_str}")
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            model.eval()
            eval_loss = eval_tok = eval_consist = 0.0
            eval_mode = {}
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_loss(model, target_compressor, batch, device, cfg)
                    eval_loss += metrics["loss"]
                    eval_tok += metrics["tok_acc"]
                    eval_consist += metrics.get("consist", 0)
                    for k in ("tok_adv", "tok_qry", "tok_id"):
                        if k in metrics:
                            eval_mode[k] = eval_mode.get(k, 0) + metrics[k]
                            eval_mode[k + "_n"] = eval_mode.get(k + "_n", 0) + 1
                    n_eval += 1

            eval_loss /= n_eval
            eval_tok /= n_eval
            eval_consist /= n_eval
            eval_mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = eval_mode.get(k + "_n", 0)
                if n > 0:
                    eval_mode_str += f" | {k.split('_')[1]}: {eval_mode[k]/n:.3f}"
            msg = f"  eval | loss {eval_loss:.4f} | consist {eval_consist:.4f} | tok {eval_tok:.3f}{eval_mode_str}"
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
                        pre_bn = bn
                        max_c = chain_len_b.max().item()
                        for s in range(1, max_c):
                            pre_bn = bn
                            bn = model.forward_dynamics(bn, mode_ids_b)

                        gen_bs = 8
                        for gi in range(0, len(input_ids), gen_bs):
                            if bleu_n >= bleu_samples:
                                break
                            ge = min(gi + gen_bs, len(input_ids))
                            gen_ids = model.generate(bn[gi:ge], pre_dynamics=pre_bn[gi:ge])
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
                torch.save(model.state_dict(), out_dir / "weights.pt")
                model_config.save(out_dir / "model_config.json")
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")


if __name__ == "__main__":
    main()
