#!/usr/bin/env python3
"""Train AR decoder with entity prediction head on TextWorld.

Entity head predicts discrete key tokens from the dynamics latent.
These become prefix conditioning for the AR decoder, providing
sharp entity disambiguation that soft cross-attention can't resolve.

Pipeline: compressor → dynamics → entity_head → [entity_prefix] + AR decoder

Usage:
  uv run python scripts/train_ar_entity.py configs/ar_entity_tw.json
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
from twm.ar_decoder import ARDecoder
from twm.entity_head import EntityHead
from twm.chain_dataset import ChainDataset

NUM_MODES = 3


@torch.no_grad()
def update_ema(online, target, decay):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(decay).add_(p_o.data, alpha=1 - decay)


class AREntityModel(nn.Module):
    """Compressor + dynamics + entity head + AR decoder."""

    def __init__(self, config, tokenizer, compressor_layers=3, dynamics_layers=2,
                 decoder_layers=2, d_ff=512, max_text_tokens=128, dropout=0.1,
                 bottleneck_dim=None, n_entity_slots=8, num_modes=NUM_MODES):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        d = config.d_model
        bn_d = bottleneck_dim or d

        self.shared_token_emb = nn.Embedding(tokenizer.vocab_size, d)
        self.shared_token_emb.weight.requires_grad = False

        self.compressor = TextCompressor(
            token_emb=self.shared_token_emb,
            d_model=d, n_heads=config.n_heads,
            n_layers=compressor_layers,
            max_triples=config.max_triples,
            max_text_tokens=max_text_tokens,
            dropout=dropout, vae=False, bottleneck_dim=bn_d,
        )

        dyn_heads = max(1, bn_d // 16)
        self.dynamics = TransformerDynamics(
            d_model=bn_d, n_heads=dyn_heads,
            n_layers=dynamics_layers,
            d_ff=bn_d * 4, dropout=dropout, zero_init=True,
        )

        self.mode_emb = nn.Embedding(num_modes * 3, bn_d)
        self.mode_role_emb = nn.Embedding(3, bn_d)

        self.entity_head = EntityHead(
            vocab_size=tokenizer.vocab_size,
            d_model=d, n_entity_slots=n_entity_slots,
            n_heads=config.n_heads, bottleneck_dim=bn_d,
            pad_id=tokenizer.pad_token_id,
        )

        self.decoder = ARDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=d, n_heads=config.n_heads,
            n_layers=decoder_layers, d_ff=d_ff,
            max_text_tokens=max_text_tokens,
            dropout=dropout, bottleneck_dim=bn_d,
            pad_id=tokenizer.pad_token_id,
        )

    def init_embeddings(self):
        with torch.no_grad():
            nn.init.normal_(self.shared_token_emb.weight, std=0.02)
            self.shared_token_emb.weight.data = F.normalize(
                self.shared_token_emb.weight.data, dim=-1
            )

    def compress(self, text_ids, text_pad):
        return self.compressor(text_ids, text_pad, self.config.max_triples)

    def _build_mode_triple(self, mode_ids):
        B = mode_ids.shape[0]
        device = mode_ids.device
        base = mode_ids * 3
        slot_ids = base.unsqueeze(1) + torch.arange(3, device=device)
        mode_triple = self.mode_emb(slot_ids)
        return mode_triple + self.mode_role_emb(torch.arange(3, device=device))

    def forward_dynamics(self, bottleneck, mode_ids):
        mode_triple = self._build_mode_triple(mode_ids)
        x = torch.cat([mode_triple, bottleneck], dim=1)
        x = self.dynamics(x)
        delta = x[:, 3:]
        return bottleneck + delta


def extract_entity_targets(target_ids, target_pad, tokenizer, n_slots):
    """Extract key content tokens from target text as entity supervision.

    Heuristic: take the first n_slots non-function-word tokens.
    Function words: determiners, prepositions, punctuation.
    """
    B, T = target_ids.shape
    device = target_ids.device
    pad_id = tokenizer.pad_token_id

    # Common function word token IDs (precompute once would be better, but fine for now)
    func_words = set()
    for w in ["the", "a", "an", "from", "on", "in", "into", "with", "of", "to",
              "is", "are", "was", ".", ",", ":", "-", "=", " ", "  ", "you", "your",
              "has", "just", "by", "one", "not", "and", "it", "that", "this"]:
        ids = tokenizer.encode(w, max_length=5)
        func_words.update(i for i in ids if i != pad_id)

    entity_targets = torch.full((B, n_slots), pad_id, dtype=torch.long, device=device)

    for b in range(B):
        non_pad = target_ids[b] != pad_id
        tgt = target_ids[b][non_pad].tolist()
        content = [t for t in tgt if t not in func_words]
        for k, tok_id in enumerate(content[:n_slots]):
            entity_targets[b, k] = tok_id

    return entity_targets


def compute_loss(model, target_compressor, batch, device, cfg):
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T = chain_ids.shape
    lambda_consist = cfg.get("consistency_weight", 1.0)
    lambda_entity = cfg.get("entity_weight", 1.0)
    n_slots = cfg.get("n_entity_slots", 8)

    bottleneck = model.compress(input_ids, input_pad)
    if isinstance(bottleneck, tuple):
        bottleneck = bottleneck[0]

    total_loss = torch.tensor(0.0, device=device)
    consist_total = entity_total = 0.0
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

        # Entity head loss
        entity_targets = extract_entity_targets(
            target_ids, target_pad, model.tokenizer, n_slots
        )
        entity_logits = model.entity_head(active_bn)  # (B', K, V)
        entity_mask = entity_targets != model.tokenizer.pad_token_id
        if entity_mask.any():
            entity_loss = F.cross_entropy(
                entity_logits[entity_mask], entity_targets[entity_mask]
            )
            total_loss = total_loss + lambda_entity * entity_loss
            entity_total += entity_loss.item()

        # AR decode with entity prefix (teacher-forced entity tokens)
        logits = model.decoder(
            active_bn, target_ids, target_pad,
            pre_dynamics=active_pre,
            entity_prefix=entity_targets,  # use ground truth entities during training
        )

        # EOS-aware CE loss
        eos_mask = ~target_pad.clone()
        for b in range(target_pad.shape[0]):
            pad_positions = target_pad[b].nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                eos_mask[b, pad_positions[0]] = True
        if eos_mask.any():
            ce_loss = F.cross_entropy(logits[eos_mask], target_ids[eos_mask])
            total_loss = total_loss + ce_loss

        with torch.no_grad():
            non_pad = ~target_pad
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
        "entity": entity_total / max(n_steps, 1),
        "tok_acc": total_tok_acc / max(n_steps, 1),
    }
    for m, name in mode_names.items():
        if mode_tok_n[m] > 0:
            metrics[f"tok_{name}"] = mode_tok_sum[m] / mode_tok_n[m]

    return total_loss, metrics


def generation_eval(model, test_loader, device, tok, n_samples=200):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()
    refs, hyps = [], []
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

            bn = model.compress(input_ids, input_pad)
            if isinstance(bn, tuple):
                bn = bn[0]
            pre_bn = bn
            max_c = chain_len_b.max().item()
            for s in range(1, max_c):
                pre_bn = bn
                bn = model.forward_dynamics(bn, mode_ids_b)

            for i in range(len(input_ids)):
                if n >= n_samples:
                    break
                last = chain_len_b[i].item() - 1
                tgt_ids = chain_ids[i, last]
                tgt_pad = chain_pad[i, last]
                non_pad = ~tgt_pad

                # Predict entities from latent
                entity_ids = model.entity_head.predict(bn[i:i+1])

                gen_ids = model.decoder.generate(
                    bn[i:i+1], pre_dynamics=pre_bn[i:i+1],
                    max_tokens=model.max_text_tokens,
                    entity_prefix=entity_ids,
                )

                ref = tok.decode(tgt_ids[non_pad].tolist())
                hyp = tok.decode(gen_ids[0].tolist())
                refs.append(ref)
                hyps.append(hyp)

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

    return chrf_score, gen_tok, mode_str, n


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
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("dynamics_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_triples=cfg.get("max_triples", 16),
        dropout=cfg.get("dropout", 0.1),
    )

    model = AREntityModel(
        config=model_config, tokenizer=tokenizer,
        compressor_layers=cfg.get("compressor_layers", 3),
        dynamics_layers=cfg.get("dynamics_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_text_tokens=max_text_tokens,
        dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=cfg.get("bottleneck_dim"),
        n_entity_slots=cfg.get("n_entity_slots", 8),
    )
    model.init_embeddings()
    model.to(device)

    target_compressor = copy.deepcopy(model.compressor)
    target_compressor.requires_grad_(False)
    target_compressor.to(device)

    ema_decay = cfg.get("ema_decay", 0.999)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable)")
    print(f"Entity slots: {cfg.get('n_entity_slots', 8)}, EMA decay: {ema_decay}")

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
        epoch_loss = epoch_tok = epoch_consist = epoch_entity = 0.0
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
            scheduler.step()

            update_ema(model.compressor, target_compressor, ema_decay)

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            epoch_consist += metrics.get("consist", 0)
            epoch_entity += metrics.get("entity", 0)
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode[k] = epoch_mode.get(k, 0) + metrics[k]
                    epoch_mode[k + "_n"] = epoch_mode.get(k + "_n", 0) + 1
            n_batches += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches
        epoch_consist /= n_batches
        epoch_entity /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = epoch_mode.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {epoch_mode[k]/n:.3f}"
            msg = (f"ep {epoch:4d} | loss {epoch_loss:.4f} | consist {epoch_consist:.4f} "
                   f"| entity {epoch_entity:.4f} | tok {epoch_tok:.3f}{mode_str} | lr {lr_now:.2e}")
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

            msg = f"  tf_eval | loss {eval_loss:.4f} | consist {eval_consist:.4f} | tok {eval_tok:.3f}{eval_mode_str}"
            print(msg)
            log_file.write(msg + "\n")

            chrf, gen_tok, gen_mode_str, n_gen = generation_eval(
                model, test_loader, device, tokenizer,
                n_samples=cfg.get("gen_samples", 200),
            )
            gap = eval_tok - gen_tok
            msg = f"  gen     | chrF {chrf:.1f} | tok {gen_tok:.3f}{gen_mode_str} ({n_gen} samples)"
            print(msg)
            log_file.write(msg + "\n")
            msg = f"  gap     | {gap:.3f} ({gap*100:.1f} points)"
            print(msg)
            log_file.write(msg + "\n")
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
    print(f"Phase 2 (no entity head): chrF 49.0")


if __name__ == "__main__":
    main()
