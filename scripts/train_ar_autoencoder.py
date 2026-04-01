#!/usr/bin/env python3
"""Phase 1: TextWorld identity autoencoder.

Compressor → AR decoder, no dynamics. Tests whether the compress→decode
pipeline can round-trip text. If chrF < 80 on literal reconstruction,
the pipeline is fundamentally broken.

Usage:
  uv run python scripts/train_ar_autoencoder.py configs/ar_autoenc_tw.json
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

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_compressor import TextCompressor
from twm.ar_decoder import ARDecoder
from twm.chain_dataset import ChainDataset


class AutoencoderModel(nn.Module):
    """Compressor → AR decoder. No dynamics."""

    def __init__(self, config, tokenizer, compressor_layers=3, decoder_layers=2,
                 d_ff=512, max_text_tokens=128, dropout=0.1, bottleneck_dim=None):
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

    def forward(self, input_ids, input_pad, target_ids, target_pad):
        bn = self.compressor(input_ids, input_pad, self.config.max_triples)
        if isinstance(bn, tuple):
            bn = bn[0]
        return self.decoder(bn, target_ids, target_pad)

    def generate(self, input_ids, input_pad):
        bn = self.compressor(input_ids, input_pad, self.config.max_triples)
        if isinstance(bn, tuple):
            bn = bn[0]
        return self.decoder.generate(bn, max_tokens=self.max_text_tokens)


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
    else:
        tokenizer = DomainBPETokenizer.load(cfg["tokenizer"], max_length=cfg.get("max_text_tokens", 128))

    max_text_tokens = cfg.get("max_text_tokens", 128)
    train_ds = ChainDataset(cfg["train_data"], tokenizer, max_text_tokens=max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tokenizer, max_text_tokens=max_text_tokens)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    model_config = ModelConfig(
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=2,  # unused but required
        d_ff=cfg.get("d_ff", 512),
        max_triples=cfg.get("max_triples", 16),
        dropout=cfg.get("dropout", 0.1),
    )

    model = AutoencoderModel(
        config=model_config, tokenizer=tokenizer,
        compressor_layers=cfg.get("compressor_layers", 3),
        decoder_layers=cfg.get("decoder_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_text_tokens=max_text_tokens,
        dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=cfg.get("bottleneck_dim"),
    )
    model.init_embeddings()
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.get("lr", 3e-4), weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg.get("epochs", 40)
    warmup = cfg.get("warmup_steps", 500)

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
        epoch_loss = epoch_tok = 0.0
        n_batches = 0

        for batch in train_loader:
            # Identity: input = target
            input_ids = batch["input_ids"].to(device)
            input_pad = batch["input_pad"].to(device)

            # For identity chains, step 0 = step 1 = same text
            # Use input as both source and target
            target_ids = input_ids
            target_pad = input_pad

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids, input_pad, target_ids, target_pad)
                # Include first pad position as EOS target so model learns to stop
                eos_mask = ~target_pad.clone()
                for b in range(target_pad.shape[0]):
                    pad_positions = target_pad[b].nonzero(as_tuple=True)[0]
                    if len(pad_positions) > 0:
                        eos_mask[b, pad_positions[0]] = True
                loss = F.cross_entropy(logits[eos_mask], target_ids[eos_mask])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                non_pad = ~target_pad
                preds = logits[non_pad].argmax(-1)
                tok_acc = (preds == target_ids[non_pad]).float().mean().item()

            epoch_loss += loss.item()
            epoch_tok += tok_acc
            n_batches += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            msg = f"ep {epoch:4d} | loss {epoch_loss:.4f} | tf_tok {epoch_tok:.3f} | lr {lr_now:.2e}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            model.eval()

            # Teacher-forced eval
            eval_loss = eval_tok = 0.0
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    input_pad = batch["input_pad"].to(device)
                    logits = model(input_ids, input_pad, input_ids, input_pad)
                    non_pad = ~input_pad
                    loss = F.cross_entropy(logits[non_pad], input_ids[non_pad], ignore_index=0)
                    preds = logits[non_pad].argmax(-1)
                    eval_loss += loss.item()
                    eval_tok += (preds == input_ids[non_pad]).float().mean().item()
                    n_eval += 1
            eval_loss /= n_eval
            eval_tok /= n_eval

            # Generation eval
            from sacrebleu.metrics import CHRF
            chrf_metric = CHRF()
            refs, hyps = [], []
            gen_correct = gen_total = 0
            n_gen = 0
            with torch.no_grad():
                for batch in test_loader:
                    if n_gen >= cfg.get("gen_samples", 200):
                        break
                    input_ids = batch["input_ids"].to(device)
                    input_pad = batch["input_pad"].to(device)

                    for i in range(len(input_ids)):
                        if n_gen >= cfg.get("gen_samples", 200):
                            break
                        gen_ids = model.generate(
                            input_ids[i:i+1], input_pad[i:i+1]
                        )
                        non_pad = ~input_pad[i]
                        ref = tokenizer.decode(input_ids[i][non_pad].tolist())
                        hyp = tokenizer.decode(gen_ids[0].tolist())
                        refs.append(ref)
                        hyps.append(hyp)

                        gen_len = min(gen_ids.shape[1], input_ids.shape[1])
                        mask = input_ids[i, :gen_len] != tokenizer.pad_token_id
                        if mask.any():
                            gen_correct += (gen_ids[0, :gen_len][mask] == input_ids[i, :gen_len][mask]).sum().item()
                            gen_total += mask.sum().item()
                        n_gen += 1

            chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
            gen_tok = gen_correct / max(gen_total, 1)
            gap = eval_tok - gen_tok

            msg = f"  tf_eval | loss {eval_loss:.4f} | tok {eval_tok:.3f}"
            print(msg)
            log_file.write(msg + "\n")
            msg = f"  gen     | chrF {chrf_score:.1f} | tok {gen_tok:.3f} ({n_gen} samples)"
            print(msg)
            log_file.write(msg + "\n")
            msg = f"  gap     | {gap:.3f} ({gap*100:.1f} points)"
            print(msg)
            log_file.write(msg + "\n")

            # Show 3 examples
            for k in range(min(3, len(refs))):
                print(f"  ref: {refs[k][:100]}")
                print(f"  gen: {hyps[k][:100]}")
                print()

            log_file.flush()

            if chrf_score > best_chrf:
                best_chrf = chrf_score
                torch.save(model.state_dict(), out_dir / "weights.pt")
                tokenizer.save(str(out_dir / "tokenizer.json"))
                print(f"  saved (best chrF: {best_chrf:.1f})")
                log_file.write(f"  saved (best chrF: {best_chrf:.1f})\n")
                log_file.flush()

    log_file.close()
    print(f"\nDone. Best chrF: {best_chrf:.1f}")
    print(f"Target: chrF > 80 = pipeline works, < 80 = fundamentally broken")


if __name__ == "__main__":
    main()
