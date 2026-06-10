#!/usr/bin/env python3
"""Train byte-level parallel decoder on frozen BPE dynamics latents.

Frozen: BPE compressor + dynamics core (oracle parity confirmed).
Trainable: byte-level parallel decoder (259 vocab, 512 positions).

Tests whether semantic latents transfer across surface representations.

Usage:
  uv run python scripts/train_parallel_frozen_bytes.py configs/parallel_frozen_bytes_v1.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset


class ByteParallelDecoder(nn.Module):
    """Parallel decoder that outputs bytes (259 classes) from BPE latents."""

    def __init__(self, d_model=128, n_heads=8, n_layers=3, max_bytes=512,
                 dropout=0.1, bottleneck_dim=None):
        super().__init__()
        self.d_model = d_model
        self.max_bytes = max_bytes
        bn_d = bottleneck_dim or d_model
        self.vocab_size = 259  # 256 bytes + pad + mask + unk

        # Learned position queries
        self.pos_queries = nn.Embedding(max_bytes, d_model)

        # Project BPE bottleneck to memory
        self.memory_proj = nn.Sequential(
            nn.Linear(bn_d, d_model),
            nn.LayerNorm(d_model),
        )

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.vocab_size)

        # Length head
        self.length_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _build_memory(self, bottleneck, pre_dynamics=None):
        projected = self.memory_proj(bottleneck)
        if pre_dynamics is not None:
            pre_projected = self.memory_proj(pre_dynamics)
            return torch.cat([pre_projected, projected], dim=1)
        return projected

    def forward(self, bottleneck, target_bytes, target_pad, pre_dynamics=None):
        """Forward: predict byte logits at all positions.

        Args:
            bottleneck: (B, N*3, d) frozen BPE latent
            target_bytes: (B, T) target byte values (0-258)
            target_pad: (B, T) True where padding
            pre_dynamics: (B, N*3, d) optional
        Returns:
            logits: (B, T, 259)
        """
        B, T = target_bytes.shape
        device = target_bytes.device
        memory = self._build_memory(bottleneck, pre_dynamics)
        queries = self.pos_queries(torch.arange(T, device=device))
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(tgt=queries, memory=memory, tgt_key_padding_mask=target_pad)
        return self.output_proj(self.ln_f(out))

    def forward_length(self, bottleneck):
        pooled = bottleneck.mean(dim=1)
        norm_hint = bottleneck.norm(dim=-1).mean(dim=-1, keepdim=True) / self.max_bytes
        return self.length_head(torch.cat([pooled, norm_hint], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def generate(self, bottleneck, max_bytes=None, pre_dynamics=None):
        B = bottleneck.shape[0]
        device = bottleneck.device
        memory = self._build_memory(bottleneck, pre_dynamics)
        if max_bytes is None:
            T = self.forward_length(bottleneck).round().long().clamp(1, self.max_bytes).max().item()
        else:
            T = max_bytes
        queries = self.pos_queries(torch.arange(T, device=device))
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(tgt=queries, memory=memory)
        logits = self.output_proj(self.ln_f(out))
        return logits.argmax(-1)

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def text_to_bytes(text, max_len, pad_id=0):
    """Convert text string to byte tensor with padding."""
    raw = list(text.encode("utf-8"))[:max_len]
    # Shift by 3 to leave room for pad=0, mask=1, unk=2
    shifted = [b + 3 for b in raw]
    padded = shifted + [pad_id] * (max_len - len(shifted))
    return padded, len(shifted)


def bytes_to_text(byte_ids):
    """Convert byte tensor back to text."""
    raw = []
    for b in byte_ids:
        b = b.item() if hasattr(b, "item") else b
        if b <= 2:  # pad/mask/unk
            continue
        raw.append(b - 3)
    try:
        return bytes(raw).decode("utf-8", errors="replace")
    except Exception:
        return "<decode error>"


def compute_loss(frozen_model, decoder, batch, device, max_bytes):
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    B, C, T_bpe = chain_ids.shape
    tok = frozen_model.tokenizer

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

        target_bpe_ids = chain_ids[active, step]
        target_bpe_pad = chain_pad[active, step]
        active_modes = mode_ids[active]
        active_bn = bottleneck[active]
        active_pre = pre_dynamics[active]
        B_active = active_bn.shape[0]

        # Convert BPE targets to byte targets
        byte_targets = []
        byte_pads = []
        byte_lengths = []
        for b in range(B_active):
            non_pad = ~target_bpe_pad[b]
            text = tok.decode(target_bpe_ids[b][non_pad].tolist())
            padded, real_len = text_to_bytes(text, max_bytes)
            byte_targets.append(padded)
            pad_mask = [False] * real_len + [True] * (max_bytes - real_len)
            byte_pads.append(pad_mask)
            byte_lengths.append(real_len)

        byte_tgt = torch.tensor(byte_targets, dtype=torch.long, device=device)
        byte_pad = torch.tensor(byte_pads, dtype=torch.bool, device=device)
        byte_lens = torch.tensor(byte_lengths, dtype=torch.float, device=device)

        # Decode
        logits = decoder(active_bn, byte_tgt, byte_pad, pre_dynamics=active_pre)
        non_pad = ~byte_pad
        if non_pad.any():
            ce_loss = F.cross_entropy(logits[non_pad], byte_tgt[non_pad], ignore_index=0)
            total_loss = total_loss + ce_loss

        # Length loss
        len_pred = decoder.forward_length(active_bn)
        total_loss = total_loss + 0.1 * F.mse_loss(len_pred, byte_lens)

        # Metrics
        with torch.no_grad():
            if non_pad.any():
                preds = logits[non_pad].argmax(-1)
                tok_acc = (preds == byte_tgt[non_pad]).float().mean().item()
                total_tok_acc += tok_acc

                for m in mode_names:
                    mask = active_modes == m
                    if mask.any():
                        m_non_pad = ~byte_pad[mask]
                        if m_non_pad.any():
                            m_preds = logits[mask][m_non_pad].argmax(-1)
                            m_tgt = byte_tgt[mask][m_non_pad]
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
    max_bytes = cfg.get("max_bytes", 512)

    # Frozen BPE model
    frozen_model = TextDynamicsModel.load(cfg["frozen_model_dir"], device=str(device))
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad = False
    tok = frozen_model.tokenizer
    d = frozen_model.config.d_model
    bn_d = frozen_model._bottleneck_dim
    print(f"Frozen model: {frozen_model.param_count():,} params (all frozen)")

    # Byte decoder
    decoder = ByteParallelDecoder(
        d_model=d, n_heads=cfg.get("n_heads", frozen_model.config.n_heads),
        n_layers=cfg.get("decoder_layers", 3),
        max_bytes=max_bytes, dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=bn_d,
    ).to(device)
    print(f"Byte decoder: {decoder.trainable_param_count():,} trainable params")

    train_ds = ChainDataset(cfg["train_data"], tok, max_text_tokens=frozen_model.max_text_tokens)
    test_ds = ChainDataset(cfg["test_data"], tok, max_text_tokens=frozen_model.max_text_tokens)
    print(f"Train: {len(train_ds)} chains, Test: {len(test_ds)} chains")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32))

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = cfg.get("epochs", 40)
    log_every = cfg.get("log_every", 2)
    eval_every = cfg.get("eval_every", 2)
    best_tok_acc = 0.0
    log_file = open(out_dir / "train.log", "w")

    for epoch in range(1, epochs + 1):
        decoder.train()
        epoch_loss = epoch_tok = 0.0
        epoch_mode = {}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = compute_loss(frozen_model, decoder, batch, device, max_bytes)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += metrics["loss"]
            epoch_tok += metrics["tok_acc"]
            for k in ("tok_adv", "tok_qry", "tok_id"):
                if k in metrics:
                    epoch_mode[k] = epoch_mode.get(k, 0) + metrics[k]
                    epoch_mode[k + "_n"] = epoch_mode.get(k + "_n", 0) + 1
            n_batches += 1

        epoch_loss /= n_batches
        epoch_tok /= n_batches

        if epoch % log_every == 0 or epoch == 1:
            mode_str = ""
            for k in ("tok_adv", "tok_qry", "tok_id"):
                n = epoch_mode.get(k + "_n", 0)
                if n > 0:
                    mode_str += f" | {k.split('_')[1]}: {epoch_mode[k]/n:.3f}"
            msg = f"ep {epoch:4d} | loss {epoch_loss:.4f} | tok {epoch_tok:.3f}{mode_str}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        if epoch % eval_every == 0:
            decoder.eval()
            eval_loss = eval_tok = 0.0
            eval_mode = {}
            n_eval = 0
            with torch.no_grad():
                for batch in test_loader:
                    _, metrics = compute_loss(frozen_model, decoder, batch, device, max_bytes)
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
            msg = f"  eval | loss {eval_loss:.4f} | tok {eval_tok:.3f}{eval_mode_str}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            # chrF: generate bytes, decode to text
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
                            if bleu_n >= bleu_samples:
                                break
                            ge = min(gi + gen_bs, len(input_ids))
                            gen_byte_ids = decoder.generate(
                                bn[gi:ge], pre_dynamics=pre_bn[gi:ge]
                            )
                            for j in range(gen_byte_ids.shape[0]):
                                if bleu_n >= bleu_samples:
                                    break
                                idx = gi + j
                                last = chain_len_b[idx].item() - 1
                                tgt_ids = chain_ids[idx, last]
                                tgt_pad = chain_pad[idx, last]
                                ref = tok.decode(tgt_ids[~tgt_pad].tolist())
                                hyp = bytes_to_text(gen_byte_ids[j])
                                refs.append(ref)
                                hyps.append(hyp)
                                bleu_n += 1

                chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
                print(f"  chrF: {chrf_score:.1f} ({bleu_n} samples)")
                log_file.write(f"  chrF: {chrf_score:.1f}\n")
                log_file.flush()

            if eval_tok > best_tok_acc:
                best_tok_acc = eval_tok
                torch.save(decoder.state_dict(), out_dir / "decoder_weights.pt")
                print(f"  saved (best tok_acc: {best_tok_acc:.3f})")

    log_file.close()
    print(f"\nDone. Best eval tok_acc: {best_tok_acc:.3f}")
    print(f"BPE parallel decoder comparison: 23.2% tok, 12.1 chrF")


if __name__ == "__main__":
    main()
