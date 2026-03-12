#!/usr/bin/env python3
"""Train v17: Text compressor/expander with shared 256d bottleneck.

Text identity reconstruction on WebNLG free text. The model discovers
its own internal triple-like structure via learned extraction queries.
Query mode probing is done at inference time, not during training.

Curriculum:
  Phase 1: t in [0.7, 1.0], fixed epochs
  Phase 2: t in [0.0, 1.0], importance sampling, early stop

Usage:
    uv run python scripts/train_v17_text.py \
        --data-dir data/webnlg_multi \
        --out-dir results/v17_text \
        --tokenizer data/webnlg_multi/shared_bpe_tokenizer.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.domain_bpe import DomainBPETokenizer
from twm.text_model import TextWorldModel
from twm.text_dataset import TextDataset
from twm.diffusion_decoder import importance_sample_timesteps


def sample_timestep(B, device, t_min, t_max, bias_power=1.0):
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    if t_min == 0.0 and t_max == 1.0 and bias_power != 1.0:
        return importance_sample_timesteps(B, device, bias_power)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


def compute_loss(model, text_ids, text_pad, text_len, device, timestep,
                 aux_ce_weight=0.1, length_weight=0.1):
    text_ids = text_ids.to(device)
    text_pad = text_pad.to(device)
    token_emb = model.shared_token_emb

    bottleneck = model.compress(text_ids, text_pad)
    pred_emb, _ = model.forward_expander(bottleneck, text_ids, text_pad, timestep=timestep)

    # MSE loss on non-pad tokens
    non_pad = ~text_pad
    metrics = {}
    if not non_pad.any():
        return torch.tensor(0.0, device=device), metrics

    target_clean = token_emb(text_ids)
    mse_loss = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])

    with torch.no_grad():
        cos = F.cosine_similarity(pred_emb[non_pad], target_clean[non_pad], dim=-1).mean()
        pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
        tgt_ids = text_ids[non_pad]
        metrics["tok_acc"] = (nn_ids == tgt_ids).float().mean().item()
        metrics["cos"] = cos.item()

    # Aux CE through decode_proj
    aux_loss = torch.tensor(0.0, device=device)
    if model.text_expander.use_decode_proj:
        logits = model.text_expander.decode_proj_logits(pred_emb)
        aux_loss = F.cross_entropy(logits[non_pad] / 0.1, text_ids[non_pad], ignore_index=0)

    # Length prediction (normalized to [0,1] to match embedding MSE scale)
    len_pred = model.forward_length(bottleneck)
    len_loss = F.mse_loss(len_pred, text_len.float().to(device) / model.max_text_tokens)

    total = mse_loss + aux_ce_weight * aux_loss + length_weight * len_loss
    metrics["loss"] = total.item()
    metrics["mse"] = mse_loss.item()
    metrics["len_loss"] = len_loss.item()
    return total, metrics


@torch.no_grad()
def run_assess(model, ds, device, tokenizer, n_examples=64, n_steps=10):
    model.eval()
    n = min(n_examples, len(ds))
    text_ids = ds._text_token_ids[:n].to(device)
    text_pad = ds._text_pad_mask[:n].to(device)
    pad_id = tokenizer.pad_token_id

    bottleneck = model.compress(text_ids, text_pad)
    gen_ids = model.generate(bottleneck, n_steps=n_steps)

    # Predict lengths and truncate
    pred_lens = model.forward_length(bottleneck)
    pred_lens = pred_lens.round().long().clamp(1, gen_ids.shape[-1])

    tok_match = total_tok = exact = len_match = 0
    for i in range(n):
        tgt = [x for x in text_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        pred = gen_ids[i].cpu().tolist()[:pl]
        if pl == len(tgt):
            len_match += 1
        # Compare up to min length
        cmp_len = min(pl, len(tgt))
        tok_match += sum(1 for a, b in zip(pred[:cmp_len], tgt[:cmp_len]) if a == b)
        total_tok += len(tgt)
        if pred == tgt:
            exact += 1

    return {
        "tok_acc": tok_match / max(total_tok, 1),
        "exact": exact / n,
        "len_acc": len_match / n,
    }


@torch.no_grad()
def print_samples(model, ds, device, tokenizer, n=5, n_steps=10):
    model.eval()
    n = min(n, len(ds))
    text_ids = ds._text_token_ids[:n].to(device)
    text_pad = ds._text_pad_mask[:n].to(device)

    bottleneck = model.compress(text_ids, text_pad)
    gen_ids = model.generate(bottleneck, n_steps=n_steps)
    pred_lens = model.forward_length(bottleneck).round().long().clamp(1, gen_ids.shape[-1])

    def _clean(s):
        """Strip ByteLevel BPE artifacts for display."""
        return s.replace("Ġ", " ").replace("Ċ", "\n").replace("âĢĵ", "-").strip()

    print(f"\n{'='*70}")
    for i in range(n):
        tgt = _clean(tokenizer.decode(text_ids[i].cpu(), skip_special_tokens=True))
        pl = pred_lens[i].item()
        pred = _clean(tokenizer.decode(gen_ids[i, :pl].cpu(), skip_special_tokens=True))
        match = "Y" if tgt == pred else "N"
        print(f"  [{i}] tgt:  {tgt}")
        print(f"       pred: {pred}  [{match}]")
    print(f"{'='*70}", flush=True)


def run_phase(model, train_ds, eval_ds, device, tokenizer, out_dir, phase_name,
              t_min, t_max, bias_power, epochs, patience,
              batch_size, lr, weight_decay, denoise_steps,
              log_every, diagnostic_every, aux_ce_weight, length_weight):
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}  t in [{t_min}, {t_max}]  bias={bias_power}")
    print(f"{'='*60}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    n_train = len(train_ds)
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    init_m = run_assess(model, eval_ds, device, tokenizer, n_examples=64, n_steps=denoise_steps)
    print(f"  init: tok={init_m['tok_acc']:.3f} exact={init_m['exact']:.3f} len={init_m.get('len_acc',0):.3f}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train)
        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            B = idx.shape[0]
            timestep = sample_timestep(B, device, t_min, t_max, bias_power)

            loss, _ = compute_loss(
                model,
                train_ds._text_token_ids[idx],
                train_ds._text_pad_mask[idx],
                train_ds._text_lengths[idx],
                device, timestep,
                aux_ce_weight=aux_ce_weight,
                length_weight=length_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1:
            model.eval()
            gen_m = run_assess(model, eval_ds, device, tokenizer,
                              n_examples=64, n_steps=denoise_steps)

            log = (f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                   f" | tok={gen_m['tok_acc']:.3f} exact={gen_m['exact']:.3f}"
                   f" len={gen_m.get('len_acc',0):.3f}")

            if gen_m["tok_acc"] > best_metric:
                best_metric = gen_m["tok_acc"]
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), phase_dir / "model_best.pt")
                log += " *"
            else:
                no_improve += log_every

            print(log, flush=True)
            history.append({"epoch": epoch, "loss": avg_loss, **gen_m})

            if epoch == 1 or epoch % diagnostic_every == 0:
                print_samples(model, eval_ds, device, tokenizer, n=5, n_steps=denoise_steps)

            if patience > 0 and no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best tok={best_metric:.4f} at epoch {best_epoch})")
                break

    torch.save(model.state_dict(), phase_dir / "model_final.pt")
    with open(phase_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{phase_name} done. Best tok: {best_metric:.4f} at epoch {best_epoch}")
    return best_metric, best_epoch


def main():
    ap = argparse.ArgumentParser(description="Train v17: Text compressor/expander")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--config", type=str, default="base", choices=list(PROFILES.keys()))
    ap.add_argument("--text-compressor-layers", type=int, default=4)
    ap.add_argument("--text-expander-layers", type=int, default=3)
    ap.add_argument("--max-text-tokens", type=int, default=64)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--phase1-epochs", type=int, default=100)
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--phase2-patience", type=int, default=50)
    ap.add_argument("--phase2-bias-power", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = DomainBPETokenizer.load(args.tokenizer, max_length=args.max_text_tokens)
    config = ModelConfig.from_profile(args.config)

    print(f"=== v17 Text Compressor/Expander ===")
    print(f"Device: {device}")
    print(f"Config: {args.config} (d={config.d_model}, h={config.n_heads})")
    print(f"BPE vocab: {tokenizer.vocab_size}")
    print(f"Bottleneck slots: {config.max_triples * 3}")

    model = TextWorldModel(
        config=config,
        domain_tokenizer=tokenizer,
        text_compressor_layers=args.text_compressor_layers,
        text_expander_layers=args.text_expander_layers,
        max_text_tokens=args.max_text_tokens,
        dropout=args.dropout,
        alpha_min=args.alpha_min,
    ).to(device)
    model.init_embeddings()

    print(f"  Compressor: {model.text_compressor.trainable_param_count():,} params")
    print(f"  Expander: {model.text_expander.trainable_param_count():,} params")
    print(f"  Total: {model.param_count():,} ({model.trainable_param_count():,} trainable)")

    train_ds = TextDataset(
        data_dir / "train.jsonl", tokenizer,
        max_text_tokens=args.max_text_tokens,
        max_examples=args.max_examples,
    )
    print(f"  Train: {len(train_ds)} examples")

    test_path = data_dir / "test.jsonl"
    eval_ds = train_ds
    if test_path.exists():
        eval_ds = TextDataset(test_path, tokenizer, max_text_tokens=args.max_text_tokens)
        print(f"  Test: {len(eval_ds)} examples")

    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "architecture": "v17_text",
            "config_profile": args.config,
            "text_compressor_layers": args.text_compressor_layers,
            "text_expander_layers": args.text_expander_layers,
            "max_text_tokens": args.max_text_tokens,
            "denoise_steps": args.denoise_steps,
            "dropout": args.dropout,
            "alpha_min": args.alpha_min,
            "bpe_vocab_size": tokenizer.vocab_size,
            "bottleneck_slots": config.max_triples * 3,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "max_examples": args.max_examples,
        }, f, indent=2)

    p1, p1_ep = run_phase(
        model, train_ds, eval_ds, device, tokenizer,
        out_dir, "phase1",
        t_min=0.7, t_max=1.0, bias_power=1.0,
        epochs=args.phase1_epochs, patience=0,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every, diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight, length_weight=args.length_weight,
    )

    p2, p2_ep = run_phase(
        model, train_ds, eval_ds, device, tokenizer,
        out_dir, "phase2",
        t_min=0.0, t_max=1.0, bias_power=args.phase2_bias_power,
        epochs=args.phase2_epochs, patience=args.phase2_patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every, diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight, length_weight=args.length_weight,
    )

    print(f"\n{'='*60}")
    print(f"DONE -- v17 Text")
    print(f"  Phase 1: best tok={p1:.4f} at epoch {p1_ep}")
    print(f"  Phase 2: best tok={p2:.4f} at epoch {p2_ep}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
