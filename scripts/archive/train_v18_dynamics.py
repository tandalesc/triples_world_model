#!/usr/bin/env python3
"""Train v18: Text dynamics with identity + Q&A modes.

Text compressor → dynamics core → text expander pipeline.
Trains on paired text data with mode conditioning.

Modes:
  - identity (0): reconstruct input text
  - qa (1): question → answer transformation

Curriculum:
  Phase 1: t in [0.7, 1.0], fixed epochs
  Phase 2: t in [0.0, 1.0], importance sampling, early stop

Usage:
    # First generate the Q&A dataset:
    uv run python scripts/generate_qa_dataset.py \
        --input data/webnlg_multi/train.jsonl \
        --output data/webnlg_multi/qa_train.jsonl
    uv run python scripts/generate_qa_dataset.py \
        --input data/webnlg_multi/test.jsonl \
        --output data/webnlg_multi/qa_test.jsonl

    # Then train:
    uv run python scripts/train_v18_dynamics.py \
        --data-dir data/webnlg_multi \
        --out-dir results/v18_dynamics \
        --tokenizer data/webnlg_multi/shared_bpe_tokenizer.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.text_pair_dataset import TextPairDataset
from twm.diffusion_decoder import importance_sample_timesteps


def sample_timestep(B, device, t_min, t_max, bias_power=1.0):
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    if t_min == 0.0 and t_max == 1.0 and bias_power != 1.0:
        return importance_sample_timesteps(B, device, bias_power)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


def compute_loss(model, input_ids, input_pad, output_ids, output_pad,
                 output_len, mode_ids, device, timestep,
                 aux_ce_weight=0.1, length_weight=0.1):
    input_ids = input_ids.to(device)
    input_pad = input_pad.to(device)
    output_ids = output_ids.to(device)
    output_pad = output_pad.to(device)
    mode_ids = mode_ids.to(device)
    token_emb = model.shared_token_emb

    # Compress input -> dynamics -> bottleneck'
    bottleneck = model.compress(input_ids, input_pad)
    bottleneck = model.forward_dynamics(bottleneck, mode_ids)

    # Expand toward output text
    pred_emb, _ = model.forward_expander(bottleneck, output_ids, output_pad, timestep=timestep)

    # MSE loss on non-pad output tokens
    non_pad = ~output_pad
    metrics = {}
    if not non_pad.any():
        return torch.tensor(0.0, device=device), metrics

    target_clean = token_emb(output_ids)
    mse_loss = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])

    with torch.no_grad():
        cos = F.cosine_similarity(pred_emb[non_pad], target_clean[non_pad], dim=-1).mean()
        pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
        tgt_ids = output_ids[non_pad]
        metrics["tok_acc"] = (nn_ids == tgt_ids).float().mean().item()
        metrics["cos"] = cos.item()

        # Per-mode accuracy
        for mode_val, mode_name in [(0, "id"), (1, "qa")]:
            mask = mode_ids == mode_val
            if mask.any():
                mode_non_pad = ~output_pad[mask]
                if mode_non_pad.any():
                    mode_pred = pred_emb[mask][mode_non_pad]
                    mode_tgt = output_ids[mask][mode_non_pad]
                    mode_pred_n = F.normalize(mode_pred, dim=-1)
                    mode_nn = torch.matmul(mode_pred_n, emb_norm.T).argmax(-1)
                    metrics[f"tok_{mode_name}"] = (mode_nn == mode_tgt).float().mean().item()

    # Aux CE through decode_proj
    aux_loss = torch.tensor(0.0, device=device)
    if model.text_expander.use_decode_proj:
        logits = model.text_expander.decode_proj_logits(pred_emb)
        aux_loss = F.cross_entropy(logits[non_pad] / 0.1, output_ids[non_pad], ignore_index=0)

    # Length prediction (target normalized to [0,1])
    len_pred = model.forward_length(bottleneck)
    len_target = output_len.float().to(device) / model.max_text_tokens
    len_loss = F.mse_loss(len_pred, len_target)

    total = mse_loss + aux_ce_weight * aux_loss + length_weight * len_loss
    metrics["loss"] = total.item()
    metrics["mse"] = mse_loss.item()
    metrics["len_loss"] = len_loss.item()
    return total, metrics


@torch.no_grad()
def run_assess(model, ds, device, tokenizer, n_examples=64, n_steps=10):
    model.eval()
    n = min(n_examples, len(ds))
    input_ids = ds._input_token_ids[:n].to(device)
    input_pad = ds._input_pad_mask[:n].to(device)
    output_ids = ds._output_token_ids[:n].to(device)
    mode_ids = ds._modes[:n].to(device)
    pad_id = tokenizer.pad_token_id

    # Compress -> dynamics -> generate
    bottleneck = model.compress(input_ids, input_pad)
    bottleneck = model.forward_dynamics(bottleneck, mode_ids)
    gen_ids = model.generate(bottleneck, n_steps=n_steps)

    # Predict lengths (scale back from [0,1])
    pred_lens = model.forward_length(bottleneck)
    pred_lens = (pred_lens * model.max_text_tokens).round().long().clamp(1, gen_ids.shape[-1])

    tok_match = total_tok = exact_count = len_match = 0
    id_tok_match = id_total = qa_tok_match = qa_total = 0

    for i in range(n):
        tgt = [x for x in output_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        pred = gen_ids[i].cpu().tolist()[:pl]
        mode = mode_ids[i].item()

        if pl == len(tgt):
            len_match += 1

        cmp_len = min(pl, len(tgt))
        matches = sum(1 for a, b in zip(pred[:cmp_len], tgt[:cmp_len]) if a == b)
        tok_match += matches
        total_tok += len(tgt)

        if mode == 0:
            id_tok_match += matches
            id_total += len(tgt)
        else:
            qa_tok_match += matches
            qa_total += len(tgt)

        if pred == tgt:
            exact_count += 1

    result = {
        "tok_acc": tok_match / max(total_tok, 1),
        "exact": exact_count / n,
        "len_acc": len_match / n,
    }
    if id_total > 0:
        result["tok_id"] = id_tok_match / id_total
    if qa_total > 0:
        result["tok_qa"] = qa_tok_match / qa_total
    return result


@torch.no_grad()
def print_samples(model, ds, device, tokenizer, n=5, n_steps=10):
    model.eval()
    n = min(n, len(ds))
    input_ids = ds._input_token_ids[:n].to(device)
    input_pad = ds._input_pad_mask[:n].to(device)
    output_ids = ds._output_token_ids[:n].to(device)
    mode_ids = ds._modes[:n].to(device)

    bottleneck = model.compress(input_ids, input_pad)
    bottleneck = model.forward_dynamics(bottleneck, mode_ids)
    gen_ids = model.generate(bottleneck, n_steps=n_steps)
    pred_lens = model.forward_length(bottleneck)
    pred_lens = (pred_lens * model.max_text_tokens).round().long().clamp(1, gen_ids.shape[-1])

    def _clean(s):
        return s.replace("\u0120", " ").replace("\u010a", "\n").replace("\u00e2\u0122\u0135", "-").strip()

    mode_names = {0: "ID", 1: "QA"}

    print(f"\n{'='*70}")
    for i in range(n):
        mode = mode_names.get(mode_ids[i].item(), "??")
        inp = _clean(tokenizer.decode(input_ids[i].cpu(), skip_special_tokens=True))
        tgt = _clean(tokenizer.decode(output_ids[i].cpu(), skip_special_tokens=True))
        pl = pred_lens[i].item()
        pred = _clean(tokenizer.decode(gen_ids[i, :pl].cpu(), skip_special_tokens=True))
        match = "Y" if tgt == pred else "N"
        print(f"  [{i}|{mode}] inp:  {inp}")
        print(f"         tgt:  {tgt}")
        print(f"         pred: {pred}  [{match}]")
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

    # Differential LR: dynamics/mode at full LR, compressor/expander at 1/10th
    dynamics_params = list(model.dynamics.parameters()) + [model.mode_embeddings, model.dynamics_gate]
    io_params = [p for p in list(model.text_compressor.parameters()) + list(model.text_expander.parameters())
                 if p.requires_grad]

    param_groups = [{"params": dynamics_params, "lr": lr}]
    if io_params:
        param_groups.append({"params": io_params, "lr": lr * 0.1})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    n_train = len(train_ds)
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    init_m = run_assess(model, eval_ds, device, tokenizer, n_examples=64, n_steps=denoise_steps)
    init_log = (f"  init: tok={init_m['tok_acc']:.3f} exact={init_m['exact']:.3f} "
                f"len={init_m.get('len_acc',0):.3f}")
    if "tok_id" in init_m:
        init_log += f" id={init_m['tok_id']:.3f}"
    if "tok_qa" in init_m:
        init_log += f" qa={init_m['tok_qa']:.3f}"
    print(init_log, flush=True)

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
                train_ds._input_token_ids[idx],
                train_ds._input_pad_mask[idx],
                train_ds._output_token_ids[idx],
                train_ds._output_pad_mask[idx],
                train_ds._output_lengths[idx],
                train_ds._modes[idx],
                device, timestep,
                aux_ce_weight=aux_ce_weight,
                length_weight=length_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            all_trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1:
            model.train(False)
            gen_m = run_assess(model, eval_ds, device, tokenizer,
                              n_examples=64, n_steps=denoise_steps)

            log = (f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                   f" | tok={gen_m['tok_acc']:.3f} exact={gen_m['exact']:.3f}"
                   f" len={gen_m.get('len_acc',0):.3f}")
            if "tok_id" in gen_m:
                log += f" id={gen_m['tok_id']:.3f}"
            if "tok_qa" in gen_m:
                log += f" qa={gen_m['tok_qa']:.3f}"

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
    ap = argparse.ArgumentParser(description="Train v18: Text dynamics (identity + Q&A)")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--config", type=str, default="base", choices=list(PROFILES.keys()))
    ap.add_argument("--text-compressor-layers", type=int, default=4)
    ap.add_argument("--text-expander-layers", type=int, default=3)
    ap.add_argument("--dynamics-layers", type=int, default=None,
                    help="Override dynamics core layers (default: from config profile)")
    ap.add_argument("--max-text-tokens", type=int, default=64)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--length-weight", type=float, default=0.1,
                    help="Weight for length loss (target normalized to [0,1])")
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--phase1-epochs", type=int, default=100)
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--phase2-patience", type=int, default=50)
    ap.add_argument("--phase2-bias-power", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--pretrained", type=str, default=None,
                    help="Path to v17 TextWorldModel checkpoint (loads compressor/expander weights)")
    ap.add_argument("--freeze-io", action="store_true",
                    help="Freeze compressor and expander (train only dynamics + mode embeddings)")
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

    print("=== v18 Text Dynamics (Identity + Q&A) ===")
    print(f"Device: {device}")
    print(f"Config: {args.config} (d={config.d_model}, h={config.n_heads})")
    print(f"BPE vocab: {tokenizer.vocab_size}")
    print(f"Bottleneck slots: {config.max_triples * 3}")

    dyn_layers = args.dynamics_layers if args.dynamics_layers is not None else config.n_layers
    model = TextDynamicsModel(
        config=config,
        domain_tokenizer=tokenizer,
        text_compressor_layers=args.text_compressor_layers,
        text_expander_layers=args.text_expander_layers,
        dynamics_layers=args.dynamics_layers,
        max_text_tokens=args.max_text_tokens,
        dropout=args.dropout,
        alpha_min=args.alpha_min,
    ).to(device)
    model.init_embeddings()

    # Load pretrained compressor/expander from v17 checkpoint
    if args.pretrained:
        print(f"\n  Loading pretrained weights from: {args.pretrained}")
        v17_state = torch.load(args.pretrained, map_location=device, weights_only=True)
        # v17 TextWorldModel keys: shared_token_emb.*, text_compressor.*, text_expander.*
        # These map directly to TextDynamicsModel (same prefixes)
        loaded = []
        skipped = []
        model_state = model.state_dict()
        for k, v in v17_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k.split(".")[0])
            else:
                skipped.append(k)
        model.load_state_dict(model_state)
        loaded_modules = sorted(set(loaded))
        print(f"  Loaded: {loaded_modules}")
        if skipped:
            print(f"  Skipped: {skipped}")

    # Freeze compressor and expander — train only dynamics + mode embeddings
    if args.freeze_io:
        for param in model.text_compressor.parameters():
            param.requires_grad = False
        for param in model.text_expander.parameters():
            param.requires_grad = False
        print("  Frozen: text_compressor, text_expander")

    dyn_params = sum(p.numel() for p in model.dynamics.parameters() if p.requires_grad)
    mode_params = model.mode_embeddings.numel()
    print(f"  Compressor: {model.text_compressor.trainable_param_count():,} params"
          f"{' (frozen)' if args.freeze_io else ''}")
    print(f"  Dynamics: {dyn_params:,} params ({dyn_layers}L)")
    print(f"  Mode embeddings: {mode_params:,} params")
    print(f"  Expander: {model.text_expander.trainable_param_count():,} params"
          f"{' (frozen)' if args.freeze_io else ''}")
    print(f"  Total: {model.param_count():,} ({model.trainable_param_count():,} trainable)")

    # Load paired Q&A dataset
    train_path = data_dir / "qa_train.jsonl"
    train_ds = TextPairDataset(
        train_path, tokenizer,
        max_text_tokens=args.max_text_tokens,
        max_examples=args.max_examples,
    )
    n_id = (train_ds._modes == 0).sum().item()
    n_qa = (train_ds._modes == 1).sum().item()
    print(f"  Train: {len(train_ds)} examples ({n_id} identity, {n_qa} Q&A)")

    test_path = data_dir / "qa_test.jsonl"
    eval_ds = train_ds
    if test_path.exists():
        eval_ds = TextPairDataset(test_path, tokenizer, max_text_tokens=args.max_text_tokens)
        n_id_e = (eval_ds._modes == 0).sum().item()
        n_qa_e = (eval_ds._modes == 1).sum().item()
        print(f"  Test: {len(eval_ds)} examples ({n_id_e} identity, {n_qa_e} Q&A)")

    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "architecture": "v18_dynamics",
            "config_profile": args.config,
            "text_compressor_layers": args.text_compressor_layers,
            "text_expander_layers": args.text_expander_layers,
            "dynamics_layers": dyn_layers,
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
            "pretrained": args.pretrained,
            "freeze_io": args.freeze_io,
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
    print("DONE -- v18 Text Dynamics")
    print(f"  Phase 1: best tok={p1:.4f} at epoch {p1_ep}")
    print(f"  Phase 2: best tok={p2:.4f} at epoch {p2_ep}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
