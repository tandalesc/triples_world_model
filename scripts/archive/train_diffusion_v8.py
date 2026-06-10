#!/usr/bin/env python3
"""Train diffusion TWM v8: fine-tune with pure mask_ratio=1.0.

Phase 2 of conditioning training. Starting from a v7 adaLN checkpoint,
fine-tune with every batch at mask_ratio=1.0. All tokens masked. The only
signal for prediction is TWM conditioning via cross-attention and adaLN.

Two modes:
  --conditioning-only: Freeze self-attention + FFN, train only adaLN projections,
                       cross-attention, projection MLPs, and output head.
  (default): Full fine-tune of all denoiser parameters.

Usage:
    uv run python scripts/train_diffusion_v8.py \
        --data-dir data/atomic_10000 \
        --out-dir results/v8_pure1.0_condonly_lr0.1x \
        --v7-checkpoint results/v7_adaln/model_best.pt \
        --v7-model-dir results/v7_adaln \
        --conditioning-only \
        --lr 3e-5 \
        --epochs 200 --patience 50 --log-every 5
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.t5_dataset import T5TripleDataset


def make_encode_fn(model_name: str, device: torch.device, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name, device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    def encode(phrases: list[str]) -> torch.Tensor:
        return st_model.encode(
            phrases, batch_size=batch_size,
            show_progress_bar=len(phrases) > 1000,
            convert_to_tensor=True, device=str(device),
        )
    return encode, st_dim


def _diffusion_loss(logits, diff_mask, target_ids, pad_mask, B, M):
    if logits.shape[0] == 0 or not diff_mask.any():
        return torch.tensor(0.0, device=logits.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]

    logits_at_mask = logits[diff_mask]
    targets_at_mask = tgt_valid[diff_mask]

    loss = F.cross_entropy(logits_at_mask, targets_at_mask)

    metrics = {}
    preds = logits_at_mask.argmax(-1)
    non_pad = targets_at_mask != 0
    if non_pad.any():
        acc = (preds[non_pad] == targets_at_mask[non_pad]).float().mean().item()
        metrics["acc"] = acc
    metrics["loss"] = loss.item()
    return loss, metrics


def compute_loss(
    model: DiffusionWorldModel,
    latent: torch.Tensor,
    tgt_attr: torch.Tensor,
    tgt_pad: torch.Tensor,
    entity_token_ids: torch.Tensor,
    value_token_ids: torch.Tensor,
    mask_ratio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = {}
    B, M = tgt_attr.shape

    discrete_logits = model.forward_discrete(latent)
    attr_logits = discrete_logits["attr"]
    V = attr_logits.shape[-1]
    logits_flat = attr_logits.reshape(-1, V)
    tgt_flat = tgt_attr.reshape(-1)
    valid = ~tgt_pad.reshape(-1)
    attr_loss = torch.tensor(0.0, device=latent.device)
    if valid.any():
        attr_loss = F.cross_entropy(logits_flat[valid], tgt_flat[valid], ignore_index=0)
        acc = (logits_flat[valid].argmax(-1) == tgt_flat[valid]).float().mean().item()
        metrics["acc_attr"] = acc

    entity_logits, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, mask_ratio=mask_ratio,
    )
    entity_loss, entity_metrics = _diffusion_loss(
        entity_logits, entity_mask, entity_token_ids, tgt_pad, B, M,
    )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]

    value_logits, value_mask = model.forward_value(
        latent, value_token_ids, tgt_pad, mask_ratio=mask_ratio,
    )
    value_loss, value_metrics = _diffusion_loss(
        value_logits, value_mask, value_token_ids, tgt_pad, B, M,
    )
    if "acc" in value_metrics:
        metrics["acc_value_tok"] = value_metrics["acc"]

    total_loss = attr_loss + entity_loss + value_loss
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


@torch.no_grad()
def run_eval_generation(
    model: DiffusionWorldModel,
    ds,
    device: torch.device,
    t5_tokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
) -> dict[str, float]:
    """Run generation-from-scratch with both token-level and exact-match metrics."""
    n = min(n_examples, len(ds))

    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    gen_entities = model.generate_entities(latent, n_steps=n_steps)
    gen_values = model.generate_values(latent, n_steps=n_steps)

    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    ent_exact_match = 0
    val_exact_match = 0
    attr_match = 0
    ent_tok_correct = 0
    ent_tok_total = 0
    val_tok_correct = 0
    val_tok_total = 0
    total = 0
    all_pred_values = []
    all_pred_entities = []

    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total += 1

            if pred_attrs[i, m].item() == tgt_attrs[i, m].item():
                attr_match += 1

            tgt_e_ids = ds._all_entity_token_ids[i, m]
            tgt_e_str = t5_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e_str = gen_entities[i][m]
            if pred_e_str == tgt_e_str:
                ent_exact_match += 1
            all_pred_entities.append(pred_e_str)

            pred_e_tok = t5_tokenizer(
                pred_e_str, padding="max_length", truncation=True,
                max_length=tgt_e_ids.shape[0], return_tensors="pt",
            )["input_ids"][0]
            non_pad = tgt_e_ids != 0
            if non_pad.any():
                match = (pred_e_tok[:len(tgt_e_ids)][non_pad] == tgt_e_ids[non_pad]).sum().item()
                ent_tok_correct += match
                ent_tok_total += non_pad.sum().item()

            tgt_v_ids = ds._all_value_token_ids[i, m]
            tgt_v_str = t5_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v_str = gen_values[i][m]
            if pred_v_str == tgt_v_str:
                val_exact_match += 1
            all_pred_values.append(pred_v_str)

            pred_v_tok = t5_tokenizer(
                pred_v_str, padding="max_length", truncation=True,
                max_length=tgt_v_ids.shape[0], return_tensors="pt",
            )["input_ids"][0]
            non_pad = tgt_v_ids != 0
            if non_pad.any():
                match = (pred_v_tok[:len(tgt_v_ids)][non_pad] == tgt_v_ids[non_pad]).sum().item()
                val_tok_correct += match
                val_tok_total += non_pad.sum().item()

    if total == 0:
        return {}

    unique_values = len(set(all_pred_values))
    unique_entities = len(set(all_pred_entities))
    val_counter = Counter(all_pred_values)
    top_value_count = val_counter.most_common(1)[0][1] if val_counter else 0

    return {
        "gen_ent_tok": ent_tok_correct / max(ent_tok_total, 1),
        "gen_ent_exact": ent_exact_match / total,
        "gen_val_tok": val_tok_correct / max(val_tok_total, 1),
        "gen_val_exact": val_exact_match / total,
        "gen_attr": attr_match / total,
        "gen_total": total,
        "unique_values": unique_values,
        "unique_entities": unique_entities,
        "top_value_count": top_value_count,
    }


@torch.no_grad()
def run_loss_vs_mask_ratio(
    model: DiffusionWorldModel,
    ds,
    device: torch.device,
    n_examples: int = 64,
    mask_ratios: list[float] | None = None,
) -> dict[str, float]:
    """Compute loss at multiple mask ratios on held-out data.

    Returns dict like {"mr_0.0_val": 1.23, "mr_0.0_ent": 0.45, ...}
    """
    if mask_ratios is None:
        mask_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    n = min(n_examples, len(ds))

    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    B = latent.shape[0]
    M = model.config.max_triples
    tgt_pad = ds._all_target_pad_masks[:n].to(device)
    ent_ids = ds._all_entity_token_ids[:n].to(device)
    val_ids = ds._all_value_token_ids[:n].to(device)

    results = {}
    for mr in mask_ratios:
        if mr == 0.0:
            # mask_ratio=0 means clamp to min=1 token per seq, effectively ~1/S
            mr_tensor = torch.zeros(B, device=device)
        else:
            mr_tensor = torch.full((B,), mr, device=device)

        # Value loss
        val_logits, val_mask = model.forward_value(latent, val_ids, tgt_pad, mask_ratio=mr_tensor)
        if val_logits.shape[0] > 0 and val_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = val_ids.reshape(B * M, -1)[valid]
            val_loss = F.cross_entropy(
                val_logits[val_mask], tgt_valid[val_mask],
            ).item()
        else:
            val_loss = 0.0

        # Entity loss
        ent_logits, ent_mask = model.forward_entity(latent, ent_ids, tgt_pad, mask_ratio=mr_tensor)
        if ent_logits.shape[0] > 0 and ent_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = ent_ids.reshape(B * M, -1)[valid]
            ent_loss = F.cross_entropy(
                ent_logits[ent_mask], tgt_valid[ent_mask],
            ).item()
        else:
            ent_loss = 0.0

        results[f"mr_{mr:.1f}_val"] = val_loss
        results[f"mr_{mr:.1f}_ent"] = ent_loss

    return results


def freeze_denoiser_core(model: DiffusionWorldModel):
    """Freeze self-attention and FFN in denoiser layers. Keep conditioning trainable.

    Trainable after freeze:
      - adaln_proj (adaLN conditioning projections)
      - cross_attn (cross-attention IS conditioning)
      - projection (cross-attention projection MLP)
      - output_head, ln_f
    Frozen:
      - self_attn, ffn in each layer
      - token_emb, pos_emb
    """
    for decoder in [model.entity_decoder, model.value_decoder]:
        # Freeze token/position embeddings
        decoder.token_emb.requires_grad_(False)
        decoder.pos_emb.requires_grad_(False)

        for layer in decoder.layers:
            # Freeze self-attention
            for p in layer.self_attn.parameters():
                p.requires_grad = False
            # Freeze FFN
            for p in layer.ffn.parameters():
                p.requires_grad = False
            # adaln_proj stays trainable (default)
            # cross_attn stays trainable (default)


def main():
    ap = argparse.ArgumentParser(description="Train diffusion TWM v8 (pure mask_ratio=1.0 fine-tune)")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--v7-checkpoint", type=str, required=True,
                    help="Path to v7 model_best.pt checkpoint")
    ap.add_argument("--v7-model-dir", type=str, required=True,
                    help="Path to v7 model directory (for config files)")
    ap.add_argument("--conditioning-only", action="store_true",
                    help="Freeze self-attn + FFN, train only conditioning pathway")
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--st-model", type=str, default=None)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--patience", type=int, default=50,
                    help="Early stopping patience on gen_val_tok")
    ap.add_argument("--diagnostic-every", type=int, default=50,
                    help="Run loss-vs-mask_ratio diagnostic every N epochs (0 to disable)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v7_model_dir = Path(args.v7_model_dir)
    v7_ckpt_path = Path(args.v7_checkpoint)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load v7 model config
    with open(v7_model_dir / "model_config.json") as f:
        mcfg = json.load(f)

    st_model_name = args.st_model or mcfg["st_model"]
    use_cross_attention = mcfg.get("use_cross_attention", True)
    use_adaln = mcfg.get("use_adaln", False)
    use_film = mcfg.get("use_film", False)

    print(f"Device: {device}")
    print(f"Mode: {'conditioning-only' if args.conditioning_only else 'full fine-tune'}")
    print(f"LR: {args.lr} (v7 base was 3e-4)")
    print(f"adaLN: {use_adaln}, Cross-attention: {use_cross_attention}, FiLM: {use_film}")
    print(f"mask_ratio: 1.0 (pure, every batch)")
    print(f"Early stopping: gen_val_tok, patience={args.patience}, max={args.epochs}")

    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(st_model_name, device)
    print(f"  ST dim: {st_dim}")

    config = ModelConfig.from_profile(args.config)
    print(f"  Config: {args.config} (d_model={config.d_model})")

    # Load vocab from v7 model dir
    vocab = PhraseVocab.load(v7_model_dir / "phrase_vocab.json")
    for role, size in vocab.vocab_sizes.items():
        print(f"  {role}: {size} phrases")
    vocab.save(out_dir / "phrase_vocab.json")

    print("Building diffusion model...")
    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    MASK_TOKEN_ID = 32099

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=mcfg["max_value_tokens"],
        n_proj_tokens=mcfg["n_proj_tokens"],
        denoiser_layers=mcfg["denoiser_layers"],
        denoiser_dim=mcfg["denoiser_dim"],
        denoiser_heads=mcfg["denoiser_heads"],
        dropout=mcfg.get("dropout", 0.1),
        mask_token_id=MASK_TOKEN_ID,
        tokenizer=t5_tokenizer,
        use_film=use_film,
        use_cross_attention=use_cross_attention,
        use_adaln=use_adaln,
        use_continuous_noise=mcfg.get("use_continuous_noise", False),
        normalize_noise=mcfg.get("normalize_noise", True),
        alpha_min=mcfg.get("alpha_min", 0.0),
        timestep_bias_power=mcfg.get("timestep_bias_power", 1.0),
        unified_decoder=mcfg.get("unified_decoder", False),
        wspace=mcfg.get("wspace", False),
        use_structured_noise=mcfg.get("structured_noise", False),
        use_mse_prediction=mcfg.get("use_mse_prediction", False),
        cond_drop_prob=mcfg.get("cond_drop_prob", 0.0),
        use_decode_proj=mcfg.get("use_decode_proj", False),
    ).to(device)

    # Load v7 checkpoint
    print(f"Loading v7 checkpoint from {v7_ckpt_path}...")
    sd = torch.load(v7_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)

    # Freeze dynamics + encoder (always frozen for diffusion fine-tuning)
    model.freeze_dynamics()
    model.freeze_encoder()

    # Optionally freeze denoiser core (self-attn + FFN)
    if args.conditioning_only:
        freeze_denoiser_core(model)
        print("  Frozen: self-attention + FFN in denoiser layers")
        print("  Trainable: adaLN projections, cross-attention, projection MLPs, output head")

    trainable = model.trainable_param_count()
    total = model.param_count()
    print(f"  Total: {total:,} params, {trainable:,} trainable", flush=True)

    print("Building datasets...")
    train_path = data_dir / "train.jsonl"
    train_ds = T5TripleDataset(
        train_path, encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=mcfg["max_value_tokens"],
    )
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = T5TripleDataset(
            test_path, encode_fn, vocab, t5_tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=mcfg["max_value_tokens"],
        )
        print(f"  Test: {len(test_ds)} examples")

    # Save configs
    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            **mcfg,
            "v8_lr": args.lr,
            "v8_conditioning_only": args.conditioning_only,
            "v8_v7_checkpoint": str(v7_ckpt_path),
            "v8_mask_ratio": 1.0,
        }, f, indent=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    all_inputs = train_ds._all_inputs.to(device)
    all_input_pads = train_ds._all_input_pad_masks.to(device)
    all_tgt_a = train_ds._all_target_attr.to(device)
    all_tgt_pads = train_ds._all_target_pad_masks.to(device)
    all_ent_ids = train_ds._all_entity_token_ids.to(device)
    all_val_ids = train_ds._all_value_token_ids.to(device)
    n_train = all_inputs.shape[0]

    best_gen_val_tok = -1.0
    best_gen_epoch = 0
    epochs_without_improvement = 0
    history = []

    # Run initial diagnostic before any training
    if test_ds is not None and args.diagnostic_every > 0:
        model.train(False)
        print("\n--- Loss-vs-mask_ratio diagnostic (epoch 0) ---")
        diag = run_loss_vs_mask_ratio(model, test_ds, device, n_examples=64)
        mr_line = "  ".join(f"mr={k.split('_')[1]}: v={diag[k]:.3f} e={diag[k.replace('_val','_ent')]:.3f}"
                           for k in sorted(diag.keys()) if k.endswith("_val"))
        print(f"  {mr_line}")
        history.append({"epoch": 0, "diagnostic": diag})

    print(f"\nFine-tuning for up to {args.epochs} epochs (pure mask_ratio=1.0)...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            B = idx.shape[0]

            latent = model.encode_dynamics(all_inputs[idx], all_input_pads[idx])

            # Pure mask_ratio=1.0 — every token masked
            mask_ratio = torch.ones(B, device=device)

            loss, _ = compute_loss(
                model, latent,
                all_tgt_a[idx], all_tgt_pads[idx],
                all_ent_ids[idx], all_val_ids[idx],
                mask_ratio=mask_ratio,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == 1:
            model.train(False)

            # Train metrics (at mask_ratio=1.0)
            n_ev = min(len(train_ds), 100)
            with torch.no_grad():
                ev_latent = model.encode_dynamics(
                    train_ds._all_inputs[:n_ev].to(device),
                    train_ds._all_input_pad_masks[:n_ev].to(device),
                )
                mr_ev = torch.ones(n_ev, device=device)
                _, ev_metrics = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    mask_ratio=mr_ev,
                )

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent {ev_metrics.get('acc_entity_tok', 0):.3f}"
                f" | attr {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val {ev_metrics.get('acc_value_tok', 0):.3f}"
            )

            test_metrics = None
            gen_metrics = {}
            if test_ds is not None:
                # Test loss at mask_ratio=1.0
                n_tev = min(len(test_ds), 100)
                with torch.no_grad():
                    tev_latent = model.encode_dynamics(
                        test_ds._all_inputs[:n_tev].to(device),
                        test_ds._all_input_pad_masks[:n_tev].to(device),
                    )
                    mr_tev = torch.ones(n_tev, device=device)
                    _, test_metrics = compute_loss(
                        model, tev_latent,
                        test_ds._all_target_attr[:n_tev].to(device),
                        test_ds._all_target_pad_masks[:n_tev].to(device),
                        test_ds._all_entity_token_ids[:n_tev].to(device),
                        test_ds._all_value_token_ids[:n_tev].to(device),
                        mask_ratio=mr_tev,
                    )

                # Generation scoring (uses iterative unmasking, the real test)
                gen_metrics = run_eval_generation(
                    model, test_ds, device, t5_tokenizer,
                    n_examples=64, n_steps=args.denoise_steps,
                )

                gvt = gen_metrics.get("gen_val_tok", 0)
                gve = gen_metrics.get("gen_val_exact", 0)
                get_ = gen_metrics.get("gen_ent_tok", 0)
                gee = gen_metrics.get("gen_ent_exact", 0)

                log += (
                    f" || t_loss {test_metrics['loss_total']:.3f}"
                    f" t_val {test_metrics.get('acc_value_tok', 0):.3f}"
                    f" t_ent {test_metrics.get('acc_entity_tok', 0):.3f}"
                    f" | gv_tok {gvt:.3f} gv_ex {gve:.3f}"
                    f" | ge_tok {get_:.3f} ge_ex {gee:.3f}"
                    f" | g_attr {gen_metrics.get('gen_attr', 0):.3f}"
                    f" | u_v {gen_metrics.get('unique_values', 0)}"
                    f" u_e {gen_metrics.get('unique_entities', 0)}"
                    f" top1 {gen_metrics.get('top_value_count', 0)}"
                )

                # Early stopping on gen_val_tok
                if gvt > best_gen_val_tok:
                    best_gen_val_tok = gvt
                    best_gen_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), out_dir / "model_best.pt")
                    log += " *"
                else:
                    epochs_without_improvement += args.log_every

            print(log, flush=True)

            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in ev_metrics.items()},
            }
            if test_metrics:
                entry.update({f"test_{k}": v for k, v in test_metrics.items()})
            if gen_metrics:
                entry.update(gen_metrics)

            # Loss-vs-mask_ratio diagnostic
            if (args.diagnostic_every > 0 and test_ds is not None
                    and epoch % args.diagnostic_every == 0):
                diag = run_loss_vs_mask_ratio(model, test_ds, device, n_examples=64)
                entry["diagnostic"] = diag
                mr_line = "  ".join(
                    f"mr={k.split('_')[1]}: v={diag[k]:.3f} e={diag[k.replace('_val','_ent')]:.3f}"
                    for k in sorted(diag.keys()) if k.endswith("_val")
                )
                print(f"  DIAG: {mr_line}", flush=True)

            history.append(entry)

            if epochs_without_improvement >= args.patience and test_ds is not None:
                print(f"\nEarly stopping at epoch {epoch} (best gen_val_tok {best_gen_val_tok:.4f} at epoch {best_gen_epoch})", flush=True)
                break

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    print(f"Best gen_val_tok: {best_gen_val_tok:.4f} at epoch {best_gen_epoch}")
    print(f"Stopped at epoch {epoch}")


if __name__ == "__main__":
    main()
