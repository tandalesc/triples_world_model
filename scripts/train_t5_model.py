#!/usr/bin/env python3
"""Train the hybrid TWM with T5 value decoder.

Entity/attr: discrete cross-entropy (small vocab, fast convergence)
Value: frozen T5 decoder generates free-text from projected TWM embeddings

Usage:
    uv run python scripts/train_t5_model.py \
        --data-dir data/atomic_small \
        --out-dir results/t5_test \
        --config base --epochs 50

    # With pretrained dynamics + encoder, unfreezing last 2 T5 layers
    uv run python scripts/train_t5_model.py \
        --data-dir data/atomic_2k \
        --out-dir results/t5_2k \
        --pretrained-dynamics pretrained/model_best.pt \
        --unfreeze-t5-layers 2 \
        --epochs 200
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.phrase_vocab import PhraseVocab
from twm.t5_model import HybridT5WorldModel
from twm.t5_dataset import T5TripleDataset


def make_encode_fn(model_name: str, device: torch.device, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name, device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    def encode(phrases: list[str]) -> torch.Tensor:
        return st_model.encode(
            phrases,
            batch_size=batch_size,
            show_progress_bar=len(phrases) > 1000,
            convert_to_tensor=True,
            device=str(device),
        )

    return encode, st_dim


def compute_loss(
    model: HybridT5WorldModel,
    latent: torch.Tensor,
    tgt_entity: torch.Tensor,
    tgt_attr: torch.Tensor,
    tgt_pad: torch.Tensor,
    value_token_ids: torch.Tensor,
    value_attention_mask: torch.Tensor,
    t5_start_token_id: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined discrete + T5 loss."""
    metrics = {}
    B, M = tgt_entity.shape

    # --- Discrete entity/attr loss ---
    discrete_logits = model.forward_discrete(latent)
    discrete_loss = torch.tensor(0.0, device=latent.device)

    for role, tgt in [("entity", tgt_entity), ("attr", tgt_attr)]:
        logits = discrete_logits[role]  # (B, M, V)
        V = logits.shape[-1]
        logits_flat = logits.reshape(-1, V)
        tgt_flat = tgt.reshape(-1)
        valid = ~tgt_pad.reshape(-1)
        if valid.any():
            role_loss = F.cross_entropy(logits_flat[valid], tgt_flat[valid], ignore_index=0)
            discrete_loss = discrete_loss + role_loss
            acc = (logits_flat[valid].argmax(-1) == tgt_flat[valid]).float().mean().item()
            metrics[f"acc_{role}"] = acc
            metrics[f"loss_{role}"] = role_loss.item()

    # --- T5 value loss ---
    seq_len = value_token_ids.shape[-1]
    val_ids_flat = value_token_ids.reshape(B * M, seq_len)
    val_mask_flat = value_attention_mask.reshape(B * M, seq_len)
    tgt_pad_flat = tgt_pad.reshape(B * M)

    valid_mask = ~tgt_pad_flat
    if valid_mask.any():
        val_ids_valid = val_ids_flat[valid_mask]
        val_mask_valid = val_mask_flat[valid_mask]

        # Build decoder input: prepend start token, shift right
        n_valid = val_ids_valid.shape[0]
        start_tok = torch.full(
            (n_valid, 1), t5_start_token_id,
            dtype=torch.long, device=latent.device,
        )
        decoder_input = torch.cat([start_tok, val_ids_valid[:, :-1]], dim=1)

        # Get full triple context (entity+attr+value) for valid triples
        triple_ctx = model._extract_triple_context(latent)  # (B, M, 3*d_model)
        ctx_flat = triple_ctx.reshape(B * M, -1)
        ctx_valid = ctx_flat[valid_mask]

        # T5 forward with full triple context
        value_logits = model.value_decoder(ctx_valid, decoder_input)

        # Cross-entropy over valid tokens
        value_logits_flat = value_logits.reshape(-1, value_logits.shape[-1])
        labels_flat = val_ids_valid.reshape(-1)
        mask_flat = val_mask_valid.reshape(-1).bool()

        if mask_flat.any():
            value_loss = F.cross_entropy(
                value_logits_flat[mask_flat],
                labels_flat[mask_flat],
                ignore_index=0,
            )
        else:
            value_loss = torch.tensor(0.0, device=latent.device)

        if mask_flat.any():
            val_preds = value_logits_flat[mask_flat].argmax(-1)
            val_acc = (val_preds == labels_flat[mask_flat]).float().mean().item()
            metrics["acc_value_tok"] = val_acc
        metrics["loss_value"] = value_loss.item()
    else:
        value_loss = torch.tensor(0.0, device=latent.device)

    total_loss = discrete_loss + value_loss
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


def build_gen_samples(
    model: HybridT5WorldModel,
    dataset: T5TripleDataset,
    device: torch.device,
    n_examples: int = 3,
) -> list[tuple[str, str]]:
    """Generate value text samples for qualitative inspection."""
    model.eval()
    n = min(len(dataset), n_examples)
    inp = dataset._all_inputs[:n].to(device)
    inp_pad = dataset._all_input_pad_masks[:n].to(device)
    tgt_pad = dataset._all_target_pad_masks[:n].to(device)
    val_ids = dataset._all_value_token_ids[:n].to(device)
    val_mask = dataset._all_value_attention_mask[:n].to(device)

    latent = model.encode_dynamics(inp, inp_pad)
    generated = model.generate_values(latent, max_length=32)

    tokenizer = model.value_decoder.tokenizer
    samples = []
    for i in range(n):
        M = model.config.max_triples
        for m in range(M):
            if not tgt_pad[i, m]:
                tgt_text = tokenizer.decode(
                    val_ids[i, m][val_mask[i, m].bool()],
                    skip_special_tokens=True,
                )
                pred_text = generated[i][m]
                samples.append((tgt_text, pred_text))
                if len(samples) >= 5:
                    return samples
    return samples


def main():
    ap = argparse.ArgumentParser(description="Train hybrid T5 TWM")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--t5-model", type=str, default="t5-small")
    ap.add_argument("--n-proj-tokens", type=int, default=8,
                    help="Number of projected tokens for T5 cross-attention")
    ap.add_argument("--unfreeze-t5-layers", type=int, default=0,
                    help="Unfreeze last N T5 decoder layers for fine-tuning")
    ap.add_argument("--pretrained-dynamics", type=str, default=None)
    ap.add_argument("--pretrained-sentence-model", type=str, default=None)
    ap.add_argument("--freeze-dynamics", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=10)
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

    print(f"Device: {device}")

    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(args.st_model, device)
    print(f"  ST dim: {st_dim}")

    config = ModelConfig.from_profile(args.config)
    print(f"  Config: {args.config} (d_model={config.d_model})")

    # Phrase vocab
    print("Building phrase vocabulary...")
    train_path = data_dir / "train.jsonl"
    train_examples = []
    with open(train_path) as f:
        for line in f:
            train_examples.append(json.loads(line))

    vocab = PhraseVocab()
    vocab.build(train_examples)
    for role, size in vocab.vocab_sizes.items():
        print(f"  {role}: {size} phrases")
    vocab.save(out_dir / "phrase_vocab.json")

    # Model
    print(f"Loading T5 decoder ({args.t5_model})...")
    model = HybridT5WorldModel(
        config, st_dim, vocab,
        t5_model_name=args.t5_model,
        n_proj_tokens=args.n_proj_tokens,
        unfreeze_last_n=args.unfreeze_t5_layers,
    ).to(device)

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...")
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)
    if args.pretrained_sentence_model:
        print(f"Loading encoder from {args.pretrained_sentence_model}...")
        model.load_encoder_from_sentence_model(args.pretrained_sentence_model)
    if args.freeze_dynamics:
        model.freeze_dynamics()
    if args.freeze_encoder:
        model.freeze_encoder()

    t5_frozen = model.value_decoder.frozen_param_count()
    t5_trainable = model.value_decoder.trainable_param_count()
    print(f"  T5 decoder: {t5_frozen:,} frozen, {t5_trainable:,} trainable (projection)")
    print(f"  Total: {model.param_count():,} params, {model.trainable_param_count():,} trainable", flush=True)

    # Dataset
    print("Building datasets...")
    t5_tokenizer = model.value_decoder.tokenizer
    train_ds = T5TripleDataset(
        train_path, encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
    )
    print(f"  Train: {len(train_ds)} examples, value_seq_len={train_ds.value_seq_len}")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = T5TripleDataset(
            test_path, encode_fn, vocab, t5_tokenizer,
            max_triples=config.max_triples,
        )
        print(f"  Test: {len(test_ds)} examples")

    config.save(out_dir / "config.json")
    with open(out_dir / "st_config.json", "w") as f:
        json.dump({
            "st_model": args.st_model, "st_dim": st_dim,
            "t5_model": args.t5_model,
            "n_proj_tokens": args.n_proj_tokens,
            "unfreeze_t5_layers": args.unfreeze_t5_layers,
        }, f, indent=2)

    # Optimizer
    param_groups = []
    proj_params = list(model.value_decoder.projection.parameters())
    param_groups.append({"params": proj_params, "lr": args.lr})

    head_params = (
        list(model.ln_f.parameters())
        + list(model.entity_head.parameters())
        + list(model.attr_head.parameters())
    )
    param_groups.append({"params": head_params, "lr": args.lr})

    t5_train_params = [
        p for p in model.value_decoder.t5_decoder.parameters() if p.requires_grad
    ]
    if t5_train_params:
        param_groups.append({"params": t5_train_params, "lr": args.lr * 0.1})

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        enc_lr = args.lr * 0.1 if args.pretrained_sentence_model else args.lr
        param_groups.append({"params": enc_params, "lr": enc_lr})

    dyn_params = [p for p in model.dynamics.parameters() if p.requires_grad]
    if dyn_params:
        dyn_lr = args.lr * 0.1 if args.pretrained_dynamics else args.lr
        param_groups.append({"params": dyn_params, "lr": dyn_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t5_start_id = model.value_decoder.t5_config.decoder_start_token_id

    # Move data to device
    all_inputs = train_ds._all_inputs.to(device)
    all_input_pads = train_ds._all_input_pad_masks.to(device)
    all_tgt_e = train_ds._all_target_entity.to(device)
    all_tgt_a = train_ds._all_target_attr.to(device)
    all_tgt_pads = train_ds._all_target_pad_masks.to(device)
    all_val_ids = train_ds._all_value_token_ids.to(device)
    all_val_mask = train_ds._all_value_attention_mask.to(device)
    n_train = all_inputs.shape[0]

    best_test_loss = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]

            latent = model.encode_dynamics(all_inputs[idx], all_input_pads[idx])
            loss, _ = compute_loss(
                model, latent,
                all_tgt_e[idx], all_tgt_a[idx], all_tgt_pads[idx],
                all_val_ids[idx], all_val_mask[idx],
                t5_start_id,
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
            train_metrics = build_gen_samples(model, train_ds, device)
            # Also compute loss metrics
            model.eval()
            n_ev = min(len(train_ds), 100)
            with torch.no_grad():
                ev_latent = model.encode_dynamics(
                    train_ds._all_inputs[:n_ev].to(device),
                    train_ds._all_input_pad_masks[:n_ev].to(device),
                )
                _, ev_metrics = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_entity[:n_ev].to(device),
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    train_ds._all_value_attention_mask[:n_ev].to(device),
                    t5_start_id,
                )

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | acc_e {ev_metrics.get('acc_entity', 0):.3f}"
                f" | acc_a {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val_tok {ev_metrics.get('acc_value_tok', 0):.3f}"
            )

            test_metrics = None
            if test_ds is not None:
                n_tev = min(len(test_ds), 100)
                with torch.no_grad():
                    tev_latent = model.encode_dynamics(
                        test_ds._all_inputs[:n_tev].to(device),
                        test_ds._all_input_pad_masks[:n_tev].to(device),
                    )
                    _, test_metrics = compute_loss(
                        model, tev_latent,
                        test_ds._all_target_entity[:n_tev].to(device),
                        test_ds._all_target_attr[:n_tev].to(device),
                        test_ds._all_target_pad_masks[:n_tev].to(device),
                        test_ds._all_value_token_ids[:n_tev].to(device),
                        test_ds._all_value_attention_mask[:n_tev].to(device),
                        t5_start_id,
                    )

                log += (
                    f" || test_val_tok {test_metrics.get('acc_value_tok', 0):.3f}"
                    f" | test_loss {test_metrics['loss_total']:.3f}"
                )
                if test_metrics["loss_total"] < best_test_loss:
                    best_test_loss = test_metrics["loss_total"]
                    torch.save(model.state_dict(), out_dir / "model_best.pt")

            print(log, flush=True)

            # Show generated samples
            if train_metrics:
                for tgt, pred in train_metrics[:3]:
                    print(f"    target: {tgt!r}  ->  pred: {pred!r}", flush=True)

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                **({f"train_{k}": v for k, v in ev_metrics.items()}),
                **({f"test_{k}": v for k, v in test_metrics.items()} if test_metrics else {}),
            })

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    if best_test_loss < float("inf"):
        print(f"Best test loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    main()
