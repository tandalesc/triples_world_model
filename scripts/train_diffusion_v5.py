#!/usr/bin/env python3
"""Train the hybrid TWM with masked diffusion decoders.

Entity: masked discrete diffusion (open vocabulary via T5 tokens)
Attr: discrete cross-entropy (small closed vocab)
Value: masked discrete diffusion -- random masking + denoising prediction

Usage:
    uv run python scripts/train_diffusion_model.py \
        --data-dir data/atomic_2k \
        --out-dir results/diffusion_2k \
        --config base --epochs 300
"""

import argparse
import json
from pathlib import Path

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


def sample_mask_ratio(B: int, device: torch.device, beta_a: float, beta_b: float) -> torch.Tensor:
    """Sample mask ratios from Beta(a,b) distribution."""
    if beta_a == 1.0 and beta_b == 1.0:
        return torch.rand(B, device=device)
    dist = torch.distributions.Beta(beta_a, beta_b)
    return dist.sample((B,)).to(device)


def _diffusion_loss(logits, diff_mask, target_ids, pad_mask, B, M):
    """Compute CE loss on masked positions for a diffusion decoder."""
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

    # --- Discrete attr loss ---
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

    # --- Diffusion entity loss ---
    entity_logits, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, mask_ratio=mask_ratio,
    )
    entity_loss, entity_metrics = _diffusion_loss(
        entity_logits, entity_mask, entity_token_ids, tgt_pad, B, M,
    )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]

    # --- Diffusion value loss ---
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
def eval_generation(
    model: DiffusionWorldModel,
    ds,
    device: torch.device,
    t5_tokenizer,
    n_examples: int = 32,
    n_steps: int = 10,
) -> dict[str, float]:
    """Evaluate generation-from-scratch quality.

    Runs full denoising loop (mask_ratio=1.0) and measures token accuracy
    against ground truth. This directly measures inference-quality generation.
    """
    model.eval()
    n = min(n_examples, len(ds))

    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    gen_entities = model.generate_entities(latent, n_steps=n_steps)
    gen_values = model.generate_values(latent, n_steps=n_steps)

    # Discrete attr
    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    # Score entity/value by re-tokenizing predictions and comparing
    entity_match = 0
    value_match = 0
    attr_match = 0
    total = 0

    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total += 1

            # Attr
            if pred_attrs[i, m].item() == tgt_attrs[i, m].item():
                attr_match += 1

            # Entity: compare decoded strings
            tgt_e_ids = ds._all_entity_token_ids[i, m]
            tgt_e = t5_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e = gen_entities[i][m]
            if pred_e == tgt_e:
                entity_match += 1

            # Value: compare decoded strings
            tgt_v_ids = ds._all_value_token_ids[i, m]
            tgt_v = t5_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v = gen_values[i][m]
            if pred_v == tgt_v:
                value_match += 1

    if total == 0:
        return {}
    return {
        "gen_ent": entity_match / total,
        "gen_attr": attr_match / total,
        "gen_val": value_match / total,
        "gen_total": total,
    }


def main():
    ap = argparse.ArgumentParser(description="Train diffusion TWM")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--pretrained-dynamics", type=str, default=None)
    ap.add_argument("--pretrained-sentence-model", type=str, default=None)
    ap.add_argument("--freeze-dynamics", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--n-proj-tokens", type=int, default=8)
    ap.add_argument("--denoiser-layers", type=int, default=4)
    ap.add_argument("--denoiser-dim", type=int, default=512)
    ap.add_argument("--denoiser-heads", type=int, default=8)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--max-value-tokens", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    # Masking schedule
    ap.add_argument("--mask-beta-a", type=float, default=2.0,
                    help="Beta distribution alpha for mask ratio (1.0 = uniform)")
    ap.add_argument("--mask-beta-b", type=float, default=1.0,
                    help="Beta distribution beta for mask ratio (1.0 = uniform)")
    # Early stopping
    ap.add_argument("--patience", type=int, default=50,
                    help="Early stopping patience (epochs without test loss improvement)")
    # FiLM conditioning
    ap.add_argument("--use-film", action="store_true",
                    help="Enable FiLM conditioning injection")
    ap.add_argument("--no-cross-attention", action="store_true",
                    help="Disable cross-attention (FiLM-only mode)")
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

    use_cross_attention = not args.no_cross_attention

    print(f"Device: {device}")
    print(f"Mask schedule: Beta({args.mask_beta_a}, {args.mask_beta_b})")
    print(f"FiLM: {args.use_film}, Cross-attention: {use_cross_attention}")

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
    print("Building diffusion model...")
    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    MASK_TOKEN_ID = 32099  # T5's <extra_id_0>

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=args.max_value_tokens,
        n_proj_tokens=args.n_proj_tokens,
        denoiser_layers=args.denoiser_layers,
        denoiser_dim=args.denoiser_dim,
        denoiser_heads=args.denoiser_heads,
        dropout=args.dropout,
        mask_token_id=MASK_TOKEN_ID,
        tokenizer=t5_tokenizer,
        use_film=args.use_film,
        use_cross_attention=use_cross_attention,
    ).to(device)

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...")
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)
    encoder_ckpt = args.pretrained_sentence_model or args.pretrained_dynamics
    if encoder_ckpt:
        print(f"Loading encoder from {encoder_ckpt}...")
        model.load_encoder_from_sentence_model(encoder_ckpt)
    if args.freeze_dynamics:
        model.freeze_dynamics()
    if args.freeze_encoder:
        model.freeze_encoder()

    print(f"  Denoiser: {args.denoiser_layers}L, {args.denoiser_dim}d, {args.denoiser_heads}H, dropout={args.dropout}")
    print(f"  FiLM: {args.use_film}, Cross-attn: {use_cross_attention}, Proj tokens: {args.n_proj_tokens}")
    print(f"  Total: {model.param_count():,} params, {model.trainable_param_count():,} trainable", flush=True)

    # Dataset
    print("Building datasets...")
    train_ds = T5TripleDataset(
        train_path, encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=args.max_value_tokens,
    )
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = T5TripleDataset(
            test_path, encode_fn, vocab, t5_tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=args.max_value_tokens,
        )
        print(f"  Test: {len(test_ds)} examples")

    # Save configs
    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "st_model": args.st_model, "st_dim": st_dim,
            "n_proj_tokens": args.n_proj_tokens,
            "denoiser_layers": args.denoiser_layers,
            "denoiser_dim": args.denoiser_dim,
            "denoiser_heads": args.denoiser_heads,
            "max_value_tokens": args.max_value_tokens,
            "denoise_steps": args.denoise_steps,
            "mask_beta_a": args.mask_beta_a,
            "mask_beta_b": args.mask_beta_b,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "use_film": args.use_film,
            "use_cross_attention": use_cross_attention,
        }, f, indent=2)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Move data to device
    all_inputs = train_ds._all_inputs.to(device)
    all_input_pads = train_ds._all_input_pad_masks.to(device)
    all_tgt_a = train_ds._all_target_attr.to(device)
    all_tgt_pads = train_ds._all_target_pad_masks.to(device)
    all_ent_ids = train_ds._all_entity_token_ids.to(device)
    all_val_ids = train_ds._all_value_token_ids.to(device)
    n_train = all_inputs.shape[0]

    best_test_loss = float("inf")
    best_test_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            B_actual = idx.shape[0]

            # Sample mask ratio from Beta distribution
            mask_ratio = sample_mask_ratio(B_actual, device, args.mask_beta_a, args.mask_beta_b)

            latent = model.encode_dynamics(all_inputs[idx], all_input_pads[idx])
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
            model.eval()
            n_ev = min(len(train_ds), 100)
            with torch.no_grad():
                ev_latent = model.encode_dynamics(
                    train_ds._all_inputs[:n_ev].to(device),
                    train_ds._all_input_pad_masks[:n_ev].to(device),
                )
                _, ev_metrics = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                )

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent_tok {ev_metrics.get('acc_entity_tok', 0):.3f}"
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
                        test_ds._all_target_attr[:n_tev].to(device),
                        test_ds._all_target_pad_masks[:n_tev].to(device),
                        test_ds._all_entity_token_ids[:n_tev].to(device),
                        test_ds._all_value_token_ids[:n_tev].to(device),
                    )

                # Generation-from-scratch eval
                gen_metrics = eval_generation(
                    model, test_ds, device, t5_tokenizer,
                    n_examples=32, n_steps=args.denoise_steps,
                )

                log += (
                    f" || t_ent {test_metrics.get('acc_entity_tok', 0):.3f}"
                    f" | t_a {test_metrics.get('acc_attr', 0):.3f}"
                    f" | t_val {test_metrics.get('acc_value_tok', 0):.3f}"
                    f" | t_loss {test_metrics['loss_total']:.3f}"
                    f" | gen_e {gen_metrics.get('gen_ent', 0):.3f}"
                    f" | gen_a {gen_metrics.get('gen_attr', 0):.3f}"
                    f" | gen_v {gen_metrics.get('gen_val', 0):.3f}"
                )

                # Early stopping on test loss
                if test_metrics["loss_total"] < best_test_loss:
                    best_test_loss = test_metrics["loss_total"]
                    best_test_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), out_dir / "model_best.pt")
                else:
                    epochs_without_improvement += args.log_every

            print(log, flush=True)

            # Generate samples periodically
            if epoch % (args.log_every * 5) == 0 or epoch == 1:
                n_gen = min(3, len(train_ds))
                with torch.no_grad():
                    gen_latent = model.encode_dynamics(
                        train_ds._all_inputs[:n_gen].to(device),
                        train_ds._all_input_pad_masks[:n_gen].to(device),
                    )
                    gen_entities = model.generate_entities(gen_latent, n_steps=args.denoise_steps)
                    gen_values = model.generate_values(gen_latent, n_steps=args.denoise_steps)
                tgt_pad = train_ds._all_target_pad_masks[:n_gen]
                for i in range(n_gen):
                    M = config.max_triples
                    for m in range(M):
                        if not tgt_pad[i, m]:
                            tgt_e_ids = train_ds._all_entity_token_ids[i, m]
                            tgt_v_ids = train_ds._all_value_token_ids[i, m]
                            tgt_e = t5_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
                            tgt_v = t5_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
                            pred_e = gen_entities[i][m]
                            pred_v = gen_values[i][m]
                            print(f"    [{i},{m}] e: {tgt_e!r}->{pred_e!r}  v: {tgt_v!r}->{pred_v!r}", flush=True)
                            break

            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in ev_metrics.items()},
            }
            if test_metrics:
                entry.update({f"test_{k}": v for k, v in test_metrics.items()})
            if gen_metrics:
                entry.update(gen_metrics)
            history.append(entry)

            # Check early stopping
            if epochs_without_improvement >= args.patience and test_ds is not None:
                print(f"\nEarly stopping at epoch {epoch} (best test loss {best_test_loss:.4f} at epoch {best_test_epoch})", flush=True)
                break

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    print(f"Best test loss: {best_test_loss:.4f} at epoch {best_test_epoch}")
    print(f"Stopped at epoch {epoch}")


if __name__ == "__main__":
    main()
