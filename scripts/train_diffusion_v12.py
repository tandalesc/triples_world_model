#!/usr/bin/env python3
"""Train diffusion TWM v12: W-space decoding.

Moves the diffusion decoder into the TWM's own 256d latent space. The
corruption process operates directly on W-space vectors. No projection
MLP bridging two unrelated spaces — conditioning and denoising are
geometrically aligned.

Key changes from v11:
  - Denoiser d_model: 256 (= TWM d_model, was 128)
  - Projection MLP eliminated — raw W-space vectors as cross-attention memory
  - 3 natural memory tokens (entity, attr, value) instead of projected tokens
  - Token embeddings: 256d, initialized from TWM encoder pipeline (W-space)
  - adaLN context: raw 768d W-space vectors (3×256), not MLP output

Usage:
    uv run python scripts/train_diffusion_v12.py \
        --data-dir data/atomic_10000 \
        --out-dir results/v12_wspace \
        --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
        --use-adaln --use-continuous-noise --unified-decoder --wspace \
        --config base --epochs 600 --patience 100
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.domain_bpe import DomainBPETokenizer
from twm.domain_dataset import DomainTripleDataset


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


def _diffusion_loss_continuous(logits, target_mask, target_ids, pad_mask, B, M):
    if logits.shape[0] == 0 or not target_mask.any():
        return torch.tensor(0.0, device=logits.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]

    logits_flat = logits[target_mask]
    targets_flat = tgt_valid[target_mask]

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

    metrics = {}
    preds = logits_flat.argmax(-1)
    non_pad = targets_flat != 0
    if non_pad.any():
        metrics["acc"] = (preds[non_pad] == targets_flat[non_pad]).float().mean().item()
    metrics["loss"] = loss.item()
    return loss, metrics


def compute_loss(
    model: DiffusionWorldModel,
    latent: torch.Tensor,
    tgt_attr: torch.Tensor,
    tgt_pad: torch.Tensor,
    entity_token_ids: torch.Tensor,
    value_token_ids: torch.Tensor,
    timestep: torch.Tensor | None = None,
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
        metrics["acc_attr"] = (logits_flat[valid].argmax(-1) == tgt_flat[valid]).float().mean().item()

    entity_logits, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, timestep=timestep,
    )
    entity_loss, entity_metrics = _diffusion_loss_continuous(
        entity_logits, entity_mask, entity_token_ids, tgt_pad, B, M,
    )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]

    value_logits, value_mask = model.forward_value(
        latent, value_token_ids, tgt_pad, timestep=timestep,
    )
    value_loss, value_metrics = _diffusion_loss_continuous(
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
    ds: DomainTripleDataset,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
) -> dict[str, float]:
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

    ent_exact = val_exact = attr_match = 0
    ent_tok_correct = ent_tok_total = 0
    val_tok_correct = val_tok_total = 0
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
            tgt_e_str = domain_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e_str = gen_entities[i][m]
            if pred_e_str == tgt_e_str:
                ent_exact += 1
            all_pred_entities.append(pred_e_str)

            pred_e_tok = torch.tensor(
                domain_tokenizer.encode(pred_e_str, max_length=tgt_e_ids.shape[0]),
                dtype=torch.long,
            )
            non_pad = tgt_e_ids != 0
            if non_pad.any():
                match = (pred_e_tok[:len(tgt_e_ids)][non_pad] == tgt_e_ids[non_pad]).sum().item()
                ent_tok_correct += match
                ent_tok_total += non_pad.sum().item()

            tgt_v_ids = ds._all_value_token_ids[i, m]
            tgt_v_str = domain_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v_str = gen_values[i][m]
            if pred_v_str == tgt_v_str:
                val_exact += 1
            all_pred_values.append(pred_v_str)

            pred_v_tok = torch.tensor(
                domain_tokenizer.encode(pred_v_str, max_length=tgt_v_ids.shape[0]),
                dtype=torch.long,
            )
            non_pad = tgt_v_ids != 0
            if non_pad.any():
                match = (pred_v_tok[:len(tgt_v_ids)][non_pad] == tgt_v_ids[non_pad]).sum().item()
                val_tok_correct += match
                val_tok_total += non_pad.sum().item()

    if total == 0:
        return {}

    return {
        "gen_ent_tok": ent_tok_correct / max(ent_tok_total, 1),
        "gen_ent_exact": ent_exact / total,
        "gen_val_tok": val_tok_correct / max(val_tok_total, 1),
        "gen_val_exact": val_exact / total,
        "gen_attr": attr_match / total,
        "gen_total": total,
        "unique_values": len(set(all_pred_values)),
        "unique_entities": len(set(all_pred_entities)),
        "top_value_count": Counter(all_pred_values).most_common(1)[0][1] if all_pred_values else 0,
    }


@torch.no_grad()
def run_loss_vs_timestep(
    model: DiffusionWorldModel,
    ds: DomainTripleDataset,
    device: torch.device,
    n_examples: int = 64,
    timesteps: list[float] | None = None,
) -> dict[str, float]:
    if timesteps is None:
        timesteps = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

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
    for t_val in timesteps:
        t_tensor = torch.full((B,), t_val, device=device)

        val_logits, val_mask = model.forward_value(latent, val_ids, tgt_pad, timestep=t_tensor)
        if val_logits.shape[0] > 0 and val_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = val_ids.reshape(B * M, -1)[valid]
            val_loss = F.cross_entropy(
                val_logits[val_mask], tgt_valid[val_mask], ignore_index=0,
            ).item()
        else:
            val_loss = 0.0

        ent_logits, ent_mask = model.forward_entity(latent, ent_ids, tgt_pad, timestep=t_tensor)
        if ent_logits.shape[0] > 0 and ent_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = ent_ids.reshape(B * M, -1)[valid]
            ent_loss = F.cross_entropy(
                ent_logits[ent_mask], tgt_valid[ent_mask], ignore_index=0,
            ).item()
        else:
            ent_loss = 0.0

        results[f"t_{t_val:.1f}_val"] = val_loss
        results[f"t_{t_val:.1f}_ent"] = ent_loss

    return results


def main():
    ap = argparse.ArgumentParser(description="Train diffusion TWM v12 (W-space decoding)")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--domain-tokenizer", type=str, required=True)
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--pretrained-dynamics", type=str, default=None)
    ap.add_argument("--pretrained-sentence-model", type=str, default=None)
    ap.add_argument("--freeze-dynamics", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--denoiser-layers", type=int, default=1)
    ap.add_argument("--denoiser-heads", type=int, default=4)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--max-value-tokens", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--use-adaln", action="store_true")
    ap.add_argument("--use-continuous-noise", action="store_true")
    ap.add_argument("--no-normalize-noise", action="store_true")
    ap.add_argument("--no-cross-attention", action="store_true")
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--timestep-bias-power", type=float, default=2.0)
    ap.add_argument("--unified-decoder", action="store_true")
    ap.add_argument("--wspace", action="store_true",
                    help="W-space decoding: denoiser operates in TWM latent space")
    ap.add_argument("--no-wspace-init", action="store_true",
                    help="Skip W-space embedding initialization (random init instead)")
    ap.add_argument("--diagnostic-every", type=int, default=50)
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
    normalize_noise = not args.no_normalize_noise

    domain_tokenizer = DomainBPETokenizer.load(args.domain_tokenizer, max_length=args.max_value_tokens)
    domain_vocab_size = domain_tokenizer.vocab_size
    mask_token_id = domain_tokenizer.mask_token_id

    config = ModelConfig.from_profile(args.config)

    # W-space: denoiser_dim = TWM d_model (256)
    denoiser_dim = config.d_model if args.wspace else 128

    print(f"Device: {device}")
    print(f"W-space: {args.wspace} (denoiser_dim={denoiser_dim})")
    print(f"Domain BPE vocab: {domain_vocab_size} tokens")
    print(f"Max output tokens: {args.max_value_tokens}")
    print(f"Unified decoder: {args.unified_decoder}")
    print(f"Noise: {'continuous' if args.use_continuous_noise else 'discrete'}, normalize: {normalize_noise}")
    print(f"Alpha min: {args.alpha_min}, Timestep bias: {args.timestep_bias_power}")
    print(f"adaLN: {args.use_adaln}, Cross-attention: {use_cross_attention}")

    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(args.st_model, device)
    print(f"  ST dim: {st_dim}, Config: {args.config} (d_model={config.d_model})")

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

    print("Building diffusion model (W-space)...")
    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=args.max_value_tokens,
        n_proj_tokens=3,  # ignored in wspace mode, but set for consistency
        denoiser_layers=args.denoiser_layers,
        denoiser_dim=denoiser_dim,
        denoiser_heads=args.denoiser_heads,
        dropout=args.dropout,
        token_vocab_size=domain_vocab_size,
        mask_token_id=mask_token_id,
        tokenizer=domain_tokenizer,
        use_film=False,
        use_cross_attention=use_cross_attention,
        use_adaln=args.use_adaln,
        use_continuous_noise=args.use_continuous_noise,
        normalize_noise=normalize_noise,
        alpha_min=args.alpha_min,
        timestep_bias_power=args.timestep_bias_power,
        unified_decoder=args.unified_decoder,
        wspace=args.wspace,
    ).to(device)

    # Embeddings are trainable (domain vocab)
    if args.unified_decoder:
        model.triple_decoder.token_emb.weight.requires_grad = True
    else:
        model.entity_decoder.token_emb.weight.requires_grad = True
        model.value_decoder.token_emb.weight.requires_grad = True

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...")
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)
    encoder_ckpt = args.pretrained_sentence_model or args.pretrained_dynamics
    if encoder_ckpt:
        print(f"Loading encoder from {encoder_ckpt}...")
        model.load_encoder_from_sentence_model(encoder_ckpt)

    # Initialize embeddings in W-space from TWM encoder pipeline
    if args.wspace and not args.no_wspace_init:
        print("Initializing domain embeddings in W-space...")
        proj_weight = model.encoder.proj.weight.detach().cpu()  # (d_model, st_dim)
        wspace_embs = domain_tokenizer.build_wspace_init_embeddings(encode_fn, proj_weight)
        if args.unified_decoder:
            model.triple_decoder.token_emb.weight.data.copy_(wspace_embs.to(device))
        else:
            model.entity_decoder.token_emb.weight.data.copy_(wspace_embs.to(device))
            model.value_decoder.token_emb.weight.data.copy_(wspace_embs.to(device))
        print(f"  Initialized {wspace_embs.shape[0]} embeddings in {wspace_embs.shape[1]}d W-space")
        norms = wspace_embs.norm(dim=1)
        print(f"  Norm stats: mean={norms.mean():.1f}, std={norms.std():.1f}")

    if args.freeze_dynamics:
        model.freeze_dynamics()
    if args.freeze_encoder:
        model.freeze_encoder()

    print(f"  Denoiser: {args.denoiser_layers}L, {denoiser_dim}d, {args.denoiser_heads}H")
    print(f"  Total: {model.param_count():,} params, {model.trainable_param_count():,} trainable")

    print("\n  Trainable parameter breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name}: {list(param.shape)} = {param.numel():,}")
    print(flush=True)

    print("Building datasets...")
    train_ds = DomainTripleDataset(
        train_path, encode_fn, vocab, domain_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=args.max_value_tokens,
    )
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = DomainTripleDataset(
            test_path, encode_fn, vocab, domain_tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=args.max_value_tokens,
        )
        print(f"  Test: {len(test_ds)} examples")

    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "st_model": args.st_model, "st_dim": st_dim,
            "n_proj_tokens": 3,
            "denoiser_layers": args.denoiser_layers,
            "denoiser_dim": denoiser_dim,
            "denoiser_heads": args.denoiser_heads,
            "max_value_tokens": args.max_value_tokens,
            "denoise_steps": args.denoise_steps,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "use_film": False,
            "use_cross_attention": use_cross_attention,
            "use_adaln": args.use_adaln,
            "use_continuous_noise": args.use_continuous_noise,
            "normalize_noise": normalize_noise,
            "alpha_min": args.alpha_min,
            "timestep_bias_power": args.timestep_bias_power,
            "unified_decoder": args.unified_decoder,
            "domain_tokenizer": args.domain_tokenizer,
            "domain_vocab_size": domain_vocab_size,
            "wspace": args.wspace,
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

    if test_ds is not None and args.diagnostic_every > 0:
        model.train(False)
        print("\n--- Loss-vs-timestep diagnostic (epoch 0) ---")
        diag = run_loss_vs_timestep(model, test_ds, device, n_examples=64)
        diag_line = "  ".join(
            f"t={k.split('_')[1]}: v={diag[k]:.3f} e={diag[k.replace('_val','_ent')]:.3f}"
            for k in sorted(diag.keys()) if k.endswith("_val")
        )
        print(f"  {diag_line}")
        history.append({"epoch": 0, "diagnostic": diag})

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}, stopping on gen_val_tok)...", flush=True)
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
                all_tgt_a[idx], all_tgt_pads[idx],
                all_ent_ids[idx], all_val_ids[idx],
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
                f" | ent {ev_metrics.get('acc_entity_tok', 0):.3f}"
                f" | attr {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val {ev_metrics.get('acc_value_tok', 0):.3f}"
            )

            test_metrics = None
            gen_metrics = {}
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

                gen_metrics = run_eval_generation(
                    model, test_ds, device, domain_tokenizer,
                    n_examples=64, n_steps=args.denoise_steps,
                )

                gvt = gen_metrics.get("gen_val_tok", 0)
                gve = gen_metrics.get("gen_val_exact", 0)
                get_ = gen_metrics.get("gen_ent_tok", 0)
                gee = gen_metrics.get("gen_ent_exact", 0)

                log += (
                    f" || t_loss {test_metrics['loss_total']:.3f}"
                    f" | gv_tok {gvt:.3f} gv_ex {gve:.3f}"
                    f" | ge_tok {get_:.3f} ge_ex {gee:.3f}"
                    f" | g_attr {gen_metrics.get('gen_attr', 0):.3f}"
                    f" | u_v {gen_metrics.get('unique_values', 0)}"
                    f" u_e {gen_metrics.get('unique_entities', 0)}"
                    f" top1 {gen_metrics.get('top_value_count', 0)}"
                )

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

            if (args.diagnostic_every > 0 and test_ds is not None
                    and epoch % args.diagnostic_every == 0):
                diag = run_loss_vs_timestep(model, test_ds, device, n_examples=64)
                entry["diagnostic"] = diag
                diag_line = "  ".join(
                    f"t={k.split('_')[1]}: v={diag[k]:.3f} e={diag[k.replace('_val','_ent')]:.3f}"
                    for k in sorted(diag.keys()) if k.endswith("_val")
                )
                print(f"  DIAG: {diag_line}", flush=True)

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
