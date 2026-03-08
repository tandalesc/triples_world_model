#!/usr/bin/env python3
"""Train diffusion TWM v12f: Cosine similarity loss + classifier-free guidance.

Cosine loss focuses on directional alignment in W-space, preventing the model
from gaming MSE by memorizing exact vectors. CFG amplifies the conditioning
signal at inference by training with conditioning dropout (15%).

Changes from v12d:
  - Cosine similarity loss instead of MSE
  - Conditioning dropout (cond_drop_prob=0.15) for CFG training
  - Guidance scale sweep at each evaluation step
  - Conditioning reliance diagnostic (conditioned vs unconditioned loss gap)

Usage:
    uv run python scripts/train_diffusion_v12f.py \
        --data-dir data/atomic_10000 \
        --out-dir results/v12f_cosine_cfg \
        --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
        --use-adaln --use-continuous-noise --unified-decoder --wspace \
        --config base --epochs 600 --patience 100 --cond-drop-prob 0.15
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
from twm.diffusion_decoder import importance_sample_timesteps
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


def _cosine_loss(pred_emb, target_mask, target_ids, pad_mask, B, M, token_emb):
    """Cosine similarity loss between predicted and target clean embeddings."""
    if pred_emb.shape[0] == 0 or not target_mask.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]

    # Get target clean embeddings
    target_clean = token_emb(tgt_valid)  # (n_valid, S, d_model)

    # Only loss on non-pad token positions (id=0 is pad)
    non_pad = tgt_valid != 0  # (n_valid, S)

    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    pred_flat = pred_emb[non_pad]  # (total_tokens, d_model)
    target_flat = target_clean[non_pad]  # (total_tokens, d_model)

    # Cosine similarity loss: minimize angular distance
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    loss = (1.0 - cos_sim).mean()

    # Metrics
    metrics = {"loss": loss.item(), "cos_sim": cos_sim.mean().item()}
    with torch.no_grad():
        # NN accuracy
        pred_norm = F.normalize(pred_flat, dim=-1)
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.T)
        nn_ids = sims.argmax(dim=-1)
        tgt_flat_ids = tgt_valid[non_pad]
        metrics["acc"] = (nn_ids == tgt_flat_ids).float().mean().item()

    return loss, metrics


def _mse_loss(pred_emb, target_mask, target_ids, pad_mask, B, M, token_emb):
    """MSE loss between predicted and target clean embeddings."""
    if pred_emb.shape[0] == 0 or not target_mask.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]
    target_clean = token_emb(tgt_valid)
    non_pad = tgt_valid != 0

    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    pred_flat = pred_emb[non_pad]
    target_flat = target_clean[non_pad]
    loss = F.mse_loss(pred_flat, target_flat)

    metrics = {"loss": loss.item()}
    with torch.no_grad():
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item()
        metrics["cos_sim"] = cos_sim
        pred_norm = F.normalize(pred_flat, dim=-1)
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.T)
        nn_ids = sims.argmax(dim=-1)
        tgt_flat_ids = tgt_valid[non_pad]
        metrics["acc"] = (nn_ids == tgt_flat_ids).float().mean().item()

    return loss, metrics


def compute_loss(
    model: DiffusionWorldModel,
    latent: torch.Tensor,
    tgt_attr: torch.Tensor,
    tgt_pad: torch.Tensor,
    entity_token_ids: torch.Tensor,
    value_token_ids: torch.Tensor,
    timestep: torch.Tensor | None = None,
    loss_type: str = "cosine",
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = {}
    B, M = tgt_attr.shape

    # Attr: still CE (discrete classification)
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

    loss_fn = _mse_loss if loss_type == "mse" else _cosine_loss

    # Entity
    decoder = model._get_decoder("entity")
    entity_pred, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, timestep=timestep,
    )
    entity_loss, entity_metrics = loss_fn(
        entity_pred, entity_mask, entity_token_ids, tgt_pad, B, M,
        decoder.token_emb,
    )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]
    if "cos_sim" in entity_metrics:
        metrics["cos_entity"] = entity_metrics["cos_sim"]

    # Value
    value_pred, value_mask = model.forward_value(
        latent, value_token_ids, tgt_pad, timestep=timestep,
    )
    value_loss, value_metrics = loss_fn(
        value_pred, value_mask, value_token_ids, tgt_pad, B, M,
        decoder.token_emb,
    )
    if "acc" in value_metrics:
        metrics["acc_value_tok"] = value_metrics["acc"]
    if "cos_sim" in value_metrics:
        metrics["cos_value"] = value_metrics["cos_sim"]

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
    guidance_scale: float = 1.0,
) -> dict[str, float]:
    n = min(n_examples, len(ds))

    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    gen_entities = model.generate_entities(
        latent, n_steps=n_steps, guidance_scale=guidance_scale,
    )
    gen_values = model.generate_values(
        latent, n_steps=n_steps, guidance_scale=guidance_scale,
    )

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
def run_guidance_sweep(
    model: DiffusionWorldModel,
    ds: DomainTripleDataset,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
    scales: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Run generation at multiple guidance scales and return per-scale metrics."""
    if scales is None:
        scales = [1.0, 2.0, 3.0, 5.0, 7.0]

    results = {}
    for scale in scales:
        m = run_eval_generation(
            model, ds, device, domain_tokenizer,
            n_examples=n_examples, n_steps=n_steps,
            guidance_scale=scale,
        )
        results[f"g{scale:.1f}"] = m
    return results


@torch.no_grad()
def run_continuous_gen_metrics(
    model: DiffusionWorldModel,
    ds: DomainTripleDataset,
    device: torch.device,
    n_examples: int = 64,
) -> dict[str, float]:
    """Cosine sim at t=1.0 (pure noise) — smooth training signal."""
    n = min(n_examples, len(ds))
    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    B = latent.shape[0]
    M = model.config.max_triples
    tgt_pad = ds._all_target_pad_masks[:n].to(device)
    val_ids = ds._all_value_token_ids[:n].to(device)
    ent_ids = ds._all_entity_token_ids[:n].to(device)

    t_tensor = torch.ones(B, device=device)

    decoder = model._get_decoder("value")

    results = {}
    for role, tgt_ids, prefix in [("value", val_ids, "val"), ("entity", ent_ids, "ent")]:
        pred_emb, mask = model._forward_diffusion(
            role, latent, tgt_ids, tgt_pad, timestep=t_tensor,
        )
        if pred_emb.shape[0] == 0:
            continue

        pad_flat = tgt_pad.reshape(B * M)
        valid = ~pad_flat
        tgt_valid = tgt_ids.reshape(B * M, -1)[valid]
        target_clean = decoder.token_emb(tgt_valid)

        non_pad = tgt_valid != 0
        if not non_pad.any():
            continue

        pred_flat = pred_emb[non_pad]
        target_flat = target_clean[non_pad]

        results[f"gen_{prefix}_cos"] = F.cosine_similarity(
            pred_flat, target_flat, dim=-1
        ).mean().item()

    return results


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
    val_ids = ds._all_value_token_ids[:n].to(device)

    decoder = model._get_decoder("value")

    results = {}
    for t_val in timesteps:
        t_tensor = torch.full((B,), t_val, device=device)

        val_pred, val_mask = model.forward_value(latent, val_ids, tgt_pad, timestep=t_tensor)
        if val_pred.shape[0] > 0 and val_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = val_ids.reshape(B * M, -1)[valid]
            target_clean = decoder.token_emb(tgt_valid)
            non_pad = tgt_valid != 0
            if non_pad.any():
                cos = F.cosine_similarity(
                    val_pred[non_pad], target_clean[non_pad], dim=-1
                ).mean().item()
                cos_loss = 1.0 - cos
            else:
                cos = cos_loss = 0.0
        else:
            cos = cos_loss = 0.0

        results[f"t_{t_val:.1f}_cos"] = cos
        results[f"t_{t_val:.1f}_loss"] = cos_loss

    return results


@torch.no_grad()
def run_conditioning_reliance(
    model: DiffusionWorldModel,
    ds: DomainTripleDataset,
    device: torch.device,
    n_examples: int = 64,
    timesteps: list[float] | None = None,
) -> dict[str, float]:
    """Conditioned vs unconditioned cosine loss gap at each timestep."""
    if timesteps is None:
        timesteps = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    n = min(n_examples, len(ds))
    latent = model.encode_dynamics(
        ds._all_inputs[:n].to(device),
        ds._all_input_pad_masks[:n].to(device),
    )

    B = latent.shape[0]
    M = model.config.max_triples
    tgt_pad = ds._all_target_pad_masks[:n].to(device)
    val_ids = ds._all_value_token_ids[:n].to(device)

    decoder = model._get_decoder("value")
    role_id = model._get_role_id("value")
    triple_ctx = model._extract_triple_context(latent)
    ctx_flat = triple_ctx.reshape(B * M, -1)
    tgt_flat = val_ids.reshape(B * M, -1)
    pad_flat = tgt_pad.reshape(B * M)
    valid = ~pad_flat

    ctx_valid = ctx_flat[valid]
    tgt_valid = tgt_flat[valid]
    ctx_zero = torch.zeros_like(ctx_valid)

    target_clean = decoder.token_emb(tgt_valid)
    non_pad = tgt_valid != 0

    results = {}
    for t_val in timesteps:
        n_valid = ctx_valid.shape[0]
        t_tensor = torch.full((n_valid,), t_val, device=device)

        # Conditioned
        pred_cond, _ = decoder(ctx_valid, tgt_valid, timestep=t_tensor, role_id=role_id)
        cos_cond = F.cosine_similarity(
            pred_cond[non_pad], target_clean[non_pad], dim=-1
        ).mean().item()

        # Unconditioned
        pred_uncond, _ = decoder(ctx_zero, tgt_valid, timestep=t_tensor, role_id=role_id)
        cos_uncond = F.cosine_similarity(
            pred_uncond[non_pad], target_clean[non_pad], dim=-1
        ).mean().item()

        results[f"t_{t_val:.1f}_cond"] = cos_cond
        results[f"t_{t_val:.1f}_uncond"] = cos_uncond
        results[f"t_{t_val:.1f}_gap"] = cos_cond - cos_uncond  # positive = conditioning helps

    return results


def main():
    ap = argparse.ArgumentParser(description="Train diffusion TWM v12f (cosine loss + CFG)")
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
    ap.add_argument("--wspace", action="store_true")
    ap.add_argument("--no-wspace-init", action="store_true")
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--cond-drop-prob", type=float, default=0.15)
    ap.add_argument("--fixed-timestep", type=float, default=None,
                    help="Fix timestep to this value (e.g. 1.0 for pure construction)")
    ap.add_argument("--loss-type", type=str, default="mse",
                    choices=["cosine", "mse"],
                    help="Loss function: MSE (default) or cosine similarity")
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
    denoiser_dim = config.d_model if args.wspace else 128

    print(f"Device: {device}")
    print(f"W-space: {args.wspace} (denoiser_dim={denoiser_dim})")
    print(f"Prediction: cosine x₀-prediction (no output head)")
    print(f"Loss: {args.loss_type} x₀-prediction")
    print(f"CFG: cond_drop_prob={args.cond_drop_prob}")
    if args.fixed_timestep is not None:
        print(f"Fixed timestep: {args.fixed_timestep}")
    print(f"Domain BPE vocab: {domain_vocab_size} tokens")
    print(f"Max output tokens: {args.max_value_tokens}")
    print(f"Unified decoder: {args.unified_decoder}")
    print(f"Noise: {'continuous' if args.use_continuous_noise else 'discrete'}, normalize: {normalize_noise}")
    print(f"Alpha min: {args.alpha_min}, Timestep bias: {args.timestep_bias_power}")

    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(args.st_model, device)

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

    print("Building diffusion model (cosine loss + CFG)...")
    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=args.max_value_tokens,
        n_proj_tokens=3,
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
        use_mse_prediction=True,
        cond_drop_prob=args.cond_drop_prob,
    ).to(device)

    # Embeddings frozen — W-space init stays fixed as the decode target
    # (token_emb.weight.requires_grad defaults to False in DiffusionDecoder)

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...")
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)
    encoder_ckpt = args.pretrained_sentence_model or args.pretrained_dynamics
    if encoder_ckpt:
        print(f"Loading encoder from {encoder_ckpt}...")
        model.load_encoder_from_sentence_model(encoder_ckpt)

    # Initialize embeddings in W-space
    if args.wspace and not args.no_wspace_init:
        print("Initializing domain embeddings in W-space...")
        proj_weight = model.encoder.proj.weight.detach().cpu()
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
            "use_mse_prediction": True,
            "cond_drop_prob": args.cond_drop_prob,
            "loss_type": args.loss_type,
            "fixed_timestep": args.fixed_timestep,
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
    best_guidance_scale = 1.0
    epochs_without_improvement = 0
    history = []

    guidance_scales = [1.0, 2.0, 3.0, 5.0, 7.0]

    def _emb_diagnostic(label: str):
        decoder = model._get_decoder("value")
        embs = decoder.token_emb.weight.detach()
        V = embs.shape[0]
        normed = F.normalize(embs, dim=-1)
        sims = normed @ normed.T
        mask = ~torch.eye(V, dtype=torch.bool, device=sims.device)
        print(f"  EMB ({label}): mean_cos={sims[mask].mean():.4f} "
              f"max_cos={sims[mask].max():.4f} "
              f"min_cos={sims[mask].min():.4f} "
              f"norm_mean={embs.norm(dim=-1).mean():.2f} "
              f"norm_std={embs.norm(dim=-1).std():.2f}", flush=True)

    if test_ds is not None and args.diagnostic_every > 0:
        model.train(False)
        print("\n--- Initial diagnostics ---")
        _emb_diagnostic("epoch=0")
        diag = run_loss_vs_timestep(model, test_ds, device, n_examples=64)
        diag_line = "  ".join(
            f"t={k.split('_')[1]}: cos={diag[k]:.3f}"
            for k in sorted(diag.keys()) if k.endswith("_cos")
        )
        print(f"  DIAG: {diag_line}")

        cont = run_continuous_gen_metrics(model, test_ds, device, n_examples=64)
        if cont:
            print(f"  GEN (t=1.0): val_cos={cont.get('gen_val_cos', 0):.3f} "
                  f"ent_cos={cont.get('gen_ent_cos', 0):.3f}")

        cond_diag = run_conditioning_reliance(model, test_ds, device, n_examples=64)
        cond_line = "  ".join(
            f"t={k.split('_')[1]}: gap={cond_diag[k]:.4f}"
            for k in sorted(cond_diag.keys()) if k.endswith("_gap")
        )
        print(f"  COND: {cond_line}", flush=True)

        history.append({"epoch": 0, "diagnostic": diag, "continuous": cont, "cond_reliance": cond_diag})

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}, stopping on gen_val_tok)...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            B_batch = idx.shape[0]
            timestep = None
            if args.fixed_timestep is not None:
                timestep = torch.full((B_batch,), args.fixed_timestep, device=device)
            latent = model.encode_dynamics(all_inputs[idx], all_input_pads[idx])
            loss, _ = compute_loss(
                model, latent,
                all_tgt_a[idx], all_tgt_pads[idx],
                all_ent_ids[idx], all_val_ids[idx],
                timestep=timestep,
                loss_type=args.loss_type,
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
                    loss_type=args.loss_type,
                )

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent {ev_metrics.get('acc_entity_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_entity', 0):.3f}"
                f" | attr {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val {ev_metrics.get('acc_value_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_value', 0):.3f}"
            )

            test_metrics = None
            gen_metrics = {}
            cont_metrics = {}
            sweep_results = {}
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
                        loss_type=args.loss_type,
                    )

                # Guidance scale sweep
                sweep_results = run_guidance_sweep(
                    model, test_ds, device, domain_tokenizer,
                    n_examples=64, n_steps=args.denoise_steps,
                    scales=guidance_scales,
                )

                # Find best guidance scale this epoch
                epoch_best_gvt = -1.0
                epoch_best_scale = 1.0
                for scale_key, m in sweep_results.items():
                    gvt = m.get("gen_val_tok", 0)
                    if gvt > epoch_best_gvt:
                        epoch_best_gvt = gvt
                        epoch_best_scale = float(scale_key[1:])

                # Use unguided (g1.0) as primary gen_metrics for display
                gen_metrics = sweep_results.get("g1.0", {})

                cont_metrics = run_continuous_gen_metrics(
                    model, test_ds, device, n_examples=64,
                )

                gvt = gen_metrics.get("gen_val_tok", 0)
                gve = gen_metrics.get("gen_val_exact", 0)
                get_ = gen_metrics.get("gen_ent_tok", 0)
                gee = gen_metrics.get("gen_ent_exact", 0)

                log += (
                    f" || t_loss {test_metrics['loss_total']:.4f}"
                    f" | gv_tok {gvt:.3f} gv_ex {gve:.3f}"
                    f" | ge_tok {get_:.3f} ge_ex {gee:.3f}"
                    f" | g_attr {gen_metrics.get('gen_attr', 0):.3f}"
                    f" | u_v {gen_metrics.get('unique_values', 0)}"
                    f" u_e {gen_metrics.get('unique_entities', 0)}"
                    f" top1 {gen_metrics.get('top_value_count', 0)}"
                )

                if cont_metrics:
                    log += f" | cos_v {cont_metrics.get('gen_val_cos', 0):.3f}"

                # CFG sweep summary
                cfg_parts = []
                for scale_key, m in sorted(sweep_results.items()):
                    cfg_parts.append(f"{scale_key}={m.get('gen_val_tok', 0):.3f}")
                log += f" | CFG [{' '.join(cfg_parts)}]"

                # Track best across all guidance scales
                if epoch_best_gvt > best_gen_val_tok:
                    best_gen_val_tok = epoch_best_gvt
                    best_gen_epoch = epoch
                    best_guidance_scale = epoch_best_scale
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), out_dir / "model_best.pt")
                    log += f" * (g={epoch_best_scale:.1f})"
                else:
                    epochs_without_improvement += args.log_every

            if epoch <= 50 or epoch % args.diagnostic_every == 0:
                _emb_diagnostic(f"epoch={epoch}")

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
            if cont_metrics:
                entry.update(cont_metrics)
            if sweep_results:
                entry["guidance_sweep"] = sweep_results

            if (args.diagnostic_every > 0 and test_ds is not None
                    and epoch % args.diagnostic_every == 0):
                diag = run_loss_vs_timestep(model, test_ds, device, n_examples=64)
                entry["diagnostic"] = diag
                diag_line = "  ".join(
                    f"t={k.split('_')[1]}: cos={diag[k]:.3f}"
                    for k in sorted(diag.keys()) if k.endswith("_cos")
                )
                print(f"  DIAG: {diag_line}", flush=True)

                cond_diag = run_conditioning_reliance(model, test_ds, device, n_examples=64)
                entry["cond_reliance"] = cond_diag
                cond_line = "  ".join(
                    f"t={k.split('_')[1]}: gap={cond_diag[k]:.4f}"
                    for k in sorted(cond_diag.keys()) if k.endswith("_gap")
                )
                print(f"  COND: {cond_line}", flush=True)

            history.append(entry)

            if epochs_without_improvement >= args.patience and test_ds is not None:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best gen_val_tok {best_gen_val_tok:.4f} at epoch {best_gen_epoch}, "
                      f"guidance_scale={best_guidance_scale:.1f})", flush=True)
                break

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    print(f"Best gen_val_tok: {best_gen_val_tok:.4f} at epoch {best_gen_epoch} (guidance_scale={best_guidance_scale:.1f})")
    print(f"Stopped at epoch {epoch}")


if __name__ == "__main__":
    main()
