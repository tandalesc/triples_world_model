#!/usr/bin/env python3
"""Train diffusion TWM v12g: Curriculum expansion with decode projection.

Phase 1 (v12f) proved conditioning works at t=1.0. This script runs phases 2-4,
progressively expanding the timestep range while keeping the conditioning pathway
alive. Adds a learned decode_proj (256→256) with auxiliary CE loss for NN decode
sharpening.

Phases:
  2: t ∈ [0.7, 1.0]  — introduce faint residual signal
  3: t ∈ [0.4, 1.0]  — more residual signal
  4: t ∈ [0.0, 1.0]  — full range with importance sampling

Usage:
    uv run python scripts/train_diffusion_v12g.py \
        --data-dir data/atomic_10000 \
        --out-dir results/v12g_curriculum \
        --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
        --phase1-checkpoint results/v12f_t1_mse/model_best.pt \
        --use-adaln --use-continuous-noise --unified-decoder --wspace \
        --config base --cond-drop-prob 0.15
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
from twm.token_dataset import TokenTripleDataset


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


def sample_timestep(B: int, device: torch.device, t_min: float, t_max: float,
                    bias_power: float = 1.0) -> torch.Tensor:
    """Sample timesteps in [t_min, t_max], optionally importance-sampled."""
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    if t_min == 0.0 and t_max == 1.0 and bias_power != 1.0:
        return importance_sample_timesteps(B, device, bias_power)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


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


def _aux_ce_loss(pred_emb, target_ids, pad_mask, B, M, decoder):
    """Auxiliary CE loss through decode_proj for NN decode sharpening."""
    if pred_emb.shape[0] == 0:
        return torch.tensor(0.0, device=pred_emb.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]
    non_pad = tgt_valid != 0

    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    # Get similarity logits through decode_proj
    logits = decoder.decode_proj_logits(pred_emb)  # (n_valid, S, vocab)
    logits_flat = logits[non_pad]  # (total_tokens, vocab)
    targets_flat = tgt_valid[non_pad]  # (total_tokens,)

    # Scale logits for softmax (temperature)
    loss = F.cross_entropy(logits_flat / 0.1, targets_flat, ignore_index=0)

    metrics = {}
    with torch.no_grad():
        preds = logits_flat.argmax(-1)
        metrics["proj_acc"] = (preds == targets_flat).float().mean().item()

    return loss, metrics


def _ce_loss(logits, target_mask, target_ids, pad_mask, B, M):
    """CE loss on output head logits."""
    if logits.shape[0] == 0 or not target_mask.any():
        return torch.tensor(0.0, device=logits.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]

    logits_flat = logits[target_mask]
    targets_flat = tgt_valid[target_mask]

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

    metrics = {}
    with torch.no_grad():
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
    aux_ce_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = {}
    B, M = tgt_attr.shape

    # Attr: CE
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

    decoder = model._get_decoder("entity")
    use_aux = decoder.use_decode_proj
    use_ce = not decoder.use_mse_prediction  # output head exists → CE loss

    # Entity
    entity_pred, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, timestep=timestep,
    )
    if use_ce:
        entity_loss, entity_metrics = _ce_loss(
            entity_pred, entity_mask, entity_token_ids, tgt_pad, B, M,
        )
    else:
        entity_loss, entity_metrics = _mse_loss(
            entity_pred, entity_mask, entity_token_ids, tgt_pad, B, M,
            decoder.token_emb,
        )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]
    if "cos_sim" in entity_metrics:
        metrics["cos_entity"] = entity_metrics["cos_sim"]

    entity_aux = torch.tensor(0.0, device=latent.device)
    if use_aux and not use_ce and entity_pred.shape[0] > 0:
        entity_aux, ent_aux_m = _aux_ce_loss(
            entity_pred, entity_token_ids, tgt_pad, B, M, decoder,
        )
        if "proj_acc" in ent_aux_m:
            metrics["proj_acc_entity"] = ent_aux_m["proj_acc"]

    # Value
    value_pred, value_mask = model.forward_value(
        latent, value_token_ids, tgt_pad, timestep=timestep,
    )
    if use_ce:
        value_loss, value_metrics = _ce_loss(
            value_pred, value_mask, value_token_ids, tgt_pad, B, M,
        )
    else:
        value_loss, value_metrics = _mse_loss(
            value_pred, value_mask, value_token_ids, tgt_pad, B, M,
            decoder.token_emb,
        )
    if "acc" in value_metrics:
        metrics["acc_value_tok"] = value_metrics["acc"]
    if "cos_sim" in value_metrics:
        metrics["cos_value"] = value_metrics["cos_sim"]

    value_aux = torch.tensor(0.0, device=latent.device)
    if use_aux and not use_ce and value_pred.shape[0] > 0:
        value_aux, val_aux_m = _aux_ce_loss(
            value_pred, value_token_ids, tgt_pad, B, M, decoder,
        )
        if "proj_acc" in val_aux_m:
            metrics["proj_acc_value"] = val_aux_m["proj_acc"]

    total_loss = attr_loss + entity_loss + value_loss
    if use_aux and not use_ce:
        total_loss = total_loss + aux_ce_weight * (entity_aux + value_aux)
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
    model, ds, device, domain_tokenizer,
    n_examples=64, n_steps=10, scales=None,
):
    if scales is None:
        scales = [1.0, 3.0, 5.0, 7.0]
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
def run_conditioning_reliance(
    model, ds, device, n_examples=64, timesteps=None,
):
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

    non_pad = tgt_valid != 0
    use_ce = not decoder.use_mse_prediction

    if not use_ce:
        target_clean = decoder.token_emb(tgt_valid)

    results = {}
    for t_val in timesteps:
        n_valid = ctx_valid.shape[0]
        t_tensor = torch.full((n_valid,), t_val, device=device)

        pred_cond, _ = decoder(ctx_valid, tgt_valid, timestep=t_tensor, role_id=role_id)
        pred_uncond, _ = decoder(ctx_zero, tgt_valid, timestep=t_tensor, role_id=role_id)

        if use_ce:
            # CE mode: measure accuracy gap
            acc_cond = (pred_cond[non_pad].argmax(-1) == tgt_valid[non_pad]).float().mean().item()
            acc_uncond = (pred_uncond[non_pad].argmax(-1) == tgt_valid[non_pad]).float().mean().item()
            results[f"t_{t_val:.1f}_cond"] = acc_cond
            results[f"t_{t_val:.1f}_uncond"] = acc_uncond
            results[f"t_{t_val:.1f}_gap"] = acc_cond - acc_uncond
        else:
            cos_cond = F.cosine_similarity(
                pred_cond[non_pad], target_clean[non_pad], dim=-1
            ).mean().item()
            cos_uncond = F.cosine_similarity(
                pred_uncond[non_pad], target_clean[non_pad], dim=-1
            ).mean().item()
            results[f"t_{t_val:.1f}_cond"] = cos_cond
            results[f"t_{t_val:.1f}_uncond"] = cos_uncond
            results[f"t_{t_val:.1f}_gap"] = cos_cond - cos_uncond

    return results


@torch.no_grad()
def run_loss_vs_timestep(model, ds, device, n_examples=64, timesteps=None):
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

    use_ce = not decoder.use_mse_prediction

    results = {}
    for t_val in timesteps:
        t_tensor = torch.full((B,), t_val, device=device)
        val_pred, val_mask = model.forward_value(latent, val_ids, tgt_pad, timestep=t_tensor)
        if val_pred.shape[0] > 0 and val_mask.any():
            pad_flat = tgt_pad.reshape(B * M)
            valid = ~pad_flat
            tgt_valid = val_ids.reshape(B * M, -1)[valid]
            non_pad = tgt_valid != 0
            if non_pad.any():
                if use_ce:
                    # val_pred is logits (N, S, vocab) — measure accuracy
                    preds = val_pred[non_pad].argmax(-1)
                    targets = tgt_valid[non_pad]
                    acc = (preds == targets).float().mean().item()
                    results[f"t_{t_val:.1f}_acc"] = acc
                else:
                    target_clean = decoder.token_emb(tgt_valid)
                    cos = F.cosine_similarity(
                        val_pred[non_pad], target_clean[non_pad], dim=-1
                    ).mean().item()
                    results[f"t_{t_val:.1f}_cos"] = cos
            else:
                if use_ce:
                    results[f"t_{t_val:.1f}_acc"] = 0.0
                else:
                    results[f"t_{t_val:.1f}_cos"] = 0.0
        else:
            if use_ce:
                results[f"t_{t_val:.1f}_acc"] = 0.0
            else:
                results[f"t_{t_val:.1f}_cos"] = 0.0

    return results


def run_phase(
    phase_name: str,
    model: DiffusionWorldModel,
    train_ds: DomainTripleDataset,
    test_ds: DomainTripleDataset | None,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    out_dir: Path,
    t_min: float,
    t_max: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    denoise_steps: int,
    log_every: int,
    diagnostic_every: int,
    timestep_bias_power: float,
    aux_ce_weight: float,
    guidance_scales: list[float],
) -> tuple[float, int]:
    """Run a single curriculum phase. Returns (best_gen_val_tok, best_epoch)."""
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name}  t ∈ [{t_min}, {t_max}]")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

    def _emb_diagnostic(label: str):
        decoder = model._get_decoder("value")
        embs = decoder.token_emb.weight.detach()
        V = embs.shape[0]
        normed = F.normalize(embs, dim=-1)
        sims = normed @ normed.T
        mask = ~torch.eye(V, dtype=torch.bool, device=sims.device)
        print(f"  EMB ({label}): mean_cos={sims[mask].mean():.4f} "
              f"max_cos={sims[mask].max():.4f} "
              f"norm_mean={embs.norm(dim=-1).mean():.2f} "
              f"norm_std={embs.norm(dim=-1).std():.2f}", flush=True)

    # Initial diagnostics
    if test_ds is not None:
        model.train(False)
        _emb_diagnostic(f"{phase_name}_start")
        sweep = run_guidance_sweep(
            model, test_ds, device, domain_tokenizer,
            n_examples=64, n_steps=denoise_steps, scales=guidance_scales,
        )
        for sk, sm in sorted(sweep.items()):
            print(f"  {phase_name} init {sk}: gv_tok={sm.get('gen_val_tok',0):.3f} "
                  f"ge_tok={sm.get('gen_ent_tok',0):.3f}", flush=True)

    print(f"\nTraining {phase_name} for up to {epochs} epochs "
          f"(patience={patience})...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            B_batch = idx.shape[0]
            timestep = sample_timestep(B_batch, device, t_min, t_max,
                                       bias_power=timestep_bias_power)
            latent = model.encode_dynamics(all_inputs[idx], all_input_pads[idx])
            loss, _ = compute_loss(
                model, latent,
                all_tgt_a[idx], all_tgt_pads[idx],
                all_ent_ids[idx], all_val_ids[idx],
                timestep=timestep,
                aux_ce_weight=aux_ce_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1:
            model.train(False)

            # Train metrics
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
                    aux_ce_weight=aux_ce_weight,
                )

            log = (
                f"[{phase_name}] Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent {ev_metrics.get('acc_entity_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_entity', 0):.3f}"
                f" | attr {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val {ev_metrics.get('acc_value_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_value', 0):.3f}"
            )
            if "proj_acc_value" in ev_metrics:
                log += f" proj {ev_metrics['proj_acc_value']:.3f}"

            sweep_results = {}
            if test_ds is not None:
                sweep_results = run_guidance_sweep(
                    model, test_ds, device, domain_tokenizer,
                    n_examples=64, n_steps=denoise_steps, scales=guidance_scales,
                )

                # Find best guidance scale
                epoch_best_gvt = -1.0
                epoch_best_scale = 1.0
                for scale_key, m in sweep_results.items():
                    gvt = m.get("gen_val_tok", 0)
                    if gvt > epoch_best_gvt:
                        epoch_best_gvt = gvt
                        epoch_best_scale = float(scale_key[1:])

                gen_metrics = sweep_results.get("g1.0", {})
                gvt = gen_metrics.get("gen_val_tok", 0)
                gve = gen_metrics.get("gen_val_exact", 0)

                log += (
                    f" || gv_tok {gvt:.3f} gv_ex {gve:.3f}"
                    f" | g_attr {gen_metrics.get('gen_attr', 0):.3f}"
                    f" | u_v {gen_metrics.get('unique_values', 0)}"
                    f" top1 {gen_metrics.get('top_value_count', 0)}"
                )

                cfg_parts = []
                for scale_key, m in sorted(sweep_results.items()):
                    cfg_parts.append(f"{scale_key}={m.get('gen_val_tok', 0):.3f}")
                log += f" | CFG [{' '.join(cfg_parts)}]"

                if epoch_best_gvt > best_gen_val_tok:
                    best_gen_val_tok = epoch_best_gvt
                    best_gen_epoch = epoch
                    best_guidance_scale = epoch_best_scale
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), phase_dir / "model_best.pt")
                    log += f" * (g={epoch_best_scale:.1f})"
                else:
                    epochs_without_improvement += log_every

            if epoch <= 50 or epoch % diagnostic_every == 0:
                _emb_diagnostic(f"epoch={epoch}")

            print(log, flush=True)

            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in ev_metrics.items()},
            }
            if sweep_results:
                entry["guidance_sweep"] = sweep_results

            if (diagnostic_every > 0 and test_ds is not None
                    and epoch % diagnostic_every == 0):
                diag = run_loss_vs_timestep(model, test_ds, device, n_examples=64)
                entry["diagnostic"] = diag
                diag_parts = []
                for k in sorted(diag.keys()):
                    t_str = k.split('_')[1]
                    if k.endswith("_cos"):
                        diag_parts.append(f"t={t_str}: cos={diag[k]:.3f}")
                    elif k.endswith("_acc"):
                        diag_parts.append(f"t={t_str}: acc={diag[k]:.3f}")
                diag_line = "  ".join(diag_parts)
                print(f"  DIAG: {diag_line}", flush=True)

                cond_diag = run_conditioning_reliance(model, test_ds, device, n_examples=64)
                entry["cond_reliance"] = cond_diag
                cond_line = "  ".join(
                    f"t={k.split('_')[1]}: gap={cond_diag[k]:.4f}"
                    for k in sorted(cond_diag.keys()) if k.endswith("_gap")
                )
                print(f"  COND: {cond_line}", flush=True)

            history.append(entry)

            if epochs_without_improvement >= patience and test_ds is not None:
                print(f"\n[{phase_name}] Early stopping at epoch {epoch} "
                      f"(best gvt {best_gen_val_tok:.4f} at epoch {best_gen_epoch}, "
                      f"g={best_guidance_scale:.1f})", flush=True)
                break

    torch.save(model.state_dict(), phase_dir / "model_final.pt")
    with open(phase_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{phase_name}] Done. Best gvt: {best_gen_val_tok:.4f} at epoch {best_gen_epoch} "
          f"(g={best_guidance_scale:.1f})")

    return best_gen_val_tok, best_gen_epoch


def main():
    ap = argparse.ArgumentParser(description="Train diffusion TWM v12g (curriculum + decode proj)")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--domain-tokenizer", type=str, required=True)
    ap.add_argument("--phase1-checkpoint", type=str, required=True,
                    help="Best checkpoint from phase 1 (v12f t=1.0 run)")
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
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
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--no-decode-proj", action="store_true",
                    help="Disable decode_proj — pure curriculum, no projection layer")
    ap.add_argument("--use-output-head", action="store_true",
                    help="Use learned output head + CE loss instead of MSE + NN decode")
    ap.add_argument("--token-level", action="store_true",
                    help="Use token-level BPE input encoding instead of sentence embeddings")
    ap.add_argument("--max-tokens-per-slot", type=int, default=12,
                    help="Max BPE tokens per triple slot (token-level mode)")
    # Phase config
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--phase3-epochs", type=int, default=200)
    ap.add_argument("--phase4-epochs", type=int, default=200)
    ap.add_argument("--phase2-patience", type=int, default=50)
    ap.add_argument("--phase3-patience", type=int, default=50)
    ap.add_argument("--phase4-patience", type=int, default=100)
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
    use_proj = not args.no_decode_proj
    use_output_head = args.use_output_head
    if use_output_head:
        print(f"Prediction: CE through learned output head (W-space denoiser → logits)")
    elif use_proj:
        print(f"Prediction: MSE x₀ + {args.aux_ce_weight}× aux CE through decode_proj")
    else:
        print(f"Prediction: MSE x₀ (no decode_proj)")
    print(f"CFG: cond_drop_prob={args.cond_drop_prob}")
    print(f"Domain BPE vocab: {domain_vocab_size} tokens")
    print(f"Unified decoder: {args.unified_decoder}")

    encode_fn, st_dim = None, 0
    if not args.token_level:
        print("Loading sentence-transformer...")
        encode_fn, st_dim = make_encode_fn(args.st_model, device)
    else:
        print("Token-level mode: skipping sentence-transformer")

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

    token_level = args.token_level
    max_tokens_per_slot = args.max_tokens_per_slot

    # For token-level mode, st_dim is not used (encoder uses W-space embeddings)
    if token_level:
        st_dim = config.d_model  # placeholder — TokenEncoder doesn't use st_dim

    print(f"Building diffusion model ({'CE output head' if use_output_head else 'MSE x₀'}"
          f"{', token-level' if token_level else ''})...")
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
        use_mse_prediction=not use_output_head,
        cond_drop_prob=args.cond_drop_prob,
        use_decode_proj=use_proj and not use_output_head,
        token_level=token_level,
        max_tokens_per_slot=max_tokens_per_slot,
    ).to(device)

    # Embeddings stay frozen
    # (token_emb.weight.requires_grad defaults to False in DiffusionDecoder)

    # Load phase 1 checkpoint
    print(f"Loading phase 1 checkpoint: {args.phase1_checkpoint}")
    sd = torch.load(args.phase1_checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys (expected for new decode_proj): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

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
    if token_level:
        # Get frozen W-space token embeddings from the decoder
        token_emb_weight = model._get_decoder("value").token_emb.weight.detach().cpu()
        train_ds = TokenTripleDataset(
            train_path, token_emb_weight, vocab, domain_tokenizer,
            max_triples=config.max_triples,
            max_tokens_per_slot=max_tokens_per_slot,
            max_value_tokens=args.max_value_tokens,
        )
    else:
        train_ds = DomainTripleDataset(
            train_path, encode_fn, vocab, domain_tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=args.max_value_tokens,
        )
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        if token_level:
            test_ds = TokenTripleDataset(
                test_path, token_emb_weight, vocab, domain_tokenizer,
                max_triples=config.max_triples,
                max_tokens_per_slot=max_tokens_per_slot,
                max_value_tokens=args.max_value_tokens,
            )
        else:
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
            "use_mse_prediction": not use_output_head,
            "cond_drop_prob": args.cond_drop_prob,
            "use_decode_proj": use_proj and not use_output_head,
            "aux_ce_weight": args.aux_ce_weight,
            "token_level": token_level,
            "max_tokens_per_slot": max_tokens_per_slot,
        }, f, indent=2)

    guidance_scales = [1.0, 3.0, 5.0, 7.0]
    common_kwargs = dict(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        domain_tokenizer=domain_tokenizer,
        out_dir=out_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every,
        diagnostic_every=args.diagnostic_every,
        timestep_bias_power=args.timestep_bias_power,
        aux_ce_weight=args.aux_ce_weight,
        guidance_scales=guidance_scales,
    )

    # Phase 2: t ∈ [0.7, 1.0]
    gvt2, ep2 = run_phase(
        phase_name="phase2",
        t_min=0.7, t_max=1.0,
        epochs=args.phase2_epochs,
        patience=args.phase2_patience,
        **common_kwargs,
    )

    # Load phase 2 best for phase 3
    p2_best = out_dir / "phase2" / "model_best.pt"
    if p2_best.exists():
        print(f"\nLoading phase 2 best checkpoint for phase 3...")
        model.load_state_dict(torch.load(p2_best, map_location=device, weights_only=True))

    # Phase 3: t ∈ [0.4, 1.0]
    gvt3, ep3 = run_phase(
        phase_name="phase3",
        t_min=0.4, t_max=1.0,
        epochs=args.phase3_epochs,
        patience=args.phase3_patience,
        **common_kwargs,
    )

    # Load phase 3 best for phase 4
    p3_best = out_dir / "phase3" / "model_best.pt"
    if p3_best.exists():
        print(f"\nLoading phase 3 best checkpoint for phase 4...")
        model.load_state_dict(torch.load(p3_best, map_location=device, weights_only=True))

    # Phase 4: t ∈ [0.0, 1.0] with importance sampling
    gvt4, ep4 = run_phase(
        phase_name="phase4",
        t_min=0.0, t_max=1.0,
        epochs=args.phase4_epochs,
        patience=args.phase4_patience,
        **common_kwargs,
    )

    # Copy overall best to out_dir root
    best_gvt = max(gvt2, gvt3, gvt4)
    if best_gvt == gvt4 and (out_dir / "phase4" / "model_best.pt").exists():
        best_phase = "phase4"
    elif best_gvt == gvt3 and (out_dir / "phase3" / "model_best.pt").exists():
        best_phase = "phase3"
    else:
        best_phase = "phase2"

    import shutil
    src = out_dir / best_phase / "model_best.pt"
    if src.exists():
        shutil.copy2(src, out_dir / "model_best.pt")
        print(f"\nOverall best: {best_phase} gvt={best_gvt:.4f} → {out_dir}/model_best.pt")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Phase 2 (t∈[0.7,1.0]): best gvt={gvt2:.4f}")
    print(f"  Phase 3 (t∈[0.4,1.0]): best gvt={gvt3:.4f}")
    print(f"  Phase 4 (t∈[0.0,1.0]): best gvt={gvt4:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
