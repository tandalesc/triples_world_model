#!/usr/bin/env python3
"""Train v15: Fresh compressor/expander with integrated length head and curriculum.

All components co-train from epoch 1: content prediction, length prediction,
dropout regularization. No pretrained checkpoint, no inherited weights.

Curriculum:
  Phase 1: t in [0.7, 1.0] -- 100 epochs (no early stopping)
  Phase 2: t in [0.0, 1.0] -- 200 epochs (importance sampling, early stop on gen_val_exact)

Usage:
    uv run python scripts/train_v15_fresh.py \
        --data-dir data/atomic_10000_identity \
        --out-dir results/v15a_1L_fresh \
        --domain-tokenizer data/atomic_10000/domain_bpe_tokenizer.json \
        --config base --denoiser-layers 1
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
from twm.diffusion_decoder import importance_sample_timesteps
from twm.domain_bpe import DomainBPETokenizer
from twm.compressor import TripleCompressor
from twm.compressor_dataset import CompressorDataset


# -- Timestep sampling ------------------------------------------------

def sample_timestep(B: int, device: torch.device, t_min: float, t_max: float,
                    bias_power: float = 1.0) -> torch.Tensor:
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    if t_min == 0.0 and t_max == 1.0 and bias_power != 1.0:
        return importance_sample_timesteps(B, device, bias_power)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


# -- Loss functions ----------------------------------------------------

def _mse_loss(pred_emb, target_mask, target_ids, pad_mask, B, M, token_emb):
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
    if pred_emb.shape[0] == 0:
        return torch.tensor(0.0, device=pred_emb.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]
    non_pad = tgt_valid != 0

    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    logits = decoder.decode_proj_logits(pred_emb)
    logits_flat = logits[non_pad]
    targets_flat = tgt_valid[non_pad]

    loss = F.cross_entropy(logits_flat / 0.1, targets_flat, ignore_index=0)

    metrics = {}
    with torch.no_grad():
        preds = logits_flat.argmax(-1)
        metrics["proj_acc"] = (preds == targets_flat).float().mean().item()

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
    length_weight: float = 0.1,
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

    # Entity: MSE + aux CE
    entity_pred, entity_mask = model.forward_entity(
        latent, entity_token_ids, tgt_pad, timestep=timestep,
    )
    entity_loss, entity_metrics = _mse_loss(
        entity_pred, entity_mask, entity_token_ids, tgt_pad, B, M,
        decoder.token_emb,
    )
    if "acc" in entity_metrics:
        metrics["acc_entity_tok"] = entity_metrics["acc"]
    if "cos_sim" in entity_metrics:
        metrics["cos_entity"] = entity_metrics["cos_sim"]

    entity_aux = torch.tensor(0.0, device=latent.device)
    if decoder.use_decode_proj and entity_pred.shape[0] > 0:
        entity_aux, ent_aux_m = _aux_ce_loss(
            entity_pred, entity_token_ids, tgt_pad, B, M, decoder,
        )
        if "proj_acc" in ent_aux_m:
            metrics["proj_acc_entity"] = ent_aux_m["proj_acc"]

    # Value: MSE + aux CE
    value_pred, value_mask = model.forward_value(
        latent, value_token_ids, tgt_pad, timestep=timestep,
    )
    value_loss, value_metrics = _mse_loss(
        value_pred, value_mask, value_token_ids, tgt_pad, B, M,
        decoder.token_emb,
    )
    if "acc" in value_metrics:
        metrics["acc_value_tok"] = value_metrics["acc"]
    if "cos_sim" in value_metrics:
        metrics["cos_value"] = value_metrics["cos_sim"]

    value_aux = torch.tensor(0.0, device=latent.device)
    if decoder.use_decode_proj and value_pred.shape[0] > 0:
        value_aux, val_aux_m = _aux_ce_loss(
            value_pred, value_token_ids, tgt_pad, B, M, decoder,
        )
        if "proj_acc" in val_aux_m:
            metrics["proj_acc_value"] = val_aux_m["proj_acc"]

    # Length prediction
    ent_len_pred, val_len_pred = model.forward_lengths(latent)
    ent_real = (entity_token_ids != 0).sum(dim=-1).float()
    val_real = (value_token_ids != 0).sum(dim=-1).float()
    valid_mask = ~tgt_pad
    if valid_mask.any():
        length_loss = (
            F.mse_loss(ent_len_pred[valid_mask], ent_real[valid_mask])
            + F.mse_loss(val_len_pred[valid_mask], val_real[valid_mask])
        )
    else:
        length_loss = torch.tensor(0.0, device=latent.device)
    metrics["length_loss"] = length_loss.item()

    total_loss = (
        attr_loss + entity_loss + value_loss
        + aux_ce_weight * (entity_aux + value_aux)
        + length_weight * length_loss
    )
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


# -- Encode helper -----------------------------------------------------

def encode_with_compressor(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    input_token_ids: torch.Tensor,
    input_token_pad: torch.Tensor,
    triple_pad: torch.Tensor,
) -> torch.Tensor:
    compressed = compressor(input_token_ids, input_token_pad, triple_pad)
    B, M = triple_pad.shape
    pad_3x = triple_pad.unsqueeze(-1).expand(B, M, 3).reshape(B, M * 3)
    model._cached_pad_mask = pad_3x
    return compressed


# -- Generation-based evaluation ---------------------------------------

@torch.no_grad()
def _encode_cached(compressor, model, ds, device, n_examples):
    n = min(n_examples, len(ds))
    return encode_with_compressor(
        compressor, model,
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    ), n


@torch.no_grad()
def run_generation_metrics(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    ds: CompressorDataset,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
) -> dict[str, float]:
    """Generation metrics with length-truncated predictions and per-length-bucket accuracy."""
    latent, n = _encode_cached(compressor, model, ds, device, n_examples)

    gen_ent_ids = model.generate_entity_ids(latent, n_steps=n_steps)
    gen_val_ids = model.generate_value_ids(latent, n_steps=n_steps)
    pred_ent_lens, pred_val_lens = model.predict_lengths(latent)
    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    ent_exact = val_exact = attr_match = 0
    ent_tok_correct = ent_tok_total = 0
    val_tok_correct = val_tok_total = 0
    ent_len_correct = val_len_correct = 0
    total = 0

    # Per-length-bucket tracking
    buckets = {"short": (1, 3), "medium": (4, 6), "long": (7, 12)}
    bucket_val_correct = {k: 0 for k in buckets}
    bucket_val_total = {k: 0 for k in buckets}
    bucket_ent_correct = {k: 0 for k in buckets}
    bucket_ent_total = {k: 0 for k in buckets}

    all_pred_values = []

    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total += 1
            if pred_attrs[i, m].item() == tgt_attrs[i, m].item():
                attr_match += 1

            # Entity: truncate to predicted length
            tgt_e_ids = ds._all_entity_token_ids[i, m]
            pred_e_ids = gen_ent_ids[i, m].cpu()
            n_real_e = (tgt_e_ids != 0).sum().item()
            pl_e = pred_ent_lens[i, m].item()
            if pl_e == n_real_e:
                ent_len_correct += 1
            cmp_e = min(pl_e, n_real_e)
            if cmp_e > 0:
                match = (pred_e_ids[:cmp_e] == tgt_e_ids[:cmp_e]).sum().item()
                ent_tok_correct += match
                ent_tok_total += n_real_e
            if pl_e == n_real_e and n_real_e > 0 and (pred_e_ids[:pl_e] == tgt_e_ids[:pl_e]).all():
                ent_exact += 1
                for bname, (lo, hi) in buckets.items():
                    if lo <= n_real_e <= hi:
                        bucket_ent_correct[bname] += 1

            for bname, (lo, hi) in buckets.items():
                if lo <= n_real_e <= hi:
                    bucket_ent_total[bname] += 1

            # Value: truncate to predicted length
            tgt_v_ids = ds._all_value_token_ids[i, m]
            pred_v_ids = gen_val_ids[i, m].cpu()
            n_real_v = (tgt_v_ids != 0).sum().item()
            pl_v = pred_val_lens[i, m].item()
            if pl_v == n_real_v:
                val_len_correct += 1
            cmp_v = min(pl_v, n_real_v)
            if cmp_v > 0:
                match = (pred_v_ids[:cmp_v] == tgt_v_ids[:cmp_v]).sum().item()
                val_tok_correct += match
                val_tok_total += n_real_v
            if pl_v == n_real_v and n_real_v > 0 and (pred_v_ids[:pl_v] == tgt_v_ids[:pl_v]).all():
                val_exact += 1
                for bname, (lo, hi) in buckets.items():
                    if lo <= n_real_v <= hi:
                        bucket_val_correct[bname] += 1

            for bname, (lo, hi) in buckets.items():
                if lo <= n_real_v <= hi:
                    bucket_val_total[bname] += 1

            pred_v_text = domain_tokenizer.decode(pred_v_ids[:pl_v], skip_special_tokens=True).strip()
            all_pred_values.append(pred_v_text)

    if total == 0:
        return {}

    result = {
        "gen_ent_tok": ent_tok_correct / max(ent_tok_total, 1),
        "gen_ent_exact": ent_exact / total,
        "gen_val_tok": val_tok_correct / max(val_tok_total, 1),
        "gen_val_exact": val_exact / total,
        "gen_attr": attr_match / total,
        "gen_total": total,
        "len_acc_ent": ent_len_correct / total,
        "len_acc_val": val_len_correct / total,
        "unique_values": len(set(all_pred_values)),
        "top_value_count": Counter(all_pred_values).most_common(1)[0][1] if all_pred_values else 0,
    }

    for bname in buckets:
        if bucket_val_total[bname] > 0:
            result[f"val_exact_{bname}"] = bucket_val_correct[bname] / bucket_val_total[bname]
            result[f"val_n_{bname}"] = bucket_val_total[bname]
        if bucket_ent_total[bname] > 0:
            result[f"ent_exact_{bname}"] = bucket_ent_correct[bname] / bucket_ent_total[bname]
            result[f"ent_n_{bname}"] = bucket_ent_total[bname]

    return result


@torch.no_grad()
def print_sample_predictions(
    compressor, model, ds, device, domain_tokenizer, n_examples=5, n_steps=10,
):
    n = min(n_examples, len(ds))
    latent = encode_with_compressor(
        compressor, model,
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    )
    gen_ent_ids = model.generate_entity_ids(latent, n_steps=n_steps)
    gen_val_ids = model.generate_value_ids(latent, n_steps=n_steps)
    pred_ent_lens, pred_val_lens = model.predict_lengths(latent)
    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)
    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    printed = 0
    print(f"\n  {'='*70}")
    print(f"  SAMPLE PREDICTIONS (n={n})")
    print(f"  {'='*70}")
    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m] or printed >= 5:
                continue
            printed += 1

            tgt_e_ids = ds._all_entity_token_ids[i, m]
            pred_e_raw = gen_ent_ids[i, m].cpu()
            pl_e = pred_ent_lens[i, m].item()
            pred_e_trunc = pred_e_raw[:pl_e].tolist()
            tgt_e_nonpad = [x for x in tgt_e_ids.tolist() if x != 0]
            tgt_e = domain_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e = domain_tokenizer.decode(pred_e_raw[:pl_e], skip_special_tokens=True).strip()
            e_match = "Y" if pred_e_trunc == tgt_e_nonpad else "N"

            tgt_a_id = tgt_attrs[i, m].item()
            pred_a_id = pred_attrs[i, m].item()
            tgt_a = ds.vocab.decode_id(tgt_a_id, "attr")
            pred_a = ds.vocab.decode_id(pred_a_id, "attr")
            a_match = "Y" if tgt_a == pred_a else "N"

            tgt_v_ids = ds._all_value_token_ids[i, m]
            pred_v_raw = gen_val_ids[i, m].cpu()
            pl_v = pred_val_lens[i, m].item()
            pred_v_trunc = pred_v_raw[:pl_v].tolist()
            tgt_v_nonpad = [x for x in tgt_v_ids.tolist() if x != 0]
            tgt_v = domain_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v = domain_tokenizer.decode(pred_v_raw[:pl_v], skip_special_tokens=True).strip()
            v_match = "Y" if pred_v_trunc == tgt_v_nonpad else "N"

            print(f"  [{i},{m}] ENTITY (len: pred={pl_e}, tgt={len(tgt_e_nonpad)})")
            print(f"    tgt:  {repr(tgt_e)}")
            print(f"    pred: {repr(pred_e)}  [{e_match}]")
            print(f"    tgt_ids:  {tgt_e_nonpad}")
            print(f"    pred_ids: {pred_e_trunc}")
            print(f"         ATTR")
            print(f"    tgt:  {repr(tgt_a)}")
            print(f"    pred: {repr(pred_a)}  [{a_match}]")
            print(f"         VALUE (len: pred={pl_v}, tgt={len(tgt_v_nonpad)})")
            print(f"    tgt:  {repr(tgt_v)}")
            print(f"    pred: {repr(pred_v)}  [{v_match}]")
            print(f"    tgt_ids:  {tgt_v_nonpad}")
            print(f"    pred_ids: {pred_v_trunc}")

            diffs = []
            maxlen = max(len(tgt_v_nonpad), len(pred_v_trunc))
            for idx in range(maxlen):
                t = tgt_v_nonpad[idx] if idx < len(tgt_v_nonpad) else -1
                p = pred_v_trunc[idx] if idx < len(pred_v_trunc) else -1
                if t != p:
                    diffs.append(f"pos{idx}: tgt={t} pred={p}")
            if diffs:
                print(f"    tok_diffs: {diffs}")
            print()
    print(f"  {'='*70}", flush=True)


@torch.no_grad()
def run_compressor_diagnostic(compressor, ds, device, n_examples=100):
    n = min(n_examples, len(ds))
    compressed = compressor(
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    )
    B, T, D = compressed.shape
    M = ds.max_triples
    triple_pad = ds._all_triple_pad[:n].to(device)
    pad_3x = triple_pad.unsqueeze(-1).expand(B, M, 3).reshape(B, M * 3)
    valid = ~pad_3x
    vectors = compressed[valid]
    if vectors.shape[0] < 2:
        return {}
    normed = F.normalize(vectors, dim=-1)
    N = min(vectors.shape[0], 500)
    sample = normed[:N]
    sims = sample @ sample.T
    mask = ~torch.eye(N, dtype=torch.bool, device=sims.device)
    return {
        "mean_cos": sims[mask].mean().item(),
        "max_cos": sims[mask].max().item(),
        "std_cos": sims[mask].std().item(),
        "norm_mean": vectors.norm(dim=-1).mean().item(),
        "norm_std": vectors.norm(dim=-1).std().item(),
    }


# -- Training phase ----------------------------------------------------

def run_phase(
    compressor, model, train_ds, ds_for_metrics, device, domain_tokenizer,
    out_dir, phase_name,
    t_min, t_max, bias_power,
    epochs, patience,
    batch_size, lr, weight_decay, denoise_steps,
    log_every, diagnostic_every,
    aux_ce_weight, length_weight,
):
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}  t in [{t_min}, {t_max}]  bias_power={bias_power}")
    print(f"{'='*60}")

    all_params = list(compressor.parameters()) + list(model.parameters())
    trainable = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    all_in_ids = train_ds._all_input_token_ids.to(device)
    all_in_pad = train_ds._all_input_token_pad.to(device)
    all_tri_pad = train_ds._all_triple_pad.to(device)
    all_tgt_a = train_ds._all_target_attr.to(device)
    all_tgt_pads = train_ds._all_target_pad_masks.to(device)
    all_ent_ids = train_ds._all_entity_token_ids.to(device)
    all_val_ids = train_ds._all_value_token_ids.to(device)
    n_train = all_in_ids.shape[0]

    best_val_exact = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # Initial diagnostics
    compressor.eval()
    model.eval()
    init_m = run_generation_metrics(
        compressor, model, ds_for_metrics, device, domain_tokenizer,
        n_examples=64, n_steps=denoise_steps,
    )
    print(f"  init: ve={init_m.get('gen_val_exact',0):.3f} "
          f"vt={init_m.get('gen_val_tok',0):.3f} "
          f"ee={init_m.get('gen_ent_exact',0):.3f} "
          f"len_v={init_m.get('len_acc_val',0):.3f}", flush=True)

    print(f"\nTraining for up to {epochs} epochs "
          f"(patience={'off' if patience <= 0 else patience})...", flush=True)

    for epoch in range(1, epochs + 1):
        compressor.train()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            B_batch = idx.shape[0]
            timestep = sample_timestep(B_batch, device, t_min, t_max, bias_power)

            latent = encode_with_compressor(
                compressor, model,
                all_in_ids[idx], all_in_pad[idx], all_tri_pad[idx],
            )
            loss, _ = compute_loss(
                model, latent,
                all_tgt_a[idx], all_tgt_pads[idx],
                all_ent_ids[idx], all_val_ids[idx],
                timestep=timestep,
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
            compressor.eval()
            model.eval()

            # Train loss metrics
            n_ev = min(len(train_ds), 100)
            with torch.no_grad():
                ev_latent = encode_with_compressor(
                    compressor, model,
                    train_ds._all_input_token_ids[:n_ev].to(device),
                    train_ds._all_input_token_pad[:n_ev].to(device),
                    train_ds._all_triple_pad[:n_ev].to(device),
                )
                _, ev_m = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    aux_ce_weight=aux_ce_weight,
                    length_weight=length_weight,
                )

                ht_t = torch.full((n_ev,), 0.9, device=device)
                _, ht_m = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    timestep=ht_t,
                    aux_ce_weight=aux_ce_weight,
                    length_weight=length_weight,
                )

            # Generation metrics
            gen_m = run_generation_metrics(
                compressor, model, ds_for_metrics, device, domain_tokenizer,
                n_examples=64, n_steps=denoise_steps,
            )

            gve = gen_m.get("gen_val_exact", 0)
            gvt = gen_m.get("gen_val_tok", 0)

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent {ev_m.get('acc_entity_tok', 0):.3f}"
                f" cos {ev_m.get('cos_entity', 0):.3f}"
                f" | attr {ev_m.get('acc_attr', 0):.3f}"
                f" | val {ev_m.get('acc_value_tok', 0):.3f}"
                f" cos {ev_m.get('cos_value', 0):.3f}"
            )
            if "proj_acc_value" in ev_m:
                log += f" proj {ev_m['proj_acc_value']:.3f}"
            log += (
                f" | ht_e {ht_m.get('acc_entity_tok', 0):.3f}"
                f" ht_v {ht_m.get('acc_value_tok', 0):.3f}"
                f" || ve {gve:.3f} vt {gvt:.3f}"
                f" ee {gen_m.get('gen_ent_exact', 0):.3f}"
                f" | attr {gen_m.get('gen_attr', 0):.3f}"
                f" | len_v {gen_m.get('len_acc_val', 0):.3f}"
                f" len_e {gen_m.get('len_acc_ent', 0):.3f}"
                f" | u_v {gen_m.get('unique_values', 0)}"
            )

            for bname in ["short", "medium", "long"]:
                key = f"val_exact_{bname}"
                if key in gen_m:
                    log += f" | {bname[0]}:{gen_m[key]:.2f}({gen_m.get(f'val_n_{bname}', 0)})"

            if gve > best_val_exact:
                best_val_exact = gve
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "compressor": compressor.state_dict(),
                    "model": model.state_dict(),
                }, phase_dir / "model_best.pt")
                log += " *"
            else:
                epochs_without_improvement += log_every

            print(log, flush=True)

            entry = {
                "epoch": epoch,
                "phase": phase_name,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in ev_m.items()},
                "gen_metrics": gen_m,
            }

            if epoch % diagnostic_every == 0:
                comp_diag = run_compressor_diagnostic(compressor, ds_for_metrics, device)
                entry["compressor_diag"] = comp_diag
                print(f"  COMPRESSOR: mean_cos={comp_diag.get('mean_cos',0):.4f} "
                      f"std_cos={comp_diag.get('std_cos',0):.4f} "
                      f"norm={comp_diag.get('norm_mean',0):.2f}"
                      f"+/-{comp_diag.get('norm_std',0):.2f}", flush=True)

            if epoch == 1 or epoch % diagnostic_every == 0:
                train_recon = run_generation_metrics(
                    compressor, model, train_ds, device, domain_tokenizer,
                    n_examples=100, n_steps=denoise_steps,
                )
                entry["train_recon"] = train_recon
                print(f"  RECON (train): ve={train_recon.get('gen_val_exact',0):.3f} "
                      f"vt={train_recon.get('gen_val_tok',0):.3f} "
                      f"ee={train_recon.get('gen_ent_exact',0):.3f} "
                      f"et={train_recon.get('gen_ent_tok',0):.3f} "
                      f"len_v={train_recon.get('len_acc_val',0):.3f}", flush=True)

                print_sample_predictions(
                    compressor, model, ds_for_metrics, device, domain_tokenizer,
                    n_examples=5, n_steps=denoise_steps,
                )

            history.append(entry)

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best ve {best_val_exact:.4f} at epoch {best_epoch})", flush=True)
                break

    torch.save({
        "compressor": compressor.state_dict(),
        "model": model.state_dict(),
    }, phase_dir / "model_final.pt")
    with open(phase_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{phase_name} done. Best ve: {best_val_exact:.4f} at epoch {best_epoch}")
    return best_val_exact, best_epoch


# -- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train v15: Fresh compressor/expander with curriculum")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--domain-tokenizer", type=str, required=True)
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--denoiser-layers", type=int, default=1)
    ap.add_argument("--denoiser-heads", type=int, default=4)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--max-value-tokens", type=int, default=12)
    ap.add_argument("--compressor-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--device", type=str, default=None)
    # Curriculum
    ap.add_argument("--phase1-epochs", type=int, default=100)
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--phase2-patience", type=int, default=50)
    ap.add_argument("--phase2-bias-power", type=float, default=2.0)
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

    domain_tokenizer = DomainBPETokenizer.load(args.domain_tokenizer, max_length=args.max_value_tokens)
    domain_vocab_size = domain_tokenizer.vocab_size
    mask_token_id = domain_tokenizer.mask_token_id

    config = ModelConfig.from_profile(args.config)
    denoiser_dim = config.d_model

    print(f"=== v15 Fresh Compressor/Expander with Curriculum ===")
    print(f"Device: {device}")
    print(f"Denoiser: {args.denoiser_layers}L, {denoiser_dim}d, {args.denoiser_heads}H")
    print(f"Domain BPE vocab: {domain_vocab_size} tokens")
    print(f"Phase 1: t in [0.7, 1.0], {args.phase1_epochs} epochs, no early stop")
    print(f"Phase 2: t in [0.0, 1.0], {args.phase2_epochs} epochs, "
          f"patience={args.phase2_patience}, bias_power={args.phase2_bias_power}")

    # Vocab
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

    # Model (fresh)
    print(f"Building model (fresh init)...")
    model = DiffusionWorldModel(
        config, config.d_model, vocab,
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
        use_cross_attention=True,
        use_adaln=True,
        use_continuous_noise=True,
        normalize_noise=True,
        alpha_min=args.alpha_min,
        unified_decoder=True,
        wspace=True,
        use_mse_prediction=True,
        cond_drop_prob=0.0,
        use_decode_proj=True,
    ).to(device)

    # Freeze unused encoder and dynamics
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.dynamics.parameters():
        p.requires_grad = False

    # Token embeddings: random unit-norm, zero specials
    print("Initializing token embeddings (random, unit-norm)...")
    decoder = model._get_decoder("entity")
    with torch.no_grad():
        nn.init.normal_(decoder.token_emb.weight, std=0.02)
        decoder.token_emb.weight.data = F.normalize(decoder.token_emb.weight.data, dim=-1)
        for special_id in (domain_tokenizer.pad_token_id,
                           domain_tokenizer.mask_token_id,
                           domain_tokenizer.unk_token_id):
            if special_id is not None:
                decoder.token_emb.weight.data[special_id] = 0.0

    # Compressor (shares frozen token_emb with decoder)
    print(f"Building compressor...")
    compressor = TripleCompressor(
        token_emb=decoder.token_emb,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=args.compressor_layers,
        n_roles=3,
        max_seq_len=args.max_value_tokens,
        max_triples=config.max_triples,
        dropout=args.dropout,
    ).to(device)

    total_trainable = compressor.trainable_param_count() + model.trainable_param_count()
    print(f"  Compressor: {compressor.trainable_param_count():,} trainable")
    print(f"  Model: {model.trainable_param_count():,} trainable")
    print(f"  Total: {total_trainable:,} trainable")
    print(f"  Length head: {sum(p.numel() for p in model.length_head.parameters()):,} params")

    # Datasets
    print("Building datasets...")
    train_ds = CompressorDataset(
        train_path, vocab, domain_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=args.max_value_tokens,
    )
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = CompressorDataset(
            test_path, vocab, domain_tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=args.max_value_tokens,
        )
        print(f"  Test: {len(test_ds)} examples")
    ds_for_metrics = test_ds or train_ds

    # Save config
    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "architecture": "v15_fresh_compressor_expander",
            "denoiser_layers": args.denoiser_layers,
            "denoiser_dim": denoiser_dim,
            "denoiser_heads": args.denoiser_heads,
            "compressor_layers": args.compressor_layers,
            "max_value_tokens": args.max_value_tokens,
            "denoise_steps": args.denoise_steps,
            "dropout": args.dropout,
            "alpha_min": args.alpha_min,
            "domain_vocab_size": domain_vocab_size,
            "aux_ce_weight": args.aux_ce_weight,
            "length_weight": args.length_weight,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
            "phase2_patience": args.phase2_patience,
            "phase2_bias_power": args.phase2_bias_power,
            "identity_advance": True,
            "curriculum": True,
        }, f, indent=2)

    # Phase 1: high noise only, no early stopping
    p1_ve, p1_ep = run_phase(
        compressor, model, train_ds, ds_for_metrics, device, domain_tokenizer,
        out_dir=out_dir, phase_name="phase1",
        t_min=0.7, t_max=1.0, bias_power=1.0,
        epochs=args.phase1_epochs, patience=0,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every,
        diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight,
        length_weight=args.length_weight,
    )

    # Phase 2: full range, importance sampling, early stop on val_exact
    p2_ve, p2_ep = run_phase(
        compressor, model, train_ds, ds_for_metrics, device, domain_tokenizer,
        out_dir=out_dir, phase_name="phase2",
        t_min=0.0, t_max=1.0, bias_power=args.phase2_bias_power,
        epochs=args.phase2_epochs, patience=args.phase2_patience,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every,
        diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight,
        length_weight=args.length_weight,
    )

    print(f"\n{'='*60}")
    print(f"DONE -- v15 Fresh Compressor/Expander")
    print(f"  Phase 1: best ve={p1_ve:.4f} at epoch {p1_ep}")
    print(f"  Phase 2: best ve={p2_ve:.4f} at epoch {p2_ep}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
