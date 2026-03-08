#!/usr/bin/env python3
"""Train v14: Compressor/Expander identity advance.

Compressor: BPE tokens -> self-attention -> role-conditioned pool -> 256d per slot
Expander: existing diffusion decoder (unchanged)
Dynamics: identity (compressor output IS expander input)

Validates that the compression bottleneck preserves enough information.

Curriculum:
  Phase 1: t in [0.7, 1.0] -- 100 epochs (no early stopping)
  Phase 2: t in [0.0, 1.0] -- 200 epochs (importance sampling, early stop)

Usage:
    uv run python scripts/train_v14_compress.py \
        --data-dir data/webnlg \
        --out-dir results/v14_compress_webnlg \
        --domain-tokenizer data/webnlg/domain_bpe_tokenizer.json \
        --config base --use-adaln --use-continuous-noise --unified-decoder --wspace
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
from twm.compressor import TripleCompressor
from twm.compressor_dataset import CompressorDataset


# -- Timestep sampling ------------------------------------------------

def sample_timestep(B: int, device: torch.device, t_min: float, t_max: float) -> torch.Tensor:
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


# -- Loss functions (same as v13) -------------------------------------

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

    # MSE only on content tokens — pad embedding is zero so pad positions
    # naturally produce near-zero outputs that are invisible to self-attention
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


def _ce_loss(logits, target_mask, target_ids, pad_mask, B, M):
    if logits.shape[0] == 0 or not target_mask.any():
        return torch.tensor(0.0, device=logits.device), {}

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]

    # CE over ALL positions — pad token ID 0 is a valid target
    logits_all = logits.reshape(-1, logits.shape[-1])
    targets_all = tgt_valid.reshape(-1)
    loss = F.cross_entropy(logits_all, targets_all)

    # Metrics on content tokens only
    metrics = {}
    with torch.no_grad():
        non_pad = targets_all != 0
        if non_pad.any():
            preds = logits_all.argmax(-1)
            metrics["acc"] = (preds[non_pad] == targets_all[non_pad]).float().mean().item()
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
    use_ce = not decoder.use_mse_prediction

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

    # Length prediction: how many real BPE tokens per entity/value slot
    length_pred = model.forward_lengths(latent)  # (B, M, 2)
    ent_real = (entity_token_ids != 0).sum(dim=-1).float()  # (B, M)
    val_real = (value_token_ids != 0).sum(dim=-1).float()  # (B, M)
    length_target = torch.stack([ent_real, val_real], dim=-1)  # (B, M, 2)
    # Only compute on non-pad triples
    valid_mask = ~tgt_pad  # (B, M)
    if valid_mask.any():
        length_loss = F.mse_loss(
            length_pred[valid_mask], length_target[valid_mask],
        )
    else:
        length_loss = torch.tensor(0.0, device=latent.device)
    metrics["length_loss"] = length_loss.item()

    total_loss = attr_loss + entity_loss + value_loss + 0.1 * length_loss
    if use_aux and not use_ce:
        total_loss = total_loss + aux_ce_weight * (entity_aux + value_aux)
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


# -- Encode helper: compressor -> latent -------------------------------

def encode_with_compressor(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    input_token_ids: torch.Tensor,
    input_token_pad: torch.Tensor,
    triple_pad: torch.Tensor,
) -> torch.Tensor:
    """Run compressor and set up model state for downstream calls.

    Returns latent (B, M*3, d_model) -- identity advance, no dynamics.
    """
    compressed = compressor(input_token_ids, input_token_pad, triple_pad)
    # Set cached pad mask so _extract_triple_context works
    # For non-token-level mode, pad_mask is (B, M*3)
    B, M = triple_pad.shape
    pad_3x = triple_pad.unsqueeze(-1).expand(B, M, 3).reshape(B, M * 3)
    model._cached_pad_mask = pad_3x
    return compressed


# -- Eval / generation ------------------------------------------------

@torch.no_grad()
def run_eval_generation(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    ds: CompressorDataset,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
    guidance_scale: float = 1.0,
) -> dict[str, float]:
    latent, n = _encode_cached(compressor, model, ds, device, n_examples)
    return _eval_from_latent(model, latent, ds, device, domain_tokenizer, n, n_steps, guidance_scale)


@torch.no_grad()
def print_sample_predictions(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    ds: CompressorDataset,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    n_examples: int = 5,
    n_steps: int = 10,
    guidance_scale: float = 1.0,
):
    n = min(n_examples, len(ds))

    latent = encode_with_compressor(
        compressor, model,
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    )
    gen_ent_ids = model.generate_entity_ids(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    gen_val_ids = model.generate_value_ids(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    pred_ent_lens, pred_val_lens = model.predict_lengths(latent)

    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)
    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    printed = 0
    print(f"\n  {'='*70}")
    print(f"  SAMPLE PREDICTIONS (n={n}, gs={guidance_scale:.1f})")
    print(f"  {'='*70}")
    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m] or printed >= 5:
                continue
            printed += 1

            # Entity: truncate to predicted length
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

            # Value: truncate to predicted length
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

            maxlen = max(len(tgt_v_nonpad), len(pred_v_trunc))
            diffs = []
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
def _encode_cached(compressor, model, ds, device, n_examples):
    """Encode once, return latent for reuse across guidance scales."""
    n = min(n_examples, len(ds))
    return encode_with_compressor(
        compressor, model,
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    ), n


@torch.no_grad()
def _eval_from_latent(
    model, latent, ds, device, domain_tokenizer, n, n_steps, guidance_scale,
):
    """Run generation eval from pre-computed latent.

    Uses raw token IDs from NN decode — no string roundtrip.
    """
    gen_ent_ids = model.generate_entity_ids(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    gen_val_ids = model.generate_value_ids(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    # Predicted lengths for truncation
    pred_ent_lens, pred_val_lens = model.predict_lengths(latent)
    # Still generate text for diversity/display metrics
    gen_entities = model.generate_entities(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    gen_values = model.generate_values(latent, n_steps=n_steps, guidance_scale=guidance_scale)
    discrete_logits = model.forward_discrete(latent)
    pred_attrs = discrete_logits["attr"].argmax(-1)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = tgt_pad.shape[1]

    ent_exact = val_exact = attr_match = 0
    ent_tok_correct = ent_tok_total = 0
    val_tok_correct = val_tok_total = 0
    # Truncated metrics: use predicted length to truncate, then compare
    ent_trunc_exact = val_trunc_exact = 0
    ent_trunc_tok_correct = ent_trunc_tok_total = 0
    val_trunc_tok_correct = val_trunc_tok_total = 0
    ent_len_correct = val_len_correct = 0
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

            # Entity: compare raw IDs at non-pad positions (original metric)
            tgt_e_ids = ds._all_entity_token_ids[i, m]
            pred_e_ids = gen_ent_ids[i, m].cpu()  # (S,)
            all_pred_entities.append(gen_entities[i][m])
            n_real_e = (tgt_e_ids != 0).sum().item()
            non_pad_e = tgt_e_ids != 0
            if non_pad_e.any():
                tok_match = pred_e_ids[non_pad_e] == tgt_e_ids[non_pad_e]
                ent_tok_correct += tok_match.sum().item()
                ent_tok_total += non_pad_e.sum().item()
                if tok_match.all():
                    ent_exact += 1

            # Entity truncated: use predicted length
            pl_e = pred_ent_lens[i, m].item()
            if pl_e == n_real_e:
                ent_len_correct += 1
            trunc_n = min(pl_e, n_real_e)
            if trunc_n > 0:
                match = (pred_e_ids[:trunc_n] == tgt_e_ids[:trunc_n]).sum().item()
                ent_trunc_tok_correct += match
                ent_trunc_tok_total += n_real_e
                if pl_e == n_real_e and (pred_e_ids[:pl_e] == tgt_e_ids[:pl_e]).all():
                    ent_trunc_exact += 1

            # Value: compare raw IDs at non-pad positions (original metric)
            tgt_v_ids = ds._all_value_token_ids[i, m]
            pred_v_ids = gen_val_ids[i, m].cpu()  # (S,)
            all_pred_values.append(gen_values[i][m])
            n_real_v = (tgt_v_ids != 0).sum().item()
            non_pad_v = tgt_v_ids != 0
            if non_pad_v.any():
                tok_match = pred_v_ids[non_pad_v] == tgt_v_ids[non_pad_v]
                val_tok_correct += tok_match.sum().item()
                val_tok_total += non_pad_v.sum().item()
                if tok_match.all():
                    val_exact += 1

            # Value truncated: use predicted length
            pl_v = pred_val_lens[i, m].item()
            if pl_v == n_real_v:
                val_len_correct += 1
            trunc_n = min(pl_v, n_real_v)
            if trunc_n > 0:
                match = (pred_v_ids[:trunc_n] == tgt_v_ids[:trunc_n]).sum().item()
                val_trunc_tok_correct += match
                val_trunc_tok_total += n_real_v
                if pl_v == n_real_v and (pred_v_ids[:pl_v] == tgt_v_ids[:pl_v]).all():
                    val_trunc_exact += 1

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
        # Truncated metrics (using predicted lengths)
        "trunc_ent_tok": ent_trunc_tok_correct / max(ent_trunc_tok_total, 1),
        "trunc_ent_exact": ent_trunc_exact / total,
        "trunc_val_tok": val_trunc_tok_correct / max(val_trunc_tok_total, 1),
        "trunc_val_exact": val_trunc_exact / total,
        "len_acc_ent": ent_len_correct / total,
        "len_acc_val": val_len_correct / total,
    }



@torch.no_grad()
def run_compressor_diagnostic(
    compressor: TripleCompressor,
    ds: CompressorDataset,
    device: torch.device,
    n_examples: int = 100,
) -> dict[str, float]:
    """Analyze compressor output distribution -- detect collapse."""
    n = min(n_examples, len(ds))
    compressed = compressor(
        ds._all_input_token_ids[:n].to(device),
        ds._all_input_token_pad[:n].to(device),
        ds._all_triple_pad[:n].to(device),
    )  # (n, M*3, D)

    # Flatten to all non-pad vectors
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

    # Token position embedding analysis
    pos_embs = compressor.token_pos_emb.weight.detach()  # (S, D)
    pos_normed = F.normalize(pos_embs, dim=-1)
    pos_sims = pos_normed @ pos_normed.T  # (S, S)
    S = pos_embs.shape[0]
    pos_mask = ~torch.eye(S, dtype=torch.bool, device=pos_sims.device)

    return {
        "mean_cos": sims[mask].mean().item(),
        "max_cos": sims[mask].max().item(),
        "min_cos": sims[mask].min().item(),
        "std_cos": sims[mask].std().item(),
        "norm_mean": vectors.norm(dim=-1).mean().item(),
        "norm_std": vectors.norm(dim=-1).std().item(),
        "n_vectors": vectors.shape[0],
        "pos_mean_cos": pos_sims[pos_mask].mean().item(),
        "pos_max_cos": pos_sims[pos_mask].max().item(),
        "pos_norm_mean": pos_embs.norm(dim=-1).mean().item(),
    }


# -- Training phase ----------------------------------------------------

def run_training(
    compressor: TripleCompressor,
    model: DiffusionWorldModel,
    train_ds: CompressorDataset,
    test_ds: CompressorDataset | None,
    device: torch.device,
    domain_tokenizer: DomainBPETokenizer,
    out_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    denoise_steps: int,
    log_every: int,
    diagnostic_every: int,
    aux_ce_weight: float,
) -> tuple[float, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training encoder/decoder  t in [0.0, 1.0]")
    print(f"{'='*60}")

    # Collect all trainable params from both compressor and model
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

    best_gen_val_tok = -1.0
    best_gen_epoch = 0
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
        compressor.eval()
        model.train(False)
        _emb_diagnostic("start")
        init_metrics = run_eval_generation(
            compressor, model, test_ds, device, domain_tokenizer,
            n_examples=64, n_steps=denoise_steps,
        )
        print(f"  init: gv_tok={init_metrics.get('gen_val_tok',0):.3f} "
              f"ge_tok={init_metrics.get('gen_ent_tok',0):.3f}", flush=True)

        comp_diag = run_compressor_diagnostic(compressor, test_ds, device)
        print(f"  COMPRESSOR init: mean_cos={comp_diag.get('mean_cos',0):.4f} "
              f"std_cos={comp_diag.get('std_cos',0):.4f} "
              f"norm={comp_diag.get('norm_mean',0):.2f}", flush=True)
        print(f"  TOK_POS init: mean_cos={comp_diag.get('pos_mean_cos',0):.4f} "
              f"max_cos={comp_diag.get('pos_max_cos',0):.4f} "
              f"norm={comp_diag.get('pos_norm_mean',0):.4f}", flush=True)

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...", flush=True)

    for epoch in range(1, epochs + 1):
        compressor.train()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]
            B_batch = idx.shape[0]
            timestep = sample_timestep(B_batch, device, 0.0, 1.0)

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
            model.train(False)

            # Train metrics
            n_ev = min(len(train_ds), 100)
            with torch.no_grad():
                ev_latent = encode_with_compressor(
                    compressor, model,
                    train_ds._all_input_token_ids[:n_ev].to(device),
                    train_ds._all_input_token_pad[:n_ev].to(device),
                    train_ds._all_triple_pad[:n_ev].to(device),
                )
                _, ev_metrics = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    aux_ce_weight=aux_ce_weight,
                )

                # High-t diagnostic: measure accuracy at generation-relevant noise
                ht_timestep = torch.full((n_ev,), 0.9, device=device)
                _, ht_metrics = compute_loss(
                    model, ev_latent,
                    train_ds._all_target_attr[:n_ev].to(device),
                    train_ds._all_target_pad_masks[:n_ev].to(device),
                    train_ds._all_entity_token_ids[:n_ev].to(device),
                    train_ds._all_value_token_ids[:n_ev].to(device),
                    timestep=ht_timestep,
                    aux_ce_weight=aux_ce_weight,
                )

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | ent {ev_metrics.get('acc_entity_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_entity', 0):.3f}"
                f" | attr {ev_metrics.get('acc_attr', 0):.3f}"
                f" | val {ev_metrics.get('acc_value_tok', 0):.3f}"
                f" cos {ev_metrics.get('cos_value', 0):.3f}"
            )
            if "proj_acc_value" in ev_metrics:
                log += f" proj {ev_metrics['proj_acc_value']:.3f}"
            log += (
                f" | ht_e {ht_metrics.get('acc_entity_tok', 0):.3f}"
                f" ht_v {ht_metrics.get('acc_value_tok', 0):.3f}"
            )

            eval_ds = test_ds or train_ds
            gen_metrics = run_eval_generation(
                compressor, model, eval_ds, device, domain_tokenizer,
                n_examples=64, n_steps=denoise_steps,
            )

            gvt = gen_metrics.get("gen_val_tok", 0)
            gve = gen_metrics.get("gen_val_exact", 0)

            log += (
                f" || gv_tok {gvt:.3f} gv_ex {gve:.3f}"
                f" | g_attr {gen_metrics.get('gen_attr', 0):.3f}"
                f" | u_v {gen_metrics.get('unique_values', 0)}"
                f" top1 {gen_metrics.get('top_value_count', 0)}"
                f" | trunc_vt {gen_metrics.get('trunc_val_tok', 0):.3f}"
                f" trunc_ve {gen_metrics.get('trunc_val_exact', 0):.3f}"
                f" len_v {gen_metrics.get('len_acc_val', 0):.3f}"
            )

            if gvt > best_gen_val_tok:
                best_gen_val_tok = gvt
                best_gen_epoch = epoch
                epochs_without_improvement = 0
                torch.save({
                    "compressor": compressor.state_dict(),
                    "model": model.state_dict(),
                }, out_dir / "model_best.pt")
                log += " *"
            else:
                epochs_without_improvement += log_every

            if epoch <= 50 or epoch % diagnostic_every == 0:
                _emb_diagnostic(f"epoch={epoch}")

            print(log, flush=True)

            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                **{f"train_{k}": v for k, v in ev_metrics.items()},
                "gen_metrics": gen_metrics,
            }

            if diagnostic_every > 0 and epoch % diagnostic_every == 0:
                comp_diag = run_compressor_diagnostic(compressor, eval_ds, device)
                entry["compressor_diag"] = comp_diag
                print(f"  COMPRESSOR: mean_cos={comp_diag.get('mean_cos',0):.4f} "
                      f"std_cos={comp_diag.get('std_cos',0):.4f} "
                      f"max_cos={comp_diag.get('max_cos',0):.4f} "
                      f"norm={comp_diag.get('norm_mean',0):.2f}"
                      f"+/-{comp_diag.get('norm_std',0):.2f}",
                      flush=True)
                print(f"  TOK_POS: mean_cos={comp_diag.get('pos_mean_cos',0):.4f} "
                      f"max_cos={comp_diag.get('pos_max_cos',0):.4f} "
                      f"norm={comp_diag.get('pos_norm_mean',0):.4f}",
                      flush=True)

            if epoch == 1 or epoch % diagnostic_every == 0:
                train_recon = run_eval_generation(
                    compressor, model, train_ds, device, domain_tokenizer,
                    n_examples=100, n_steps=denoise_steps,
                )
                entry["train_recon"] = train_recon
                print(f"  RECON (train): val_tok={train_recon.get('gen_val_tok',0):.3f} "
                      f"val_ex={train_recon.get('gen_val_exact',0):.3f} "
                      f"ent_tok={train_recon.get('gen_ent_tok',0):.3f} "
                      f"ent_ex={train_recon.get('gen_ent_exact',0):.3f} "
                      f"| trunc_vt={train_recon.get('trunc_val_tok',0):.3f} "
                      f"trunc_ve={train_recon.get('trunc_val_exact',0):.3f} "
                      f"len_v={train_recon.get('len_acc_val',0):.3f} "
                      f"len_e={train_recon.get('len_acc_ent',0):.3f}", flush=True)

                print_sample_predictions(
                    compressor, model, eval_ds, device, domain_tokenizer,
                    n_examples=5, n_steps=denoise_steps,
                )

            history.append(entry)

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best gvt {best_gen_val_tok:.4f} at epoch {best_gen_epoch})", flush=True)
                break

    torch.save({
        "compressor": compressor.state_dict(),
        "model": model.state_dict(),
    }, out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best gvt: {best_gen_val_tok:.4f} at epoch {best_gen_epoch}")

    return best_gen_val_tok, best_gen_epoch


# -- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train v14: Compressor/Expander identity advance")
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
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use-adaln", action="store_true")
    ap.add_argument("--use-continuous-noise", action="store_true")
    ap.add_argument("--no-normalize-noise", action="store_true")
    ap.add_argument("--no-cross-attention", action="store_true")
    ap.add_argument("--alpha-min", type=float, default=0.01)
    ap.add_argument("--unified-decoder", action="store_true")
    ap.add_argument("--wspace", action="store_true")
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--use-decode-proj", action="store_true")
    ap.add_argument("--use-output-head", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    # Resume
    ap.add_argument("--resume-checkpoint", type=str, default=None)
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

    use_proj = args.use_decode_proj
    use_output_head = args.use_output_head

    print(f"=== v14 Compressor/Expander Identity Advance ===")
    print(f"Device: {device}")
    print(f"W-space: {args.wspace} (denoiser_dim={denoiser_dim})")
    if use_output_head:
        print(f"Prediction: CE through learned output head")
    elif use_proj:
        print(f"Prediction: MSE x0 + {args.aux_ce_weight}x aux CE through decode_proj")
    else:
        print(f"Prediction: MSE x0 (no decode_proj)")
    print(f"Domain BPE vocab: {domain_vocab_size} tokens")
    print(f"Compressor: {args.compressor_layers}L self-attention + role-conditioned pool")
    print(f"NO DYNAMICS -- identity advance (compressor -> expander directly)")

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

    # Build model (decoder/expander + attr head).
    # SentenceEncoder is created but unused -- compressor replaces it.
    # st_dim matches d_model so proj becomes Identity (unused anyway).
    print(f"Building model (expander/decoder)...")
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
        use_cross_attention=use_cross_attention,
        use_adaln=args.use_adaln,
        use_continuous_noise=args.use_continuous_noise,
        normalize_noise=normalize_noise,
        alpha_min=args.alpha_min,
        unified_decoder=args.unified_decoder,
        wspace=args.wspace,
        use_mse_prediction=not use_output_head,
        cond_drop_prob=0.0,
        use_decode_proj=use_proj,
    ).to(device)

    # Freeze the unused encoder and dynamics
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.dynamics.parameters():
        p.requires_grad = False

    # Initialize token embeddings: random unit-norm vectors.
    # Uniform norm means NN cosine decode only cares about direction.
    # Compressor and decoder jointly learn what each direction means.
    print("Initializing token embeddings (random, unit-norm)...")
    decoder = model._get_decoder("entity")
    with torch.no_grad():
        nn.init.normal_(decoder.token_emb.weight, std=0.02)
        decoder.token_emb.weight.data = F.normalize(decoder.token_emb.weight.data, dim=-1)
        # Zero out special tokens — pad must be zero so it contributes nothing
        # to self-attention during generation (prevents attractor/repetition)
        for special_id in (domain_tokenizer.pad_token_id,
                           domain_tokenizer.mask_token_id,
                           domain_tokenizer.unk_token_id):
            if special_id is not None:
                decoder.token_emb.weight.data[special_id] = 0.0
    norms = decoder.token_emb.weight.data.norm(dim=-1)
    print(f"  {domain_vocab_size} embeddings, {config.d_model}d")
    print(f"  Norm stats: mean={norms.mean():.3f}, std={norms.std():.3f}")

    # Build compressor -- shares frozen token_emb with decoder
    print(f"Building compressor (shared token_emb with decoder)...")
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

    # Resume from checkpoint
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        print(f"Loading resume checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "compressor" in sd:
            compressor.load_state_dict(sd["compressor"], strict=False)
            model.load_state_dict(sd["model"], strict=False)
            print(f"  Loaded compressor + model state")
        else:
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded model state (no compressor state found)")

    print(f"  Compressor: {args.compressor_layers}L, {config.d_model}d, {config.n_heads}H")
    print(f"  Compressor: {compressor.trainable_param_count():,} trainable params")
    print(f"  Decoder: {args.denoiser_layers}L, {denoiser_dim}d, {args.denoiser_heads}H")

    total_trainable = compressor.trainable_param_count() + model.trainable_param_count()
    print(f"  Total trainable: {total_trainable:,}")

    print("\n  Trainable parameters (compressor):")
    for name, param in compressor.named_parameters():
        if param.requires_grad:
            print(f"    compressor.{name}: {list(param.shape)} = {param.numel():,}")
    print("  Trainable parameters (model/decoder):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    model.{name}: {list(param.shape)} = {param.numel():,}")
    print(flush=True)

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

    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "n_proj_tokens": 3,
            "denoiser_layers": args.denoiser_layers,
            "denoiser_dim": denoiser_dim,
            "denoiser_heads": args.denoiser_heads,
            "compressor_layers": args.compressor_layers,
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
            "unified_decoder": args.unified_decoder,
            "domain_tokenizer": args.domain_tokenizer,
            "domain_vocab_size": domain_vocab_size,
            "wspace": args.wspace,
            "use_mse_prediction": not use_output_head,
            "cond_drop_prob": 0.0,
            "use_decode_proj": use_proj,
            "aux_ce_weight": args.aux_ce_weight,
            "identity_advance": True,
            "architecture": "compressor_expander",
        }, f, indent=2)

    best_gvt, best_ep = run_training(
        compressor=compressor,
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        domain_tokenizer=domain_tokenizer,
        out_dir=out_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every,
        diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight,
    )

    print(f"\n{'='*60}")
    print(f"DONE -- Compressor/Expander Identity Advance")
    print(f"  Best gvt: {best_gvt:.4f} at epoch {best_ep}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
