#!/usr/bin/env python3
"""Train v16: Multimodal triple ↔ text with shared 256d bottleneck.

Four tasks on WebNLG data:
  1. triple → triple (identity, anchor task)
  2. triple → text   (generation)
  3. text → triple   (extraction)
  4. text → text     (paraphrase, lower weight)

Curriculum:
  Phase 1: t in [0.7, 1.0], fixed epochs, no early stopping
  Phase 2: t in [0.0, 1.0], importance sampling, early stop

Usage:
    uv run python scripts/train_v16_multimodal.py \
        --data-dir data/webnlg_multi \
        --out-dir results/v16_multimodal \
        --tokenizer data/webnlg_multi/shared_bpe_tokenizer.json
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from twm.config import ModelConfig, PROFILES
from twm.phrase_vocab import PhraseVocab
from twm.domain_bpe import DomainBPETokenizer
from twm.multimodal_model import MultimodalWorldModel
from twm.webnlg_dataset import (
    WebNLGMultimodalDataset,
    TASK_TRIPLE_TRIPLE, TASK_TRIPLE_TEXT,
    TASK_TEXT_TRIPLE, TASK_TEXT_TEXT,
    TASK_NAMES,
)
from twm.diffusion_decoder import importance_sample_timesteps


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

def _mse_loss(pred_emb, target_ids, pad_mask, B, M, token_emb):
    """MSE loss in embedding space for diffusion expander outputs."""
    if pred_emb.shape[0] == 0:
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


def _text_mse_loss(pred_emb, target_ids, text_pad_mask, token_emb):
    """MSE loss for text expander outputs."""
    non_pad = ~text_pad_mask
    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device), {}

    target_clean = token_emb(target_ids)
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
        tgt_flat_ids = target_ids[non_pad]
        metrics["acc"] = (nn_ids == tgt_flat_ids).float().mean().item()

    return loss, metrics


def _aux_ce_loss(pred_emb, target_ids, pad_mask, B, M, decoder):
    """Auxiliary CE through decode_proj for triple expander."""
    if pred_emb.shape[0] == 0:
        return torch.tensor(0.0, device=pred_emb.device)

    pad_flat = pad_mask.reshape(B * M)
    valid = ~pad_flat
    tgt_valid = target_ids.reshape(B * M, -1)[valid]
    non_pad = tgt_valid != 0

    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device)

    logits = decoder.decode_proj_logits(pred_emb)
    return F.cross_entropy(logits[non_pad] / 0.1, tgt_valid[non_pad], ignore_index=0)


def _text_aux_ce_loss(pred_emb, target_ids, text_pad_mask, expander):
    """Auxiliary CE through decode_proj for text expander."""
    non_pad = ~text_pad_mask
    if not non_pad.any():
        return torch.tensor(0.0, device=pred_emb.device)

    logits = expander.decode_proj_logits(pred_emb)
    return F.cross_entropy(logits[non_pad] / 0.1, target_ids[non_pad], ignore_index=0)


def compute_task_loss(
    model: MultimodalWorldModel,
    batch: dict,
    task: int,
    device: torch.device,
    timestep: torch.Tensor,
    aux_ce_weight: float = 0.1,
    length_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute loss for a single task on a batch."""
    metrics = {}
    token_emb = model.shared_token_emb

    # Select compressor
    if task in (TASK_TRIPLE_TRIPLE, TASK_TRIPLE_TEXT):
        bottleneck = model.compress_triples(
            batch["triple_token_ids"].to(device),
            batch["triple_token_pad"].to(device),
            batch["triple_pad"].to(device),
        )
    else:  # TEXT input
        n_triples = model.config.max_triples
        bottleneck = model.compress_text(
            batch["text_token_ids"].to(device),
            batch["text_pad_mask"].to(device),
            n_triples,
        )

    # Select expander and compute loss
    if task in (TASK_TRIPLE_TRIPLE, TASK_TEXT_TRIPLE):
        # Triple output: attr CE + entity MSE + value MSE + lengths
        B = batch["target_pad"].shape[0]
        M = model.config.max_triples
        tgt_pad = batch["target_pad"].to(device)
        tgt_attr = batch["target_attr"].to(device)
        ent_ids = batch["entity_token_ids"].to(device)
        val_ids = batch["value_token_ids"].to(device)

        # Attr
        attr_logits = model.forward_attr(bottleneck)
        V = attr_logits.shape[-1]
        valid = ~tgt_pad.reshape(-1)
        attr_loss = torch.tensor(0.0, device=device)
        if valid.any():
            attr_loss = F.cross_entropy(
                attr_logits.reshape(-1, V)[valid],
                tgt_attr.reshape(-1)[valid],
                ignore_index=0,
            )
            metrics["acc_attr"] = (
                attr_logits.reshape(-1, V)[valid].argmax(-1)
                == tgt_attr.reshape(-1)[valid]
            ).float().mean().item()

        # Entity
        ent_pred, _ = model.forward_triple_expander(
            "entity", bottleneck, ent_ids, tgt_pad, timestep=timestep,
        )
        ent_loss, ent_m = _mse_loss(ent_pred, ent_ids, tgt_pad, B, M, token_emb)
        if "acc" in ent_m:
            metrics["acc_ent_tok"] = ent_m["acc"]

        ent_aux = torch.tensor(0.0, device=device)
        if model.triple_expander.use_decode_proj and ent_pred.shape[0] > 0:
            ent_aux = _aux_ce_loss(ent_pred, ent_ids, tgt_pad, B, M, model.triple_expander)

        # Value
        val_pred, _ = model.forward_triple_expander(
            "value", bottleneck, val_ids, tgt_pad, timestep=timestep,
        )
        val_loss, val_m = _mse_loss(val_pred, val_ids, tgt_pad, B, M, token_emb)
        if "acc" in val_m:
            metrics["acc_val_tok"] = val_m["acc"]

        val_aux = torch.tensor(0.0, device=device)
        if model.triple_expander.use_decode_proj and val_pred.shape[0] > 0:
            val_aux = _aux_ce_loss(val_pred, val_ids, tgt_pad, B, M, model.triple_expander)

        # Length prediction
        ent_len_pred, val_len_pred = model.forward_triple_lengths(bottleneck)
        ent_real = (ent_ids != 0).sum(dim=-1).float()
        val_real = (val_ids != 0).sum(dim=-1).float()
        valid_mask = ~tgt_pad
        length_loss = torch.tensor(0.0, device=device)
        if valid_mask.any():
            length_loss = (
                F.mse_loss(ent_len_pred[valid_mask], ent_real[valid_mask])
                + F.mse_loss(val_len_pred[valid_mask], val_real[valid_mask])
            )

        total = (
            attr_loss + ent_loss + val_loss
            + aux_ce_weight * (ent_aux + val_aux)
            + length_weight * length_loss
        )

    else:
        # Text output: MSE + aux CE + length
        text_ids = batch["text_token_ids"].to(device)
        text_pad = batch["text_pad_mask"].to(device)
        triple_pad = batch["triple_pad"].to(device)

        text_pred, _ = model.forward_text_expander(
            bottleneck, text_ids, text_pad,
            triple_pad_mask=triple_pad, timestep=timestep,
        )
        text_loss, text_m = _text_mse_loss(text_pred, text_ids, text_pad, token_emb)
        if "acc" in text_m:
            metrics["acc_text_tok"] = text_m["acc"]
        if "cos_sim" in text_m:
            metrics["cos_text"] = text_m["cos_sim"]

        text_aux = torch.tensor(0.0, device=device)
        if model.text_expander.use_decode_proj:
            text_aux = _text_aux_ce_loss(text_pred, text_ids, text_pad, model.text_expander)

        # Text length prediction
        text_len_pred = model.forward_text_length(bottleneck, triple_pad)
        text_real = batch["text_length"].float().to(device)
        text_len_loss = F.mse_loss(text_len_pred, text_real)
        metrics["text_len_loss"] = text_len_loss.item()

        total = text_loss + aux_ce_weight * text_aux + length_weight * text_len_loss

    metrics["loss_total"] = total.item()
    return total, metrics


# -- Generation-based assessment ---------------------------------------

@torch.no_grad()
def run_assessment(
    model: MultimodalWorldModel,
    ds: WebNLGMultimodalDataset,
    device: torch.device,
    tokenizer: DomainBPETokenizer,
    n_examples: int = 64,
    n_steps: int = 10,
) -> dict[str, float]:
    """Assess all four tasks with generation metrics."""
    model.eval()
    n = min(n_examples, len(ds))
    metrics = {}

    # Get data
    tri_ids = ds._triple_token_ids[:n].to(device)
    tri_pad = ds._triple_token_pad[:n].to(device)
    t_pad = ds._triple_pad[:n].to(device)
    txt_ids = ds._text_token_ids[:n].to(device)
    txt_pad = ds._text_pad_mask[:n].to(device)
    tgt_attr = ds._target_attr[:n].to(device)
    tgt_pad = ds._target_pad[:n].to(device)
    ent_ids = ds._entity_token_ids[:n]
    val_ids = ds._value_token_ids[:n]
    M = model.config.max_triples

    # Compress both ways
    tri_bottleneck = model.compress_triples(tri_ids, tri_pad, t_pad)
    txt_bottleneck = model.compress_text(txt_ids, txt_pad, model.config.max_triples)

    # Bottleneck alignment
    metrics["alignment"] = model.bottleneck_alignment(tri_bottleneck, txt_bottleneck, t_pad)

    # Task 1: triple -> triple (generate from triple bottleneck)
    gen_ent, gen_val, gen_attr = model.generate_triple_ids(tri_bottleneck, n_steps=n_steps)
    ent_exact = val_exact = attr_match = total = 0
    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total += 1
            if gen_attr[i, m].item() == tgt_attr[i, m].item():
                attr_match += 1
            tgt_e = [x for x in ent_ids[i, m].tolist() if x != 0]
            pred_e = gen_ent[i, m].cpu().tolist()[:len(tgt_e)]
            if pred_e == tgt_e:
                ent_exact += 1
            tgt_v = [x for x in val_ids[i, m].tolist() if x != 0]
            pred_v = gen_val[i, m].cpu().tolist()[:len(tgt_v)]
            if pred_v == tgt_v:
                val_exact += 1
    if total > 0:
        metrics["t1_ent_exact"] = ent_exact / total
        metrics["t1_val_exact"] = val_exact / total
        metrics["t1_attr"] = attr_match / total

    # Task 2: triple -> text (generate text from triple bottleneck)
    gen_text_ids = model.generate_text_ids(tri_bottleneck, triple_pad_mask=t_pad, n_steps=n_steps)
    text_tok_match = text_total = text_exact = 0
    for i in range(n):
        tgt = [x for x in txt_ids[i].tolist() if x != tokenizer.pad_token_id]
        pred = gen_text_ids[i].cpu().tolist()[:len(tgt)]
        matches = sum(1 for a, b in zip(pred, tgt) if a == b)
        text_tok_match += matches
        text_total += len(tgt)
        if pred == tgt:
            text_exact += 1
    if text_total > 0:
        metrics["t2_text_tok"] = text_tok_match / text_total
        metrics["t2_text_exact"] = text_exact / n

    # Task 3: text -> triple (generate triples from text bottleneck)
    gen_ent3, gen_val3, gen_attr3 = model.generate_triple_ids(txt_bottleneck, n_steps=n_steps)
    ent_exact3 = val_exact3 = attr_match3 = total3 = 0
    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total3 += 1
            if gen_attr3[i, m].item() == tgt_attr[i, m].item():
                attr_match3 += 1
            tgt_e = [x for x in ent_ids[i, m].tolist() if x != 0]
            pred_e = gen_ent3[i, m].cpu().tolist()[:len(tgt_e)]
            if pred_e == tgt_e:
                ent_exact3 += 1
            tgt_v = [x for x in val_ids[i, m].tolist() if x != 0]
            pred_v = gen_val3[i, m].cpu().tolist()[:len(tgt_v)]
            if pred_v == tgt_v:
                val_exact3 += 1
    if total3 > 0:
        metrics["t3_ent_exact"] = ent_exact3 / total3
        metrics["t3_val_exact"] = val_exact3 / total3
        metrics["t3_attr"] = attr_match3 / total3

    return metrics


@torch.no_grad()
def print_samples(
    model: MultimodalWorldModel,
    ds: WebNLGMultimodalDataset,
    device: torch.device,
    tokenizer: DomainBPETokenizer,
    n_examples: int = 3,
    n_steps: int = 10,
):
    """Print sample predictions for each task."""
    model.eval()
    n = min(n_examples, len(ds))

    tri_ids = ds._triple_token_ids[:n].to(device)
    tri_pad = ds._triple_token_pad[:n].to(device)
    t_pad = ds._triple_pad[:n].to(device)
    txt_ids = ds._text_token_ids[:n].to(device)
    txt_pad = ds._text_pad_mask[:n].to(device)
    M = model.config.max_triples

    tri_bn = model.compress_triples(tri_ids, tri_pad, t_pad)
    txt_bn = model.compress_text(txt_ids, txt_pad, model.config.max_triples)

    gen_ent, gen_val, gen_attr = model.generate_triple_ids(tri_bn, n_steps=n_steps)
    gen_text = model.generate_text_ids(tri_bn, triple_pad_mask=t_pad, n_steps=n_steps)
    gen_ent3, gen_val3, gen_attr3 = model.generate_triple_ids(txt_bn, n_steps=n_steps)

    print(f"\n{'='*70}")
    print(f"SAMPLE PREDICTIONS")
    print(f"{'='*70}")

    for i in range(n):
        ex = ds.examples[i]
        print(f"\n--- Example {i} ---")
        print(f"  INPUT triples: {ex['triples']}")
        print(f"  INPUT text: {ex['text']}")

        # T1: triple -> triple
        print(f"\n  T1 (triple->triple):")
        for m in range(M):
            if ds._target_pad[i, m]:
                continue
            tgt_e = tokenizer.decode(ds._entity_token_ids[i, m], skip_special_tokens=True)
            pred_e = tokenizer.decode(gen_ent[i, m].cpu(), skip_special_tokens=True)
            tgt_a = ds.vocab.decode_id(ds._target_attr[i, m].item(), "attr")
            pred_a = ds.vocab.decode_id(gen_attr[i, m].item(), "attr")
            tgt_v = tokenizer.decode(ds._value_token_ids[i, m], skip_special_tokens=True)
            pred_v = tokenizer.decode(gen_val[i, m].cpu(), skip_special_tokens=True)
            match = "Y" if (tgt_e.strip() == pred_e.strip() and
                          tgt_a == pred_a and tgt_v.strip() == pred_v.strip()) else "N"
            print(f"    [{m}] tgt:  ({tgt_e.strip()}, {tgt_a}, {tgt_v.strip()})")
            print(f"         pred: ({pred_e.strip()}, {pred_a}, {pred_v.strip()}) [{match}]")

        # T2: triple -> text
        pred_text = tokenizer.decode(gen_text[i].cpu(), skip_special_tokens=True)
        print(f"\n  T2 (triple->text):")
        print(f"    tgt:  {ex['text']}")
        print(f"    pred: {pred_text.strip()}")

        # T3: text -> triple
        print(f"\n  T3 (text->triple):")
        for m in range(M):
            if ds._target_pad[i, m]:
                continue
            tgt_e = tokenizer.decode(ds._entity_token_ids[i, m], skip_special_tokens=True)
            pred_e = tokenizer.decode(gen_ent3[i, m].cpu(), skip_special_tokens=True)
            tgt_a = ds.vocab.decode_id(ds._target_attr[i, m].item(), "attr")
            pred_a = ds.vocab.decode_id(gen_attr3[i, m].item(), "attr")
            tgt_v = tokenizer.decode(ds._value_token_ids[i, m], skip_special_tokens=True)
            pred_v = tokenizer.decode(gen_val3[i, m].cpu(), skip_special_tokens=True)
            match = "Y" if (tgt_e.strip() == pred_e.strip() and
                          tgt_a == pred_a and tgt_v.strip() == pred_v.strip()) else "N"
            print(f"    [{m}] tgt:  ({tgt_e.strip()}, {tgt_a}, {tgt_v.strip()})")
            print(f"         pred: ({pred_e.strip()}, {pred_a}, {pred_v.strip()}) [{match}]")

    print(f"{'='*70}", flush=True)


# -- Training phase ----------------------------------------------------

def run_phase(
    model: MultimodalWorldModel,
    train_ds: WebNLGMultimodalDataset,
    eval_ds: WebNLGMultimodalDataset,
    device: torch.device,
    tokenizer: DomainBPETokenizer,
    out_dir: Path,
    phase_name: str,
    t_min: float, t_max: float, bias_power: float,
    epochs: int, patience: int,
    batch_size: int, lr: float, weight_decay: float,
    denoise_steps: int,
    log_every: int, diagnostic_every: int,
    aux_ce_weight: float, length_weight: float,
    task_weights: dict[int, float],
    lr_scale_existing: float = 1.0,
):
    phase_dir = out_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}  t in [{t_min}, {t_max}]  bias={bias_power}")
    print(f"{'='*60}")

    # Separate param groups for existing (fine-tune) vs new (from scratch)
    existing_params = []
    new_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "triple_compressor" in name or "triple_expander" in name:
            existing_params.append(p)
        else:
            new_params.append(p)

    param_groups = [
        {"params": existing_params, "lr": lr * lr_scale_existing},
        {"params": new_params, "lr": lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    total_trainable = sum(p.numel() for group in param_groups for p in group["params"])
    print(f"  Trainable: {total_trainable:,} "
          f"(existing: {sum(p.numel() for p in existing_params):,} @ {lr*lr_scale_existing:.1e}, "
          f"new: {sum(p.numel() for p in new_params):,} @ {lr:.1e})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    n_train = len(train_ds)
    tasks = list(task_weights.keys())
    weights = [task_weights[t] for t in tasks]

    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # Initial assessment
    model.eval()
    init_m = run_assessment(model, eval_ds, device, tokenizer, n_examples=64, n_steps=denoise_steps)
    log_parts = ["  init:"]
    for k, v in sorted(init_m.items()):
        log_parts.append(f"{k}={v:.3f}")
    print(" ".join(log_parts), flush=True)

    print(f"\nTraining for up to {epochs} epochs "
          f"(patience={'off' if patience <= 0 else patience})...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = {t: [] for t in tasks}
        epoch_loss_total = 0.0
        n_batches = 0

        indices = list(range(n_train))
        random.shuffle(indices)

        for start in range(0, n_train - batch_size + 1, batch_size):
            batch_idx = indices[start:start + batch_size]
            B = len(batch_idx)

            # Sample task for this batch
            task = random.choices(tasks, weights=weights, k=1)[0]

            # Collate batch
            batch = {}
            for key in train_ds[0].keys():
                batch[key] = torch.stack([train_ds[i][key] for i in batch_idx])

            timestep = sample_timestep(B, device, t_min, t_max, bias_power)

            loss, metrics = compute_task_loss(
                model, batch, task, device, timestep,
                aux_ce_weight=aux_ce_weight,
                length_weight=length_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            optimizer.step()

            epoch_losses[task].append(loss.item())
            epoch_loss_total += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss_total / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1:
            model.eval()

            # Per-task training losses
            task_loss_str = " | ".join(
                f"{TASK_NAMES[t]}: {sum(epoch_losses[t])/max(len(epoch_losses[t]),1):.4f}"
                for t in tasks if epoch_losses[t]
            )

            # Generation assessment
            gen_m = run_assessment(model, eval_ds, device, tokenizer,
                            n_examples=64, n_steps=denoise_steps)

            # Primary metric: average of t1_val_exact and t3_val_exact
            primary = (gen_m.get("t1_val_exact", 0) + gen_m.get("t3_val_exact", 0)) / 2

            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f} | {task_loss_str}"
                f" || t1_ve={gen_m.get('t1_val_exact', 0):.3f}"
                f" t1_ee={gen_m.get('t1_ent_exact', 0):.3f}"
                f" t1_a={gen_m.get('t1_attr', 0):.3f}"
                f" | t2_tt={gen_m.get('t2_text_tok', 0):.3f}"
                f" | t3_ve={gen_m.get('t3_val_exact', 0):.3f}"
                f" t3_ee={gen_m.get('t3_ent_exact', 0):.3f}"
                f" t3_a={gen_m.get('t3_attr', 0):.3f}"
                f" | align={gen_m.get('alignment', 0):.3f}"
            )

            if primary > best_metric:
                best_metric = primary
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), phase_dir / "model_best.pt")
                log += " *"
            else:
                epochs_without_improvement += log_every

            print(log, flush=True)

            entry = {
                "epoch": epoch,
                "phase": phase_name,
                "train_loss": avg_loss,
                "gen_metrics": gen_m,
                "primary_metric": primary,
            }

            if epoch == 1 or epoch % diagnostic_every == 0:
                print_samples(model, eval_ds, device, tokenizer,
                            n_examples=3, n_steps=denoise_steps)

            history.append(entry)

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best primary {best_metric:.4f} at epoch {best_epoch})")
                break

    torch.save(model.state_dict(), phase_dir / "model_final.pt")
    with open(phase_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{phase_name} done. Best primary: {best_metric:.4f} at epoch {best_epoch}")
    return best_metric, best_epoch


# -- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train v16: Multimodal triple <-> text")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, required=True)
    ap.add_argument("--config", type=str, default="base", choices=list(PROFILES.keys()))
    # Architecture
    ap.add_argument("--compressor-layers", type=int, default=2)
    ap.add_argument("--text-compressor-layers", type=int, default=4)
    ap.add_argument("--denoiser-layers", type=int, default=1)
    ap.add_argument("--text-expander-layers", type=int, default=3)
    ap.add_argument("--max-slot-tokens", type=int, default=12)
    ap.add_argument("--max-text-tokens", type=int, default=64)
    ap.add_argument("--denoise-steps", type=int, default=10)
    # Training
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-scale-existing", type=float, default=0.1,
                    help="LR multiplier for triple compressor/expander (fine-tune)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aux-ce-weight", type=float, default=0.1)
    ap.add_argument("--length-weight", type=float, default=0.1)
    ap.add_argument("--alpha-min", type=float, default=0.01)
    # Curriculum
    ap.add_argument("--phase1-epochs", type=int, default=100)
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--phase2-patience", type=int, default=50)
    ap.add_argument("--phase2-bias-power", type=float, default=2.0)
    # Task weights
    ap.add_argument("--w-tt", type=float, default=1.0, help="triple->triple weight")
    ap.add_argument("--w-tx", type=float, default=1.0, help="triple->text weight")
    ap.add_argument("--w-xt", type=float, default=1.0, help="text->triple weight")
    ap.add_argument("--w-xx", type=float, default=0.5, help="text->text weight")
    # Logging
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--diagnostic-every", type=int, default=50)
    ap.add_argument("--device", type=str, default=None)
    # Optional: load pre-trained triple compressor/expander
    ap.add_argument("--max-examples", type=int, default=0,
                    help="Limit training examples (0 = use all)")
    ap.add_argument("--pretrained-checkpoint", type=str, default=None,
                    help="Path to v15 checkpoint for triple compressor/expander init")
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

    task_weights = {
        TASK_TRIPLE_TRIPLE: args.w_tt,
        TASK_TRIPLE_TEXT: args.w_tx,
        TASK_TEXT_TRIPLE: args.w_xt,
        TASK_TEXT_TEXT: args.w_xx,
    }

    print(f"=== v16 Multimodal Triple <-> Text ===")
    print(f"Device: {device}")
    print(f"Config: {args.config} (d={config.d_model}, h={config.n_heads}, L={config.n_layers})")
    print(f"BPE vocab: {tokenizer.vocab_size}")
    print(f"Task weights: {', '.join(f'{TASK_NAMES[t]}={w}' for t, w in task_weights.items())}")
    print(f"Phase 1: t in [0.7,1.0], {args.phase1_epochs} epochs")
    print(f"Phase 2: t in [0.0,1.0], {args.phase2_epochs} epochs, patience={args.phase2_patience}")

    # Build phrase vocab
    print("Building phrase vocabulary...")
    train_path = data_dir / "train.jsonl"
    train_examples = []
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            # Convert to state_t/state_t+1 format for PhraseVocab
            train_examples.append({
                "state_t": ex["triples"],
                "state_t+1": ex["triples"],
            })

    vocab = PhraseVocab()
    vocab.build(train_examples)
    for role, size in vocab.vocab_sizes.items():
        print(f"  {role}: {size} phrases")
    vocab.save(out_dir / "phrase_vocab.json")

    # Build model
    print("Building multimodal model...")
    model = MultimodalWorldModel(
        config=config,
        vocab=vocab,
        domain_tokenizer=tokenizer,
        compressor_layers=args.compressor_layers,
        text_compressor_layers=args.text_compressor_layers,
        max_slot_tokens=args.max_slot_tokens,
        max_text_tokens=args.max_text_tokens,
        denoiser_layers=args.denoiser_layers,
        text_expander_layers=args.text_expander_layers,
        dropout=args.dropout,
        alpha_min=args.alpha_min,
    ).to(device)

    # Init embeddings
    model.init_embeddings()

    # Optionally load pre-trained triple compressor/expander
    if args.pretrained_checkpoint:
        print(f"Loading pre-trained checkpoint: {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=True)
        if "compressor" in ckpt:
            model.triple_compressor.load_state_dict(ckpt["compressor"], strict=False)
            print("  Loaded triple compressor weights")
        if "model" in ckpt:
            # Extract triple_decoder weights
            decoder_sd = {}
            for k, v in ckpt["model"].items():
                if k.startswith("triple_decoder."):
                    decoder_sd[k.removeprefix("triple_decoder.")] = v
            if decoder_sd:
                model.triple_expander.load_state_dict(decoder_sd, strict=False)
                print(f"  Loaded {len(decoder_sd)} triple expander weights")

    # Print param counts
    counts = model.component_param_counts()
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
    print(f"  Total: {model.param_count():,} ({model.trainable_param_count():,} trainable)")

    # Build datasets
    print("Building datasets...")
    train_ds = WebNLGMultimodalDataset(
        train_path, vocab, tokenizer,
        max_triples=config.max_triples,
        max_slot_tokens=args.max_slot_tokens,
        max_text_tokens=args.max_text_tokens,
        task_weights=task_weights,
        max_examples=args.max_examples,
    )
    print(f"  Train: {len(train_ds)} examples")

    test_path = data_dir / "test.jsonl"
    eval_ds = train_ds
    if test_path.exists():
        test_ds = WebNLGMultimodalDataset(
            test_path, vocab, tokenizer,
            max_triples=config.max_triples,
            max_slot_tokens=args.max_slot_tokens,
            max_text_tokens=args.max_text_tokens,
        )
        eval_ds = test_ds
        print(f"  Test: {len(test_ds)} examples")

    # Save config
    config.save(out_dir / "config.json")
    with open(out_dir / "model_config.json", "w") as f:
        json.dump({
            "architecture": "v16_multimodal",
            "config_profile": args.config,
            "compressor_layers": args.compressor_layers,
            "text_compressor_layers": args.text_compressor_layers,
            "denoiser_layers": args.denoiser_layers,
            "text_expander_layers": args.text_expander_layers,
            "max_slot_tokens": args.max_slot_tokens,
            "max_text_tokens": args.max_text_tokens,
            "denoise_steps": args.denoise_steps,
            "dropout": args.dropout,
            "alpha_min": args.alpha_min,
            "bpe_vocab_size": tokenizer.vocab_size,
            "task_weights": {TASK_NAMES[t]: w for t, w in task_weights.items()},
            "lr": args.lr,
            "lr_scale_existing": args.lr_scale_existing,
            "phase1_epochs": args.phase1_epochs,
            "phase2_epochs": args.phase2_epochs,
        }, f, indent=2)

    # Phase 1: high noise, no early stopping
    p1, p1_ep = run_phase(
        model, train_ds, eval_ds, device, tokenizer,
        out_dir=out_dir, phase_name="phase1",
        t_min=0.7, t_max=1.0, bias_power=1.0,
        epochs=args.phase1_epochs, patience=0,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every, diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight, length_weight=args.length_weight,
        task_weights=task_weights,
        lr_scale_existing=args.lr_scale_existing,
    )

    # Phase 2: full range, importance sampling, early stop
    p2, p2_ep = run_phase(
        model, train_ds, eval_ds, device, tokenizer,
        out_dir=out_dir, phase_name="phase2",
        t_min=0.0, t_max=1.0, bias_power=args.phase2_bias_power,
        epochs=args.phase2_epochs, patience=args.phase2_patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        denoise_steps=args.denoise_steps,
        log_every=args.log_every, diagnostic_every=args.diagnostic_every,
        aux_ce_weight=args.aux_ce_weight, length_weight=args.length_weight,
        task_weights=task_weights,
        lr_scale_existing=args.lr_scale_existing,
    )

    print(f"\n{'='*60}")
    print(f"DONE -- v16 Multimodal")
    print(f"  Phase 1: best={p1:.4f} at epoch {p1_ep}")
    print(f"  Phase 2: best={p2:.4f} at epoch {p2_ep}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
