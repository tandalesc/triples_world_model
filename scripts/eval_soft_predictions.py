#!/usr/bin/env python3
"""Evaluate soft predictions on a trained diffusion TWM checkpoint.

Sweeps temperature and step count, comparing hard argmax vs soft embeddings.

Usage:
    uv run python scripts/eval_soft_predictions.py \
        --model-dir results/v10_unified \
        --data-dir data/atomic_10000 \
        --device cuda:1
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch

from twm.config import ModelConfig
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.t5_dataset import T5TripleDataset


def run_generation(model, ds, device, t5_tokenizer, n_examples, n_steps,
                   temperature, soft):
    """Run generation and return metrics."""
    n = min(n_examples, len(ds))

    with torch.no_grad():
        latent = model.encode_dynamics(
            ds._all_inputs[:n].to(device),
            ds._all_input_pad_masks[:n].to(device),
        )

        gen_entities = model.generate_entities(
            latent, n_steps=n_steps, temperature=temperature, soft=soft,
        )
        gen_values = model.generate_values(
            latent, n_steps=n_steps, temperature=temperature, soft=soft,
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
            tgt_e_str = t5_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e_str = gen_entities[i][m]
            if pred_e_str == tgt_e_str:
                ent_exact += 1
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
                val_exact += 1
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

    return {
        "gen_ent_tok": ent_tok_correct / max(ent_tok_total, 1),
        "gen_ent_exact": ent_exact / total,
        "gen_val_tok": val_tok_correct / max(val_tok_total, 1),
        "gen_val_exact": val_exact / total,
        "gen_attr": attr_match / total,
        "unique_values": len(set(all_pred_values)),
        "unique_entities": len(set(all_pred_entities)),
        "top_value_count": Counter(all_pred_values).most_common(1)[0][1] if all_pred_values else 0,
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser(description="Soft predictions eval sweep")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="model_best.pt")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n-examples", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = ModelConfig.load(model_dir / "config.json")
    with open(model_dir / "model_config.json") as f:
        mcfg = json.load(f)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(mcfg["st_model"], device=str(device))
    st_dim = st.get_sentence_embedding_dimension()

    def encode_fn(phrases):
        return st.encode(phrases, batch_size=256, show_progress_bar=False,
                         convert_to_tensor=True, device=str(device))

    vocab = PhraseVocab.load(model_dir / "phrase_vocab.json")

    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=mcfg["max_value_tokens"],
        n_proj_tokens=mcfg["n_proj_tokens"],
        denoiser_layers=mcfg["denoiser_layers"],
        denoiser_dim=mcfg["denoiser_dim"],
        denoiser_heads=mcfg["denoiser_heads"],
        mask_token_id=32099,
        tokenizer=t5_tokenizer,
        use_film=mcfg.get("use_film", False),
        use_cross_attention=mcfg.get("use_cross_attention", True),
        use_adaln=mcfg.get("use_adaln", False),
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

    ckpt_path = model_dir / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model_final.pt"
    print(f"Loading {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.train(False)

    data_file = data_dir / f"{args.split}.jsonl"
    ds = T5TripleDataset(
        data_file, encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=mcfg["max_value_tokens"],
    )
    print(f"Dataset: {len(ds)} examples, evaluating {min(args.n_examples, len(ds))}")

    # Sweep configurations
    configs = [
        # Baseline: hard argmax
        {"soft": False, "temperature": 0.0, "n_steps": 10, "label": "hard_t0.0_s10"},
        # Soft predictions with temperature sweep
        {"soft": True, "temperature": 0.1, "n_steps": 10, "label": "soft_t0.1_s10"},
        {"soft": True, "temperature": 0.5, "n_steps": 10, "label": "soft_t0.5_s10"},
        {"soft": True, "temperature": 1.0, "n_steps": 10, "label": "soft_t1.0_s10"},
        {"soft": True, "temperature": 2.0, "n_steps": 10, "label": "soft_t2.0_s10"},
    ]

    results = {}
    best_temp = None
    best_gvt = -1.0

    print(f"\n{'label':<20s}  {'gv_tok':>7s}  {'gv_ex':>6s}  {'ge_tok':>7s}  {'ge_ex':>6s}  {'g_attr':>6s}  {'u_v':>4s}  {'u_e':>4s}  {'top1':>4s}")
    print("-" * 85)

    for cfg in configs:
        m = run_generation(
            model, ds, device, t5_tokenizer,
            n_examples=args.n_examples,
            n_steps=cfg["n_steps"],
            temperature=cfg["temperature"],
            soft=cfg["soft"],
        )
        results[cfg["label"]] = m

        gvt = m.get("gen_val_tok", 0)
        if cfg["soft"] and gvt > best_gvt:
            best_gvt = gvt
            best_temp = cfg["temperature"]

        print(f"{cfg['label']:<20s}  {m.get('gen_val_tok',0):7.3f}  {m.get('gen_val_exact',0):6.3f}  "
              f"{m.get('gen_ent_tok',0):7.3f}  {m.get('gen_ent_exact',0):6.3f}  "
              f"{m.get('gen_attr',0):6.3f}  {m.get('unique_values',0):4d}  "
              f"{m.get('unique_entities',0):4d}  {m.get('top_value_count',0):4d}")

    # If we found a best soft temperature, sweep step counts
    if best_temp is not None:
        print(f"\nBest soft temperature: {best_temp}")
        print(f"\nStep count sweep at temp={best_temp}:")
        print(f"{'label':<20s}  {'gv_tok':>7s}  {'gv_ex':>6s}  {'ge_tok':>7s}  {'ge_ex':>6s}")
        print("-" * 55)

        for n_steps in [5, 20, 50]:
            label = f"soft_t{best_temp}_s{n_steps}"
            m = run_generation(
                model, ds, device, t5_tokenizer,
                n_examples=args.n_examples,
                n_steps=n_steps,
                temperature=best_temp,
                soft=True,
            )
            results[label] = m
            print(f"{label:<20s}  {m.get('gen_val_tok',0):7.3f}  {m.get('gen_val_exact',0):6.3f}  "
                  f"{m.get('gen_ent_tok',0):7.3f}  {m.get('gen_ent_exact',0):6.3f}")

    # Save results
    out_path = model_dir / "soft_predictions_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
