#!/usr/bin/env python3
"""Run v6's generation scoring on the v2 600-epoch checkpoint.

Resolves the 65% vs 14% discrepancy: is the diagnostic inflated,
or does the v2 checkpoint genuinely perform better?
"""

import json
from pathlib import Path
from collections import Counter

import torch

from twm.config import ModelConfig
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.t5_dataset import T5TripleDataset


@torch.no_grad()
def run_gen_scoring(model, ds, device, t5_tokenizer, n_examples=64, n_steps=10):
    """Exact same logic as v6 eval_generation."""
    model.eval()  # noqa: B009
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

            # Entity
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

            # Value
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

    unique_values = len(set(all_pred_values))
    unique_entities = len(set(all_pred_entities))
    val_counter = Counter(all_pred_values)
    top_value_count = val_counter.most_common(1)[0][1] if val_counter else 0

    print(f"Total triples: {total}")
    print(f"gen_val_tok:   {val_tok_correct}/{val_tok_total} = {val_tok_correct/max(val_tok_total,1)*100:.1f}%")
    print(f"gen_val_exact: {val_exact_match}/{total} = {val_exact_match/total*100:.1f}%")
    print(f"gen_ent_tok:   {ent_tok_correct}/{ent_tok_total} = {ent_tok_correct/max(ent_tok_total,1)*100:.1f}%")
    print(f"gen_ent_exact: {ent_exact_match}/{total} = {ent_exact_match/total*100:.1f}%")
    print(f"gen_attr:      {attr_match}/{total} = {attr_match/total*100:.1f}%")
    print(f"unique_values: {unique_values}")
    print(f"unique_entities: {unique_entities}")
    print(f"top_value_count: {top_value_count}")

    # Print a few examples
    print("\n--- Sample predictions (first 5 non-pad triples) ---")
    shown = 0
    for i in range(n):
        for m_idx in range(M):
            if tgt_pad[i, m_idx]:
                continue
            tgt_v_ids = ds._all_value_token_ids[i, m_idx]
            tgt_v = t5_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v = gen_values[i][m_idx]
            tgt_e_ids = ds._all_entity_token_ids[i, m_idx]
            tgt_e = t5_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e = gen_entities[i][m_idx]
            print(f"  [{i},{m_idx}] ent: {tgt_e!r} -> {pred_e!r}")
            print(f"         val: {tgt_v!r} -> {pred_v!r}")
            shown += 1
            if shown >= 5:
                break
        if shown >= 5:
            break


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="results/v2_1L128d_10k_pre")
    ap.add_argument("--data-dir", type=str, default="data/atomic_10000")
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

    def encode_fn(phrases):
        return st.encode(phrases, batch_size=256,
                         show_progress_bar=False,
                         convert_to_tensor=True, device=str(device))

    vocab = PhraseVocab.load(model_dir / "phrase_vocab.json")

    from transformers import T5Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    MASK_TOKEN_ID = 32099
    st_dim = st.get_sentence_embedding_dimension()

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=mcfg["max_value_tokens"],
        n_proj_tokens=mcfg["n_proj_tokens"],
        denoiser_layers=mcfg["denoiser_layers"],
        denoiser_dim=mcfg["denoiser_dim"],
        denoiser_heads=mcfg["denoiser_heads"],
        mask_token_id=MASK_TOKEN_ID,
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

    ckpt_path = model_dir / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model_final.pt"
    print(f"Loading {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # v2 used "denoiser.layers" but current code uses "layers"
    fixed_sd = {}
    for k, v in sd.items():
        fixed_sd[k.replace(".denoiser.layers.", ".layers.")] = v
    model.load_state_dict(fixed_sd)

    ds = T5TripleDataset(
        data_dir / "test.jsonl", encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=mcfg["max_value_tokens"],
    )

    print(f"\nRunning v6 gen scoring on v2 checkpoint ({len(ds)} test examples, using 64)")
    print("=" * 60)
    run_gen_scoring(model, ds, device, t5_tokenizer, n_examples=64, n_steps=10)


if __name__ == "__main__":
    main()
