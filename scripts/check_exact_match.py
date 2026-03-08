#!/usr/bin/env python3
"""Quick diagnostic: print predictions vs targets to debug exact match metric.

Loads a checkpoint and prints 10 side-by-side comparisons with repr(),
token IDs, and character diffs.

Usage:
    uv run python scripts/check_exact_match.py \
        --model-dir results/v13_identity_webnlg/phase1 \
        --data-dir data/webnlg \
        --device cuda:1
"""

import argparse
import json
from pathlib import Path

import torch

from twm.config import ModelConfig
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.domain_bpe import DomainBPETokenizer
from twm.domain_dataset import DomainTripleDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="model_best.pt")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--st-model", type=str, default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Find config -- check parent dir too (phase subdirs)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        config_path = model_dir.parent / "config.json"
    config = ModelConfig.load(config_path)

    mcfg_path = model_dir / "model_config.json"
    if not mcfg_path.exists():
        mcfg_path = model_dir.parent / "model_config.json"
    with open(mcfg_path) as f:
        mcfg = json.load(f)

    vocab_path = model_dir / "phrase_vocab.json"
    if not vocab_path.exists():
        vocab_path = model_dir.parent / "phrase_vocab.json"
    vocab = PhraseVocab.load(vocab_path)

    domain_tokenizer = DomainBPETokenizer.load(
        mcfg["domain_tokenizer"], max_length=mcfg["max_value_tokens"],
    )

    st_model_name = args.st_model or mcfg.get("st_model", "all-MiniLM-L6-v2")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name, device=str(device))
    st_dim = st.get_sentence_embedding_dimension()

    def encode_fn(phrases):
        return st.encode(phrases, batch_size=256,
                         show_progress_bar=False,
                         convert_to_tensor=True, device=str(device))

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=mcfg["max_value_tokens"],
        n_proj_tokens=mcfg.get("n_proj_tokens", 3),
        denoiser_layers=mcfg["denoiser_layers"],
        denoiser_dim=mcfg["denoiser_dim"],
        denoiser_heads=mcfg["denoiser_heads"],
        token_vocab_size=mcfg.get("domain_vocab_size", domain_tokenizer.vocab_size),
        mask_token_id=domain_tokenizer.mask_token_id,
        tokenizer=domain_tokenizer,
        use_film=mcfg.get("use_film", False),
        use_cross_attention=mcfg.get("use_cross_attention", True),
        use_adaln=mcfg.get("use_adaln", False),
        use_continuous_noise=mcfg.get("use_continuous_noise", False),
        normalize_noise=mcfg.get("normalize_noise", True),
        alpha_min=mcfg.get("alpha_min", 0.0),
        timestep_bias_power=mcfg.get("timestep_bias_power", 1.0),
        unified_decoder=mcfg.get("unified_decoder", False),
        wspace=mcfg.get("wspace", False),
        use_mse_prediction=mcfg.get("use_mse_prediction", True),
        cond_drop_prob=mcfg.get("cond_drop_prob", 0.0),
        use_decode_proj=mcfg.get("use_decode_proj", False),
    ).to(device)

    ckpt_path = model_dir / args.checkpoint
    print(f"Loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    data_file = data_dir / f"{args.split}.jsonl"
    ds = DomainTripleDataset(
        data_file, encode_fn, vocab, domain_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=mcfg["max_value_tokens"],
    )

    n = min(args.n_examples, len(ds))
    print(f"\n{args.split}: {len(ds)} examples, checking {n}")
    print(f"Denoise steps: {args.denoise_steps}, guidance: {args.guidance_scale}")

    with torch.no_grad():
        latent = model.encode_dynamics(
            ds._all_inputs[:n].to(device),
            ds._all_input_pad_masks[:n].to(device),
        )
        gen_entities = model.generate_entities(
            latent, n_steps=args.denoise_steps, guidance_scale=args.guidance_scale,
        )
        gen_values = model.generate_values(
            latent, n_steps=args.denoise_steps, guidance_scale=args.guidance_scale,
        )
        discrete_logits = model.forward_discrete(latent)
        pred_attrs = discrete_logits["attr"].argmax(-1)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n].to(device)
    M = config.max_triples

    ent_exact = val_exact = total = 0
    ent_tok_correct = ent_tok_total = 0
    val_tok_correct = val_tok_total = 0
    printed = 0

    print(f"\n{'='*80}")
    for i in range(n):
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total += 1

            tgt_e_ids = ds._all_entity_token_ids[i, m]
            tgt_e = domain_tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e = gen_entities[i][m]
            pred_e_reenc = domain_tokenizer.encode(pred_e, max_length=tgt_e_ids.shape[0])

            tgt_v_ids = ds._all_value_token_ids[i, m]
            tgt_v = domain_tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v = gen_values[i][m]
            pred_v_reenc = domain_tokenizer.encode(pred_v, max_length=tgt_v_ids.shape[0])

            tgt_a_id = tgt_attrs[i, m].item()
            pred_a_id = pred_attrs[i, m].item()
            tgt_a = vocab.decode_id(tgt_a_id, "attr")
            pred_a = vocab.decode_id(pred_a_id, "attr")

            # Token-level comparison (re-encode prediction, compare at non-pad positions)
            tgt_e_list = tgt_e_ids.tolist()
            non_pad_e = [j for j, x in enumerate(tgt_e_list) if x != 0]
            e_all_match = True
            for j in non_pad_e:
                ent_tok_total += 1
                if j < len(pred_e_reenc) and pred_e_reenc[j] == tgt_e_list[j]:
                    ent_tok_correct += 1
                else:
                    e_all_match = False
            if e_all_match and non_pad_e:
                ent_exact += 1

            tgt_v_list = tgt_v_ids.tolist()
            non_pad_v = [j for j, x in enumerate(tgt_v_list) if x != 0]
            v_all_match = True
            for j in non_pad_v:
                val_tok_total += 1
                if j < len(pred_v_reenc) and pred_v_reenc[j] == tgt_v_list[j]:
                    val_tok_correct += 1
                else:
                    v_all_match = False
            if v_all_match and non_pad_v:
                val_exact += 1

            # String comparison for diagnostics
            e_str_exact = tgt_e == pred_e
            v_str_exact = tgt_v == pred_v

            if printed < 10:
                printed += 1
                tgt_e_nonpad = [x for x in tgt_e_list if x != 0]
                pred_e_nonpad = [x for x in pred_e_reenc if x != 0]
                tgt_v_nonpad = [x for x in tgt_v_list if x != 0]
                pred_v_nonpad = [x for x in pred_v_reenc if x != 0]

                print(f"\n[{i},{m}] ENTITY  tok_exact={e_all_match}  str_exact={e_str_exact}")
                print(f"  tgt:  {repr(tgt_e)}")
                print(f"  pred: {repr(pred_e)}")
                print(f"  tgt_ids:  {tgt_e_nonpad}")
                print(f"  pred_ids: {pred_e_nonpad}")
                if not e_str_exact and e_all_match:
                    print(f"  NOTE: tokens match but strings differ (extra tokens at pad positions)")

                print(f"  ATTR  match={tgt_a_id == pred_a_id}")
                print(f"  tgt:  {repr(tgt_a)}")
                print(f"  pred: {repr(pred_a)}")

                print(f"  VALUE  tok_exact={v_all_match}  str_exact={v_str_exact}")
                print(f"  tgt:  {repr(tgt_v)}")
                print(f"  pred: {repr(pred_v)}")
                print(f"  tgt_ids:  {tgt_v_nonpad}")
                print(f"  pred_ids: {pred_v_nonpad}")
                if not v_str_exact and v_all_match:
                    print(f"  NOTE: tokens match but strings differ (extra tokens at pad positions)")

                # Show diffs at non-pad target positions
                diffs = []
                for j in non_pad_v:
                    t = tgt_v_list[j]
                    p = pred_v_reenc[j] if j < len(pred_v_reenc) else -1
                    if t != p:
                        diffs.append(f"pos{j}: tgt={t} pred={p}")
                if diffs:
                    print(f"  tok_diffs: {diffs}")
                else:
                    print(f"  tok_diffs: NONE (all target-position tokens match!)")

    print(f"\n{'='*80}")
    print(f"SUMMARY ({total} triples)")
    print(f"  Entity: tok={ent_tok_correct}/{ent_tok_total} "
          f"({ent_tok_correct/max(ent_tok_total,1)*100:.1f}%), "
          f"exact={ent_exact}/{total} ({ent_exact/max(total,1)*100:.1f}%)")
    print(f"  Value:  tok={val_tok_correct}/{val_tok_total} "
          f"({val_tok_correct/max(val_tok_total,1)*100:.1f}%), "
          f"exact={val_exact}/{total} ({val_exact/max(total,1)*100:.1f}%)")

    if val_tok_correct / max(val_tok_total, 1) > 0.8 and val_exact / max(total, 1) < 0.1:
        print("\n  NOTE: high token accuracy but low exact match.")
        print("  If str_exact=False but tok_exact=True above, the model predicts")
        print("  correct content tokens but extra garbage at pad positions.")


if __name__ == "__main__":
    main()
