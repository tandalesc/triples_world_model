#!/usr/bin/env python3
"""Run inference with a trained diffusion TWM.

Supports configurable denoise steps, temperature, and cosine schedule.

Usage:
    uv run python scripts/infer_diffusion_model.py \
        --model-dir results/diff_2L256d_10k \
        --data-dir data/atomic_10000 \
        --denoise-steps 50 --temperature 0.5 --split test --n-examples 20
"""

import argparse
import json
from pathlib import Path

import torch

from twm.config import ModelConfig
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.t5_dataset import T5TripleDataset
from twm.token_dataset import TokenTripleDataset


def main():
    ap = argparse.ArgumentParser(description="Diffusion TWM inference")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint file (default: model_best.pt)")
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-cosine", action="store_true",
                    help="Use linear schedule instead of cosine")
    ap.add_argument("--guidance-scale", type=float, default=1.0,
                    help="Classifier-free guidance scale (1.0 = no guidance)")
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "test"])
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--st-model", type=str, default=None)
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

    # Load configs
    config = ModelConfig.load(model_dir / "config.json")
    with open(model_dir / "model_config.json") as f:
        mcfg = json.load(f)

    token_level = mcfg.get("token_level", False)
    max_tokens_per_slot = mcfg.get("max_tokens_per_slot", 12)

    encode_fn, st_dim = None, 0
    if not token_level:
        st_model_name = args.st_model or mcfg["st_model"]
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(st_model_name, device=str(device))
        st_dim = st.get_sentence_embedding_dimension()

        def encode_fn(phrases):
            return st.encode(phrases, batch_size=256,
                             show_progress_bar=False,
                             convert_to_tensor=True, device=str(device))

    # Vocab
    vocab = PhraseVocab.load(model_dir / "phrase_vocab.json")

    # Tokenizer: domain BPE or T5
    use_domain_vocab = "domain_tokenizer" in mcfg
    if use_domain_vocab:
        from twm.domain_bpe import DomainBPETokenizer
        tokenizer = DomainBPETokenizer.load(
            mcfg["domain_tokenizer"],
            max_length=mcfg["max_value_tokens"],
        )
        token_vocab_size = mcfg["domain_vocab_size"]
        mask_token_id = tokenizer.mask_token_id
    else:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        token_vocab_size = 32100
        mask_token_id = 32099  # T5's <extra_id_0>

    if token_level:
        st_dim = config.d_model  # placeholder — TokenEncoder doesn't use st_dim

    model = DiffusionWorldModel(
        config, st_dim, vocab,
        max_value_tokens=mcfg["max_value_tokens"],
        n_proj_tokens=mcfg["n_proj_tokens"],
        denoiser_layers=mcfg["denoiser_layers"],
        denoiser_dim=mcfg["denoiser_dim"],
        denoiser_heads=mcfg["denoiser_heads"],
        token_vocab_size=token_vocab_size,
        mask_token_id=mask_token_id,
        tokenizer=tokenizer,
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
        token_level=token_level,
        max_tokens_per_slot=max_tokens_per_slot,
    ).to(device)

    ckpt = args.checkpoint or "model_best.pt"
    ckpt_path = model_dir / ckpt
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model_final.pt"
    print(f"Loading {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Dataset
    data_file = data_dir / f"{args.split}.jsonl"
    if not data_file.exists():
        data_file = data_dir / "test.jsonl" if args.split == "test" else data_dir / "train.jsonl"
    if token_level:
        token_emb_weight = model._get_decoder("value").token_emb.weight.detach().cpu()
        ds = TokenTripleDataset(
            data_file, token_emb_weight, vocab, tokenizer,
            max_triples=config.max_triples,
            max_tokens_per_slot=max_tokens_per_slot,
            max_value_tokens=mcfg["max_value_tokens"],
        )
    elif use_domain_vocab:
        from twm.domain_dataset import DomainTripleDataset
        ds = DomainTripleDataset(
            data_file, encode_fn, vocab, tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=mcfg["max_value_tokens"],
        )
    else:
        ds = T5TripleDataset(
            data_file, encode_fn, vocab, tokenizer,
            max_triples=config.max_triples,
            max_value_tokens=mcfg["max_value_tokens"],
        )

    n = min(args.n_examples, len(ds))
    cosine = not args.no_cosine
    print(f"Split: {args.split} ({len(ds)} examples), showing {n}")
    print(f"Denoise steps: {args.denoise_steps}, temperature: {args.temperature}, cosine: {cosine}, guidance_scale: {args.guidance_scale}")
    print()

    gen_kwargs = dict(
        n_steps=args.denoise_steps,
        temperature=args.temperature,
        cosine_schedule=cosine,
        guidance_scale=args.guidance_scale,
    )

    with torch.no_grad():
        latent = model.encode_dynamics(
            ds._all_inputs[:n].to(device),
            ds._all_input_pad_masks[:n].to(device),
        )

        # Discrete attr predictions
        discrete_logits = model.forward_discrete(latent)
        pred_attrs = discrete_logits["attr"].argmax(-1)

        # Diffusion entity + value generation
        gen_entities = model.generate_entities(latent, **gen_kwargs)
        gen_values = model.generate_values(latent, **gen_kwargs)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_attrs = ds._all_target_attr[:n]

    attr_correct = 0
    total_triples = 0

    for i in range(n):
        print(f"--- Example {i} ---")
        M = config.max_triples
        for m in range(M):
            if tgt_pad[i, m]:
                continue
            total_triples += 1

            # Entity (diffusion)
            tgt_e_ids = ds._all_entity_token_ids[i, m]
            tgt_e_str = tokenizer.decode(tgt_e_ids, skip_special_tokens=True).strip()
            pred_e_str = gen_entities[i][m]

            # Attr (classification)
            tgt_a = tgt_attrs[i, m].item()
            pred_a = pred_attrs[i, m].item()
            a_match = tgt_a == pred_a
            attr_correct += int(a_match)
            tgt_a_str = vocab.decode_id(tgt_a, "attr")
            pred_a_str = vocab.decode_id(pred_a, "attr")

            # Value (diffusion)
            tgt_v_ids = ds._all_value_token_ids[i, m]
            tgt_v_str = tokenizer.decode(tgt_v_ids, skip_special_tokens=True).strip()
            pred_v_str = gen_values[i][m]

            mark_a = "Y" if a_match else "N"
            print(f"  [{m}] entity: {tgt_e_str!r} -> {pred_e_str!r}")
            print(f"       attr:   {tgt_a_str} -> {pred_a_str} {mark_a}")
            print(f"       value:  {tgt_v_str!r} -> {pred_v_str!r}")
        print()

    print(f"Attr acc: {attr_correct}/{total_triples} ({attr_correct/max(total_triples,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
