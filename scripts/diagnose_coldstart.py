#!/usr/bin/env python3
"""Phase 1: Diagnose cold-start problem in diffusion generation.

Test generation at multiple mask ratios to find the cliff where
accuracy and diversity collapse. No training needed — uses existing checkpoint.

Usage:
    uv run python scripts/diagnose_coldstart.py \
        --model-dir results/v2_1L128d_10k_pre \
        --data-dir data/atomic_10000 \
        --n-examples 64 --device cuda:1
"""

import argparse
import json
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

from twm.config import ModelConfig
from twm.phrase_vocab import PhraseVocab
from twm.diffusion_model import DiffusionWorldModel
from twm.t5_dataset import T5TripleDataset


def generate_from_partial(
    decoder,
    triple_context: torch.Tensor,
    target_ids: torch.Tensor,
    mask_ratio: float,
    n_steps: int = 10,
    cosine_schedule: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate with partial masking: mask `mask_ratio` fraction of positions,
    keep the rest as ground truth. Returns predicted ids and the mask.

    Args:
        decoder: DiffusionDecoder
        triple_context: (B, 3*twm_dim) conditioning
        target_ids: (B, S) ground truth token ids
        mask_ratio: fraction of positions to mask (1.0 = fully masked)
        n_steps: denoising steps
        cosine_schedule: use cosine unmasking schedule

    Returns:
        pred_ids: (B, S) final predicted token ids
        mask: (B, S) bool, True where positions were originally masked
    """
    B, S = target_ids.shape
    device = target_ids.device

    # Create mask: randomly select positions to mask
    n_to_mask = max(1, int(mask_ratio * S))
    if n_to_mask >= S:
        # Full mask — all positions
        mask = torch.ones(B, S, dtype=torch.bool, device=device)
    else:
        rand_scores = torch.rand(B, S, device=device)
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        for i in range(B):
            _, indices = rand_scores[i].topk(n_to_mask, largest=False)
            mask[i, indices] = True

    # Initialize: masked positions get MASK token, rest keep ground truth
    ids = target_ids.clone()
    ids[mask] = decoder.mask_token_id
    is_masked = mask.clone()

    pos_idx = torch.arange(S, device=device).unsqueeze(0)

    for step in range(n_steps):
        if cosine_schedule:
            t = (step + 1) / n_steps
            frac_unmasked = 1.0 - math.cos(t * math.pi / 2)
            n_target_unmasked_total = max(1, int(frac_unmasked * S))
        else:
            n_target_unmasked_total = max(1, int((step + 1) / n_steps * S))

        x = decoder.token_emb(ids) + decoder.pos_emb(pos_idx)
        decoded = decoder._run_denoiser(x, triple_context)
        decoded = decoder.ln_f(decoded)
        logits = decoder.output_head(decoded)

        probs = F.softmax(logits, dim=-1)
        confidence, pred_ids = probs.max(dim=-1)
        confidence[~is_masked] = float('inf')

        for i in range(B):
            n_currently_unmasked = (~is_masked[i]).sum().item()
            n_to_reveal = max(0, n_target_unmasked_total - n_currently_unmasked)
            n_reveal = min(n_to_reveal, is_masked[i].sum().item())
            if n_reveal > 0:
                masked_confidence = confidence[i].clone()
                masked_confidence[~is_masked[i]] = -1.0
                _, top_idx = masked_confidence.topk(n_reveal)
                ids[i, top_idx] = pred_ids[i, top_idx]
                is_masked[i, top_idx] = False

    # Final pass
    if is_masked.any():
        x = decoder.token_emb(ids) + decoder.pos_emb(pos_idx)
        decoded = decoder._run_denoiser(x, triple_context)
        decoded = decoder.ln_f(decoded)
        logits = decoder.output_head(decoded)
        pred_ids = logits.argmax(dim=-1)
        ids[is_masked] = pred_ids[is_masked]

    return ids, mask


def main():
    ap = argparse.ArgumentParser(description="Diagnose cold-start in diffusion generation")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--n-examples", type=int, default=64)
    ap.add_argument("--denoise-steps", type=int, default=10)
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

    st_model_name = args.st_model or mcfg["st_model"]

    # Sentence encoder
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name, device=str(device))

    def encode_fn(phrases):
        return st.encode(phrases, batch_size=256,
                         show_progress_bar=False,
                         convert_to_tensor=True, device=str(device))

    # Vocab
    vocab = PhraseVocab.load(model_dir / "phrase_vocab.json")

    # Model
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

    ckpt = args.checkpoint or "model_best.pt"
    ckpt_path = model_dir / ckpt
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model_final.pt"
    print(f"Loading {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # v2 checkpoints used "denoiser.layers" but v5 code uses "layers" directly
    fixed_sd = {}
    for k, v in sd.items():
        new_k = k.replace(".denoiser.layers.", ".layers.")
        fixed_sd[new_k] = v
    model.load_state_dict(fixed_sd)
    model.eval()

    # Dataset
    ds = T5TripleDataset(
        data_dir / "test.jsonl", encode_fn, vocab, t5_tokenizer,
        max_triples=config.max_triples,
        max_value_tokens=mcfg["max_value_tokens"],
    )

    n = min(args.n_examples, len(ds))
    M = config.max_triples

    print(f"Testing {n} examples, {M} triples each, {args.denoise_steps} denoise steps")
    print()

    # Encode dynamics once
    with torch.no_grad():
        latent = model.encode_dynamics(
            ds._all_inputs[:n].to(device),
            ds._all_input_pad_masks[:n].to(device),
        )
        triple_ctx = model._extract_triple_context(latent)  # (n, M, 3*d_model)

    tgt_pad = ds._all_target_pad_masks[:n]
    tgt_entity_ids = ds._all_entity_token_ids[:n]  # (n, M, S)
    tgt_value_ids = ds._all_value_token_ids[:n]  # (n, M, S)

    mask_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    # Header
    print(f"{'mask_ratio':>10} | {'val_tok_acc':>10} | {'ent_tok_acc':>10} | {'unique_vals':>11} | {'unique_ents':>11} | top_3_values")
    print("-" * 100)

    for mr in mask_ratios:
        val_correct = 0
        val_total = 0
        ent_correct = 0
        ent_total = 0
        all_pred_values = []
        all_pred_entities = []

        with torch.no_grad():
            for i in range(n):
                for m in range(M):
                    if tgt_pad[i, m]:
                        continue

                    ctx = triple_ctx[i, m].unsqueeze(0)  # (1, 3*d_model)

                    # Value generation
                    val_tgt = tgt_value_ids[i, m].unsqueeze(0).to(device)
                    val_pred, val_mask = generate_from_partial(
                        model.value_decoder, ctx, val_tgt, mr,
                        n_steps=args.denoise_steps,
                    )
                    masked_correct = (val_pred[0][val_mask[0]] == val_tgt[0][val_mask[0]]).sum().item()
                    masked_total = val_mask[0].sum().item()
                    val_correct += masked_correct
                    val_total += masked_total

                    pred_val_str = t5_tokenizer.decode(val_pred[0], skip_special_tokens=True).strip()
                    all_pred_values.append(pred_val_str)

                    # Entity generation
                    ent_tgt = tgt_entity_ids[i, m].unsqueeze(0).to(device)
                    ent_pred, ent_mask = generate_from_partial(
                        model.entity_decoder, ctx, ent_tgt, mr,
                        n_steps=args.denoise_steps,
                    )
                    masked_correct = (ent_pred[0][ent_mask[0]] == ent_tgt[0][ent_mask[0]]).sum().item()
                    masked_total = ent_mask[0].sum().item()
                    ent_correct += masked_correct
                    ent_total += masked_total

                    pred_ent_str = t5_tokenizer.decode(ent_pred[0], skip_special_tokens=True).strip()
                    all_pred_entities.append(pred_ent_str)

        val_acc = val_correct / max(val_total, 1) * 100
        ent_acc = ent_correct / max(ent_total, 1) * 100
        unique_vals = len(set(all_pred_values))
        unique_ents = len(set(all_pred_entities))

        val_counter = Counter(all_pred_values)
        top3_vals = val_counter.most_common(3)
        top3_str = ", ".join(f"'{v}'({c})" for v, c in top3_vals)

        print(f"{mr:>10.2f} | {val_acc:>9.1f}% | {ent_acc:>9.1f}% | {unique_vals:>11} | {unique_ents:>11} | {top3_str}")

    print()
    print("Look for a sharp cliff in accuracy and diversity between mask ratios.")
    print("If cliff exists between e.g. 0.8 and 0.9, cold-start hypothesis is confirmed.")


if __name__ == "__main__":
    main()
