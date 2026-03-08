#!/usr/bin/env python3
"""Export pet_sim_v2 model weights to JSON for browser inference.

Run once:  python export_weights.py
Creates:   model_weights.json (~280KB)
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from twm.serve import WorldModel


def to_list(t, decimals=6):
    """Convert tensor to nested Python lists, rounding for compact JSON."""
    if t.dim() == 0:
        return round(t.item(), decimals)
    return [to_list(x, decimals) for x in t]


def main():
    wm = WorldModel(str(ROOT / "results" / "pet_sim_v2"), device="cpu")
    sd = wm.model.state_dict()

    weights = {
        "config": {
            "vocab_size": wm.config.vocab_size,
            "d_model": wm.config.d_model,
            "n_heads": wm.config.n_heads,
            "n_layers": wm.config.n_layers,
            "d_ff": wm.config.d_ff,
            "max_triples": wm.config.max_triples,
        },
        "vocab": wm.vocab.token2id,
        "id2token": {str(k): v for k, v in wm.vocab.id2token.items()},
        "embeddings": {
            "token": to_list(sd["triple_encoder.token_emb.weight"]),
            "triple_pos": to_list(sd["triple_encoder.triple_pos_emb.weight"]),
            "role": to_list(sd["triple_encoder.role_emb.weight"]),
        },
        "layers": [],
        "decoder": {
            "ln_weight": to_list(sd["triple_decoder.ln_f.weight"]),
            "ln_bias": to_list(sd["triple_decoder.ln_f.bias"]),
            "head_weight": to_list(sd["triple_decoder.head.weight"]),
            "head_bias": to_list(sd["triple_decoder.head.bias"]),
        },
    }

    for i in range(wm.config.n_layers):
        p = f"dynamics.encoder.layers.{i}"
        weights["layers"].append({
            "norm1_weight": to_list(sd[f"{p}.norm1.weight"]),
            "norm1_bias": to_list(sd[f"{p}.norm1.bias"]),
            "attn_in_proj_weight": to_list(sd[f"{p}.self_attn.in_proj_weight"]),
            "attn_in_proj_bias": to_list(sd[f"{p}.self_attn.in_proj_bias"]),
            "attn_out_proj_weight": to_list(sd[f"{p}.self_attn.out_proj.weight"]),
            "attn_out_proj_bias": to_list(sd[f"{p}.self_attn.out_proj.bias"]),
            "norm2_weight": to_list(sd[f"{p}.norm2.weight"]),
            "norm2_bias": to_list(sd[f"{p}.norm2.bias"]),
            "ff_linear1_weight": to_list(sd[f"{p}.linear1.weight"]),
            "ff_linear1_bias": to_list(sd[f"{p}.linear1.bias"]),
            "ff_linear2_weight": to_list(sd[f"{p}.linear2.weight"]),
            "ff_linear2_bias": to_list(sd[f"{p}.linear2.bias"]),
        })

    out_path = Path(__file__).parent / "model_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f)

    size_kb = out_path.stat().st_size / 1024
    print(f"Exported {wm.model.param_count():,} params to {out_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
