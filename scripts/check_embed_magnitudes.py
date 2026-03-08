#!/usr/bin/env python3
"""Check if MASK embedding magnitude drowns out positional embeddings."""

import json
from pathlib import Path
import torch

model_dir = Path("/tmp/results/v2_1L128d_10k_pre")

with open(model_dir / "model_config.json") as f:
    mcfg = json.load(f)

sd = torch.load(model_dir / "model_best.pt", map_location="cpu", weights_only=True)

MASK_ID = 32099

for name, prefix in [("entity_decoder", "entity_decoder"), ("value_decoder", "value_decoder")]:
    tok_emb = sd[f"{prefix}.token_emb.weight"]  # (vocab, d_model)
    pos_emb = sd[f"{prefix}.pos_emb.weight"]     # (max_seq_len, d_model)

    mask_vec = tok_emb[MASK_ID]
    mask_norm = mask_vec.norm().item()

    pos_norms = pos_emb.norm(dim=-1)
    pos_mean = pos_norms.mean().item()
    pos_min = pos_norms.min().item()
    pos_max = pos_norms.max().item()

    # How different are positions from each other?
    pos_pairwise = torch.cdist(pos_emb.unsqueeze(0), pos_emb.unsqueeze(0)).squeeze(0)
    pos_spread = pos_pairwise.mean().item()

    # Cosine similarity between MASK+pos[i] vectors
    masked_vecs = mask_vec.unsqueeze(0) + pos_emb  # (S, d_model)
    masked_normed = masked_vecs / masked_vecs.norm(dim=-1, keepdim=True)
    cos_sim = (masked_normed @ masked_normed.T)
    # Average off-diagonal cosine similarity
    S = cos_sim.shape[0]
    off_diag = cos_sim[~torch.eye(S, dtype=torch.bool)]
    avg_cos = off_diag.mean().item()
    min_cos = off_diag.min().item()

    print(f"=== {name} ===")
    print(f"  MASK embedding norm:     {mask_norm:.4f}")
    print(f"  Position embedding norm: mean={pos_mean:.4f}, min={pos_min:.4f}, max={pos_max:.4f}")
    print(f"  Ratio (MASK/pos_mean):   {mask_norm/pos_mean:.2f}x")
    print(f"  Position pairwise dist:  {pos_spread:.4f}")
    print(f"  MASK+pos cosine sim:     avg={avg_cos:.4f}, min={min_cos:.4f}")
    print(f"    (1.0 = identical, <0.9 = distinguishable)")
    print()
