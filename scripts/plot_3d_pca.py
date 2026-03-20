#!/usr/bin/env python3
"""3D PCA scatter of bottleneck embeddings, colored by mode (identity vs QA).

Usage:
    uv run python scripts/plot_3d_pca.py \
        --config configs/v49_diffcomp_k5_joint.json \
        --checkpoint results/v49_diffcomp_k5_joint/joint_all_phase1/model_best.pt \
        --out plot_3d.png
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from twm.config import ModelConfig
from twm.text_dynamics_model import TextDynamicsModel
from twm.domain_bpe import DomainBPETokenizer
from twm.text_pair_dataset import TextPairDataset


def load_model(config_path, checkpoint_path):
    with open(config_path) as f:
        cfg = json.load(f)
    tok = DomainBPETokenizer.load(cfg["tokenizer_path"], max_length=cfg["max_text_tokens"])
    mc = ModelConfig.from_profile(cfg["profile"], max_triples=cfg["max_triples"])
    if "d_model" in cfg:
        mc.d_model = cfg["d_model"]
    mc.d_ff = mc.d_model * 4

    compressor_kwargs = {}
    if cfg.get("compressor_type"):
        compressor_kwargs["compressor_type"] = cfg["compressor_type"]
        compressor_kwargs["compressor_denoise_steps"] = cfg.get("compressor_denoise_steps", 5)
        compressor_kwargs["compressor_denoise_layers"] = cfg.get("compressor_denoise_layers")
        compressor_kwargs["compressor_random_k"] = cfg.get("compressor_random_k", False)
        compressor_kwargs["compressor_k_min"] = cfg.get("compressor_k_min", 1)

    model = TextDynamicsModel(
        config=mc, domain_tokenizer=tok,
        text_compressor_layers=cfg["text_compressor_layers"],
        text_expander_layers=cfg["text_expander_layers"],
        dynamics_layers=cfg.get("dynamics_layers", 4),
        max_text_tokens=cfg["max_text_tokens"],
        dropout=cfg["dropout"], alpha_min=cfg["alpha_min"],
        vae=cfg.get("vae", False),
        **compressor_kwargs,
    )
    model.init_embeddings()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    return model, tok, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="plot_3d.png")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    model, tok, cfg = load_model(args.config, args.checkpoint)
    data_dir = Path(cfg["data_dir"])
    test_file = data_dir / "qa_balanced_test.jsonl"
    if not test_file.exists():
        test_file = data_dir / "test.jsonl"
    dataset = TextPairDataset(str(test_file), tok, max_text_tokens=cfg["max_text_tokens"])

    n = min(args.n, len(dataset))
    input_ids = dataset._input_token_ids[:n]
    input_pad = dataset._input_pad_mask[:n]
    modes = dataset._modes[:n].numpy()

    with torch.no_grad():
        compress_out = model.compress(input_ids, input_pad)
        bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out

    flat = bottleneck.reshape(n, -1).numpy()
    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat)
    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    mode_labels = {0: "identity", 1: "QA"}
    colors = {0: "#4C72B0", 1: "#DD8452"}

    for m in sorted(set(modes)):
        mask = modes == m
        ax.scatter(
            proj[mask, 0], proj[mask, 1], proj[mask, 2],
            c=colors.get(m, "gray"), label=mode_labels.get(m, f"mode {m}"),
            alpha=0.5, s=8, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var[2]:.1%})")
    ax.legend(fontsize=11, markerscale=3)
    ax.set_title(f"v49 Bottleneck PCA — {var[:3].sum():.1%} variance explained", fontsize=14)

    ax.view_init(elev=25, azim=135)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
