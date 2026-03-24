#!/usr/bin/env python3
"""Plot PCA of bottleneck space from chain dynamics training.

Shows compressor output and post-dynamics bottleneck colored by mode (advance/query/identity),
with arrows showing the dynamics transformation.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset
from torch.utils.data import DataLoader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TextDynamicsModel.load(run_dir, device=str(device))
    model.eval()

    # Load data
    tokenizer = model.tokenizer
    ds = ChainDataset(args.data, tokenizer, max_text_tokens=model.max_text_tokens)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # Collect bottlenecks
    pre_dyn = []   # compressor output (before dynamics)
    post_dyn = []  # after dynamics
    modes = []

    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            input_pad = batch["input_pad"].to(device)
            mode_id = batch["mode_id"].to(device)

            bn = model.compress(input_ids, input_pad)
            if isinstance(bn, tuple):
                bn = bn[0]

            bn_post = model.forward_dynamics(bn, mode_id)

            # Mean-pool across triple positions for visualization
            pre_dyn.append(bn.mean(dim=1).cpu().numpy())
            post_dyn.append(bn_post.mean(dim=1).cpu().numpy())
            modes.append(mode_id.cpu().numpy())

            n += len(input_ids)
            if n >= args.n_examples:
                break

    pre_dyn = np.concatenate(pre_dyn)[:args.n_examples]
    post_dyn = np.concatenate(post_dyn)[:args.n_examples]
    modes = np.concatenate(modes)[:args.n_examples]

    # PCA on combined pre+post
    combined = np.concatenate([pre_dyn, post_dyn])
    pca = PCA(n_components=2)
    pca.fit(combined)
    pre_2d = pca.transform(pre_dyn)
    post_2d = pca.transform(post_dyn)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    mode_names = {0: "advance", 1: "query", 2: "identity"}
    colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}

    # Panel 1: Pre-dynamics colored by mode
    ax = axes[0]
    for m, name in mode_names.items():
        mask = modes == m
        if mask.any():
            ax.scatter(pre_2d[mask, 0], pre_2d[mask, 1], c=colors[m],
                      alpha=0.4, s=10, label=name)
    ax.set_title("Compressor output (pre-dynamics)")
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Panel 2: Post-dynamics colored by mode
    ax = axes[1]
    for m, name in mode_names.items():
        mask = modes == m
        if mask.any():
            ax.scatter(post_2d[mask, 0], post_2d[mask, 1], c=colors[m],
                      alpha=0.4, s=10, label=name)
    ax.set_title("Post-dynamics output")
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Panel 3: Flow arrows (pre → post)
    ax = axes[2]
    step = max(1, len(pre_2d) // 200)  # subsample for readability
    for m, name in mode_names.items():
        mask = modes == m
        idx = np.where(mask)[0][::step]
        ax.quiver(pre_2d[idx, 0], pre_2d[idx, 1],
                 post_2d[idx, 0] - pre_2d[idx, 0],
                 post_2d[idx, 1] - pre_2d[idx, 1],
                 color=colors[m], alpha=0.5, scale=1, scale_units="xy",
                 angles="xy", width=0.003, label=name)
    ax.set_title("Dynamics flow (pre → post)")
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    plt.suptitle(f"Chain Dynamics PCA — {run_dir.name}", fontsize=14)
    plt.tight_layout()

    out_path = args.output or str(run_dir / "pca_chain.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
