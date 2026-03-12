"""Export static PNG figures for pet sim dynamics analysis.

Generates publication-quality matplotlib figures from the same data
as the interactive plotly visualizations.

Usage:
    uv run python scripts/export_pet_sim_figures.py \
        --checkpoint results/pet_sim \
        --output research/sprint4_figures
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys
sys.path.insert(0, str(Path(__file__).parent))
from visualize_dynamics import (
    enumerate_pet_states, encode_states, extract_latents,
)
from twm.serve import WorldModel
from twm.analysis import dynamics_jacobian


def plot_all_panels(pre_3d, post_3d, labels, var_ratio, eigenvalues, output_path):
    """Single figure with 3D latent space, flow field, and eigenspectrum side by side."""
    fig = plt.figure(figsize=(18, 5.5))

    pets = sorted(set(l["pet"] for l in labels))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(pets)))
    pet_color = {p: colors[i] for i, p in enumerate(pets)}

    # --- Panel 1: 3D latent space ---
    ax = fig.add_subplot(131, projection="3d")
    for pet in pets:
        idx = [i for i, l in enumerate(labels) if l["pet"] == pet]
        c = pet_color[pet]
        ax.scatter(pre_3d[idx, 0], pre_3d[idx, 1], pre_3d[idx, 2],
                   c=[c], s=3, alpha=0.3, label=pet, rasterized=True)
        ax.scatter(post_3d[idx, 0], post_3d[idx, 1], post_3d[idx, 2],
                   c=[c], s=3, alpha=0.3, marker="^", rasterized=True)
    step = max(1, len(pre_3d) // 150)
    for i in range(0, len(pre_3d), step):
        ax.plot([pre_3d[i, 0], post_3d[i, 0]],
                [pre_3d[i, 1], post_3d[i, 1]],
                [pre_3d[i, 2], post_3d[i, 2]],
                color="gray", alpha=0.12, linewidth=0.4)
    total_var = sum(var_ratio[:3])
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%})", fontsize=8)
    ax.set_zlabel(f"PC3 ({var_ratio[2]:.0%})", fontsize=8)
    ax.set_title(f"Latent Space (PCA {total_var:.0%})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", markerscale=2)
    ax.view_init(elev=25, azim=135)
    ax.tick_params(labelsize=7)

    # --- Panel 2: Flow field ---
    ax = fig.add_subplot(132)
    for pet in pets:
        idx = [i for i, l in enumerate(labels) if l["pet"] == pet]
        c = pet_color[pet]
        ax.scatter(pre_3d[idx, 0], pre_3d[idx, 1],
                   c=[c], s=4, alpha=0.35, label=pet, rasterized=True)
    step = max(1, len(pre_3d) // 200)
    disp = post_3d - pre_3d
    for i in range(0, len(pre_3d), step):
        ax.annotate("", xy=(pre_3d[i, 0] + disp[i, 0], pre_3d[i, 1] + disp[i, 1]),
                     xytext=(pre_3d[i, 0], pre_3d[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3, lw=0.5))
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%})", fontsize=8)
    ax.set_title("Flow Field", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)

    # --- Panel 3: Eigenspectrum (complex plane) ---
    ax = fig.add_subplot(133)
    mags = np.abs(eigenvalues)
    scatter = ax.scatter(eigenvalues.real, eigenvalues.imag, c=mags, cmap="coolwarm",
                         s=6, alpha=0.7, vmin=0, vmax=min(mags.max(), 5), rasterized=True)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Re(λ)", fontsize=8)
    ax.set_ylabel("Im(λ)", fontsize=8)
    n_exp = (mags > 1).sum()
    n_con = (mags < 1).sum()
    ax.set_title(f"Jacobian Eigenspectrum\n{n_exp} expansive, {n_con} contractive, mean |λ|={mags.mean():.2f}",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)
    plt.colorbar(scatter, ax=ax, label="|λ|", shrink=0.7)

    fig.suptitle("Pet Sim Dynamics Core — 28K params, 3,780 states", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/pet_sim")
    parser.add_argument("--output", default="research/sprint4_figures")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    wm = WorldModel(args.checkpoint)
    print(f"  {wm.model.param_count():,} params on {wm.device}")

    # Enumerate and encode
    print("\nEnumerating pet states...")
    states, labels = enumerate_pet_states()
    print(f"  {len(states)} states")

    print("Encoding and extracting latents...")
    input_ids = encode_states(wm, states)
    pre_latents, post_latents = extract_latents(wm, input_ids)

    # PCA
    combined = np.concatenate([pre_latents, post_latents])
    pca = PCA(n_components=3)
    pca.fit(combined)
    pre_3d = pca.transform(pre_latents)
    post_3d = pca.transform(post_latents)
    var = pca.explained_variance_ratio_
    print(f"  PCA variance: {var[0]:.1%} + {var[1]:.1%} + {var[2]:.1%} = {sum(var):.1%}")

    # Eigenspectrum
    print("\nComputing Jacobian eigenspectrum...")
    sample_state = [
        ["#mode", "type", "advance"],
        ["Daisy", "hunger", "hungry"],
        ["Daisy", "energy", "tired"],
        ["Daisy", "mood", "content"],
        ["Daisy", "cleanliness", "messy"],
        ["Daisy", "action", "feed"],
    ]
    from twm.dataset import _sort_triples, _pad_triples, _flatten_triples
    sorted_state = _sort_triples(sample_state)
    padded = _pad_triples(sorted_state, wm.config.max_triples)
    ids = _flatten_triples(padded, wm.vocab)
    sample_ids = torch.tensor([ids], dtype=torch.long)
    eigenvalues, J = dynamics_jacobian(wm.model, sample_ids, device=str(wm.device))

    # Combined figure
    print("\nGenerating combined panel figure...")
    plot_all_panels(pre_3d, post_3d, labels, var, eigenvalues, out / "dynamics_analysis.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
