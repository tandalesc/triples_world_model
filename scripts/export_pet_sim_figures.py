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


def plot_latent_space_3d(pre_3d, post_3d, labels, var_ratio, output_path):
    """3D scatter of pre/post dynamics, colored by pet."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    pets = sorted(set(l["pet"] for l in labels))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(pets)))
    pet_color = {p: colors[i] for i, p in enumerate(pets)}

    for pet in pets:
        idx = [i for i, l in enumerate(labels) if l["pet"] == pet]
        c = pet_color[pet]
        ax.scatter(pre_3d[idx, 0], pre_3d[idx, 1], pre_3d[idx, 2],
                   c=[c], s=4, alpha=0.3, label=f"{pet} (pre)", rasterized=True)
        ax.scatter(post_3d[idx, 0], post_3d[idx, 1], post_3d[idx, 2],
                   c=[c], s=4, alpha=0.3, marker="^", rasterized=True)

    # Flow lines (subsample)
    step = max(1, len(pre_3d) // 200)
    for i in range(0, len(pre_3d), step):
        ax.plot([pre_3d[i, 0], post_3d[i, 0]],
                [pre_3d[i, 1], post_3d[i, 1]],
                [pre_3d[i, 2], post_3d[i, 2]],
                color="gray", alpha=0.15, linewidth=0.5)

    total_var = sum(var_ratio[:3])
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%})")
    ax.set_zlabel(f"PC3 ({var_ratio[2]:.0%})")
    ax.set_title(f"Pet Sim Dynamics Latent Space\n3,780 states, PCA {total_var:.1%} variance")
    ax.legend(fontsize=8, loc="upper left", markerscale=2)
    ax.view_init(elev=25, azim=135)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_flow_field_2d(pre_3d, post_3d, labels, var_ratio, output_path):
    """2D flow field with displacement arrows, colored by pet."""
    fig, ax = plt.subplots(figsize=(9, 7))

    pets = sorted(set(l["pet"] for l in labels))
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(pets)))
    pet_color = {p: colors[i] for i, p in enumerate(pets)}

    # Plot points
    for pet in pets:
        idx = [i for i, l in enumerate(labels) if l["pet"] == pet]
        c = pet_color[pet]
        ax.scatter(pre_3d[idx, 0], pre_3d[idx, 1],
                   c=[c], s=6, alpha=0.4, label=pet, zorder=2, rasterized=True)

    # Displacement arrows (subsample)
    step = max(1, len(pre_3d) // 250)
    disp = post_3d - pre_3d
    for i in range(0, len(pre_3d), step):
        ax.annotate("", xy=(pre_3d[i, 0] + disp[i, 0], pre_3d[i, 1] + disp[i, 1]),
                     xytext=(pre_3d[i, 0], pre_3d[i, 1]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.35, lw=0.7))

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%})")
    ax.set_title("Dynamics Flow Field (PC1 vs PC2)")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_eigenspectrum(eigenvalues, output_path):
    """Eigenvalue magnitude distribution + complex plane plot."""
    mags = np.abs(eigenvalues)
    reals = eigenvalues.real
    imags = eigenvalues.imag

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Magnitude histogram
    ax = axes[0]
    ax.hist(mags, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="|λ| = 1")
    ax.set_xlabel("|λ|")
    ax.set_ylabel("Count")
    ax.set_title(f"Eigenvalue Magnitudes\nmean={mags.mean():.2f}, range=[{mags.min():.3f}, {mags.max():.2f}]")
    ax.legend()

    n_exp = (mags > 1).sum()
    n_con = (mags < 1).sum()
    ax.text(0.97, 0.95, f"expansive: {n_exp}\ncontractive: {n_con}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Complex plane
    ax = axes[1]
    scatter = ax.scatter(reals, imags, c=mags, cmap="coolwarm", s=8, alpha=0.7,
                         vmin=0, vmax=min(mags.max(), 5))
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Eigenvalues in Complex Plane")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.colorbar(scatter, ax=ax, label="|λ|", shrink=0.8)

    fig.suptitle(f"Jacobian Eigenspectrum ({len(eigenvalues)}×{len(eigenvalues)} dynamics map)",
                 fontsize=13, y=1.02)
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

    # Plot 1: 3D latent space
    print("\nGenerating 3D latent space plot...")
    plot_latent_space_3d(pre_3d, post_3d, labels, var, out / "latent_space.png")

    # Plot 2: Flow field
    print("Generating flow field...")
    plot_flow_field_2d(pre_3d, post_3d, labels, var, out / "flow_field.png")

    # Plot 3: Eigenspectrum
    print("Computing Jacobian eigenspectrum...")
    # Use a representative state: Daisy, hungry, tired, content, messy, feed
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
    plot_eigenspectrum(eigenvalues, out / "eigenspectrum.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
