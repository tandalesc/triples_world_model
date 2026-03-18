"""Latent space visualization for TWM models.

Generates two visualizations:
1. Role-conditioned covariance heatmap — shows pairwise distances between
   E/A/V slots and how role structure evolves.
2. Bottleneck decode grid — samples a grid of points in PC1/PC2 space,
   decodes each one, shows what text comes out at each region.

Usage:
    uv run python scripts/visualize_latent_space.py \
        --config configs/v38_balanced_joint.json \
        --checkpoint results/v38_balanced_joint/joint_all_phase2/model_best.pt \
        --out-dir results/v38_balanced_joint/visualizations
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
    cfg = json.load(open(config_path))
    tok = DomainBPETokenizer.load(cfg["tokenizer_path"], max_length=cfg["max_text_tokens"])
    mc = ModelConfig.from_profile(cfg["profile"], max_triples=cfg["max_triples"])
    if "d_model" in cfg:
        mc.d_model = cfg["d_model"]
    mc.d_ff = mc.d_model * 4

    model = TextDynamicsModel(
        config=mc, domain_tokenizer=tok,
        text_compressor_layers=cfg["text_compressor_layers"],
        text_expander_layers=cfg["text_expander_layers"],
        dynamics_layers=cfg.get("dynamics_layers", 4),
        max_text_tokens=cfg["max_text_tokens"],
        dropout=cfg["dropout"], alpha_min=cfg["alpha_min"],
        vae=cfg.get("vae", False),
    )
    model.init_embeddings()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    return model, tok, cfg


def get_bottlenecks(model, dataset, n=500, device="cpu"):
    """Compress dataset examples and return bottleneck vectors + metadata."""
    n = min(n, len(dataset))

    if hasattr(dataset, "_input_token_ids"):
        input_ids = dataset._input_token_ids[:n].to(device)
        input_pad = dataset._input_pad_mask[:n].to(device)
        modes = dataset._modes[:n]
    else:
        input_ids = dataset._text_token_ids[:n].to(device)
        input_pad = dataset._text_pad_mask[:n].to(device)
        modes = torch.zeros(n, dtype=torch.long)

    with torch.no_grad():
        compress_out = model.compress(input_ids, input_pad)
        bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out

    return bottleneck.cpu(), modes


def plot_role_covariance(bottleneck, max_triples, out_path):
    """Plot role-conditioned covariance heatmap."""
    B, N3, D = bottleneck.shape
    N = max_triples

    slot_means = bottleneck[:, :N*3, :].mean(dim=0)

    norms = slot_means.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = slot_means / norms
    cos_sim = (normalized @ normalized.T).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8)

    role_labels = []
    for i in range(N):
        role_labels.extend([f"E{i}", f"A{i}", f"V{i}"])
    ax.set_xticks(range(N*3))
    ax.set_xticklabels(role_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(N*3))
    ax.set_yticklabels(role_labels, fontsize=6)
    ax.set_title("Slot Cosine Similarity (mean vectors)")

    for i in range(1, N):
        ax.axhline(i*3 - 0.5, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(i*3 - 0.5, color="black", linewidth=0.5, alpha=0.3)

    role_names = ["Entity", "Attribute", "Value"]
    role_avg = np.zeros((3, 3))
    for r1 in range(3):
        for r2 in range(3):
            vals = []
            for i in range(N):
                for j in range(N):
                    vals.append(cos_sim[i*3 + r1, j*3 + r2])
            role_avg[r1, r2] = np.mean(vals)

    ax = axes[1]
    im = ax.imshow(role_avg, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(role_names, fontsize=12)
    ax.set_yticks(range(3))
    ax.set_yticklabels(role_names, fontsize=12)
    ax.set_title("Role-Averaged Cosine Similarity")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{role_avg[i,j]:.3f}",
                   ha="center", va="center", fontsize=14,
                   color="white" if abs(role_avg[i,j]) > 0.5 else "black")

    fig.suptitle("Role-Conditioned Bottleneck Structure", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved role covariance: {out_path}")


def plot_decode_grid(model, bottleneck, tokenizer, out_path,
                     grid_size=8, n_steps=10):
    """Sample a grid in PC1/PC2 space, decode each point, show text."""
    B, N3, D = bottleneck.shape

    flat = bottleneck.reshape(B, -1).numpy()
    pca = PCA(n_components=2)
    projected = pca.fit_transform(flat)

    p5, p95 = np.percentile(projected, [5, 95], axis=0)
    x_range = np.linspace(p5[0], p95[0], grid_size)
    y_range = np.linspace(p95[1], p5[1], grid_size)

    fig, ax = plt.subplots(figsize=(grid_size * 3, grid_size * 2.5))
    ax.set_xlim(p5[0], p95[0])
    ax.set_ylim(p5[1], p95[1])

    ax.scatter(projected[:, 0], projected[:, 1], alpha=0.1, s=5, c="gray")

    with torch.no_grad():
        for y in y_range:
            for x in x_range:
                pca_point = np.array([[x, y]])
                flat_point = pca.inverse_transform(pca_point)
                bn_point = torch.tensor(flat_point, dtype=torch.float32).reshape(1, N3, D)

                mode_ids = torch.zeros(1, dtype=torch.long)
                bn_dyn = model.forward_dynamics(bn_point, mode_ids)

                gen_ids = model.generate(bn_dyn, n_steps=n_steps)
                pred_len = model.forward_length(bn_dyn)
                pl = pred_len.round().long().clamp(1, gen_ids.shape[-1]).item()

                text = tokenizer.decode(gen_ids[0][:pl].cpu())
                if len(text) > 40:
                    text = text[:37] + "..."

                ax.text(x, y, text, fontsize=5, ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                alpha=0.8, edgecolor="lightgray"),
                       clip_on=True)

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Latent Space Decode Grid (identity mode)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved decode grid: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tok, cfg = load_model(args.config, args.checkpoint)
    max_triples = cfg["max_triples"]

    data_dir = Path(args.data_dir or cfg["data_dir"])
    print("Loading dataset...")
    dataset = TextPairDataset(
        data_dir / "qa_balanced_train.jsonl", tok,
        max_text_tokens=cfg["max_text_tokens"],
        max_examples=args.n_examples,
    )

    print("Computing bottlenecks...")
    bottleneck, modes = get_bottlenecks(model, dataset, n=args.n_examples)

    print("Generating role covariance heatmap...")
    plot_role_covariance(bottleneck, max_triples, out_dir / "role_covariance.png")

    print("Generating decode grid...")
    plot_decode_grid(model, bottleneck, tok, out_dir / "decode_grid.png",
                    grid_size=args.grid_size, n_steps=args.n_steps)

    print("Done!")


if __name__ == "__main__":
    main()
