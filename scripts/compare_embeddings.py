#!/usr/bin/env python3
"""Compare distributional role embeddings vs compressor bottleneck space.

Generates a side-by-side PCA visualization showing:
- Left: distributional embeddings (from spaCy + sentence encoder)
- Right: compressor bottleneck embeddings (from trained model)
Both colored by role (entity/subject, attribute/predicate, value/object).

Usage:
    uv run python scripts/compare_embeddings.py \
        --config configs/v50_diffcomp_k5_joint_d128.json \
        --checkpoint results/v50_diffcomp_k5_joint_d128/joint_all_phase1/model_best.pt \
        --distributional data/distributional_triples/ \
        --out results/embedding_comparison.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from twm.config import ModelConfig
from twm.text_dynamics_model import TextDynamicsModel
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dataset import TextDataset


def load_distributional(dist_dir):
    """Load distributional role embeddings."""
    dist_dir = Path(dist_dir)
    role_map = {"subject": "E", "predicate": "A", "object": "V"}
    all_embs = []
    all_roles = []
    all_spans = []

    for role_name, label in role_map.items():
        embs = np.load(dist_dir / f"{role_name}_embeddings.npy")
        with open(dist_dir / f"{role_name}_spans.json") as f:
            meta = json.load(f)

        # Filter to spans with enough occurrences
        counts = meta["counts"]
        spans = meta["spans"]
        mask = [c >= 10 for c in counts]
        embs = embs[mask]
        spans_filtered = [s for s, m in zip(spans, mask) if m]

        all_embs.append(embs)
        all_roles.extend([label] * len(embs))
        all_spans.extend(spans_filtered)

    return np.vstack(all_embs), all_roles, all_spans


def load_compressor_bottleneck(config_path, checkpoint_path, max_samples=500):
    """Load model and compute bottleneck embeddings."""
    with open(config_path) as f:
        cfg = json.load(f)

    tokenizer = DomainBPETokenizer.load(cfg["tokenizer_path"], max_length=cfg["max_text_tokens"])
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
        config=mc, domain_tokenizer=tokenizer,
        text_compressor_layers=cfg["text_compressor_layers"],
        text_expander_layers=cfg["text_expander_layers"],
        dynamics_layers=cfg.get("dynamics_layers", 4),
        max_text_tokens=cfg["max_text_tokens"],
        dropout=cfg["dropout"], alpha_min=cfg["alpha_min"],
        vae=cfg.get("vae", False),
        **compressor_kwargs,
    )
    model.init_embeddings()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load dataset
    ds = TextDataset(
        str(Path(cfg["data_dir"]) / "identity_train.jsonl"),
        tokenizer, cfg["max_text_tokens"],
        max_examples=max_samples,
    )

    all_embs = []
    all_roles = []

    with torch.no_grad():
        # Process in batches
        batch_size = 64
        for start in range(0, len(ds._text_token_ids), batch_size):
            end = min(start + batch_size, len(ds._text_token_ids))
            ids = ds._text_token_ids[start:end]
            pad = ds._text_pad_mask[start:end]

            bottleneck = model.compress(ids, pad)
            if isinstance(bottleneck, tuple):
                bottleneck = bottleneck[1].get("mu", bottleneck[0])

            # bottleneck shape: (B, N*3, d)
            B, T, d = bottleneck.shape
            embs = bottleneck.numpy()

            for b in range(B):
                for t in range(T):
                    role_idx = t % 3
                    role = ["E", "A", "V"][role_idx]
                    all_embs.append(embs[b, t])
                    all_roles.append(role)

    return np.stack(all_embs), all_roles


def plot_comparison(dist_embs, dist_roles, comp_embs, comp_roles, out_path):
    """Side-by-side PCA plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {"E": "#e74c3c", "A": "#3498db", "V": "#2ecc71"}
    labels_map = {"E": "Entity/Subject", "A": "Attribute/Predicate", "V": "Value/Object"}

    # Distributional embeddings
    ax = axes[0]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(dist_embs)

    for role in ["E", "A", "V"]:
        mask = [r == role for r in dist_roles]
        ax.scatter(proj[mask, 0], proj[mask, 1], c=colors[role],
                   label=labels_map[role], alpha=0.5, s=15)

    var_ratio = pca.explained_variance_ratio_
    ax.set_title(f"Distributional Role Embeddings\n"
                 f"({len(dist_embs)} spans, {dist_embs.shape[1]}d)\n"
                 f"PC1={var_ratio[0]:.1%}  PC2={var_ratio[1]:.1%}", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=9)

    # Compute per-role PC1 variance
    role_list = ["E", "A", "V"]
    for i, role in enumerate(role_list):
        mask = [r == role for r in dist_roles]
        role_var = np.var(proj[mask, 0])
        ax.text(0.02, 0.98 - i * 0.05,
                f"{role} PC1 var={role_var:.3f}",
                transform=ax.transAxes, fontsize=8, va='top', color=colors[role])

    # Compressor bottleneck
    ax = axes[1]

    # Subsample if too many points
    if len(comp_embs) > 5000:
        idx = np.random.choice(len(comp_embs), 5000, replace=False)
        comp_embs_plot = comp_embs[idx]
        comp_roles_plot = [comp_roles[i] for i in idx]
    else:
        comp_embs_plot = comp_embs
        comp_roles_plot = comp_roles

    pca2 = PCA(n_components=2)
    proj2 = pca2.fit_transform(comp_embs_plot)

    for role in ["E", "A", "V"]:
        mask = [r == role for r in comp_roles_plot]
        ax.scatter(proj2[mask, 0], proj2[mask, 1], c=colors[role],
                   label=labels_map[role], alpha=0.5, s=15)

    var_ratio2 = pca2.explained_variance_ratio_
    ax.set_title(f"Compressor Bottleneck (v50 d=128)\n"
                 f"({len(comp_embs_plot)} slots, {comp_embs.shape[1]}d)\n"
                 f"PC1={var_ratio2[0]:.1%}  PC2={var_ratio2[1]:.1%}", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=9)

    for i, role in enumerate(role_list):
        mask = [r == role for r in comp_roles_plot]
        role_var = np.var(proj2[mask, 0])
        ax.text(0.02, 0.98 - i * 0.05,
                f"{role} PC1 var={role_var:.3f}",
                transform=ax.transAxes, fontsize=8, va='top', color=colors[role])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # Print summary stats
    print(f"\n=== Geometry Comparison ===")
    print(f"Distributional: {dist_embs.shape[0]} points, {dist_embs.shape[1]}d")
    print(f"  PCA variance: {pca.explained_variance_ratio_[:5]}")
    print(f"Compressor:     {comp_embs.shape[0]} points, {comp_embs.shape[1]}d")
    print(f"  PCA variance: {pca2.explained_variance_ratio_[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--distributional", required=True)
    parser.add_argument("--out", default="results/embedding_comparison.png")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    print("Loading distributional embeddings...")
    dist_embs, dist_roles, dist_spans = load_distributional(args.distributional)
    print(f"  {len(dist_embs)} role embeddings")

    print("Loading compressor and computing bottleneck...")
    comp_embs, comp_roles = load_compressor_bottleneck(
        args.config, args.checkpoint, args.max_samples
    )
    print(f"  {len(comp_embs)} bottleneck slots")

    print("Plotting...")
    plot_comparison(dist_embs, dist_roles, comp_embs, comp_roles, args.out)


if __name__ == "__main__":
    main()
