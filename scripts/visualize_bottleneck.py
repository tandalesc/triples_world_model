"""Visualize open-vocab bottleneck geometry and dynamics effect.

Plot 1: Compressor output PCA colored by mode (identity vs QA)
Plot 2: Compressor output PCA colored by relation type
Plot 3: Post-dynamics PCA comparing IO checkpoint vs mode_warmup checkpoint

Usage:
    uv run python scripts/visualize_bottleneck.py \
        --io-checkpoint results/v20_mini64/io_phase1/model_best.pt \
        --warmup-checkpoint results/v20_dyn/mode_warmup_phase1/model_best.pt \
        --config configs/v20_dynamics_only.json \
        --output results/v20_bottleneck_geometry
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twm.training_config import TrainingConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.text_pair_dataset import TextPairDataset


def build_model(config: TrainingConfig, tokenizer, device="cpu"):
    model_config = config.build_model_config()
    dyn_layers = config.dynamics_layers if config.dynamics_layers is not None else model_config.n_layers
    model = TextDynamicsModel(
        config=model_config, domain_tokenizer=tokenizer,
        text_compressor_layers=config.text_compressor_layers,
        text_expander_layers=config.text_expander_layers,
        dynamics_layers=dyn_layers,
        max_text_tokens=config.max_text_tokens,
        dropout=config.dropout, alpha_min=config.alpha_min,
    )
    model.init_embeddings()
    return model.to(device)


def load_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    loaded = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded.append(k.split(".")[0])
    model.load_state_dict(model_state)
    print(f"  Loaded from {path}: {sorted(set(loaded))}")


def extract_relation(text: str) -> str:
    """Extract relation type from QA text."""
    m = re.search(r'what (?:is the |are the )?(\w[\w\s]{0,20}?) (?:of|for|in)\b', text, re.I)
    if m:
        return m.group(1).strip().lower()
    if text.lower().startswith("where"):
        return "location"
    return "other"


@torch.no_grad()
def get_bottlenecks(model, dataset, device, n=500):
    """Compress inputs and return pooled bottleneck vectors + metadata."""
    n = min(n, len(dataset))
    input_ids = dataset._input_token_ids[:n].to(device)
    input_pad = dataset._input_pad_mask[:n].to(device)
    modes = dataset._modes[:n].numpy()

    bottleneck = model.compress(input_ids, input_pad)
    pooled = bottleneck.mean(dim=1).cpu().numpy()

    texts = [dataset.examples[i]["input_text"] for i in range(n)]
    relations = [extract_relation(t) for t in texts]

    return pooled, modes, texts, relations


@torch.no_grad()
def get_post_dynamics(model, dataset, device, mode_id, n=500):
    """Run dynamics with a specific mode and return pooled bottleneck."""
    n = min(n, len(dataset))
    input_ids = dataset._input_token_ids[:n].to(device)
    input_pad = dataset._input_pad_mask[:n].to(device)

    bottleneck = model.compress(input_ids, input_pad)
    modes = torch.full((n,), mode_id, dtype=torch.long, device=device)
    post = model.forward_dynamics(bottleneck, modes)

    return post.mean(dim=1).cpu().numpy()


def plot_pca_colored(ax, points_2d, labels, title):
    """Scatter plot with categorical coloring."""
    unique = sorted(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 0.9, max(len(unique), 2)))
    for i, lab in enumerate(unique):
        mask = np.array([l == lab for l in labels])
        ax.scatter(points_2d[mask, 0], points_2d[mask, 1],
                   c=[colors[i]], label=lab, s=15, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", markerscale=1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--io-checkpoint", required=True)
    parser.add_argument("--warmup-checkpoint", default=None)
    parser.add_argument("--dynamics-checkpoint", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/v20_bottleneck_geometry")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig.load(args.config)
    device = "cpu"
    tokenizer = DomainBPETokenizer.load(config.tokenizer_path, max_length=config.max_text_tokens)
    data_dir = Path(config.data_dir)

    qa_ds = TextPairDataset(data_dir / "qa_test.jsonl", tokenizer,
                            max_text_tokens=config.max_text_tokens)
    print(f"QA test: {len(qa_ds)} examples")

    # --- Plot 1 & 2: Compressor output geometry (IO checkpoint) ---
    print("\nLoading IO checkpoint...")
    model = build_model(config, tokenizer, device)
    load_checkpoint(model, args.io_checkpoint, device)
    model.eval()

    pooled, modes, texts, relations = get_bottlenecks(model, qa_ds, device, args.n)
    pca = PCA(n_components=2)
    pts = pca.fit_transform(pooled)
    var = pca.explained_variance_ratio_
    print(f"PCA variance explained: {var[0]:.1%} + {var[1]:.1%} = {sum(var):.1%}")

    mode_labels = ["identity" if m == 0 else "qa" for m in modes]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_pca_colored(axes[0], pts, mode_labels,
                     f"Compressor bottleneck by mode\n(PCA {sum(var):.0%} var)")

    rel_counts = Counter(relations)
    top_rels = {r for r, _ in rel_counts.most_common(8)}
    rel_labels = [r if r in top_rels else "other" for r in relations]
    plot_pca_colored(axes[1], pts, rel_labels,
                     f"Compressor bottleneck by relation type\n(PCA {sum(var):.0%} var)")

    fig.tight_layout()
    fig.savefig(out / "compressor_bottleneck_geometry.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {out / 'compressor_bottleneck_geometry.png'}")
    plt.close(fig)

    # --- Plot 3: Post-dynamics comparison (IO vs mode_warmup) ---
    if args.warmup_checkpoint:
        print("\nLoading mode_warmup checkpoint...")
        model_warmup = build_model(config, tokenizer, device)
        load_checkpoint(model_warmup, args.warmup_checkpoint, device)
        model_warmup.eval()

        n_sub = min(200, args.n)

        io_post_id = get_post_dynamics(model, qa_ds, device, mode_id=0, n=n_sub)
        io_post_qa = get_post_dynamics(model, qa_ds, device, mode_id=1, n=n_sub)

        wu_post_id = get_post_dynamics(model_warmup, qa_ds, device, mode_id=0, n=n_sub)
        wu_post_qa = get_post_dynamics(model_warmup, qa_ds, device, mode_id=1, n=n_sub)
        wu_post_rev = get_post_dynamics(model_warmup, qa_ds, device, mode_id=2, n=n_sub)

        all_post = np.concatenate([io_post_id, io_post_qa, wu_post_id, wu_post_qa, wu_post_rev])
        pca2 = PCA(n_components=2)
        all_2d = pca2.fit_transform(all_post)
        var2 = pca2.explained_variance_ratio_

        io_id_2d = all_2d[:n_sub]
        io_qa_2d = all_2d[n_sub:2*n_sub]
        wu_id_2d = all_2d[2*n_sub:3*n_sub]
        wu_qa_2d = all_2d[3*n_sub:4*n_sub]
        wu_rev_2d = all_2d[4*n_sub:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        ax.scatter(io_id_2d[:, 0], io_id_2d[:, 1], c="tab:blue", s=15, alpha=0.5, label="identity")
        ax.scatter(io_qa_2d[:, 0], io_qa_2d[:, 1], c="tab:orange", s=15, alpha=0.5, label="qa")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"IO checkpoint: post-dynamics by mode\n(PCA {sum(var2):.0%} var)")
        ax.legend()

        ax = axes[1]
        ax.scatter(wu_id_2d[:, 0], wu_id_2d[:, 1], c="tab:blue", s=15, alpha=0.5, label="identity")
        ax.scatter(wu_qa_2d[:, 0], wu_qa_2d[:, 1], c="tab:orange", s=15, alpha=0.5, label="qa")
        ax.scatter(wu_rev_2d[:, 0], wu_rev_2d[:, 1], c="tab:green", s=15, alpha=0.5, label="reverse")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Mode warmup checkpoint: post-dynamics by mode\n(PCA {sum(var2):.0%} var)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(out / "post_dynamics_io_vs_warmup.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {out / 'post_dynamics_io_vs_warmup.png'}")
        plt.close(fig)

    # --- Optional: dynamics checkpoint ---
    if args.dynamics_checkpoint and args.warmup_checkpoint:
        print("\nLoading dynamics checkpoint...")
        model_dyn = build_model(config, tokenizer, device)
        load_checkpoint(model_dyn, args.dynamics_checkpoint, device)
        model_dyn.eval()

        n_sub = min(200, args.n)
        dyn_post_id = get_post_dynamics(model_dyn, qa_ds, device, mode_id=0, n=n_sub)
        dyn_post_qa = get_post_dynamics(model_dyn, qa_ds, device, mode_id=1, n=n_sub)

        all_dyn = np.concatenate([wu_post_id, wu_post_qa, dyn_post_id, dyn_post_qa])
        pca3 = PCA(n_components=2)
        all_3d = pca3.fit_transform(all_dyn)
        var3 = pca3.explained_variance_ratio_

        wu_id_3 = all_3d[:n_sub]
        wu_qa_3 = all_3d[n_sub:2*n_sub]
        dyn_id_3 = all_3d[2*n_sub:3*n_sub]
        dyn_qa_3 = all_3d[3*n_sub:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        ax.scatter(wu_id_3[:, 0], wu_id_3[:, 1], c="tab:blue", s=15, alpha=0.5, label="identity")
        ax.scatter(wu_qa_3[:, 0], wu_qa_3[:, 1], c="tab:orange", s=15, alpha=0.5, label="qa")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Mode warmup: post-dynamics\n(PCA {sum(var3):.0%} var)")
        ax.legend()

        ax = axes[1]
        ax.scatter(dyn_id_3[:, 0], dyn_id_3[:, 1], c="tab:blue", s=15, alpha=0.5, label="identity")
        ax.scatter(dyn_qa_3[:, 0], dyn_qa_3[:, 1], c="tab:orange", s=15, alpha=0.5, label="qa")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Dynamics checkpoint: post-dynamics\n(PCA {sum(var3):.0%} var)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(out / "post_dynamics_warmup_vs_dynamics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {out / 'post_dynamics_warmup_vs_dynamics.png'}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
