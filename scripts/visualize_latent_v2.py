"""Rich latent space visualization for TWM models.

Generates:
1. Semantic landscape — 6-panel view: mode coloring, length, norm density,
   PC1vPC3, KDE density contour, per-role PCA spectrum.
2. Role geometry — per-role density in shared PCA space.
3. Interpolation strips — smooth walks between example pairs.

Usage:
    uv run python scripts/visualize_latent_v2.py \
        --config configs/v38_balanced_joint.json \
        --checkpoint results/v38_balanced_joint/joint_all_phase2/model_best.pt \
        --out-dir results/v38_balanced_joint/visualizations_v2
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

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
    model.return_value = None
    model.eval()
    return model, tok, cfg


def get_data(model, dataset, n=1000, device="cpu"):
    n = min(n, len(dataset))
    input_ids = dataset._input_token_ids[:n].to(device)
    input_pad = dataset._input_pad_mask[:n].to(device)
    modes = dataset._modes[:n].numpy()
    pad_id = dataset.tokenizer.pad_token_id
    lengths = np.array([sum(1 for t in input_ids[i].tolist() if t != pad_id) for i in range(n)])
    with torch.no_grad():
        compress_out = model.compress(input_ids, input_pad)
        bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out
    return bottleneck.cpu(), modes, lengths, input_ids.cpu()


def plot_semantic_landscape(bottleneck, modes, lengths, max_triples, out_path):
    B, N3, D = bottleneck.shape
    N = max_triples
    flat = bottleneck.reshape(B, -1).numpy()
    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat)
    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Mode
    ax = fig.add_subplot(gs[0, 0])
    mc = np.where(modes == 0, 0.0, 1.0)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=mc, cmap="coolwarm", s=8, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Mode: Identity (blue) vs QA (red)", fontsize=13)
    cb = plt.colorbar(sc, ax=ax, ticks=[0, 1]); cb.set_ticklabels(["Identity", "QA"])

    # Panel 2: Length
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=lengths, cmap="viridis", s=8, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Input Text Length (tokens)", fontsize=13)
    plt.colorbar(sc, ax=ax, label="Tokens")

    # Panel 3: Norm
    bn_norms = bottleneck.norm(dim=-1).mean(dim=-1).numpy()
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=bn_norms, cmap="magma", s=8, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Bottleneck Norm (info density)", fontsize=13)
    plt.colorbar(sc, ax=ax, label="Mean L2")

    # Panel 4: PC1 vs PC3
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(proj[:, 0], proj[:, 2], c=mc, cmap="coolwarm", s=8, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC3")
    ax.set_title("PC1 vs PC3 (mode coloring)", fontsize=13)

    # Panel 5: Density
    ax = fig.add_subplot(gs[1, 1])
    try:
        kde = gaussian_kde(proj[:, :2].T)
        xg = np.linspace(proj[:, 0].min(), proj[:, 0].max(), 100)
        yg = np.linspace(proj[:, 1].min(), proj[:, 1].max(), 100)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contourf(X, Y, Z, levels=20, cmap="Blues")
        ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.5, alpha=0.5)
        ax.scatter(proj[:, 0], proj[:, 1], c="black", s=1, alpha=0.1)
    except Exception:
        ax.scatter(proj[:, 0], proj[:, 1], c="steelblue", s=5, alpha=0.3)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Density Landscape", fontsize=13)

    # Panel 6: Per-role variance
    ax = fig.add_subplot(gs[1, 2])
    bn_r = bottleneck[:, :N*3, :].reshape(B, N, 3, D)
    role_names = ["Entity", "Attribute", "Value"]
    role_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    x = np.arange(5)
    w = 0.25
    for r in range(3):
        rv = bn_r[:, :, r, :].reshape(-1, D).numpy()
        rp = PCA(n_components=min(10, D)).fit(rv)
        ax.bar(x + r * w, rp.explained_variance_ratio_[:5], w,
               label=role_names[r], color=role_colors[r], alpha=0.8)
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Explained")
    ax.set_title("Per-Role PCA Spectrum", fontsize=13)
    ax.set_xticks(x + w); ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
    ax.legend(fontsize=10)

    fig.suptitle(f"v38 Latent Space — PC1: {var[0]:.1%}, PC2: {var[1]:.1%}, PC3: {var[2]:.1%}",
                fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_role_geometry(bottleneck, max_triples, out_path):
    B, N3, D = bottleneck.shape
    N = max_triples
    bn = bottleneck[:, :N*3, :].reshape(B, N, 3, D)

    all_vecs = bottleneck[:, :N*3, :].reshape(-1, D).numpy()
    pca = PCA(n_components=2)
    pca.fit(all_vecs)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    names = ["Entity", "Attribute", "Value"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    cmaps = ["Reds", "Greens", "Blues"]

    for r, (ax, name, color, cmap) in enumerate(zip(axes, names, colors, cmaps)):
        rv = bn[:, :, r, :].reshape(-1, D).numpy()
        p = pca.transform(rv)
        try:
            kde = gaussian_kde(p.T)
            xg = np.linspace(p[:, 0].min(), p[:, 0].max(), 80)
            yg = np.linspace(p[:, 1].min(), p[:, 1].max(), 80)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.7)
        except Exception:
            pass
        ax.scatter(p[:, 0], p[:, 1], c=color, s=2, alpha=0.2)
        ax.set_xlabel("Shared PC1", fontsize=11)
        ax.set_ylabel("Shared PC2", fontsize=11)
        ax.set_title(f"{name} slots", fontsize=14, color=color, fontweight="bold")

    v = pca.explained_variance_ratio_
    fig.suptitle(f"Role Geometry (shared PCA: PC1 {v[0]:.1%}, PC2 {v[1]:.1%})",
                fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_interpolations(model, bottleneck, modes, tokenizer, out_path, n_steps=12):
    B, N3, D = bottleneck.shape
    id_idx = np.where(modes == 0)[0]
    qa_idx = np.where(modes == 1)[0]

    pairs = []
    if len(id_idx) >= 2:
        flat = bottleneck[id_idx].reshape(len(id_idx), -1)
        dists = torch.cdist(flat, flat)
        i, j = divmod(dists.argmax().item(), len(id_idx))
        pairs.append(("Identity A -> Identity B", id_idx[i], id_idx[j]))

    if len(qa_idx) >= 2:
        flat = bottleneck[qa_idx].reshape(len(qa_idx), -1)
        dists = torch.cdist(flat, flat)
        i, j = divmod(dists.argmax().item(), len(qa_idx))
        pairs.append(("QA A -> QA B", qa_idx[i], qa_idx[j]))

    if len(id_idx) > 0 and len(qa_idx) > 0:
        flat_id = bottleneck[id_idx].reshape(len(id_idx), -1)
        flat_qa = bottleneck[qa_idx].reshape(len(qa_idx), -1)
        dists = torch.cdist(flat_id, flat_qa)
        i, j = divmod(dists.argmin().item(), len(qa_idx))
        pairs.append(("Identity -> nearest QA", id_idx[i], qa_idx[j]))

    if not pairs:
        print("  No valid pairs")
        return

    fig, axes = plt.subplots(len(pairs), 1, figsize=(28, 5 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, (label, idx_a, idx_b) in zip(axes, pairs):
            bn_a = bottleneck[idx_a:idx_a+1]
            bn_b = bottleneck[idx_b:idx_b+1]
            texts = []
            alphas = np.linspace(0, 1, n_steps)

            for alpha in alphas:
                bn_interp = (1 - alpha) * bn_a + alpha * bn_b
                mode_ids = torch.zeros(1, dtype=torch.long)
                bn_dyn = model.forward_dynamics(bn_interp, mode_ids)
                gen_ids = model.generate(bn_dyn, n_steps=10)
                pred_len = model.forward_length(bn_dyn)
                pl = pred_len.round().long().clamp(1, gen_ids.shape[-1]).item()
                text = tokenizer.decode(gen_ids[0][:pl].cpu())
                texts.append(text)

            ax.set_xlim(0, n_steps)
            ax.set_ylim(0, 1)
            ax.set_title(label, fontsize=14, fontweight="bold")

            for i, (alpha, text) in enumerate(zip(alphas, texts)):
                color = plt.cm.coolwarm(alpha)
                ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, alpha=0.25))
                display = text[:60] + ("..." if len(text) > 60 else "")
                ax.text(i + 0.5, 0.5, display, ha="center", va="center",
                       fontsize=7, rotation=45,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

            ax.set_xticks([0.5, n_steps/2, n_steps - 0.5])
            ax.set_xticklabels(["Source", "Midpoint", "Target"], fontsize=11)
            ax.set_yticks([])

    fig.suptitle("Latent Space Interpolations", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-examples", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, tok, cfg = load_model(args.config, args.checkpoint)

    data_dir = Path(args.data_dir or cfg["data_dir"])
    print("Loading dataset...")
    dataset = TextPairDataset(
        data_dir / "qa_balanced_train.jsonl", tok,
        max_text_tokens=cfg["max_text_tokens"],
        max_examples=args.n_examples,
    )

    print("Computing bottlenecks...")
    bottleneck, modes, lengths, input_ids = get_data(model, dataset, n=args.n_examples)

    print("1/3 Semantic landscape...")
    plot_semantic_landscape(bottleneck, modes, lengths, cfg["max_triples"],
                          out_dir / "semantic_landscape.png")

    print("2/3 Role geometry...")
    plot_role_geometry(bottleneck, cfg["max_triples"], out_dir / "role_geometry.png")

    print("3/3 Interpolations...")
    plot_interpolations(model, bottleneck, modes, tok, out_dir / "interpolations.png")

    print("Done!")


if __name__ == "__main__":
    main()
