#!/usr/bin/env python3
"""Plot training curves and comparison charts for TWM model family benchmark.

Usage:
  uv run python scripts/plot_family.py --results-dir results/family_benchmark
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

VARIANTS = {
    "twm_base": {"label": "Base (256d, GloVe)", "color": "#2196F3", "ls": "-"},
    "twm_base_split": {"label": "Base Split", "color": "#1565C0", "ls": "--"},
    "twm_mini": {"label": "Mini (32d)", "color": "#E91E63", "ls": "-"},
    "twm_micro": {"label": "Micro (16d)", "color": "#FF9800", "ls": "-"},
    "twm_micro_split": {"label": "Micro Split", "color": "#E65100", "ls": "--"},
    "twm_micro_split_qat": {"label": "Micro Split+QAT", "color": "#9C27B0", "ls": ":"},
    "twm_micro_qat": {"label": "Micro QAT (shared)", "color": "#4CAF50", "ls": "-."},
}

SPLITS = ["train", "test_comp", "test_seen", "test_context"]
SPLIT_LABELS = {
    "train": "Train",
    "test_comp": "Comp. Gen.",
    "test_seen": "Seen Combos",
    "test_context": "Context-Dep.",
}


def load_train_log(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_training_curves(results_dir: Path, out_path: Path):
    """Plot F1 over epochs for all variants, one subplot per test split."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("TWM Model Family — Training Curves", fontsize=14, fontweight="bold")

    for ax, split in zip(axes.flat, SPLITS):
        ax.set_title(SPLIT_LABELS[split], fontsize=12)
        ax.set_ylabel("F1")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        for variant_name, style in VARIANTS.items():
            log_path = results_dir / variant_name / "train_log.jsonl"
            if not log_path.exists():
                continue
            rows = load_train_log(log_path)
            epochs = [r["epoch"] for r in rows]
            f1_key = f"{split}/f1"
            f1s = [r.get(f1_key, 0) for r in rows]
            if any(v > 0 for v in f1s):
                ax.plot(epochs, f1s, label=style["label"],
                        color=style["color"], linestyle=style["ls"], linewidth=2)

        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_final_comparison(results_dir: Path, out_path: Path):
    """Bar chart comparing final F1 across variants and splits."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("TWM Model Family — Final F1 Comparison", fontsize=14, fontweight="bold")

    bar_width = 0.12
    x_positions = list(range(len(SPLITS)))

    variant_names = []
    for i, (variant_name, style) in enumerate(VARIANTS.items()):
        log_path = results_dir / variant_name / "train_log.jsonl"
        if not log_path.exists():
            continue
        rows = load_train_log(log_path)
        if not rows:
            continue
        final = rows[-1]
        f1s = [final.get(f"{split}/f1", 0) for split in SPLITS]

        config_path = results_dir / variant_name / "config.json"
        config = load_config(config_path) if config_path.exists() else {}
        param_count = "?"

        # Try to compute param count from config
        label = style["label"]

        offsets = [x + i * bar_width for x in x_positions]
        bars = ax.bar(offsets, f1s, bar_width, label=label,
                      color=style["color"], alpha=0.85, edgecolor="white")

        # Add value labels on bars
        for bar, val in zip(bars, f1s):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        variant_names.append(variant_name)

    ax.set_xticks([x + bar_width * (len(variant_names) - 1) / 2 for x in x_positions])
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 1.15),
              ncol=3, framealpha=0.9, edgecolor="gray")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_efficiency(results_dir: Path, out_path: Path):
    """Scatter plot: param count vs F1, showing efficiency frontier."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("TWM Model Family — Size vs. Accuracy Tradeoff", fontsize=14, fontweight="bold")

    for ax, split in zip(axes, ["test_comp", "test_seen", "test_context"]):
        ax.set_title(SPLIT_LABELS[split], fontsize=12)
        ax.set_xlabel("Parameters (log scale)")
        ax.set_ylabel("F1")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Collect all data points first for smart label placement
        points = []
        for variant_name, style in VARIANTS.items():
            log_path = results_dir / variant_name / "train_log.jsonl"
            config_path = results_dir / variant_name / "config.json"
            if not log_path.exists():
                continue

            rows = load_train_log(log_path)
            if not rows:
                continue
            final = rows[-1]
            f1 = final.get(f"{split}/f1", 0)

            from twm.config import ModelConfig
            from twm.model import TripleWorldModel
            mc = ModelConfig.load(config_path)
            model = TripleWorldModel(mc)
            params = model.param_count()

            points.append((params, f1, variant_name, style))

        # Sort by f1 within each param cluster for label offset stacking
        micro_pts = sorted([p for p in points if p[0] < 100_000], key=lambda p: -p[1])
        mid_pts = sorted([p for p in points if 100_000 <= p[0] < 500_000], key=lambda p: -p[1])
        base_pts = sorted([p for p in points if p[0] >= 500_000], key=lambda p: -p[1])

        # Assign vertical offsets to avoid overlap within clusters
        def plot_cluster(cluster, base_offset_x, offsets_y):
            for i, (params, f1, vname, style) in enumerate(cluster):
                ax.scatter(params, f1, s=150, color=style["color"],
                           edgecolors="black", linewidths=0.8, zorder=5,
                           marker="o")
                oy = offsets_y[i] if i < len(offsets_y) else offsets_y[-1]
                ax.annotate(style["label"], (params, f1),
                            textcoords="offset points", xytext=(base_offset_x, oy),
                            fontsize=8.5, fontweight="bold", color=style["color"],
                            arrowprops=dict(arrowstyle="-", color=style["color"],
                                            alpha=0.4, lw=0.8))

        # Micro cluster: spread labels to the right, stacked vertically
        plot_cluster(micro_pts, 12, [28, 14, 0, -14, -28, -42])
        # Mini cluster: labels above
        plot_cluster(mid_pts, 12, [18, -18])
        # Base cluster: spread labels to the left
        plot_cluster(base_pts, -12, [18, -18])

        # Zoom to data range with padding
        all_f1 = [p[1] for p in points]
        if all_f1:
            f1_min = min(all_f1)
            f1_max = max(all_f1)
            margin = (f1_max - f1_min) * 0.25
            ax.set_ylim(max(0, f1_min - margin), min(1.05, f1_max + margin))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary_table(results_dir: Path):
    """Print markdown summary table."""
    print("\n## Model Family Benchmark Results\n")
    print(f"| Model | Params | KB (f32) | Comp Gen F1 | Seen F1 | Context F1 | Train F1 |")
    print(f"|-------|-------:|---------:|:-----------:|:-------:|:----------:|:--------:|")

    for variant_name, style in VARIANTS.items():
        log_path = results_dir / variant_name / "train_log.jsonl"
        config_path = results_dir / variant_name / "config.json"
        if not log_path.exists():
            continue

        rows = load_train_log(log_path)
        if not rows:
            continue
        final = rows[-1]

        from twm.config import ModelConfig
        from twm.model import TripleWorldModel
        mc = ModelConfig.load(config_path)
        model = TripleWorldModel(mc)
        params = model.param_count()
        mem_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024

        comp = final.get("test_comp/f1", 0)
        seen = final.get("test_seen/f1", 0)
        ctx = final.get("test_context/f1", 0)
        train_f1 = final.get("train/f1", 0)

        print(f"| {style['label']} | {params:,} | {mem_kb:.1f} | {comp:.3f} | {seen:.3f} | {ctx:.3f} | {train_f1:.3f} |")


def main():
    parser = argparse.ArgumentParser(description="Plot TWM family benchmark results")
    parser.add_argument("--results-dir", default="results/family_benchmark")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(results_dir, out_dir / "training_curves.png")
    plot_final_comparison(results_dir, out_dir / "final_comparison.png")
    plot_efficiency(results_dir, out_dir / "efficiency.png")
    print_summary_table(results_dir)


if __name__ == "__main__":
    main()
