"""Visualize TWM dynamics latent space.

Pet sim scatter plot with pre/post dynamics flow, eigenspectrum analysis,
and flow field visualization.

Usage:
    uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim --output results/pet_sim/latent_space.html
    uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim --eigenspectrum
    uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim --flow-field
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from twm.serve import WorldModel
from twm.dataset import _sort_triples, _pad_triples, _flatten_triples


# Pet sim state space
PETS = ["Daisy", "Luna", "Buddy", "Max", "Rocky"]
HUNGER_VALS = ["starving", "hungry", "content", "full"]
ENERGY_VALS = ["exhausted", "tired", "rested"]
MOOD_VALS = ["sad", "content", "happy"]
CLEANLINESS_VALS = ["dirty", "messy", "clean"]
ACTIONS = ["ignore", "pet", "play", "feed", "nap", "walk", "bathe"]


def enumerate_pet_states():
    """Generate all valid pet input states (mode + 4 attrs + action)."""
    states = []
    labels = []
    for pet in PETS:
        combos = itertools.product(HUNGER_VALS, ENERGY_VALS, MOOD_VALS, CLEANLINESS_VALS, ACTIONS)
        for hunger, energy, mood, clean, action in combos:
            triples = [
                ["#mode", "type", "advance"],
                [pet, "hunger", hunger],
                [pet, "energy", energy],
                [pet, "mood", mood],
                [pet, "cleanliness", clean],
                [pet, "action", action],
            ]
            states.append(triples)
            labels.append({
                "pet": pet,
                "hunger": hunger,
                "energy": energy,
                "mood": mood,
                "cleanliness": clean,
                "action": action,
            })
    return states, labels


def encode_states(wm, states):
    """Encode all states to input_ids tensor."""
    all_ids = []
    for state in states:
        sorted_state = _sort_triples(state)
        padded = _pad_triples(sorted_state, wm.config.max_triples)
        ids = _flatten_triples(padded, wm.vocab)
        all_ids.append(ids)
    return torch.tensor(all_ids, dtype=torch.long, device=wm.device)


def extract_latents(wm, input_ids, batch_size=256):
    """Extract pre and post dynamics latents, mean-pooled."""
    model = wm.model
    model.eval()

    pre_all, post_all = [], []
    n = input_ids.shape[0]

    for i in range(0, n, batch_size):
        batch = input_ids[i : i + batch_size]
        with torch.no_grad():
            latent, _ = model.triple_encoder(batch)
            pad_mask = batch == 0
            pre = latent.clone()
            post = model.dynamics(latent, src_key_padding_mask=pad_mask)

        # Mean-pool excluding pad positions
        active = (~pad_mask).unsqueeze(-1).float()
        pre_pooled = (pre * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
        post_pooled = (post * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)

        pre_all.append(pre_pooled.cpu().numpy())
        post_all.append(post_pooled.cpu().numpy())

    return np.concatenate(pre_all), np.concatenate(post_all)


def build_scatter_plot(pre_3d, post_3d, labels, color_by="pet"):
    """Build 3D scatter with flow arrows."""
    import plotly.graph_objects as go
    import plotly.colors as pc

    color_key = color_by
    unique_vals = sorted(set(l[color_key] for l in labels))
    palette = pc.qualitative.Set1[:len(unique_vals)]
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(unique_vals)}

    hover_texts = []
    for l in labels:
        text = "<br>".join(f"{k}: {v}" for k, v in l.items())
        hover_texts.append(text)

    fig = go.Figure()

    # Group by color_by value for legend
    for val in unique_vals:
        idx = [i for i, l in enumerate(labels) if l[color_key] == val]
        color = color_map[val]

        fig.add_trace(go.Scatter3d(
            x=pre_3d[idx, 0], y=pre_3d[idx, 1], z=pre_3d[idx, 2],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.6),
            text=[hover_texts[i] for i in idx],
            hovertemplate="%{text}<extra>pre " + val + "</extra>",
            name=f"{val} (pre)",
            legendgroup=val,
        ))

        fig.add_trace(go.Scatter3d(
            x=post_3d[idx, 0], y=post_3d[idx, 1], z=post_3d[idx, 2],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.6, symbol="diamond"),
            text=[hover_texts[i] for i in idx],
            hovertemplate="%{text}<extra>post " + val + "</extra>",
            name=f"{val} (post)",
            legendgroup=val,
            showlegend=False,
        ))

    # Flow lines (subsample for performance)
    n = len(pre_3d)
    step = max(1, n // 500)
    for i in range(0, n, step):
        fig.add_trace(go.Scatter3d(
            x=[pre_3d[i, 0], post_3d[i, 0]],
            y=[pre_3d[i, 1], post_3d[i, 1]],
            z=[pre_3d[i, 2], post_3d[i, 2]],
            mode="lines",
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"TWM Dynamics Latent Space (colored by {color_by})",
        scene=dict(
            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
        ),
        width=1000, height=800,
    )

    return fig


def run_scatter(wm, output_path, color_by="pet"):
    """Main scatter plot pipeline."""
    print("Enumerating pet states...")
    states, labels = enumerate_pet_states()
    print(f"  {len(states)} states")

    print("Encoding states...")
    input_ids = encode_states(wm, states)

    print("Extracting latents...")
    pre_latents, post_latents = extract_latents(wm, input_ids)

    print("Fitting PCA...")
    combined = np.concatenate([pre_latents, post_latents])
    pca = PCA(n_components=3)
    pca.fit(combined)
    pre_3d = pca.transform(pre_latents)
    post_3d = pca.transform(post_latents)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print("Building plot...")
    fig = build_scatter_plot(pre_3d, post_3d, labels, color_by=color_by)
    fig.write_html(str(output_path))
    print(f"  Saved to {output_path}")

    return pca  # for reuse in flow field


def run_eigenspectrum(wm, output_path):
    """Compute and plot Jacobian eigenspectrum for a sample state."""
    from twm.analysis import dynamics_jacobian, eigenspectrum_plot

    # Use a representative state
    sample_state = [
        ["#mode", "type", "advance"],
        ["Daisy", "hunger", "hungry"],
        ["Daisy", "energy", "tired"],
        ["Daisy", "mood", "content"],
        ["Daisy", "cleanliness", "messy"],
        ["Daisy", "action", "feed"],
    ]
    sorted_state = _sort_triples(sample_state)
    padded = _pad_triples(sorted_state, wm.config.max_triples)
    ids = _flatten_triples(padded, wm.vocab)
    input_ids = torch.tensor([ids], dtype=torch.long)

    print("Computing Jacobian (this may take a moment)...")
    eigenvalues, J = dynamics_jacobian(wm.model, input_ids, device=str(wm.device))

    mags = np.abs(eigenvalues)
    print(f"  Eigenvalue magnitude range: [{mags.min():.4f}, {mags.max():.4f}]")
    print(f"  Mean magnitude: {mags.mean():.4f}")
    print(f"  # with |lambda| > 1 (expansive): {(mags > 1).sum()}")
    print(f"  # with |lambda| < 1 (contractive): {(mags < 1).sum()}")

    eigen_path = output_path.parent / "eigenspectrum.html"
    eigenspectrum_plot(eigenvalues, output_path=eigen_path)
    print(f"  Saved to {eigen_path}")


def run_flow_field(wm, output_path):
    """Visualize flow field in PCA space."""
    from twm.analysis import flow_field
    import plotly.graph_objects as go

    print("Enumerating states for flow field...")
    states, labels = enumerate_pet_states()
    input_ids = encode_states(wm, states)

    # Fit PCA on pre-dynamics latents
    pre_latents, post_latents = extract_latents(wm, input_ids)
    combined = np.concatenate([pre_latents, post_latents])
    pca = PCA(n_components=3)
    pca.fit(combined)

    print("Computing flow field...")
    origins, displacements = flow_field(wm.model, input_ids, pca, device=str(wm.device))

    # 2D projection (PC1 vs PC2) with quiver-like arrows
    fig = go.Figure()

    # Subsample for readability
    n = len(origins)
    step = max(1, n // 300)
    idx = list(range(0, n, step))

    import plotly.colors as pc
    palette = pc.qualitative.Set1[:len(PETS)]
    pet_colors = {p: palette[i] for i, p in enumerate(PETS)}

    for pet in PETS:
        pet_idx = [j for j in idx if labels[j]["pet"] == pet]
        if not pet_idx:
            continue
        fig.add_trace(go.Scatter(
            x=origins[pet_idx, 0], y=origins[pet_idx, 1],
            mode="markers",
            marker=dict(size=4, color=pet_colors[pet]),
            text=[f"{labels[i]['pet']} {labels[i]['action']}" for i in pet_idx],
            hovertemplate="%{text}<extra></extra>",
            name=pet,
        ))

    # Arrows as annotations
    scale = 1.0
    for j in idx:
        fig.add_annotation(
            x=origins[j, 0] + displacements[j, 0] * scale,
            y=origins[j, 1] + displacements[j, 1] * scale,
            ax=origins[j, 0], ay=origins[j, 1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=0.8, arrowwidth=1,
            arrowcolor="rgba(100,100,100,0.4)",
        )

    fig.update_layout(
        title="Dynamics Flow Field (PC1 vs PC2)",
        xaxis_title="PC1", yaxis_title="PC2",
        width=900, height=700,
    )

    flow_path = output_path.parent / "flow_field.html"
    fig.write_html(str(flow_path))
    print(f"  Saved to {flow_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TWM dynamics latent space")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to run directory")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--color-by", type=str, default="pet",
                        choices=["pet", "action", "hunger", "energy", "mood", "cleanliness"],
                        help="Color scatter points by this attribute")
    parser.add_argument("--eigenspectrum", action="store_true", help="Compute Jacobian eigenspectrum")
    parser.add_argument("--flow-field", action="store_true", help="Visualize dynamics flow field")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    ckpt_dir = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else ckpt_dir / "latent_space.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_dir}...")
    wm = WorldModel(ckpt_dir, device=args.device)
    print(f"  {wm.model.param_count():,} params on {wm.device}")

    if args.eigenspectrum:
        run_eigenspectrum(wm, output_path)
    elif args.flow_field:
        run_flow_field(wm, output_path)
    else:
        run_scatter(wm, output_path, color_by=args.color_by)


if __name__ == "__main__":
    main()
