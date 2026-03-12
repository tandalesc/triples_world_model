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
import json
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


def _is_triple_list(x):
    return isinstance(x, list) and all(isinstance(t, list) and len(t) == 3 for t in x)


def load_states_file(path: Path):
    """Load custom states from JSON/JSONL.

    Accepted line/object formats:
      - [[e,a,v], ...]
      - {"state_t": [[e,a,v], ...]}
    """
    states = []
    labels = []

    if path.suffix.lower() == ".jsonl":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
        objs = obj if isinstance(obj, list) else [obj]

    for i, obj in enumerate(objs):
        if _is_triple_list(obj):
            st = obj
        elif isinstance(obj, dict) and _is_triple_list(obj.get("state_t")):
            st = obj["state_t"]
        else:
            continue

        st = _sort_triples(st)
        states.append(st)

        d = {(e, a): v for e, a, v in st}
        label = {
            "index": str(i),
            "mode": d.get(("#mode", "state"), d.get(("#mode", "type"), "unknown")),
            "user": d.get(("user", "state"), "n/a"),
            "task": d.get(("task", "state"), "n/a"),
            "energy": d.get(("energy", "state"), "n/a"),
            "focus": d.get(("focus", "state"), "n/a"),
            "calendar": d.get(("calendar", "state"), "n/a"),
            "urgency": d.get(("urgency", "state"), "n/a"),
        }
        labels.append(label)

    if not states:
        raise ValueError(f"No valid states found in {path}")
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

    color_key = color_by if color_by in labels[0] else "index"
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


def run_scatter(wm, output_path, color_by="pet", states=None, labels=None):
    """Main scatter plot pipeline."""
    if states is None or labels is None:
        print("Enumerating pet states...")
        states, labels = enumerate_pet_states()
    else:
        print("Using custom states...")
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


def _auto_sample_state_from_vocab(wm):
    toks = [t for t in wm.vocab.token2id.keys() if t != "<pad>"]
    rel_pref = ["state", "type", "relation", "attr", "action"]
    rel = next((r for r in rel_pref if r in wm.vocab.token2id), toks[0])

    mode_vals = [m for m in ["advance", "identity", "query", "solve"] if m in wm.vocab.token2id]
    mode_v = mode_vals[0] if mode_vals else toks[min(1, len(toks)-1)]

    entities = [t for t in toks if t not in {"#mode", rel, mode_v}][: max(2, wm.config.max_triples - 1)]
    values = [t for t in toks if t not in {"#mode", rel, *entities}]
    if not values:
        values = entities

    sample = [["#mode", rel, mode_v]]
    for i, e in enumerate(entities[: wm.config.max_triples - 1]):
        sample.append([e, rel, values[i % len(values)]])
    return sample


def run_eigenspectrum(wm, output_path, sample_state=None):
    """Compute and plot Jacobian eigenspectrum for a sample state."""
    from twm.analysis import dynamics_jacobian, eigenspectrum_plot

    if sample_state is None:
        sample_state = _auto_sample_state_from_vocab(wm)
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
    try:
        eigenspectrum_plot(eigenvalues, output_path=eigen_path)
        print(f"  Saved to {eigen_path}")
    except ModuleNotFoundError:
        fallback = output_path.parent / "eigenspectrum_values.json"
        with fallback.open("w", encoding="utf-8") as f:
            json.dump({
                "real": [float(x.real) for x in eigenvalues],
                "imag": [float(x.imag) for x in eigenvalues],
            }, f)
        print(f"  plotly not installed; wrote eigenvalues to {fallback}")


def run_flow_field(wm, output_path, states=None, labels=None):
    """Visualize flow field in PCA space."""
    from twm.analysis import flow_field
    import plotly.graph_objects as go

    if states is None or labels is None:
        print("Enumerating states for flow field...")
        states, labels = enumerate_pet_states()
    else:
        print("Using custom states for flow field...")
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
    group_key = "pet" if "pet" in labels[0] else ("mode" if "mode" in labels[0] else "index")
    groups = sorted(set(l.get(group_key, "unknown") for l in labels))
    palette = pc.qualitative.Set1
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    for g in groups:
        g_idx = [j for j in idx if labels[j].get(group_key, "unknown") == g]
        if not g_idx:
            continue
        fig.add_trace(go.Scatter(
            x=origins[g_idx, 0], y=origins[g_idx, 1],
            mode="markers",
            marker=dict(size=4, color=group_colors[g]),
            text=[" | ".join(f"{k}:{v}" for k, v in labels[i].items()) for i in g_idx],
            hovertemplate="%{text}<extra></extra>",
            name=str(g),
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
                        help="Color scatter points by this attribute (default: pet)")
    parser.add_argument("--states-file", type=str, default=None,
                        help="Optional JSON/JSONL file with custom states for scatter/flow")
    parser.add_argument("--sample-state-json", type=str, default=None,
                        help="Optional JSON triple-list for eigenspectrum sample state")
    parser.add_argument("--sample-state-file", type=str, default=None,
                        help="Optional file containing sample state JSON triple-list")
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

    states = labels = None
    if args.states_file:
        states, labels = load_states_file(Path(args.states_file))

    sample_state = None
    if args.sample_state_json:
        sample_state = json.loads(args.sample_state_json)
    elif args.sample_state_file:
        sample_state = json.loads(Path(args.sample_state_file).read_text(encoding="utf-8"))

    if args.eigenspectrum:
        if sample_state is None and states:
            sample_state = states[0]
        run_eigenspectrum(wm, output_path, sample_state=sample_state)
    elif args.flow_field:
        run_flow_field(wm, output_path, states=states, labels=labels)
    else:
        run_scatter(wm, output_path, color_by=args.color_by, states=states, labels=labels)


if __name__ == "__main__":
    main()
