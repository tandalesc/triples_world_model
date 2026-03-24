#!/usr/bin/env python3
"""3D interactive visualization of dynamics trajectories in bottleneck space.

Shows how states flow through the latent space during multi-step dynamics.
Each trajectory is a line from compress(input) through N dynamics steps.

Usage:
  uv run python scripts/visualize_dynamics_3d.py --run-dir results/glucose_chain_v4 \
    --data data/glucose/augmented_chain_general_test.jsonl --n-examples 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.domain_bpe import DomainBPETokenizer
from twm.text_dynamics_model import TextDynamicsModel
from twm.chain_dataset import ChainDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--extra-steps", type=int, default=4,
                        help="Extra dynamics steps beyond chain length (explore past training)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextDynamicsModel.load(run_dir, device=str(device))
    model.eval()

    tokenizer = model.tokenizer
    ds = ChainDataset(args.data, tokenizer, max_text_tokens=model.max_text_tokens)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # Collect trajectories: each is a list of bottleneck snapshots
    trajectories = []  # list of (mode, [bn_step0, bn_step1, ...])
    mode_names = {0: "advance", 1: "query", 2: "identity"}

    n = 0
    max_steps = 3 + args.extra_steps  # chain steps + extra unrolling

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            input_pad = batch["input_pad"].to(device)
            mode_ids = batch["mode_id"].to(device)

            bn = model.compress(input_ids, input_pad)
            if isinstance(bn, tuple):
                bn = bn[0]

            B = bn.shape[0]
            # Mean-pool across triple positions
            snapshots = [bn.mean(dim=1).cpu().numpy()]  # step 0

            for step in range(max_steps):
                bn = model.forward_dynamics(bn, mode_ids)
                snapshots.append(bn.mean(dim=1).cpu().numpy())

            # Store per-example trajectories
            for i in range(B):
                traj = [s[i] for s in snapshots]
                trajectories.append((mode_ids[i].item(), traj))

            n += B
            if n >= args.n_examples:
                break

    trajectories = trajectories[:args.n_examples]

    # PCA on all points
    all_points = []
    for mode, traj in trajectories:
        all_points.extend(traj)
    all_points = np.array(all_points)

    pca = PCA(n_components=3)
    pca.fit(all_points)
    var = pca.explained_variance_ratio_

    # Transform trajectories to 3D
    trajs_3d = []
    for mode, traj in trajectories:
        traj_3d = pca.transform(np.array(traj))
        trajs_3d.append((mode, traj_3d))

    # Build plotly figure
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = {0: "rgb(231,76,60)", 1: "rgb(52,152,219)", 2: "rgb(46,204,113)"}
    color_names = {0: "advance", 1: "query", 2: "identity"}

    # Add trajectories as lines
    for mode, traj_3d in trajs_3d:
        fig.add_trace(go.Scatter3d(
            x=traj_3d[:, 0], y=traj_3d[:, 1], z=traj_3d[:, 2],
            mode="lines",
            line=dict(color=colors[mode], width=2),
            opacity=0.3,
            name=color_names[mode],
            showlegend=False,
        ))

    # Add start points (larger, opaque)
    for mode in [0, 1, 2]:
        starts = np.array([t[0] for m, t in trajs_3d if m == mode])
        if len(starts) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=starts[:, 0], y=starts[:, 1], z=starts[:, 2],
            mode="markers",
            marker=dict(size=4, color=colors[mode], opacity=0.8),
            name=f"{color_names[mode]} (start)",
        ))

    # Add end points (diamonds, semi-transparent)
    for mode in [0, 1, 2]:
        ends = np.array([t[-1] for m, t in trajs_3d if m == mode])
        if len(ends) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=ends[:, 0], y=ends[:, 1], z=ends[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors[mode], opacity=0.4, symbol="diamond"),
            name=f"{color_names[mode]} (end, +{args.extra_steps} steps)",
        ))

    # Add arrows for a subset (too many clutters)
    arrow_step = max(1, len(trajs_3d) // 50)
    for i, (mode, traj_3d) in enumerate(trajs_3d):
        if i % arrow_step != 0:
            continue
        # Arrow from second-to-last to last point
        p1 = traj_3d[-2]
        p2 = traj_3d[-1]
        fig.add_trace(go.Cone(
            x=[p2[0]], y=[p2[1]], z=[p2[2]],
            u=[p2[0]-p1[0]], v=[p2[1]-p1[1]], w=[p2[2]-p1[2]],
            colorscale=[[0, colors[mode]], [1, colors[mode]]],
            showscale=False,
            sizemode="absolute",
            sizeref=0.5,
            opacity=0.6,
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Dynamics Trajectories in Bottleneck Space — {run_dir.name}<br>"
              f"<sub>PCA: {var[0]:.1%} / {var[1]:.1%} / {var[2]:.1%} variance. "
              f"{len(trajectories)} examples, {max_steps+1} steps each "
              f"({max_steps-2} past training)</sub>",
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1%})",
            yaxis_title=f"PC2 ({var[1]:.1%})",
            zaxis_title=f"PC3 ({var[2]:.1%})",
        ),
        width=1200,
        height=800,
        template="plotly_dark",
    )

    out_path = args.output or str(run_dir / "dynamics_3d.html")
    fig.write_html(out_path)
    print(f"Saved: {out_path}")

    # Also save a static image if kaleido is available
    try:
        fig.write_image(str(run_dir / "dynamics_3d.png"), width=1200, height=800)
        print(f"Saved: {run_dir / 'dynamics_3d.png'}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
