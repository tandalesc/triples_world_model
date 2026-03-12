"""Reusable dynamics analysis tools: Jacobian eigenspectrum and flow fields.

Usage:
    from twm.analysis import dynamics_jacobian, flow_field
"""

import torch
import numpy as np
from torch.autograd.functional import jacobian


def dynamics_jacobian(model, input_ids, device="cpu"):
    """Compute Jacobian of the dynamics map at a given input state.

    Args:
        model: TripleWorldModel instance (in eval mode)
        input_ids: (1, T) tensor of token IDs
        device: device string

    Returns:
        eigenvalues: complex numpy array of eigenvalues
        jacobian_matrix: (T*d, T*d) numpy array
    """
    model.eval()
    input_ids = input_ids.to(device)

    # Encode to get latent
    with torch.no_grad():
        latent, _ = model.triple_encoder(input_ids)
        pad_mask = input_ids == 0

    latent_flat = latent.squeeze(0).reshape(-1).detach().clone().requires_grad_(True)
    shape = latent.shape[1:]  # (T, d_model)

    def dynamics_fn(flat_in):
        x = flat_in.reshape(1, *shape)
        out = model.dynamics(x, src_key_padding_mask=pad_mask)
        return out.reshape(-1)

    J = jacobian(dynamics_fn, latent_flat)
    J_np = J.detach().cpu().numpy()
    eigenvalues = np.linalg.eigvals(J_np)
    return eigenvalues, J_np


def flow_field(model, input_ids_batch, pca, device="cpu"):
    """Compute pre→post dynamics displacement in PCA space.

    Args:
        model: TripleWorldModel instance
        input_ids_batch: (N, T) tensor of token IDs
        pca: fitted sklearn PCA object (from latent → 3D)
        device: device string

    Returns:
        origins: (N, 3) PCA coords of pre-dynamics latents
        displacements: (N, 3) PCA displacement vectors (post - pre)
    """
    model.eval()
    input_ids_batch = input_ids_batch.to(device)

    with torch.no_grad():
        latent, _ = model.triple_encoder(input_ids_batch)
        pad_mask = input_ids_batch == 0
        pre = latent.clone()
        post = model.dynamics(latent, src_key_padding_mask=pad_mask)

    # Mean-pool excluding pad positions
    active = (~pad_mask).unsqueeze(-1).float()  # (B, T, 1)
    pre_pooled = (pre * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
    post_pooled = (post * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)

    pre_np = pre_pooled.cpu().numpy()
    post_np = post_pooled.cpu().numpy()

    origins = pca.transform(pre_np)
    endpoints = pca.transform(post_np)
    displacements = endpoints - origins

    return origins, displacements


def eigenspectrum_plot(eigenvalues, output_path=None):
    """Create a plotly scatter of eigenvalue magnitudes and phases.

    Args:
        eigenvalues: complex numpy array
        output_path: optional path to save HTML

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    mags = np.abs(eigenvalues)
    phases = np.angle(eigenvalues, deg=True)
    real = eigenvalues.real
    imag = eigenvalues.imag

    fig = go.Figure()

    # Complex plane view
    fig.add_trace(go.Scatter(
        x=real, y=imag,
        mode="markers",
        marker=dict(size=4, color=mags, colorscale="Viridis", colorbar=dict(title="|λ|")),
        text=[f"|λ|={m:.3f}, φ={p:.1f}°" for m, p in zip(mags, phases)],
        hovertemplate="Re: %{x:.3f}<br>Im: %{y:.3f}<br>%{text}<extra></extra>",
    ))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))

    fig.update_layout(
        title="Dynamics Jacobian Eigenspectrum",
        xaxis_title="Re(λ)", yaxis_title="Im(λ)",
        xaxis=dict(scaleanchor="y"), width=700, height=700,
    )

    if output_path:
        fig.write_html(str(output_path))
    return fig
