"""Unified loss computation for diffusion-based text models.

Supports both IO (identity reconstruction) and dynamics (transformation) modes
via a single `compute_diffusion_loss` function.
"""

import torch
import torch.nn.functional as F

from .diffusion_decoder import importance_sample_timesteps


def sample_timestep(B: int, device: torch.device, t_min: float, t_max: float,
                    bias_power: float = 1.0) -> torch.Tensor:
    """Sample diffusion timesteps with optional importance sampling."""
    if t_min == t_max:
        return torch.full((B,), t_min, device=device)
    if t_min == 0.0 and t_max == 1.0 and bias_power != 1.0:
        return importance_sample_timesteps(B, device, bias_power)
    u = torch.rand(B, device=device)
    return t_min + (t_max - t_min) * u


def _clean(s: str) -> str:
    """Clean BPE token artifacts for display."""
    return s.replace("\u0120", " ").replace("\u010a", "\n").replace("\u00e2\u0122\u0135", "-").strip()


def _compute_spectral_loss(bottleneck, n_roles=3):
    """Penalize bottleneck covariance collapse per role.

    For each role (E, A, V), computes the covariance of bottleneck vectors
    across the batch, then penalizes when the top eigenvalue dominates
    (PC1 explained variance ratio → 1.0 means 1D collapse).

    Loss = mean over roles of (λ_max / Σλ), which is 1/d when perfectly
    spread and approaches 1.0 when collapsed to 1D.

    Args:
        bottleneck: (B, N*3, d_model)

    Returns:
        scalar loss, metrics dict with per-role PC1 ratios
    """
    B, T, d = bottleneck.shape
    device = bottleneck.device

    role_idx = torch.arange(T, device=device) % n_roles
    role_names = ["entity", "attribute", "value"]
    total = torch.tensor(0.0, device=device)
    metrics = {}

    for r in range(n_roles):
        mask = role_idx == r
        # (B, n_slots_per_role, d) → (B * n_slots_per_role, d)
        vecs = bottleneck[:, mask].reshape(-1, d)
        if vecs.shape[0] < 2:
            continue
        # Center
        vecs = vecs - vecs.mean(dim=0, keepdim=True)
        # Covariance: (d, d)
        cov = (vecs.T @ vecs) / (vecs.shape[0] - 1)
        # Eigenvalues (symmetric → real)
        eigvals = torch.linalg.eigvalsh(cov)  # ascending order
        eigvals = eigvals.clamp(min=1e-8)
        # PC1 ratio = max eigenvalue / sum
        pc1_ratio = eigvals[-1] / eigvals.sum()
        total = total + pc1_ratio
        metrics[f"spec_{role_names[r]}"] = pc1_ratio.item()

    total = total / n_roles
    metrics["spec_loss"] = total.item()
    return total, metrics


def _compute_role_prior_loss(bottleneck, role_centroids):
    """Pull each slot toward its role-conditioned centroid.

    Slots repeat in E, A, V pattern: positions 0,3,6,... are entity,
    1,4,7,... are attribute, 2,5,8,... are value.

    Args:
        bottleneck: (B, N*3, d_model)
        role_centroids: nn.Embedding(3, d_model) — learned centroids

    Returns:
        scalar MSE loss
    """
    B, T, d = bottleneck.shape
    device = bottleneck.device

    # Build role index for each slot: [0,1,2, 0,1,2, ...]
    role_idx = torch.arange(3, device=device).repeat(T // 3)  # (T,)
    centroids = role_centroids(role_idx)  # (T, d)

    # MSE between each slot and its role centroid
    return F.mse_loss(bottleneck, centroids.unsqueeze(0).expand_as(bottleneck))


def _compute_role_decomposed_bn_loss(bottleneck, target, bn_role_weights):
    """Role-decomposed bottleneck MSE: separate loss for E, A, V slots.

    Entity and attribute slots should be preserved (Q and A share them).
    Value slots should transform (that's where new information lives).

    Args:
        bottleneck: (B, N*3, d) dynamics output
        target: (B, N*3, d) compressor output for the answer
        bn_role_weights: (entity_w, attr_w, value_w) loss weights

    Returns:
        weighted scalar loss, per-role metrics dict
    """
    B, T, d = bottleneck.shape
    device = bottleneck.device
    w_e, w_a, w_v = bn_role_weights

    # Role masks: E=0,3,6,...  A=1,4,7,...  V=2,5,8,...
    role_idx = torch.arange(T, device=device) % 3
    e_mask = role_idx == 0
    a_mask = role_idx == 1
    v_mask = role_idx == 2

    e_loss = F.mse_loss(bottleneck[:, e_mask], target[:, e_mask])
    a_loss = F.mse_loss(bottleneck[:, a_mask], target[:, a_mask])
    v_loss = F.mse_loss(bottleneck[:, v_mask], target[:, v_mask])

    total = w_e * e_loss + w_a * a_loss + w_v * v_loss
    role_metrics = {
        "bn_e": e_loss.item(),
        "bn_a": a_loss.item(),
        "bn_v": v_loss.item(),
    }
    return total, role_metrics


def compute_diffusion_loss(model, input_ids, input_pad, output_ids, output_pad,
                           output_len, device, timestep, mode_ids=None,
                           aux_ce_weight=0.1, length_weight=0.1,
                           bottleneck_weight=0.0, role_prior_weight=0.0,
                           bn_role_weights=None, detach_dynamics_expander=False,
                           kl_weight=0.0, spectral_weight=0.0):
    """Unified diffusion loss for both IO and dynamics modes.

    When mode_ids is None: IO mode (input is target, no dynamics).
    When mode_ids is provided: dynamics mode (run dynamics core, output is target).

    Args:
        bn_role_weights: (entity_w, attr_w, value_w) for role-decomposed bn loss.
            If None, uses uniform bottleneck_weight on all slots.
        detach_dynamics_expander: if True, detach bottleneck before expander
            during dynamics training. Core trains on bn loss only, no token
            gradients fighting its exploration of W-space.
        kl_weight: VAE KL weight (β), already annealed by caller.
        spectral_weight: weight for spectral penalty (prevents bottleneck
            collapse to 1D manifold by penalizing dominant eigenvalues).
    """
    input_ids = input_ids.to(device)
    input_pad = input_pad.to(device)
    token_emb = model.shared_token_emb

    compress_out = model.compress(input_ids, input_pad)

    # Handle VAE vs deterministic compressor.
    # For VAE: condition expander on mu (clean), not z (noisy). The KL loss
    # still regularizes via z, but the expander sees the same clean signal
    # in training and eval — no double-noise problem.
    vae_info = {}
    if isinstance(compress_out, tuple):
        bottleneck, vae_info = compress_out
        # Expander/length head see mu; dynamics/KL see z (bottleneck)
        expander_bn = vae_info.get("mu", bottleneck)
    else:
        bottleneck = compress_out
        expander_bn = bottleneck

    # Save pre-dynamics bottleneck for length prediction — length is a
    # property of the input, not the dynamics transformation.
    length_bn = expander_bn

    # Role-conditioned prior: pull each slot toward its role centroid (legacy).
    role_loss = torch.tensor(0.0, device=device)
    if role_prior_weight > 0 and hasattr(model, "role_centroids"):
        role_loss = _compute_role_prior_loss(bottleneck, model.role_centroids)

    bottleneck_target = None
    if mode_ids is not None:
        # Dynamics mode: transform through dynamics core
        mode_ids = mode_ids.to(device)
        output_ids = output_ids.to(device)
        output_pad = output_pad.to(device)

        # Compute target bottleneck for direct supervision (detached).
        # Use mu (clean) for consistency with dynamics input.
        if bottleneck_weight > 0 or bn_role_weights is not None:
            with torch.no_grad():
                target_out = model.compress(output_ids, output_pad)
                if isinstance(target_out, tuple):
                    _, target_info = target_out
                    bottleneck_target = target_info.get("mu", target_out[0])
                else:
                    bottleneck_target = target_out

        # Run dynamics on mu (clean) — the expander and bn_loss both see
        # clean conditioning. KL regularization comes from the compressor's
        # z, not from the dynamics path.
        bottleneck = model.forward_dynamics(expander_bn, mode_ids)
        expander_bn = bottleneck
        target_ids = output_ids
        target_pad = output_pad
    else:
        # IO mode: input is the target
        target_ids = input_ids
        target_pad = input_pad

    # Detach bottleneck before expander during dynamics: core trains on
    # bn loss only, token-level gradients don't fight W-space exploration.
    expander_input = expander_bn.detach() if (detach_dynamics_expander and mode_ids is not None) else expander_bn
    pred_emb, _ = model.forward_expander(expander_input, target_ids, target_pad, timestep=timestep)

    non_pad = ~target_pad
    metrics = {}
    if not non_pad.any():
        return torch.tensor(0.0, device=device), metrics

    target_clean = token_emb(target_ids)
    mse_loss = F.mse_loss(pred_emb[non_pad], target_clean[non_pad])

    with torch.no_grad():
        cos = F.cosine_similarity(pred_emb[non_pad], target_clean[non_pad], dim=-1).mean()
        pred_norm = F.normalize(pred_emb[non_pad], dim=-1)
        emb_norm = F.normalize(token_emb.weight, dim=-1)
        nn_ids = torch.matmul(pred_norm, emb_norm.T).argmax(-1)
        tgt_ids = target_ids[non_pad]
        metrics["tok_acc"] = (nn_ids == tgt_ids).float().mean().item()
        metrics["cos"] = cos.item()

        # Per-mode token accuracy (dynamics only)
        if mode_ids is not None:
            for mode_val, mode_name in [(0, "id"), (1, "qa"), (2, "rev")]:
                mask = mode_ids == mode_val
                if mask.any():
                    mode_non_pad = ~target_pad[mask]
                    if mode_non_pad.any():
                        mode_pred = pred_emb[mask][mode_non_pad]
                        mode_tgt = target_ids[mask][mode_non_pad]
                        mode_pred_n = F.normalize(mode_pred, dim=-1)
                        mode_nn = torch.matmul(mode_pred_n, emb_norm.T).argmax(-1)
                        metrics[f"tok_{mode_name}"] = (mode_nn == mode_tgt).float().mean().item()

    aux_loss = torch.tensor(0.0, device=device)
    if model.text_expander.use_decode_proj:
        logits = model.text_expander.decode_proj_logits(pred_emb)
        aux_loss = F.cross_entropy(logits[non_pad] / 0.1, target_ids[non_pad], ignore_index=0)

    len_pred = model.forward_length(length_bn)
    len_loss = F.mse_loss(len_pred, output_len.float().to(device))

    # Bottleneck loss: role-decomposed or uniform
    # Apply to all non-identity modes (qa=1, reverse=2, etc.)
    bn_loss = torch.tensor(0.0, device=device)
    if bottleneck_target is not None:
        transform_mask = mode_ids != 0
        if transform_mask.any():
            if bn_role_weights is not None:
                bn_loss, bn_metrics = _compute_role_decomposed_bn_loss(
                    bottleneck[transform_mask], bottleneck_target[transform_mask], bn_role_weights
                )
                metrics.update(bn_metrics)
            elif bottleneck_weight > 0:
                bn_loss = F.mse_loss(bottleneck[transform_mask], bottleneck_target[transform_mask])

    bn_w = bottleneck_weight if bn_role_weights is None else 1.0

    # VAE KL loss
    kl_loss = torch.tensor(0.0, device=device)
    if kl_weight > 0 and "kl_loss" in vae_info:
        kl_loss = vae_info["kl_loss"]

    # Spectral penalty: prevent bottleneck collapse to 1D.
    # Use mu (not sampled z) for VAE models — sampling noise masks collapse.
    spec_loss = torch.tensor(0.0, device=device)
    if spectral_weight > 0:
        spec_input = vae_info.get("mu", bottleneck)
        spec_loss, spec_metrics = _compute_spectral_loss(spec_input)
        metrics.update(spec_metrics)

    total = (mse_loss + aux_ce_weight * aux_loss + length_weight * len_loss
             + bn_w * bn_loss + role_prior_weight * role_loss
             + kl_weight * kl_loss + spectral_weight * spec_loss)
    metrics["loss"] = total.item()
    metrics["mse"] = mse_loss.item()
    metrics["ce"] = aux_loss.item()
    metrics["len_loss"] = len_loss.item()
    if bottleneck_weight > 0 or bn_role_weights is not None:
        metrics["bn_loss"] = bn_loss.item()
    if role_prior_weight > 0:
        metrics["role_loss"] = role_loss.item()
    if kl_weight > 0 and vae_info:
        metrics["kl"] = kl_loss.item()
        for k in ("kl_entity", "kl_attribute", "kl_value"):
            if k in vae_info:
                metrics[k] = vae_info[k]
    if spectral_weight > 0:
        metrics["spec"] = spec_loss.item()
    return total, metrics
