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


def compute_diffusion_loss(model, input_ids, input_pad, output_ids, output_pad,
                           output_len, device, timestep, mode_ids=None,
                           aux_ce_weight=0.1, length_weight=0.1):
    """Unified diffusion loss for both IO and dynamics modes.

    When mode_ids is None: IO mode (input is target, no dynamics).
    When mode_ids is provided: dynamics mode (run dynamics core, output is target).
    """
    input_ids = input_ids.to(device)
    input_pad = input_pad.to(device)
    token_emb = model.shared_token_emb

    bottleneck = model.compress(input_ids, input_pad)

    if mode_ids is not None:
        # Dynamics mode: transform through dynamics core
        mode_ids = mode_ids.to(device)
        output_ids = output_ids.to(device)
        output_pad = output_pad.to(device)
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)
        target_ids = output_ids
        target_pad = output_pad
    else:
        # IO mode: input is the target
        target_ids = input_ids
        target_pad = input_pad

    pred_emb, _ = model.forward_expander(bottleneck, target_ids, target_pad, timestep=timestep)

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
            for mode_val, mode_name in [(0, "id"), (1, "qa")]:
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

    len_pred = model.forward_length(bottleneck)
    len_loss = F.mse_loss(len_pred, output_len.float().to(device))

    total = mse_loss + aux_ce_weight * aux_loss + length_weight * len_loss
    metrics["loss"] = total.item()
    metrics["mse"] = mse_loss.item()
    metrics["ce"] = aux_loss.item()
    metrics["len_loss"] = len_loss.item()
    return total, metrics
