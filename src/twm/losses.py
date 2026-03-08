"""Pluggable loss functions for Triple World Model training.

Provides a common interface for different training objectives:
- CrossEntropyLoss: standard classification over phrase vocabulary
- cosine_embedding_loss: existing cosine similarity loss (for NN decoder)

Future strategies to implement:
- VariationalLoss: VAE-style ELBO with KL divergence on latent
- ContrastiveLoss: pull correct triples together, push wrong ones apart
- EnergyLoss: energy-based model scoring valid vs invalid transitions
- DPOLoss: direct preference optimization over triple predictions
- RLHFLoss: reward model + PPO for triple generation
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleLoss(ABC):
    """Abstract interface for TWM training losses.

    All losses take model outputs and targets, return a scalar loss
    and an optional dict of extra metrics for logging.
    """

    @abstractmethod
    def __call__(
        self,
        predictions: dict,
        targets: dict,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: model output (format depends on decoder type)
            targets: ground truth (format depends on decoder type)
            pad_mask: (B, T) True where padded

        Returns:
            loss: scalar tensor
            metrics: dict of loggable values
        """
        ...


class Seq2SeqCrossEntropyLoss(TripleLoss):
    """Per-role cross-entropy loss for seq2seq decoder.

    Computes cross-entropy separately for entity, attr, and value positions,
    masking out pad triples. Returns combined loss and per-role breakdowns.
    """

    def __init__(self, pad_id: int = 0, label_smoothing: float = 0.0):
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: {"entity": (B, M, V_e), "attr": (B, M, V_a), "value": (B, M, V_v)}
            targets: {"entity": (B, M), "attr": (B, M), "value": (B, M)} integer IDs
            pad_mask: (B, M) True where triple is padding (applied per-triple, not per-position)
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        metrics = {}

        for role in ("entity", "attr", "value"):
            logits = predictions[role]  # (B, M, V_role)
            tgt = targets[role]          # (B, M)
            B, M, V = logits.shape

            # Flatten for cross-entropy
            logits_flat = logits.reshape(-1, V)
            tgt_flat = tgt.reshape(-1)

            if pad_mask is not None:
                # Mask: ignore positions where the triple is padding
                valid = ~pad_mask.reshape(-1)
                logits_flat = logits_flat[valid]
                tgt_flat = tgt_flat[valid]

            if tgt_flat.numel() == 0:
                role_loss = torch.tensor(0.0, device=logits.device)
            else:
                role_loss = F.cross_entropy(
                    logits_flat, tgt_flat,
                    ignore_index=self.pad_id,
                    label_smoothing=self.label_smoothing,
                )

            total_loss = total_loss + role_loss
            metrics[f"loss_{role}"] = role_loss.item()

            # Per-role accuracy
            if tgt_flat.numel() > 0:
                preds = logits_flat.argmax(dim=-1)
                acc = (preds == tgt_flat).float().mean().item()
                metrics[f"acc_{role}"] = acc

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics


class RoundTripContrastiveLoss(TripleLoss):
    """Contrastive round-trip consistency loss.

    Checks that decoded output, when re-embedded through the phrase embedding
    space, is cosine-similar to the target ST embeddings. Fully differentiable
    via soft lookup: softmax(logits) @ phrase_embeddings = expected embedding.

    This addresses the NN decoder ceiling by providing a continuous training
    signal that rewards semantically correct outputs even when the exact
    phrase ID doesn't match.
    """

    def __init__(
        self,
        phrase_embeddings: dict[str, torch.Tensor],
        temperature: float = 1.0,
    ):
        """
        Args:
            phrase_embeddings: per-role ST embeddings, e.g.
                {"entity": (V_e, st_dim), "attr": (V_a, st_dim), "value": (V_v, st_dim)}
            temperature: softmax temperature (lower = sharper, closer to argmax)
        """
        self.phrase_embeddings = phrase_embeddings  # kept on CPU, moved lazily
        self.temperature = temperature
        self._device_cache: dict[str, torch.Tensor] = {}

    def _get_embeddings(self, role: str, device: torch.device) -> torch.Tensor:
        key = f"{role}_{device}"
        if key not in self._device_cache:
            # Clone to detach from inference mode (sentence-transformers outputs)
            self._device_cache[key] = self.phrase_embeddings[role].clone().detach().to(device)
        return self._device_cache[key]

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: {"entity": (B, M, V_e), ...} logits from seq2seq decoder
            targets: {"entity_embeds": (B, M, st_dim), ...} target ST embeddings
            pad_mask: (B, M) True where padded
        """
        device = next(iter(predictions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        for role in ("entity", "attr", "value"):
            logits = predictions[role]  # (B, M, V_role)
            tgt_emb = targets[f"{role}_embeds"]  # (B, M, st_dim)
            phrase_emb = self._get_embeddings(role, device)  # (V_role, st_dim)

            # Soft lookup: softmax(logits/T) @ phrase_embeddings
            probs = F.softmax(logits / self.temperature, dim=-1)  # (B, M, V_role)
            pred_emb = probs @ phrase_emb  # (B, M, st_dim)

            # Cosine similarity per position
            pred_norm = F.normalize(pred_emb, dim=-1)
            tgt_norm = F.normalize(tgt_emb, dim=-1)
            cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # (B, M)

            if pad_mask is not None:
                cos_sim = cos_sim.masked_fill(pad_mask, 0.0)
                n_valid = (~pad_mask).sum().clamp(min=1)
                role_loss = 1.0 - cos_sim.sum() / n_valid
            else:
                role_loss = 1.0 - cos_sim.mean()

            total_loss = total_loss + role_loss
            metrics[f"roundtrip_{role}"] = role_loss.item()

        metrics["loss_roundtrip"] = total_loss.item()
        return total_loss, metrics


class CombinedLoss(TripleLoss):
    """Weighted combination of multiple losses."""

    def __init__(self, losses: list[tuple[TripleLoss, float]]):
        """
        Args:
            losses: list of (loss_fn, weight) tuples
        """
        self.losses = losses

    def __call__(
        self,
        predictions: dict,
        targets: dict,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = next(iter(predictions.values())).device
        total = torch.tensor(0.0, device=device)
        all_metrics = {}
        for loss_fn, weight in self.losses:
            loss, metrics = loss_fn(predictions, targets, pad_mask)
            total = total + weight * loss
            all_metrics.update(metrics)
        all_metrics["loss_total"] = total.item()
        return total, all_metrics


class CosineEmbeddingLoss(TripleLoss):
    """Wraps existing cosine loss for the NN decoder path."""

    def __call__(
        self,
        predictions: dict,
        targets: dict,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = predictions["embeds"]   # (B, T, st_dim)
        tgt = targets["embeds"]        # (B, T, st_dim)
        loss = cosine_embedding_loss(pred, tgt, pad_mask)
        return loss, {"loss_cosine": loss.item()}


def cosine_embedding_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cosine similarity loss: 1 - mean(cosine_similarity) over non-pad positions."""
    pred_norm = F.normalize(pred, dim=-1)
    tgt_norm = F.normalize(target, dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)

    if pad_mask is not None:
        cos_sim = cos_sim.masked_fill(pad_mask, 0.0)
        n_valid = (~pad_mask).sum().clamp(min=1)
        return 1.0 - cos_sim.sum() / n_valid
    else:
        return 1.0 - cos_sim.mean()
