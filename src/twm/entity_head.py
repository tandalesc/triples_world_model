"""Discrete entity prediction head for TextWorld dynamics latents.

Three classification heads predict which entities are involved
in a state transition:
  - Action type (16 classes): take, put, drop, cook, etc.
  - Object (195 classes): white onion, red potato, chicken leg, etc.
  - Place (27 classes): 13 rooms + 14 locations (kitchen, fridge, etc.)

Predictions become prefix tokens for the AR decoder, providing
sharp entity disambiguation that soft cross-attention cannot resolve.

Uses a closed inventory from data/tw_entity_inventory.json.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityHead(nn.Module):
    """Three-head entity classifier over dynamics latent.

    Architecture:
        latent (B, N*3, d) → attention-pool → shared representation
        → action_head (128 → 256 → 16)
        → object_head (128 → 256 → 195)
        → place_head  (128 → 256 → 27)
    """

    def __init__(
        self,
        inventory_path: str | Path,
        d_model: int = 128,
        n_heads: int = 4,
        bottleneck_dim: int | None = None,
    ):
        super().__init__()
        bn_d = bottleneck_dim or d_model

        # Load inventory
        with open(inventory_path) as f:
            inv = json.load(f)
        cats = inv["categories"]

        self.actions = cats["actions"]
        self.objects = cats["objects"]
        self.places = cats["rooms"] + cats["locations"]

        self.n_actions = len(self.actions)
        self.n_objects = len(self.objects)
        self.n_places = len(self.places)

        # Index lookups
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.object_to_idx = {o: i for i, o in enumerate(self.objects)}
        self.place_to_idx = {p: i for i, p in enumerate(self.places)}

        # Attention pool over latent positions
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_proj = nn.Linear(bn_d, d_model) if bn_d != d_model else nn.Identity()
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1,
        )
        self.pool_ln = nn.LayerNorm(d_model)

        # Classification heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.n_actions),
        )
        self.object_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.n_objects),
        )
        self.place_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.n_places),
        )

    def _pool(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Attention-pool latent positions to single vector."""
        B = bottleneck.shape[0]
        kv = self.pool_proj(bottleneck)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, kv, kv)
        return self.pool_ln(pooled.squeeze(1))  # (B, d)

    def forward(self, bottleneck: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict entity logits.

        Returns dict with 'action', 'object', 'place' logits.
        """
        pooled = self._pool(bottleneck)
        return {
            "action": self.action_head(pooled),   # (B, 16)
            "object": self.object_head(pooled),   # (B, 195)
            "place": self.place_head(pooled),     # (B, 27)
        }

    @torch.no_grad()
    def predict(self, bottleneck: torch.Tensor) -> dict[str, list[str]]:
        """Predict entity strings.

        Returns dict with 'action', 'object', 'place' string lists.
        """
        logits = self.forward(bottleneck)
        action_idx = logits["action"].argmax(-1)  # (B,)
        object_idx = logits["object"].argmax(-1)
        place_idx = logits["place"].argmax(-1)

        return {
            "action": [self.actions[i] for i in action_idx.tolist()],
            "object": [self.objects[i] for i in object_idx.tolist()],
            "place": [self.places[i] for i in place_idx.tolist()],
        }

    def predict_prefix_strings(self, bottleneck: torch.Tensor) -> list[str]:
        """Predict entity prefix as tokenizable strings.

        Returns list of B strings like "take | white onion | table"
        that can be tokenized and fed to the AR decoder as prefix.
        """
        preds = self.predict(bottleneck)
        prefixes = []
        for a, o, p in zip(preds["action"], preds["object"], preds["place"]):
            prefixes.append(f"{a} | {o} | {p}")
        return prefixes

    def compute_loss(
        self,
        bottleneck: torch.Tensor,
        action_targets: torch.Tensor,
        object_targets: torch.Tensor,
        place_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute CE loss on all three heads.

        Targets use -1 for unknown/not-applicable (ignored in loss).
        """
        logits = self.forward(bottleneck)

        losses = {}
        accs = {}
        total = torch.tensor(0.0, device=bottleneck.device)

        for name, lg, tgt in [
            ("action", logits["action"], action_targets),
            ("object", logits["object"], object_targets),
            ("place", logits["place"], place_targets),
        ]:
            valid = tgt >= 0
            if valid.any():
                loss = F.cross_entropy(lg[valid], tgt[valid])
                total = total + loss
                losses[f"{name}_loss"] = loss.item()
                preds = lg[valid].argmax(-1)
                accs[f"{name}_acc"] = (preds == tgt[valid]).float().mean().item()
            else:
                losses[f"{name}_loss"] = 0.0
                accs[f"{name}_acc"] = 0.0

        return total, {**losses, **accs}

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
