"""How far off is the student likely to be? A learned error estimate.

The student's own probabilities are not a confidence: the teacher's answers are
often spread out, so a flat answer can be exactly right. ``ErrorHead`` instead
predicts the student's KL to the teacher, in nats, from what the student
computed on the way to its answer:

    features = [mean and max of the offered options' final states (2d),
                normalized entropy of the answer, its largest probability,
                share of the question's triples the vocabulary does not know]
    log(KL + 0.001) ≈ head(features)        (two small linear layers)

It is trained on rows the student never trained on (held-out towns), so it
learns what being wrong looks like; train_error_head.py then checks it on
another unseen set. Set models only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EPS = 1e-3


def features(model, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (logits, features) for a batch; features are (B, 2d + 3)."""
    logits, hidden = model(b, return_hidden=True)
    mask = b["opt_mask"].unsqueeze(-1)
    k = b["opt_mask"].sum(dim=1).clamp_min(1)
    mean = (hidden * mask).sum(dim=1) / k.unsqueeze(-1)
    mx = hidden.masked_fill(~mask, float("-inf")).max(dim=1).values
    p = torch.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).masked_fill(~b["opt_mask"], 0).sum(dim=1) / torch.log(k.float().clamp_min(2))
    unk = (((b["attr"] == 1) | (b["value"] == 1)) & b["tok_mask"]).sum(dim=1) / b["tok_mask"].sum(dim=1).clamp_min(1)
    return logits, torch.cat([mean, mx, ent.unsqueeze(-1), p.max(dim=-1).values.unsqueeze(-1), unk.float().unsqueeze(-1)], dim=-1)


class ErrorHead(nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, math.log(0.02 + EPS))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Predicted log(KL + EPS)."""
        return self.net(f).squeeze(-1)

    @staticmethod
    def expected_kl(pred_log: torch.Tensor) -> torch.Tensor:
        return (pred_log.exp() - EPS).clamp_min(0)
