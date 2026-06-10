"""Integration regression tests (integrator-owned).

Guards the two findings the reviewers flagged that the per-task unit suites could
not catch because they exercise only single modules:

  1. JEPALoss must NOT auto-register the operator as a submodule (else the operator
     codebook lands in AdamW twice -> 2x effective LR + a duplicate-param error).
  2. The trainer's optimizer param list must contain no id-duplicate parameters.
  3. The full model builds + does one train-style forward/backward end to end with
     finite losses and gradients flowing only to online params (not EMA).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from twm.jepa.config import JEPAConfig
from twm.jepa.model import build_jepa_model
from twm.jepa.losses import JEPALoss


def _nano_cfg():
    return JEPAConfig.from_dict({"profile": "jepa_nano"})


def test_loss_does_not_register_operator_submodule():
    from twm.jepa.operator import RotationScaleOperator

    op = RotationScaleOperator(n_verbs=8, d_noun=32, block=2)
    loss = JEPALoss(operator=op)
    # The loss only needs READ access to theta/log_r; it must own zero params.
    assert len(list(loss.parameters())) == 0
    assert "operator" not in dict(loss.named_modules())
    # But it can still read the codebook for the spread penalty.
    params = loss._operator_params()
    assert params is not None and len(params) == 2


def test_optimizer_param_list_has_no_duplicates():
    cfg = _nano_cfg()
    token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    token_emb.weight.requires_grad_(False)
    model = build_jepa_model(cfg, token_emb)
    loss_fn = JEPALoss(operator=model.operator)

    merged = list(model.online_parameters()) + [
        p for p in loss_fn.parameters() if p.requires_grad
    ]
    ids = [id(p) for p in merged]
    assert len(ids) == len(set(ids)), "operator params duplicated in optimizer list"

    # operator codebook is present exactly once via online_parameters().
    op_ids = {id(model.operator.theta), id(model.operator.log_r)}
    online_ids = [id(p) for p in model.online_parameters()]
    for oid in op_ids:
        assert online_ids.count(oid) == 1


def test_end_to_end_forward_backward_finite():
    torch.manual_seed(0)
    cfg = _nano_cfg()
    token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    token_emb.weight.requires_grad_(False)
    model = build_jepa_model(cfg, token_emb)
    loss_fn = JEPALoss(operator=model.operator)

    B, T = 4, cfg.data.max_text_tokens
    src_ids = torch.randint(0, cfg.data.vocab_size, (B, T))
    tgt_ids = torch.randint(0, cfg.data.vocab_size, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)

    out = model(src_ids, pad, tgt_ids, pad, gumbel_tau=1.0, hard=False)
    assert "z_target" in out
    total, comps = loss_fn(
        out["k"], out["verb_logits"], out["zhat"], out["z_target"], 1.0, False
    )
    assert torch.isfinite(total)
    for key in ("L_pred", "L_sigreg", "L_div"):
        assert key in comps and torch.isfinite(torch.tensor(comps[key]))

    total.backward()
    # EMA params receive no gradient.
    for p in model.ema.parameters():
        assert p.grad is None
    # At least the operator codebook + readout receive gradient.
    assert model.operator.theta.grad is not None
