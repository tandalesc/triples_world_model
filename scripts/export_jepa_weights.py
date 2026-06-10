#!/usr/bin/env python3
"""Export JEPA nano model weights for browser inference — spec §8.

INT8 weight-only (per-channel symmetric) for attention weights;
fp16 for small scale/bias parameters; operator bake() returns
(cos, sin) INT8-quantized in [-1,1] with scale 1/127 and r as fp16.

Extends the pet-sim export_weights.py to_list() + JSON pattern.

Activation quantization of cross-attention is explicitly NOT done
(documented DETR collapse mode — cross-attn activations stay fp16).

Usage:
    uv run python scripts/export_jepa_weights.py <checkpoint.pt> [out.json]

The script asserts that the nano export fits in <= 303 KB.
mini exports fp16 (~1.9 MB) for the research rig; not the browser target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# quantization helpers
# ---------------------------------------------------------------------------

def _int8_quantize_per_channel(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric INT8 quantisation of a 2-D weight matrix.

    Activations remain fp16 (not quantized) — see module docstring.
    Returns (q_int8, scale_fp32) where q_int8 is np.int8 (out_ch, in_ch)
    and scale_fp32 is np.float32 (out_ch,).  Reconstructed weight ≈ q * scale[:,None].
    """
    w_fp = w.detach().float()  # (out, in)
    scale = w_fp.abs().amax(dim=1) / 127.0  # (out,)
    scale = scale.clamp(min=1e-8)
    q = (w_fp / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return q.numpy(), scale.numpy().astype(np.float32)


def _int8_quantize_cos_sin(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize (cos,sin) values in [-1,1] to INT8 with fixed scale 1/127.

    Returns (q_int8, scale) where scale = 1/127 (fixed, not per-channel).
    """
    scale = 1.0 / 127.0
    q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def _to_fp16_list(t: torch.Tensor) -> list:
    """Convert tensor to fp16, then to a flat Python list (compact JSON)."""
    return t.detach().half().cpu().numpy().ravel().tolist()


def _to_fp32_list(t: torch.Tensor) -> list:
    return t.detach().float().cpu().numpy().ravel().tolist()


# ---------------------------------------------------------------------------
# weight extraction helpers
# ---------------------------------------------------------------------------

def _export_linear_int8(module: torch.nn.Module, name_prefix: str) -> dict:
    """Export a single nn.Linear weight as INT8 (per-channel); bias as fp16."""
    result = {}
    if hasattr(module, "weight"):
        w = module.weight
        if w.dim() == 2:
            q, sc = _int8_quantize_per_channel(w)
            result[f"{name_prefix}_weight_int8"] = q.tolist()
            result[f"{name_prefix}_weight_scale_fp32"] = sc.tolist()
        else:
            result[f"{name_prefix}_weight_fp16"] = _to_fp16_list(w)
    if hasattr(module, "bias") and module.bias is not None:
        result[f"{name_prefix}_bias_fp16"] = _to_fp16_list(module.bias)
    return result


def _export_layer_norm(module: torch.nn.Module, name_prefix: str) -> dict:
    """Export LayerNorm weight and bias as fp16."""
    result = {}
    if hasattr(module, "weight") and module.weight is not None:
        result[f"{name_prefix}_weight_fp16"] = _to_fp16_list(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        result[f"{name_prefix}_bias_fp16"] = _to_fp16_list(module.bias)
    return result


def _export_mha_int8(mha_module: torch.nn.Module, name_prefix: str) -> dict:
    """Export a MultiheadAttention or custom attn module weights as INT8.

    Handles both PyTorch nn.MultiheadAttention and simple
    projection-weight modules. Activations stay fp16.
    """
    result = {}
    sd = {k: v for k, v in mha_module.named_parameters(recurse=False)}
    # Try PyTorch MHA layout first
    for wname in ("in_proj_weight", "out_proj_weight"):
        if wname in sd:
            q, sc = _int8_quantize_per_channel(sd[wname])
            result[f"{name_prefix}_{wname}_int8"] = q.tolist()
            result[f"{name_prefix}_{wname}_scale_fp32"] = sc.tolist()
    for bname in ("in_proj_bias", "out_proj_bias"):
        if bname in sd:
            result[f"{name_prefix}_{bname}_fp16"] = _to_fp16_list(sd[bname])
    # Nested out_proj (nn.Linear inside MHA)
    if hasattr(mha_module, "out_proj"):
        op = mha_module.out_proj
        if hasattr(op, "weight"):
            q, sc = _int8_quantize_per_channel(op.weight)
            result[f"{name_prefix}_out_proj_weight_int8"] = q.tolist()
            result[f"{name_prefix}_out_proj_weight_scale_fp32"] = sc.tolist()
        if hasattr(op, "bias") and op.bias is not None:
            result[f"{name_prefix}_out_proj_bias_fp16"] = _to_fp16_list(op.bias)
    return result


# ---------------------------------------------------------------------------
# operator export  (spec §8: bake() INT8 + r fp16)
# ---------------------------------------------------------------------------

def _export_operator(op_module: torch.nn.Module) -> dict:
    """Export operator via bake() -> (cos, sin, r) per verb per block.

    cos/sin are INT8 with scale 1/127 (values in [-1,1]).
    r is fp16 (one scalar per block per verb; small tensor).
    """
    baked = op_module.bake()  # {"cos": ..., "sin": ..., "r": ...}
    cos_t = baked["cos"]   # (V, dn//2)
    sin_t = baked["sin"]   # (V, dn//2)
    r_t = baked["r"]       # (V, dn//2)

    cos_np = cos_t.detach().float().cpu().numpy()
    sin_np = sin_t.detach().float().cpu().numpy()
    r_np = r_t.detach().float().cpu().numpy()

    cos_q, cos_scale = _int8_quantize_cos_sin(cos_np)
    sin_q, sin_scale = _int8_quantize_cos_sin(sin_np)

    V, n_blocks = cos_np.shape
    return {
        "n_verbs": V,
        "n_blocks": n_blocks,
        "cos_int8": cos_q.tolist(),
        "cos_scale": float(cos_scale),    # fixed 1/127
        "sin_int8": sin_q.tolist(),
        "sin_scale": float(sin_scale),
        "r_fp16": r_np.astype(np.float16).tolist(),
    }


# ---------------------------------------------------------------------------
# token_emb INT8 shared table
# ---------------------------------------------------------------------------

def _export_token_emb_int8(emb: torch.nn.Embedding) -> dict:
    """Export token embedding table as INT8 (per-channel = per-embedding-dim)."""
    w = emb.weight  # (vocab, d)
    # treat each row (token) as a channel for per-channel symmetric quant
    q, sc = _int8_quantize_per_channel(w)  # (vocab, d) int8; (vocab,) fp32 scale
    return {
        "vocab_size": w.shape[0],
        "d_model": w.shape[1],
        "weight_int8": q.tolist(),
        "weight_scale_fp32": sc.tolist(),
        "note": "frozen domain BPE embedding table, NOT exported as dense FP32",
    }


# ---------------------------------------------------------------------------
# slot encoder export
# ---------------------------------------------------------------------------

def _export_slot_encoder(se: torch.nn.Module) -> dict:
    """Walk the slot encoder and export all attn weights as INT8, norms as fp16."""
    out = {}

    # text self-attention block (ALBERT-tied)
    if hasattr(se, "text_attn"):
        out.update(_export_mha_int8(se.text_attn, "text_self_attn"))
    if hasattr(se, "text_ffn"):
        # FFN: two linear layers
        ffn = se.text_ffn
        for i, layer in enumerate(ffn.children() if hasattr(ffn, "children") else []):
            if isinstance(layer, torch.nn.Linear):
                out.update(_export_linear_int8(layer, f"text_ffn_{i}"))
    if hasattr(se, "text_norm1"):
        out.update(_export_layer_norm(se.text_norm1, "text_norm1"))
    if hasattr(se, "text_norm2"):
        out.update(_export_layer_norm(se.text_norm2, "text_norm2"))

    # slot queries
    if hasattr(se, "slot_queries"):
        out["slot_queries_fp16"] = _to_fp16_list(se.slot_queries)
    if hasattr(se, "slot_mu"):
        out["slot_mu_fp16"] = _to_fp16_list(se.slot_mu)
    if hasattr(se, "slot_log_sigma"):
        out["slot_log_sigma_fp16"] = _to_fp16_list(se.slot_log_sigma)

    # cross-attention (attn activations NOT quantized — DETR collapse mode)
    if hasattr(se, "cross_attn"):
        out.update(_export_mha_int8(se.cross_attn, "cross_attn"))
    if hasattr(se, "cross_norm"):
        out.update(_export_layer_norm(se.cross_norm, "cross_norm"))

    # slot self-attention coordination
    if hasattr(se, "slot_self_attn"):
        out.update(_export_mha_int8(se.slot_self_attn, "slot_self_attn"))
    if hasattr(se, "slot_norm"):
        out.update(_export_layer_norm(se.slot_norm, "slot_norm"))

    # NounHead
    if hasattr(se, "noun_head"):
        out.update(_export_linear_int8(se.noun_head, "noun_head"))
    if hasattr(se, "noun_ln"):
        out.update(_export_layer_norm(se.noun_ln, "noun_ln"))

    # VerbHead
    if hasattr(se, "verb_head"):
        out.update(_export_linear_int8(se.verb_head, "verb_head"))

    return out


# ---------------------------------------------------------------------------
# readout export
# ---------------------------------------------------------------------------

def _export_readout(readout: torch.nn.Module) -> dict:
    out = {}
    if hasattr(readout, "cond_query"):
        out["cond_query_fp16"] = _to_fp16_list(readout.cond_query)
    if hasattr(readout, "attn"):
        out.update(_export_mha_int8(readout.attn, "readout_attn"))
    if hasattr(readout, "ln"):
        out.update(_export_layer_norm(readout.ln, "readout_ln"))
    # Also handle nn.Linear predictor
    if hasattr(readout, "predictor"):
        pred = readout.predictor
        for i, layer in enumerate(pred.children() if hasattr(pred, "children") else []):
            if isinstance(layer, torch.nn.Linear):
                out.update(_export_linear_int8(layer, f"predictor_{i}"))
    return out


# ---------------------------------------------------------------------------
# byte size estimation
# ---------------------------------------------------------------------------

def _count_bytes(obj) -> int:
    """Recursively count bytes in a nested list/dict of int/float."""
    if isinstance(obj, dict):
        return sum(_count_bytes(v) for v in obj.values())
    if isinstance(obj, list):
        if len(obj) == 0:
            return 0
        flat = _flatten(obj)
        # heuristic: int8 lists are lists of ints in [-127,127] -> 1 byte each
        # fp16 lists are floats stored as 2 bytes each
        # fp32 lists are floats stored as 4 bytes each
        if all(isinstance(x, int) and -128 <= x <= 127 for x in flat[:16]):
            return len(flat)  # 1 byte per INT8
        return len(flat) * 2  # fp16 assumption
    if isinstance(obj, (int, float)):
        return 4
    return 0


def _flatten(lst) -> list:
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# main export function
# ---------------------------------------------------------------------------

def export_jepa_weights(
    model,
    out_path: str | Path,
    profile: str = "jepa_nano",
    assert_under_303kb: bool = True,
) -> dict:
    """Export JEPA model weights to JSON in the pet-sim format.

    Only exports online encoder + operator + readout. EMA, predictor
    (used during training only), and SIGReg are training-only and excluded.

    Returns the weights dict (also written to out_path).
    """
    model.eval()
    cfg = model.cfg if hasattr(model, "cfg") else {}

    weights: dict = {
        "format": "jepa_v1_int8",
        "profile": profile,
        "config": _serialize_config(cfg),
        "note": (
            "INT8 weight-only quantization for attn weights (per-channel symmetric). "
            "Activation fp16 assumed. Cross-attn activations NOT quantized (DETR collapse mode). "
            "Operator: (cos,sin) INT8 with scale=1/127; r as fp16."
        ),
    }

    # token_emb (frozen, exported as INT8 shared table)
    if hasattr(model, "slot_encoder") and hasattr(model.slot_encoder, "token_emb"):
        weights["token_emb"] = _export_token_emb_int8(model.slot_encoder.token_emb)
    elif hasattr(model, "token_emb"):
        weights["token_emb"] = _export_token_emb_int8(model.token_emb)

    # text positional embedding (fp16 — small)
    if hasattr(model, "slot_encoder") and hasattr(model.slot_encoder, "text_pos_emb"):
        weights["text_pos_emb_fp16"] = _to_fp16_list(model.slot_encoder.text_pos_emb.weight)
    elif hasattr(model, "text_pos_emb"):
        weights["text_pos_emb_fp16"] = _to_fp16_list(model.text_pos_emb.weight)

    # slot encoder (ALBERT-tied self-attn + cross-attn + coordination)
    if hasattr(model, "slot_encoder"):
        weights["slot_encoder"] = _export_slot_encoder(model.slot_encoder)

    # operator (baked cos/sin/r)
    if hasattr(model, "operator"):
        weights["operator"] = _export_operator(model.operator)

    # readout (attention-pool over a*)
    if hasattr(model, "readout"):
        weights["readout"] = _export_readout(model.readout)

    # JS API documentation embedded in export
    weights["js_api"] = {
        "step_latent": "a_star = operator_apply(k, verb_idx)  // RoPE-style elementwise",
        "undo_latent": "k = operator_inverse_apply(a_star, verb_idx)  // structural exact undo",
        "apply_rope": "(x', y') = (r*(x*cos - y*sin), r*(x*sin + y*cos))  // per 2-block",
        "inverse_rope": "(x, y) = ((x'*cos + y'*sin)/r, (-x'*sin + y'*cos)/r)",
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weights, f)

    size_bytes = out_path.stat().st_size
    size_kb = size_bytes / 1024
    print(f"Exported JEPA weights to {out_path.name}  ({size_kb:.1f} KB)")

    if profile == "jepa_nano" and assert_under_303kb:
        assert size_kb <= 303, (
            f"nano export is {size_kb:.1f} KB — exceeds 303 KB pet-sim envelope! "
            "Check that attn weights are INT8, emb table is quantized, "
            "and no large fp32 tensors snuck in."
        )
    else:
        target_kb = 303 if profile == "jepa_nano" else 2000
        print(f"  ({profile} target: {'<= 303 KB' if profile == 'jepa_nano' else '~1.9 MB fp16'})")

    return weights


def _serialize_config(cfg) -> dict:
    """Convert config object or dict to a serializable dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # dataclass or object
    result = {}
    for attr in ("d_model", "d_noun", "n_slots", "n_verbs", "block",
                 "n_text_layers", "tie_text_layers", "n_heads", "n_slot_iters",
                 "operator_group", "n_steps_T", "vocab_size", "max_text_tokens"):
        if hasattr(cfg, attr):
            result[attr] = getattr(cfg, attr)
    return result


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: export_jepa_weights.py <checkpoint.pt> [out.json] [--no-assert]")
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else \
        ckpt_path.parent / "jepa_weights.json"
    assert_size = "--no-assert" not in sys.argv

    # load model — supports checkpoints that store either a full model object or a
    # state_dict (the train_jepa.py trainer saves `{"model": state_dict, "config":
    # cfg.to_dict(), ...}`). Rebuild via build_jepa_model when we have a state_dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _rebuild_from_state(state_dict, cfg_dict):
        import torch.nn as nn
        from twm.jepa.model import build_jepa_model
        from twm.jepa.config import JEPAConfig

        cfg = JEPAConfig.from_dict(cfg_dict) if cfg_dict else JEPAConfig()
        token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
        token_emb.weight.requires_grad_(False)
        model = build_jepa_model(cfg, token_emb)
        model.load_state_dict(state_dict, strict=False)
        return model

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        # trainer layout: state_dict under "model", config under "config".
        model = _rebuild_from_state(ckpt["model"], ckpt.get("config", {}))
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]  # full model object stored directly
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model = _rebuild_from_state(ckpt["model_state_dict"], ckpt.get("config", {}))
    else:
        model = ckpt  # assume full model object

    profile = "jepa_nano"
    d_model = None
    if hasattr(model, "cfg") and hasattr(model.cfg, "d_model"):
        d_model = model.cfg.d_model
    elif hasattr(model, "encoder") and hasattr(model.encoder, "d_model"):
        d_model = model.encoder.d_model
    if d_model is not None:
        profile = "jepa_nano" if d_model <= 64 else "jepa_mini"

    export_jepa_weights(model, out_path, profile=profile, assert_under_303kb=assert_size)


if __name__ == "__main__":
    main()
