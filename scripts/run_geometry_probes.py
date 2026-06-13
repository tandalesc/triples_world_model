#!/usr/bin/env python3
"""Standalone runner for the four entity-world geometry probes (new-probe-designs.md).

Loads a FROZEN JEPA v2 checkpoint and runs, under torch.no_grad on CPU/MPS:
  1. counterfactual locality  -> ent_cf_locality_*
  2. latent-NN purity         -> ent_latent_nn_purity*
  3. discriminative-variance  -> ent_disc_var_ratio / ent_disc_delta_r2
  4. slot occupancy           -> ent_slot_occupancy / ent_slot_occupancy_frac

Prints a JSON dict of all ent_* scalars to stdout (nothing else on stdout).

Usage:
    uv run python scripts/run_geometry_probes.py \
        --ckpt results/jepa_v4_smoke/model_latest.pt \
        --labeled_dir data/entity_world --split test_iid

If --config is omitted the hparams are read from the checkpoint's embedded `config`
dict (every train_jepa_v2 checkpoint stores it). The model-load path mirrors
scripts/probe_retraction.py (build_jepa_model_v2 + load_state_dict(strict=False)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _pick_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to model_latest.pt (frozen).")
    p.add_argument("--config", default=None,
                   help="Model config JSON. If omitted, read from the checkpoint's "
                        "embedded `config` dict.")
    p.add_argument("--labeled_dir", default="data/entity_world",
                   help="Dir holding {split}_labeled.jsonl + manifest.json.")
    p.add_argument("--split", default="test_iid", help="Labeled split to probe.")
    p.add_argument("--max_chains", type=int, default=128,
                   help="Cap on chains loaded for the probes.")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (auto if unset).")
    args = p.parse_args()

    device = _pick_device(args.device)

    from twm.jepa.config import JEPAConfig
    from twm.jepa.model import build_jepa_model_v2
    from twm.domain_bpe import DomainBPETokenizer
    from twm.jepa import diagnostics as diag
    import torch.nn as nn

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # Config: explicit file wins, else the checkpoint's embedded dict.
    if args.config is not None:
        cfg = JEPAConfig.from_json(args.config)
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        cfg = JEPAConfig.from_dict(ckpt["config"])
    else:
        sys.exit("ERROR: no --config and the checkpoint has no embedded `config` dict.")

    max_text_tokens = cfg.data.max_text_tokens
    append_eos = getattr(cfg.data, "append_eos", True)

    # Build + load the frozen model (mirrors probe_retraction.main).
    token_emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, token_emb).to(device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = DomainBPETokenizer.load(cfg.data.tokenizer, max_length=max_text_tokens)

    # Oracle module (deterministic, no GPU).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_entity_world", REPO / "scripts" / "generate_entity_world.py"
    )
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)

    manifest = diag._load_manifest(args.labeled_dir)
    chains = diag._load_labeled_split(
        args.labeled_dir, args.split, tokenizer, max_text_tokens, append_eos,
        max_chains=args.max_chains,
    )

    out: dict = {}
    with torch.no_grad():
        # (1) counterfactual locality
        try:
            out.update(diag._counterfactual_locality(
                model, tokenizer, chains, device, gen_mod, manifest,
                max_text_tokens, append_eos,
            ))
        except Exception as exc:  # keep going; record the error
            out["_ent_cf_locality_error"] = f"{type(exc).__name__}: {exc}"
        # (2) latent-NN purity
        try:
            out.update(diag._latent_nn_purity(model, chains, device, manifest))
        except Exception as exc:
            out["_ent_latent_nn_purity_error"] = f"{type(exc).__name__}: {exc}"
        # (3) discriminative-variance ratio
        try:
            out.update(diag._discriminative_variance_ratio(
                model, chains, device, gen_mod, manifest,
            ))
        except Exception as exc:
            out["_ent_disc_var_ratio_error"] = f"{type(exc).__name__}: {exc}"
        # (4) slot occupancy
        try:
            out.update(diag._slot_occupancy(model, chains, device))
        except Exception as exc:
            out["_ent_slot_occupancy_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
