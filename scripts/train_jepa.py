"""Train the JEPA Operator world model (spec §10).

Usage:
    uv run python scripts/train_jepa.py configs/jepa_nano.json

Loop (spec §10): per batch — forward (Gumbel-softmax verbs) → JEPALoss → backward →
clip_grad_norm_(online, 1.0) → optimizer.step() → model.ema_update(τ) → anneal τ_g →
periodic eval_diagnostics. AdamW + CosineAnnealingLR with linear warmup, on ONLINE
params only. EMA params are excluded from the optimizer and from grad clipping.

This script owns the training orchestration. The model, operator, slot encoder,
losses, data, and diagnostics are concurrently-developed sibling modules; this
script composes them through the frozen contracts in twm.jepa.__init__.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make `twm` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.jepa.config import JEPAConfig
from twm.jepa.model import build_jepa_model


def resolve_device(device_str: str | None = None) -> torch.device:
    """Device order: cuda -> mps -> cpu (spec / CLAUDE.md convention)."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def gumbel_tau_at(step: int, total_steps: int, vc) -> float:
    """Annealed Gumbel temperature τ_g: start -> end over the first `anneal_frac`
    of training, then held at `end` (spec §3 VerbHead fix)."""
    anneal_steps = max(1, int(vc.anneal_frac * total_steps))
    if step >= anneal_steps:
        return vc.gumbel_tau_end
    frac = step / anneal_steps
    return vc.gumbel_tau_start + (vc.gumbel_tau_end - vc.gumbel_tau_start) * frac


def lr_factor_at(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 0 — multiplicative on base lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_token_emb(vocab_size: int, d_model: int) -> nn.Embedding:
    """Frozen domain-BPE token embedding (spec §2 token_emb, NOT trained)."""
    emb = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(emb.weight, std=0.02)
    emb.weight.requires_grad_(False)  # frozen
    return emb


def build_loss(cfg, operator):
    """Construct JEPALoss (T3), passing operator + loss config flexibly.

    The §3 angle/scale-spread term of L_div needs the operator's theta/log_r, so the
    loss is constructed with a reference to the operator. Constructor kwargs are
    filtered to what JEPALoss actually accepts (sibling-module naming drift safety).
    """
    from twm.jepa import JEPALoss
    import inspect

    lc = cfg.loss
    candidate = dict(
        operator=operator,
        w_pred=lc.w_pred,
        w_sigreg=lc.w_sigreg,
        w_div=lc.w_div,
        w_scale_reg=lc.w_scale_reg,
        n_slices=lc.sigreg.n_slices,
        n_knots=lc.sigreg.n_knots,
        knot_max=lc.sigreg.knot_max,
        standardize=lc.sigreg.standardize,
        n_verbs=cfg.model.n_verbs,
    )
    sig = inspect.signature(JEPALoss.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return JEPALoss(**candidate)
    accepted = {n for n in sig.parameters if n != "self"}
    return JEPALoss(**{k: v for k, v in candidate.items() if k in accepted})


def _get_batch(dataset, idx: torch.Tensor):
    """Return (src_ids, src_pad, tgt_ids, tgt_pad) for an index tensor.

    Prefer the contiguous-tensor layout the dataset exposes (spec §6: stored as
    src_ids/src_pad/tgt_ids/tgt_pad CPU tensors, direct index slicing). Fall back to
    stacking per-item dicts via __getitem__ if those attributes are not present.
    """
    for attrs in (
        ("src_ids", "src_pad", "tgt_ids", "tgt_pad"),
        ("_src_ids", "_src_pad", "_tgt_ids", "_tgt_pad"),
    ):
        if all(hasattr(dataset, a) for a in attrs):
            t = [getattr(dataset, a) for a in attrs]
            return tuple(x[idx] for x in t)
    items = [dataset[int(i)] for i in idx]
    return (
        torch.stack([it["src_ids"] for it in items]),
        torch.stack([it["src_pad"] for it in items]),
        torch.stack([it["tgt_ids"] for it in items]),
        torch.stack([it["tgt_pad"] for it in items]),
    )


def maybe_eval_diagnostics(model, dataset, device, cfg, epoch):
    """Run §5 diagnostics if the module is available (import-guarded so training
    survives diagnostics being mid-build)."""
    try:
        from twm.jepa.diagnostics import eval_diagnostics
    except Exception as e:  # module not written yet / broken — never block training
        print(f"  [diagnostics skipped: {e}]", flush=True)
        return
    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics = eval_diagnostics(
            model, dataset, device,
            n_examples=cfg.eval.n_examples, out_dir=str(out_dir),
        )
        flat = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"  diag[ep{epoch}] {flat}", flush=True)
    except Exception as e:
        print(f"  [diagnostics error: {e}]", flush=True)


def save_checkpoint(model, cfg, step, epoch, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(ckpt, out_dir / "model_latest.pt")


def train(config_path: str):
    cfg = JEPAConfig.from_json(config_path)
    seed_everything(cfg.seed)
    device = resolve_device()

    print(f"=== JEPA Operator training ===")
    print(f"Config: {config_path}")
    print(f"Profile: {cfg.profile}  device: {device}")

    # ---- data ----
    from twm.jepa.data import JEPAChainDataset
    from twm.domain_bpe import DomainBPETokenizer

    # JEPAChainDataset takes a pre-constructed tokenizer object (not a path) and has
    # no `pairing` kwarg — pairing is structurally adjacent (spec §6). Load the BPE
    # artifact here, then hand the dataset the tokenizer.
    tokenizer = DomainBPETokenizer.load(
        cfg.data.tokenizer, max_length=cfg.data.max_text_tokens
    )
    dataset = JEPAChainDataset(
        path=cfg.data.path,
        tokenizer=tokenizer,
        max_text_tokens=cfg.data.max_text_tokens,
    )
    n_train = len(dataset)
    if getattr(cfg.data, "max_chains", None):
        # Smoke / debug cap: keep only the first max_chains*2 pairs (chain len 3 ->
        # 2 adjacent pairs each). Slicing the contiguous tensors keeps direct-index
        # slicing intact.
        cap = int(cfg.data.max_chains) * 2
        if cap < n_train:
            dataset._src_ids = dataset._src_ids[:cap].contiguous()
            dataset._src_pad = dataset._src_pad[:cap].contiguous()
            dataset._tgt_ids = dataset._tgt_ids[:cap].contiguous()
            dataset._tgt_pad = dataset._tgt_pad[:cap].contiguous()
            dataset._src_texts = dataset._src_texts[:cap]
            dataset._tgt_texts = dataset._tgt_texts[:cap]
            n_train = len(dataset)
    print(f"Dataset: {n_train} pairs")

    # ---- model ----
    token_emb = build_token_emb(cfg.data.vocab_size, cfg.model.d_model)
    model = build_jepa_model(cfg, token_emb).to(device)
    loss_fn = build_loss(cfg, model.operator).to(device)

    n_online = sum(p.numel() for p in model.online_parameters())
    print(f"Online (trainable) params: {n_online:,}")

    # ---- optimizer (ONLINE params only) + cosine/warmup ----
    # The operator's theta/log_r are already in model.online_parameters(); the loss
    # holds only a (non-registered) read reference to the operator, so it has no
    # independent trainable params. We still merge + dedup-by-id defensively so a
    # future loss-owned param lands exactly once (no 2x-LR duplicate-param bug).
    o = cfg.optim
    seen_ids: set[int] = set()
    opt_params = []
    for p in list(model.online_parameters()) + [
        p for p in loss_fn.parameters() if p.requires_grad
    ]:
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        opt_params.append(p)
    optimizer = torch.optim.AdamW(
        opt_params, lr=o.lr, weight_decay=o.weight_decay,
    )
    bs = o.batch_size
    steps_per_epoch = max(1, n_train // bs)
    total_steps = steps_per_epoch * o.epochs

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "jepa_config.json")

    global_step = 0
    for epoch in range(1, o.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_total = ep_pred = ep_sig = ep_div = 0.0
        n_batches = 0

        for start in range(0, n_train - bs + 1, bs):
            idx = perm[start:start + bs]
            src_ids, src_pad, tgt_ids, tgt_pad = _get_batch(dataset, idx)
            src_ids = src_ids.to(device)
            src_pad = src_pad.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_pad = tgt_pad.to(device)

            tau_g = gumbel_tau_at(global_step, total_steps, cfg.loss.verb)

            # warmup + cosine lr (manual; on online param group)
            factor = lr_factor_at(global_step, o.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = o.lr * factor

            out = model(
                src_ids, src_pad, tgt_ids, tgt_pad,
                gumbel_tau=tau_g, hard=False,
            )
            loss, comps = loss_fn(
                out["k"], out["verb_logits"], out["zhat"], out["z_target"],
                tau_g, False,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online_parameters(), o.grad_clip)
            optimizer.step()
            model.ema_update(cfg.ema.tau)  # AFTER optimizer.step()

            global_step += 1
            ep_total += float(loss.item())
            ep_pred += float(comps.get("L_pred", 0.0))
            ep_sig += float(comps.get("L_sigreg", 0.0))
            ep_div += float(comps.get("L_div", 0.0))
            n_batches += 1

        nb = max(1, n_batches)
        print(
            f"Epoch {epoch:4d} | loss {ep_total/nb:.4f} "
            f"L_pred={ep_pred/nb:.4f} L_sigreg={ep_sig/nb:.4f} L_div={ep_div/nb:.4f} "
            f"| tau_g={tau_g:.3f} lr={o.lr*factor:.2e}",
            flush=True,
        )

        save_checkpoint(model, cfg, global_step, epoch, out_dir)

        if epoch % cfg.eval.every_epochs == 0 or epoch == 1:
            model.eval()
            maybe_eval_diagnostics(model, dataset, device, cfg, epoch)

    print("Training complete.")
    save_checkpoint(model, cfg, global_step, o.epochs, out_dir)


def main():
    if len(sys.argv) < 2:
        print("usage: train_jepa.py <config.json>")
        raise SystemExit(2)
    train(sys.argv[1])


if __name__ == "__main__":
    main()
