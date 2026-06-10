"""Train the JEPA v2 latent-action world model (design §1/§5/§8/§10).

Usage:
    uv run python scripts/train_jepa_v2.py configs/jepa_nano_v2.json

Loop (design §5/§8): per batch — forward (Gumbel-ST hard posterior action) ->
JEPALossV2 (L_token primary + L_prior + L_sigreg + L_pred aux) -> backward ->
clip_grad_norm_(online, grad_clip) -> optimizer.step() -> model.ema_update(τ_ema) ->
anneal posterior τ -> periodic eval_diagnostics_v2 (generated text samples + v-ablation
CE gap + hard-neg MRR regression). AdamW + linear-warmup-then-cosine on ONLINE params
only; EMA params excluded from the optimizer and grad clipping.

PER-EVAL CHECKPOINT RETENTION (design §11 / explicit task requirement): a distinct
`model_ep{N}.pt` is written every eval epoch (NOT just an overwritten model_latest.pt).
This bug bit v1 twice — the only checkpoint kept was the last one, so a collapse in the
final epochs erased every recoverable earlier state. v2 keeps every eval checkpoint.

This script composes the v2 model/loss/diagnostics through the frozen contracts; the
sibling modules (transition, decoder, losses_v2, diagnostics_v2) are import-guarded so
training survives a sibling being mid-build.
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
from twm.jepa.model_v2 import build_jepa_model_v2


def resolve_device(device_str: str | None = None) -> torch.device:
    """Device order: cuda -> mps -> cpu (CLAUDE.md convention)."""
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
    """Annealed posterior Gumbel temperature τ: start -> end over the first
    `anneal_frac` of training, then held at `end` (design §2.4: 3.0 -> 1.0 over 50%)."""
    anneal_steps = max(1, int(vc.anneal_frac * total_steps))
    if step >= anneal_steps:
        return vc.gumbel_tau_end
    frac = step / anneal_steps
    return vc.gumbel_tau_start + (vc.gumbel_tau_end - vc.gumbel_tau_start) * frac


def lr_factor_at(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 0 — multiplicative on base lr (v1 convention)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_token_emb(vocab_size: int, d_model: int) -> nn.Embedding:
    """Frozen domain-BPE ENCODER token embedding (design §2; NOT trained). The token
    decoder owns a SEPARATE, trainable embedding (§9 budget)."""
    emb = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(emb.weight, std=0.02)
    emb.weight.requires_grad_(False)
    return emb


def build_loss_v2(cfg, operator):
    """Construct JEPALossV2 (Task D). Filters kwargs to what the class accepts."""
    from twm.jepa import JEPALossV2
    import inspect

    lc = cfg.loss
    candidate = dict(
        operator=operator,
        w_token=lc.w_token,
        w_prior=lc.w_prior,
        w_sigreg=lc.w_sigreg,
        w_pred=lc.w_pred,
        n_slices=lc.sigreg.n_slices,
        n_knots=lc.sigreg.n_knots,
        knot_max=lc.sigreg.knot_max,
        standardize=lc.sigreg.standardize,
        pad_id=0,
        n_verbs=cfg.model.n_verbs,
    )
    sig = inspect.signature(JEPALossV2.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return JEPALossV2(**candidate)
    accepted = {n for n in sig.parameters if n != "self"}
    return JEPALossV2(**{k: v for k, v in candidate.items() if k in accepted})


def _get_batch(dataset, idx: torch.Tensor):
    """Return (src_ids, src_pad, tgt_ids, tgt_pad) for an index tensor via the
    contiguous-tensor layout (design §6 storage); fall back to per-item stacking."""
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


def maybe_eval_diagnostics(model, dataset, device, cfg, epoch, tokenizer):
    """Run §8 v2 diagnostics if the module is available (import-guarded so training
    survives diagnostics being mid-build)."""
    try:
        from twm.jepa.diagnostics_v2 import eval_diagnostics_v2
    except Exception as e:
        print(f"  [diagnostics_v2 skipped: {e}]", flush=True)
        return
    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = dict(
        model=model,
        dataset=dataset,
        device=device,
        n_examples=cfg.eval.n_examples,
        out_dir=str(out_dir),
        n_text_samples=cfg.eval.n_text_samples,
        temperatures=cfg.eval.temperatures,
        tokenizer=tokenizer,
        epoch=epoch,
    )
    try:
        import inspect

        sig = inspect.signature(eval_diagnostics_v2)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            accepted = {n for n in sig.parameters}
            candidate = {k: v for k, v in candidate.items() if k in accepted}
        metrics = eval_diagnostics_v2(**candidate)
        if isinstance(metrics, dict):
            flat = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            )
            print(f"  diag_v2[ep{epoch}] {flat}", flush=True)
    except Exception as e:
        print(f"  [diagnostics_v2 error: {e}]", flush=True)


def save_checkpoint(model, cfg, step, epoch, out_dir: Path, tag: str = "latest"):
    """Save a checkpoint. tag='latest' overwrites model_latest.pt; tag='ep{N}' keeps a
    per-eval checkpoint (design §11 per-eval retention — never overwrite earlier evals)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(ckpt, out_dir / f"model_{tag}.pt")


def train(config_path: str):
    cfg = JEPAConfig.from_json(config_path)
    seed_everything(cfg.seed)
    device = resolve_device()

    print("=== JEPA v2 latent-action training ===")
    print(f"Config: {config_path}")
    print(f"Profile: {cfg.profile}  device: {device}")

    # ---- data (append_eos=True so the AR decoder learns to stop; design §7) ----
    from twm.jepa.data import JEPAChainDataset
    from twm.domain_bpe import DomainBPETokenizer

    tokenizer = DomainBPETokenizer.load(
        cfg.data.tokenizer, max_length=cfg.data.max_text_tokens
    )
    dataset = JEPAChainDataset(
        path=cfg.data.path,
        tokenizer=tokenizer,
        max_text_tokens=cfg.data.max_text_tokens,
        append_eos=cfg.data.append_eos,
    )
    n_train = len(dataset)
    if getattr(cfg.data, "max_chains", None):
        cap = int(cfg.data.max_chains) * 2  # chain len 3 -> 2 adjacent pairs each
        if cap < n_train:
            dataset._src_ids = dataset._src_ids[:cap].contiguous()
            dataset._src_pad = dataset._src_pad[:cap].contiguous()
            dataset._tgt_ids = dataset._tgt_ids[:cap].contiguous()
            dataset._tgt_pad = dataset._tgt_pad[:cap].contiguous()
            dataset._src_texts = dataset._src_texts[:cap]
            dataset._tgt_texts = dataset._tgt_texts[:cap]
            n_train = len(dataset)
    print(f"Dataset: {n_train} pairs (append_eos={cfg.data.append_eos})")

    # ---- model ----
    token_emb = build_token_emb(cfg.data.vocab_size, cfg.model.d_model)
    model = build_jepa_model_v2(cfg, token_emb).to(device)
    loss_fn = build_loss_v2(cfg, model.operator).to(device)

    n_online = model.trainable_param_count()
    print(f"Online (trainable) params: {n_online:,}")
    if n_online > 250_000:
        print(f"  WARNING: trainable params {n_online:,} exceed the nano-v2 budget (250K).")

    # ---- optimizer (ONLINE params only; dedup-by-id so a loss-owned param lands once) ----
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
    optimizer = torch.optim.AdamW(opt_params, lr=o.lr, weight_decay=o.weight_decay)

    bs = o.batch_size
    steps_per_epoch = max(1, n_train // bs)
    total_steps = steps_per_epoch * o.epochs

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "jepa_v2_config.json")

    global_step = 0
    factor = 1.0
    tau = cfg.loss.verb.gumbel_tau_start
    for epoch in range(1, o.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_total = ep_token = ep_prior = ep_sig = ep_pred = 0.0
        n_batches = 0

        for start in range(0, n_train - bs + 1, bs):
            idx = perm[start:start + bs]
            src_ids, src_pad, tgt_ids, tgt_pad = _get_batch(dataset, idx)
            src_ids = src_ids.to(device)
            src_pad = src_pad.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_pad = tgt_pad.to(device)

            tau = gumbel_tau_at(global_step, total_steps, cfg.loss.verb)

            factor = lr_factor_at(global_step, o.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = o.lr * factor

            # Hard ST one-hot posterior action (design §6 L3: bounds future->decoder bits).
            out = model(src_ids, src_pad, tgt_ids, tgt_pad, tau=tau, hard=True)
            loss, comps = loss_fn(
                logits=out["logits"],
                tgt_ids=tgt_ids,
                tgt_pad=tgt_pad,
                k=out["k"],
                v_logits=out["v_logits"],
                p_logits=out["p_logits"],
                zhat=out["zhat"],
                z_target=out["z_target"],
                tau=tau,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.online_parameters(), o.grad_clip)
            optimizer.step()
            model.ema_update(cfg.ema.tau)  # AFTER optimizer.step()

            global_step += 1
            ep_total += float(loss.item())
            ep_token += float(comps.get("L_token", 0.0))
            ep_prior += float(comps.get("L_prior", 0.0))
            ep_sig += float(comps.get("L_sigreg", 0.0))
            ep_pred += float(comps.get("L_pred", 0.0))
            n_batches += 1

        nb = max(1, n_batches)
        print(
            f"Epoch {epoch:4d} | loss {ep_total/nb:.4f} "
            f"L_token={ep_token/nb:.4f} L_prior={ep_prior/nb:.4f} "
            f"L_sigreg={ep_sig/nb:.4f} L_pred={ep_pred/nb:.4f} "
            f"| tau={tau:.3f} lr={o.lr*factor:.2e}",
            flush=True,
        )

        # Always overwrite the rolling 'latest'; ADDITIONALLY keep a per-eval snapshot.
        save_checkpoint(model, cfg, global_step, epoch, out_dir, tag="latest")

        if epoch % cfg.eval.every_epochs == 0 or epoch == 1:
            # Per-eval checkpoint retention (design §11) — distinct file, never clobbered.
            save_checkpoint(model, cfg, global_step, epoch, out_dir, tag=f"ep{epoch}")
            model.eval()
            maybe_eval_diagnostics(model, dataset, device, cfg, epoch, tokenizer)

    print("Training complete.")
    save_checkpoint(model, cfg, global_step, o.epochs, out_dir, tag="latest")
    save_checkpoint(model, cfg, global_step, o.epochs, out_dir, tag=f"ep{o.epochs}")


def main():
    if len(sys.argv) < 2:
        print("usage: train_jepa_v2.py <config.json>")
        raise SystemExit(2)
    train(sys.argv[1])


if __name__ == "__main__":
    main()
