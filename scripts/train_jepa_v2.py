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
sibling modules (transition, decoder, losses, diagnostics) are import-guarded so
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
from twm.jepa.model import build_jepa_model_v2


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
        # v3 InfoNCE (design §1.7): w_nce takes over w_pred's slot; nce_temperature τ=0.1.
        # Defaults (0.0 / 0.1) reproduce v2.1 — the loss skips the matmul when w_nce=0.
        w_nce=getattr(lc, "w_nce", 0.0),
        nce_temperature=getattr(getattr(lc, "nce", None), "temperature", 0.1),
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


def _get_triple_batch(dataset, idx: torch.Tensor):
    """Return (s0_ids, s0_pad, s1_ids, s1_pad, s2_ids, s2_pad, chain_ids) for an index
    tensor in triple mode (design §2.1). chain_ids is a (B,) long tensor of originating
    chain ids — one per example — for InfoNCE same-chain bookkeeping."""
    s0i, s0p = dataset._s0_ids[idx], dataset._s0_pad[idx]
    s1i, s1p = dataset._s1_ids[idx], dataset._s1_pad[idx]
    s2i, s2p = dataset._s2_ids[idx], dataset._s2_pad[idx]
    cids = torch.tensor(
        [dataset._chain_ids[int(i)] for i in idx], dtype=torch.long
    )
    return s0i, s0p, s1i, s1p, s2i, s2p, cids


def chain_contiguous_perm(chain_ids: list[int], generator=None) -> torch.Tensor:
    """Chain-grouped shuffle (design §1.5): shuffle at the CHAIN level, then flatten so a
    chain's sibling examples are adjacent in the permutation. With batch_size a multiple of
    the per-chain example count (64 % 2 == 0 for pairs), siblings co-occur in one batch, so
    the in-batch (B,B) InfoNCE matrix already contains the same-chain hard negative.

    This does NOT change the loss math — only which negatives populate the matrix. For
    triple mode the hard negative is per-example (s1 vs s2), so contiguity is not required;
    we still group by chain for consistency and reproducibility.
    """
    # Map chain_id -> list of dataset indices belonging to it, in first-seen chain order.
    order: list[int] = []
    buckets: dict[int, list[int]] = {}
    for i, cid in enumerate(chain_ids):
        if cid not in buckets:
            buckets[cid] = []
            order.append(cid)
        buckets[cid].append(i)
    # Shuffle the chain order (not the within-chain item order, which stays contiguous).
    perm_chains = torch.randperm(len(order), generator=generator).tolist()
    flat: list[int] = []
    for ci in perm_chains:
        flat.extend(buckets[order[ci]])
    return torch.tensor(flat, dtype=torch.long)


def _pair_step(model, loss_fn, dataset, idx, device, tau, w_nce):
    """One v2 pairs-mode training step. When w_nce==0 this is BITWISE the v2.1 step
    (no chain_ids passed → loss skips InfoNCE). When w_nce>0, same-chain negatives are
    enabled via the batch's chain_ids (the in-batch (B,B) matrix already holds the hard
    negative thanks to chain-contiguous batching, design §1.5)."""
    src_ids, src_pad, tgt_ids, tgt_pad = _get_batch(dataset, idx)
    src_ids = src_ids.to(device)
    src_pad = src_pad.to(device)
    tgt_ids = tgt_ids.to(device)
    tgt_pad = tgt_pad.to(device)

    chain_ids = None
    if w_nce > 0:
        chain_ids = torch.tensor(
            [dataset._chain_ids[int(i)] for i in idx], dtype=torch.long, device=device
        )

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
        chain_ids=chain_ids,
    )
    return loss, comps


def _unroll_step(model, loss_fn, dataset, idx, device, tau, hop_weights):
    """One v3 triple-mode two-hop unroll step (design §2.3/§2.4).

    Calls model.forward_unroll (Task C) to get per-hop outputs, then assembles the total
    loss = Σ_h hop_weights[h] · loss_fn(hop_h). The cross-hop hard negative is supplied
    per example (design §2.4): for the hop-1 anchor, z_target of hop 2 (= EMA pool of s2)
    is the same-chain negative; for the hop-2 anchor, z_target of hop 1 (= EMA pool of s1)
    is. These enter as nce_neg_keys (B, 1, dn). The loss is called ONCE PER HOP and summed
    here with the hop weights (design §6 Task B: hops live in the trainer, the loss is
    hop-agnostic)."""
    s0i, s0p, s1i, s1p, s2i, s2p, cids = _get_triple_batch(dataset, idx)
    s0i, s0p = s0i.to(device), s0p.to(device)
    s1i, s1p = s1i.to(device), s1p.to(device)
    s2i, s2p = s2i.to(device), s2p.to(device)
    cids = cids.to(device)

    hops = model.forward_unroll(s0i, s0p, s1i, s1p, s2i, s2p, tau=tau, hard=True)
    hop_tgt = [(s1i, s1p), (s2i, s2p)]

    # Cross-hop hard negatives: hop h's negative is the OTHER hop's z_target (the EMA pool
    # of the sibling future), shaped (B, 1, dn). Only available when use_pred built the EMA
    # head (z_target not None). Guarded so a model without the anchor head still runs.
    z_targets = [h.get("z_target") for h in hops]
    neg_for_hop = [None, None]
    if all(z is not None for z in z_targets):
        neg_for_hop[0] = z_targets[1].unsqueeze(1)  # hop-1 anchor: s2 pool is the hard neg
        neg_for_hop[1] = z_targets[0].unsqueeze(1)  # hop-2 anchor: s1 pool is the hard neg

    total = None
    agg: dict = {}
    for h, (out, (tids, tpad)) in enumerate(zip(hops, hop_tgt)):
        wt = hop_weights[h] if h < len(hop_weights) else 1.0
        loss_h, comps_h = loss_fn(
            logits=out["logits"],
            tgt_ids=tids,
            tgt_pad=tpad,
            k=out["k"],
            v_logits=out["v_logits"],
            p_logits=out["p_logits"],
            zhat=out["zhat"],
            z_target=out["z_target"],
            tau=tau,
            chain_ids=cids,
            nce_neg_keys=neg_for_hop[h],
        )
        total = wt * loss_h if total is None else total + wt * loss_h
        # Aggregate components for logging: weighted sums (matching `total`) plus per-hop CE.
        for key in ("L_token", "L_prior", "L_sigreg", "L_pred", "L_nce"):
            agg[key] = agg.get(key, 0.0) + wt * float(comps_h.get(key, 0.0))
        agg[f"L_token_h{h + 1}"] = float(comps_h.get("L_token", 0.0))

    return total, agg


class _TripleHop1View:
    """Pairs-compatible view of a triple-mode dataset for the v2 diagnostics harness.

    `eval_diagnostics_v2` expects `dataset[idx] -> {src_ids,src_pad,tgt_ids,tgt_pad}`
    (the v2 pair interface) plus `.tokenizer` and `.chain_ids`. Triple mode stores
    `s0/s1/s2`, so this thin adapter surfaces the hop-1 pair (s0 -> s1) — the single-step
    transition the v2 diagnostics were written against. Diagnostics is owned by another
    task; this adapter lives in the train script so triple-mode runs can still report
    the same CE-gap / hard-neg-MRR metrics without touching diagnostics.py."""

    def __init__(self, ds):
        self._ds = ds
        self.tokenizer = ds.tokenizer
        self.chain_ids = ds._chain_ids

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return {
            "src_ids": self._ds._s0_ids[idx],
            "src_pad": self._ds._s0_pad[idx],
            "tgt_ids": self._ds._s1_ids[idx],
            "tgt_pad": self._ds._s1_pad[idx],
        }


def maybe_eval_diagnostics(model, dataset, device, cfg, epoch, tokenizer):
    """Run §8 v2 diagnostics if the module is available (import-guarded so training
    survives diagnostics being mid-build)."""
    # Triple-mode datasets store s0/s1/s2; wrap as the hop-1 (s0->s1) pair view the v2
    # diagnostics expect.
    if getattr(dataset, "mode", "pairs") == "triples":
        dataset = _TripleHop1View(dataset)
    try:
        from twm.jepa.diagnostics import eval_diagnostics_v2
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
        if not isinstance(metrics, dict):
            metrics = {}

        # Entity-world diagnostics hook (campaign §3.5): only when enabled. The labeled
        # splits are loaded directly by eval_entity_world (not the train dataset), so this
        # uses `cfg` + the real tokenizer rather than the wrapped hop-1 view above.
        ew_cfg = getattr(cfg.eval, "entity_world", None)
        if ew_cfg is not None and getattr(ew_cfg, "enabled", False):
            try:
                from twm.jepa.diagnostics import eval_entity_world

                ent_metrics = eval_entity_world(
                    model, ew_cfg, device, tokenizer,
                    max_text_tokens=cfg.data.max_text_tokens,
                    out_dir=str(out_dir), epoch=epoch,
                    append_eos=getattr(cfg.data, "append_eos", True),
                )
                metrics.update(ent_metrics)
            except Exception as e:
                print(f"  [entity_world eval error: {e}]", flush=True)

        if metrics:
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
    mode = getattr(cfg.data, "mode", "pairs")
    dataset = JEPAChainDataset(
        path=cfg.data.path,
        tokenizer=tokenizer,
        max_text_tokens=cfg.data.max_text_tokens,
        append_eos=cfg.data.append_eos,
        mode=mode,
    )
    n_train = len(dataset)
    if getattr(cfg.data, "max_chains", None):
        n_chains = int(cfg.data.max_chains)
        if mode == "triples":
            # Triple mode: one example per chain — cap is the chain count directly.
            cap = n_chains
            if cap < n_train:
                dataset._s0_ids = dataset._s0_ids[:cap].contiguous()
                dataset._s0_pad = dataset._s0_pad[:cap].contiguous()
                dataset._s1_ids = dataset._s1_ids[:cap].contiguous()
                dataset._s1_pad = dataset._s1_pad[:cap].contiguous()
                dataset._s2_ids = dataset._s2_ids[:cap].contiguous()
                dataset._s2_pad = dataset._s2_pad[:cap].contiguous()
                dataset._s0_texts = dataset._s0_texts[:cap]
                dataset._s1_texts = dataset._s1_texts[:cap]
                dataset._s2_texts = dataset._s2_texts[:cap]
                dataset._chain_ids = dataset._chain_ids[:cap]
                n_train = len(dataset)
        else:
            cap = n_chains * 2  # chain len 3 -> 2 adjacent pairs each
            if cap < n_train:
                dataset._src_ids = dataset._src_ids[:cap].contiguous()
                dataset._src_pad = dataset._src_pad[:cap].contiguous()
                dataset._tgt_ids = dataset._tgt_ids[:cap].contiguous()
                dataset._tgt_pad = dataset._tgt_pad[:cap].contiguous()
                dataset._src_texts = dataset._src_texts[:cap]
                dataset._tgt_texts = dataset._tgt_texts[:cap]
                # Keep chain_ids aligned with the truncated dataset (design §8.2).
                dataset._chain_ids = dataset._chain_ids[:cap]
                n_train = len(dataset)
    unit = "triples (chains)" if mode == "triples" else "pairs"
    print(f"Dataset: {n_train} {unit} (mode={mode}, append_eos={cfg.data.append_eos})")
    if mode == "triples" and getattr(dataset, "n_skipped", 0):
        print(f"  triple mode skipped {dataset.n_skipped} chains of length < 3")

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

    # v3 (design §1.5): chain-contiguous batching guarantees same-chain hard negatives
    # for InfoNCE. It is only needed when InfoNCE is on (w_nce>0) in PAIRS mode; when
    # w_nce==0 (v2.1) we keep the original `randperm` so the behavior-preservation gate
    # holds bitwise. In TRIPLE mode the hard negative is per-example (s1 vs s2), so the
    # ordering does not affect correctness, but we still group by chain for consistency.
    w_nce = getattr(cfg.loss, "w_nce", 0.0)
    use_chain_contig = (mode == "triples") or (w_nce > 0)
    hop_weights = list(getattr(getattr(cfg.loss, "unroll", None), "hop_weights", [1.0, 0.5]))

    global_step = 0
    factor = 1.0
    tau = cfg.loss.verb.gumbel_tau_start
    for epoch in range(1, o.epochs + 1):
        model.train()
        if use_chain_contig:
            perm = chain_contiguous_perm(dataset._chain_ids)
        else:
            perm = torch.randperm(n_train)
        ep_total = ep_token = ep_prior = ep_sig = ep_pred = ep_nce = 0.0
        ep_tok_h1 = ep_tok_h2 = 0.0
        n_batches = 0

        for start in range(0, n_train - bs + 1, bs):
            idx = perm[start:start + bs]

            tau = gumbel_tau_at(global_step, total_steps, cfg.loss.verb)
            factor = lr_factor_at(global_step, o.warmup_steps, total_steps)
            for g in optimizer.param_groups:
                g["lr"] = o.lr * factor

            if mode == "triples":
                loss, comps = _unroll_step(
                    model, loss_fn, dataset, idx, device, tau, hop_weights
                )
            else:
                loss, comps = _pair_step(
                    model, loss_fn, dataset, idx, device, tau, w_nce
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
            ep_nce += float(comps.get("L_nce", 0.0))
            ep_tok_h1 += float(comps.get("L_token_h1", 0.0))
            ep_tok_h2 += float(comps.get("L_token_h2", 0.0))
            n_batches += 1

        nb = max(1, n_batches)
        line = (
            f"Epoch {epoch:4d} | loss {ep_total/nb:.4f} "
            f"L_token={ep_token/nb:.4f} L_prior={ep_prior/nb:.4f} "
            f"L_sigreg={ep_sig/nb:.4f} L_pred={ep_pred/nb:.4f} L_nce={ep_nce/nb:.4f} "
        )
        if mode == "triples":
            line += f"L_tok_h1={ep_tok_h1/nb:.4f} L_tok_h2={ep_tok_h2/nb:.4f} "
        line += f"| tau={tau:.3f} lr={o.lr*factor:.2e}"
        print(line, flush=True)

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
