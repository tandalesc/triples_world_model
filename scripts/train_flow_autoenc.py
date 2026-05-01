#!/usr/bin/env python3
"""Flow-matching dynamics on top of a frozen identity autoencoder.

v2 of the flow architecture. Difference from v1 (train_flow_dynamics.py):
v1 used the dual-AR base, whose sparse channel was trained on dynamics
outputs (different distribution than compressor outputs). v1 confirmed
the manifold-mismatch hypothesis on identity mode but plateaued because
the decoder's sparse projection rejected on-manifold samples in
advance/query modes.

v2 uses the identity autoencoder (compressor + single-memory ARDecoder,
chrF 95.9 on reconstruction). The decoder consumes the bottleneck
through a single memory_proj layer that was trained on compressor
outputs — exactly the distribution the flow learns to sample from.

Architecture:
    compressor(input)  ─┐
                        ├──▶ FlowDynamics ──▶ z_sampled ──▶ ARDecoder.generate
    mode_id ───────────┘                              (frozen autoencoder)

Frozen: compressor, ARDecoder, shared_token_emb.
Trained: FlowDynamics velocity field.

Usage:
    uv run python scripts/train_flow_autoenc.py configs/flow_autoenc_v2.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.chain_dataset import ChainDataset
from twm.flow_dynamics import FlowDynamics, flow_loss
from train_ar_autoencoder import AutoencoderModel  # noqa: E402

NUM_MODES = 3
MODE_NAMES = {0: "adv", 1: "qry", 2: "id"}


def build_autoencoder(base_cfg, tokenizer, device):
    model_config = ModelConfig(
        d_model=base_cfg.get("d_model", 128),
        n_heads=base_cfg.get("n_heads", 4),
        n_layers=2,
        d_ff=base_cfg.get("d_ff", 512),
        max_triples=base_cfg.get("max_triples", 16),
        dropout=base_cfg.get("dropout", 0.1),
    )
    return AutoencoderModel(
        config=model_config, tokenizer=tokenizer,
        compressor_layers=base_cfg.get("compressor_layers", 3),
        decoder_layers=base_cfg.get("decoder_layers", 2),
        d_ff=base_cfg.get("d_ff", 512),
        max_text_tokens=base_cfg.get("max_text_tokens", 128),
        dropout=base_cfg.get("dropout", 0.1),
        bottleneck_dim=base_cfg.get("bottleneck_dim"),
    ).to(device)


@torch.no_grad()
def encode_pairs(base_model, batch, device):
    """For each chain transition, return (z_prev, z_target, mode) triples.

    Latent teacher forcing across the chain: z_prev for step k+1 is
    compressor(target_k), not the flow's sampled z. This isolates the
    flow's learning signal step-by-step.
    """
    input_ids = batch["input_ids"].to(device)
    input_pad = batch["input_pad"].to(device)
    chain_ids = batch["chain_ids"].to(device)
    chain_pad = batch["chain_pad"].to(device)
    chain_len = batch["chain_len"].to(device)
    mode_ids = batch["mode_id"].to(device)

    z_prev = base_model.compressor(
        input_ids, input_pad, base_model.config.max_triples
    )
    if isinstance(z_prev, tuple):
        z_prev = z_prev[0]

    pairs = []
    max_chain = chain_len.max().item()

    for step in range(1, max_chain):
        active = chain_len > step
        if not active.any():
            break

        target_ids = chain_ids[active, step]
        target_pad = chain_pad[active, step]
        z_target = base_model.compressor(
            target_ids, target_pad, base_model.config.max_triples
        )
        if isinstance(z_target, tuple):
            z_target = z_target[0]

        pairs.append((
            z_prev[active].detach(),
            z_target.detach(),
            mode_ids[active],
        ))
        # Latent TF: next step's z_prev = compressor(target).
        z_prev_step = z_prev.clone()
        z_prev_step[active] = z_target
        z_prev = z_prev_step

    return pairs


def train_step(base_model, flow, batch, device, optimizer):
    pairs = encode_pairs(base_model, batch, device)
    if not pairs:
        return None, {}

    optimizer.zero_grad()
    total = 0.0
    n = 0
    per_mode = {0: [0.0, 0], 1: [0.0, 0], 2: [0.0, 0]}

    for z_prev, z_target, mode_ids in pairs:
        loss = flow_loss(flow, z_prev, z_target, mode_ids)
        loss.backward()
        total += loss.item()
        n += 1
        # Per-mode mse for diagnostic.
        with torch.no_grad():
            for m in per_mode:
                mask = mode_ids == m
                if mask.any():
                    sub_loss = flow_loss(flow, z_prev[mask], z_target[mask], mode_ids[mask])
                    per_mode[m][0] += sub_loss.item()
                    per_mode[m][1] += 1
    optimizer.step()
    avg = total / max(n, 1)
    by_mode = {
        MODE_NAMES[m]: (v[0] / v[1] if v[1] else float("nan"))
        for m, v in per_mode.items()
    }
    return avg, by_mode


@torch.no_grad()
def generation_eval(base_model, flow, test_loader, device, n_samples, n_ode_steps):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()
    refs, hyps, modes = [], [], []
    gen_correct = gen_total = 0
    mode_gen = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    n = 0

    tok = base_model.tokenizer
    pad_id = tok.pad_token_id

    for batch in test_loader:
        if n >= n_samples:
            break
        input_ids = batch["input_ids"].to(device)
        input_pad = batch["input_pad"].to(device)
        chain_ids = batch["chain_ids"].to(device)
        chain_pad = batch["chain_pad"].to(device)
        chain_len_b = batch["chain_len"].to(device)
        mode_ids_b = batch["mode_id"].to(device)

        z_prev = base_model.compressor(
            input_ids, input_pad, base_model.config.max_triples
        )
        if isinstance(z_prev, tuple):
            z_prev = z_prev[0]

        # Unroll flow across the chain (sampled trajectory at gen time).
        max_c = chain_len_b.max().item()
        for s in range(1, max_c):
            z_prev = flow.sample(z_prev, mode_ids_b, n_steps=n_ode_steps)

        for i in range(len(input_ids)):
            if n >= n_samples:
                break
            last = chain_len_b[i].item() - 1
            tgt_ids = chain_ids[i, last]
            tgt_pad = chain_pad[i, last]
            non_pad = ~tgt_pad

            gen_ids = base_model.decoder.generate(
                bottleneck=z_prev[i:i + 1],
                max_tokens=base_model.max_text_tokens,
            )

            ref = tok.decode(tgt_ids[non_pad].tolist())
            hyp = tok.decode(gen_ids[0].tolist())
            refs.append(ref)
            hyps.append(hyp)
            modes.append(mode_ids_b[i].item())

            gen_len = min(gen_ids.shape[1], tgt_ids.shape[0])
            mask = tgt_ids[:gen_len] != pad_id
            if mask.any():
                c = (gen_ids[0, :gen_len][mask] == tgt_ids[:gen_len][mask]).sum().item()
                gen_correct += c
                gen_total += mask.sum().item()
                m = mode_ids_b[i].item()
                mode_gen[m][0] += c
                mode_gen[m][1] += mask.sum().item()
            n += 1

    chrf_score = chrf_metric.corpus_score(hyps, [refs]).score
    gen_tok = gen_correct / max(gen_total, 1)
    per_mode_chrf = {}
    per_mode_tok = {}
    for m, name in MODE_NAMES.items():
        sel = [(h, r) for h, r, mm in zip(hyps, refs, modes) if mm == m]
        if sel:
            mh, mr = zip(*sel)
            per_mode_chrf[name] = chrf_metric.corpus_score(list(mh), [list(mr)]).score
        else:
            per_mode_chrf[name] = float("nan")
        mt = mode_gen[m]
        per_mode_tok[name] = mt[0] / mt[1] if mt[1] else float("nan")

    return {
        "chrf": chrf_score,
        "tok": gen_tok,
        "per_mode_chrf": per_mode_chrf,
        "per_mode_tok": per_mode_tok,
        "n": n,
        "examples": list(zip(refs[:5], hyps[:5], modes[:5])),
    }


@torch.no_grad()
def oracle_eval(base_model, test_loader, device, n_samples):
    """Decode compressor(target) directly. Tells us the upper ceiling
    the flow could possibly hit if it lands perfectly on the manifold."""
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()
    refs, hyps = [], []
    n = 0
    tok = base_model.tokenizer

    for batch in test_loader:
        if n >= n_samples:
            break
        chain_ids = batch["chain_ids"].to(device)
        chain_pad = batch["chain_pad"].to(device)
        chain_len_b = batch["chain_len"].to(device)
        for i in range(chain_ids.shape[0]):
            if n >= n_samples:
                break
            last = chain_len_b[i].item() - 1
            tgt_ids = chain_ids[i:i + 1, last]
            tgt_pad = chain_pad[i:i + 1, last]
            z = base_model.compressor(tgt_ids, tgt_pad, base_model.config.max_triples)
            if isinstance(z, tuple):
                z = z[0]
            gen = base_model.decoder.generate(
                bottleneck=z, max_tokens=base_model.max_text_tokens
            )
            ref = tok.decode(tgt_ids[0][~tgt_pad[0]].tolist())
            hyp = tok.decode(gen[0].tolist())
            refs.append(ref)
            hyps.append(hyp)
            n += 1
    return chrf_metric.corpus_score(hyps, [refs]).score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device(cfg.get("device", "cuda"))

    base_dir = Path(cfg["base_run_dir"])
    with open(base_dir / "config.json") as f:
        base_cfg = json.load(f)

    if base_cfg.get("tokenizer_pretrained"):
        tokenizer = DomainBPETokenizer.from_pretrained(
            base_cfg["tokenizer_pretrained"],
            max_length=base_cfg.get("max_text_tokens", 128),
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            base_dir / "tokenizer.json",
            max_length=base_cfg.get("max_text_tokens", 128),
        )

    train_ds = ChainDataset(
        cfg["train_data"], tokenizer,
        max_text_tokens=base_cfg.get("max_text_tokens", 128),
    )
    test_ds = ChainDataset(
        cfg["test_data"], tokenizer,
        max_text_tokens=base_cfg.get("max_text_tokens", 128),
    )
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    base_model = build_autoencoder(base_cfg, tokenizer, device)
    base_state = torch.load(base_dir / "weights.pt", map_location=device)
    base_model.load_state_dict(base_state)
    for p in base_model.parameters():
        p.requires_grad_(False)
    base_model.eval()
    print(f"Loaded frozen autoencoder from {base_dir}")

    # Sanity: oracle ceiling for the frozen decoder.
    print("Computing oracle ceiling (decode compressor(target) directly)...")
    oracle_chrf = oracle_eval(base_model, test_loader, device, n_samples=200)
    print(f"Oracle chrF (compressor->decoder, no flow): {oracle_chrf:.1f}")
    print("This is the theoretical ceiling for the flow + frozen decoder pipeline.")
    print()

    bn_d = base_cfg.get("bottleneck_dim") or base_cfg.get("d_model", 128)
    flow = FlowDynamics(
        d_model=bn_d,
        n_heads=cfg.get("flow_n_heads", max(1, bn_d // 16)),
        n_layers=cfg.get("flow_n_layers", 4),
        num_modes=NUM_MODES,
        max_triples=base_cfg.get("max_triples", 16),
        d_ff=cfg.get("flow_d_ff", bn_d * 4),
        dropout=cfg.get("flow_dropout", 0.1),
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"FlowDynamics: {n_params:,} trainable params")

    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    log_file = open(out_dir / "train.log", "w")
    best_chrf = -1.0
    n_ode_steps = cfg.get("ode_steps", 10)

    for epoch in range(cfg.get("epochs", 30)):
        flow.train()
        running = 0.0
        n_batches = 0
        running_per_mode = {n: [0.0, 0] for n in MODE_NAMES.values()}
        for batch in train_loader:
            avg, by_mode = train_step(base_model, flow, batch, device, optimizer)
            if avg is None:
                continue
            running += avg
            n_batches += 1
            for name, val in by_mode.items():
                if val == val:  # not nan
                    running_per_mode[name][0] += val
                    running_per_mode[name][1] += 1
        avg_loss = running / max(n_batches, 1)
        per_mode_loss = {
            n: (s / c if c else float("nan"))
            for n, (s, c) in running_per_mode.items()
        }

        log_line = (
            f"epoch {epoch:3d} | flow_mse {avg_loss:.4f}"
            f" (adv {per_mode_loss['adv']:.3f}"
            f" qry {per_mode_loss['qry']:.3f}"
            f" id {per_mode_loss['id']:.3f})"
        )

        if (epoch + 1) % cfg.get("eval_every", 3) == 0 or epoch == 0:
            flow.eval()
            res = generation_eval(
                base_model, flow, test_loader, device,
                n_samples=cfg.get("gen_samples", 200),
                n_ode_steps=n_ode_steps,
            )
            c = res["per_mode_chrf"]
            t = res["per_mode_tok"]
            log_line += (
                f" | gen chrF {res['chrf']:.1f} tok {res['tok']:.3f}"
                f" | adv {c['adv']:.1f}/{t['adv']:.3f}"
                f" qry {c['qry']:.1f}/{t['qry']:.3f}"
                f" id {c['id']:.1f}/{t['id']:.3f}"
            )
            for ref, hyp, m in res["examples"][:2]:
                print(f"  [{MODE_NAMES[m]}] ref: {ref[:90]}")
                print(f"  [{MODE_NAMES[m]}] gen: {hyp[:90]}")
            if res["chrf"] > best_chrf:
                best_chrf = res["chrf"]
                torch.save(flow.state_dict(), out_dir / "flow_best.pt")
                log_line += f"  saved (best {best_chrf:.1f})"

        print(log_line, flush=True)
        log_file.write(log_line + "\n")
        log_file.flush()

    torch.save(flow.state_dict(), out_dir / "flow_final.pt")
    print(f"\nDone. Best gen chrF: {best_chrf:.1f}")
    print(f"Oracle ceiling (compressor->decoder): {oracle_chrf:.1f}")
    print(f"v1 dual-AR base flow:                 best chrF ~33 (id 51 ceiling)")
    print(f"AR baseline (single memory):          chrF 49.0")
    log_file.close()


if __name__ == "__main__":
    main()
