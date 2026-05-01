#!/usr/bin/env python3
"""Beam-search diagnostic for dual-AR runs.

Compares greedy vs beam(k) decoding on a saved DualARModel checkpoint.
Probes whether greedy decoding is leaving signal on the table at the
chrF~49 TextWorld advance wall — no retraining required.

Usage:
  uv run python scripts/eval_beam_search.py \\
    --run-dir results/dual_ar_v4_vq_entity \\
    --test-data data/tw_all_test.jsonl \\
    --beams 1 4 8 \\
    --n-samples 200
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.chain_dataset import ChainDataset

# Reuse the wrapper class defined in the trainer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dual_ar import DualARModel  # noqa: E402


MODE_NAMES = {0: "adv", 1: "qry", 2: "id"}


def build_model(cfg, tokenizer, device):
    model_config = ModelConfig(
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("dynamics_layers", 2),
        d_ff=cfg.get("d_ff", 512),
        max_triples=cfg.get("max_triples", 16),
        dropout=cfg.get("dropout", 0.1),
    )
    model = DualARModel(
        config=model_config,
        tokenizer=tokenizer,
        compressor_layers=cfg.get("compressor_layers", 3),
        dynamics_layers=cfg.get("dynamics_layers", 2),
        decoder_layers=cfg.get("decoder_layers", 3),
        d_ff=cfg.get("d_ff", 512),
        max_text_tokens=cfg.get("max_text_tokens", 128),
        dropout=cfg.get("dropout", 0.1),
        bottleneck_dim=cfg.get("bottleneck_dim"),
        dense_dropout=cfg.get("dense_dropout", 0.3),
        vq_enabled=cfg.get("vq_enabled", False),
        vq_num_codes=cfg.get("vq_num_codes", 1024),
        vq_beta=cfg.get("vq_beta", 0.25),
        vq_entity_only=cfg.get("vq_entity_only", False),
    ).to(device)
    return model


def decode_one(model, batch, device, strategy, num_beams, length_penalty, n_samples):
    """Run dynamics + decoder over the test batch, returning hyps/refs/per-mode hits."""
    refs, hyps, modes = [], [], []
    gen_correct = gen_total = 0
    mode_gen = {0: [0, 0], 1: [0, 0], 2: [0, 0]}

    n = 0
    for b in batch:
        if n >= n_samples:
            break
        input_ids = b["input_ids"].to(device)
        input_pad = b["input_pad"].to(device)
        chain_ids = b["chain_ids"].to(device)
        chain_pad = b["chain_pad"].to(device)
        chain_len_b = b["chain_len"].to(device)
        mode_ids_b = b["mode_id"].to(device)

        compressor_out = model.compress(input_ids, input_pad)
        if isinstance(compressor_out, tuple):
            compressor_out = compressor_out[0]

        bn = compressor_out
        max_c = chain_len_b.max().item()
        for s in range(1, max_c):
            bn = model.forward_dynamics(bn, mode_ids_b)
            bn, _, _ = model.quantize(bn)

        dense_pred = model.dense_proj(bn)

        for i in range(len(input_ids)):
            if n >= n_samples:
                break
            last = chain_len_b[i].item() - 1
            tgt_ids = chain_ids[i, last]
            tgt_pad = chain_pad[i, last]
            non_pad = ~tgt_pad

            if strategy == "greedy":
                gen_ids = model.decoder.generate(
                    dynamics_out=bn[i:i + 1],
                    compressor_out=dense_pred[i:i + 1],
                    max_tokens=model.max_text_tokens,
                )
            else:
                gen_ids = model.decoder.generate_beam(
                    dynamics_out=bn[i:i + 1],
                    compressor_out=dense_pred[i:i + 1],
                    num_beams=num_beams,
                    max_tokens=model.max_text_tokens,
                    length_penalty=length_penalty,
                )

            ref = model.tokenizer.decode(tgt_ids[non_pad].tolist())
            hyp = model.tokenizer.decode(gen_ids[0].tolist())
            refs.append(ref)
            hyps.append(hyp)
            modes.append(mode_ids_b[i].item())

            gen_len = min(gen_ids.shape[1], tgt_ids.shape[0])
            mask = tgt_ids[:gen_len] != model.tokenizer.pad_token_id
            if mask.any():
                c = (gen_ids[0, :gen_len][mask] == tgt_ids[:gen_len][mask]).sum().item()
                gen_correct += c
                gen_total += mask.sum().item()
                m = mode_ids_b[i].item()
                mode_gen[m][0] += c
                mode_gen[m][1] += mask.sum().item()
            n += 1

    return refs, hyps, modes, gen_correct, gen_total, mode_gen


@torch.no_grad()
def evaluate(model, test_loader, device, strategy, num_beams, length_penalty, n_samples):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()

    # Materialize batches once — same data across all strategies.
    batches = []
    n = 0
    for b in test_loader:
        batches.append(b)
        n += b["input_ids"].shape[0]
        if n >= n_samples:
            break

    refs, hyps, modes, gen_correct, gen_total, mode_gen = decode_one(
        model, batches, device, strategy, num_beams, length_penalty, n_samples
    )

    chrf = chrf_metric.corpus_score(hyps, [refs]).score
    tok = gen_correct / max(gen_total, 1)
    per_mode_chrf = {}
    for m, name in MODE_NAMES.items():
        sel = [(h, r) for h, r, mm in zip(hyps, refs, modes) if mm == m]
        if sel:
            mh, mr = zip(*sel)
            per_mode_chrf[name] = chrf_metric.corpus_score(list(mh), [list(mr)]).score
        else:
            per_mode_chrf[name] = float("nan")
    per_mode_tok = {
        MODE_NAMES[m]: (mode_gen[m][0] / mode_gen[m][1] if mode_gen[m][1] else float("nan"))
        for m in MODE_NAMES
    }

    return {
        "chrf": chrf,
        "tok": tok,
        "per_mode_chrf": per_mode_chrf,
        "per_mode_tok": per_mode_tok,
        "n": len(refs),
        "refs": refs,
        "hyps": hyps,
        "modes": modes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--test-data", required=True)
    p.add_argument("--beams", nargs="+", type=int, default=[1, 4, 8],
                   help="Beam widths to compare. 1 = greedy.")
    p.add_argument("--length-penalty", type=float, default=1.0,
                   help="GNMT length norm exponent. 1.0 = none.")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--show-examples", type=int, default=8)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    tok_path = run_dir / "tokenizer.json"
    if cfg.get("tokenizer_pretrained"):
        # Same call as training: preserves normalize=False for GPT-2.
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"],
            max_length=cfg.get("max_text_tokens", 128),
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            tok_path, max_length=cfg.get("max_text_tokens", 128)
        )

    test_ds = ChainDataset(
        args.test_data, tokenizer,
        max_text_tokens=cfg.get("max_text_tokens", 128),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    model = build_model(cfg, tokenizer, device)
    state = torch.load(run_dir / "weights.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Run: {run_dir}")
    print(f"Test: {args.test_data}  ({len(test_ds)} chains)")
    print(f"Samples: {args.n_samples}  length_penalty: {args.length_penalty}")
    print()

    results = {}
    for k in args.beams:
        strategy = "greedy" if k == 1 else "beam"
        label = "greedy" if k == 1 else f"beam-{k}"
        print(f"  evaluating {label}...", flush=True)
        results[label] = evaluate(
            model, test_loader, device,
            strategy, k, args.length_penalty, args.n_samples,
        )

    # Summary table
    print()
    print(f"{'strategy':<10} {'chrF':>6} {'tok':>6}   "
          f"{'chrF_adv':>8} {'chrF_qry':>8} {'chrF_id':>7}   "
          f"{'tok_adv':>7} {'tok_qry':>7} {'tok_id':>7}")
    print("-" * 90)
    for label, r in results.items():
        c = r["per_mode_chrf"]
        t = r["per_mode_tok"]
        print(f"{label:<10} {r['chrf']:>6.1f} {r['tok']:>6.3f}   "
              f"{c['adv']:>8.1f} {c['qry']:>8.1f} {c['id']:>7.1f}   "
              f"{t['adv']:>7.3f} {t['qry']:>7.3f} {t['id']:>7.3f}")
    print()

    # Side-by-side examples — pick the same indices across strategies.
    if args.show_examples > 0 and results:
        labels = list(results.keys())
        base = results[labels[0]]
        n_show = min(args.show_examples, base["n"])
        print(f"Side-by-side examples (first {n_show}):")
        for i in range(n_show):
            mode_name = MODE_NAMES[base["modes"][i]]
            print(f"\n  [{i}] mode={mode_name}")
            print(f"    ref:    {base['refs'][i][:100]}")
            for label in labels:
                print(f"    {label:<7}: {results[label]['hyps'][i][:100]}")

    # Quick interpretation hint.
    print()
    if len(results) >= 2:
        greedy_chrf = results[list(results.keys())[0]]["chrf"]
        best = max(results.values(), key=lambda r: r["chrf"])
        delta = best["chrf"] - greedy_chrf
        print(f"Best - greedy chrF delta: {delta:+.2f}")
        if abs(delta) < 1.0:
            print("Interpretation: search is NOT leaving signal — wall is in the model's "
                  "distribution, not the decode strategy. Move on to scheduled sampling.")
        elif delta >= 2.0:
            print("Interpretation: search recovers meaningful gain — investigate why "
                  "greedy collapses. Argmax may be stuck in local mode.")
        else:
            print("Interpretation: marginal gain. Beam helps a little but doesn't break "
                  "the wall — exposure bias is still the dominant problem.")


if __name__ == "__main__":
    main()
