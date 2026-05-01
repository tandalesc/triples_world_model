#!/usr/bin/env python3
"""Partial-observability diagnostic for the TextWorld AR wall.

Splits the test set into two partitions:
  - "in_input":  every target token (non-special) appears in the input
  - "out_input": at least one target token is NOT in the input

If chrF on "in_input" jumps far above the wall (~49) while "out_input" stays
flat, the wall is partial observability — the model is being asked for
unobservable info. If both partitions sit at the wall, the wall is the
decoder's distribution / exposure bias and architecture must change.

Usage:
  uv run python scripts/eval_observability.py \\
    --run-dir results/dual_ar_v4_vq_entity \\
    --test-data data/tw_all_test.jsonl \\
    --target-per-partition 200
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.chain_dataset import ChainDataset
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
    return DualARModel(
        config=model_config, tokenizer=tokenizer,
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


def is_in_input(input_ids_row, target_ids_row, target_pad_row, special_ids):
    """Return True if every non-special target token appears in input."""
    inp_set = set(input_ids_row.tolist()) - special_ids
    tgt_tokens = target_ids_row[~target_pad_row].tolist()
    tgt_content = [t for t in tgt_tokens if t not in special_ids]
    if not tgt_content:
        return True  # vacuous: nothing to predict
    return all(t in inp_set for t in tgt_content)


def coverage_ratio(input_ids_row, target_ids_row, target_pad_row, special_ids):
    """Fraction of target content tokens present in input."""
    inp_set = set(input_ids_row.tolist()) - special_ids
    tgt_tokens = target_ids_row[~target_pad_row].tolist()
    tgt_content = [t for t in tgt_tokens if t not in special_ids]
    if not tgt_content:
        return 1.0
    return sum(1 for t in tgt_content if t in inp_set) / len(tgt_content)


@torch.no_grad()
def run(model, test_loader, device, target_per_partition, special_ids):
    """Iterate test set, partition examples, generate greedy, collect refs/hyps."""
    partitions = {
        "in_input": {"refs": [], "hyps": [], "modes": [], "cov": [],
                     "tok_correct": 0, "tok_total": 0,
                     "mode_tok": {0: [0, 0], 1: [0, 0], 2: [0, 0]}},
        "out_input": {"refs": [], "hyps": [], "modes": [], "cov": [],
                      "tok_correct": 0, "tok_total": 0,
                      "mode_tok": {0: [0, 0], 1: [0, 0], 2: [0, 0]}},
    }
    pad_id = model.tokenizer.pad_token_id

    def both_full():
        return all(
            len(p["refs"]) >= target_per_partition for p in partitions.values()
        )

    for batch in test_loader:
        if both_full():
            break
        input_ids = batch["input_ids"].to(device)
        input_pad = batch["input_pad"].to(device)
        chain_ids = batch["chain_ids"].to(device)
        chain_pad = batch["chain_pad"].to(device)
        chain_len_b = batch["chain_len"].to(device)
        mode_ids_b = batch["mode_id"].to(device)

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
            if both_full():
                break
            last = chain_len_b[i].item() - 1
            tgt_ids = chain_ids[i, last]
            tgt_pad = chain_pad[i, last]

            label = "in_input" if is_in_input(
                input_ids[i], tgt_ids, tgt_pad, special_ids
            ) else "out_input"
            cov = coverage_ratio(input_ids[i], tgt_ids, tgt_pad, special_ids)
            slot = partitions[label]
            if len(slot["refs"]) >= target_per_partition:
                continue

            gen_ids = model.decoder.generate(
                dynamics_out=bn[i:i + 1],
                compressor_out=dense_pred[i:i + 1],
                max_tokens=model.max_text_tokens,
            )

            non_pad = ~tgt_pad
            ref = model.tokenizer.decode(tgt_ids[non_pad].tolist())
            hyp = model.tokenizer.decode(gen_ids[0].tolist())

            slot["refs"].append(ref)
            slot["hyps"].append(hyp)
            slot["modes"].append(mode_ids_b[i].item())
            slot["cov"].append(cov)

            gen_len = min(gen_ids.shape[1], tgt_ids.shape[0])
            mask = tgt_ids[:gen_len] != pad_id
            if mask.any():
                c = (gen_ids[0, :gen_len][mask] == tgt_ids[:gen_len][mask]).sum().item()
                t = mask.sum().item()
                slot["tok_correct"] += c
                slot["tok_total"] += t
                m = mode_ids_b[i].item()
                slot["mode_tok"][m][0] += c
                slot["mode_tok"][m][1] += t

    return partitions


def summarize(partitions):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()
    out = {}
    for name, p in partitions.items():
        if not p["refs"]:
            out[name] = None
            continue
        chrf = chrf_metric.corpus_score(p["hyps"], [p["refs"]]).score
        tok = p["tok_correct"] / max(p["tok_total"], 1)
        per_mode_chrf = {}
        per_mode_tok = {}
        for m, mname in MODE_NAMES.items():
            sel = [(h, r) for h, r, mm in zip(p["hyps"], p["refs"], p["modes"]) if mm == m]
            if sel:
                mh, mr = zip(*sel)
                per_mode_chrf[mname] = chrf_metric.corpus_score(list(mh), [list(mr)]).score
            else:
                per_mode_chrf[mname] = float("nan")
            mt = p["mode_tok"][m]
            per_mode_tok[mname] = mt[0] / mt[1] if mt[1] else float("nan")
        out[name] = {
            "n": len(p["refs"]),
            "chrf": chrf,
            "tok": tok,
            "avg_cov": sum(p["cov"]) / len(p["cov"]),
            "per_mode_chrf": per_mode_chrf,
            "per_mode_tok": per_mode_tok,
            "examples": list(zip(p["refs"][:5], p["hyps"][:5], p["modes"][:5], p["cov"][:5])),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--test-data", required=True)
    p.add_argument("--target-per-partition", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--show-examples", type=int, default=4)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    if cfg.get("tokenizer_pretrained"):
        tokenizer = DomainBPETokenizer.from_pretrained(
            cfg["tokenizer_pretrained"],
            max_length=cfg.get("max_text_tokens", 128),
        )
    else:
        tokenizer = DomainBPETokenizer.load(
            run_dir / "tokenizer.json",
            max_length=cfg.get("max_text_tokens", 128),
        )

    test_ds = ChainDataset(
        args.test_data, tokenizer,
        max_text_tokens=cfg.get("max_text_tokens", 128),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 64))

    model = build_model(cfg, tokenizer, device)
    model.load_state_dict(torch.load(run_dir / "weights.pt", map_location=device))
    model.eval()

    # Treat pad/mask/unk as non-content. Common-vocab tokens (the, a, .) are
    # NOT excluded — they really do appear in input, so they're legit signal.
    special_ids = set()
    for attr in ("pad_token_id", "mask_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            special_ids.add(v)

    print(f"Run: {run_dir}")
    print(f"Test: {args.test_data}  ({len(test_ds)} chains)")
    print(f"Target per partition: {args.target_per_partition}")
    print(f"Special ids excluded from membership: {sorted(special_ids)}")
    print()
    print("Streaming + partitioning + generating greedy...")

    partitions = run(model, test_loader, device, args.target_per_partition, special_ids)
    summary = summarize(partitions)

    print()
    print(f"{'partition':<10} {'n':>4} {'avg_cov':>8} {'chrF':>6} {'tok':>6}   "
          f"{'chrF_adv':>8} {'chrF_qry':>8} {'chrF_id':>7}   "
          f"{'tok_adv':>7} {'tok_qry':>7} {'tok_id':>7}")
    print("-" * 100)
    for name in ("in_input", "out_input"):
        s = summary[name]
        if s is None:
            print(f"{name:<10}  (no samples)")
            continue
        c = s["per_mode_chrf"]
        t = s["per_mode_tok"]
        print(f"{name:<10} {s['n']:>4} {s['avg_cov']:>8.2f} {s['chrf']:>6.1f} {s['tok']:>6.3f}   "
              f"{c['adv']:>8.1f} {c['qry']:>8.1f} {c['id']:>7.1f}   "
              f"{t['adv']:>7.3f} {t['qry']:>7.3f} {t['id']:>7.3f}")
    print()

    # Sample-level inspection.
    if args.show_examples > 0:
        for name in ("in_input", "out_input"):
            s = summary[name]
            if s is None:
                continue
            print(f"--- {name} examples ---")
            for ref, hyp, m, cov in s["examples"][:args.show_examples]:
                print(f"  mode={MODE_NAMES[m]} cov={cov:.2f}")
                print(f"    ref: {ref[:120]}")
                print(f"    gen: {hyp[:120]}")
            print()

    # Decision rule.
    if summary["in_input"] and summary["out_input"]:
        delta = summary["in_input"]["chrf"] - summary["out_input"]["chrf"]
        adv_in = summary["in_input"]["per_mode_chrf"]["adv"]
        adv_out = summary["out_input"]["per_mode_chrf"]["adv"]
        adv_delta = adv_in - adv_out
        print(f"chrF delta (in - out): {delta:+.1f}")
        print(f"chrF_adv delta (in - out): {adv_delta:+.1f}")
        if adv_delta >= 20:
            print("Verdict: PARTIAL OBSERVABILITY. Architecture move = persistent "
                  "memory / accumulating triple state across chain.")
        elif adv_delta >= 10:
            print("Verdict: MIXED. Some signal is information-access, some is "
                  "decoder distribution. Both directions worth pursuing.")
        else:
            print("Verdict: DECODER DISTRIBUTION. Information access is not the "
                  "wall. Architecture move = stochastic latent dynamics OR "
                  "structured triple-output with separate surface decoder.")


if __name__ == "__main__":
    main()
