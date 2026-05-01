#!/usr/bin/env python3
"""Multi-sample diagnostic for a trained flow checkpoint.

Tests whether the flow has collapsed to deterministic point prediction
or genuinely produces multimodal outputs. Per input, samples N
candidates with different noise seeds, decodes each, and reports:

  - Greedy chrF (single sample, what training reports)
  - Best-of-N chrF (oracle picker, what we'd get with a perfect ranker)
  - Mean candidate chrF (average over the N samples)
  - Inter-candidate diversity: pairwise chrF between candidates. Low
    inter-chrF = candidates differ from each other (multimodal flow).
    High inter-chrF = candidates are near-duplicates (collapsed flow).

Decision rule:
  - best_of_N(adv) >> greedy(adv): multimodal sampling is the answer.
    Flow has the diversity, we just need a better picker. Pursue
    eval-time techniques (reranker, self-consistency).
  - best_of_N(adv) ≈ greedy(adv) AND high inter-candidate chrF: flow
    collapsed to deterministic. Need explicit per-sample latent
    conditioning (v3) to recover multimodality.
  - best_of_N(adv) ≈ greedy(adv) AND low inter-candidate chrF: flow
    IS multimodal but all modes are equally wrong. Indicates a deeper
    problem — the conditional distribution doesn't contain the correct
    answer. Worth investigating training data / mode conditioning.

Usage:
  uv run python scripts/eval_flow_multisample.py \\
    --base-run-dir results/ar_autoenc_tw \\
    --flow-checkpoint results/flow_autoenc_v2/flow_best.pt \\
    --test-data data/tw_all_test.jsonl \\
    --n-candidates 4 \\
    --n-samples 200
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
from twm.flow_dynamics import FlowDynamics
from train_ar_autoencoder import AutoencoderModel  # noqa: E402
from train_flow_autoenc import build_autoencoder  # noqa: E402

NUM_MODES = 3
MODE_NAMES = {0: "adv", 1: "qry", 2: "id"}


def load_base(base_run_dir, device):
    base_dir = Path(base_run_dir)
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
    base_model = build_autoencoder(base_cfg, tokenizer, device)
    state = torch.load(base_dir / "weights.pt", map_location=device)

    pe_key = "decoder.pos_emb.weight"
    if pe_key in state:
        saved = state[pe_key]
        current = base_model.decoder.pos_emb.weight
        if saved.shape != current.shape and saved.shape[1] == current.shape[1]:
            new = current.data.clone()
            n = min(saved.shape[0], new.shape[0])
            new[:n] = saved[:n]
            state[pe_key] = new
    base_model.load_state_dict(state)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    return base_model, base_cfg, tokenizer


def load_flow(flow_path, base_cfg, device, n_layers=4, n_heads=8, d_ff=512):
    bn_d = base_cfg.get("bottleneck_dim") or base_cfg.get("d_model", 128)
    flow = FlowDynamics(
        d_model=bn_d, n_heads=n_heads, n_layers=n_layers,
        num_modes=NUM_MODES,
        max_triples=base_cfg.get("max_triples", 16),
        d_ff=d_ff, dropout=0.0,
    ).to(device)
    flow.load_state_dict(torch.load(flow_path, map_location=device))
    flow.eval()
    return flow


@torch.no_grad()
def multi_sample_eval(base_model, flow, test_loader, device,
                     n_candidates, n_samples, n_ode_steps):
    from sacrebleu.metrics import CHRF
    chrf_metric = CHRF()

    # For each test example, store: ref, list of N hyps, mode
    rows = []
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

        z_prev_init = base_model.compressor(
            input_ids, input_pad, base_model.config.max_triples,
        )
        if isinstance(z_prev_init, tuple):
            z_prev_init = z_prev_init[0]

        # Run flow N times with different noise per candidate.
        # We unroll the chain step-by-step, but use the same fresh noise
        # at each step within a single candidate run.
        candidate_outputs = []
        for k in range(n_candidates):
            z_prev = z_prev_init.clone()
            max_c = chain_len_b.max().item()
            for s in range(1, max_c):
                z_prev = flow.sample(z_prev, mode_ids_b, n_steps=n_ode_steps)
            candidate_outputs.append(z_prev)

        for i in range(len(input_ids)):
            if n >= n_samples:
                break
            last = chain_len_b[i].item() - 1
            tgt_ids = chain_ids[i, last]
            tgt_pad = chain_pad[i, last]
            non_pad = ~tgt_pad
            ref = tok.decode(tgt_ids[non_pad].tolist())
            mode = mode_ids_b[i].item()

            hyps = []
            for k in range(n_candidates):
                gen_ids = base_model.decoder.generate(
                    bottleneck=candidate_outputs[k][i:i + 1],
                    max_tokens=base_model.max_text_tokens,
                )
                hyps.append(tok.decode(gen_ids[0].tolist()))

            rows.append({"ref": ref, "hyps": hyps, "mode": mode})
            n += 1

    # Aggregate metrics.
    greedy_refs, greedy_hyps = [], []
    best_refs, best_hyps = [], []
    mean_refs, mean_hyps = [], []
    per_mode = {m: {"greedy": ([], []), "best": ([], []), "mean": ([], []),
                    "diversity": []} for m in MODE_NAMES}

    for row in rows:
        ref = row["ref"]
        hyps = row["hyps"]
        mode = row["mode"]

        # Per-candidate chrF, used for best-of-N and mean.
        per_cand = [chrf_metric.sentence_score(h, [ref]).score for h in hyps]
        best_idx = max(range(len(per_cand)), key=lambda i: per_cand[i])

        greedy_refs.append(ref); greedy_hyps.append(hyps[0])
        best_refs.append(ref); best_hyps.append(hyps[best_idx])
        for h in hyps:
            mean_refs.append(ref); mean_hyps.append(h)

        per_mode[mode]["greedy"][0].append(ref)
        per_mode[mode]["greedy"][1].append(hyps[0])
        per_mode[mode]["best"][0].append(ref)
        per_mode[mode]["best"][1].append(hyps[best_idx])
        for h in hyps:
            per_mode[mode]["mean"][0].append(ref)
            per_mode[mode]["mean"][1].append(h)

        # Inter-candidate diversity: pairwise chrF among the candidates.
        # Low = candidates are similar (collapsed). High = candidates diverge.
        if len(hyps) >= 2:
            pairs = [
                chrf_metric.sentence_score(hyps[i], [hyps[j]]).score
                for i in range(len(hyps)) for j in range(i + 1, len(hyps))
            ]
            per_mode[mode]["diversity"].append(sum(pairs) / len(pairs))

    def score(refs, hyps):
        if not refs:
            return float("nan")
        return chrf_metric.corpus_score(hyps, [refs]).score

    overall = {
        "greedy": score(greedy_refs, greedy_hyps),
        "best": score(best_refs, best_hyps),
        "mean": score(mean_refs, mean_hyps),
    }
    by_mode = {}
    for m, name in MODE_NAMES.items():
        d = per_mode[m]
        diversity = (sum(d["diversity"]) / len(d["diversity"])
                     if d["diversity"] else float("nan"))
        by_mode[name] = {
            "greedy": score(*d["greedy"]),
            "best": score(*d["best"]),
            "mean": score(*d["mean"]),
            "inter_chrf": diversity,
            "n": len(d["greedy"][0]),
        }
    return overall, by_mode, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-run-dir", required=True)
    p.add_argument("--flow-checkpoint", required=True)
    p.add_argument("--flow-config", default=None,
                   help="JSON config the flow was trained with (for layer/head shape).")
    p.add_argument("--test-data", required=True)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-ode-steps", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--show-examples", type=int, default=4)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Base: {args.base_run_dir}")
    print(f"Flow: {args.flow_checkpoint}")
    print(f"Test: {args.test_data}")
    print(f"Candidates per input: {args.n_candidates} | ODE steps: {args.n_ode_steps} | Samples: {args.n_samples}")
    print()

    base_model, base_cfg, tokenizer = load_base(args.base_run_dir, device)

    flow_layers = flow_heads = flow_d_ff = None
    if args.flow_config:
        with open(args.flow_config) as f:
            fc = json.load(f)
        flow_layers = fc.get("flow_n_layers", 4)
        flow_heads = fc.get("flow_n_heads", 8)
        flow_d_ff = fc.get("flow_d_ff", 512)
    flow = load_flow(
        args.flow_checkpoint, base_cfg, device,
        n_layers=flow_layers or 4,
        n_heads=flow_heads or 8,
        d_ff=flow_d_ff or 512,
    )

    test_ds = ChainDataset(
        args.test_data, tokenizer,
        max_text_tokens=base_cfg.get("max_text_tokens", 128),
    )
    test_loader = DataLoader(test_ds, batch_size=base_cfg.get("batch_size", 64))

    overall, by_mode, rows = multi_sample_eval(
        base_model, flow, test_loader, device,
        n_candidates=args.n_candidates,
        n_samples=args.n_samples,
        n_ode_steps=args.n_ode_steps,
    )

    print(f"{'mode':<8} {'n':>4} {'greedy':>7} {'best-of-N':>10} {'mean':>6} {'inter_chrF':>11}")
    print("-" * 60)
    for name, d in by_mode.items():
        print(f"{name:<8} {d['n']:>4} {d['greedy']:>7.1f} {d['best']:>10.1f} "
              f"{d['mean']:>6.1f} {d['inter_chrf']:>11.1f}")
    print("-" * 60)
    print(f"{'overall':<8} {sum(d['n'] for d in by_mode.values()):>4} "
          f"{overall['greedy']:>7.1f} {overall['best']:>10.1f} {overall['mean']:>6.1f}")
    print()

    if args.show_examples > 0:
        print(f"Side-by-side (first {args.show_examples}):")
        for row in rows[:args.show_examples]:
            print(f"\n  mode={MODE_NAMES[row['mode']]}")
            print(f"    ref:  {row['ref'][:100]}")
            for k, h in enumerate(row["hyps"]):
                print(f"    h[{k}]: {h[:100]}")

    print()
    adv = by_mode["adv"]
    delta = adv["best"] - adv["greedy"]
    inter = adv["inter_chrf"]
    print(f"adv: best-of-N - greedy = {delta:+.1f}  (inter_chrF {inter:.1f})")
    if delta >= 8:
        print("Verdict: MULTIMODAL FLOW. Greedy underexploits; pursue eval-time "
              "ranker / self-consistency / reranker.")
    elif delta < 4 and inter >= 50:
        print("Verdict: COLLAPSED FLOW. Candidates are near-duplicates. "
              "Need explicit per-sample latent z conditioning (v3).")
    elif delta < 4 and inter < 30:
        print("Verdict: DIVERSE BUT WRONG. Flow IS multimodal but no candidate "
              "matches the target. Investigate conditioning / data.")
    else:
        print("Verdict: AMBIGUOUS. Marginal gain from multi-sample.")


if __name__ == "__main__":
    main()
