# AGENTS — operating manual

The agent-facing manual for working this repo. Complements `CLAUDE.md` (the project surface
and architecture) — this file is the *how-to-operate* layer: running jobs on the homelab,
moving artifacts, diagnostics conventions, and probe discipline. When the two disagree,
`CLAUDE.md` wins on architecture; this file wins on operations.

## Where training actually runs

Training runs on the homelab GPU server (`rrh-llm-1`, dual RTX 3090) via the **wartable**
MCP, not locally. Local CPU/MPS is fine for smoke runs, probes, and the operator-group fit
(see `docs/REPRODUCING.md`). The server has a mirror checkout at
`~/triples_world_model_Glucose` and the large server-only data files.

### The 3GB-slice reality (important — the old 22GB rule is dead)

vLLM now runs **tensor-parallel across BOTH 3090s** (~19 GB shard per card + an EngineCore
proc). The historical `gpu_vram_min_gb=22` routing rule is **unsatisfiable** — there is no
free 22 GB card. For these models (all sub-1M params, Python-bound not GPU-bound):

- Request **`gpu_vram_min_gb=3`** and run in the margins. Several concurrent small trainers
  are fine — GPU util stays <20%.
- Do **not** hardcode `CUDA_VISIBLE_DEVICES`; let the scheduler assign.
- For quick debug jobs that need almost no VRAM, omit `gpu_vram_min_gb` entirely.

### Submit pattern

```
mcp__wartable__submit_job(
    name="<experiment-name>",
    command=(
        "cd ~/triples_world_model_Glucose && "
        "git fetch origin feature/glucose-converter && "
        "git reset --hard origin/feature/glucose-converter && "
        "rm -rf <out_dir> && "
        "uv run python scripts/train_jepa_v2.py configs/jepa/<config>.json"
    ),
    gpu_count=1,
    gpu_vram_min_gb=3,
    tags=["jepa", "<variant>"],
)
```

### Subscribe to filtered logs

After submission, subscribe with a regex filter so only meaningful lines come back as
channel events. The subscription auto-stops when the job completes.

```
mcp__wartable_channel__subscribe_job_logs(
    job_id=<id>,
    interval_seconds=600,        # tighten if early validation matters
    pattern="Epoch|loss|ce_true|ce_gap|hard_mrr|chrf|diag_v2|Error|Traceback",
    tail_lines=30,
)
```

## /tmp staging + bot.claude scp (foreign-checkout access)

Probe and analysis scripts (`results/jepa_v2_probe/probe_v2_battery.py`,
`results/jepa_matrix/anchor_stability.py`, `extract_matrix.py`) run **server-side as the
repo owner** and assume the `~/triples_world_model_Glucose/` checkout for the tokenizer and
test chains. They write results to a staging dir, conventionally **`/tmp/jepa_stage/`**.

To pull staged artifacts back to the local checkout, scp as the light-access bot user:

```bash
scp -i ~/.ssh/hgs_lab_ed25519 \
    bot.claude@rrh-llm-1:/tmp/jepa_stage/anchor_stability.json \
    results/jepa_matrix/staged/
```

Direct SSH for light tests/debugging (NOT training — vLLM shares the box; use a few-GB
slice or CPU, and `nvidia-smi` first):

```bash
ssh -i ~/.ssh/hgs_lab_ed25519 bot.claude@rrh-llm-1
```

The staging pattern exists because the wartable job's stdout is the only channel back by
default, and large JSON/PNG artifacts are better scp'd than echoed. Stage to `/tmp`, then
pull what the synthesis needs.

## Diagnostics conventions

Every v2 eval prints a single flat line:

```
diag_v2[ep<N>] key=val key=val ...
```

emitted by `eval_diagnostics_v2` (in `src/twm/jepa/diagnostics.py`). Each metric is a float
or int; the trainer flattens the dict so logs grep cleanly. Key metrics and what they mean:

| metric | meaning | healthy / threshold |
|---|---|---|
| `ce_true_nats` | teacher-forced token CE of the true next-state | lower; nano baseline ~1.39, decoder arm ~0.95 |
| `ce_gap_nats` | CE increase when the verb is ablated (constant-verb) | higher = action carries signal; **seed-noisy, see below** |
| `gap_passes_threshold` | `ce_gap` above its bar | informational |
| `hard_mrr` / `easy_mrr` | next-state retrieval MRR on hard / easy pool | compare to `chance_hard_mrr` / `chance_easy_mrr` |
| `gen_chrf_greedy` / `gen_exact_greedy` | generation quality vs gold next-state | chrF ~0.37–0.38, exact ~0.0 currently |
| `noun_eff_rank` / `modulus_eff_rank` | effective rank of noun / modulus space | warn flags fire below thresholds (8 / 4) |
| `v_usage_ppl_posterior/prior` | codebook usage perplexity | healthy >4; near 1 = collapse |
| `n_action_codes_used` | distinct codes used | want = n_verbs (8) |
| `identity_persistence_pass` | pure rotations preserve modulus | structural check, should pass |

### V-relative thresholds, not absolutes

Read retrieval and gap metrics **relative to their chance/V baseline**, never as absolutes.
`hard_mrr=0.07` is *below* `chance_hard_mrr=0.10` → a refutation, not a weak positive. The
diag line ships the chance baselines next to the metric for exactly this reason. The same
holds for NMI: compare to the shuffle baseline in the same JSON.

### Seed-variance rule

`ce_gap` swings ~63% across identical-config seeds (0.082 vs 0.134). **Treat any sub-0.05-nat
effect as null.** For any claim you intend to keep, run **≥3 seeds and report the band**
before calling an effect real.

### The permutation-probe rule: never trust usage statistics

A codebook can show healthy usage perplexity (e.g. 6.2/8 in v1) while carrying **zero**
task semantics — v1's verbs were assigned per slot *position*, gaming the usage entropy.
**Never read "codebook is well-used" as "codebook is meaningful."** Always validate with a
permutation/shuffle baseline (NMI vs shuffled labels) or an ablation (LOO, constant-verb
`ce_gap`). If the observed statistic doesn't beat its own shuffle, the structure is
decorative.

## Probes — where they live and how to rerun

| Probe set | Location | What it tests |
|---|---|---|
| v1 refutation | `results/jepa_nano_probe/` (`probe1_retrieval.py`, `probe2_verb_mi.py`, `probe2_glucose_dim.py`, `probe3_slot_purity.py`) | the three v1 failures (retrieval, verb MI, slot purity) |
| v2 battery | `results/jepa_v2_probe/probe_v2_battery.py` | decoder-likelihood retrieval, slot LOO, action NMI, samples |
| anchor gate | `results/jepa_matrix/anchor_stability.py` | relative-representation seed stability (LLM-mount gate) |
| matrix extract | `results/jepa_matrix/extract_matrix.py` | re-derive overwritten decoder-arm diag, stage JSONs |

Rerun commands and expected numbers are in `docs/REPRODUCING.md` §2, §5, §6. All probes are
CPU-only and read-only against checkpoints. Locally, edit the hardcoded
`~/triples_world_model_Glucose/data/glucose/` paths to the in-repo `data/glucose/`.

## Checkpoint retention

`scripts/train_jepa_v2.py` writes a **distinct `model_ep{N}.pt` every eval epoch** plus a
rolling `model_latest.pt`. This is deliberate and load-bearing: the v1 trainer
(`scripts/legacy/train_jepa.py`) overwrote a single `model_latest.pt`, so when a run
collapsed mid-training no recoverable earlier state survived — this bit twice
(`results/jepa_nano_viz/REPORT.md`). **Never strip per-eval checkpoints from the v2
trainer.** When pruning disk, keep at minimum the final `model_ep{last}.pt` and any eval
where a diagnostic flipped; the per-eval series is what enables PCA-evolution animations and
before/after geometry contrasts.

## Conventions checklist

- Edit on `feature/glucose-converter` (not `main`); commit + push before submitting a job
  that resets the server checkout to your commit.
- Device order is `cuda -> mps -> cpu`, resolved automatically — don't pin it.
- Smoke before full: every config has a `*_smoke.json` (`max_chains` capped, 3 epochs).
  Run it locally first; it's a plumbing check, not a capability run.
- ≥3 seeds before any sub-0.05-nat claim. Shuffle baseline before any "codebook is
  meaningful" claim.
