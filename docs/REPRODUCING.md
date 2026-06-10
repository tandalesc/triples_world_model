# Reproducing TWM results

This is the reproducibility backbone for the JEPA latent-action program (the v1 → v2 →
v2.1 line) plus the standalone operator-group fit. Every command here is verified against
the actual scripts and configs in this checkout.

## Environment

```bash
# uv + Python 3.13 (pyproject pins >=3.11; the dev box runs 3.13)
uv sync                      # core deps (torch, numpy, matplotlib, sentence-transformers, ...)
# optional extras only needed for some plotting / clustering:
uv sync --extra viz          # plotly + scikit-learn
```

- **Device:** all JEPA training/eval resolves `cuda -> mps -> cpu` automatically
  (`resolve_device` in `scripts/train_jepa_v2.py`). CPU and Apple MPS run every smoke and
  probe here. Full 100-epoch nano runs are fine on MPS but slow; the canonical 100-epoch
  matrix runs were done on the homelab dual-3090 (see `docs/AGENTS.md`).
- **scikit-learn** is required for the operator-group fit and for the probe-battery NMI
  (the probe falls back to a hand-rolled MI if sklearn is absent).

### Data prerequisites — what is in-repo vs server-only

| Artifact | Where | Used by |
|---|---|---|
| `data/glucose/chain_general_train.jsonl` (15 MB, 4049 chains) | **in-repo** | all JEPA training/probes |
| `data/glucose/chain_general_test.jsonl` (1.7 MB) | **in-repo** | probes, anchor experiment |
| `data/glucose/jepa_bpe_512.json` (vocab 512) | **in-repo** | JEPA tokenizer |
| `data/glucose/GLUCOSE_training_data_final.csv` (194 MB) | **in-repo** | operator-group fit only |
| `data/tw_all_*.jsonl`, WebNLG augmented files | **server-only** | TextWorld / open-vocab configs (NOT covered here) |

Everything in the sections below is **fully local** — no server checkout needed — except
the canonical 100-epoch matrix numbers, which are noted as homelab with a local smoke
equivalent given.

---

## 1. Operator-group fit (fully local, ~3–5 min)

**What it answers.** Which group should the transition operator `B` (with `B·z_t ≈ z_t+1`)
live in: pure rotation `U(1)`, rotation+scale, full orthogonal, or general-linear — split
by reversibility class. This is the empirical justification for the `rotation_scale`
operator the whole JEPA line uses.

```bash
uv run --with sentence-transformers --with scikit-learn --with matplotlib \
    python scripts/operator_group_fit.py
```

- **Data:** `data/glucose/GLUCOSE_training_data_final.csv` (in-repo). Extracts up to 12,000
  cause→effect pairs across the 10 GLUCOSE causal dimensions, embeds both sides with
  `all-MiniLM-L6-v2` (falls back to TF-IDF+SVD-128 if sentence-transformers can't load).
- **Seeded** (`SEED=0`), end-to-end.
- **Wall clock:** ~3–5 min on CPU (dominated by the MiniLM embedding of ~24K sentences;
  first run also downloads the MiniLM weights).
- **Artifacts:** `results/operator_fit/operator_fit_results.json`,
  `residual_by_family_class.png`, `singular_spectra.png`.

**Expected headline (verified, MiniLM backend, d=384, n=12000, label agreement 0.657):**

| class | identity | mean_shift | u1_rotation | rot_scale | orthogonal | general |
|---|---:|---:|---:|---:|---:|---:|
| all     | 0.929 | 0.925 | 0.929 | **0.709** | 0.927 | 0.643 |
| dim_irr | 0.828 | 0.781 | 0.824 | **0.649** | 0.868 | 0.750 |
| kw_irr  | 0.896 | 0.868 | 0.893 | **0.689** | 1.047 | 1.178 |

(held-out residual MSE on unit vectors; lower better). **Verdict:** `rot_scale` is the
smallest family that beats identity by a wide, consistent margin and stays robust under
per-cluster fitting; pure rotation and full orthogonal don't beat identity; general-linear
wins globally but overfits per-cluster. → the operator is parameterized `rotation_scale`.
Full writeup: `research/operator_group_fit.md`.

---

## 2. v1 testbed + refutation probes (legacy path)

v1 is a **negative result, retained on purpose.** It trains a slot/operator world model in
latent space (no token decoder), predicting the next state's verb from `state_t` alone.
Three probes refute it. v2 exists *because* of these refutations.

### 2a. v1 training (legacy)

```bash
uv run python scripts/legacy/train_jepa.py configs/archive/jepa_nano.json
```

- 100 epochs, ~82K online params, GLUCOSE adjacent pairs. Writes a **single overwritten**
  `model_latest.pt` (the checkpoint-retention bug — see below).
- Canonical run was homelab; local MPS works but is slow. Artifacts from the canonical run
  live at `results/jepa_nano_smoke/` (final ckpt) and `results/jepa_nano_viz/` (geometry
  report `REPORT.md`).

### 2b. v1 refutation probes

The v1 probe methodology lives in `results/jepa_nano_probe/`
(`probe1_retrieval.py`, `probe2_verb_mi.py`, `probe2_glucose_dim.py`,
`probe3_slot_purity.py`; shared helpers in `_probe_common.py`). Each probe loads the v1
checkpoint and writes a JSON next to it.

**Expected headline (the three independent refutations, verified in the committed JSONs):**

- **probe1 retrieval** (`probe1_retrieval.json`): hard-pool MRR **0.070 vs chance 0.104** —
  *below* chance. `easy_minus_hard_mrr = −0.041`. The model ranks the wrong same-chain
  state above the right next-state → `L_pred` was satisfied by topic-prior similarity, not
  transition modeling.
- **probe2 verb–label MI** (`probe2_verb_mi.json`): `nmi_minus_shuffle ≈ 0.0002` — codebook
  carries no transition semantics; usage ppl 6.2/8 looks healthy but is positional.
- **probe3 slot purity** (`probe3_slot_purity.json`): every slot's LOO residual increase
  `< 0` (masking a slot *helps*); per-slot verb histograms near-deterministic → the encoder
  hardwired one verb per slot position; the operator path is decorative.

Root cause (all three): predicting the next causal step from the present alone is
ill-posed on narrative data. Full reading: `research/jepa_v2_latent_actions.md` §0.

---

## 3. v2 latent-actions training

The live path. A training-time **posterior** sees both `state_t` and `state_t+1` and emits
ONE sequence-level discrete action `v`; a token decoder must reconstruct `text_t+1`, so the
action's bits are *necessary*. Token CE is the primary loss; codebook usage is a diagnostic
only.

### 3a. Smoke (fully local, ~2–4 min on MPS — VERIFIED)

```bash
uv run python scripts/train_jepa_v2.py configs/jepa_nano_v2_smoke.json
```

- 3 epochs, `max_chains=2000` (4000 pairs), eval every epoch. 236,568 online params.
- Writes `results/jepa_nano_v2_smoke/model_ep{1,2,3}.pt` + `model_latest.pt` and prints a
  `diag_v2[epN]` line per eval.
- **Expected (verified):** loss falls 6.11 → 4.02 over 3 epochs; `ce_true_nats` ~3.9 by
  ep3; `ce_gap_nats` ~0.04–0.08; `gen_chrf_greedy` ~0.26; `gen_exact_greedy` 0.0;
  `n_action_codes_used` 2 (codebook hasn't differentiated yet at 3 epochs). This smoke is a
  *plumbing check*, not a capability run — the headline numbers come from §3b.

### 3b. Full run (homelab; local MPS possible but slow)

```bash
uv run python scripts/train_jepa_v2.py configs/jepa_nano_v2.json
```

- 100 epochs, full GLUCOSE train (~72.9K pairs), eval every 5 epochs.
- **Expected headline (ep100, the matrix baseline):** `ce_true ≈ 1.39`, `ce_gap ≈ 0.166`,
  `hard_mrr ≈ 0.071`, `chrF ≈ 0.384`, all 8 codes used, `noun_eff_rank ≈ 11`.
- Artifacts land in `results/jepa_nano_v2/` (per-eval `model_ep{N}.pt` retained).
- Canonical numbers: synthesis table in `research/jepa_matrix_synthesis.md` §1.

Other v2 arms (same script, different config):
- `configs/jepa_nano_v2_m32.json` — 32 slots instead of 8.
- `configs/jepa/jepa_mini_v2.json` — mini profile (d_model 128). Note: this config still
  carries v1-style loss keys (`w_div`, `w_scale_reg`); treat as experimental.

---

## 4. v2.1 polar (modulus = identity, phase = state)

Same training script and loss; adds polar conditioning (`use_polar_conditioning: true`).
The modulus profile `|z_b|` is the persistent identity; the verb is a per-block complex
multiply. Behaviorally neutral vs v2.0 at this scale (see synthesis §2), kept because it is
free and non-harmful and powers the kind/identity fingerprint.

### 4a. Smoke (fully local — same runtime profile as §3a)

```bash
uv run python scripts/train_jepa_v2.py configs/jepa/jepa_nano_v21_smoke.json
```

→ `results/jepa_nano_v21_smoke/`.

### 4b. Full runs (homelab)

```bash
uv run python scripts/train_jepa_v2.py configs/jepa/jepa_nano_v21.json          # seed 0
uv run python scripts/train_jepa_v2.py configs/jepa/jepa_nano_v21_seed1.json    # seed 1
uv run python scripts/train_jepa_v2.py configs/jepa/jepa_nano_v21_dn64.json     # d_noun 32->64
uv run python scripts/train_jepa_v2.py configs/jepa/jepa_small_v21_dec.json     # d_dec 128, 2L decoder arm
```

- 100 epochs each. Param counts (verified): nano v21 236,824; dn64 260,632;
  **decoder arm 657,304**.
- **Headline (ep100):** the decoder arm is the one real win — `ce_true` **0.953 nats** (vs
  baseline 1.39) and grammatical GLUCOSE sentences instead of BPE soup. polar seed0/seed1
  `ce_gap` 0.082 vs 0.134 (the 63% seed-variance finding). dn64 lifts modulus_eff_rank
  6.5 → 10.8 and nothing downstream. Full matrix: `research/jepa_matrix_synthesis.md` §1.

> Two seeds (v21 + v21_seed1) are the inputs to the anchor experiment (§6). Run both before §6.

---

## 5. v2 probe battery

`results/jepa_v2_probe/probe_v2_battery.py` — decoder-likelihood retrieval, slot LOO,
action-semantics NMI, and text samples against a v2/v2.1 checkpoint. CPU-only, read-only.

```bash
# expects model + tokenizer reachable; written to run server-side as the repo owner:
CUDA_VISIBLE_DEVICES= uv run python results/jepa_v2_probe/probe_v2_battery.py \
    --ckpt <path>/model_ep100.pt --out <out_dir>
```

> Note: the loader hardcodes `~/triples_world_model_Glucose/data/glucose/...` for the
> tokenizer and test chains (homelab path). To run locally, point those at the in-repo
> `data/glucose/` (see the `load()` / `main()` paths in the script). The committed JSONs in
> `results/jepa_v2_probe/` are from the homelab nano-v2 ep100 checkpoint.

**Expected headline (committed JSONs, the three persistent capability gaps):**

- **probe1** (`probe1_retrieval.json`): hard-pool MRR 0.117 vs chance 0.104; recall@1 0.037
  vs 0.024 — next-state retrieval barely beats chance.
- **probe2** (`probe2_slot_loo.json`): **all 8 slots constructive** — masking any slot
  *increases* CE (the qualitative flip vs v1, where every slot LOO ≤ 0); coarse-to-fine
  monotone 1.86 → 1.46.
- **probe3** (`probe3_action_semantics.json`): codebook well-used (ppl 7.6/8, all 8 active)
  but **NMI(action; reversibility) = 0.004, below its 0.012 shuffle baseline** — the action
  aligns with nothing nameable.
- **probe4** (`probe4_samples.json`): fluent-but-wrong samples; gen_exact 0.0.

---

## 6. Anchor-stability experiment (the LLM-mount stage-0 gate)

`results/jepa_matrix/anchor_stability.py` — relative-representation seed stability. Tests
whether a probe state's vector of cosine similarities to a fixed 64-anchor set is stable
across two random-seed checkpoints. The gate for mounting this representation inside an LLM.

```bash
CUDA_VISIBLE_DEVICES= uv run python results/jepa_matrix/anchor_stability.py \
    --ckpt0 results/jepa_nano_v21/model_ep100.pt \
    --ckpt1 results/jepa_nano_v21_seed1/model_ep100.pt \
    --chains data/glucose/chain_general_test.jsonl \
    --out results/jepa_matrix/staged/anchor_stability.json
```

- Requires both seed checkpoints from §4b. CPU-only, read-only.
- **Expected headline (verified JSON):** anchor-sim Pearson **0.431 readout / 0.456
  slotmean** — both in the 0.4–0.7 grey zone, below the 0.7 interface-viable bar.
  Raw cross-seed cosine ~0 (frames differ) but Procrustes-aligned readout cosine **0.925**
  (geometry IS shared up to rotation). **Verdict: MARGINAL — interface not yet viable;
  next move is an anchor M-sweep, not abandonment.** Full reading:
  `research/jepa_matrix_synthesis.md` §3.

`results/jepa_matrix/extract_matrix.py` is the companion that re-derives the decoder-arm
final diag (overwritten in its training log) and stages the sample/probe JSONs the
synthesis cites. Both are homelab/owner-checkout scripts.

---

## 7. Experiment matrix + synthesis

The numbers-first writeup tying §3–§6 together — the five-run overnight matrix, per-arm
verdicts, the seed-variance caveat, the three capability gaps, and the v3 recipe — is
`research/jepa_matrix_synthesis.md`. Start there for "what do we actually know."

---

## Known reproducibility caveats

- **Checkpoint-retention bug (v1 only, fixed in v2).** `scripts/legacy/train_jepa.py`
  overwrites a single `model_latest.pt` every epoch, so no mid-training (collapse-window)
  checkpoint survives — this blocked the v1 before/after geometry report
  (`results/jepa_nano_viz/REPORT.md`). `scripts/train_jepa_v2.py` fixes this: it writes a
  distinct `model_ep{N}.pt` every eval. Always run v2 for anything you may want to probe
  later.
- **Probe/anchor scripts assume the homelab owner checkout.** `probe_v2_battery.py`,
  `anchor_stability.py`, and `extract_matrix.py` hardcode `~/triples_world_model_Glucose/`
  for the tokenizer/test data. They run as-is server-side; locally, edit those paths to the
  in-repo `data/glucose/`.
- **`jepa_mini_v2.json` mixes v1 and v2 loss keys** (`w_div`, `w_scale_reg`,
  `operator_fit_pass2`). The v2 trainer/loss filter unknown kwargs, so it runs, but the
  mini arm is not part of the verified matrix — treat it as experimental.
- **Eval-set caveat in the matrix.** Several matrix diag numbers are on *train* chains
  (n=512), and the decoder-arm diag was *re-derived* from its checkpoint because the live
  log was overwritten by the dn64 run. See `research/jepa_matrix_synthesis.md` §1 for which
  number came from where.
