# Sprint 4: Config-Driven Training & Dynamics Core

## Goal

Refactor training from 600-line scripts into config-driven experiments, then train a dynamics core that transforms question bottlenecks into answer bottlenecks — the first real "world model" behavior over open-vocab text.

## Key Results (In Progress)

| Experiment | IO Exact | QA tok_acc | Notes |
|-----------|:--------:|:----------:|-------|
| v19 initial (broken) | 80% plateau | — | adaLN init + missing LayerNorms |
| v19 fixed init | 93.9% @ ep200 | — | Warm-start conditioning proj + restore compressor LNs |
| v19 graduated curriculum | **96.9%** | — | 3-phase t-range: [0.7,1]→[0.4,1]→[0.0,1] |
| v19 dynamics (2L, frozen len) | 99.6% id | 12% qa | Dynamics core too small + length head frozen |
| v19 dynamics (4L, unfrozen len) | — | TBD | Current run |

## Infrastructure: Config-Driven Training

Replaced per-experiment scripts with a JSON config + `Trainer` class.

**New files:**
- `src/twm/training_config.py` — `TrainingConfig → StageConfig → PhaseConfig` dataclass hierarchy
- `src/twm/training_losses.py` — `compute_diffusion_loss()` for both IO and dynamics
- `src/twm/training_eval.py` — `assess()` + `print_samples()` with shared generation
- `src/twm/trainer.py` — `Trainer` orchestrator with stage/phase/freeze management
- `scripts/train.py` — 10-line universal entry point
- `configs/v19_mini64.json` — example config

**Key features:**
- Staged training: IO → dynamics with auto-checkpoint chaining
- Per-phase t-range curriculum, metric selection, patience
- Named freeze system: `["compressor", "expander"]` with auto-unfreeze of length head
- Shape-compatible partial weight loading between stages

## Bug Fixes & Findings

### 1. adaLN Initialization Symmetry Breaking

**Problem:** v19's factored adaLN has 3 projections (conditioning, timestep, position). All three were zero-initialized, creating a symmetry-breaking problem — the denoiser started with zero modulation from all sources, couldn't distinguish conditioning signal from noise.

**Fix:** Warm-start the conditioning projection (index 0) with default Kaiming init. Timestep and position projections stay zero-init. The model starts in v18's single-projection optimization landscape and grows into v19's factored design.

**File:** `src/twm/diffusion_decoder.py` — `AdaLNZeroLayer.__init__`

### 2. Compressor LayerNorms

**Problem:** Internal LayerNorms (`cross_ln`, `query_self_ln`, `query_ffn_ln`) were removed in v19 refactor. Without them, bottleneck vectors had inconsistent magnitudes, and the expander couldn't learn a stable mapping.

**Fix:** Restored all three LayerNorms in the extraction pipeline.

**File:** `src/twm/text_compressor.py`

### 3. Tokenizer Missing `?`

**Problem:** The BPE tokenizer was trained on declarative WebNLG sentences only. Questions (generated later by `generate_qa_dataset.py`) use `?`, which encoded as `<unk>` (id=2, zero embedding). 33.6% of identity_test examples had trailing `<unk>` — impossible to reconstruct.

**Diagnosis chain:**
- Systematic last-position errors in diagnostics: `(pos 15: 'aleks'!='')`
- Target decoded to empty because `_clean()` strips whitespace — initially suspected trailing space token (id=87)
- Only 1/5000 examples had trailing space → couldn't explain systematic pattern
- Found actual culprit: `?` → `<unk>` (id=2), and `decode([2])` → `''`
- 33.6% of test examples are questions ending with `?`

**Fix:** Added `initial_alphabet` to `BpeTrainer` in `prepare_webnlg_multimodal.py` ensuring `?` and other common punctuation always get vocab entries. Retrained tokenizer, regenerated all data. Result: 0/9614 test examples contain `<unk>`.

**Files:** `scripts/prepare_webnlg_multimodal.py`, all data in `data/webnlg_multi/`

### 4. Stochastic Eval Mismatch

**Problem:** `assess()` and `print_samples()` each called `model.generate()` independently. Generation starts from `torch.randn` noise, so the two calls produced different outputs. Metrics showed 89% exact but all 5 displayed samples were wrong — different random draws.

**Fix:** `assess()` now returns generation results via `_gen` key. `print_samples()` accepts `gen_cache` parameter to reuse the same generation. Diagnostics now reflect exactly what the metrics measured.

**File:** `src/twm/training_eval.py`, `src/twm/trainer.py`

### 5. Frozen Length Head During Dynamics

**Problem:** The length head is part of the expander. When `freeze: ["expander"]`, the length head freezes too. It was trained to predict length from identity bottlenecks (compressor outputs). Dynamics-transformed bottlenecks are out-of-distribution for the frozen length head.

**Loss decomposition:** Total loss 5.1, MSE 0.01, CE×0.1=0.35. Remaining ~4.7 is length loss. The frozen length head generates massive gradients that dominate training, and the dynamics core spends capacity satisfying the length head rather than learning QA transformations.

**Evidence:** id=0.996 (identity passes through unchanged, length head sees familiar bottlenecks), qa=0.12 (transformed bottlenecks are OOD for frozen length head).

**Fix:** Auto-unfreeze the length head when the expander is frozen. Added to `_apply_freeze()` in `trainer.py`. Length head is tiny (~8K params), adapts quickly to transformed bottleneck distribution.

**File:** `src/twm/trainer.py` — `_apply_freeze()`

### 6. Dynamics Core Capacity

**Problem:** Mini profile gives 2 transformer layers for dynamics. Identity is trivial (near-identity residual), but question→answer requires restructuring the bottleneck — different word order, length, and content. 2 layers couldn't learn the mapping: qa tok_acc plateaued at 12% after 560 epochs.

**Fix:** Added `dynamics_layers: 4` to config (top-level field, not per-stage). Doubles dynamics depth without touching frozen compressor/expander. IO checkpoint loads via shape-compatible partial loading; new dynamics layers stay randomly initialized.

### 7. ByteLevel BPE Position-Dependent Tokenization

**Problem:** ByteLevel BPE with `add_prefix_space=False` tokenizes the same word differently depending on position. At sentence start (no `Ġ` prefix), `amdavad` → `['amdavad']` (1 token). Mid-sentence (with `Ġ` prefix), `amdavad` → `['Ġam', 'davad']` (2 tokens). The dynamics core has to learn that these different token sequences represent the same word — an unnecessary burden.

**Scope:** Affects every QA pair where entities appear at different positions in question vs answer (essentially all of them).

**Fix:** Changed `add_prefix_space=False` → `add_prefix_space=True` in `train_bpe()`. Now every word gets a consistent `Ġ` prefix regardless of position. Retrained tokenizer, regenerated all data.

**Bonus:** Consistent prefix space lets BPE learn better merges. `ahmedabad` went from 3 tokens (`ah` + `med` + `abad`) to 1 token (`Ġahmedabad`). Shorter sequences = less work for the dynamics core.

**File:** `scripts/prepare_webnlg_multimodal.py` — `train_bpe()`

## Architecture Notes

### Graduated t-Range Curriculum

Instead of training on full noise range [0,1] from the start, use graduated phases:
1. **Phase 1** [0.7, 1.0] — high noise only, learn coarse structure. Gate on `tok_acc`.
2. **Phase 2** [0.4, 1.0] — extend into medium noise. Gate on `exact`.
3. **Phase 3** [0.0, 1.0] — full range, fine-tune boundaries. Gate on `exact`.

Each phase extends competence incrementally rather than shocking the model with low-noise gradients that destroy high-noise knowledge.

### Attention Pool vs Mean Pool

Mean pool (`bottleneck.mean(dim=1)`) divides gradients by N×3 (=36). Attention pool (learned query cross-attending to bottleneck) preserves per-position gradient flow. Both the conditioning vector and length head read from the same attention-pooled vector.

### Length Prediction Architecture

Length lives in the expander, not the compressor:
```
bottleneck → cond_attn pool → cond_proj → length_head (2-layer MLP) → scalar
```
During dynamics, the length head reads from the **post-dynamics** bottleneck, predicting the **output** length. The dynamics core must reshape the bottleneck geometry so the length head reads the correct output length.

### Natural MSE/CE Curriculum

MSE dominates early (large gradients when far from targets). CE becomes effective late (at cell boundaries). Explicit CE weight annealing fights this natural process. Keep CE weight constant.

## Dynamics Geometry Analysis (Pet Sim)

To understand *how* the dynamics core transforms state, we ran geometry analysis on the pet sim checkpoint (28K params, mini profile, 98.9% exact match). Tools: `scripts/visualize_dynamics.py` and `src/twm/analysis.py`.

### Latent Space Structure

![Latent space scatter](sprint4_figures/latent_space.png)

3,780 states (5 pets × 756 attribute/action combos), PCA to 3D (68.4% variance explained). Each pet starts from a tight pre-dynamics cluster, then fans out into a larger downstream region after the dynamics step. The inputs are encoded compactly, while most of the variation appears in the transition map itself. The downstream clouds differ by pet, so the model is not ignoring identity, but the overall geometry shows dog-specific variation around a common transition mechanism.

### Flow Field

![Flow field](sprint4_figures/flow_field.png)

PC1 vs PC2 with displacement arrows. Most transitions move in a broadly similar direction, with different magnitudes and branching angles. The learned dynamics have a dominant global transport component — a shared progression axis corresponding to "advance this state forward" — with smaller local deviations depending on which pet/state you started from.

### Jacobian Eigenspectrum

![Eigenspectrum](sprint4_figures/eigenspectrum.png)

Jacobian of the dynamics map at a representative state (Daisy, hungry, tired, content, messy, feed). 768×768 Jacobian (24 positions × 32 d_model).

- Eigenvalue magnitude range: [0.0006, 4.88]
- Mean |λ|: 1.0
- 304 expansive directions (|λ| > 1), 455 contractive (|λ| < 1)

The local operators are not simple contractions or random noise — they show heterogeneous structure, including expansive directions and coupled modes. This supports the claim that the core learned real latent dynamics.

### Takeaway

In the pet sim, the dynamics core learned one main next-state prediction function, with pet identity acting mostly as a conditioning signal that slightly changes the shape of the flow rather than selecting entirely different dynamics. This is consistent with the architecture's design: decomposed triples let the transformer share structure across entities, and the input residual means the dynamics only needs to learn the delta.

## Cedric Mode Geometry Probe (Micro vs Mini, closed-vocab)

Cedric is an OpenClaw assistant running on a hardened local LXC. He performs a number of monitoring, alert, reminder, automation, and report generation tasks. To sanity-check whether Mini is merely a larger Micro or actually learns a different latent organization, I ran a focused mode-conditioned geometry probe on a structured assistant dataset (`data/cedric_mode_probe_v2`) and compared both families with the same analysis tooling.

### Snapshot results

- Micro (`results/cedric_mode_probe_v2_micro`): comp/context F1 ~0.91 with notably lower exact-match on hard splits.
- Mini (`results/cedric_mode_probe_v2_mini`): 1.00 across F1/exact on current v2 splits.

### Geometry interpretation

- Micro shows stronger mode overlap/entanglement in post-dynamics latent projections.
- Mini shows clearer regime organization and cleaner mode-conditioned transport.
- Mode delta vector analysis (`mode_delta_vectors.png` + `mode_delta_stats.json`) indicates Mini operators are more coherent across states.

### Probe focus

For this sprint log, the key explanatory visuals are the paired micro-vs-mini probe plots:
- pre-dynamics latent embeddings,
- post-dynamics latent space by mode,
- mode delta vectors (vs identity).

These three views most directly answer the question we care about: whether Mini learns a cleaner mode-conditioned operator geometry than Micro.

### Recommendation

Use Mini as the default policy-reasoning core for mode-conditioned assistant behavior; keep Micro as a footprint-first fallback.

## Current Config

```json
{
    "model_type": "dynamics",
    "profile": "mini",
    "d_model": 64,
    "dynamics_layers": 4,
    "max_triples": 12,
    "text_compressor_layers": 3,
    "text_expander_layers": 3,
    "max_text_tokens": 64,
    "dropout": 0.1,
    "alpha_min": 0.01,
    "data_dir": "data/webnlg_multi",
    "tokenizer_path": "data/webnlg_multi/shared_bpe_tokenizer.json",
    "out_dir": "results/v19_mini64",
    "batch_size": 64,
    "denoise_steps": 10,
    "aux_ce_weight": 0.1,
    "length_weight": 0.25,
    "log_every": 10,
    "diagnostic_every": 50,
    "stages": [
        {
            "name": "io",
            "dataset": "identity",
            "lr": 3e-4,
            "max_examples": 15000,
            "phases": [
                {"t_min": 0.7, "t_max": 1.0, "epochs": 400, "patience": 100},
                {"t_min": 0.4, "t_max": 1.0, "epochs": 400, "patience": 100, "metric": "exact"},
                {"t_min": 0.0, "t_max": 1.0, "epochs": 800, "patience": 150, "metric": "exact"}
            ]
        },
        {
            "name": "dynamics",
            "dataset": "qa",
            "freeze": ["compressor", "expander"],
            "lr": 3e-4,
            "max_examples": 30000,
            "phases": [
                {"t_min": 0.0, "t_max": 1.0, "bias_power": 2.0, "epochs": 800, "patience": 150}
            ]
        }
    ]
}
```
