# Training Recipes

Config-driven training via `scripts/train.py`. Each recipe is a JSON config that defines the full experiment: model architecture, data, training stages, and curriculum.

```bash
uv run python scripts/train.py configs/your_config.json
```

## Profiles

| Profile | d_model | Layers | Heads | d_ff | Max Triples | Use Case |
|---------|--------:|-------:|------:|-----:|------------:|----------|
| micro   |      16 |      1 |     2 |   32 |           8 | Edge devices (ESP32, browser) |
| mini    |      32 |      2 |     2 |  128 |           8 | Fast iteration, proof of concept |
| base    |     256 |      4 |     4 | 1024 |           8 | Production, full accuracy |
| atomic  |     256 |      4 |     4 | 1024 |          12 | Large knowledge bases (12 triples) |

Override `d_model` and `max_triples` at the top level to scale a profile without changing layer/head counts.

## Key Concepts

**Stages** run sequentially. Each stage loads the previous stage's best checkpoint automatically.

**Phases** within a stage control the diffusion noise curriculum. The model learns coarse structure first (high noise), then refines (low noise).

**Freeze/unfreeze** controls which components train. The length head auto-unfreezes when the expander is frozen (set `"unfreeze": []` to disable).

**Metrics**: `tok_acc` (token-level accuracy) for early training, `exact` (full sequence match) once the model is in the high-accuracy regime.

---

## Recipe 1: Micro — Edge Device (ESP32 / Browser)

Closed-vocab model with fixed entity/attribute/value tokens. No compressor/expander needed. Tiny footprint (~29K params, <1MB weights).

Best for: fixed domain with known token set, client-side inference, IoT.

```json
{
    "model_type": "io",
    "profile": "micro",
    "max_triples": 4,
    "max_text_tokens": 12,
    "dropout": 0.05,
    "data_dir": "data/your_domain",
    "tokenizer_path": "data/your_domain/domain_bpe_tokenizer.json",
    "out_dir": "results/micro_edge",
    "batch_size": 128,
    "denoise_steps": 8,
    "aux_ce_weight": 0.1,
    "length_weight": 0.1,
    "log_every": 20,
    "diagnostic_every": 100,
    "stages": [
        {
            "name": "io",
            "dataset": "identity",
            "lr": 1e-3,
            "phases": [
                {"t_min": 0.5, "t_max": 1.0, "epochs": 200, "patience": 50},
                {"t_min": 0.0, "t_max": 1.0, "epochs": 400, "patience": 100, "metric": "exact"}
            ]
        }
    ]
}
```

**Notes:**
- Higher learning rate (1e-3) — small models benefit from aggressive updates.
- 2-phase curriculum is enough for small vocab. Skip the 3-phase graduated descent.
- `max_text_tokens: 12` keeps sequence length short for fast inference.
- `dropout: 0.05` — less regularization for tiny models that are already capacity-limited.
- Export weights to JSON for browser/embedded inference (see `demo/pet_simulation/export_weights.py`).

---

## Recipe 2: Mini IO — Compressor/Expander Training

Train the compressor and expander to encode and decode free text through a bottleneck. No dynamics — just identity reconstruction. This is stage 1 of any open-vocab experiment.

Best for: validating your data pipeline, testing tokenizer coverage, establishing an IO baseline before dynamics.

```json
{
    "model_type": "io",
    "profile": "mini",
    "d_model": 64,
    "max_triples": 12,
    "text_compressor_layers": 3,
    "text_expander_layers": 3,
    "max_text_tokens": 64,
    "dropout": 0.1,
    "alpha_min": 0.01,
    "data_dir": "data/webnlg_multi",
    "tokenizer_path": "data/webnlg_multi/shared_bpe_tokenizer.json",
    "out_dir": "results/mini_io",
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
        }
    ]
}
```

**Curriculum explained:**
1. **Phase 1** `[0.7, 1.0]` — High noise only. The model learns coarse token placement without worrying about fine boundaries. Gate on `tok_acc` because exact match is near-zero early on.
2. **Phase 2** `[0.4, 1.0]` — Extend into medium noise. The model refines token identity at moderate corruption levels. Gate on `exact` — the model should be getting some sequences perfectly right.
3. **Phase 3** `[0.0, 1.0]` — Full range. Fine-tune the boundary between "almost right" and "exact". Longer patience (150) because improvements are incremental.

**Expected trajectory:** tok_acc climbs quickly in phase 1 (>0.90 by epoch 100), exact match starts climbing in phase 2, reaches 90%+ by end of phase 3.

---

## Recipe 3: Mini Full — IO + Dynamics

End-to-end training: first learn to encode/decode, then freeze the IO pipeline and train the dynamics core to transform questions into answers.

Best for: open-vocab Q&A, knowledge graph reasoning, state transformations over natural language.

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
    "out_dir": "results/mini_full",
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

**Key design decisions:**
- `dynamics_layers: 4` — The dynamics core needs more depth than IO. Question-to-answer requires restructuring the bottleneck (different word order, length, content). 2 layers is not enough.
- `freeze: ["compressor", "expander"]` — The IO pipeline is fixed. Only the dynamics core + length head train.
- The length head auto-unfreezes (even though it's part of the expander) because it must adapt to transformed bottlenecks. To disable: add `"unfreeze": []`.
- `bias_power: 2.0` — Importance sampling that focuses on harder (higher noise) timesteps.
- Stage 2 auto-loads the best checkpoint from `io_phase3/model_best.pt`.

**Skipping IO training:** If you already have an IO checkpoint, remove the IO stage and point directly:
```json
{
    "stages": [
        {
            "name": "dynamics",
            "pretrained": "results/mini_io/io_phase3/model_best.pt",
            "dataset": "qa",
            "freeze": ["compressor", "expander"],
            ...
        }
    ]
}
```

---

## Recipe 4: Base — Full Scale

Production-quality model with full-size architecture. Slower to train but higher ceiling.

```json
{
    "model_type": "dynamics",
    "profile": "base",
    "dynamics_layers": 6,
    "max_triples": 12,
    "text_compressor_layers": 4,
    "text_expander_layers": 4,
    "max_text_tokens": 128,
    "dropout": 0.1,
    "alpha_min": 0.01,
    "data_dir": "data/webnlg_multi",
    "tokenizer_path": "data/webnlg_multi/shared_bpe_tokenizer.json",
    "out_dir": "results/base_full",
    "batch_size": 32,
    "denoise_steps": 20,
    "aux_ce_weight": 0.1,
    "length_weight": 0.25,
    "log_every": 5,
    "diagnostic_every": 25,
    "stages": [
        {
            "name": "io",
            "dataset": "identity",
            "lr": 1e-4,
            "phases": [
                {"t_min": 0.7, "t_max": 1.0, "epochs": 200, "patience": 50},
                {"t_min": 0.4, "t_max": 1.0, "epochs": 200, "patience": 50, "metric": "exact"},
                {"t_min": 0.0, "t_max": 1.0, "epochs": 400, "patience": 100, "metric": "exact"}
            ]
        },
        {
            "name": "dynamics",
            "dataset": "qa",
            "freeze": ["compressor", "expander"],
            "lr": 1e-4,
            "phases": [
                {"t_min": 0.0, "t_max": 1.0, "bias_power": 2.0, "epochs": 400, "patience": 100}
            ]
        }
    ],
    "device": "cuda:0"
}
```

**Notes:**
- Lower learning rate (1e-4) — larger models are more sensitive to LR.
- Smaller batch size (32) — fits in GPU memory with d_model=256.
- More denoising steps (20) — larger embedding space benefits from finer denoising.
- `dynamics_layers: 6` — more depth than profile default (4) for complex transformations.
- `max_text_tokens: 128` — longer text sequences for richer descriptions.
- Fewer epochs needed — larger models converge faster per epoch.

---

## Curriculum Design Guide

### When to use graduated t-range

Use 3 phases when training compressor/expander (IO stage). The denoiser must learn to handle different corruption levels, and dumping the full range at once causes low-noise gradients to destroy high-noise knowledge.

| Phase | t-range | Gate metric | Why |
|-------|---------|-------------|-----|
| 1 | [0.7, 1.0] | tok_acc | Learn coarse structure from heavy corruption |
| 2 | [0.4, 1.0] | exact | Extend into medium noise, refine token identity |
| 3 | [0.0, 1.0] | exact | Full range, fine-tune boundaries |

### When to skip graduated t-range

- **Dynamics stage**: The denoiser is frozen. Only the dynamics core trains. Use full range `[0.0, 1.0]` with `bias_power: 2.0`.
- **Micro/edge models**: Small vocab + short sequences. 2 phases is enough: `[0.5, 1.0]` then `[0.0, 1.0]`.
- **Fine-tuning**: If starting from a pretrained IO checkpoint, the denoiser already handles all noise levels. Use full range.

### Bias power

`bias_power` > 1.0 samples higher timesteps more often (importance sampling). Use 2.0 for dynamics training where the model needs to handle the full noise range but harder examples (high noise) are more informative.

### Patience and metric selection

- **Phase 1 (from scratch):** Use `tok_acc` with moderate patience (50-100). Exact match is near-zero early — gating on it would stop training immediately.
- **Later phases:** Switch to `exact`. The model is in the high-accuracy regime where `tok_acc` saturates but exact match keeps climbing.
- **Set patience to 0** to disable early stopping (run all epochs). Useful for ablations.

## Freeze/Unfreeze Reference

| Component | What it covers |
|-----------|---------------|
| `compressor` | Text compressor (BPE → bottleneck) |
| `expander` | Text expander (bottleneck → BPE via denoising), including length head |
| `dynamics` | Dynamics transformer + mode embeddings |
| `embeddings` | Shared token embedding table (always frozen by default) |

**Unfreeze overrides** (applied after freeze):

| Override | What it does |
|----------|-------------|
| `length_head` | Unfreezes the length prediction head inside the expander |

**Auto-behaviors:**
- Dynamics is auto-frozen during IO stages (identity dataset), even if not listed in `freeze`.
- Length head is auto-unfrozen when expander is frozen (it must adapt to transformed bottlenecks). Set `"unfreeze": []` to disable this — useful for uniform-length data where the length head doesn't need to adapt.

## Data Preparation

1. **Prepare raw data + tokenizer:**
   ```bash
   uv run python scripts/prepare_webnlg_multimodal.py --out-dir data/webnlg_multi
   ```

2. **Generate identity + QA datasets:**
   ```bash
   uv run python scripts/generate_qa_dataset.py \
       --input data/webnlg_multi/train.jsonl \
       --output-dir data/webnlg_multi --split train

   uv run python scripts/generate_qa_dataset.py \
       --input data/webnlg_multi/test.jsonl \
       --output-dir data/webnlg_multi --split test
   ```

This produces:
- `identity_train.jsonl` / `identity_test.jsonl` — every unique text paired with itself
- `qa_train.jsonl` / `qa_test.jsonl` — question→answer pairs + identity pairs
- `shared_bpe_tokenizer.json` — BPE tokenizer covering all text
