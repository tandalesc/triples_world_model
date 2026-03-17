# Triple World Model (TWM)

## What This Is

A minimal world model that learns state dynamics over structured (entity, attribute, value) triples using a vanilla transformer. The core claim: a small transformer over decomposed triple tokens can learn compositional state transformations that generalize to novel entity-state combinations never seen in training — and it needs cross-position attention to do it.

Compositional generalization is confirmed. The current focus is extending to open-vocabulary values via a compressor/expander architecture.

## Architecture

At the center is the **dynamics core** — a transformer that processes triples in latent space. You can use it directly with a fixed token set, or wrap it with a **compressor/expander** for open-vocabulary and other use cases.

```
  Direct (fixed token set)              With I/O wrappers (open-vocab)
  ────────────────────────              ─────────────────────────────
  token IDs → Encoder                  BPE text → Compressor
                  │                                    │
                  ▼                                    ▼
             ┌──────────┐                        ┌──────────┐
             │ Dynamics │                        │ Dynamics │
             │  (core)  │                        │  (core)  │
             └────┬─────┘                        └────┬─────┘
                  │                                    │
                  ▼                                    ▼
         Decoder → logits               Expander → BPE text
                                        (iterative denoising)
```

The dynamics core sees the same shaped input either way: `(B, max_triples × 3, d_model)`.

### Terminology

| Term | What It Is | Code |
|------|-----------|------|
| **Dynamics** | Transformer core. The world model. | `TransformerDynamics` |
| **Encoder/Decoder** | Thin wrappers for fixed-vocab I/O | `TripleEncoder`, `TripleDecoder` |
| **Compressor** | BPE tokens → 256d latent per slot | `TripleCompressor` |
| **Expander** | 256d latent → BPE tokens via denoising | `DiffusionDecoder` |
| **Denoiser** | Transformer layers inside the expander | (internal to `DiffusionDecoder`) |

### Mode Conditioning

A mode triple `(#mode, type, advance)` is prepended as a regular triple — no architecture changes. The transformer learns to condition on it. Modes are just training data:
- `advance`: predict state after transformation
- `identity`: predict same state (validates reconstruction)
- Future: `query`, `instruct`, etc.

## Key Design Decisions

- **Decomposed triples, not sentence embeddings.** Each entity/attribute/value is its own token. Compositionality comes from structure, not embedding space.
- **Set-to-set, not autoregressive.** Parallel prediction of all output positions. Closer to BERT than GPT.
- **Embedding-agnostic.** The dynamics core doesn't prescribe an embedding space. Closed-vocab uses learned embeddings; open-vocab uses BPE compressor/expander.
- **Input residual.** Most state persists across transformations. The model learns the delta. In the open-vocab path, `forward_dynamics` returns `bottleneck + out_gate(transformer(x))` where the gate is zero-initialized for identity at init.

## Config Profiles

| Profile | d_model | Layers | Heads | d_ff | Max Triples |
|---------|--------:|-------:|------:|-----:|------------:|
| base | 256 | 4 | 4 | 1024 | 8 |
| mini | 32 | 2 | 2 | 128 | 8 |
| micro | 16 | 1 | 2 | 32 | 8 |
| atomic | 256 | 4 | 4 | 1024 | 12 |

## Project Structure

```
├── CLAUDE.md
├── README.md
├── src/twm/
│   ├── config.py            # ModelConfig with profiles
│   ├── modules.py           # TripleEncoder, TransformerDynamics, TripleDecoder
│   ├── model.py             # TripleWorldModel (closed-vocab wrapper)
│   ├── compressor.py        # TripleCompressor (open-vocab input)
│   ├── diffusion_decoder.py # DiffusionDecoder / expander (open-vocab output)
│   ├── diffusion_model.py   # DiffusionWorldModel (open-vocab wrapper)
│   ├── dataset.py           # Triple dataset + collation
│   ├── sentence_dataset.py  # Sentence-level dataset for open-vocab
│   ├── train.py             # Training loop
│   ├── eval.py              # Evaluation + attention visualization
│   ├── analysis.py          # Dynamics geometry tools (Jacobian, flow field)
│   ├── serve.py             # Inference server
│   └── vocab.py             # Vocabulary builder
├── scripts/                 # Training and plotting scripts
├── data/                    # Training data (JSONL triple pairs)
├── demo/pet_simulation/     # Client-side JS inference demo (303 KB)
├── results/                 # Checkpoints, logs, plots per experiment
└── research/                # Architecture docs, references, experiment logs
    ├── architecture.md      # Full architecture with diagrams
    ├── references.md        # Papers and systems referenced
    ├── theoretical_foundations.md  # Geometric framework
    └── sprint3_diffusion_decoder.md  # Experiment log
```

## Key Results

- **Compositional generalization**: 0.74 F1 on novel entity-state combos (3-domain benchmark)
- **Context-dependent reasoning**: 0.978 F1 where output depends on other triples present
- **Mini matches Base**: 178K params (25x smaller) gets identical context-dep F1
- **Micro is viable**: 80K params (57x smaller), 0.91 context-dep F1
- **Pet sim demo**: 29K params, 98.9% exact match, runs client-side in 303 KB JS
- **Open-vocab (staged)**: 81.1% exact match on ATOMIC 10K with compressor/expander (identity mode)
- **Open-vocab (joint+dynamics, VAE)**: 19% exact / 73% tok_acc on WebNLG IO with VAE bottleneck (v25); 26% QA tok_acc with dynamics (v27)
- **Open-vocab (no VAE)**: 96.9% exact / 99.3% tok_acc on WebNLG IO (v35, d64, 16 triples). 36.8% QA tok_acc. Best architecture: no VAE, joint training, spectral penalty, 16 triples, t_min=0.5/0.3 schedule.
- **Triples sweep (d32)**: 4→57%, 8→74%, 12→83%, 16→85% IO tok_acc. Big jump at 8→12 triples. QA needs d64+ (d32 caps at ~8% regardless of triples).
- **Beats frontier LLMs**: 100% attr accuracy vs 4-8/8 for Claude/Gemini/GPT on ATOMIC

## VAE + Diffusion Gotchas

- **Spectral loss must measure mu, not z.** VAE sampling noise masks bottleneck collapse. Always compute geometry metrics on the deterministic mu.
- **Condition diffusion expander on mu, not z.** Double-noise (VAE + diffusion) causes train/eval mismatch — the expander learns to denoise against noisy conditioning but sees clean mu at eval, converging to a single attractor.
- **Length head reads pre-dynamics mu.** Length is a property of the input, not the transformation. Reading post-dynamics bottleneck makes the length head chase a moving target.
- **Joint training prevents bottleneck collapse.** Staged IO→dynamics fails because the compressor collapses to 1D before dynamics arrives. Joint training (dynamics from epoch 1 with zero-init gate) keeps spectral loss at 0.04 vs 1.0 for staged. Use `StageConfig.joint=true`.
- **Warmup is counterproductive.** Pre-trained IO geometry gets destroyed when random dynamics comes online. Co-evolution from scratch works better.
- **Drop the VAE.** With joint training + spectral penalty, VAE is pure overhead. No-VAE (v30) trains 5-10x faster and reaches 95% tok_acc / 61% exact in IO phase 1. The VAE introduced 3 bugs (spectral on noise, double-noise mismatch, z/mu divergence) that took hours to fix. Without VAE, the bottleneck is deterministic — no train/eval mismatch, no noise masking collapse.
- **Eval must route through dynamics for joint IO.** In joint training, identity data routes through dynamics with mode=0. If eval skips dynamics for TextDataset, the length head and expander see a different bottleneck than training — causes systematic N-1 length on questions. Fixed in `training_eval.py`.
- **Use t_min=0.5/0.3, patience 200.** The old schedule (t_min=0.7/0.4, patience 100) early-stops before geometry is built. Lower noise from the start lets tokens resolve in phase 1. v35 hit 92% exact at ep180 where v31 needed 400+ epochs for 83%.
- **Staged IO→QA causes geometry collapse.** IO-first builds a 1D+noise manifold. When QA arrives, dynamics collapses the geometry for cheap mode separation instead of learning transforms (PC1 0.28→0.57). Freezing compressor preserves geometry but limits QA to ~8%. Next approach: joint identity+QA from epoch 1 (v37, untested).

## Training

Config-driven via JSON: `uv run python scripts/train.py configs/<name>.json`
Training configs define stages (io, joint_io, dynamics) with phases (graduated noise curriculum).
Key configs: `v35_d64_t16.json` (best IO model — 99.3% tok, 36.8% QA tok), `v37_joint_all.json` (next experiment — joint identity+QA from epoch 1).
Submit to GPU server via wartable MCP: `mcp__wartable__submit_job`.

## Data Format

```json
{"state_t": [["glass", "state", "full"], ["person", "state", "thirsty"]], "state_t+1": [["glass", "state", "empty"], ["person", "state", "satisfied"]]}
```

Test splits: `test_comp` (novel combos), `test_seen` (seen combos), `test_context` (cross-entity).

## Hardware

Dual RTX 3090. Models are small enough that CPU training works for closed-vocab experiments.
