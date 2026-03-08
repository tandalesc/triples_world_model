# Sprint 3: Diffusion Decoder (Expander)

> **Terminology note:** This log predates our standardized naming. "Decoder" here
> refers to what we now call the **expander** (latent → BPE tokens via denoising).
> "Encoder" refers to the **compressor** (BPE tokens → latent). See
> [architecture.md](architecture.md) for the full terminology map.

## Expander Architecture

The expander reconstructs BPE tokens from the dynamics core's 256d latent vectors
via iterative denoising. Here's the full data flow:

```
                         COMPRESSOR (input)
                         ─────────────────
  "to be helpful"        BPE tokenize: [to, be, help, ful]
       │
       ▼
  ┌──────────────┐
  │ Frozen BPE   │       Look up frozen token embeddings
  │ Embeddings   │       (shared with expander)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 2L Self-Attn │       Contextualize within slot
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ Role Pool    │       Cross-attn with learned query
  │ (query)      │       → single 256d vector
  └──────┬───────┘
         │
         ▼
     1 × 256d            One vector per triple slot
         │                (entity, attr, value each get one)
         │
─────────┼───────────────────────────────────────────────
         │
         ▼
  ┌──────────────┐
  │  Dynamics    │       TransformerEncoder over all slots
  │   Core       │       (frozen or trainable)
  │  256d → 256d │       Attends across all triple positions
  └──────┬───────┘
         │
─────────┼───────────────────────────────────────────────
         │
         ▼               EXPANDER (output)
     1 × 256d            ─────────────────
  ┌──────┴───────┐
  │  Length Head  │──→ predicted token count (e.g., 4)
  │  (256 params)│     used for truncation at inference
  └──────────────┘

  Per denoising step (T=50 steps at inference):
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   x_noisy = sqrt(α) · x_clean + sqrt(1-α) · noise  │
  │   (at training: random t, at inference: t=1→0)      │
  │                                                     │
  │        x_noisy (S positions, 256d each)             │
  │              │                                      │
  │              ▼                                      │
  │   ┌────────────────────┐                            │
  │   │ + Position Emb     │  (noise-free, via adaLN)   │
  │   └────────┬───────────┘                            │
  │            │                                        │
  │            ▼              conditioning from         │
  │   ┌─────────────────┐    dynamics (256d)            │
  │   │  adaLN-Zero     │◄──────────────────────┐       │
  │   │  Self-Attention │    modulates γ,β,gate │       │
  │   └────────┬────────┘                       │       │
  │            │                                │       │
  │            ▼              W-space memory    │       │
  │   ┌─────────────────┐    (3 × 256d)        │       │
  │   │  adaLN-Zero     │◄─────────────────────┤       │
  │   │  Cross-Attention│    attends to triple  │       │
  │   └────────┬────────┘    slot context       │       │
  │            │                                │       │
  │            ▼                                │       │
  │   ┌─────────────────┐                       │       │
  │   │  adaLN-Zero     │◄─────────────────────┘       │
  │   │  FFN            │                               │
  │   └────────┬────────┘                               │
  │            │                                        │
  │            ▼                                        │
  │      x_pred (predicted clean embeddings)            │
  │            │                                        │
  │   ×1-3 layers (depth = denoiser depth)              │
  └─────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────┐
  │  Nearest-Neighbor│   cosine similarity against
  │  Lookup          │   frozen BPE embedding table
  │                  │   → closest token per position
  └──────┬───────────┘
         │
         ▼
  [to, be, help, ful]   Truncate to length head prediction
         │
         ▼
  "to be helpful"       Detokenize
```

### Key design choices visible in the diagram

- **Frozen BPE embeddings** are shared between compressor and expander.
  The expander's NN lookup searches the same table the compressor reads from.
- **Position routes through adaLN**, not through the noisy input. At high noise,
  positional embeddings added to x_noisy would be buried. adaLN is noise-free.
- **Cross-attention** connects each denoising position to the W-space conditioning
  (the dynamics output for that triple slot). This is where the expander learns
  *what* to reconstruct.
- **adaLN-Zero** modulates *how* to reconstruct — the gate starts at zero
  (identity init) and gradually turns on during training.
- **x0-prediction**: the denoiser predicts the clean embedding directly (not the noise).
  MSE loss in embedding space. At inference, predicted embeddings are decoded via NN lookup.

## Summary

Built a continuous diffusion decoder that reconstructs natural language from TWM's compressed triple representations. Started with discrete masked diffusion, iterated through 12 major design problems, arrived at a BPE compressor/expander architecture achieving 81.1% exact match on 10K examples.

## Key Results

| Phase | Architecture | Gen Exact | Gen Token | Notes |
|-------|-------------|:---------:|:---------:|-------|
| 1 | Sentence enc + discrete masking | — | 48.6%* | *Conditional completion (some visible tokens), not generation |
| 2 | Sentence enc + continuous noise | 0% | 21.7% | First gen-from-scratch metric; solved conditioning collapse |
| 3 | BPE compressor/expander (1L) | 65.0% | 91.5% | Token-level enc/dec fixes meaning→spelling gap |
| 3 | + length head + 3L denoiser | **81.1%** | **93.4%** | Depth + truncation |

Note: Metrics changed between phases. Phase 1 measured conditional completion (50% visible tokens).
Phase 2+ measured generation from scratch (0% visible tokens). Numbers are not directly comparable.

## Detailed Experiment Tables

### Phase 1: Discrete Masking

| Run | Denoiser | Data | Pretrained | Peak Test Val Tok | Test Attr |
|-----|----------|------|:---:|:-:|:-:|
| 1L/128d | 1L, 128d | 2K | No | 37.2% | 72% |
| 1L/128d | 1L, 128d | 10K | No | 42.7% | 74% |
| 2L/256d | 2L, 256d | 10K | Yes (frozen) | 48.6% | 76% |
| 1L/128d | 1L, 128d | 10K | Yes (frozen) | 47.0% | 76% |

### Phase 2: Continuous Noise (gen_val_tok — generation from scratch)

| Run | Change | gen_val_tok | gen_val_exact | Notes |
|-----|--------|:-:|:-:|-------|
| v6 baseline | Cross-attn only | 14.2% | 0% | Peaked epoch 10, flatlined |
| v6 FiLM | + FiLM conditioning | 15.8% | 0% | Slow climb, still below ceiling |
| v7 adaLN | + adaLN-Zero | 17.4% | 0% | Higher peak, same decline pattern |
| v9 cosine | Continuous noise (cosine schedule) | 20.0% | 0% | First run without conditioning collapse |
| v9 alpha_min | + alpha_min=0.01, importance sampling | 21.0% | 0% | Stable at ~21%, no decline |
| v10 unified | Unified decoder, 6.5M params | 21.7% | 0% | Matched quality at half parameters |

### Phase 3: Compressor/Expander Identity

| Run | Decoder | Encoder | Val Exact | Val Tok | Ent Exact | Params |
|-----|---------|---------|:-:|:-:|:-:|-------:|
| v13 sentence-enc | 1L, 256d | Sentence-transformer | 0% | 34% | 0% | ~6.5M |
| v13 identity (WebNLG) | 1L, 256d | BPE compressor 2L | **100%** | **100%** | **100%** | ~10M |
| v15a 1L (ATOMIC) | 1L, 256d | BPE compressor 2L | 55.1% | 80.5% | 62.0% | ~10M |
| v15b 2L (ATOMIC) | 2L, 256d | BPE compressor 2L | 71.1% | 89.4% | 63.3% | ~11M |
| v15c 3L (ATOMIC) | 3L, 256d | BPE compressor 2L | **81.1%** | **93.4%** | **75.5%** | ~12M |

#### Per-length breakdown (v15c 3L, best run)

| Phrase Length | Exact Match | Examples |
|--------------|:-:|:-:|
| Short (1-3 BPE tokens) | 100% | 125 |
| Medium (4-6 BPE tokens) | 89% | 222 |
| Long (7+ BPE tokens) | 46% | 114 |

## Problem/Solution Log

### 1. Discrete masking discontinuity
**Problem:** Masked discrete diffusion has a hard discontinuity at mask_ratio=1.0. At 0.95, a few real tokens are visible — the model denoises. At 1.0, all tokens are identical MASK embeddings — the model must generate from nothing. These are qualitatively different tasks. The model learns denoising but can't generate.

**Solution:** Replace discrete masking with continuous Gaussian noise in embedding space. The corruption `x_t = sqrt(α) * x_0 + sqrt(1-α) * noise` always preserves a faint residual of the original signal, even at high noise. The task is the same at every noise level, just harder.

### 2. Loss cliff at t=1.0
**Problem:** Even with continuous noise, the cosine schedule produced a 25-50x loss cliff between t=0.9 and t=1.0. The model solved t=0.0 through t=0.8 trivially (loss ~0.10) and couldn't touch t=1.0 (loss ~4.4). 95% of training compute was wasted on already-solved timesteps.

**Solution:** Clamp alpha_min=0.01 so the signal never fully disappears. Combined with importance sampling (bias_power=2.0) to allocate more training to high-noise timesteps. The loss-vs-timestep curve went from a 50x cliff to a smooth 3x gradient.

### 3. Conditioning collapse (path A vs path B)
**Problem:** The model has two information pathways: visible/corrupted tokens (path A) and TWM conditioning (path B). Path A is always easier. Gradient descent follows least resistance, so the model learns to ignore conditioning. gen_val_tok peaks at epoch 10 and declines as the model commits to path A.

**Solution:** Multiple fixes contributed. Continuous noise helped (no discrete shortcut). alpha_min helped (no singularity). Ultimately, joint encoder-decoder training via the compressor/expander was what fully solved it — the conditioning pathway becomes strong enough naturally when both sides co-train.

### 4. Metric confusion: token accuracy vs exact match
**Problem:** We spent three experiment rounds (v3-v5) debugging architecture based on gen_val=2%. A diagnostic showed 65% on the same checkpoint. Turns out gen_val measured exact phrase match while the diagnostic measured per-token accuracy. Both were real numbers measuring different things. 65% token accuracy on 16-position sequences produces ~2% exact phrase match.

**Solution:** Log both gen_val_tok (per-token) and gen_val_exact (phrase-level) going forward. Early stop on the metric you actually care about.

### 5. Early stopping on the wrong metric
**Problem:** We early-stopped on test loss, which measures conditional completion quality (50% visible tokens). Generation quality (0% visible tokens) is anti-correlated with test loss after early training — the model gets more specific and confident (worse test loss) but better at generation.

**Solution:** Early stop on gen_val_tok (generation from scratch), not test loss. Later switched to gen_val_exact when that became the real target.

### 6. Trainable embedding collapse
**Problem:** Under continuous noise, trainable token embeddings collapse into a narrow cone during training. Mean pairwise cosine went from 0.21 to 0.96 in 30 epochs. The model satisfies the loss by pointing at the centroid of the collapsed space — no conditioning needed because all targets look the same.

**Solution:** Freeze the token embeddings. The fixed landmarks force the model to predict precise directions for specific tokens, which requires conditioning. T5 embeddings were accidentally frozen the whole time — that's why T5 worked.

### 7. MSE on frozen embeddings enabled gaming
**Problem:** With frozen embeddings in W-space, MSE loss at moderate noise still allowed the model to reconstruct from the corrupted input alone (proximity shortcut). The corrupted embedding was still closest to its clean origin in the shared semantic space.

**Solution:** Not fully solved through loss function changes alone. Classifier-free guidance (CFG) with 15% conditioning dropout partially helped. Ultimately, the compressor/expander architecture dissolved the problem by making the conditioning pathway so strong through joint training that the model uses it natively (g=1.0 always optimal).

### 8. Sentence encoder meaning→spelling mismatch
**Problem:** The sentence-transformer encoder produces one 256d vector per phrase capturing meaning ("transcripts" ≈ "records") but not spelling. The decoder produces BPE tokens that require spelling. Result: "prove" → "show" (same meaning, wrong word). Entities fail on unseen strings because meaning→spelling doesn't generalize.

**Solution:** Replace sentence encoder with a BPE-level compressor. Both sides operate at token granularity. The compressor reads BPE tokens, contextualizes through self-attention, and compresses to one 256d vector via learned pool query. The reconstruction loss teaches it to preserve spelling information, not just meaning.

### 9. Pad position garbage
**Problem:** Pad positions were unsupervised (loss detached, zero-vector target with undefined cosine similarity for NN decode). The model produced real-token-like embeddings at pad positions, causing trailing garbage in decoded strings. 91.5% token accuracy with 0% exact match because every prediction had correct content plus trailing nonsense.

**Solution:** Give pad a real unit-norm embedding like any other token. Supervise pad positions through the main loss path. Add a length prediction head (256 params) that predicts the number of real tokens from the conditioning vector. Truncate predictions at inference. Length head learned instantly (100% accuracy).

### 10. Over-engineering the pad fix
**Problem:** We tried to fix pad behavior by changing the loss function, the pad embedding, the generation loop, and the supervision path simultaneously. Every change broke something else. The model went from 67.7% exact match to 6%.

**Solution:** Revert everything. Go back to the working checkpoint. Add only the length head for truncation. The model already generates correct content — it just doesn't know when to stop. That's a 256-parameter post-processing fix, not an architectural redesign.

### 11. Premature convergence calls
**Problem:** We repeatedly declared the model "converged" or "plateaued" and stopped training or changed architecture. Every time, longer training proved the model was still improving. The 2L run went from 71.1% at epoch 200 to 81.1% at epoch 320.

**Solution:** Set longer max epochs. Be patient. The model trains slowly on hard examples (long phrases) and the gains are invisible until they accumulate enough for exact matches to flip from wrong to right.

### 12. Denoiser depth matters for long sequences
**Problem:** 1L denoiser got the right tokens but couldn't order them correctly for longer phrases. One round of cross-attention and adaLN couldn't resolve 7+ BPE token positions.

**Solution:** Deeper denoiser. 1L→2L: +16% exact match. 2L→3L: +0.5% value but +8% entity. Each layer adds a refinement round. Per-length-bucket metrics revealed the gains concentrate on medium and long phrases.

## Distilled Principles

1. **Continuous > discrete** — Gaussian noise eliminates the masking task discontinuity (#1)
2. **Freeze embeddings** — trainable embeddings collapse under continuous noise (#6)
3. **Spelling ≠ meaning** — BPE compression preserves orthography; sentence encoders don't (#8)
4. **Depth buys ordering** — each denoiser layer is a position refinement pass (#12)
5. **Change one thing** — multi-fix went 67.7% → 6%; single-fix (length head) solved it (#10)
6. **Measure what matters** — 65% token accuracy ≈ 2% exact match (#4)
7. **Be patient** — every "plateau" was premature (#11)

See also: [theoretical foundations](theoretical_foundations.md) for geometric justification of principles 1-3.
