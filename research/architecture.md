# Architecture

At the center of TWM is the **dynamics core** — a transformer that processes triples
in latent space. You can use it directly with a fixed token set, or wrap it with
a **compressor/expander** pair for open-vocabulary and other use cases.

## Terminology

| Term | What It Is | Code |
|------|-----------|------|
| **Dynamics** | The transformer that processes triples in latent space. The core world model. | `TransformerDynamics` |
| **Compressor** | BPE tokens → fixed-width 256d latent per triple slot. Open-vocab input stage. | `TripleCompressor` |
| **Expander** | 256d latent → BPE tokens via iterative denoising. Open-vocab output stage. | `DiffusionDecoder` |
| **Denoiser** | The transformer layers inside the expander that refine noisy embeddings. Not a separate class — it's the expander's internal mechanism. | (inside `DiffusionDecoder`) |
| **Encoder** | Token IDs → latent vectors. Closed-vocab input stage. | `TripleEncoder` |
| **Decoder** | Latent vectors → token logits. Closed-vocab output stage. | `TripleDecoder` |

"Encoder/decoder" is the general concept. "Compressor/expander" is the specific
open-vocabulary implementation. "Denoiser" is the iterative refinement process
inside the expander.

## Using the Core Directly

With a fixed token set (pet simulator, family benchmark), you use the dynamics core
directly with thin encoder/decoder wrappers.

```
  Input token IDs          Output token logits
       │                         ▲
       ▼                         │
  ┌─────────┐              ┌─────────┐
  │ Encoder │              │ Decoder │
  │         │              │         │
  │ tok IDs │              │ latent  │
  │   → d   │              │  → V    │
  └────┬────┘              └────┬────┘
       │     ┌───────────┐      │
       └────►│ Dynamics  ├──────┘
             │           │
             │ d → d     │
             │ (+ resid) │
             └───────────┘

  Encoder:  nn.Embedding(V, d) + positional encoding
  Dynamics: TransformerEncoder (n layers, n heads, d_model)
  Decoder:  LayerNorm + Linear(d, V) + input residual skip
```

Everything is discrete tokens. Cross-entropy loss. The encoder and decoder are thin
wrappers — most of the parameters are in the dynamics core and the embedding table.

**Example sizes:**
- Mini (pet sim): V=53, d=32, 2 layers → 29K params
- Base (3-domain): V=2340, d=256, 4 layers → 4.5M params

## With Compressor/Expander Wrappers

For open-vocabulary domains (ATOMIC, free-text values), wrap the dynamics core
with a **compressor** (input) and **expander** (output) that handle variable-length
BPE token sequences.

```
  BPE tokens per slot (up to 12)     Reconstructed BPE tokens
       │                                      ▲
       ▼                                      │
  ┌────────────┐                      ┌───────────────┐
  │ Compressor │                      │   Expander    │
  │            │                      │  (Denoiser)   │
  │ S tokens   │                      │               │
  │   → 1×256d │                      │ 1×256d → S tok│
  └─────┬──────┘                      └───────┬───────┘
        │     ┌───────────────────┐           │
        └────►│    Dynamics       ├───────────┘
              │  (frozen or not)  │
              │                   │
              │    256d → 256d    │
              └───────────────────┘

  Compressor: frozen BPE embeddings → 2L self-attention → learned pool query → 256d
  Dynamics:   same TransformerEncoder as closed-vocab (operates on 256d latents)
  Expander:   conditioned denoising transformer (1-3 layers)
              cross-attention + adaLN-Zero conditioning from dynamics output
              iterative refinement: noise → clean BPE embeddings → nearest-neighbor decode
  Length head: predicts token count per slot for truncation (256 params)
```

The dynamics core sees **identical input** regardless of pipeline — it always operates
on (B, max_triples × 3, 256) latent tensors. The compressor/expander pair is
transparent to it.

### Full Open-Vocab Data Flow

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
     3×N × 256d          One vector per slot (entity, attr, value)
         │                × N triples (scales with max_triples)
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
         ▼               EXPANDER (output, per slot)
     3×N × 256d          ─────────────────
  ┌──────┴───────┐
  │  Length Head │──→  predicted token count (e.g., 4)
  │  (256 params)│     used for truncation at inference
  └──────────────┘

  Per denoising step (T=50 steps at inference):
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   x_noisy = sqrt(α) · x_clean + sqrt(1-α) · noise   │
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
  │   ┌─────────────────┐    (3 × 256d)         │       │
  │   │  adaLN-Zero     │◄──────────────────────┤       │
  │   │  Cross-Attention│    attends to triple  │       │
  │   └────────┬────────┘    slot context       │       │
  │            │                                │       │
  │            ▼                                │       │
  │   ┌─────────────────┐                       │       │
  │   │  adaLN-Zero     │◄──────────────────────┘       │
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

### Key design choices

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

## How They Connect

```
                    ┌─────────────────────────────────────────┐
                    │           TWM Dynamics Core             │
                    │     (same architecture either way)      │
                    └──────────┬──────────────────┬───────────┘
                               │                  │
              ┌────────────────┴──┐          ┌────┴────────────────┐
              │  Direct I/O       │          │   Wrapped I/O       │
              │                   │          │                     │
              │  Encoder: tok→d   │          │  Compressor: BPE→d  │
              │  Decoder: d→tok   │          │  Expander:   d→BPE  │
              │                   │          │                     │
              │  (thin wrappers)  │          │  (learned pipeline) │
              └───────────────────┘          └─────────────────────┘
```

The dynamics core is the world model. The I/O layers are interchangeable interfaces.
A pet sim TWM and an ATOMIC TWM share the same dynamics architecture — only the
I/O differs.

## File Map

```
src/twm/
├── modules.py          # TripleEncoder, TransformerDynamics, TripleDecoder
├── model.py            # TripleWorldModel (closed-vocab wrapper)
├── compressor.py       # TripleCompressor (open-vocab input)
├── diffusion_decoder.py # DiffusionDecoder (open-vocab output / expander)
├── diffusion_model.py  # DiffusionWorldModel (open-vocab wrapper)
└── config.py           # ModelConfig with profiles (base/mini/micro/atomic)
```
