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

### How the Expander Works

The expander reconstructs BPE tokens from a 256d conditioning vector via diffusion:

1. **Training**: corrupt clean BPE embeddings with Gaussian noise at random timestep t.
   The denoiser predicts the clean embeddings (x0-prediction), conditioned on the
   dynamics output via cross-attention and adaLN-Zero.

2. **Inference**: start from pure noise, iteratively denoise over T steps.
   At each step, predict clean embeddings, re-noise to t-1, repeat.
   Final embeddings are decoded via nearest-neighbor lookup in the frozen BPE table.

3. **Length head**: a small linear layer predicts how many real tokens each slot
   contains. Output is truncated accordingly.

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
