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
- **Input residual.** Most state persists across transformations. The model learns the delta.

## Config Profiles

| Profile | d_model | Layers | Heads | d_ff | Max Triples |
|---------|--------:|-------:|------:|-----:|------------:|
| base | 256 | 4 | 4 | 1024 | 8 |
| mini | 32 | 2 | 2 | 128 | 8 |
| micro | 16 | 1 | 2 | 32 | 8 |
| atomic | 256 | 4 | 4 | 1024 | 12 |

## Project Structure

See [`research/architecture.md`](research/architecture.md#project-structure) for the full file map.

## Key Results

- **Compositional generalization**: 0.74 F1 on novel entity-state combos (3-domain benchmark)
- **Context-dependent reasoning**: 0.978 F1 where output depends on other triples present
- **Mini matches Base**: 178K params (25x smaller) gets identical context-dep F1
- **Micro is viable**: 80K params (57x smaller), 0.91 context-dep F1
- **Pet sim demo**: 29K params, 98.9% exact match, runs client-side in 303 KB JS
- **Open-vocab**: 81.1% exact match on ATOMIC 10K with compressor/expander (identity mode)
- **Beats frontier LLMs**: 100% attr accuracy vs 4-8/8 for Claude/Gemini/GPT on ATOMIC

## Data Format

```json
{"state_t": [["glass", "state", "full"], ["person", "state", "thirsty"]], "state_t+1": [["glass", "state", "empty"], ["person", "state", "satisfied"]]}
```

Test splits: `test_comp` (novel combos), `test_seen` (seen combos), `test_context` (cross-entity).

## Hardware

Dual RTX 3090. Models are small enough that CPU training works for closed-vocab experiments.
