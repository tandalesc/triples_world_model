# Triple World Model (TWM)

## What This Is

A minimal world model that learns temporal state dynamics over structured triples using a vanilla transformer. The core hypothesis: if you decompose world states into (entity, relation, value) triples and treat each component as a separate token, a small transformer can learn state transition dynamics that generalize compositionally to novel entity-state combinations it never saw in training.

This is a research prototype. The primary question is whether compositional generalization emerges — i.e., is this a world model or just compression.

## Architecture

```
State at time t (set of triples):
  (glass, state, full), (person, state, thirsty)

Tokenized (3 tokens per triple):
  [glass] [state] [full] [person] [state] [thirsty]
  + positional encoding marking triple boundaries
    (triple_index, role: entity/relation/value)

  → Small transformer encoder (4 layers, 4 heads, 128-dim)
  → Linear output head (128 → vocab_size) per position
  → Predicted next-state tokens

Target:
  [glass] [state] [empty] [person] [state] [satisfied]

Loss: cross-entropy per position
```

- Learned token embeddings: `nn.Embedding(vocab_size, 128)`
- Learned positional embeddings encoding (which_triple, which_role)
- Attention mask handles variable number of triples, padded to max_triples (16)
- ~3M parameters total, trainable on a single GPU in minutes

## Key Design Decisions

- **Decomposed triples, not sentence embeddings.** Each entity/relation/value is its own token. "full" and "empty" are different tokens, not similar sentences. Compositionality comes from structure, not embedding space geometry.
- **One advance function.** There is one physics, one set of dynamics. Multiple representational spaces were considered and rejected — the transformer's attention heads can learn whatever factoring is useful.
- **Set-to-set, not autoregressive.** Input triples have no natural ordering. This is parallel prediction of next-state, not sequential generation. Closer to BERT than GPT.
- **No pretrained embeddings.** The vocabulary is small enough (likely 40-80 tokens for initial experiments) that learned embeddings from scratch are fine and avoid the semantic-similarity-vs-state-discrimination problem of sentence transformers.

## Project Phases

### Phase 1: Handwritten Dataset (CURRENT)
Build 50-100 state transition pairs by hand across 3-4 domains:
- Kitchen physics (glass full → glass empty, food raw → food cooked)
- Weather/environment (sky clear → sky cloudy → sky raining)
- Social interactions (person lonely + person_b nearby → person social)
- Simple mechanics (ball high + ball unsupported → ball low)

Each example: 2-4 input triples → 2-4 output triples.

**Critical:** Hold out ~20 examples that recombine known entities with known states in combinations never seen in training. This is the compositional generalization test.

Data format (JSONL):
```json
{"state_t": [["glass", "state", "full"], ["person", "state", "thirsty"]], "state_t+1": [["glass", "state", "empty"], ["person", "state", "satisfied"]]}
```

### Phase 2: Model Implementation
PyTorch model with:
- Vocabulary builder: scan dataset, assign IDs to all unique tokens
- Positional encoding scheme: learned embeddings for (triple_index, role)
- TransformerEncoder: 4 layers, 4 heads, 128-dim, with padding mask
- Output head: Linear(128, vocab_size) predicting token logits per position
- Training loop: AdamW, lr=1e-4, cosine schedule, cross-entropy loss

### Phase 3: Evaluation
Three-tier eval:
1. **Memorization** — accuracy on training set (should be ~100% after overfitting)
2. **Compositional generalization** — accuracy on held-out novel combinations of seen entities/states. THIS IS THE ONLY TEST THAT MATTERS.
3. **Copy baseline** — predict state_t+1 = state_t. Must beat this to demonstrate the model learns transitions, not just persistence.

Report per-position accuracy and full-triple accuracy (all 3 components correct).
Visualize attention patterns to see if the model learns causal structure (which input triples drive which output changes).

### Phase 4 (Future): Scale Up
- LLM-generated training data (triple extraction from temporal corpora)
- Larger vocabularies, more domains
- Variable output length (predict different number of triples than input via stop token)
- Multi-step prediction (advance multiple times, check for drift)
- Comparison against serialized-text baseline (same data, GPT-2 fine-tuned on text versions of the transitions)

## File Structure
```
twm/
├── CLAUDE.md          # this file
├── data/
│   ├── train.jsonl    # handwritten training pairs
│   ├── test_comp.jsonl # held-out compositional generalization test
│   └── test_seen.jsonl # held-out seen-combination test
├── src/
│   ├── vocab.py       # vocabulary builder from dataset
│   ├── dataset.py     # PyTorch dataset + collation with padding
│   ├── model.py       # TripleWorldModel
│   ├── train.py       # training loop
│   └── eval.py        # evaluation + attention visualization
└── results/
    └── ...
```

## Hardware
Dual RTX 3090 homelab, but this model is so small a single GPU is overkill. CPU training might even be fine for Phase 1.

## Context
This emerged from exploring JEPA-style world models, MoE expert specialization, and temporal knowledge graph forecasting. The insight was reductive: strip away every architectural novelty until you're left with the minimal testable claim. That claim is: a vanilla transformer over decomposed triple tokens can learn compositional state dynamics. Everything else (scale, data pipelines, multi-step prediction, real-world applications) depends on whether that claim holds.

Related but distinct from existing work:
- Temporal KG forecasting (RE-NET, TiRGN, TANGO): uses GNNs over discrete entity IDs with fixed vocabularies
- KG embeddings (TransE, RotatE): static scoring, not temporal dynamics
- JEPA: operates in continuous latent space, not structured tokens
- Standard LLMs: learn dynamics implicitly but without compositional token structure

The gap this tests: vanilla transformer + decomposed triple tokens + temporal prediction. Simple enough to implement in an afternoon, novel enough that the result is interesting either way.
