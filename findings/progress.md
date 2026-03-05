# Triple World Model — Progress Log

## What This Is

A minimal world model that learns temporal state dynamics over structured (entity, relation, value) triples using a vanilla transformer encoder. The core question: does compositional generalization emerge from this structure, or is the model just memorizing?

## Architecture (v2, current)

- **Set-to-set transformer encoder**: input triples → encoder → predicted next-state triples (single forward pass, parallel prediction)
- **Input residual**: skip connection from input token embeddings to encoder output before the prediction head. Most of the world persists between timesteps — the model only needs to learn the delta.
- **Positional encoding**: learned embeddings for (triple_index, role) where role ∈ {entity, relation, value}. No sequence position — triples are a set.
- **Canonical ordering**: both input and output triples sorted alphabetically. Aligns positions for the residual and removes ordering ambiguity.
- **Padding mask**: pad positions can attend to real positions (getting useful representations for output prediction at new triple slots), but real positions don't attend to pad.
- **3.2M parameters**: 4 layers, 4 heads, 256-dim, 1024 FFN. ~149 token vocabulary.

## Dataset (v2, current)

- **121 training examples** across 4 domains: kitchen physics, weather, social interactions, simple mechanics
- **28 test_comp examples** (compositional generalization): novel entity-state combinations and novel entities never seen in training (jug, fireplace, stone, etc.)
- **14 test_seen examples**: seen combinations with extra context triples
- **Variable-length output**: some examples have different input/output triple counts (2→3, 3→2, 4→3). Empty output slots predicted as `<pad>`.

## Results

### v1 → v2 improvements

Three changes were made simultaneously between v1 and v2:

1. **Set-based evaluation** (was positional comparison, penalized correct-but-reordered predictions)
2. **Input residual connection** (persistence is free — model only learns what changes)
3. **More diverse training data** (30 additional examples with new entities: mug, bottle, flask, grill, campfire, rock, brick, pan, person_c, road, garden, button)

| Metric | v1 (500ep) | v2 (500ep) | Delta |
|---|---|---|---|
| Comp gen F1 | 0.50 | **0.77** | +0.27 |
| Seen test F1 | 0.54 | **0.97** | +0.43 |
| Seen exact match | 0.08 | **0.83** | +0.75 |
| Comp exact match | 0.30 | **0.63** | +0.33 |
| Copy baseline F1 | 0.44 | 0.44 | — |

### v1 error analysis (informed v2 changes)

Per-role accuracy breakdown revealed:
- **Relations**: 92% (easy — "state" stays "state")
- **Entities**: 67% (hard — model memorized specific entities instead of learning role-agnostic patterns)
- **Values**: 60% (hardest — the actual state change prediction)

Key failure modes:
- **Entity substitution**: `block→ball`, `kettle→cup` (memorized co-occurrences instead of learning the pattern)
- **Context corruption**: 3rd "bystander" triples that should pass through unchanged got corrupted
- **Ordering confusion**: correct triples placed at wrong output positions (set-based eval fixed this)

### v2 training dynamics

- Model converges by **epoch ~20** (train F1 = 1.0, loss < 0.01)
- **No overfitting**: generalization gap is flat from epoch 20–500. The gap to comp gen (0.23) is a generalization ceiling, not overfitting drift.
- Loss has transient spikes (epochs 55-75, 170-210) from cosine LR schedule noise on small dataset. No impact on final metrics.
- Context preservation nearly solved: seen test F1 0.97 (was 0.54 in v1)

### What the comp gen gap (0.23) consists of

The remaining errors on test_comp are:
- **Truly novel entities** (jug, fireplace, stone): the model has never seen these tokens and can't generalize from embedding space (tokens are learned from scratch, no pretrained semantics)
- **Entity-swap generalization** (block for ball, kettle for pot): improved but not perfect — needs more training diversity
- **Variable-length edge cases**: predicting output triple counts different from input

## Next Steps

- More training data diversity (the clearest lever — every new entity for an existing pattern teaches the model that the pattern is entity-agnostic)
- Investigate whether the novel-entity failure (jug, fireplace, stone) is fundamentally a vocabulary problem (unseen tokens have random embeddings) vs. a generalization problem
- Multi-step prediction: `advance_n()` already exists but needs evaluation for drift
- Attention pattern visualization to verify causal structure learning
