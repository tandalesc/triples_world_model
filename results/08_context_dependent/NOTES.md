# Run 08: Context-Dependent (Attention Stress Test) [CANONICAL]

Proves transformer attention is essential, not just helpful. Added 83 training
examples where the same triple transitions differently depending on other
triples in the state. Renamed person_a/b/c to alice/bob/carol for distinct
GloVe embeddings (cosine sim 0.28 vs 0.62 with person_a/person_b).

- **Data**: 1371 train (121 HW + 738 ProPara + 429 OpenPI + 83 context-dependent)
- **Test**: 55 comp gen, 26 seen, 30 context-dependent (NEW)
- **Vocab**: 2340 tokens, GloVe 300d, alice/bob/carol as entity names
- **Config**: d_model=256, 4 layers, 4 heads

## Context-Dependent Test Results (the attention test)
| Model | Exact F1 | Exact Match | Delta F1 |
|-------|:---:|:---:|:---:|
| **TWM (transformer)** | **0.989** | **0.967** | **0.983** |
| MLP (no attention) | 0.756 | 0.500 | 0.725 |
| Qwen3-VL 8B 5-shot | 0.589 | 0.200 | — |

## Full 4-Way Comparison (Exact F1)
| Model | Context (30) | Comp Gen (55) | Seen (26) |
|-------|:---:|:---:|:---:|
| Copy baseline | 0.29 | 0.29 | 0.29 |
| Qwen3-VL 8B 5-shot | 0.59 | 0.56 | 0.57 |
| MLP + GloVe | 0.76 | 0.70 | 0.64 |
| **TWM (ours)** | **0.99** | **0.74** | **0.77** |

## Takeaway
On standard tests, TWM leads MLP by 4-8%. On context-dependent tests requiring
cross-entity reasoning, the gap explodes to +23% F1 / +47% exact match. The MLP
fails specifically on "no-change" cases (flask stays full when nobody's thirsty,
door stays closed when nobody's pulling) because it can't check other positions.
TWM wins 15/30 examples the MLP gets wrong; MLP wins 0 the TWM gets wrong.
