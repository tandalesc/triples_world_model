# Run 01: Handwritten Only

First working model. Trained on 121 handwritten examples only.

- **Data**: 121 train, 27 comp gen test, 12 seen test
- **Vocab**: 149 tokens, random embeddings
- **Config**: d_model=256, 4 layers, 4 heads

## Results
- Train: 100% exact match (full memorization)
- Comp gen: F1=0.770, EM=63%
- Seen: F1=0.971, EM=83%

## Takeaway
Compositional generalization works on small handwritten data. The model learns
transitions, not just memorization — 77% F1 on novel entity-state combos.
