# Run 02: Added ProPara

First multi-domain run. Added ProPara procedural text data.

- **Data**: 859 train (121 HW + 738 ProPara), same 27/12 test sets
- **Vocab**: 802 tokens, random embeddings
- **Config**: d_model=256, 4 layers, 4 heads

## Results
- Comp gen: F1=0.758 (slight drop from 0.770)
- Seen: F1=0.943
- ProPara dev: F1=0.244 (terrible — location-only triples, process-specific)

## Takeaway
ProPara adds training volume but its location-only triples don't directly help
comp gen. ProPara dev is unfair (requires process-specific reasoning the model
can't learn from triple structure alone).
