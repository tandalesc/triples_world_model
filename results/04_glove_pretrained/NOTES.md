# Run 04: GloVe Pretrained

Same setup as run 03 but with GloVe 300d embeddings projected to 256-dim.

- **Data**: 859 train (121 HW + 738 ProPara, with conflict dedup)
- **Vocab**: 798 tokens, GloVe 300d pretrained
- **Config**: d_model=256, 4 layers, 4 heads, embed_dim=300 -> 256 projection

## Results
- Comp gen: F1=0.775 (+2.3% vs random), token_acc=0.887 (+5.4%)
- Seen: F1=0.905

## Takeaway
GloVe helps, especially at token-level accuracy. Modest F1 gain but consistent
improvement across metrics. Worth keeping for all future runs.
