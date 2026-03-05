# Run 03: GloVe Control (Random Init)

Control run for the GloVe embedding experiment. Same data as run_pretrained
but with random embedding initialization.

- **Data**: 859 train (121 HW + 738 ProPara, with conflict dedup)
- **Vocab**: 798 tokens, random embeddings
- **Config**: d_model=256, 4 layers, 4 heads

## Results
- Comp gen: F1=0.752, token_acc=0.833
- Seen: F1=0.933

## Takeaway
Baseline for A/B comparison with GloVe pretrained embeddings.
