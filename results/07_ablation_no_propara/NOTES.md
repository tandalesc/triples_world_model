# Run 07: Ablation — No ProPara

Tests whether ProPara training data helps or hurts. Trained without ProPara
to isolate its contribution.

- **Data**: 550 train (121 HW + 429 OpenPI), expanded 55/26 tests
- **Vocab**: 1756 tokens, GloVe 300d pretrained
- **Config**: d_model=256, 4 layers, 4 heads

## Results
- Comp gen: F1=0.717 (-2.4% vs with ProPara)
- Seen: F1=0.709 (-2.9%)
- OpenPI dev: F1=0.833 (same)

## Takeaway
ProPara hurts its own dev set but helps comp gen via training volume and
location-tracking patterns. Keep it in the training mix.
