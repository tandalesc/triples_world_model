# Run 06: Expanded Test Sets [CANONICAL]

Same model as run 05, but evaluated on expanded, harder test sets with
cross-domain coverage. This is the canonical run for all final comparisons.

- **Data**: 1288 train (121 HW + 738 ProPara + 429 OpenPI)
- **Test**: 55 comp gen (was 27), 26 seen (was 12) — cross-domain combos
- **Vocab**: 2340 tokens, GloVe 300d pretrained
- **Config**: d_model=256, 4 layers, 4 heads

## Results (Exact)
- Comp gen: F1=0.741, EM=53%
- Seen: F1=0.738, EM=54%
- OpenPI dev: F1=0.833, EM=75%

## Results (Semantic, via Embedding Gemma)
- Comp gen: sem_F1=0.818
- Seen: sem_F1=0.856
- OpenPI dev: sem_F1=0.917

## Takeaway
Harder tests lower absolute numbers but give a truer measure of generalization.
This run is the basis for the 4-way model comparison (see comparisons/).
