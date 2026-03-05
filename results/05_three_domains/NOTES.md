# Run 05: Three Domains (HW + ProPara + OpenPI)

Added OpenPI (wikiHow state changes) as third training domain. Evaluated on
original (smaller) test sets.

- **Data**: 1288 train (121 HW + 738 ProPara + 429 OpenPI), original 27/12 tests
- **Vocab**: 2340 tokens, GloVe 300d pretrained
- **Config**: d_model=256, 4 layers, 4 heads

## Results
- Comp gen: F1=0.836 (best on original tests, +6% over two-domain)
- Seen: F1=0.877
- OpenPI dev: F1=0.833

## Takeaway
Adding OpenPI significantly boosts generalization. Three-domain training provides
diverse attributes (temperature, cleanness, moisture, ownership) that help the
model learn more general state transition patterns.
