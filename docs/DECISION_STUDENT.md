# Decision students: distilling Hearthvale's decision model into the browser

Hearthvale's non-player actors ask a decision model ("System 1", LiquidAI d1-3B
served on `rrh-llm-1`) what to do: a resident's routine, a household's move, a
faction's stance. Every question is a banded snapshot of the town plus a few
facts about who is asking, and every answer is a probability distribution over
a fixed set of options. This directory's code trains small students that copy
those distributions so the game can answer locally, in plain JS, in the sim
worker.

Code: `src/twm/decision/` (data, models), `scripts/train_decision_student.py`,
`scripts/train_decision_gbdt.py` (XGBoost baseline),
`scripts/relabel_decision_student.py` (second-stage distillation),
`scripts/export_decision_student.py` (INT8 browser bundle, reusing
`export_jepa_weights.py`'s quantization). Hearthvale side: `src/decision-features.js`
(question → triples), `scripts/distill-*.js` (collect, label, export, check),
`src/decision-student.js` (JS forward pass), `src/decision-local.js` (wiring).
Run date 21 October 2026.

## Data

- **Collection.** 64 headless towns grown to year 150 under 10 stand-in council
  styles and 7 water archetypes; a recording decision bus captured every
  question: about 4.75M distinct questions over 483k distinct snapshots in the
  first 40 towns alone (`resident_routine` 57%).
- **Labels.** 345,376 questions answered by d1-3B through the game's own
  request path (byte-identical prompts), sampled per kind (up to 15k each, and
  150k routine) with several questions per snapshot to use the service's state
  cache. 0 failures; 110–220 q/s.
- **Input.** Each question becomes 19–72 `(scope, attribute, value, num)`
  triples generated from the kind's schema (no per-kind feature code); free-text
  names are dropped.
- **Split.** By town: 20% of towns are held out entirely. 99.9% of held-out
  rows have a snapshot never seen in training.

## Models

| name | shape | in-browser cost |
|---|---|---|
| **set** | one token per triple (scope + attribute + value embeddings, plus `num · num_dir[attr]`), no positions; state triples → self-attention encoder (cached per snapshot); question triples + one token per offered option → decoder layers attending to the state; one logit per option | cached state: 1 decoder pass |
| **twm** | the TWM core's layout: three tokens per triple, `role_emb` + `triple_pos_emb` | ~3× the tokens |
| **fm** | factorization machine, multi-output: `logit(o) = <u_o, LN(½((Σe)² − Σe²)) + LN(Σe)>`; order 3 adds `(p1³ − 3p1p2 + 2p3)/6` over a projection | O(triples · d) |
| **mlp** | the same triple embeddings, summed, then an MLP (dropout) | O(triples · d + hidden²) |
| **xgb** | XGBoost per kind, one-hot triples, soft labels by row expansion | trees |

All neural students train on KL(teacher ‖ student), 20 epochs, MPS.

## Results (held-out towns, 30,667 rows)

Per-kind-average baseline KL: 0.150.

| model | KL | top-1 | shipped (weights + vocab, INT8) | JS µs / question (cached state) |
|---|---:|---:|---:|---:|
| set d32, 2+2 layers | **0.0253** | **85.7%** | 307 KB | ~260 |
| set d32, 1+1 | 0.0258 | 85.3% | 273 KB | ~135 |
| set d16, 1+1 | 0.0294 | 84.4% | 194 KB | ~60 |
| xgb depth 6 (all kinds) | 0.0283 | 85.4% | ~12.7 MB (1.3M nodes) | ~5,400 node visits |
| fm³ d32 + set labels | 0.0303 | 84.6% | ~240 KB | ~130 |
| fm d32 + set labels | 0.0314 | 84.0% | 239 KB | ~21 |
| fm³ d64 | 0.0321 | 84.3% | 358 KB | ~200 |
| fm d32 | 0.0360 | 82.6% | 239 KB | ~21 |
| mlp d64, h256, dropout 0.3 + set labels | 0.0540 | 81.4% | — | — |
| mlp d64, h256, dropout 0.3 (best of 4) | 0.0620 | 80.5% | 484 KB | ~93 |

JS timings: Node on an M5 Pro, one thread, INT8 weights; parity with PyTorch
under 7e-7 on every bundle. The d1-3B teacher costs ~5–9 ms of GPU time per
question at 110–220 q/s.

"+ set labels": trained on d1's labels plus 1.2M questions from the training
towns labelled by the set model (second-stage distillation); about −13% KL for
every cheap model.

## Compositional test: held-out pairs

Six pairs of triples, chosen for strong interactions, are removed from training
(rows with either half alone stay in) and scored separately (8,571 rows):

| model | held-out towns KL | held-out pairs KL | pairs ÷ towns |
|---|---:|---:|---:|
| set d32, 2+2 | 0.0238 | **0.0248** | **1.04×** |
| set d16, 1+1 | 0.0288 | 0.0332 | 1.15× |
| fm³ d32 | 0.0324 | 0.0366 | 1.13× |
| fm d32 | 0.0372 | 0.0427 | 1.15× |
| mlp | 0.0583 | 0.2194 | 3.8× |
| xgb (5 kinds) | — | 0.0223 | — |

The MLP fails outright on `choose_trade` × (no open farm jobs, poor wellbeing):
KL 0.69 against a baseline of 0.75; the set model scores 0.038. The FMs win the
purest pairwise cases (crafters × "helps craft": fm³ 0.020 vs set 0.074).

## Town outcomes

Four unseen towns grown to year 150, answered by d1, by students, and by the
game's fallback rules: both set students land in d1's regime (median population
1,399 and 1,729 vs d1's 1,098; fallback rules 7,409) with routine shares within a
few points of d1's (worship slightly over-picked). End to end a student-run
town takes 82–288 s against 2,336–2,540 s waiting on d1. Four towns is low power.

## The faithful TWM layout

`twm` (three tokens per triple with role and triple positions) trained ~4×
slower and sat at ~2× the set model's KL at every epoch it reached (stopped
after 4 of 20). Likely causes: binding three tokens through a position tag must
be learned (the set model gets it by summation), and list fields shift triple
positions. A run with the triple order shuffled per batch (positions as a pure
binding tag) is the fair follow-up.

## Reproduce

```bash
# Hearthvale (worktree with data/distill/)
node scripts/distill-collect.js data/distill/v1 --seeds 40 --years 150
node scripts/distill-label.js data/distill/v1 --per-kind 15000 --per-state 8 --concurrency 4
node scripts/distill-export.js data/distill/v1 --out train.jsonl
# TWM
uv run python scripts/train_decision_student.py train.jsonl --arch set --epochs 20 --batch-size 1024 --lr 3e-3 --out results/decision_student/set
uv run python scripts/train_decision_student.py train.jsonl --arch set --holdout-pairs configs/decision_student/holdout_pairs.json ...
uv run python scripts/export_decision_student.py results/decision_student/set set.json --data train.jsonl
# Hearthvale
node scripts/distill-check-student.js set.json
```
