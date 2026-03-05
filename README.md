# Triple World Model (TWM)

A minimal world model that learns temporal state dynamics over structured
(entity, attribute, value) triples using a vanilla transformer encoder.

**The core claim**: a small transformer (~3M params) over decomposed triple
tokens can learn compositional state transitions that generalize to novel
entity-state combinations never seen in training — and it needs cross-position
attention to do it.

## Results

| Model | Context-Dep (30) | Comp Gen (55) | Seen (26) |
|-------|:---:|:---:|:---:|
| Copy baseline | 0.29 | 0.29 | 0.29 |
| Qwen3-VL 8B (5-shot) | 0.59 | 0.56 | 0.57 |
| MLP + GloVe (no attention) | 0.76 | 0.70 | 0.64 |
| **TWM (ours)** | **0.99** | **0.74** | **0.77** |

The context-dependent test is the key result: when a triple's next state
depends on what other triples are present (glass stays full if nobody's thirsty,
fire goes out if wind is gusty), the MLP can't solve it because it processes
each position independently. The transformer attends across positions and
gets +23% F1.

## How It Works

```
State at time t (set of triples):
  (glass, state, full), (alice, state, thirsty), (bob, state, resting)

Tokenized (3 tokens per triple, sorted alphabetically):
  [alice] [state] [thirsty] [bob] [state] [resting] [glass] [state] [full]
  + positional encoding: (triple_index, role: entity/attr/value)

  → Transformer encoder (4 layers, 4 heads, 256-dim)
  → Linear head per position → predicted next-state tokens

Predicted next state:
  (glass, state, empty), (alice, state, satisfied), (bob, state, resting)
```

- GloVe 300d pretrained token embeddings, projected 300→256
- Set-to-set prediction (not autoregressive) — triples have no natural order
- Input residual: most of the world persists, model only learns the delta
- Padding mask for variable-length triple sets (up to 8 triples)

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Train the model (takes ~5 min on MPS/GPU, ~15 min CPU)
uv run python -m twm.train \
  --data-dir data/combined \
  --out-dir results/my_run \
  --pretrained-embeds data/combined/pretrained_embeds.pt \
  --epochs 500 --batch-size 32 --lr 1e-3

# Evaluate on all test splits
uv run python -m twm.metrics \
  --checkpoint results/my_run \
  --data-dir data/combined \
  --split all
```

### Inference tool (pretrain + predict)

If you want a single command that can **train missing weights** and then run inference:

```bash
uv run python scripts/inference_tool.py \
  --checkpoint results/inference_ready \
  --train-if-missing \
  --data-dir data/combined \
  --pretrained-embeds data/combined/pretrained_embeds.pt \
  --epochs 200 \
  --input '[["glass","state","full"],["alice","state","thirsty"]]'
```

For existing checkpoints (already trained), you can skip `--train-if-missing`.

If your input contains out-of-vocabulary tokens, enable canonicalization:

```bash
uv run python scripts/inference_tool.py \
  --checkpoint results/inference_ready \
  --canonicalize-oov --show-canonicalization \
  --input '[["cedric","state","curious"],["alice","state","thirsty"]]'
```

### Rebuilding from scratch

If you want to rebuild everything from raw data:

```bash
# 1. Build GloVe pretrained embeddings (downloads ~1GB model on first run)
uv run python scripts/build_pretrained_embeds.py \
  --vocab data/combined/vocab.json \
  --output data/combined/pretrained_embeds.pt

# 2. Train the transformer
uv run python -m twm.train \
  --data-dir data/combined \
  --out-dir results/my_run \
  --pretrained-embeds data/combined/pretrained_embeds.pt

# 3. Run MLP baseline comparison
uv run python scripts/run_mlp_baseline.py
```

### Data pipeline (if modifying training data)

```bash
# Convert ProPara and OpenPI to triple format (requires raw data)
uv run python scripts/convert_propara.py
uv run python scripts/convert_openpi.py

# Merge all sources into combined dataset
uv run python scripts/merge_datasets.py

# Rebuild vocab and embeddings
uv run python scripts/build_pretrained_embeds.py \
  --vocab data/combined/vocab.json \
  --output data/combined/pretrained_embeds.pt
```

### LLM benchmark (requires local inference server)

The LLM comparison requires a local vLLM/Ollama server and an embedding
model server. Update the URLs in `scripts/benchmark_llm.py` to match your setup.

```bash
uv run python scripts/benchmark_llm.py --split test_context --few-shot 5
```

## Project Structure

```
src/twm/
  model.py          Transformer world model (TripleWorldModel)
  mlp_baseline.py   MLP baseline (no cross-position attention)
  dataset.py        Triple transition dataset + collation
  vocab.py          Token vocabulary builder
  train.py          Training loop with eval
  metrics.py        Set-based F1, exact match, delta metrics

scripts/
  build_pretrained_embeds.py   GloVe embedding initialization
  run_mlp_baseline.py          Train + compare MLP vs transformer
  benchmark_llm.py             LLM benchmark with few-shot + semantic eval
  semantic_eval_all.py         Semantic similarity evaluation
  convert_propara.py           ProPara → triple format
  convert_openpi.py            OpenPI → triple format
  merge_datasets.py            Merge all data sources
  normalize_openpi_llm.py      LLM-based value normalization for OpenPI

data/
  train.jsonl                  Handwritten training examples (121)
  context_dependent_train.jsonl  Cross-entity interaction examples (83)
  test_comp.jsonl              Compositional generalization test (27)
  test_seen.jsonl              Seen-combination test (12)
  test_context.jsonl           Context-dependent attention test (30)
  combined/                    Merged dataset (1371 train, all test splits)
    train.jsonl, test_*.jsonl, vocab.json, pretrained_embeds.pt

results/
  01-08_*/                     Numbered experiment runs with NOTES.md each
  comparisons/                 Cross-model comparison charts and data
  README.md                    Full results summary and progression
```

## Training Data

Three sources, merged into `data/combined/`:

| Source | Examples | What it adds |
|--------|:---:|-------------|
| Handwritten | 121 | Kitchen physics, weather, social, mechanics |
| ProPara | 738 | Location tracking from procedural text |
| OpenPI | 429 | Diverse attributes: cleanness, temperature, moisture |
| Context-dependent | 83 | Cross-entity interactions requiring attention |
| **Total** | **1371** | |

Test sets are held-out and non-overlapping with training.

## Key Design Decisions

- **Decomposed triples, not sentence embeddings.** Each entity/attribute/value
  is its own token. Compositionality comes from structure, not embedding space.
- **Set-to-set, not autoregressive.** Parallel prediction of all output
  positions. Closer to BERT than GPT.
- **GloVe with real names.** Entity tokens are real words (`alice`, `bob`,
  `glass`, `campfire`) with distinct pretrained embeddings. Compound tokens
  like `person_a` get near-identical vectors and break entity disambiguation.
- **Input residual.** Most of the world persists between timesteps. The model
  only needs to learn what changes.
