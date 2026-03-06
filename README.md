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

### Model Family Scaling

TWM scales down to embedded/edge hardware. Micro (16d, 1 layer, 2 heads) retains
the attention advantage at 57x fewer parameters:

![Final Comparison](results/family_benchmark/plots/final_comparison.png)

| Model | Params | Context F1 | Comp Gen F1 | Seen F1 |
|-------|-------:|:---:|:---:|:---:|
| Base (256d, GloVe) | 4.5M | 0.978 | 0.748 | 0.778 |
| Micro (16d, shared) | 80K | 0.911 | 0.671 | 0.640 |
| Micro QAT (int8-ready) | 80K | 0.893 | 0.632 | 0.706 |

![Size vs Accuracy](results/family_benchmark/plots/efficiency.png)

Key scaling findings:
- **Attention works at 16d/2-head** — context-dependent F1 drops only 7% (0.978 -> 0.911)
- **Split embedding tables help at base scale** (+0.011 context) but **hurt at micro** (-0.089) — not enough dimensions for separate tables to learn useful representations
- **QAT is essentially free** — simulated int8 quantization noise costs <2% F1
- **ESP32 target met**: ~4.4K params / ~5 KB at int8 with domain-specific vocab

See [results/README.md](results/README.md) for full experiment progression and analysis.

## How It Works

```
State at time t (set of triples):
  (glass, state, full), (alice, state, thirsty), (bob, state, resting)

Tokenized (3 tokens per triple, sorted alphabetically):
  [alice] [state] [thirsty] [bob] [state] [resting] [glass] [state] [full]
  + positional encoding: (triple_index, role: entity/attr/value)

  -> Transformer encoder (4 layers, 4 heads, 256-dim)
  -> Linear head per position -> predicted next-state tokens

Predicted next state:
  (glass, state, empty), (alice, state, satisfied), (bob, state, resting)
```

- GloVe 300d pretrained token embeddings, projected 300->256
- Set-to-set prediction (not autoregressive) — triples have no natural order
- Input residual: most of the world persists, model only learns the delta
- Padding mask for variable-length triple sets (up to 8 triples)

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Train the base model
uv run python -m twm.train \
  --data-dir data/combined \
  --out-dir results/my_run \
  --config base \
  --pretrained-embeds data/combined/pretrained_embeds.pt \
  --epochs 500

# Train the micro model (for embedded deployment)
uv run python -m twm.train \
  --data-dir data/combined \
  --out-dir results/my_micro_run \
  --config micro \
  --epochs 500

# Evaluate on all test splits
uv run python -m twm.metrics \
  --checkpoint results/my_run \
  --data-dir data/combined \
  --split all
```

### Config profiles

| Profile | d_model | Layers | Heads | d_ff | Target |
|---------|--------:|-------:|------:|-----:|--------|
| `base` | 256 | 4 | 4 | 1024 | GPU training/inference |
| `micro` | 16 | 1 | 2 | 32 | ESP32 / edge deployment |
| `atomic` | 256 | 4 | 4 | 1024 | ATOMIC 2020 (12 triples) |

Additional flags:
- `--split-embeddings` — separate entity/attr/value embedding tables
- `--quantize-aware` — simulate int8 quantization during training

### Inference tool (pretrain + predict)

```bash
uv run python scripts/inference_tool.py \
  --checkpoint results/inference_ready \
  --train-if-missing \
  --data-dir data/combined \
  --pretrained-embeds data/combined/pretrained_embeds.pt \
  --epochs 200 \
  --input '[["glass","state","full"],["alice","state","thirsty"]]'
```

For existing checkpoints, skip `--train-if-missing`. For OOV tokens:

```bash
uv run python scripts/inference_tool.py \
  --checkpoint results/inference_ready \
  --canonicalize-oov --show-canonicalization \
  --input '[["cedric","state","curious"],["alice","state","thirsty"]]'
```

### LLM bridge (prototype)

TWM as structured reasoning middleware for LLMs:

```python
from twm.llm_bridge import TWMBridge

bridge = TWMBridge(checkpoint_dir="results/08_context_dependent")

# Full pipeline: natural language -> triples -> predict -> explain
result = bridge.reason("Alice has a full glass and she's thirsty")

# Or skip LLM, use structured triples directly
result = bridge.reason_no_llm(
    [["glass", "state", "full"], ["alice", "state", "thirsty"]]
)
```

### Rebuilding from scratch

```bash
# 1. Build GloVe pretrained embeddings (downloads ~1GB model on first run)
uv run python scripts/build_pretrained_embeds.py \
  --vocab data/combined/vocab.json \
  --output data/combined/pretrained_embeds.pt

# 2. Train the transformer
uv run python -m twm.train \
  --data-dir data/combined \
  --out-dir results/my_run \
  --config base \
  --pretrained-embeds data/combined/pretrained_embeds.pt

# 3. Run MLP baseline comparison
uv run python scripts/run_mlp_baseline.py

# 4. Run model family benchmark
uv run python scripts/benchmark_family.py \
  --data-dir data/combined \
  --results-dir results/family_benchmark \
  --epochs 500
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

```bash
uv run python scripts/benchmark_llm.py --split test_context --few-shot 5
```

## Project Structure

```
src/twm/
  config.py          Model config profiles (base, micro, atomic)
  model.py           Transformer world model (TripleWorldModel)
  mlp_baseline.py    MLP baseline (no cross-position attention)
  dataset.py         Triple transition dataset + collation
  vocab.py           Token vocabulary builder (shared + role-split)
  train.py           Training loop with eval, QAT support
  metrics.py         Set-based F1, exact match, delta metrics
  serve.py           Inference wrapper (WorldModel)
  llm_bridge.py      LLM<->TWM bridge for structured reasoning

scripts/
  benchmark_family.py        Train + eval all model variants
  plot_family.py             Generate comparison charts
  convert_atomic.py          ATOMIC 2020 -> TWM triple format
  build_pretrained_embeds.py GloVe embedding initialization
  run_mlp_baseline.py        Train + compare MLP vs transformer
  benchmark_llm.py           LLM benchmark with few-shot + semantic eval
  semantic_eval_all.py       Semantic similarity evaluation
  inference_tool.py          Train-if-missing + inference CLI
  convert_propara.py         ProPara -> triple format
  convert_openpi.py          OpenPI -> triple format
  merge_datasets.py          Merge all data sources

data/
  combined/                  Merged dataset (1371 train, all test splits)
  train.jsonl                Handwritten training examples (121)
  test_comp.jsonl            Compositional generalization test
  test_context.jsonl         Context-dependent attention test (30)

results/
  01-08_*/                   Numbered experiment runs with NOTES.md
  family_benchmark/          Model family scaling experiments
  comparisons/               Cross-model comparison charts and data
  README.md                  Full results summary and progression
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
