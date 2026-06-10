# Triple World Model (TWM)

A small, structured world model that learns **state dynamics** — how a situation changes
into the next one — over decomposed (entity, attribute, value) triples, using a vanilla
transformer instead of a giant language model. The bet: a tiny model that represents state
as structure (not as a sentence embedding) can learn compositional, reversible
transformations that generalize to novel state combinations — at thousands of transitions
per second on a laptop.

## The current architecture in one paragraph

Each entity is a **noun + adjective** pair, and dynamics is a **verb** acting on it. Nouns
are points in a complex latent space; the adjective is the **modulus profile** — a
persistent identity that rotations structurally cannot touch. The verb is a **discrete
latent action** applied as per-block complex multiplication, inferred unsupervised from
state pairs (no action labels). Because the action is complex multiplication, **persistence
is structural** (rotation can't change what something *is*), **the inverse is exact
division** (undo is free, not learned), **composition is angle addition**, and **the future
enters only through the verb's few bits** — so the model is forced to compress "what
changed" into a small, discrete, invertible operator. This is the JEPA v2/v2.1 line; the
older closed-vocab transformer core (below) is the lineage it grew out of.

## 60-second quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Smoke-train the latent-action world model (3 epochs, ~2-4 min on CPU/MPS)
uv run python scripts/train_jepa_v2.py configs/jepa_nano_v2_smoke.json
```

Watch the `diag_v2[epN]` line each epoch: `ce_true_nats` falls 6.1 → 4.0, the codebook
starts getting used, and generated GLUCOSE next-state samples + diagnostics land in
`results/jepa_nano_v2_smoke/`. That's the whole loop — encode a state into noun slots,
infer the discrete verb that took it to the next state, apply the operator, decode the
result back to text. For the full 100-epoch run and every other result, see
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Key results

| Result | Headline | Where |
|---|---|---|
| **Compositional generalization (closed-vocab v1)** | 0.74 F1 on novel entity-state combos; Mini (178K) matches Base | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| **Pet-sim demo** | 29K params, 98.9% exact match, runs client-side in 303 KB JS | [demo/](demo/) |
| **JEPA v2 latent actions** | slot-LOO flips positive (all 8 slots constructive); `ce_gap` 0.166 | [research/jepa_matrix_synthesis.md](research/jepa_matrix_synthesis.md) |
| **JEPA v2.1 decoder arm** | bigger decoder cuts teacher-forced CE 1.39 → **0.95 nats**; fluent GLUCOSE | [research/jepa_matrix_synthesis.md](research/jepa_matrix_synthesis.md) |

**Honest status:** the v2 line has a working structural prior and a clean decoder, but
three capability gaps persist — next-state retrieval barely beats chance, the discrete
action carries no recoverable semantics yet, and generations are fluent but generically
wrong. The relative-representation interface (for mounting this inside an LLM) is *marginal*,
not yet green-lit. The numbers and the v3 recipe are in the synthesis below.

## Links

- **[docs/REPRODUCING.md](docs/REPRODUCING.md)** — exact commands, data prerequisites,
  wall-clock, and expected numbers for every result.
- **[research/jepa_matrix_synthesis.md](research/jepa_matrix_synthesis.md)** — latest
  synthesis: the experiment matrix, per-arm verdicts, capability gaps, v3 recipe.
- **[research/jepa_v2_latent_actions.md](research/jepa_v2_latent_actions.md)** +
  **[research/jepa_v21_polar.md](research/jepa_v21_polar.md)** — the v2 / v2.1 designs.
- **[docs/AGENTS.md](docs/AGENTS.md)** — agent operating manual (wartable jobs, probes,
  diagnostics conventions).
- **[CLAUDE.md](CLAUDE.md)** — the agent surface / project instructions.
- **[demo/](demo/)** — client-side pet-simulation inference demo.

## How the closed-vocab core works (the lineage)

The JEPA line sits on top of an earlier, simpler idea that still underpins the demo and the
compositional-generalization results:

```
text / triples → Compressor → [mode | bottleneck] → Dynamics → Expander → text / triples
                                                         ↺
                                                    (unroll N times)
```

The **dynamics core** is a transformer that processes triples in latent space, seeing
`(B, max_triples × 3, d_model)` tensors regardless of input format. A **mode triple**
(advance / query / identity) is prepended to condition the transformation, and the core
uses an **input residual** — it learns deltas, not full outputs. Decomposed triples (each
entity/attribute/value is its own token) are what give it compositionality. For the full
file map and diagrams see [research/architecture.md](research/architecture.md).

## Inference (closed-vocab)

```python
from twm import TextDynamicsModel
model = TextDynamicsModel.load("results/my_run/")
# model.compress() -> model.forward_dynamics() -> model.generate()
```

## Dynamics analysis

Tools for understanding how the dynamics core transforms state. Requires `uv sync --extra viz`.

```bash
uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim
uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim --eigenspectrum
uv run python scripts/visualize_dynamics.py --checkpoint results/pet_sim --flow-field
```
