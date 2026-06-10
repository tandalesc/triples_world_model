# Entity-World Trace Generator — Design Note

Synthetic training data for the **engram-wm** program's OOD-entity generalization
experiment (substrate objective `engram-wm`, task 2: *continuous modulus-identity memory*),
and eventually the pet-sim demo. Spiritual descendant of `scripts/generate_pet_sim.py`,
re-targeted from closed-vocab triples to the JEPA chain-text format.

- Generator: `scripts/generate_entity_world.py` (seeded, config-dict at top of file)
- Dedicated BPE builder: `scripts/build_entity_world_bpe.py`
- Output: `data/entity_world/`
- Tests: `tests/jepa/test_entity_world.py`

## 1. World spec

**Entity types.** Each type has a typed **attribute schema** (a subset of the shared
attribute pool) and a **response profile** (how it reacts to each action). The point of the
world is that types respond to the *same* action *differently* — that differential response
is what a continuous identity representation should capture and a lookup table can only
memorize.

**Attributes.** Eight ordinal ladders in the shared pool (`ATTRIBUTE_POOL`), index 0 = best:

| attribute | ladder (best → worst) |
|---|---|
| hunger | full, fed, hungry, starving |
| energy | lively, rested, tired, worn out |
| mood | cheerful, happy, calm, sad |
| cleanliness | spotless, clean, messy, dirty |
| thirst | watered, fresh, thirsty, parched |
| power | on, warming, cooling, off |
| fill | full, filling, draining, empty |
| open | wide open, open, ajar, shut |

**Actions.** One shared vocabulary (`feed, play, wash, rest, water, switch on, switch off,
fill, open, close, wait`). An action with no entry in a type's profile is a **no-op** for
that type — so the *same* action sequence produces type-specific trajectories.

**Oracle dynamics.** Deterministic ordinal shifts: each effect moves an attribute one rung
"up" (better) or "down" (worse), clamped at the ends. **Conditional** overrides (first match
wins) model state-dependent responses — e.g. `play` when the dog is *worn out* lowers mood
instead of raising it; overwatering the fern (`water` when *watered*) hurts mood. Dynamics
are deterministic (`STOCHASTIC=False`) so a trace is **exactly replayable** from its initial
state + action labels (the oracle-consistency contract the action-recovery eval relies on).

**Rendering.** States render as 2–4 short GLUCOSE-register sentences (`The dog feel(s)
hungry. The dog is messy.`), surfacing the two most salient attributes per entity. Each
chain step after the first is `<action sentence> <resulting state>`. 1–2 entities per chain.

## 2. Split design

Type roles are config-controlled and **disjoint** (`TYPE_LIBRARY[*]["split_role"]`):

| role | types | schema | profile |
|---|---|---|---|
| train | dog, cat, horse, fern, lamp, kettle, box | seen | seen |
| near_ood | puppy, pony, sprout | **same** as a train type | **small perturbation** of a train profile |
| far_ood | terrarium, robot pet | **novel recombination** of seen schemas | structurally novel (cross-attribute coupling) |

**Graded-similarity knob.** Documented per-type in `TYPE_LIBRARY` (`derived_from`,
`similarity`) and surfaced in the manifest:

- **near-OOD** = one knob turned on a base type. *puppy* = dog but `wash` lifts mood instead
  of lowering it; *pony* = horse but `play` no longer dirties and `rest` now lifts mood;
  *sprout* = fern with the overwatering penalty removed and `wait` also lowering mood. Same
  schema, ~one changed effect → an identity space that interpolates should place these near
  their base type.
- **far-OOD** = novel schema recombination + structurally novel coupling. *terrarium* =
  container (fill/open) + plant (thirst) where `water` also fills and `open` dries it out;
  *robot pet* = device (power/fill) + living (mood/energy) where charging lifts mood and
  playing drains the battery. No training type couples these attributes — pure memorization
  has nothing to address.

## 3. What each test split discriminates

| split | file | discriminates |
|---|---|---|
| train | `train.jsonl` | in-distribution dynamics (the substrate to learn from) |
| test_iid | `test_iid.jsonl` | generalization to fresh states/actions on **seen** types — the floor; failure here means the model didn't even learn the dynamics |
| test_ood_near | `test_ood_near.jsonl` | **identity-space interpolation** — small profile perturbation of a seen schema; a continuous modulus-identity memory should address these, a lookup table can't |
| test_ood_far | `test_ood_far.jsonl` | **novel schema recombination** + unseen cross-attribute coupling — the hardest test of compositional identity addressing |

The expected ordering for the continuous-identity claim is
`iid ≥ near > far ≫ lookup-baseline-on-OOD`. A lookup memory should collapse to chance on
`near`/`far` (no row for an unseen entity type); the continuous memory's whole value
proposition is non-trivial accuracy there.

## 4. Action-labeled variants

Every split has a `*_labeled.jsonl` twin: same `chain`, plus a parallel `actions` field
(`["<action>@<entity_index>", ...]`, length `len(chain)-1`) and the `types` per chain. These
ground-truth oracle labels enable **action-recovery evaluation** of the JEPA pipeline's
*unsupervised* latent actions: cluster/probe the inferred latent transition codes against
the true action label per step and measure recovery accuracy / mutual information. The
`@index` actor tag lets recovery align both verb and which entity moved.

## 5. How the OOD-entity experiment consumes it

The JEPA pipeline (`src/twm/jepa/data.py`) reads `{"chain": [...]}` as adjacent
`(state_t, state_{t+1})` cross-state pairs, tokenized by a 512-token domain BPE. Consumption:

1. **Train** the world model / memory on `train.jsonl` (train types only).
2. Build the baseline (the program's rule: every component must beat a black-box baseline on
   a named metric or be cut). The **lookup-table memory** baseline is the relevant ceiling
   here — it structurally cannot address `near`/`far` entity types.
3. **Evaluate** next-state prediction (and action-recovery, via the `*_labeled` twins) on all
   four splits. The named metric for task 2 is **OOD-entity generalization**: the gap between
   the continuous modulus-identity memory and the lookup baseline on `test_ood_near` /
   `test_ood_far`. Continuous identity wins iff it interpolates to unseen types.
4. The graded near/far axis turns a binary pass/fail into a **curve**: how far in identity
   space the memory can address before it breaks.

`manifest.json` ships the full schema + response profile of every type so the eval harness
can compute oracle next-states for any split without re-running the generator.

## 6. Tokenizer coverage

The existing GLUCOSE 512 BPE (`data/glucose/jepa_bpe_512.json`) **does not transfer well**:
mean ~40 tokens/state, ~13 byte-fallback fragments/state, and up to 4.4% of far-OOD states
overflow `max_text_tokens=64` (it never saw *kettle / lamp / terrarium / "worn out"*). The
generator detects this (`coverage_report.json`, verdict POOR) and the design ships a
**dedicated BPE**: `scripts/build_entity_world_bpe.py` →
`data/entity_world/bpe_512.json` (actual vocab 278; the domain is low-entropy). It cuts mean
tokens/state to ~25.6, P95 39, **0% overflow**. OOD surface forms are included at *build*
time for vocabulary coverage only (character statistics, no dynamics → no label leakage).

## 7. Reproducing / regenerating

```
uv run python scripts/generate_entity_world.py        # writes data/entity_world/*
uv run python scripts/build_entity_world_bpe.py        # writes bpe_512.json
uv run --with pytest python -m pytest tests/jepa/test_entity_world.py -q
```

All randomness derives from `CONFIG["seed"]` (per-split offset streams), so output is
byte-reproducible. Knobs (chain count, length range, entities/chain, stochasticity, type
roster) live in the `CONFIG` dict and `TYPE_LIBRARY` at the top of the generator.
