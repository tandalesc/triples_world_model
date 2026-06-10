# JEPA v2.1 — Polar Decomposition: Modulus = Identity, Phase = State

Status: design (user-approved architecture; this doc formalizes and resolves details).
Builds on: `research/jepa_v2_latent_actions.md` (v2.0). Supersedes the v1 predictive-verb
path (`research/jepa_operator_v1_design.md` §3 VerbHead-as-predictor), which is
probe-confirmed dead (`results/jepa_nano_probe/`, `results/jepa_nano_viz/REPORT.md`).

The in-flight GPU run is pinned to commit `6d1b094` and is **not** affected by anything
here — v2.1 is purely additive on top of the live v2 path (operator, slot-encoder trunk,
SIGReg, `anneal_tau`, data pipeline, EMA readout).

---

## 0. One-paragraph thesis

A noun `k ∈ ℝ^dn` is already read by the operator as `dn/2` 2×2 blocks — i.e. `dn/2`
**complex coordinates** `z_b = x_b + i·y_b`. v2.1 commits to the polar reading of those
coordinates and assigns *semantics to the two factors*: the **modulus profile** `|z_b|`
is the object's persistent identity ("what it is"); the **phase profile** `arg(z_b)` is its
mutable state ("how it is right now"). The verb operator `B_v` (the existing
`RotationScaleOperator`) is a diagonal complex map `z_b ↦ r_b · e^{iθ_b} · z_b`. A **pure
rotation** (`r_b ≡ 1`) is exactly the modulus-preserving subgroup, so it *structurally*
preserves identity; `r_b ≠ 1` is reserved for irreversible change. v2.1 adds a tiny,
zero-init, state-dependent angle offset (the "adjective replacement") so a verb's effect
can depend on *what the object is* — while keeping composition angle-additive and the
inverse exact.

v2.1 == v2.0 at initialization (the one new map `H` is zero-init). It is a refinement of
the operator's *conditioning*, not a new model. The behavior-preservation gate (§9) is a
hard requirement precisely because of this.

---

## 1. The polar reading is already in the code

`RotationScaleOperator` (`src/twm/jepa/operator.py`) stores, per verb `v`, per block `b`:

```
theta[v,b]    # block angle  θ
log_r[v,b]    # block log-scale, r = exp(log_r) > 0 (positivity structural)
```

and applies, per block, the RoPE-style elementwise map

```
(x', y') = ( r·(x cosθ − y sinθ),  r·(x sinθ + y cosθ) )    # = r·e^{iθ}·(x + iy)
```

Reading `z_b = x_b + i·y_b`:

```
|z'_b| = r_b · |z_b|              (modulus scales by r_b only)
arg(z'_b) = arg(z_b) + θ_b        (phase rotates by θ_b only)
```

This is the entire v2.1 claim made explicit. **The factors are already decoupled in the
math.** v2.1 adds (a) a name and a loss-side check for that decoupling, (b) a
state-dependent phase offset, and (c) diagnostics that read the two profiles separately.

Definitions used throughout (`nb = dn/2` blocks):

| Quantity | Formula | Shape | Meaning |
|---|---|---|---|
| modulus profile `m(k)` | `m_b = sqrt(x_b² + y_b²)` | `(B,M,nb)` | persistent identity |
| phase profile `φ(k)` | `φ_b = atan2(y_b, x_b)` | `(B,M,nb)` | mutable state |
| verb scale | `r_b = exp(log_r[v,b])` | `(nb,)` | per-verb modulus gain |
| verb angle | `θ_b = theta[v,b]` | `(nb,)` | per-verb phase advance |

---

## 2. Polar split — identity / state semantics

- **Modulus profile `|z|` = identity.** Two slots are "the same kind of object" iff their
  modulus profiles match (up to the SIGReg-imposed Rayleigh scale). Identity is what a
  verb must *not* change unless the change is irreversible.
- **Phase profile `arg(z)` = state.** This is what `advance`/`query`-type verbs move.
- **`B_v` is a diagonal complex transform.** Pure rotation (`r_b ≡ 1`) is the
  `U(1)^nb` subgroup; it preserves `|z|` exactly (a numerical identity, asserted in §8).
- **`r_b ≠ 1` is reserved for irreversible change.** Contraction/expansion of a modulus is
  the only way identity can be destroyed/created. Per-verb `log_r` is *global* (not
  state-conditioned) in v2.1 — see §3 — so irreversibility is a property of the *verb*, not
  the verb×object pair. This is deliberate: "the kettle boiled dry" should be irreversible
  regardless of which kettle.

This requires **no change to `RotationScaleOperator.apply` / `inverse_apply`**. The polar
reading is interpretation, not new arithmetic. The new arithmetic is the conditioning map.

---

## 3. Conditioning map `H` — the "adjective" replacement

v2.0's per-pair action `v` is a single global verb shared across all `M` slots
(`model_v2.py::_apply_action` broadcasts `v_onehot (B,V) → (B,M,V)`). That means *every
object in the scene undergoes the same θ*. v2.1 makes the per-slot phase advance depend on
the slot's **own identity** (its modulus profile), so "open" rotates a *door* slot
differently from a *jar* slot — without re-introducing per-slot verb prediction (the dead
v1 cheat).

### 3.1 Definition

For slot `i` with noun `k_i`, modulus profile `m_i = |k_i| ∈ ℝ^nb`:

```
θ_i = θ_v + H(m_i)
```

where:
- `θ_v ∈ ℝ^nb` is the verb's global block-angle codebook row (existing `theta[v]`).
- `H : ℝ^nb → ℝ^nb` is a **single linear map** `nn.Linear(nb, nb, bias=False)`,
  **zero-initialized** (`weight.zero_()`). At init `H(m_i) = 0` ⟹ `θ_i = θ_v` ⟹ v2.1 == v2.0.
- `m_i` is detached for the conditioning input? **No — keep gradient.** The whole point is
  to let the encoder shape the modulus profile so it is a useful conditioning signal.
  Detaching would sever that learning channel. (We *do* `stop_grad` modulus only inside the
  identity-persistence *diagnostic* in §8, never in the forward.)

`H` is **shared across verbs** in v2.1 (a single `(nb, nb)` matrix). Rationale: the
adjective-style modulation ("how much this object's kind bends a given verb") is a property
of the *space*, not of individual verbs; per-verb `H` would be `V·nb²` params and invites
the same per-verb overfitting the v1 path died of. One shared `H` is `nb² = 256` params at
`dn=32` (§6).

### 3.2 What is and isn't conditioned

| Factor | Conditioned on identity in v2.1? | Why |
|---|---|---|
| phase advance `θ` | **Yes** — `θ_i = θ_v + H(m_i)` | the adjective slot: verb effect depends on object kind |
| log-scale `log_r` | **No** — stays `log_r[v]`, global per verb | irreversibility is a verb property (§2); conditioning it would make "boiled dry" reversible for some kettles. Reserved for a future v2.2 if needed. |

### 3.3 Composition stays angle-additive; inverse stays exact

Because `H` only shifts the *angle* and `log_r` is untouched, the per-slot operator is still
a diagonal complex map `z_b ↦ r_b · e^{iθ_{i,b}} · z_b`. Therefore:

- **Composition** of two v2.1 steps on the *same* slot is `r_u r_v · e^{i(θ_{i,u}+θ_{i,v})}`
  — angle-additive, exactly as in v2.0. (Caveat: `H` reads the *current* modulus, and a
  pure-rotation verb leaves modulus unchanged, so the offset is stable under composition of
  rotation-only verbs; under an `r≠1` verb the modulus changes and the next step's offset
  shifts accordingly — this is the *correct* state-dependent behavior, not a bug.)

- **Inverse is exact** and computed by subtracting *both* terms. The undo of a v2.1 step is
  `z_b ↦ (1/r_b) · e^{-iθ_{i,b}} · z_b`. Since modulus is preserved under pure rotation, for
  a pure-rotation verb `H(m_i)` evaluated on the post-step noun equals `H(m_i)` on the
  pre-step noun (modulus unchanged), so the angle to subtract is recoverable. **Implementation
  note:** the operator's `inverse_apply` must be passed the *same* `θ_i` used in the forward.
  We therefore make the conditioned angle an explicit argument rather than recomputing it
  inside `inverse_apply` from a possibly-mutated modulus. See §4.2.

---

## 4. Operator API changes

### 4.1 New: conditioned-apply path

`RotationScaleOperator` gains an **optional** angle-offset channel. The existing
`apply(k, v)` / `inverse_apply(a, v)` signatures are **unchanged** (v2.0 callers and the
in-flight run are untouched). v2.1 adds:

```python
def apply(self, k, v, theta_offset=None):       # theta_offset: (..., nb) or None
def inverse_apply(self, a, v, theta_offset=None):
```

When `theta_offset is None` the methods are bitwise-identical to today (default arg). When
provided, the per-position block coefficients become:

```
θ_eff = θ_v + theta_offset            # broadcast (V-gathered θ_v) + (B,M,nb)
a = r · cos(θ_eff)
b = r · sin(θ_eff)
```

`r` is still the verb's global `exp(log_r[v])` (unconditioned). `_gather_blocks` is
refactored to return `(θ_v_per_pos, r_per_pos)` and the cos/sin is taken *after* adding the
offset, so the soft-mix (Gumbel) path still works: for a hard one-hot `v` the gathered
`θ_v` is exact; the offset is added in angle space before the trig. (For a genuinely soft
`v` the block-linear "expected operator" identity of `operator.py` §1.6 no longer holds once
a nonlinear `H` and post-add trig enter — but v2.0/v2.1 train with **hard** ST one-hots
(`hard=True`), so this is a non-issue in practice. We document it and assert `hard=True` in
the v2.1 model path.)

### 4.2 The model owns `H`, not the operator

`H` is a learned `nn.Linear`. It belongs to the **model** (`JEPAOperatorModelV21`), not the
operator codebook, because:
- the operator is the frozen-interface "algebra" object (bake/export, JS demo);
- `H` is a conditioning network that the operator should stay ignorant of.

Forward (`model_v2.1`):

```python
m_i = k.detach().pow(2).reshape(B, M, nb, 2).sum(-1).sqrt()   # NO — keep grad:
m_i = k.pow(2).reshape(B, M, nb, 2).sum(-1).sqrt()            # (B, M, nb)  |z| per block
theta_offset = self.H(m_i)                                    # (B, M, nb), zero at init
a = self.operator.apply(k, v_slots, theta_offset=theta_offset)
```

The undo path (`undo_latent`, rollout) recomputes `theta_offset` from the *pre-step* modulus
and passes it explicitly to `inverse_apply` — never recomputed from the post-step noun.
This is why §3.3 requires the offset to be an explicit argument.

### 4.3 `RotationOperator` (pure-rotation ablation) under v2.1

`RotationOperator` freezes `log_r ≡ 0`. Under v2.1 it becomes the **exact
identity-preserving** operator: `|z'| = |z|` for every block, every slot, every verb,
*including* the conditioned-angle path (the offset only touches phase). The
identity-persistence assertion in §8 must pass with error `< 1e-5` for this operator on
every verb. This is the cleanest test of the whole polar claim.

---

## 5. SIGReg per-factor — keep the base, add two diagnostics

### 5.1 Why full SIGReg replacement is unnecessary

The v2.1 target distribution per block is an **isotropic complex Gaussian**
`z_b ~ 𝒞𝒩(0, σ²)`. An isotropic complex Gaussian in `nb` complex dims **is** an isotropic
real Gaussian in `2·nb = dn` real dims — they are the *same distribution* on `ℝ^dn`. The
existing `sigreg_loss` (`losses.py`) already does a sliced Epps–Pulley GoF test for an
isotropic real Gaussian on the standardized nouns. Therefore:

> **The existing real-vector SIGReg already enforces the v2.1 per-factor target.** No new
> loss term is needed; replacing it with a bespoke phase+modulus loss would be strictly more
> code testing strictly less (it would test the marginals, not the joint isotropy SIGReg
> already covers). We keep `sigreg_loss` **unchanged** and weight it the same (`w_sigreg=0.05`).

The factor structure (phase uniform, modulus Rayleigh) is a *consequence* of isotropic
complex Gaussianity, made precise:

- `z_b ~ 𝒞𝒩(0,σ²)` ⟹ `arg(z_b) ~ Uniform(−π, π)`, independent of modulus.
- `z_b ~ 𝒞𝒩(0,σ²)` ⟹ `|z_b| ~ Rayleigh(σ/√2)`.

So if SIGReg passes, phases *should* be uniform and moduli *should* be Rayleigh. We don't
need to *train* on those marginals — we **measure** them as cheap diagnostics to confirm the
joint test is doing its job factor-wise (and to surface collapse modes SIGReg's slicing
might average over).

### 5.2 Two added factor diagnostics (NOT losses)

Both live in `diagnostics_v2.py` (or its v2.1 successor `diagnostics.py`, §10) and run only at
eval cadence. Neither contributes gradient.

1. **Phase-uniformity score** (circular statistics). For each block `b`, collect phases
   `φ_b` over the eval set and compute the **mean resultant length**
   `R_b = |(1/N) Σ_n e^{iφ_{n,b}}| ∈ [0,1]`. `R_b → 0` for uniform phases; `R_b → 1` for a
   concentrated (collapsed) phase. Report `phase_uniformity = 1 − mean_b R_b` (1.0 = ideal
   uniform). Also report the Rayleigh test z-statistic `z_b = N·R_b²` per block; flag any
   block with `z_b` above the 99% χ²₂ critical value as "phase-collapsed".

2. **Modulus-profile effective rank.** Build the `nb × nb` covariance of the modulus profiles
   `m(k)` over the eval set and reuse the existing `_effective_rank` helper
   (`diagnostics_v2.py` imports `_effective_rank` from v1 diagnostics). Report
   `modulus_eff_rank`. A healthy identity space uses many modulus dimensions; collapse to a
   handful means identity has degenerated into a 1-D "size" knob. Threshold heuristic:
   warn if `modulus_eff_rank < nb/4` (mirrors the existing `noun_eff_rank` threshold
   `dn/4`).

Both are pure-numpy reductions over already-collected `k` (the diagnostics harness already
gathers `all_k`), so the cost is negligible.

---

## 6. Param-delta table

`dn = 32 ⟹ nb = dn/2 = 16`.

| Component | v2.0 params | v2.1 params | Δ |
|---|---:|---:|---:|
| SlotEncoder (trunk, heads) | unchanged | unchanged | 0 |
| `RotationScaleOperator` (`theta`,`log_r`) | `2·V·nb = 256` | `2·V·nb = 256` | 0 |
| TransitionEncoder / PriorHead | unchanged | unchanged | 0 |
| TokenDecoder | unchanged | unchanged | 0 |
| **`H` conditioning map** `Linear(nb→nb, bias=False)` | — | `nb² = 256` | **+256** |
| (optional) kind head (§7), default OFF | — | `0` (off) | 0 |
| **Total non-embedding** | (v2.0 nano ≈ as built) | +256 | **+256** |

256 params on a ~250K budget is **0.1%** — budget unaffected. With `bias=True` it would be
`+272`; we keep `bias=False` (a zero modulus profile should give zero offset, and bias would
break the v2.1==v2.0-at-init guarantee unless also zero-init — simpler to omit).

Optional kind head (§7), if enabled: a codebook `(K, nb)` of `K` modulus prototypes, e.g.
`K=16 ⟹ 16·16 = 256` params, **VQ/argmax, not a Linear classifier head** (no per-class
weight matrix). Off by default; never on the budget-critical path.

---

## 7. Optional discrete "kind" head (diagnostic/demo only)

Config flag `model.use_kind_head` (default **false**). When on, a `VectorQuantizer`-style
or plain argmax readout over the **modulus profile** `m(k)` assigns each slot a discrete
"kind" id `∈ {0..K−1}`:

```
kind_codebook: (K, nb)                # K modulus prototypes
kind_id(k_i)  = argmin_j || m(k_i) − kind_codebook[j] ||²
```

Strict constraints:
- **Never load-bearing for routing.** The kind id does not gate `H`, does not select a verb,
  does not touch the operator. It is read *off* the modulus profile purely to label slots in
  diagnostics / the demo ("this slot is a `jar`-kind object").
- Trained, if at all, by a small VQ commitment loss **only when the flag is on** (mirrors the
  repo VQ gotchas: codebook `normal(0,1)` init, fp32 under autocast-off, low `λ_vq=0.25`).
  Default config leaves it untrained and unbuilt.
- Output surfaces as a **kind-cluster × example table** in diagnostics (§8): for each kind
  id, the top-N example slots/texts assigned to it.

This exists to make the "modulus = identity" claim legible to a human, not to make the model
work. It is a microscope, not a gear.

---

## 8. Diagnostics additions (`diagnostics_v2.py` → v2.1)

Added to the existing `eval_diagnostics_v2` output dict:

| Metric / artifact | Definition | Pass condition |
|---|---|---|
| `phase_uniformity` | `1 − mean_b R_b` (§5.2.1) | report; warn if `< 0.8` |
| `phase_collapsed_blocks` | count of blocks failing the Rayleigh z-test | report; warn if `> 0` |
| `modulus_eff_rank` | eff-rank of modulus-profile cov (§5.2.2) | warn if `< nb/4` |
| `identity_persistence_err` | see below | **assert `< 1e-5`** for pure-rotation verbs |
| `kind_cluster_table` | kind id → top-N example texts (only if `use_kind_head`) | artifact only |

### 8.1 Identity-persistence check (exact assertion)

For each verb `v`, take a probe batch of nouns `k`, compute the conditioned offset
`θ_off = H(|k|)`, apply the operator with that offset, and measure modulus drift:

```
a       = operator.apply(k, v, theta_offset=H(|k|))
drift_v = || |a| − |k| ||  / (|| |k| || + eps)        # relative modulus drift
```

- For a **pure-rotation** verb (`r_v ≡ 1`, i.e. the `RotationOperator` ablation, or any verb
  whose `log_r` row is exactly 0): `drift_v` must be `< 1e-5`. This is a *structural*
  identity (rotation preserves complex modulus); the offset `H(|k|)` only moves phase, so it
  cannot perturb modulus. **The diagnostic asserts this.** A failure means a code bug
  (e.g. the offset leaked into `r`, or the polar split is mis-wired), not a training issue.
- For a **scaling** verb (`r_v ≠ 1`): `drift_v ≈ |r_v − 1|` profile-weighted; reported as
  `modulus_drift[v]`, *not* asserted (this is the intended irreversible-change channel).

This is the load-bearing test of the entire v2.1 thesis: rotation = identity-preserving must
hold to machine precision.

### 8.2 `chain_ids` exposure on `JEPAChainDataset` (REQUIRED — integrator gap)

`diagnostics_v2.py`'s hard-negative MRR (`_compute_retrieval_mrr`, lines ~181–231) builds
*same-chain* negative pools via:

```python
if hasattr(dataset, "chain_ids") and idx < len(dataset.chain_ids):
    cid = dataset.chain_ids[idx]
elif hasattr(dataset, "pairs") ...:
    cid = getattr(dataset.pairs[idx], "chain_id", idx)
else:
    cid = idx                      # ← FALLBACK: every pair is its own chain
```

`JEPAChainDataset` (`data.py`) currently exposes **neither** `chain_ids` nor `pairs`, so
every pair falls into the `cid = idx` branch ⟹ `same_chain[i]` is always empty ⟹ the hard
pool degenerates into the easy pool ⟹ `easy_minus_hard_mrr` is structurally `0` and the
v1-regression guard (`easy_minus_hard_mrr ≥ 0`) passes *vacuously*. The integrator flagged
this; v2.1 fixes it.

**Fix (data.py):** record the originating chain index while flattening. In `__init__`,
alongside `src_texts`/`tgt_texts`, append `chain_idx` per pair:

```python
self._chain_ids: list[int] = []
...
for chain_no, line in enumerate(f):
    chain = json.loads(line)["chain"]
    for i in range(len(chain) - 1):
        src_texts.append(chain[i]); tgt_texts.append(chain[i+1])
        self._chain_ids.append(chain_no)          # both pairs of a chain share an id
```

Expose it as a public attribute `chain_ids` (a list, len == `len(dataset)`), and slice it in
the `max_chains` cap path of `train_jepa_v2.py` (lines 217–222) so the truncated dataset's
`chain_ids` stays aligned (`dataset._chain_ids = dataset._chain_ids[:cap]`; add a `chain_ids`
property returning `self._chain_ids`). With chain length 3, each chain contributes exactly 2
adjacent pairs that now share a `chain_id`, so `same_chain[i]` finds its sibling pair and the
hard pool is non-trivial. **Test:** add a `data.py` unit test asserting
`len(set(ds.chain_ids)) == n_chains` and that adjacent pairs from one chain share an id.

---

## 9. Decoder phase-sensitivity check (unit test)

The whole architecture assumes the **decoder can read phase differences** in `a* = B_v k`.
If the `TokenDecoder` cross-attention were phase-blind (e.g. it only attended to magnitudes),
the operator's rotation would be invisible and the latent action would carry no information.

**Unit test** (`tests/jepa/test_decoder_v2.py` or a new `test_polar.py`):

```
Construct two memories that differ ONLY in phase:
    a1 = a*                         # arbitrary (B, M, dn)
    a2 = rotate_each_block(a1, δ)   # apply a fixed nonzero angle δ per block,
                                    # modulus identical: |a2| == |a1| (assert < 1e-5)
Feed both through the (randomly-initialized) TokenDecoder with the SAME teacher-forced
target prefix:
    logits1 = decoder(a1, tgt_ids, tgt_pad)
    logits2 = decoder(a2, tgt_ids, tgt_pad)
Assert: (logits1 − logits2).abs().max() > 1e-3      # decoder is NOT phase-blind
```

Rationale: a model that produces identical logits for phase-rotated-but-equal-modulus
memories cannot use the state channel at all. The test is on a *random-init* decoder (we are
testing the *architecture's* sensitivity, not a trained behavior), so it is fast and
deterministic under a fixed seed. (The existing `ARDecoder.memory_proj` is a full `Linear`
over the `dn` real coordinates, so it provably is phase-sensitive — the test guards against
future "simplifications" that might collapse the memory to magnitudes.)

---

## 10. Config schema additions

New keys (all optional, defaulting to v2.0-equivalent behavior):

```jsonc
"model": {
  ...                                  // unchanged v2.0 keys
  "use_polar_conditioning": true,      // master switch for the H offset path; default false
                                       //   (false ⟹ exactly v2.0). Setting true with
                                       //   zero-init H is still numerically v2.0 at step 0.
  "use_kind_head": false,              // §7 diagnostic kind readout; default false
  "kind_codebook_size": 16             // K, only read when use_kind_head=true
}
```

Dataclass additions (`twm/jepa/__init__.py` `ModelHParams`): `use_polar_conditioning: bool =
False`, `use_kind_head: bool = False`, `kind_codebook_size: int = 16`. These are
forward-compatible (default-False), so **every existing v2.0 config still parses and still
builds an identical model** — critical for the §11 behavior-preservation gate. No new loss
weights (SIGReg unchanged, §5). The optional kind-VQ weight, if ever enabled, reuses the
existing `LossConfig` pattern and is gated entirely behind `use_kind_head`.

`jepa_nano_v2.json` stays a pure v2.0 config (no new keys). A new
`configs/jepa/jepa_nano_v21.json` (post-refactor location, §R) sets
`use_polar_conditioning: true`.

---

# PART II — Refactor Plan

Goal: a clean `src/twm/jepa` where the live v2 path is the *primary* path, dead v1
predictive-verb code is removed, version suffixes follow ONE convention, configs are
namespaced, and tests mirror the source. **Public entry points that MUST keep working
through the entire refactor: `scripts/train_jepa_v2.py` + `configs/jepa_nano_v2.json` (the
in-flight GPU run's interface) and `pytest tests/jepa`.** The in-flight run is pinned to
`6d1b094`, so it cannot be broken by working-tree edits — but the *interface* (script path,
config path, config schema) must remain valid so a resubmit at HEAD still works.

## R.1 Naming convention — pick ONE

**Decision: drop the `_v2` suffix; the v2 path is THE path.** v1 is either deleted or
demoted to an explicit `legacy/` location (see R.3). The `losses_v2 → losses` style of
"keep version suffixes" loses to "no suffixes on the live path" because:
- v2.1 is a *refinement* of v2, not a third parallel codebase. Perpetuating `_v2` invites a
  `_v21`, `_v3`, … sprawl.
- The integrator and diagnostics already duck-type on `forward_v2`; we keep that *method*
  alias for back-compat but rename the *files*.

The one exception: **`scripts/train_jepa_v2.py` and `configs/jepa_nano_v2.json` keep their
names** (frozen public interface of the in-flight run). We add `train_jepa.py`'s replacement
as needed but do NOT rename the v2 entry script. So the convention is: *module files lose the
suffix; the two frozen public entry points keep theirs.*

## R.2 Target layout

```
src/twm/jepa/
  __init__.py            # contracts, JEPA_PROFILES, config dataclasses (+ v2.1 keys §10)
  config.py              # from_dict parsing (unchanged shape)
  slot_encoder.py        # LIVE — unchanged (trunk + heads)
  operator.py            # LIVE — + optional theta_offset arg (§4.1)
  conditioning.py        # NEW — H map + (optional) kind head (§3, §7)
  transition.py          # LIVE — posterior + prior
  decoder.py             # LIVE — TokenDecoder
  model.py               # ← renamed from model_v2.py (the v2/v2.1 composed model)
  losses.py              # ← MERGED: v2 losses + the v1 utilities they import
  data.py                # LIVE — + chain_ids (§8.2)
  diagnostics.py         # ← renamed from diagnostics_v2.py (+ §5.2/§8 additions)
  legacy/                # frozen v1 baseline, IF kept (R.3)
    model_v1.py          # ← old model.py (Readout, Predictor, _EncoderReadout, JEPAOperatorModel)
    losses_v1.py         # ← old losses.py L_div / JEPALoss / usage_entropy / spread_penalty
    diagnostics_v1.py    # ← old diagnostics.py
scripts/
  train_jepa_v2.py       # FROZEN NAME (in-flight interface) — imports updated to new module paths
  train_jepa.py          # DELETE or move to legacy (R.3)
configs/jepa/            # NEW namespace dir
  jepa_nano_v2.json      # MOVED here? NO — see R.5 (keep at configs/ root for the frozen path)
  jepa_nano_v2_smoke.json
  jepa_nano_v21.json     # NEW (§10)
  jepa_mini_v2.json
configs/archive/         # dead v1 configs land here
tests/jepa/
  test_slot_encoder.py  test_operator.py  test_transition.py
  test_decoder.py        # ← from test_decoder_v2.py
  test_model.py          # ← from test_model_v2.py (v1 model tests → legacy or deleted, R.3)
  test_losses.py         # ← merged losses tests
  test_data.py  test_diagnostics.py  test_integration.py
  test_polar.py          # NEW — §8.1 identity-persistence, §9 phase-sensitivity
```

### Subtlety: the `model_v2 → model` rename and the `Readout/Predictor/_EncoderReadout` import

`model_v2.py` imports `Readout, Predictor, _EncoderReadout` from the v1 `model.py`. After the
rename, the v2 model becomes `model.py`, and those three helpers (which are **live** — the
EMA L_pred branch uses them) must move *with* it. So: lift `Readout`, `Predictor`,
`_EncoderReadout` out of old `model.py` into the new `model.py` (they are tiny). The v1-only
`JEPAOperatorModel` and its Gumbel `_verb_transform` go to `legacy/model_v1.py` (or are
deleted, R.3). Update `model_v2.py`'s `from .model import Readout, ...` to the new in-file
definitions.

## R.3 Dead-code list (exact)

**v1 predictive-verb path — DELETE (or demote to `legacy/`, see decision below):**

| Symbol / file | Where | Why dead |
|---|---|---|
| `JEPAOperatorModel._verb_transform` | `model.py` | per-slot Gumbel verb *prediction* from state_t — the probe-confirmed cheat |
| `JEPAOperatorModel.forward` (verb-predict branch) | `model.py` | predicts next verb from state_t alone (ill-posed) |
| `usage_entropy`, `spread_penalty` | `losses.py` | only feed `L_div` |
| `L_div` term in `JEPALoss.forward` | `losses.py` | "gameable regularizer", revoked by v2 §5 |
| `JEPALoss` (the v1 aggregator) | `losses.py` | superseded by `JEPALossV2` |
| `_OperatorRef` / theta-log_r read for L_div | `losses.py` | only L_div needs it |
| `w_div`, `w_scale_reg` weights | `LossConfig` | keep the *fields* (configs parse) but mark deprecated; v2 loss ignores them |
| `VerbHead` *predictive use* | `slot_encoder.py` forward returns `verb_logits` | **keep `VerbHead` class** (cheap, returned-but-ignored), but its output is dead in v2; document. NOT deleted to avoid touching the slot encoder trunk the in-flight run uses. |
| `scripts/train_jepa.py` | scripts | v1 train loop (Gumbel verbs + L_div) |

**MUST KEEP (the v2 path imports these — do NOT delete):**
`RotationScaleOperator`, `RotationOperator` (operator.py); `SlotEncoder` + trunk +
`NounHead` + `encode_text` (slot_encoder.py); `sigreg_loss`, `anneal_tau`,
`gumbel_softmax_sample` (losses.py — imported by transition.py and losses_v2.py);
`JEPAChainDataset` (data.py); `Readout`, `Predictor`, `_EncoderReadout` (the EMA L_pred
branch); `_effective_rank` (diagnostics, reused by §5.2.2).

**Decision on v1 baseline (frozen-baseline-vs-delete):**
> Demote, don't delete — but only the *files*, behind `legacy/`. Moving `model.py`→
> `legacy/model_v1.py`, `losses.py`→`legacy/losses_v1.py` (after lifting the shared utils
> out), and `scripts/train_jepa.py`→`scripts/legacy/train_jepa.py` is **cheap** (file moves
> + import-path fixups) and preserves the ability to reproduce the v1 negative result that
> motivated v2 (`results/jepa_nano_probe`). The v1 *configs* (`jepa_nano.json`,
> `jepa_mini.json`, `jepa_nano_seed1.json`, `jepa_nano_tau06.json`, `jepa_nano_v16.json`)
> move to `configs/archive/`. Rationale: the post-mortem value of the v1 baseline is high and
> the carry cost is one `legacy/` dir that nothing on the live path imports. If a future
> sweep proves it unused for 2+ iterations, delete then.

The genuinely *dead* (not legacy-worthy) pieces — `L_div`, `usage_entropy`, `spread_penalty`,
`_verb_transform` — are **deleted outright**, not moved, because they are probe-disproven and
keeping them invites re-use of a known-broken idea.

## R.4 The shared-utility extraction (losses merge)

`losses_v2.py` imports `sigreg_loss, anneal_tau` from `losses.py`; `transition.py` imports
`gumbel_softmax_sample` from `losses.py`. Merge plan for the new `losses.py`:

1. New `losses.py` = `{sigreg_loss, _standardize, _epps_pulley_gof, anneal_tau,
   gumbel_softmax_sample}` (the live utils) **+** `{token_ce, prior_kl, JEPALossV2}` (from
   `losses_v2.py`). Rename `JEPALossV2 → JEPALoss` (the suffix-drop convention) but **keep a
   `JEPALossV2 = JEPALoss` alias** so `train_jepa_v2.py`'s `from twm.jepa import JEPALossV2`
   keeps working (frozen interface).
2. The dead `{usage_entropy, spread_penalty, L_div, old JEPALoss}` → deleted (or, if v1 is
   kept runnable, the old `JEPALoss` + helpers go to `legacy/losses_v1.py` and the live
   `losses.py` does NOT import them).
3. `transition.py` import becomes `from .losses import gumbel_softmax_sample` (already correct
   after merge).

`__init__.py` re-exports: keep `JEPALossV2` in `__all__` (alias) AND add `JEPALoss` pointing
at the same class. Add `H`/conditioning exports (`PolarConditioner`, optional `KindHead`).

## R.5 Configs

- **Frozen public configs stay put:** `configs/jepa_nano_v2.json` and
  `configs/jepa_nano_v2_smoke.json` remain at `configs/` root (the in-flight resubmit path).
  Optionally *also* symlink/copy them under `configs/jepa/` for discoverability, but the
  canonical paths the script docs reference are unchanged.
- **New v2.1 config:** `configs/jepa/jepa_nano_v21.json` (sets `use_polar_conditioning:true`).
- **Live v2 configs** (`jepa_mini_v2.json`) → `configs/jepa/`.
- **Dead v1 configs** → `configs/archive/` (per R.3).
- The CLAUDE.md "Active configs" list is updated to point train_jepa_v2 at
  `configs/jepa/*` for v2.1 while noting the two frozen root-level v2 configs.

## R.6 Tests reorganized to mirror

- `test_model_v2.py → test_model.py`; `test_decoder_v2.py → test_decoder.py`;
  `test_losses_v2.py` merged into `test_losses.py` (keep both v1-util and v2-loss tests; v1
  L_div tests deleted with the code or moved to a `tests/jepa/legacy/` if v1 kept).
- v1 model tests (`test_model.py` as it exists for `JEPAOperatorModel`) → `legacy/` or
  deleted alongside the v1 model.
- **NEW `test_polar.py`:**
  - identity-persistence: `RotationOperator` modulus drift `< 1e-5` for all verbs, *with*
    a nonzero `theta_offset` (§8.1);
  - `theta_offset=None` path is bitwise-identical to pre-v2.1 `apply` (regression);
  - `H` zero-init ⟹ `apply(k,v,H(|k|)) == apply(k,v)` at step 0 (the v2.1==v2.0 guarantee);
  - decoder phase-sensitivity (§9);
  - `inverse_apply` with explicit `theta_offset` round-trips to `< 1e-5` (§3.3).
- **NEW `test_data.py` case:** `chain_ids` correctness (§8.2).
- Keep `pytest tests/jepa -x -q` green throughout; the 172 existing tests must stay passing
  (any renamed test keeps its assertions).

## R.7 Behavior-preservation gate (HARD requirement)

Before and after the refactor, the fixed-seed 1-epoch smoke must reproduce the **epoch-1
total loss `6.113 ± 0.001`**:

```
uv run python scripts/train_jepa_v2.py configs/jepa_nano_v2_smoke.json
# epoch-1 reported total loss must be 6.113 ± 0.001
```

This is well-defined because `seed_everything(cfg.seed=0)` is called before model
construction and the global RNG drives both `randperm` and `sigreg_loss`'s slicing directions
(`train_jepa_v2.py:193,262`; `losses.py:179`), making a fixed-device 1-epoch run
deterministic. The gate is run **twice**:
1. Pre-refactor (establish the baseline reproduces 6.113 on the dev box / GPU box used).
2. Post-refactor, with `jepa_nano_v2_smoke.json` **unchanged** and `use_polar_conditioning`
   absent (⟹ default false ⟹ no `H` in the graph) — the merged/renamed modules must yield
   the *identical* 6.113.

A *third*, optional check: `jepa_nano_v21.json` with `use_polar_conditioning:true` and
zero-init `H` should *also* hit 6.113 at epoch 1 (the v2.1==v2.0-at-init guarantee), drifting
only as `H` learns in later epochs. If it does not, `H` is not zero-init or the offset is
leaking into a non-phase channel — a bug, caught immediately.

Note: the gate must run on the SAME device class the baseline was measured on (cuda vs mps vs
cpu differ in floating-point accumulation; 6.113 is a per-device constant). Record which
device produced the canonical 6.113 in the run log.

---

## Summary of decisions resolved in this doc

1. Polar split is *interpretation* of the existing 2×2 blocks — no new operator arithmetic;
   `apply`/`inverse_apply` gain only an optional `theta_offset` arg (default None ⟹
   bitwise-identical to v2.0).
2. `H` is one shared `Linear(nb→nb, bias=False)`, zero-init, owned by the model, gradient
   kept (not detached). Conditions phase only; `log_r` stays global per verb (irreversibility
   is a verb property).
3. SIGReg is **not** replaced — isotropic complex Gaussian ≡ isotropic real Gaussian in 2×
   dims, which the existing `sigreg_loss` already enforces. Two cheap *diagnostics*
   (phase-uniformity via mean resultant length, modulus eff-rank) confirm the factors.
4. Identity-persistence is asserted exactly (`< 1e-5`) for pure-rotation verbs — the
   load-bearing test of the polar claim.
5. `chain_ids` added to `JEPAChainDataset` (the integrator's flagged gap) so hard-negative
   MRR forms true same-chain pools instead of degenerating to the easy pool.
6. Kind head is optional, default-off, diagnostic-only, never routes.
7. Param delta: **+256** (one `nb²` matrix) — budget unaffected.
8. Refactor convention: drop `_v2` suffix on module files; v2 is the path; v1 demoted to
   `legacy/` (cheap, preserves the negative-result baseline) with dead L_div/verb-predict
   bits deleted outright. The two frozen public entry points
   (`scripts/train_jepa_v2.py`, `configs/jepa_nano_v2*.json`) keep their names/paths.
9. Behavior-preservation gate: `jepa_nano_v2_smoke.json` reproduces epoch-1 total loss
   `6.113 ± 0.001` before and after refactor (and at v2.1 init).
