# JEPA v5 — Discriminative-First Objective (Step-1a)

**Status: IMPLEMENTED (pending GPU validation).** Supersedes the earlier
`jepa_v5_usepressure_design.md` draft (slot-use pressure). The Step-0 baseline retired
that direction: slot occupancy is ~7/8 — slots ARE used, so collapse-onto-few-slots is
**not** the failure. The lever is a **discriminative** signal, not slot scaling.

## 0. What Step-0 told us, and what Step-1a does

Step-0 (v4 at `n_slots=8`, no new terms) measured the root failure on the geometry probes:

- `disc_delta_r2 ≈ 0` — the action's causal delta is **not linearly present** in the noun
  latent. The readout pool of the state cannot predict "which (attr, direction) changed".
- `separation_auc ≈ 0.50` — gold-vs-sibling next-states are **indistinguishable** in the
  pooled latent the AUC probes.
- `slot_occupancy ≈ 7/8` — slots are used. **Not** a capacity problem.

So Step-1a adds **two new, independently config-gated, default-0.0 discriminative terms**.
It **holds `n_slots = 8`** (no slot sweep — that's a later step). Every existing v3/v4
config parses and trains **bitwise-identically** (the established gating pattern: each new
weight defaults to its neutral value, the loss skips the term when the weight is 0, and the
trainer attaches/threads nothing extra at the default).

Both terms are **supervised** by the entity-world oracle, attached only on the
labeled path (a `*_labeled.jsonl` twin must exist). A GLUCOSE/unlabeled config never reads
labels and pays nothing.

---

## 1. Term 1 — `L_verb_anchor`: sparse oracle-verb supervision of the bottleneck

### Idea

A tiny **training-only** aux head predicts the oracle verb from the posterior's pooled
transition features. On a deterministic ~5% slice of in-batch transitions that carry an
oracle verb label, add the CE. This is the **LAM-augmentation / "1% labels flip
PCA-on-noise → PCA-on-action"** lever: a handful of action labels biases the bottleneck to
encode the *action factor* rather than the highest-variance surface direction. The head is
dropped at inference (never used in `rollout`).

### Loss math

```
aux_logits = verb_anchor_head(verb_features)          # (B, N_oracle_verbs=11)
valid      = verb_id >= 0                              # rows with an oracle label
sel        = a deterministic ~verb_anchor_frac slice of `valid` rows (≥1 if any)
L_verb_anchor = CE(aux_logits[sel], verb_id[sel])      # 0.0 if no labeled row
```

- `verb_features` = the posterior's **pre-logits pair features** `h (B, mlp_hidden=128)`:
  `h = LayerNorm(GELU(fc1([pool_t, pool_t1, pool_t1−pool_t])))` — the vector right before
  the V-way verb logits in `TransitionEncoder`. (The `[pool_t, pool_t1, delta]` pair vector
  is an equally-valid hook; we chose the pre-logits `h` so the head sees the posterior's own
  learned transition feature, not raw pools.)
- **Aux-head shape:** `nn.Linear(mlp_hidden, N_oracle_verbs)` = `nn.Linear(128, 11)`. Lives on
  the loss module (so the optimizer picks it up via `loss.parameters()`); built ONLY when
  `w_verb_anchor > 0`.
- **Sparse by construction:** only ~`verb_anchor_frac` (5%) of each batch's labeled rows are
  supervised, chosen by a seeded shuffle (reproducible). At least 1 row is kept when any label
  exists. No labeled row ⟹ 0.0 (bitwise-neutral on unlabeled data).
- **Cardinality:** the head predicts the **11** oracle verbs (`generate_entity_world.ACTIONS`).
  The model codebook has **V=8**. We do NOT force codebook == oracle — this is a bias on the
  bottleneck, not a relabeling of the latent codes.

### Hook points

| where | file:function | what |
|---|---|---|
| feature source | `transition.py:TransitionEncoder.forward` | stashes `self._last_pair_features = h`; new prop `pair_features_dim → fc2.in_features` |
| surfaced | `model.py:JEPAOperatorModelV2.forward` / `forward_unroll` | `out["verb_features"] = transition._last_pair_features` |
| aux head + CE | `losses.py:verb_anchor_ce` + `JEPALossV2` (head built in `__init__` when `w_verb_anchor>0`, CE in `forward`) | `nn.Linear(verb_anchor_in_dim, n_oracle_verbs)` |
| labels | `data.py:JEPAChainDataset` (`entity_labels.py`) | `verb_id` per transition from `actions` `<verb>@<idx>` → `ORACLE_VERBS` index; -1 if absent |
| threaded | `train_jepa_v2.py:_pair_step` / `_unroll_step` | passes `verb_features`, `verb_ids` only when `w_verb_anchor>0` and labels exist |
| head dim | `train_jepa_v2.py:train` | `verb_anchor_in_dim = model.transition.pair_features_dim` |

### Config fields + defaults

| field | default | when read |
|---|---|---|
| `w_verb_anchor` | `0.0` | `> 0 AND aux head built AND verb_features+verb_ids supplied` |
| `verb_anchor_frac` | `0.05` | only when `w_verb_anchor > 0` |

(`n_oracle_verbs` is fixed at 11 in the loss; the head input dim is read from the model.)

---

## 2. Term 2 — `L_sep`: sibling-contrastive next-state separation (SupCon)

### Idea

A **supervised-contrastive (SupCon)** loss on the **predicted next-latent** the
separation/disc probes read. Labels = the oracle **canonical next-state id** (a stable hash
of the oracle next-state string, computed from `initial_states + actions` via the oracle).
Positives = transitions with the **same** canonical next-state (this captures
**paraphrase-invariance** — `entity_world_para` renders one underlying state many ways, all
collapsing to one id). Negatives = different next-states; in-batch negatives that share the
anchor's start/chain (**siblings** — "what else could change from here") are weighted as
**HARD** negatives. This packs distinct next-state sub-manifolds into compact, separated
clusters at the granularity `disc_delta_r2` / `separation_auc` measure.

### Readout choice (load-bearing)

`L_sep` operates on **`sep_z = out["zhat"]`** — the pooled predicted-next-state vector that
is **exactly** `diagnostics._separation_auc`'s query (`zhat_vecs` at `diagnostics.py:1257`).
`zhat = Predictor(Readout(a*))`. We deliberately match the diagnostic's query so the term
trains the geometry the AUC scores. (The disc-delta probe uses `readout(k)` of the *start*
state; pressuring `zhat` lifts the predicted-next-state manifold, which is the AUC axis. If a
later step wants to also pressure `readout(a)` directly, that is a one-line swap of `sep_z`.)

### Loss math (SupCon, cosine / temperature)

```
zn   = normalize(sep_z)                               # (B, dn)
s    = (zn @ znᵀ) / τ                                 # (B, B)
P(i) = { j≠i : canon_id_j == canon_id_i, both labeled }   # positives
ω_ia = sep_hard_neg_weight  if a is a same-chain sibling of i (and not a positive) else 1
L_i  = −1/|P(i)| Σ_{p∈P(i)} log( exp(s_ip) / Σ_{a≠i} ω_ia·exp(s_ia) )
L    = mean over anchors with ≥1 positive            # 0.0 if none
```

- No stop-grad: gradient flows into the predicted-next-state readout (the AUC geometry) — the
  point of the term.
- Numerically stable (row-max shift, detached); the self/diagonal is excluded from the
  denominator; `(−inf)·0` poisoning is avoided with a `where` (not a multiply) over positives.
- Edge cases all return 0.0: no in-batch positive, all rows unlabeled, B==1.

### Hook points

| where | file:function | what |
|---|---|---|
| SupCon | `losses.py:sep_supcon` + `JEPALossV2.forward` (`if w_sep>0 and sep_z and canon_ids`) | the loss above |
| readout | `train_jepa_v2.py:_pair_step` / `_unroll_step` | `sep_z = out["zhat"]` (the AUC query) |
| labels | `data.py` (`entity_labels.replay_canonical_states` + `CanonicalStateRegistry`) | `canon_id` per transition: oracle replay → canonical string → stable int; -1 if unavailable |
| chain hard-neg | existing `chain_ids` plumbing | same-chain siblings up-weighted in the denominator |

### Config fields + defaults

| field | default | when read |
|---|---|---|
| `w_sep` | `0.0` | `> 0 AND sep_z+canon_ids supplied` |
| `sep_temperature` | `0.1` | only when `w_sep > 0` |
| `sep_hard_neg_weight` | `2.0` | only when `w_sep > 0` |

---

## 3. Data pipeline

`entity_labels.py` (new) owns the oracle-label computation; `data.py:JEPAChainDataset`
attaches the labels at load:

- New flag `attach_labels` (default **False**). When True AND a `<stem>_labeled.jsonl` twin
  exists beside the chain file, the dataset reads the twin and resolves `self.has_labels`.
  Absent twin ⟹ `has_labels=False`, no label tensors, bitwise-unchanged.
- Per transition the batch now carries:
  - `verb_id` (int) — `ORACLE_VERBS` index of `actions[i]`'s verb; -1 if absent.
  - `canon_id` (int) — stable id of the oracle canonical next-state (replay from
    `initial_states + actions`, canonicalize to a sorted multi-entity string, intern in a
    per-dataset `CanonicalStateRegistry`). Same canonical state ⟹ same id everywhere.
  - **pairs mode:** `_verb_id`, `_canon_id` (N,). **triples mode:** per-hop
    `_verb_id_h1/_h2`, `_canon_id_h1/_h2` (one row per chain; hop-1 = s0→s1, hop-2 = s1→s2).
- Precomputed once at load; `max_chains` cap keeps the label tensors aligned (train script).
- The trainer sets `attach_labels = (w_verb_anchor>0 or w_sep>0)`, so only v5 runs read labels.

---

## 4. Revised v5 configs (Step-1a)

`n_slots=8` HELD. Old `w_slot_balance` / `w_slot_nce` / `tau_slot_nce` weights **removed**.

| config | operator | `use_targeted_actions` | `w_sep` | `w_verb_anchor` |
|---|---|---|---|---|
| `jepa_v5_s0.json` (structured, rotation_scale) | rotation_scale | true | 0.5 | 0.25 |
| `jepa_v5_blackbox_s0.json` (gated_mlp) | gated_mlp | false | 0.5 | 0.25 |
| `jepa_v5_smoke.json` (tiny, local) | rotation_scale | true | 0.5 | 0.25 |

Common: `verb_anchor_frac=0.05`, `sep_temperature=0.1`, `sep_hard_neg_weight=2.0`. Both terms
are operator-independent (they read `verb_features` from the posterior and `zhat` from the
readout), so they are active in **both** the structured and blackbox paths (unlike the old
targeted-gate term, which was blackbox-dead). The s0/blackbox configs point at
`data/entity_world/{train,*_labeled}.jsonl`; the smoke uses `max_chains=96`, 3 epochs.

**Starting weights are GUESSES** (no Step-1a GPU result yet):
- `w_sep = 0.5` — a primary discriminative term (peer of `w_nce=0.25`/`w_margin=0.25`, set
  higher because it carries the headline-metric supervision). Drop toward 0.25 if it
  dominates / destabilizes `L_token`.
- `w_verb_anchor = 0.25` — a light bias on the bottleneck (only ~5% of rows supervised, so its
  raw magnitude is small). Raise toward 0.5 if `disc_delta_r2` barely moves.

## 5. Provenance

- **`L_verb_anchor`**: latent-action-model (LAM) augmentation with sparse action labels — the
  "1% labels flip PCA-on-noise → PCA-on-action" observation. A small supervised anchor on the
  action factor reshapes the unsupervised bottleneck's principal axis.
- **`L_sep`**: supervised contrastive learning (SupCon, Khosla et al. 2020) + manifold-packing.
  Same-class compactness + cross-class separation, with paraphrase-invariance from the
  canonical-state label and "what-else-could-change" siblings as hard negatives.

## 6. Success readout (geometry probes)

`scripts/run_geometry_probes.py --labeled_dir data/entity_world --split test_iid`:

| metric | direction | source |
|---|---|---|
| `ent_disc_delta_r2` | **up** (from ~0) | `diagnostics._discriminative_variance_ratio` — `L_verb_anchor` directly biases the bottleneck toward the action delta |
| `ent_separation_auc` | **off 0.50, toward >0.7** | `diagnostics._separation_auc` — `L_sep` trains its query (`zhat`) geometry directly |
| `ent_latent_nn_purity*` | **up** | `diagnostics._latent_nn_purity` — packing same-next-state neighbors |

**Guard (must not regress):** `L_token` (primary CE), `ent_rollout_tf_exact_d4` (dynamics
fidelity), `ent_action_nmi_verb` (verb informativeness). If `L_token` climbs materially the
weights are too high.

## 7. Smoke gate

`jepa_v5_smoke.json` (3 epochs, `max_chains=96`, batch 32) must: (a) run to completion +
write a checkpoint; (b) emit `L_sep > 0` and `L_verb_anchor > 0` from epoch 1 (terms wired +
active) and keep them finite / trending down; (c) keep `L_token` in the v4-smoke ballpark.
A run of any v4 config (terms still default-0.0) stays **bitwise-identical** — the regression
gate that the default-0.0 contract holds.
