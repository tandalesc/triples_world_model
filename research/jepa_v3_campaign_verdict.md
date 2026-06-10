# JEPA Entity-World v3 Campaign — Final-Checkpoint Verdict

Status: **final.** Both v3 entity checkpoints are FINAL ep100, banked, jobs closed.
Battery run on the homelab box (CPU, thread-cap 8), read-only on `results/`; checkpoints
were snapshot-copied to `/tmp/jepa_battery/ck/` before loading. Server checkout `ef07f40`.
Probe outputs: `results/jepa_v3_final_battery/*.json`.

Checkpoints under test:
- **structured** — `results/jepa_ent_s0/model_latest.pt`: `rotation_scale` operator +
  polar conditioning (`use_polar_conditioning=true`) + norm budget (`use_norm_budget=true`).
- **black-box** — `results/jepa_ent_blackbox_s0/model_latest.pt`: `GatedMLPTransition`
  (no polar, norm-budget flag is a no-op; `inverse_apply` raises `NotImplementedError`).

The v3 *training* logs carried no `ent_*` entity diagnostics (the in-loop entity-eval hook
errored and was caught). The ep100 entity finals below were re-extracted from the dedicated
v3 diagnostic jobs that evaluated these exact final checkpoints (structured = wartable
`60a91b6b`, seed-0 block; black-box = `869435af`, seed-0 block); they match the pre-registered
anchors in `research/jepa_entity_campaign.md`.

---

## Pre-registered targets (`research/jepa_entity_campaign.md`)

> - structured **hard-MRR > 0.4** on `test_iid`
> - **action-recovery NMI ≥ 0.2** vs oracle labels (with the `<0.25` failure clause)
> - **OOD ladder** evals (iid ≥ near > far)
> - **rollout exact-match** vs oracle at depth 4 (target ≥ 0.3)

## ep100 entity-diagnostic finals (re-extracted from the checkpoints' diagnostic jobs)

| metric (ep100) | structured (s0) | black-box (s0) |
|---|---:|---:|
| `ent_action_nmi_verb` | **0.3091** | **0.2494** |
| `ent_action_nmi_verb_entity` | 0.2903 | 0.2642 |
| `ent_action_nmi_shuffle` (baseline) | 0.0062 | 0.0066 |
| `ent_action_nmi_verb_pass` (≥0.2) | True | True |
| `ent_test_iid_hard_mrr` | **0.1187** | **0.1070** |
| `ent_test_ood_near_hard_mrr` | 0.1286 | 0.1291 |
| `ent_test_ood_far_hard_mrr` | 0.1085 | 0.1155 |
| `ent_test_iid_ce` (nats) | 1.2803 | 1.6112 |
| `ent_test_ood_near_ce` | 4.8347 | 4.0091 |
| `ent_test_ood_far_ce` | **6.0886** | **7.4228** |
| `ent_test_iid_chrf` | 0.4690 | 0.5208 |
| `ent_test_ood_far_chrf` | 0.3026 | 0.3732 |
| `ent_ladder_monotone_mrr` | False | False |
| `ent_rollout_tf_exact_d4` | **0.0000** | **0.0000** |
| `ent_rollout_pr_exact_d4` | **0.0000** | **0.0000** |
| `ce_gap_nats` (ablation) | 0.5037 | 0.2025 |

---

## VERDICT TABLE vs pre-registered thresholds

| pre-registered claim | threshold | structured | black-box | held? |
|---|---|---:|---:|---|
| action-recovery NMI | ≥ 0.20 | **0.3091** (pass) | **0.2494** (pass) | **BOTH PASS** (≫ shuffle ~0.006) |
| structured hard-MRR (iid) | ≥ 0.40 | 0.1187 | 0.1070 | **BOTH FAIL** |
| hard-MRR `<0.25` failure clause | fires if MRR < 0.25 | 0.1187 < 0.25 | 0.1070 < 0.25 | **CLAUSE FIRES (both)** — retrieval geometry never formed |
| OOD ladder monotone (iid ≥ near > far) | MRR non-increasing | False (near 0.129 > iid 0.119) | False (near 0.129 > iid 0.107) | **BOTH FAIL** — MRR is flat-to-non-monotone across the ladder (all ≈ 0.11) |
| rollout exact @ d4 | ≥ 0.30 | **0.0000** | **0.0000** | **BOTH FAIL** (0.0 exact, TF and PR) |
| structured-vs-blackbox far-CE | structured lower | far-CE 6.09 | far-CE 7.42 | **structured better** by 1.33 nats — the operator's one clean OOD win |

### Battery-derived verdicts

| probe | result | verdict |
|---|---|---|
| RETRACTION (structured) | retract beats do-nothing (0.225 vs 0.216 cos); bracket **not** monotone; dynamics_gap = **0.818** | retraction works *directionally* but is swamped by rollout infidelity |
| RETRACTION (black-box) | `inverse_apply` raises `NotImplementedError`; `inverse_supported:false`, exits 0 | **asymmetry confirmed** — the capability gap |
| COMMUTATOR pure-matrix | mean 2.6e-07 (max 1.5e-06) | abelian sanity **PASS** (~0) |
| COMMUTATOR selection-incl. | mean 1.124 (rel 0.070) | SELECTIONS are path-dependent (H non-commutes) |
| binding_disagreement | 0.561 (model_same 0.922 vs oracle_same 0.463) | **poor grounding** — model collapses most actions onto one slot |
| SOFT-QUOTIENT | latent_dist 3.06, pred-JS 0.023, ratio **0.0076** | **ledger is LIVE** — decoder performs the quotient |
| ENGRAM OOD hit-rate | iid 0.99995, near 0.99987, far 0.99396 | **prediction REFUTED** — table does **not** collapse OOD |

---

## 1. RETRACTION four-point bracket (structured, n=200, K=4, j=2)

All rungs are cosine to the SAME reference: a fresh encode of the oracle's true *j-deleted*
final state.

| rung | cos ↑ | meaning |
|---|---:|---|
| do_nothing (rolled WITH event j) | 0.2158 | worst — j never removed |
| algebraic_retraction (inverse of j on contaminated roll) | **0.2249** | structured inverse applied |
| model_replay (honest j-deleted re-roll) | 0.1819 | clean counterfactual rollout |
| reencode_ceiling (reference identity) | 1.0000 | upper bound |

- **`retract_beats_donothing = True`** (0.2249 > 0.2158): the structured inverse *does* move
  the rolled latent toward the oracle-without-j target. The capability is real.
- **`bracket_monotone = False`**: the bracket is NOT do_nothing ≤ retract ≤ replay ≤ ceiling.
  `model_replay` (0.182) sits *below* `algebraic_retraction` (0.225), so:
  - **selection_drift = −0.043** (model/selection side). Negative ⇒ the algebraic inverse
    on the contaminated path *beats* the honest fresh re-roll — i.e. the H-path-contamination
    penalty is dwarfed by the re-roll's own accumulated error. There is no selection-side
    penalty to fix; the honest counterfactual is simply *worse* than the surgical undo.
  - **dynamics_gap = +0.818** (world/dynamics side). This is the entire story: even the best
    re-roll lands at cos ≈ 0.18 against a fresh encode. The rolled latent is badly misaligned
    with the encoder manifold (consistent with iid-MRR 0.12 and held-out-cos ≈ 0.25). The
    bottleneck is **rollout/encode fidelity, not the inverse algebra.** A better inverse buys
    nothing here; a faithful world model would.

**Black-box:** `inverse_apply` raised `NotImplementedError` ("GatedMLPTransition is a black-box
transition with no structural inverse"). Recorded `{"backend":"blackbox","inverse_supported":
false}`, exit 0. The honest cost asymmetry: the structured operator retracts an arbitrary
mid-chain event in **O(1)** (one stored-offset+scale inverse application); the black-box can
only approximate retraction by **suffix-replay from a prior snapshot — O(K−j)** — and exposes
no exact inverse at all. That capability/cost gap is the operator's clearest surviving win.

## 2. COMMUTATOR — model-side vs the frozen world invoice

**Frozen world invoice** (`results/world_commutator_invoice_v1.json`, reproduced exactly this
run): overall commute 94.8%, **disjoint-entity 100%**, **same-entity 88.0%**. The world is
non-commutative *only* in the same-entity sector (ordinal saturation + conditional effects);
disjoint actions commute by construction.

**Model-side (structured, n=512):**
- `pure_matrix_defect` mean **2.6e-07** — the operator matrices are abelian to fp32 eps
  (commute exactly), matching the disjoint-100% world cell.
- `selection_defect` mean **1.124** (rel 0.070) — once H is in the loop, the *selection* (the
  effective phase angle H picks from the current modulus) is **path-dependent**, so the
  model-side composition does NOT commute even though its matrices do. This is where the
  model's non-commutativity lives: in the conditioner, not the algebra.
- **Sector correspondence:** the world's non-commutativity is same-entity (88%); the model's
  is carried entirely by H's path-dependence. But the binding is wrong (next bullet), so the
  model's selection-defect is **not** aligned to the world's same-entity sector — it is a
  generic H artifact rather than a learned image of the world's conditional structure.
- `binding_disagreement_rate` **0.561**: model_same_rate 0.922 vs oracle_same_rate 0.463. The
  model assigns ~92% of action pairs to the *same* slot while the oracle says only 46% share
  an entity — the model **collapses most actions onto one dominant slot**, so its slot-binding
  does not track ground-truth entity identity. The commutator instrument confirms the
  grounding failure the binding metric flags.

## 3. SOFT-QUOTIENT readout check (oracle-merged pairs, n=512)

Oracle-merged pairs: distinct pre-images `s1 ≠ s2` that an action saturates to the same image.

- `mean_latent_dist` = **3.062** (nonzero) — the renormalized spine keeps the two pre-images
  apart in noun geometry, exactly as designed (the budget annotates, does not destroy).
- `mean_pred_js_divergence` = **0.0234** (≈ 0) — the decoder's next-state token distributions
  for the two pre-images **converge** onto the shared merged target.
- `pred_div_per_latent` = **0.0076** (small) ⇒ **the decoder genuinely performs the quotient:**
  latent kept apart, predictions merged.

**Verdict: the norm-budget scale ledger is LIVE bookkeeping, not dead.** Irreversibility is
logged in the scalar AND the readout/decoder actually applies the many-to-one map — the
failure mode (logged-but-unmodeled quotient) does NOT fire. Inverse round-trip stays exact
(max error 1.5e-06) and swapping the table offsets for H's leaves identity-persistence and
the inverse ledger bit-for-bit equivalent.

## 4. ENGRAM OOD extension — the registered prediction

Lookup table (key = kmeans-cluster(per-slot modulus profile) × verb, value = H's offset) built
from **training trajectories only** (423 keys, 64/64 clusters used, per-key offset variance
mean 2.5e-3 — H is near-deterministic per key), then evaluated on each split with the SAME
table.

| split | hit-rate (overall) | swap-in CE − pure-H CE |
|---|---:|---:|
| test_iid | 0.99995 | +0.0016 |
| test_ood_near | 0.99987 | −0.0102 |
| test_ood_far | **0.99396** | +0.0159 |

- **PREDICTION (table hit-rate collapses on unseen entity types while H generalizes):
  REFUTED.** Hit-rate is monotone non-increasing iid→near→far but barely moves — only a
  **−0.6%** drop at far-OOD. The modulus-profile codebook fit on training types still covers
  novel entity types almost perfectly, so the deterministic table generalizes *as well as* the
  continuous H conditioner, not worse.
- **Why:** H is near-constant within each (modulus-cluster, verb) key in-domain (low per-key
  variance), and OOD entity states still land in the same 64 modulus clusters. The
  continuous-identity advantage the prediction bet on does not materialize at this scale/world
  — the lookup table is a faithful, OOD-robust stand-in for H here. The CE delta stays ≈ 0
  across all splits (table reproduces H, in- and out-of-domain).
- This is a clean **null result against the "continuous-identity beats lookup" claim** for the
  selection/conditioner component, on this entity world.

---

## Honest campaign summary (what v3 proved and didn't)

The structured operator's headline mechanisms each hold at the *unit* level — the matrices are
exactly abelian (commutator 2.6e-07), the norm-budget inverse round-trips to fp32 eps, the
scale ledger is live (the decoder performs the oracle quotient), and the structured inverse
retracts a mid-chain event in O(1) where the black-box `inverse_apply` simply raises — but the
*system-level* pre-registered targets failed across the board: hard-MRR sits at 0.11–0.12
(the `<0.25` collapse clause fires for both arms), the OOD MRR ladder is non-monotone, depth-4
rollout exact-match is 0.0 for both operators, and the retraction bracket is dominated by a
0.82 dynamics_gap showing the rolled latent never aligns with the encoder manifold. Two
registered bets were outright refuted: action-NMI passed (≥0.2, both arms), but the engram-OOD
prediction that a lookup table would collapse on unseen entity types did not hold (far-OOD hit
0.994 ≈ in-domain 0.9999), and the binding instrument exposes that the model collapses ~92% of
actions onto one slot versus 46% oracle co-location, so its non-commutativity is a generic H
artifact rather than a learned image of the world's same-entity structure. The operator's
surviving case is therefore narrow and real: **discrete capability (O(1) exact retraction the
black-box cannot do), the soft-quotient ledger working as designed, and a clean +1.33-nat
far-OOD CE edge over the black-box** — capabilities, not benchmark wins. The retrieval geometry
and faithful multi-hop rollout that would have turned those capabilities into the registered
metrics never formed at d_noun=32 / 100 epochs.
