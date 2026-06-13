# Latent Geometry Spec v1 — Locked

Status: **v1, locked.** This is the working contract for what a "good" entity-world
latent must look like. It supersedes the ad-hoc per-experiment latent checks scattered
across the v4.x logs. Five properties, one demoted property, and a small battery of
probes that read them off.

**This spec is not a gate sheet.** The metrics below are *relative-tracking dials*, not
pass/fail thresholds. The only real acceptance gate is demo-usability (see §2). Treat
every number in this doc as a tracking instrument: structured-vs-black-box, variant-vs-
variant, before-vs-after. v1 is the first draft of the instrument set, not a verdict.

---

## TL;DR

- The latent is **Healthy** (isotropic, full-rank, SIGReg solved it) and **not much else
  that matters.** Healthy is necessary, but it is doing none of the work the demo needs.
- Three of the five properties are **BROKEN**: Addressable (collapses to ~1 usable slot),
  Discriminative (separation AUC stuck ~0.50), Faithful-under-dynamics (rollout exact@d4
  = 0.0, dynamics_gap 0.82).
- **Properties 1 and 3 dissociate, and that is the diagnosis.** Token-CE rewards getting
  the template right; isotropy spreads points evenly. Neither objective contains a term
  that *separates confusable-but-distinct items*. The latent is round but unaimed. The
  fix is a **missing objective** (a discriminative term + a slot-use pressure), not more
  capacity and not a different operator shape.
- The v4.x evidence is consistent and damning: separation AUC ~0.50 across
  v4.0/v4.0.1/v4.2/v4.3; the structured operator ties-or-loses to a gated-MLP black-box;
  rollout exact@d4 = 0.0 everywhere; v4.1 pooled-InfoNCE (the one config that targets the
  AUC geometry directly) never ran. There is currently **no loss that pressures slot use**
  — only a zero-gradient `slot_entropy` diagnostic.
- The plan is **usage-first**: Step0 runs four new probes on frozen v4.x checkpoints to
  baseline and confirm the diagnosis; Step1a adds use-pressure + a discriminative term at
  **fixed M=8** and asks whether occupancy and disc-var-ratio rise; Step1b scales M *only
  if* usage is fixed and the discriminative term helps. We do not scale capacity before we
  fix usage.

---

## 1. Acceptance model

### Relative dials, not gates

The five-property battery is an **instrument panel**. Each metric is read three ways:

1. **Structured vs black-box** — does the structured (polar / rotation+scale / norm-budget)
   operator beat the `GatedMLPTransition` black-box on this dial? (v4.x answer so far:
   mostly no.)
2. **Variant vs variant** — does v4.x+use-pressure beat v4.x? Does M=16 beat M=8 *after*
   usage is fixed?
3. **Before vs after** — does the dial move when we add the objective the diagnosis says
   is missing?

No single number on this panel is a release criterion. A model can post a great
`ent_separation_auc` and still be useless in the demo, or a mediocre AUC and still drive a
believable pet. The dials exist to *localize where the latent is failing* and to *tell us
whether an intervention moved the thing it was supposed to move*.

### The one real gate: demo-usability

The acceptance gate is whether the latent **drives a believable toy pet-sim**: you can
inspect a belief, edit one belief, advance the world, and have the change behave locally
and stay on-manifold across a short rollout — convincingly enough that the demo reads as
"this entity remembers and updates." Everything in the property table is in service of
that gate. If a property is BROKEN on the dials but the demo is fine, the demo wins; if the
dials are green but the demo is incoherent, the dials were measuring the wrong thing and
v2 of the spec fixes them.

This is why properties are scored *relative* and why property 5 is explicitly negotiable
(see below): the means are tradeable, the demo is not.

---

## 2. Properties

Critical path = **{2 Addressable, 3 Discriminative, 4 Faithful}**. Properties 1 and 6 are
either solved or demoted; property 5 is a negotiable means judged only by its effect on
2/3/4.

| # | Property | Definition | Metrics (NEW probes in **bold**) | Current status |
|---|----------|------------|-----------------------------------|----------------|
| 1 | **Healthy** | Isotropic, full-rank, no collapse — the bottleneck uses its dimensions. A *marginal-distribution* property. | `noun_eff_rank`, `modulus_eff_rank`, `noun_per_dim_var` | **SOLVED** (SIGReg). The latent is round and full-rank; this is no longer where the work is. |
| 2 | **Addressable** | Latent factors into MANY cleanly-usable units (NOT a required slot=entity map); you can inspect/edit one belief. | **`ent_slot_occupancy`** (participation ratio of per-slot energy) + LOO-constructive (masking any slot raises CE) | **BROKEN.** Collapses to ~1 usable slot. There is NO use-pressure loss — only a zero-gradient `slot_entropy` *diagnostic* that the optimizer cannot follow. |
| 3 | **Discriminative** | Gold next-state separable from confusable distractors; latent-NN = semantic-NN. A *pairwise-separation* property. | `ent_separation_auc`, `hard_mrr`, **`ent_latent_nn_purity`**, **`ent_disc_var_ratio`** | **BROKEN.** AUC ~0.50 (chance). The latent is round-but-unaimed: roundness ≠ separation. |
| 4 | **Faithful-under-dynamics** | Rollout stays on-manifold; edits are LOCAL — editing one belief perturbs only its causal cone. | `ent_rollout_tf_exact_d4`, `dynamics_gap`, `held_out_cos_mean`, **`ent_cf_locality`** | **BROKEN.** Rollout exact@d4 = 0.0; `dynamics_gap` = 0.82 — the rolled latent never re-aligns with the encoder manifold. |
| 5 | **Interpretably-structured** (NEGOTIABLE MEANS) | noun = identity / adjective = state is *readable*. Judged ONLY by whether it improves 2/3/4. | `ent_action_nmi_verb`, modulus/phase decodability, `identity_persistence_err` | **Present, usefulness unproven.** The polar split decodes, but it has not yet been shown to move the critical-path dials. If it never does, it is overhead. |
| 6 | ~~Retrain-stable~~ (DEMOTED) | ~~Seed-invariant latent frame.~~ Mount via a trained projection adapter instead, so seed-invariance is NOT required. | — (mount adapter absorbs the frame nuisance) | **DEMOTED.** We pay for retrain-instability at the mount, not in the latent. The adapter is trained per-checkpoint; the latent owes nothing to a downstream seed. |

### The four NEW probes (property 2/3/4 instruments)

These are the additions v1 introduces. They exist because the v3/v4 battery measured
*health* and *capability* well but measured *usage* and *separation* poorly.

- **`ent_slot_occupancy`** (prop 2) — participation ratio of per-slot energy across the M
  slots. PR near M = energy spread across all slots; PR near 1 = collapse to a single slot.
  This is the *usage* dial. Pairs with LOO-constructive (does masking each slot raise CE?)
  to confirm the spread is *load-bearing* and not decorative spread.
- **`ent_latent_nn_purity`** (prop 3) — fraction of a state's latent nearest-neighbors that
  are its true semantic neighbors. Directly tests "latent-NN = semantic-NN." Complements
  the AUC: AUC measures gold-vs-distractor separability, purity measures local neighborhood
  correctness.
- **`ent_disc_var_ratio`** (prop 3) — ratio of between-class variance to within-class
  variance in the latent (a Fisher-style separation ratio over confusable groups). This is
  the *aimed-ness* dial: a round-but-unaimed latent has ratio ≈ 1; a discriminative latent
  has ratio ≫ 1. It is the metric the missing discriminative objective should move.
- **`ent_cf_locality`** (prop 4) — counterfactual locality: edit one belief in the latent,
  advance, and measure how much of the *unrelated* state changed. Local edits perturb only
  the causal cone; non-local edits smear. This is the *surgery-quality* dial the pet-sim
  demo depends on directly.

---

## 3. Root cause — properties 1 and 3 dissociate

The crux of v1 is a dissociation that the existing evidence already forced:

> **A latent can be Healthy and not Discriminative at the same time, and the two
> properties are constrained by *different objectives*.**

- **Healthy is a marginal-distribution property.** "Round, full-rank, no collapse" is a
  statement about how the *whole cloud* of latents is distributed. Isotropy regularization
  (SIGReg / LeJEPA-style) enforces exactly this: spread the points uniformly, fill the
  dimensions. It says nothing about *which* point goes where relative to its confusers.
- **Discriminative is a pairwise-separation property.** "Confusable-but-distinct items get
  distinct codes" is a statement about *pairs*: near-duplicates must land apart. No
  marginal-distribution constraint implies this.

Token cross-entropy does not close the gap. CE rewards getting the *template/topic* right
— it is satisfied as soon as the decoder emits the right surface for the right family of
states. Two confusable-but-distinct states that share a template incur near-zero extra CE
for being collapsed together. So:

- CE pressures **template correctness** (right words),
- isotropy pressures **roundness** (even spread),
- and **nothing pressures separation** (confusable pairs land apart).

The result is the observed signature: **round but unaimed.** The points fill the space
evenly (AUC's true-positive and distractor distributions overlap because the latent has no
reason to push them apart), so `ent_separation_auc` sits at ~0.50 even while
`noun_eff_rank` is near full.

**This is a missing-objective problem, not a capacity or operator-shape problem.** Adding
dimensions (the dn64 capacity arm) gives the cloud more room to be round in — it does not
add a separation term. Changing the operator (rotation vs rotation+scale vs gated-MLP)
changes the *transition*, not the *separation pressure on the state cloud*; consistent
with v4.x, the structured operator ties-or-loses to the black-box on the critical-path
dials because the thing that is broken is upstream of the operator. The fix is to *add the
term that is missing*:

1. a **discriminative term** (contrastive / hard-negative separation) that directly
   pressures `ent_disc_var_ratio` and `ent_separation_auc`; and
2. a **slot-use pressure** with a real gradient (not the zero-gradient `slot_entropy`
   diagnostic) that pressures `ent_slot_occupancy` and LOO-constructiveness.

Property 5 (interpretable structure) is downstream of this, not a substitute for it: a
readable noun/adjective split does not separate confusers unless a separation objective
makes it do so.

---

## 4. The v4.x evidence that motivated the spec

The spec exists because the v4.x family kept failing in the *same* way regardless of which
knob was turned. The pattern is what told us the problem was a missing objective, not a
missing capability.

| Observation (v4.x) | What it rules out |
|---|---|
| **`ent_separation_auc` stuck ~0.50** across v4.0, v4.0.1, v4.2, v4.3 | Not a tuning/seed fluke; chance-level separation is *stable* across four configs → the objective set never contained separation pressure. |
| **Structured operator ties-or-loses to the gated-MLP black-box** on the critical-path dials | The breakage is *upstream of the operator*. A better/worse transition does not move a state cloud that has no separation pressure on it. (Mirrors the v3 verdict: structured wins only on *capability* dials — O(1) retraction, far-OOD CE — not on the system-level retrieval/rollout metrics.) |
| **`ent_rollout_tf_exact_d4` = 0.0 everywhere**; `dynamics_gap` ≈ 0.82 | Faithful-under-dynamics is broken independently of operator choice; the rolled latent never re-aligns with the encoder manifold (consistent with hard-MRR ~0.11–0.12 and held-out-cos ~0.25 in v3). |
| **No slot-use-pressure loss exists** — only a zero-gradient `slot_entropy` diagnostic | The optimizer was never *told* to use the slots. Addressable collapse to ~1 slot is the expected outcome of measuring usage without pressuring it. (Mirrors v3's binding_disagreement 0.561 / model_same 0.922 — ~92% of actions collapsed onto one dominant slot.) |
| **v4.1 pooled-InfoNCE never ran** | The one config that targets the AUC geometry *directly* (a pooled hard-pool InfoNCE term, `L_pool_nce`) is untested — so "we tried separation pressure and it didn't help" is **not** a claim we can make. The discriminative lever is unexplored, not refuted. |

The honest read: v4.x explored *transition shape* (v4.0 operator variants), *objective
placement* (v4.2 masked-diff, v4.3 paraphrase), and *I/O* thoroughly, but never put a
gradient on either of the two things the diagnosis says are missing — separation and slot
use. The single config that would have ([`L_pool_nce`], v4.1) was authored and shelved.

---

## 5. Plan — usage-first sequencing

The sequencing is deliberate and the order is load-bearing: **fix usage before scaling
capacity.** Scaling M before slots are constructively bound just adds decorative slots; the
v1→v2 slot-LOO flip evidence (same dn, opposite addressability, fixed by an objective
change) says capacity was never the gate.

### Step0 — baseline + confirm the diagnosis (frozen checkpoints)

Run the four NEW probes (`ent_slot_occupancy`, `ent_latent_nn_purity`,
`ent_disc_var_ratio`, `ent_cf_locality`) on **frozen v4.x checkpoints** — no training. Two
jobs:

1. baseline the dials on the existing structured and black-box arms; and
2. confirm the diagnosis directly: a Healthy latent (`noun_eff_rank` near full) that posts
   `ent_disc_var_ratio` ≈ 1 and `ent_slot_occupancy` ≈ 1 is the round-but-unaimed,
   single-slot signature §3 predicts.

Step0 produces no model; it produces the before-numbers every later intervention is read
against. It is read-only on `results/`.

### Step1a — add use-pressure + discriminative term, **hold M = 8**

Add both missing objectives at **fixed M = 8 / fixed dn**:

- a **slot-use pressure** with a real gradient (replacing the zero-gradient `slot_entropy`
  diagnostic); and
- a **discriminative term** (the shelved `L_pool_nce` pooled-InfoNCE is the natural first
  candidate; v4.1's design exists).

The question Step1a answers, on the Step0 baseline: **does `ent_slot_occupancy` rise (slots
get used) and does `ent_disc_var_ratio` rise (the latent gets aimed)?** Capacity is held
constant precisely so any movement is attributable to the objective, not to room.

### Step1b — scale M, **only if usage is fixed and helps**

Scale M (e.g. M=8 → M=16) **only after** Step1a shows usage is fixed *and* the
discriminative term helps. If Step1a does not move occupancy/disc-var-ratio, adding slots
is wasted — the gate is the objective, not the slot count. If Step1a *does* move them, then
M-scaling is the test of whether more addressable atoms convert into more
critical-path performance — at which point it is a clean capacity test rather than a
confound.

---

## 6. Mapping onto the substrate open questions

This spec is the resolution-criteria layer for two narrowed substrate open questions. The
critical-path dials are the falsifiable tests those questions have been waiting for.

- **`bottleneck-capacity-vs-geometry`** — The question asks whether the
  "right-templates-wrong-items" pathology is (a) capacity, (b) discreteness, or (c)/(d)
  operation-shape. This spec lands squarely on the substrate's *narrowed* verdict: §3's
  Healthy-but-not-Discriminative dissociation is the operation-first framing made
  measurable — a latent that uses its dimensions (capacity present) yet fails separation
  (objective absent) **falsifies framing (a)** at this scale. The discriminative term in
  Step1a is the test of framing (c)/(d): does shaping the objective to the operation the
  demo needs (separate confusers, use slots) fix it at fixed capacity? Step1b's M-scaling
  is the explicit capacity discriminator the question still wants (the analogue of the
  queued dn64 arm).

- **`right-addressable-atom`** — The question's own verdict is that *addressability is not
  a property the atom has; it is a property the training objective must force* — usage
  statistics cannot certify it, only content-dependence probes can. Property 2 of this
  spec is exactly that, operationalized: `ent_slot_occupancy` is the usage statistic (which
  the substrate warns is insufficient on its own), and LOO-constructiveness is the
  content-dependence probe that *earns* the "addressable" label. The current "collapse to
  ~1 slot, only a zero-gradient diagnostic" status is the negative instance the question
  predicts; Step1a's slot-use pressure is the attempt to force the constructive binding the
  substrate's v1→v2 slot-LOO flip showed was achievable at fixed capacity.

These mappings are pointers, not new substrate claims. The two proposed-but-uncreated nodes
the reconciliation digest flagged — `slot-budget-gates-addressable-capacity` (capacity
necessary, use-pressure also necessary; a 2-D slot-count × use-pressure failure surface)
and `reconstructive-objective-yields-unaimed-latent` (CE + isotropy give round-but-unaimed)
— are precisely the §3 root-cause and the §5 Step1a/Step1b plan restated as substrate
hypotheses, and are the natural write-back targets once Step0/Step1a land numbers.

---

*v1 is a tracking instrument, not a verdict. Every status above is a snapshot of the dials,
not a pass/fail. The next revision of this spec should be driven by Step0 baselines and the
first Step1a before/after.*
