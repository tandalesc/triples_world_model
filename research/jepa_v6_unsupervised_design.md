# JEPA v6 — Unsupervised (Label-Free) Latent-Action World Model

**Status: IMPLEMENTED (local smoke + invariance contract test green; pending GPU
validation).** Supersedes the DEPRECATED v5 supervised arm
(`research/jepa_v5_discriminative_design.md`). v6 is the LABEL-FREE realization of the
project's load-bearing commitment: an UNSUPERVISED world model whose value is transfer to
UNSEEN entities/processes. **No oracle/ground-truth signal enters the TRAINING loss.** The
oracle (`src/twm/jepa/entity_labels.py`, the `*_labeled.jsonl` twins) is retained for
EVAL/DIAGNOSTICS ONLY — it is the measuring stick, never a training input.

## 0. Why v5 was wrong and what v6 replaces it with

Step-0 (v4 at `n_slots=8`) measured the root failure on the label-free geometry probes:
`disc_delta_r2 ≈ 0`, `separation_auc ≈ 0.50`, `slot_occupancy ≈ 7/8` (slots ARE used — not
capacity), `rollout exact@d4 = 0`, `dynamics_gap 0.82`. The latent is round but **unaimed**:
CE pressures template correctness, isotropy pressures roundness, and *nothing pressures the
controllable-delta separation*. v5 fixed the aim with ORACLE labels (`L_verb_anchor` = GT
verb CE; `L_sep` = SupCon keyed on oracle canonical-next-state / changed-attribute ids).
That violates the thesis — the project already burned itself once on a gameable injected
signal (the v1 `L_div` usage-entropy term), and the substrate verdict is explicit:
"addressability is not a property the atom *has*; it is a property the training objective
must *force* — usage statistics cannot certify it." The fix must arrive as **structural
necessity / self-supervised pressure**, at fixed capacity (M=8, dn=32, V=8 HELD), and be
provably label-free.

| Removed (supervised, v5) | Replaced by (label-free, v6) |
|---|---|
| `L_verb_anchor` — oracle-verb CE re-aims the bottleneck PCA axis | **`L_lam_inv`** — surface-paraphrase invariance re-aims the axis to the controllable delta, no labels (PRIMARY) |
| `L_sep` — SupCon on oracle canon-id / changed_attr, oracle-sibling hard negs | **`L_self_nce`** — paraphrase / inferred-action self-positive InfoNCE on `zhat`, chain-sibling hard negs (SECONDARY, default-off) |

## 1. PRIMARY — `L_lam_inv`: LAM surface-augmentation invariance (arXiv:2506.15691)

**Grounding.** "What do latent action models actually learn?" (arXiv:2506.15691) shows the
LAM objective is *PCA-on-variance*: the inferred action `v` aligns with the highest-variance
**surface** direction (template/word choice) rather than the controllable delta. The fix is
purely augmentational — and it is the exact mechanism the paper verifies. It is the
label-free version of the "1% labels flip PCA-on-noise → PCA-on-action" lever that the
deprecated oracle CE was illegitimately providing.

**Mechanism (no labels anywhere).** Apply the SAME surface frame φ to BOTH posterior inputs
`(s_t, s_{t+1})`, and a DIFFERENT frame φ′ to the decoder's teacher-forced CE target
`s_{t+1}`:

- Posterior sees `q(v | φ(s_t), φ(s_{t+1}))`. Surface variance *common to both* (held
  constant across the φ pair) carries no transition information, so the posterior **cannot
  route it into `v`**.
- The decoder is teacher-forced to reconstruct `φ′(s_{t+1})` from `a* = B_v k` (memory =
  `a*` only). Producing surface-frame φ′ from conditioning built in frame φ means the only
  frame-invariant signal the decoder can rely on through the ⌈log₂V⌉-bit `v` is the
  **controllable semantic delta** — surface variance is cancelled by construction.
- Net: `v` (and via `a*`, the readout geometry the AUC/disc probes read) is pushed onto
  *what changed*, not *how it was worded*.

**Why this transfers to unseen entities (not iid overfit).** Invariance to surface form is a
property of the *rendering*, not of any entity's identity or dynamics. A model forced to be
surface-invariant on seen types learns a representation indexed by *semantic change* — the
shared axis across all types, including unseen `puppy`/`terrarium`. Memorizing iid surface
co-occurrences is exactly what the augmentation destroys. This is the JEPA
"structure-substitutes-for-scale" bet applied to the *objective*.

**Implementation (this is wiring, not a new loss-math term).** `L_lam_inv` is the
`L_token` CE of an EXTRA augmented forward, scaled by `w_lam_inv`:
- `src/twm/jepa/model.py` `forward(...)` gains two optional overrides (default None ⟹
  bitwise v4):
  - `posterior_inputs = (p_src_ids, p_src_pad, p_tgt_ids, p_tgt_pad)` — frame φ for the
    posterior AND the noun-path encoder of `s_t` (and the targeted-mask `k_tgt`, which reads
    the φ frame of `s_{t+1}` — a location signal from the same φ pair).
  - `decoder_target = (d_tgt_ids, d_tgt_pad)` — frame φ′ for the decoder CE target AND the
    EMA `z_target` (so the InfoNCE/L_pred anchor is contrasted against the same φ′ frame).
- `scripts/train_jepa_v2.py`: when `w_lam_inv>0` and the dataset is `lam_augment`, `_pair_step`
  / `_unroll_step` run one extra `model(...)` with `decoder_target=(φ′)` and add
  `w_lam_inv · token_ce(aug_logits, φ′_ids)`. The base forward stays the φ-frame path.
- **Leakage invariant preserved** (`research/jepa_v2_latent_actions.md` §6): decoder memory is
  still ONLY `a*`; φ′ enters only as the AR teacher-forced TARGET; `v` is inferred only from
  the φ pair. Pinned by `tests/jepa/test_lam_invariance.py` (perturbing φ′ does NOT change
  `v`'s argmax / `k` / `a*` / posterior logits, while the decoder logits DO change).

## 2. PLUGGABLE AUGMENTATION INTERFACE (transfer-honesty)

The augmentation is a PLUGGABLE step that yields **"two independent surface views of the same
transition"** — it is NOT the synthetic renderer wired into the loss. The contract the loss
sees is rendered **TEXT only**: a `chain` field (frame φ) and a `chain_aug` field (frame φ′)
per chain, both rendering the SAME underlying state sequence. The loss/trainer NEVER touch an
oracle state or label — the underlying-state access that produced the two frames belongs to
the **data generator**, not the trainer (the same legitimacy line the lead drew: augmentation
= input transform = allowed; oracle-derived training target = forbidden).

- **Concrete entity_world provider (this PR).** `scripts/generate_entity_world.py` gains a
  default-off `emit_lam_aug` config flag. When set, each chain ALSO emits `chain_aug`: a second
  independent surface frame rendered via `render_state(..., surface_variety=True, seed=φ′)` /
  `render_action(..., seed=φ′)` with a **fresh `chain_template_seed`** (independent of the φ
  frame's seed). Both frames render the identical state sequence (the renderer is a pure
  function of `(seed, state)` — an input transform). The `chain_aug` text is the only thing the
  data loader reads; no oracle field enters the unlabeled `.jsonl`.
- **Data loader (`src/twm/jepa/data.py`).** `JEPAChainDataset(lam_augment=True)` reads the
  `chain_aug` field when present and exposes the φ posterior tensors (aliases of the existing
  `chain` tensors) and the φ′ decoder-target tensors. Off / field-absent ⟹ bitwise v4.
- **Swappability (the transfer-honesty note).** A future TEXT-LEVEL augmenter — a paraphrase
  model, back-translation, an LLM rewrite — slots in by supplying the SAME `chain_aug` text
  field (or, equivalently, an on-the-fly callable that maps a transition's text to a second
  surface view). NOTHING in the model/loss is renderer-specific: the model takes φ/φ′ token
  tensors, the loss takes the φ′-target CE. **Risk/assumption:** the renderer is a
  domain-specific augmenter (entity_world templates). Its surface-entropy is what the LAM-
  invariance lever consumes; a domain without a clean paraphraser (free-form prose) needs a
  text-level augmenter to supply `chain_aug`. The interface is built so that swap is a
  data-side change only — the prerequisite for honest cross-domain transfer claims.

## 3. SECONDARY — `L_self_nce`: self-supervised contrastive on `zhat` (default-off)

InfoNCE on the pooled predicted-next-state readout `zhat` (the vector
`diagnostics._separation_auc` queries) — the direct discriminative gradient the geometry is
missing. Re-targets the SHAPE of the deprecated v5 `sep_supcon` onto **self-supervised**
positives (the model's own signal, never an oracle id):
- `self_nce_positive="paraphrase"`: a different surface frame of the SAME transition (the φ′
  augmentation source) — in pairs mode realized as the nearest in-batch same-chain sibling.
- `self_nce_positive="inferred_action"`: in-batch transitions sharing the model's OWN argmax
  inferred action code `v` (`out["v"]` — a self-label, NOT the oracle verb).
- Negatives = other in-batch; same-chain siblings up-weighted (`self_nce_hard_neg_weight`)
  via the existing LABEL-FREE `chain_ids` plumbing. The positive column is detached
  (BYOL/MoCo asymmetry — a self-label cannot drive representational collapse).
- Config (default-off): `w_self_nce=0.0`, `self_nce_temperature=0.1`,
  `self_nce_hard_neg_weight=2.0`, `self_nce_positive="paraphrase"`.

**Mode caveat (logged).** In `triples` mode the dataset emits ONE example per chain, so
same-chain in-batch siblings do not exist → the `"paraphrase"` positive finds no in-batch
positive and contributes 0 (correct, documented). For triples mode use
`self_nce_positive="inferred_action"` (groups by the model's own argmax `v` across the batch),
which was verified to activate (`L_self_nce ≈ 6.2` at smoke ep1). `L_self_nce` is recommended
default-off; `L_lam_inv` is the primary lever (it fixes the root PCA-on-variance pathology),
with `L_self_nce` as a stackable arm to test whether explicit pairwise separation adds beyond
invariance.

## 4. Configs (capacity HELD: M=8, dn=32, V=8; structured block from v4)

- `configs/jepa/jepa_v6_lam_s0.json` — structured (rotation_scale, polar, norm_budget,
  targeted), `lam_augment=true`, `w_lam_inv=0.5`, `w_self_nce=0.0`. Data
  `data/entity_world_para_lam/` (paired-frame, regenerated by the GPU job with
  `emit_lam_aug=true`).
- `configs/jepa/jepa_v6_blackbox_s0.json` — `operator_group=gated_mlp`, SAME losses. The
  structured-vs-blackbox A/B is an **architecture** comparison (allowed — it is not a
  supervision ablation).
- `configs/jepa/jepa_v6_smoke.json` — tiny (`max_chains=64`, 3 epochs) on the local
  `data/entity_world_para_lam/` paired-frame dataset.

The v5 supervised configs are retired to `configs/archive/jepa_v5_supervised/`.

## 5. Validation (local, M5 MPS/CPU)

1. **Bitwise-v4 regression (§A):** a v4 config builds an IDENTICAL model + computes an
   IDENTICAL loss before vs after the v5 removal (loss components, TOTAL=10.532975196838379,
   param count 666842 all bitwise-equal via a fixed-seed forward, `git stash` toggling the
   tree). The removed terms were default-0 and additive ⟹ this holds.
2. **Smoke:** `jepa_v6_smoke.json` runs 3 epochs — `L_lam_inv` path active (`≈9.69`),
   `L_self_nce=0.0` (off by default), no NaN, `L_token≈9.66` (v4 ballpark), checkpoint
   written.
3. **Invariance contract test:** `tests/jepa/test_lam_invariance.py` — 3 tests green.
4. **Geometry probes:** `run_geometry_probes.py --labeled_dir data/entity_world_para_lam`
   produces finite `ent_*` metrics on the smoke checkpoint.

## 6. SUCCESS READOUT (eval-side; oracle = ruler only)

The headline is the **OOD-ladder gap** — the metric gap between `test_iid` and
`test_ood_near`/`test_ood_far` should SHRINK (`ent_disc_delta_r2` up from ~0;
`ent_separation_auc` off 0.50 toward >0.7; `ent_latent_nn_purity` up; rollout faithfulness
up). Per `research/latent_geometry_spec.md`, these are **tracking dials, not targets** — read
structured-vs-blackbox and before-vs-after; the REAL gate is **demo-usability** (a believable
pet-sim with local on-manifold edits). **Do NOT chase iid AUC.** Guards that must not regress:
`L_token`, `ent_action_nmi`, the `v`-ablation CE gap. The oracle (`*_labeled.jsonl`,
`entity_labels.py`) is the measuring stick for these eval probes ONLY — it never enters the
training loss.
