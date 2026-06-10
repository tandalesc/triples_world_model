# JEPA Entity-World Campaign — Norm-Budget Operator, OOD-Entity Evals, Retraction & Calibration

Status: **design, fully resolved.** Builds on the live v3 substrate (`research/jepa_v3_design.md`,
`research/entity_world_design.md`, `research/jepa_matrix_synthesis.md`) and the entity-world
generator (`scripts/generate_entity_world.py`, `data/entity_world/`). Every decision a
builder needs is fixed here; there are no open choices.

This campaign serves the substrate `engram-wm` objective, task 2 (continuous
modulus-identity memory, OOD-entity generalization). It adds **one architectural change**
(the norm budget), **entity-data wiring**, an **entity-world eval suite**, a standalone
**retraction probe**, a **retrieval-ceiling calibration script**, and **entity configs**.

PRE-REGISTERED targets this build serves (do **not** weaken):
- structured **hard-MRR > 0.4** on `test_iid`
- **action-recovery NMI ≥ 0.2** vs oracle labels
- **OOD ladder** evals (iid ≥ near > far)
- **rollout exact-match** vs oracle at depth 4

Everything in §1 is config-gated and **DEFAULT FALSE**, so an unmodified v3 GLUCOSE config
builds a bitwise-identical model (the v3 behavior-preservation gate still holds). Entity
configs flip the new flags on.

---

# 1. NORM BUDGET — the one architectural change

## 1.0 Problem and the mechanism

The v3 polar operator splits each 2×2 block into a **modulus** (persistent identity) and a
**phase** (mutable state). `RotationScaleOperator` applies `r·R(θ)`: a pure rotation
(`log_r=0`) preserves modulus exactly (invertible, identity-preserving), but a scaling verb
(`r≠1`) drives the modulus toward 0 or ∞ over an unroll, and that scale change is the only
*irreversible* part of the map. The entity world's whole point is that **types respond to
the same action differently** — including state-dependent (conditional) effects — and the
continuous-identity claim needs (a) the modulus profile to stay a stable identity signal
across a multi-hop chain, and (b) the irreversibility that a real state change carries to be
*recoverable* (the retraction probe, §4) rather than lost.

**Norm budget = renormalize-and-track.** After each operator application, renormalize each
slot's modulus profile back to its **pre-application** modulus, and accumulate the extracted
global scale as an explicit per-slot scalar `s_i` in the **log domain**. The renormalized
moduli keep the *shape* of the identity profile stable across hops (only the global radius is
factored out); the extracted log-scale becomes part of world-state (readout-visible), so the
irreversibility information is **not lost** — it is *relocated* from the noun vector into an
explicit scalar that the inverse can replay exactly.

This is the invertibility-preserving mechanism: `apply()` returns `(a, scale_delta)`;
`inverse_apply` re-applies the tracked scale so the round-trip is exact.

## 1.1 Exact tensor flows

Notation (matches the live code): nouns `k (B, M, dn)`; blocks `nb = dn//2`; per-block
modulus `m_b = sqrt(x_b²+y_b²)` via `conditioning.block_modulus`; verb `v_onehot (B,V)`
broadcast to `v_slots (B,M,V)`; phase offset `θ_off = H(|·|) (B,M,nb)`.

State carried alongside the nouns is a **per-slot log-scale accumulator** `s (B, M)` (one
scalar per slot, NOT per block — the budget extracts a single global radius per slot so the
identity readout sees one irreversibility scalar per slot, mirroring the pet-sim "how far has
this entity drifted" intuition). Initialized to **zeros** (scale 1.0) at encode time.

### apply() — new return contract

`RotationScaleOperator.apply(k, v, theta_offset=None, *, norm_budget=False)`:

```
# v3 default path (norm_budget=False) — BITWISE the current apply(); returns a* tensor only.
if not norm_budget:
    return a_star                                   # exactly today's behavior

# norm-budget path (norm_budget=True):
m_pre   = block_modulus(k)                          # (B,M,nb)  pre-application moduli
a_raw   = <existing rotation+scale apply with theta_offset>   # (B,M,dn)
m_post  = block_modulus(a_raw)                      # (B,M,nb)  post-application moduli

# Per-slot global scale extracted this step = ratio of total modulus energy.
# Use the L2 norm over blocks so a single scalar summarizes the slot's radius change.
#   ρ_i = ||m_post_i|| / ||m_pre_i||   (per slot, over the nb blocks)
eps      = 1e-8
norm_post = m_post.norm(dim=-1)                     # (B,M)
norm_pre  = m_pre.norm(dim=-1).clamp_min(eps)       # (B,M)
rho       = (norm_post / norm_pre).clamp_min(eps)   # (B,M)  per-slot extracted scale
log_rho   = rho.log()                               # (B,M)  scale_delta (log domain)

# Renormalize each slot's modulus profile to its pre-application norm: divide the
# whole noun by ρ_i (broadcast over dn). This restores ||m|| to its pre-step value
# while PRESERVING the modulus-profile SHAPE that the rotation+scale produced
# (only the global radius is factored out — see §1.2).
a = a_raw / rho.unsqueeze(-1).clamp_min(eps)        # (B,M,dn)  renormalized nouns
return a, log_rho                                   # scale_delta is per-slot log-scale
```

All of this runs in fp32 under `_autocast_off` (the existing operator numerics gotcha; the
norms/log/div are bf16-unstable at large magnitude).

**Why ρ from the L2-of-moduli ratio, not per-block:** a per-block renormalization would erase
the *relative* modulus pattern across blocks — but that pattern IS the identity profile we
want to keep (`modulus_eff_rank` diagnostic). Extracting one scalar per slot factors out only
the uniform radius, leaving the inter-block shape intact. Justified in §1.2.

### Global scale as explicit world-state

The accumulator updates after each hop:

```
s_next = s + log_rho                                # (B,M)  log-domain accumulate
```

`s` is **concatenated into the readout input** so it is part of world-state and visible to
the InfoNCE anchor / decoder-conditioning geometry — irreversibility info is not lost. The
readout (`model.Readout`) pools over `a* (B,M,dn)`; with the budget on, the model forms an
**augmented slot** `a_aug = concat([a, s.unsqueeze(-1)], dim=-1)  (B,M,dn+1)` ONLY for the
readout/anchor path. The **decoder memory stays `a (B,M,dn)`** — the leakage invariant
(decoder sees only `a*`) is unchanged; `s` enters the *anchor/readout* geometry (which is
already a stop-grad-target contrastive head, not the decoder). A 1-line `Linear(dn+1 → dn)`
projection (`scale_readout_proj`, built only when `use_norm_budget`) maps `a_aug` back to `dn`
before the existing `Readout`, so the readout module is unchanged and the InfoNCE pooled
vector now carries the scale signal. This projection is the **only** new trainable parameter
the norm budget adds (`(dn+1)*dn`, e.g. 33*32 = 1056 for nano — counted in the budget; the
operator itself adds zero params).

### inverse_apply() — exact undo with tracked scale

`inverse_apply(a, v, theta_offset=None, *, norm_budget=False, scale_delta=None)`:

```
if not norm_budget:
    return <existing inverse_apply>                 # bitwise today's behavior

# Re-apply the tracked scale BEFORE inverting (undo the renormalization), then invert.
assert scale_delta is not None, "norm-budget inverse needs the stored per-slot log_rho"
rho   = scale_delta.exp().unsqueeze(-1)             # (B,M,1)
a_raw = a * rho                                     # restore the pre-renormalization radius
k     = <existing inverse_apply on a_raw with theta_offset>   # exact structural inverse
return k
```

Because the forward stored `log_rho` and the inverse multiplies it back, the round-trip
`inverse_apply(apply(k,v)..., scale_delta=log_rho) == k` is exact (to fp32 eps). This is the
invertibility-preservation guarantee, asserted by a test (§7-A) and exercised by the
retraction probe (§4). The black-box baseline has no inverse — that asymmetry IS the
experiment (§4, §1.4).

## 1.2 Conditioning H reads the RENORMALIZED moduli — decision + justification

**Decision: at hop `h`, `H` reads the modulus of the operator's INPUT at that hop, which is
the RENORMALIZED noun from hop `h-1` (`|a_{h-1}|`), exactly as the no-budget path reads
`|a_{h-1}|`.** I.e. the conditioner is fed the same tensor it is fed today (`_apply_action`
computes `conditioner(<its k arg>)`), and with the budget on, that `k` arg is the
renormalized output of the previous hop.

Justification:
1. **The renormalization preserves the modulus-profile SHAPE, only removes the global
   radius.** `H` is `Linear(nb → nb, bias=False)`, zero-init. Its job (the "adjective") is to
   make a verb's phase effect depend on the *relative* modulus pattern across blocks (what the
   object currently *is*), not on the absolute radius (how big it has gotten). Feeding `H` the
   renormalized moduli is therefore the *correct* conditioning input: it removes the
   confound of cumulative radius drift (which under a scaling chain would otherwise push `H`'s
   input off-distribution every hop) while keeping the identity-relevant shape. The design's
   own semantics (`jepa_v21_polar §3.3`: "H reads the *current* modulus") are honored — the
   "current modulus" is now the renormalized one, which is the *stable* identity signal.
2. **Interaction note (load-bearing):** with renormalization on, the modulus profile *shape*
   still varies hop-to-hop (a scaling verb changes inter-block ratios before the global radius
   is factored out), so `H` still receives a meaningfully varying, state-dependent input — the
   budget does NOT collapse `H` to a constant. Only the global scale is extracted; the shape
   that `H` keys on is intact. The `modulus_eff_rank` diagnostic (which already measures the
   modulus-profile rank on the *renormalized* nouns post-budget) confirms the shape space stays
   high-rank.
3. **Inverse exactness is independent of the H input choice.** The inverse re-applies the
   stored `log_rho` and uses the SAME `theta_offset` that the forward used (the offset is an
   explicit argument, never recomputed — the v2.1 contract). So feeding `H` the renormalized
   moduli does not threaten the round-trip; it only changes *which* phase offset is computed,
   and that offset is stored and replayed identically.

## 1.3 Unroll two-hop spec with the budget

`model.forward_unroll` already threads `k0 → a1 → a2`. With the budget on, it additionally
threads the scale accumulator and stores per-hop `scale_delta` for the inverse:

```
_, k0, _ = encoder(s0)                              # start nouns
s_acc = zeros(B, M)                                 # per-slot log-scale, scale=1.0 at start

# hop 1
v1 = posterior(s0, s1)
θoff_1 = H(|k0|)                                    # k0 is un-renormalized start (s_acc=0)
a1, log_rho_1 = operator.apply(k0, v1, theta_offset=θoff_1, norm_budget=True)
s_acc1 = s_acc + log_rho_1                          # (B,M)
logits_1 = decoder(a1, s1_ids, s1_pad)              # decoder memory = a1 (renormalized), NO scale
zhat_1 = predictor(readout(scale_readout_proj(concat[a1, s_acc1])))   # anchor sees scale

# hop 2
v2 = posterior(s1, s2)
θoff_2 = H(|a1|)                                    # a1 is the RENORMALIZED hop-1 output (§1.2)
a2, log_rho_2 = operator.apply(a1, v2, theta_offset=θoff_2, norm_budget=True)
s_acc2 = s_acc1 + log_rho_2
logits_2 = decoder(a2, s2_ids, s2_pad)
zhat_2 = predictor(readout(scale_readout_proj(concat[a2, s_acc2])))
```

Each hop dict (the existing `forward_unroll` per-hop return) gains two keys when the budget is
on: `scale_delta (B,M)` (this hop's `log_rho`) and `s_acc (B,M)` (the accumulated log-scale
*after* this hop). The single-hop `forward()` gets the same treatment: it returns `scale_delta`
and `s_acc` (= `scale_delta`, since `s` starts at 0) when `use_norm_budget`. Both are `None`
when the budget is off (back-compat; downstream guards on `is not None`). The retraction probe
(§4) consumes `scale_delta` per hop to invert exactly.

**`_apply_action` change (model.py, Task A):** it gains a `norm_budget` flag (read from
`self.use_norm_budget`) and returns `(a, scale_delta)` when on, `a` when off. Callers
(`forward`, `forward_unroll`, `rollout`, `step_latent`) are updated to unpack accordingly. The
`step_latent`/`undo_latent` pet-demo path threads `scale_delta` the same way `theta_offset` is
threaded today (returned by `step_latent`, passed back to `undo_latent`).

## 1.4 GatedMLPTransition handling of the flag — NO-OP + WARNING

`GatedMLPTransition` has no modulus/phase split and no scale notion. When `apply` is called
with `norm_budget=True`:
- It **accepts and ignores** the flag (signature parity, exactly like `theta_offset`), logs a
  **one-time warning** (`logging.getLogger(__name__).warning(...)`, guarded by a class-level
  `_warned_norm_budget` bool so it fires once per process, not per step), and returns the
  same `(a, scale_delta)` arity as the structured operator so `model._apply_action` stays
  branch-free: it returns `scale_delta = torch.zeros(B, M)` (log-scale 0 ⇒ scale 1.0, i.e. "no
  tracked scale"). The decoder/readout path then sees `s_acc` that stays 0 throughout — the
  black-box simply has no irreversibility scalar to contribute, which is the honest
  representation of "this transition does not expose a scale."
- `inverse_apply` still **raises** `NotImplementedError` regardless of `norm_budget` — the
  black-box has no inverse, and that asymmetry is the point of the retraction probe (§4): the
  structured operator can retract an event; the black-box raises. Documented in the docstring
  and asserted by a test (§7-A: `inverse_apply(..., norm_budget=True)` still raises).

So entity black-box configs set `operator_group="gated_mlp"` and `use_norm_budget=true`; the
flag is harmlessly ignored (warning once), `s_acc` stays 0, and the retraction probe records
the raise.

## 1.5 Config gating

```python
# ModelHParams (twm/jepa/__init__.py), Task A field spec → B applies to schema:
use_norm_budget: bool = False   # master switch; entity (structured) configs set true.
```

`use_norm_budget=False` ⇒ the operator's `apply` returns a bare tensor, `_apply_action`
returns a bare tensor, no `scale_readout_proj` is built, no `s_acc` is threaded — **bitwise
v3 GLUCOSE behavior**. The model `build_jepa_model_v2` reads
`getattr(m, "use_norm_budget", False)` and only constructs `scale_readout_proj` when true.

---

# 2. ENTITY DATA WIRING

## 2.1 Config data block

Entity configs point at the dedicated entity-world data + BPE:

```jsonc
"data": {
  "path": "data/entity_world/train.jsonl",
  "tokenizer": "data/entity_world/bpe_512.json",
  "vocab_size": 512,
  "max_text_tokens": 64,          // STAYS 64 (entity BPE: mean 26 tok/state, P95 39, 0% overflow)
  "mode": "triples",              // two-hop unroll over the leading (s0,s1,s2) of each chain
  "append_eos": true
}
```

Entity chains are length 4–8 (`chain_len_min=4`, `chain_len_max=8`); triples mode in
`data.py` already takes the **leading** triple `(chain[0], chain[1], chain[2])` and logs
`n_skipped` for chains shorter than 3 (none here). No `data.py` change is required — entity
data is just GLUCOSE-shaped `{"chain": [...]}` JSONL the existing loader reads. The eval suite
(§3) loads the labeled twins and the other splits directly (not through the train dataset).

## 2.2 The GPU job MUST regenerate data first

`data/entity_world/*.jsonl` and `bpe_512.json` are **seeded-generator output and are NOT
committed** (the 22MB+ train file would bloat the repo; the generator is byte-reproducible
from `CONFIG["seed"]=7`). Every entity training/eval job on the GPU server must regenerate
the data *before* training, after resetting the checkout to the pushed commit.

**Job command prefix** (prepend to the train command in the §5/§6 submit pattern):

```
cd ~/triples_world_model_Glucose && \
git fetch origin feature/glucose-converter && \
git reset --hard origin/feature/glucose-converter && \
uv run python scripts/generate_entity_world.py && \
uv run python scripts/build_entity_world_bpe.py && \
rm -rf <out_dir> && \
uv run python scripts/train_jepa_v2.py configs/jepa/<config>.json
```

`generate_entity_world.py` writes all 8 JSONL splits (+ `_labeled` twins), `manifest.json`,
and `coverage_report.json`; `build_entity_world_bpe.py` writes `bpe_512.json`. Both are
deterministic (seed 7), so re-running on the server reproduces the exact local data. The
generator exposes `apply_action` and `replay_chain` (used by the retraction probe and the
oracle backend, §4/§5).

---

# 3. EVAL SUITE — entity-world diagnostics

All additions are config-gated under a new `eval.entity_world` block; when absent or
`enabled=false` the v3 diagnostics run exactly as today (GLUCOSE behavior unchanged). These
are **Task B** (diagnostics + labeled-data loading + any train-script wiring).

## 3.0 Config schema (`EvalConfig`, Task B-owned)

```python
@dataclass
class EntityWorldEvalConfig:                # eval.entity_world
    enabled: bool = False
    labeled_dir: str = "data/entity_world"  # holds {split}_labeled.jsonl + manifest.json
    splits: list[str] = field(default_factory=lambda: ["test_iid","test_ood_near","test_ood_far"])
    subsample: int = 512                    # per-split cap for the ladder (§3b)
    n_rollout_chains: int = 128             # §3c rollout fidelity
    rollout_max_depth: int = 4              # depth 1..4 greedy decode vs gold
    action_recovery_split: str = "test_iid" # §3a NMI split (uses its _labeled twin)
# EvalConfig gains:
entity_world: EntityWorldEvalConfig = field(default_factory=EntityWorldEvalConfig)
```

`config.py::JEPAConfig.from_dict` parses `eval.entity_world` with `_only_known`
(EvalConfig nested-block parse, mirroring the loss nested blocks). The train script
(`maybe_eval_diagnostics`) passes `cfg.eval.entity_world` to a new
`diagnostics.eval_entity_world(...)` call, invoked the same epochs as `eval_diagnostics_v2`,
**only when `cfg.eval.entity_world.enabled`**. All three metric families below are written
into the returned flat dict (prefixed `ent_`) and into the per-epoch artifact JSON.

## 3.1 Labeled-data loader (Task B, in `diagnostics.py` or a small helper)

A `_load_labeled_split(labeled_dir, split, tokenizer, max_text_tokens) -> list[dict]` that
reads `{split}_labeled.jsonl`, each record `{"chain":[...], "actions":["<verb>@<idx>",...],
"types":[...]}`, and tokenizes each state to `(ids, pad)` with the SAME `_encode`/`append_eos`
logic the dataset uses (import the encode helper or replicate the 4-line BPE+eos+pad). Returns
per-chain dicts with tokenized states, the raw `actions` list, and `types`. Used by §3a and
§3c. `manifest.json` is loaded once (schema/profiles) for the oracle replay in §3c/§5.

## 3.2 (a) Action-recovery NMI

For the `action_recovery_split` labeled twin, for **each adjacent pair** `(state_t,
state_{t+1})` in every chain (this is the unit the posterior scores):
1. Run the posterior `q(v | s_t, s_{t+1})` → **hard argmax latent action** `v_hat ∈ {0..V-1}`
   (the `model.transition` head, `hard=True`, same call the forward uses).
2. Oracle label = the chain's `actions[i]`, a string `"<verb>@<entity_idx>"`. Build **two**
   label vectors:
   - `verb_only`: strip `@<entity_idx>` → the verb (`feed`, `play`, ...). This is the primary
     NMI (does the latent code recover *which action*?).
   - `verb_entity`: the full `"<verb>@<idx>"` (does it also recover *which entity moved*?).
3. Compute **NMI** between the latent-cluster assignment `v_hat` and each label vector
   (`sklearn.metrics.normalized_mutual_info_score`; if sklearn unavailable, a 12-line NMI
   from `np.histogram2d`-style joint counts — implement the fallback so the GPU env without
   sklearn still reports it).
4. **Shuffle baseline:** permute `v_hat` and recompute NMI; report `ent_action_nmi_shuffle`.
   The pre-registered bar is `ent_action_nmi_verb ≥ 0.2` AND comfortably above the shuffle
   baseline.

Reported scalars: `ent_action_nmi_verb`, `ent_action_nmi_verb_entity`,
`ent_action_nmi_shuffle`, `ent_action_nmi_verb_pass` (bool, ≥0.2), plus the cluster→verb
contingency saved to `action_nmi_contingency_epoch{N}.json`.

## 3.3 (b) OOD ladder — CE + hard-MRR + chrF per split, each diag epoch

For each split in `eval.entity_world.splits` (`test_iid`, `test_ood_near`, `test_ood_far`),
subsample `subsample` (512) **adjacent pairs** (deterministic: first 512 pairs flattened from
the split's chains) and compute, **reusing the existing v2 diagnostic machinery** on that
split's pairs:
- **CE**: teacher-forced `token_ce(logits, tgt_ids)` mean nats (the existing `token_ce`).
- **hard-MRR**: `_compute_retrieval_mrr(Zhat, Z, chain_ids, hard_nn_per_query=40)` — the
  existing hard-pool MRR (same-chain + NN distractors), run on this split's pooled
  anchor/target vectors. `chain_ids` = the split's per-pair originating chain index.
- **chrF**: greedy-decode each subsampled pair's `a*` and `_chrf(gen, gold)` mean.

Reported per split: `ent_{split}_ce`, `ent_{split}_hard_mrr`, `ent_{split}_chrf`. The
pre-registered `hard-MRR > 0.4 on test_iid` reads `ent_test_iid_hard_mrr`. The OOD-ladder
claim reads the ordering `ent_test_iid_* ≥ ent_test_ood_near_* > ent_test_ood_far_*`; a
convenience `ent_ladder_monotone_mrr` (bool) flags whether hard-MRR is monotone non-increasing
iid→near→far. Each split's generated samples table is saved to
`entity_samples_{split}_epoch{N}.json`.

## 3.4 (c) Rollout fidelity — depth 1..4, teacher-forced AND prior-sampled actions

For `n_rollout_chains` (128) chains from `test_iid` (use chains of length ≥ 5 so depth 4 has a
gold target; skip shorter, log the count):
1. **Encode `s0`** → `k0` (the start nouns).
2. **Two action sources, run separately:**
   - **teacher-forced (TF):** at each hop `h`, infer the action from the **pair posterior**
     `q(v | s_{h-1}, s_h)` using the gold intermediate states (the chain's actual states) —
     this measures "given the right actions, can the operator+decoder reproduce the chain?".
   - **prior-sampled (PR):** at each hop, sample the action from the **prior** `p(v | pooled
     current latent)` (the autonomous-rollout path, `model.rollout` / prior argmax) — this
     measures "does the model's own action prior drive a faithful rollout?". For PR, the
     current latent is the *rolled* state (apply operator stepwise), never the gold state.
3. **Apply operators stepwise** from `k0`, threading polar conditioning and (entity configs)
   the norm budget: `k_h = apply(k_{h-1}, v_h, theta_offset=H(|k_{h-1}|), norm_budget=...)`.
4. At each depth `d ∈ {1,2,3,4}` **greedy-decode** `decoder.generate(a_d)` → text, compute
   **exact-match** (string-equal to gold `chain[d]`, after the same special-token strip the
   v2 diagnostics use) and **chrF** vs gold.

Reported per depth and source: `ent_rollout_{tf,pr}_exact_d{1..4}`,
`ent_rollout_{tf,pr}_chrf_d{1..4}`. The pre-registered **rollout exact-match at depth 4**
reads `ent_rollout_tf_exact_d4` (primary) and `ent_rollout_pr_exact_d4` (autonomous). A
per-chain rollout transcript table (gold vs TF vs PR at each depth) saved to
`entity_rollout_epoch{N}.json`.

## 3.5 Train-script wiring (Task B, explicit)

`scripts/train_jepa_v2.py::maybe_eval_diagnostics` adds, after the existing
`eval_diagnostics_v2` call and only when `cfg.eval.entity_world.enabled`:
```python
if getattr(cfg.eval, "entity_world", None) and cfg.eval.entity_world.enabled:
    from twm.jepa.diagnostics import eval_entity_world
    ent_metrics = eval_entity_world(model, cfg.eval.entity_world, device, tokenizer,
                                    max_text_tokens=cfg.data.max_text_tokens,
                                    out_dir=cfg.eval.out_dir, epoch=epoch)
    metrics.update(ent_metrics)
```
This is the **only** train-script change for the eval suite. `eval_entity_world` lives in
`diagnostics.py` (Task B) and reuses the module-level helpers (`token_ce`, `_chrf`,
`_compute_retrieval_mrr`, `_decode_ids`). It is robust to a black-box model (the budget
no-op): rollout still runs (operator is the gated MLP); retraction is NOT part of this suite
(it is the standalone probe, §4).

---

# 4. RETRACTION PROBE — `scripts/probe_retraction.py` (standalone)

The headline engram-wm demonstration: **retract a past event** from a rolled latent state by
applying the structured inverse of that event (with its stored phase offset + scale delta),
and show it lands near the oracle replay *without* that event — something the black-box
literally cannot do (its `inverse_apply` raises). Standalone script, not part of the diag
suite. Owned by **Task C**.

## 4.1 Inputs and setup

```
uv run python scripts/probe_retraction.py \
    --ckpt results/jepa_ent_s0/model_latest.pt \
    --config configs/jepa/jepa_ent_s0.json \
    --labeled data/entity_world/test_iid_labeled.jsonl \
    --K 4 --n_chains 256 --retract_j 2 \
    --out results/jepa_ent_s0/retraction.json
```

Loads the model from the checkpoint + config (same `build_jepa_model_v2` path the trainer
uses), the labeled split, and the entity-world generator module
(`import scripts.generate_entity_world as ew`) for the oracle replay helper (`ew.apply_action`,
`ew.replay_chain`, `ew.TYPE_LIBRARY`). **Structured operator only** — if the loaded config has
`operator_group=="gated_mlp"`, the probe prints "black-box raises on inverse — that asymmetry
IS the experiment", attempts the first `inverse_apply` to capture the `NotImplementedError`,
records `{"backend":"blackbox","inverse_supported":false}`, and exits 0 (the negative result
is the data point).

## 4.2 Per-chain procedure (K-event chains, structured)

For each of `n_chains` chains with at least `K` events (`len(actions) ≥ K`; the first K events
define the rollable prefix):
1. **Reference (direct encode of `s_K`):** encode the chain's gold state at step `K`
   (`chain[K]`) → `z_ref = readout(scale_readout_proj(concat[k_K, s_acc=0]))` pooled latent
   (the canonical "what the final state encodes to"). (Pool with `s_acc=0` since this is a
   fresh encode, not a rollout.)
2. **Rolled state:** encode `s0` → `k0`, then apply the **posterior** actions 1..K stepwise
   (`v_h = q(s_{h-1}, s_h)` teacher-forced on gold intermediate states), threading
   `theta_offset_h = H(|k_{h-1}|)` and the norm budget. **Store per hop** `(v_h,
   theta_offset_h, scale_delta_h)`. Result: rolled nouns `k_K^roll` and accumulator `s_K^roll`.
3. **Retract event `j` (`--retract_j`, 1-based, `1 ≤ j ≤ K`):** apply the structured inverse
   of *only* event `j` to the rolled state, using the STORED offset and scale delta of hop `j`:
   ```
   k_retract = operator.inverse_apply(k_K^roll, v_j, theta_offset=theta_offset_j,
                                      norm_budget=True, scale_delta=scale_delta_j)
   s_retract = s_K^roll - scale_delta_j     # undo the accumulated log-scale of event j
   z_retract = readout(scale_readout_proj(concat[k_retract, s_retract]))
   ```
   (Retracting a *non-terminal* event `j<K` is the hard case — it commutes the inverse of `j`
   past the later events; the angle-additive + tracked-scale algebra makes this exact for the
   abelian operator. Report it; it is the interesting probe. `j=K` is the easy round-trip.)
4. **Oracle-without-j target:** use the generator to replay the chain **omitting event `j`**:
   `actions_minus_j = actions[:j-1] + actions[j:]` (drop the j-th action), then
   `snapshots = ew.replay_chain(types, initial_states, actions_minus_j)`; render the resulting
   K-1-event final state to text (the generator's `render_state`), tokenize, and encode →
   `z_oracle_minus_j`. (Initial states are recoverable from the chain's `s0` render via the
   manifest schema, OR — simpler and exact — add a tiny helper `ew.initial_states_from_chain`
   that the generator exposes: since generation is deterministic and `_labeled` carries
   `types`, re-seed is unnecessary; we reconstruct initial states by parsing `chain[0]`'s
   rendered values against the manifest ladders. **Task C adds `ew.parse_state(text, type)`**
   — a small inverse of `render_state` for the two salient attributes — OR, to avoid parsing
   fragility, the generator is extended to also emit `initial_states` in the `_labeled` records
   (preferred: a 1-line addition to `build_split`, regenerated by the GPU job). **Decision:
   emit `initial_states` in `_labeled` records** — deterministic, parse-free, regenerated by
   the seeded job; Task C adds the field to `generate_entity_world.py::build_split` and the
   probe reads it.)

## 4.3 Metrics and baselines

Compare `z_retract` against `z_oracle_minus_j` (the target — what the state *should* encode to
with event j removed):
- **cosine** and **MSE** of `z_retract` vs `z_oracle_minus_j` (primary: retraction should land
  *close*).
Against two baselines:
- **(a) do-nothing (un-retracted):** cosine/MSE of `z_K^roll` (the rolled state *with* event j)
  vs `z_oracle_minus_j`. Retraction must beat do-nothing (moving toward the oracle-without-j).
- **(b) ceiling (full re-encode):** cosine/MSE of `z_oracle_minus_j` vs itself is trivially 1/0;
  the meaningful ceiling is **`z_retract_via_full_reencode`** — re-roll the chain from scratch
  omitting event j (apply posterior actions for `actions_minus_j` stepwise) and encode that
  rolled state. This is the "best a rollout can do" upper bound; report cosine/MSE of
  `z_retract` against it too. The headline result: `cos(z_retract, z_oracle_minus_j)` should
  be **between do-nothing (worse) and full-re-encode ceiling (best)**, demonstrating the
  inverse genuinely removes the event.

Output JSON (`--out`): per-chain and aggregate `{retract_cos, retract_mse, donothing_cos,
donothing_mse, ceiling_cos, ceiling_mse, n_chains, K, retract_j, backend:"structured"}`, plus
`retract_beats_donothing` (bool: mean retract_cos > mean donothing_cos). Tests (§7-C): a 2-hop
exact round-trip (`j=K`) gives `retract_cos ≈ 1.0` on a tiny synthetic chain.

---

# 5. CALIBRATION — `scripts/calibrate_retrieval_ceiling.py`

Establishes the **retrieval ceiling**: the best achievable next-state retrieval if you had a
perfect oracle (entity-world) or a frontier LLM judge (GLUCOSE + entity hard pools). Owned by
**Task C**.

```
uv run python scripts/calibrate_retrieval_ceiling.py --backend oracle \
    --labeled data/entity_world/test_iid_labeled.jsonl --n 512 --out results/calib_oracle.json
uv run python scripts/calibrate_retrieval_ceiling.py --backend anthropic \
    --glucose data/glucose/chain_general_test.jsonl \
    --entity data/entity_world/test_iid_labeled.jsonl --n 200 --out results/calib_haiku.json
```

## 5.1 `--backend oracle` (entity-world; free, deterministic)

The oracle ceiling = **exact-match retrieval via oracle replay**. For each query pair
`(s_{t}, s_{t+1})` with its gold action label, the "retrieval" is: among the hard pool
(same-chain siblings + NN distractors built the same way `_compute_retrieval_mrr` does over
the *text*), the oracle ranks candidates by **whether `ew.apply_action(type, parse(s_t),
action)` exactly equals the candidate's rendered state**. Because the oracle deterministically
computes the true next state, the gold sits at rank 1 by construction ⇒ **MRR = 1.0** when the
pool contains the gold, modulo ties (distinct states never tie). This is the structural
ceiling: it confirms the hard pool is *solvable* (gold is uniquely identifiable from dynamics),
so a model's `hard_mrr < 1.0` is a model gap, not a pool defect. Reports `oracle_hard_mrr`
(≈1.0 expected), `pool_solvable_frac` (fraction of queries whose gold is uniquely oracle-rankable),
and flags any non-1.0 as a generator/pool bug. Runs free (no API), deterministic from seed 7.

## 5.2 `--backend anthropic` (LLM-judge ceiling; graceful skip)

Ranks the GLUCOSE and entity hard pools with **`claude-haiku-4-5`** via the official
**`anthropic` Python SDK**, IF `ANTHROPIC_API_KEY` is present; otherwise prints
`"ANTHROPIC_API_KEY not set — skipping anthropic backend (oracle backend is the free
ceiling)"` and exits 0 (graceful skip, never hard-fails a pipeline).

Minimal SDK usage (model id `claude-haiku-4-5`; Haiku does NOT support the `effort` param —
do not pass it):
```python
import os, anthropic
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ANTHROPIC_API_KEY not set — skipping anthropic backend"); return
client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY
def rank_pool(state_t: str, candidates: list[str]) -> int:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=16,
        messages=[{"role": "user", "content":
            f"Given the current state:\n{state_t}\n\n"
            f"Which numbered option is the most likely NEXT state? "
            f"Answer with only the number.\n{numbered}"}],
    )
    text = next((b.text for b in msg.content if b.type == "text"), "").strip()
    return int("".join(ch for ch in text if ch.isdigit()) or "-1")
```
For each of `--n` queries, build the hard pool (gold at index 0, shuffled into the candidate
list — track the shuffled gold position), ask Haiku to pick the next state, score
reciprocal-rank as 1.0 if Haiku's pick == gold position else 0 (LLM gives a single pick, so
RR is 1/1 or 0; report **judge accuracy** = fraction correct, the LLM-judge ceiling on hard
pools). Wrap each call in try/except (`anthropic.APIError`, rate-limit) with the SDK's built-in
retry (`anthropic.Anthropic(max_retries=4)`); on persistent failure, skip that query and log.
Output JSON: `{backend:"anthropic", glucose_judge_acc, entity_judge_acc, n, model:"claude-haiku-4-5"}`.

This gives the human-frontier ceiling to compare against the model's `hard_mrr` and the oracle
ceiling: oracle (≈1.0, structural) ≥ Haiku (frontier) ≥ trained JEPA (`ent_test_iid_hard_mrr`).

---

# 6. CONFIGS — `{jepa_ent, jepa_ent_blackbox} × {s0,s1,s2}` + smoke each

8 configs in `configs/jepa/`. All use `profile="jepa_v3"`, the entity data block (§2.1), the
v3 decoder block, InfoNCE+unroll (`w_nce=0.25`, `mode="triples"`, `hop_weights=[1.0,0.5]`),
and the `eval.entity_world` block (enabled). Structured family sets
`use_norm_budget=true`; black-box family sets `operator_group="gated_mlp"` +
`use_norm_budget=true` (no-op + warning, §1.4) + `use_polar_conditioning=false`.

| config | seed | operator_group | use_norm_budget | polar | out_dir |
|---|---:|---|---|---|---|
| `jepa_ent_s0.json` | 0 | rotation_scale | true | true | `results/jepa_ent_s0` |
| `jepa_ent_s1.json` | 1 | rotation_scale | true | true | `results/jepa_ent_s1` |
| `jepa_ent_s2.json` | 2 | rotation_scale | true | true | `results/jepa_ent_s2` |
| `jepa_ent_smoke.json` | 0 | rotation_scale | true | true | `results/jepa_ent_smoke` |
| `jepa_ent_blackbox_s0.json` | 0 | gated_mlp | true | false | `results/jepa_ent_blackbox_s0` |
| `jepa_ent_blackbox_s1.json` | 1 | gated_mlp | true | false | `results/jepa_ent_blackbox_s1` |
| `jepa_ent_blackbox_s2.json` | 2 | gated_mlp | true | false | `results/jepa_ent_blackbox_s2` |
| `jepa_ent_blackbox_smoke.json` | 0 | gated_mlp | true | false | `results/jepa_ent_blackbox_smoke` |

Common `eval` block:
```jsonc
"eval": {
  "every_epochs": 5, "n_examples": 512,
  "out_dir": "results/jepa_ent_s0", "n_text_samples": 16, "temperatures": [0.7, 1.0],
  "entity_world": {
    "enabled": true, "labeled_dir": "data/entity_world",
    "splits": ["test_iid","test_ood_near","test_ood_far"],
    "subsample": 512, "n_rollout_chains": 128, "rollout_max_depth": 4,
    "action_recovery_split": "test_iid"
  }
}
```
Smoke variants: `data.max_chains=64`, `optim.epochs=3`, `eval.every_epochs=1`,
`eval.entity_world.subsample=64`, `n_rollout_chains=16` — fast end-to-end CI check of both
operator families, triple mode, the norm budget on/off path, and the entity eval suite.
Black-box `model.gated_mlp={d_e:4,d_h:8}`.

---

# 7. WORK BREAKDOWN — disjoint

Strict file ownership, no overlaps. Each task ships its own tests. Train-script changes are
assigned to **B** explicitly.

### Task A — operator.py / model.py norm budget + tests
**Files:** `src/twm/jepa/operator.py`, `src/twm/jepa/model.py`, `src/twm/jepa/__init__.py`
(the one `use_norm_budget` ModelHParams field — A owns this dataclass field; coordinate with B
which owns the schema file, same one-directional handoff as v3), `tests/jepa/test_operator.py`
+ `tests/jepa/test_model.py` (extend).
- `RotationScaleOperator.apply/inverse_apply`: add `norm_budget` kwarg; on=`(a, log_rho)`
  return + renormalize (§1.1), off=bitwise-today. `_gather_*` unchanged.
- `model.py`: `_apply_action` gains `norm_budget` (reads `self.use_norm_budget`), returns
  `(a, scale_delta)` when on; thread `s_acc` in `forward`/`forward_unroll`/`rollout`/
  `step_latent`/`undo_latent`; build `scale_readout_proj (Linear(dn+1→dn))` only when
  `use_norm_budget`; feed `concat[a, s_acc]` through it before `readout` for the anchor path;
  decoder memory stays `a` (leakage unchanged). `build_jepa_model_v2` reads
  `getattr(m,"use_norm_budget",False)`.
- **GatedMLPTransition** (lives in `baseline_transition.py`, but the no-op+warning is part of
  the *operator interface contract* — Task A edits `baseline_transition.py` too): `apply`
  accepts+ignores `norm_budget`, warns once, returns `(a, zeros(B,M))`; `inverse_apply` still
  raises regardless of `norm_budget`.
- Tests: exact round-trip `inverse_apply(apply(k,v,norm_budget=True), scale_delta=log_rho)≈k`;
  renormalized `||m||` == pre-`||m||`; modulus-profile shape preserved (per-block ratios
  unchanged up to global scale); `use_norm_budget=False` bitwise-identical to today (regression
  vs stored baseline); `forward_unroll` returns `scale_delta`/`s_acc` keys on, `None` off;
  gated-mlp `norm_budget=True` warns once + returns zeros scale + `inverse_apply` raises;
  decoder memory unaffected by scale (perturb `s_acc` → decoder logits unchanged).

### Task B — diagnostics eval suite + labeled loading + configs + train-script wiring + tests
**Files:** `src/twm/jepa/diagnostics.py` (add `eval_entity_world` + `_load_labeled_split`),
`src/twm/jepa/__init__.py` + `src/twm/jepa/config.py` (the `EntityWorldEvalConfig` schema +
`EvalConfig.entity_world` + parse — B owns the schema files), `scripts/train_jepa_v2.py` (the
`maybe_eval_diagnostics` entity hook, §3.5 — **train-script change assigned to B**), the 8
config JSONs (§6), `tests/jepa/test_diagnostics.py` + `tests/jepa/test_config.py` (extend).
- §3a NMI, §3b OOD ladder, §3c rollout fidelity (TF + PR), all under `eval.entity_world`.
- Labeled loader + manifest load; sklearn-optional NMI fallback.
- Tests: `_load_labeled_split` shapes/keys; NMI on a synthetic perfect-recovery case == 1.0
  and shuffle ≈ 0; ladder returns the 9 scalars; rollout returns depth-1..4 exact/chrF for
  both sources on a tiny labeled fixture; `eval.entity_world.enabled=false` ⇒ no entity
  metrics (back-compat); the 8 configs parse and build (incl. `use_norm_budget`/gated_mlp).

### Task C — probe_retraction.py + calibrate_retrieval_ceiling.py + generator helper + tests
**Files:** `scripts/probe_retraction.py` (new), `scripts/calibrate_retrieval_ceiling.py`
(new), `scripts/generate_entity_world.py` (the `initial_states` field in `_labeled` records,
§4.2 — C owns this generator edit), `tests/jepa/test_probe_retraction.py` +
`tests/jepa/test_calibrate.py` (new).
- §4 retraction probe (structured; black-box records the raise). §5 oracle + anthropic
  backends (anthropic gated on `ANTHROPIC_API_KEY`, SDK `claude-haiku-4-5`, no `effort`).
- Generator: add `initial_states` to `_labeled` records (1-line in `build_split`), regenerated
  by the seeded GPU job (§2.2) — deterministic, parse-free retraction targets.
- Tests: retraction `j=K` exact round-trip `retract_cos≈1.0` on a synthetic 2-hop chain;
  retract beats do-nothing on a constructed case; black-box backend records `inverse_supported:
  false` and exits 0; oracle calibration on a tiny labeled fixture gives `oracle_hard_mrr≈1.0`;
  anthropic backend with no key prints the skip message and returns cleanly (monkeypatch env).

**Disjoint-file guarantee:** A=`operator.py`,`baseline_transition.py`,`model.py`(+tests),
plus the single `use_norm_budget` field; B=`diagnostics.py`,`train_jepa_v2.py`,the schema
files (`__init__.py`/`config.py`),the 8 configs(+tests); C=the two new scripts +
`generate_entity_world.py` generator edit(+tests). `model.py`→A (norm budget is operator/model
algebra), `train_jepa_v2.py`→B (eval orchestration), the two new scripts→C. The schema files
(`__init__.py`/`config.py`) are B-owned; A delivers the `use_norm_budget` field spec as a diff
B applies (one-directional handoff, same as v3).

---

## Summary of resolved decisions
1. **Norm budget:** renormalize each slot's modulus profile to its pre-step norm; track the
   extracted per-slot **log-scale** `s_i` as explicit world-state; `apply→(a, log_rho)`;
   `inverse_apply` re-applies `log_rho` for an **exact** inverse. **H reads the renormalized
   moduli** (shape preserved, only global radius removed — the correct state-dependent input;
   inverse exactness independent of this choice). Scale enters the **anchor/readout** path
   (decoder leakage unchanged). Config-gated `use_norm_budget`, **default false** (bitwise v3
   GLUCOSE when off). GatedMLPTransition: **no-op + one-time warning**, returns zeros scale,
   `inverse_apply` still raises (the asymmetry is the experiment).
2. **Entity wiring:** configs point at `data/entity_world/{train.jsonl,bpe_512.json}`;
   `max_text_tokens` stays **64**; the GPU job **regenerates** data first
   (`generate_entity_world.py` + `build_entity_world_bpe.py`, seeded, before training).
3. **Eval suite** (under `eval.entity_world`): (a) action-recovery **NMI** (verb + verb@entity,
   shuffle baseline, ≥0.2 bar); (b) **OOD ladder** CE+hard-MRR+chrF on iid/near/far (512 each);
   (c) **rollout fidelity** depth 1..4, **teacher-forced AND prior-sampled** actions, exact+chrF.
   Train-script hook added in B's `maybe_eval_diagnostics`.
4. **Retraction probe** (`scripts/probe_retraction.py`): roll K events, retract event `j` via
   structured inverse with stored offsets+scale deltas, compare vs **oracle-replay-without-j**;
   baselines do-nothing + full-re-encode ceiling; black-box raises on inverse (recorded). Oracle
   target via generator `replay_chain` + emitted `initial_states`.
5. **Calibration** (`scripts/calibrate_retrieval_ceiling.py`): `--backend oracle` (free,
   exact-match retrieval ceiling ≈1.0 via oracle replay) and `--backend anthropic`
   (`claude-haiku-4-5` SDK ranker, **graceful skip** without `ANTHROPIC_API_KEY`, no `effort`).
6. **Configs:** `{jepa_ent, jepa_ent_blackbox}×{s0,s1,s2}` + smoke each (8), entity BPE,
   `use_norm_budget=true`, `eval.entity_world` block.
7. **Work breakdown:** A=norm budget in `operator.py`/`baseline_transition.py`/`model.py`
   +tests; B=`diagnostics.py` eval suite + labeled loading + schema + 8 configs + **the
   train-script hook** +tests; C=`probe_retraction.py` + `calibrate_retrieval_ceiling.py` +
   generator `initial_states` edit +tests. No file overlaps.
