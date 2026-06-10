# JEPA v3 — InfoNCE Next-State + Multi-Step Unroll + Black-Box Transition Baseline

Status: **design, fully resolved.** Every decision a builder needs is fixed here; there
are no open choices. Builds on `research/jepa_v2_latent_actions.md` (v2.0),
`research/jepa_v21_polar.md` (v2.1 polar, the post-refactor live path), and the
matrix verdicts in `research/jepa_matrix_synthesis.md` (the empirical basis for the v3
recipe). v2.1 polar conditioning is **kept on** (free, non-harmful per the matrix); the
two capability levers are the InfoNCE term and multi-step unroll.

The in-flight/legacy v2 interface (`scripts/train_jepa_v2.py`,
`configs/jepa_nano_v2*.json`, `from twm.jepa import JEPALossV2`) stays valid. v3 is
**config-gated and default-off**: an unmodified v2 config builds and trains an identical
v2.1 model with InfoNCE off, single-hop, no baseline. This is a hard requirement (the
§R behavior-preservation gate of v2.1 still applies and is re-asserted here).

---

## 0. The v3 recipe (matrix-decided, one place)

From `jepa_matrix_synthesis.md §6`:

1. **Decoder `d_dec=128 / n_layers=2` is the standard profile.** It is the only change
   that bought a real metric (ce_true 1.39 → 0.95 nats) and it fixes the BPE
   degeneration. Non-negotiable. (≈ 657K params total; the `jepa_small_v21_dec` arm.)
2. **InfoNCE next-state contrastive REPLACES the vestigial EMA-aux `L_pred`.** The
   retrieval gap (hard_mrr ≈ chance) is the core failure; teacher-forced CE alone never
   pressures the slots to *discriminate* the right next-state. The slot-LOO flip
   (probe2: all 8 slots constructive in v2) says the slots *can* carry it; InfoNCE gives
   the gradient. Config-gated so the old `L_pred` behavior is recoverable (§1.7).
3. **Multi-step unroll** on chain triples `(t, t+1, t+2)` puts real pressure on action
   composition — single-step lets the operator stay near-abelian and decorative (§2).
4. **Polar conditioning kept** (free; matrix §2). **`d_noun` stays 32** (dn64 bought
   modulus rank and nothing downstream; matrix §"dn64").
5. **Every config ships seed-0/1/2 variants.** The 63% seed spread on `ce_gap`
   (matrix §2) makes any single-seed claim null.
6. **ALSO build the black-box `GatedMLPTransition` baseline** for the engram-wm program:
   the operator interface, but a verb-embedding-gated MLP with **no inverse** (§4).
   Selectable via `operator_group="gated_mlp"`, trains through the identical pipeline.

---

# 1. InfoNCE next-state contrastive (`L_nce`)

## 1.1 What it replaces and why (weight justification)

v2's `L_pred = MSE(zhat, sg(z_target))` is an EMA-aux objective with `w_pred=0.25`. The
matrix shows it is **vestigial**: it moved no headline metric, and an MSE-to-a-moving-
target on a *pooled* readout exerts no *discriminative* pressure (it pulls `zhat` toward
the single correct `z_target` but never pushes it away from plausible wrong next-states —
the exact failure mode of hard_mrr ≈ chance). InfoNCE is the same prediction head
turned into a **discriminative** loss: it pulls toward the true next-state *and* pushes
away from negatives, which is precisely the retrieval objective the matrix says is
missing.

**Weight: `w_nce = 0.25`, taking over `w_pred`'s slot in the budget.** Justification:
(a) it occupies the same architectural slot (the readout→predict head over `a*` vs the
EMA target), so swapping at equal weight is the minimal, comparable change;
(b) `L_token` (w=1.0) must remain the dominant grounding signal — InfoNCE is a
*regularizer that shapes the readout geometry*, not a co-equal objective; at 0.25 it is
strong enough to move hard_mrr (the InfoNCE-ablation test) without destabilizing the CE
that fixes generation fluency; (c) keeping the total auxiliary budget unchanged
(0.25) preserves the v2.1 loss-scale balance so the existing LR/warmup schedule carries
over without re-tuning. **`w_pred` and `w_nce` are mutually exclusive on the live path**
(§1.7): the default v3 config sets `w_pred=0.0, w_nce=0.25`; setting `w_nce=0.0,
w_pred=0.25` recovers exact v2.1 behavior.

## 1.2 Which representation pair (pooled vs slot-set — DECIDED: pooled readout)

**Decision: pooled readout, reusing the existing `Readout` + `Predictor` + EMA head.**

- **Anchor (query):** `zhat = Predictor(Readout(a*))  ∈ ℝ^(B,dn)` — the model's predicted
  next-state pool over the operator-transformed slots `a*`. This is the *exact* tensor
  already computed in `model.forward()` (`out["zhat"]`) and already used as the retrieval
  query `Zhat` in `diagnostics._compute_retrieval_mrr`. **No new head.**
- **Positive (key):** `z_target = EMA.pool_raw(tgt_ids, tgt_pad)  ∈ ℝ^(B,dn)` — the
  stop-grad EMA raw-noun readout pool of the *true* next state `text_{t+1}`. Also already
  computed (`out["z_target"]`) and already the retrieval gallery `Z` in diagnostics.

Reasoning for pooled over slot-set:

1. **It makes `L_nce` measure the exact quantity diagnostics scores.** hard_mrr in
   `diagnostics._compute_retrieval_mrr` is decoder-likelihood/`Zhat`-vs-`Z` retrieval
   over these pooled vectors. Training the same pooled pair *is* the direct
   intervention on the metric we want to move (the matrix's "give them the gradient").
2. **Slot-set contrastive needs an alignment/matching step** (the M slots are a
   permutation-invariant set; a set-to-set InfoNCE requires Hungarian or Sinkhorn
   matching or a symmetric set kernel) — added complexity and a new failure surface,
   with no evidence it helps. The slot-LOO flip already confirms slot *content* is
   informative; we don't need set-level contrast to exploit it, only a readout that
   discriminates, which the pooled head provides.
3. **Zero new parameters.** The `Readout`/`Predictor`/EMA bundle is already built and is
   the natural anchor encoder. InfoNCE reuses it; the EMA target side stays stop-grad
   exactly as in `L_pred`, so the representation is consistent train↔eval.

The EMA target side keeps its `@torch.no_grad()` / stop-grad (the key encoder is the
slow EMA copy — standard MoCo/BYOL-style asymmetry, already in `model.forward`'s
`with torch.no_grad(): z = self.ema.pool_raw(...)`). InfoNCE therefore inherits v2.1's
collapse-resistance (the EMA target cannot follow a degenerate anchor).

## 1.3 The loss

Standard InfoNCE with cosine similarity and temperature `τ_nce`:

```
zhat  : (B, dn)   anchor (gradient)         = Predictor(Readout(a*))
ztgt  : (B, dn)   positive key (stop-grad)  = sg(EMA.pool_raw(text_{t+1}))

qn = L2_normalize(zhat, dim=-1)        # (B, dn)
kn = L2_normalize(ztgt, dim=-1)        # (B, dn)   (already detached upstream)

logits = (qn @ kn.T) / τ_nce           # (B, B) similarity to every key in the pool
                                        #   + extra same-chain negative columns (§1.4)
labels = arange(B)                      # row i's positive is key i (the diagonal)
L_nce  = cross_entropy(logits, labels)  # InfoNCE = softmax-CE over the gallery
```

- **Temperature `τ_nce = 0.1`** (fixed; the SimCLR/MoCo default that works across
  scales). Not annealed — distinct from the Gumbel posterior `τ`; do **not** reuse the
  posterior temperature. A separate `loss.nce.temperature` config field.
- Cosine (L2-normalized) similarity, not raw dot — keeps the loss scale-invariant to the
  readout magnitude (which SIGReg/standardization already moves around).
- The positive key is the **stop-grad EMA** vector; only the anchor receives gradient.
  This is the same asymmetry as `L_pred`, so it cannot drive representational collapse
  (BYOL/MoCo result).

## 1.4 Negatives: same-chain + in-batch

Negatives are the union of:

1. **In-batch negatives** — every other key `kn[j], j≠i` in the batch (the standard
   off-diagonal of the `(B,B)` similarity matrix). Free.
2. **Same-chain negatives** — the *other* adjacent next-states from the same chain as
   row `i`. `chain_ids` already exists on `JEPAChainDataset` (v2.1 §8.2, present in
   `data.py`). The trainer passes the batch's `chain_ids` to the loss; for each anchor
   `i` we append, as extra negative **columns**, the EMA keys of same-chain siblings that
   are *not* `i`'s own positive. Concretely: a length-3 chain yields pairs
   `(t0→t1)` and `(t1→t2)`; for the `(t0→t1)` anchor, the `t2` key (the sibling pair's
   positive) is a *hard* negative — it is on the same narrative but is the wrong target.
   This is exactly the "hard pool" `diagnostics._compute_retrieval_mrr` builds via
   `same_chain[i]`, now used as a *training* signal, not just an eval probe.

Implementation: the loss receives `chain_ids: (B,)` long for the batch. It builds a
`(B, B)` same-chain mask `M[i,j] = (chain_ids[i]==chain_ids[j]) and (i≠j)`. Same-chain
off-diagonal entries are already columns of the in-batch `(B,B)` logits — so when two
siblings of one chain land in the same batch, the in-batch matrix *already* contains the
hard negative; no extra columns are needed for that case. To **guarantee** at least one
hard negative per anchor regardless of batch composition, v3 uses **chain-contiguous
batching** (§1.5): each batch is assembled so siblings co-occur. The mask is then only
used to (a) confirm hard negatives are present (a logged diagnostic
`frac_anchors_with_hard_neg`) and (b) **exclude the anchor's own positive** from the
negative set if a chain ever contributes the same target twice (defensive; the
`(i≠j)` and diagonal-is-positive structure already handles the common case).

There is **no separate "extra negative column" tensor** in the common path — the design
keeps InfoNCE to the clean `(B,B)` matrix and *guarantees hard negatives by batching*.
This is simpler than gathering a ragged same-chain pool and is numerically identical.

## 1.5 Chain-contiguous batching (the data-side enabler)

The current trainer does `perm = torch.randperm(n_train)` — fully shuffled, so a chain's
two sibling pairs almost never share a batch, and in-batch hard negatives are absent.
v3 adds **chain-grouped shuffling**: shuffle at the *chain* level, then flatten, so the
two pairs of each chain are adjacent in the permutation and (with `batch_size` a multiple
of the chain's pair-count, which it is: 64 % 2 == 0) co-occur in the same batch. Spec in
§6 Task B (train-script change). This is the cheapest way to make same-chain negatives
real and is required for `L_nce` to have hard negatives. It does **not** change the loss
math — it changes which negatives populate the in-batch matrix.

## 1.6 Leakage analysis (the load-bearing audit)

The v2 leakage invariant (`jepa_v2_latent_actions.md §6`): the **decoder** may see only
`a* = B_v k` as memory; the only `text_{t+1}` path into the decoder is the discrete `v`
(⌈log₂V⌉ bits). InfoNCE must not create a new path from the future into the decoder.

**Audit:**

1. **The InfoNCE positive key (`z_target`) reads `text_{t+1}`** — but it always has, via
   the v2 `L_pred` EMA target, and it is **stop-grad and never touches the decoder**. The
   key encoder is the EMA copy; its output feeds only the contrastive softmax, never the
   `a*` memory. No new decoder-side leakage.
2. **The anchor (`zhat`) is a function of `a*` only** — `a* = B_v k`, `k = f(text_t)`.
   InfoNCE gradient flows anchor → `Readout` → `Predictor` → `a*` → operator/`k`/
   posterior-`v`. The *only* `text_{t+1}` dependence in that path is, as before, the
   discrete `v` from the posterior. InfoNCE adds **no** continuous `text_{t+1}` channel
   into `a*`.
3. **The same-chain negatives are keys, not memory.** They are EMA pools of *other*
   states; they appear only as negative columns in the contrastive softmax. A negative
   cannot inject information into the decoder — it can only push the *anchor's* readout
   away, which shapes `a*`'s geometry, not its information content from the future. The
   gradient from a negative key flows into the *anchor* (and thence to `v`/`k`), never
   into the decoder memory by a non-`v` route.
4. **The negative-sampling path must not leak future text into the decoder.** It does
   not: negatives are stop-grad EMA vectors entering a similarity matrix. The decoder's
   `forward(a*, tgt_ids, tgt_pad)` signature is unchanged and still has no posterior /
   future-encoding argument. **Verified at the signature level** (decoder API untouched
   by v3) and asserted by a leakage test (§6 Task A): the InfoNCE term, given a decoder
   stubbed to record its memory argument, must never receive anything but `a*`.

**Conclusion:** InfoNCE is decoder-leakage-neutral. It reuses the already-audited EMA
target path (stop-grad, decoder-isolated) and adds only a contrastive softmax over pooled
readouts. The 3-bit bottleneck on the decoder is preserved exactly.

## 1.7 Config gating (recoverable old behavior)

```jsonc
"loss": {
  "w_pred": 0.0,          // v3 default: L_pred OFF (replaced by InfoNCE)
  "w_nce":  0.25,         // v3 default: InfoNCE ON, taking w_pred's slot
  "nce": { "temperature": 0.1 }
}
```

- `w_nce > 0` ⟹ InfoNCE active; `w_pred > 0` ⟹ EMA-MSE active. Both can be nonzero
  (sum into the total) but the **default v3 recipe is mutually exclusive**: exactly one
  of the two aux terms is on. Setting `w_nce=0.0, w_pred=0.25` reproduces v2.1 exactly
  (the recoverability requirement). When `w_nce=0`, `L_nce` is **not computed** (skip the
  similarity matmul) so v2.1 configs pay zero cost.
- The EMA head, `Readout`, `Predictor` are built whenever `use_pred or use_nce` (i.e.
  `w_pred>0 or w_nce>0`); the v3 config keeps them built (they are the anchor encoder).

---

# 2. Multi-step unroll (predict t+1 AND t+2)

## 2.1 Dataset triple mode (`data.py` spec)

`JEPAChainDataset` currently flattens each length-3 chain into two *adjacent pairs*. v3
adds an **optional triple mode** that instead emits one example per chain holding all
three states, so the trainer can unroll two hops from the same start.

**Spec (`data.py`):**

- New constructor arg `mode: str = "pairs"` (default = exactly today's behavior) with
  the alternative `"triples"`. Config field `data.mode`.
- In `"triples"` mode, for each chain `[s0, s1, s2]` emit **one** example with tokenized
  tensors for all three states:
  `s0_ids/s0_pad` (start), `s1_ids/s1_pad` (hop-1 target), `s2_ids/s2_pad` (hop-2 target).
  Reuse the existing `_insert_eos` / pad-mask logic per state. Chains of length < 3 are
  **skipped** in triple mode (GLUCOSE `chain_general` is uniformly length 3, so none are
  dropped; assert and log the count).
- `__getitem__`/`get_batch` in triple mode return keys
  `{"s0_ids","s0_pad","s1_ids","s1_pad","s2_ids","s2_pad","chain_id"}`.
- `chain_ids` semantics in triple mode: one id per chain (==the example index's chain),
  so same-chain negatives in InfoNCE still work — but with one example per chain the
  hard-negative now comes from **the cross-hop targets within the same example** (s1 vs
  s2 are both "futures of s0" — s2 is a hard negative for the hop-1 anchor and vice
  versa). The trainer builds these per-example hard negatives directly (§2.4), so triple
  mode does not depend on batch composition for hard negatives.
- **Back-compat:** `"pairs"` mode is untouched (same tensors, same keys, same
  `chain_ids` from v2.1 §8.2). The triple-mode tensors are additive; the existing
  `_src_ids/_tgt_ids` attributes are populated **only** in pairs mode, and the new
  `_s0/_s1/_s2` attributes only in triples mode. `_get_batch` in the train script gains a
  triple-aware branch (§6 Task B).

## 2.2 Per-hop posterior actions

Each hop gets its **own** posterior action from the existing `TransitionEncoder`:

- **Hop 1:** `v1 = posterior(s0, s1)` — the action carrying `s0 → s1`.
- **Hop 2:** `v2 = posterior(s1, s2)` — the action carrying `s1 → s2`.

Both use the same `TransitionEncoder` (shared trunk, no new params), called twice with
different `(src, tgt)` pairs. Each emits a hard ST one-hot `(B,V)` plus logits, exactly
as in v2. The prior is distilled per hop (`p1 = prior(pool_s0)`, `p2 = prior(pool_s1)`),
and `L_prior` is summed over both hops (each hop's `KL(sg q_h ‖ p_h)`).

## 2.3 Composed application via the angle-additive path

The composed two-hop latent is built by applying the operator twice, threading the polar
conditioning per hop:

```
k0          = encoder(s0).k                        # (B,M,dn)  start nouns
# --- hop 1 ---
θoff_1      = H(|k0|)                               # conditioning reads hop-0 modulus
a1          = operator.apply(k0, v1, theta_offset=θoff_1)   # = B_{v1} k0
logits_1    = decoder(a1, s1_ids, s1_pad)          # token CE at hop 1
# --- hop 2 ---
θoff_2      = H(|a1|)                               # conditioning reads hop-1 modulus (post-step)
a2          = operator.apply(a1, v2, theta_offset=θoff_2)   # = B_{v2}(B_{v1} k0)
logits_2    = decoder(a2, s2_ids, s2_pad)          # token CE at hop 2
```

This is exactly the existing angle-additive composition (`jepa_v21_polar.md §3.3`): two
diagonal complex maps compose as `r_{v2} r_{v1} · e^{i(θ_{v2}+θ_{v1}+offsets)}`. No new
operator arithmetic — just a second `operator.apply` on `a1`.

**H conditioning: reads hop-1 moduli (post-step `|a1|`) for hop 2 — JUSTIFIED.** The H
map is the "adjective" that makes a verb's effect depend on *what the object currently
is*. At hop 2 the object's identity has been (possibly) updated by hop 1, so the
conditioning input must be the **current** modulus `|a1|`, not the stale `|k0|`. This is
the design's own stated semantics (`jepa_v21_polar.md §3.3`: "H reads the *current*
modulus … under an r≠1 verb the modulus changes and the next step's offset shifts
accordingly — this is the *correct* state-dependent behavior, not a bug"). For a
pure-rotation hop-1 verb, `|a1| = |k0|` exactly, so the choice is moot; under a scaling
hop-1 verb, reading `|a1|` is the only correct option. **So: hop-`h` conditioning reads
the modulus of the operator's input at hop `h`** (`|k0|` for hop 1, `|a1|` for hop 2).

The conditioner is reused as-is — `model._apply_action` already computes
`theta_offset = self.conditioner(k)` from whatever noun tensor it is handed, so passing
`a1` at hop 2 needs no conditioner change, only the unroll loop calling it with `a1`.

## 2.4 Token CE at both hops

```
L_token = 1.0 · CE(logits_1, s1_ids) + 0.5 · CE(logits_2, s2_ids)
```

Per-hop weights **1.0 / 0.5** (hop-2 downweighted): hop 2 is strictly harder (compounded
action + the decoder must render a state two steps out), and full weight would let the
harder, noisier hop-2 gradient dominate the cleaner hop-1 signal that built v2's fluency.
0.5 keeps hop-2 as a *composition pressure* without destabilizing hop-1 grounding. The
weights are config fields (`loss.unroll.hop_weights = [1.0, 0.5]`) so they can be swept.

InfoNCE in triple mode runs at **both hops** with the same hop weights: anchor
`zhat_h = Predictor(Readout(a_h))`, positive key `z_h = sg(EMA.pool_raw(s_h))`, and the
**cross-hop hard negative is built per example**: for the hop-1 anchor, `z_2` (the EMA
pool of `s2`) is a same-chain negative column, and for the hop-2 anchor, `z_1` is. These
are added to the in-batch `(B,B)` matrix as the guaranteed hard negatives (replacing the
batch-composition dependence of §1.5 in triple mode — the hard negative is always present
because both targets live in the same example). `L_nce = 1.0·L_nce^(1) + 0.5·L_nce^(2)`.

## 2.5 Leakage audit extension (hop-2 text reaches the decoder only through v2's bits)

The hop-2 decoder call is `decoder(a2, s2_ids, s2_pad)`. We must show `s2`'s content
reaches the decoder *only* through `v2`'s discrete bits.

- `a2 = B_{v2}(B_{v1} k0)`. `k0 = f(s0)` (start only). `v1 = posterior(s0, s1)` —
  carries `s1` info as ⌈log₂V⌉ bits. `v2 = posterior(s1, s2)` — carries `s2` info as
  ⌈log₂V⌉ bits. The conditioning offsets `θoff_1 = H(|k0|)`, `θoff_2 = H(|a1|)` are
  functions of `k0` and `v1` only (no `s2`).
- Therefore the **only** path from `s2` into `a2` (the hop-2 decoder memory) is the
  discrete `v2`. The hop-2 decoder's teacher-forced context is `s2_ids` (the standard AR
  target — allowed, identical to v2's single-hop teacher forcing). The InfoNCE hop-2
  positive/negative keys are stop-grad EMA pools that never enter the decoder.
- **Net bound:** hop-2's continuous conditioning carries `s0`-info (`k0`) + `s1`-info
  (`v1`, 3 bits) and only `v2`'s 3 bits of `s2`. The composition is forced to *route*
  `s2`'s causal step through a discrete action — exactly the v2 bottleneck, now applied
  to a second hop. **Asserted by a leakage test** (§6 Task B/A boundary — the unroll loop
  is Task B, the contrastive leakage assertion Task A): the hop-2 decoder memory must be
  a pure function of `(k0, v1, v2)` with no continuous `s2` channel.

---

# 3. Profile table — `jepa_v3` is the standard

`jepa_v3` = nano encoder/operator (unchanged shapes) + the **d128/2L decoder** (the matrix
winner) + polar conditioning on. `d_noun=32`, `n_verbs=8`, `n_slots=8`, `d_model=64`.

## 3.1 Profile

```python
"jepa_v3": {
    "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
    "block": 2, "n_text_layers": 2, "tie_text_layers": True,
    "n_heads": 4, "n_slot_iters": 3,
    "operator_group": "rotation_scale", "n_steps_T": 1,
    "vocab_size": 512, "max_text_tokens": 64,
},
```

The decoder size (`d_dec=128, n_layers=2`), polar flag, mode, and loss weights come from
the JSON `model`/`loss`/`data` blocks (the profile stays minimal, per v2 §10). The
`jepa_v3_baseline` family reuses this profile and only flips `operator_group="gated_mlp"`.

## 3.2 Param arithmetic (≈ 660K, matching the matrix `dec` arm)

From `jepa_matrix_synthesis.md`: the `jepa_small_v21_dec` arm (d_dec128/2L) is **657,304**
params. v3 adds only the polar `H` (`nb² = 16² = 256`, already in v2.1 → already counted
in the 657K-with-polar lineage) and **zero** new trainable params for InfoNCE (reuses the
Readout/Predictor/EMA head that `L_pred` already built). So:

| Component | params | note |
|---|---:|---|
| Encoder trunk + heads (nano) | ~as v2.1 nano | unchanged |
| Operator `RotationScaleOperator` (θ, log_r) | `2·V·nb = 2·8·16 = 256` | unchanged |
| Transition + Prior heads | ~as v2 | unchanged |
| **TokenDecoder d_dec=128 / 2L (+ own token_emb)** | dominant term | the matrix `dec` arm |
| Readout + Predictor + EMA (anchor head) | ~as v2.1 | reused by InfoNCE, **+0** |
| Polar `H` `Linear(16→16, bias=False)` | `256` | v2.1, kept |
| **InfoNCE** | **0** | no new head |
| **Total (online, non-embedding)** | **≈ 657K** | == matrix `jepa_small_v21_dec` |

The `jepa_v3_baseline` swaps the operator (256 params) for `GatedMLPTransition`
(param-matched within 2×, §4.2) — total still ≈ 657K ± the operator delta, well within
budget. The decoder dominates either way.

---

# 4. `GatedMLPTransition` — the black-box baseline (engram-wm)

## 4.1 Purpose and interface contract

The engram-wm program needs a **black-box** transition behind the *same* `apply(k, v)`
interface as the structured operator, to isolate "what does the polar/rotation structure
buy over a generic learned transition?". `GatedMLPTransition` is a verb-conditioned gated
MLP that is a **drop-in `Operator` subclass** — `model.py`'s `op_cls` dispatch picks it
via `operator_group="gated_mlp"`, and the rest of the pipeline (forward, unroll, decoder,
InfoNCE, losses) is **byte-identical**.

**Interface (must satisfy the `Operator` ABC, `twm/jepa/__init__.py`):**

```python
class GatedMLPTransition(Operator):
    # apply(k, v, theta_offset=None) -> a*       # theta_offset ACCEPTED & IGNORED (see below)
    # inverse_apply(a, v, ...)        -> raises NotImplementedError  (NO inverse — documented)
    # velocity(k, v)                  -> apply(k,v) - k   (for the dormant T-step seam parity)
    # integrate(k, v, T=1)            -> apply (T must be 1; raise otherwise)
    # structural_sanity(v)            -> {"bbT_err": nan, "inv_err": nan}  (no inverse; report NaN, documented)
    # bake()                          -> raises NotImplementedError ("black-box, not JS-exportable")
    # n_verbs property
```

- `apply` accepts a `theta_offset=None` kwarg for **signature compatibility** with the
  polar-conditioning call site in `model._apply_action`, but **ignores it** (the gated
  MLP has no phase/modulus split). Documented in the docstring and asserted by a test
  (passing a nonzero offset changes nothing). This keeps `model._apply_action` branch-free
  across operator families.
- **No inverse.** `inverse_apply` raises `NotImplementedError("GatedMLPTransition is a
  black-box transition with no structural inverse (engram-wm baseline)")`. The model's
  `undo_latent` / pet-demo path is **not used in v3 training**, so the missing inverse
  never executes in the train loop; the baseline is documented as *non-reversible* — that
  is the *point* of contrasting it with the invertible operator. A test asserts the raise.
- `bake()` raises (not JS-exportable). `structural_sanity` returns NaN-filled dict so the
  diagnostics harness (which may call it) does not crash — documented as "no inverse →
  invertibility metrics undefined".

## 4.2 Architecture (param-matched within ~2× of the operator path)

The operator path's *learnable* transition params are the operator codebook
(`2·V·nb = 256`) **plus** the polar `H` (`256`) = 512 conditioning params. A param-match
"within ~2×" means the gated MLP should land in roughly **256–1024** transition params —
but a useful MLP needs a hidden layer, so we target the **low-2× end** with a deliberately
*narrow* design and document that the operator is parameter-cheaper by construction (the
structured prior's whole pitch):

```python
# Per verb v: a gating + transform over the noun. Verb enters as a learned embedding.
verb_emb : nn.Embedding(V, d_e)            #  V·d_e
# Gated MLP on (k, verb_emb): h = GELU(W1 [k ; e_v]) ; gate = sigmoid(Wg [k ; e_v])
# a* = k + gate ⊙ (W2 h)                   # residual + gate (zero-init Wg bias -> near-identity start)
W1 : Linear(dn + d_e, d_h)                 # (dn+d_e)·d_h
W2 : Linear(d_h, dn)                       # d_h·dn
Wg : Linear(dn + d_e, dn)                  # (dn+d_e)·dn   (the learned gate)
```

**Sizing for nano (dn=32, V=8):** pick `d_e=8`, `d_h=16`.
- `verb_emb`: 8·8 = 64
- `W1`: (32+8)·16 = 640
- `W2`: 16·32 = 512
- `Wg`: (32+8)·32 = 1280
- **Total ≈ 2,496** transition params.

This is ~5× the operator's 512 conditioning params — *above* the strict 2× band. To honor
"within ~2×", **shrink to `d_h=8, d_e=4`**:
- `verb_emb`: 8·4 = 32; `W1`: 36·8 = 288; `W2`: 8·32 = 256; `Wg`: 36·32 = 1152;
  **total ≈ 1,728** (still ~3.4×). The gate `Wg` dominates.

Resolution: **drop the separate full-width gate.** Use a **per-verb scalar-vector gate**
(a learned `(V, dn)` gate table, like the operator's own `(V, nb)` shape) instead of a
`Linear(dn+d_e, dn)`:

```python
verb_emb : Embedding(V, d_e=4)             # 32
W1 : Linear(dn + d_e, d_h=8)               # 36·8 = 288
W2 : Linear(d_h, dn)                       # 8·32 = 256
gate : Parameter(V, dn) (zero-init)        # 8·32 = 256   (per-verb gate, sigmoid)
# a* = k + sigmoid(gate[v]) ⊙ W2(GELU(W1([k; e_v])))
```
- **Total = 32 + 288 + 256 + 256 = 832** transition params vs the operator's **512**
  (op 256 + H 256) → **1.6×**. **Within the ~2× requirement.** Documented in the config
  (`model.gated_mlp = {d_e:4, d_h:8}`).

The gate is **zero-init** so `sigmoid(0)=0.5`-scaled residual? No — zero-init gate gives
`sigmoid(0)=0.5`, a half-residual, not identity. To match the operator's near-identity
init (the v2.1 zero-init-H spirit), init the **gate bias to a large negative** so
`sigmoid(gate)≈0` at start → `a* ≈ k` (near-identity transition at init, like the
operator's small-θ init). Use `gate` init `= -4.0` (sigmoid(-4)≈0.018). Documented.

`d_e`, `d_h` are config fields under `model.gated_mlp` (default `{d_e:4, d_h:8}`); the
factory in `model.py` reads them when `operator_group=="gated_mlp"`.

## 4.3 It trains through the identical pipeline

- `build_jepa_model_v2`'s `op_cls` dict gains `"gated_mlp": GatedMLPTransition`. The
  `_construct` kwarg-filtering already drops unknown kwargs, so passing
  `n_verbs, d_noun, block, d_e, d_h` is safe (the operator classes that don't take
  `d_e/d_h` ignore them).
- `model._apply_action` calls `operator.apply(k, v_slots, theta_offset=...)` — the
  gated MLP accepts and ignores `theta_offset`, so **no model branch is needed**.
- The unroll loop (§2.3) calls `apply` twice; the gated MLP composes by re-application
  (`a2 = apply(apply(k0,v1), v2)`) just like the operator — no special-casing.
- Losses (`L_token`, `L_nce`, `L_prior`, `L_sigreg`) are operator-agnostic. The baseline
  trains with InfoNCE + unroll exactly like `jepa_v3`, isolating the structured-prior
  contribution.
- The only differences a builder will hit: `inverse_apply`/`bake` raise (never called in
  training), and the diagnostics' invertibility/modulus metrics are NaN/skipped for the
  baseline (guard with `try/except NotImplementedError` already idiomatic in the harness).

---

# 5. Config schema additions + the 8 configs

## 5.1 Schema additions (`ModelHParams`, `LossConfig`, `DataConfig`)

```python
# DataConfig (twm/jepa/__init__.py + config.py):
mode: str = "pairs"                 # "pairs" (v2 default) | "triples" (unroll)

# ModelHParams:
#   operator_group gains "gated_mlp" as a valid value (no schema change, just a new string)
@dataclass
class GatedMLPConfig:               # NEW nested block, model.gated_mlp
    d_e: int = 4                    # verb embedding width
    d_h: int = 8                    # MLP hidden width
gated_mlp: GatedMLPConfig = field(default_factory=GatedMLPConfig)

# LossConfig:
w_nce: float = 0.0                  # InfoNCE weight; v3 sets 0.25 (replaces w_pred)
@dataclass
class NCEConfig:                    # NEW nested block, loss.nce
    temperature: float = 0.1
nce: NCEConfig = field(default_factory=NCEConfig)
@dataclass
class UnrollConfig:                 # NEW nested block, loss.unroll
    hop_weights: list[float] = field(default_factory=lambda: [1.0, 0.5])
unroll: UnrollConfig = field(default_factory=UnrollConfig)
```

All default to **v2.1-equivalent behavior** (`mode="pairs"`, `w_nce=0.0`,
`operator_group` unchanged), so every existing config parses and builds an identical
model — the behavior-preservation gate holds. `config.py`'s `_build_model` parses
`gated_mlp` like the other nested blocks; `_build_loss` parses `nce`/`unroll`; `DataConfig`
gains `mode` via `_only_known`.

## 5.2 The decoder block in every v3 config

```jsonc
"decoder": { "d_dec": 128, "n_layers": 2, "n_heads": 4, "d_ff": 256 }
```

(d_ff=256 = 2·d_dec, matching the matrix `dec` arm; n_heads stays 4.)

## 5.3 The 8 configs (+ smoke variants)

Two families × three seeds, plus a smoke per family. Out_dirs as specified.

**Family `v3` (rotation_scale operator + InfoNCE + unroll + polar):**

| config | seed | operator_group | mode | w_nce | w_pred | out_dir |
|---|---:|---|---|---:|---:|---|
| `configs/jepa/jepa_v3_s0.json` | 0 | rotation_scale | triples | 0.25 | 0.0 | `results/jepa_v3_s0` |
| `configs/jepa/jepa_v3_s1.json` | 1 | rotation_scale | triples | 0.25 | 0.0 | `results/jepa_v3_s1` |
| `configs/jepa/jepa_v3_s2.json` | 2 | rotation_scale | triples | 0.25 | 0.0 | `results/jepa_v3_s2` |
| `configs/jepa/jepa_v3_smoke.json` | 0 | rotation_scale | triples | 0.25 | 0.0 | `results/jepa_v3_smoke` |

**Family `v3_baseline` (gated_mlp black-box + InfoNCE + unroll, NO polar — the MLP ignores it):**

| config | seed | operator_group | mode | w_nce | w_pred | out_dir |
|---|---:|---|---|---:|---:|---|
| `configs/jepa/jepa_v3_baseline_s0.json` | 0 | gated_mlp | triples | 0.25 | 0.0 | `results/jepa_v3_blackbox_s0` |
| `configs/jepa/jepa_v3_baseline_s1.json` | 1 | gated_mlp | triples | 0.25 | 0.0 | `results/jepa_v3_blackbox_s1` |
| `configs/jepa/jepa_v3_baseline_s2.json` | 2 | gated_mlp | triples | 0.25 | 0.0 | `results/jepa_v3_blackbox_s2` |
| `configs/jepa/jepa_v3_baseline_smoke.json` | 0 | gated_mlp | triples | 0.25 | 0.0 | `results/jepa_v3_blackbox_smoke` |

That is **8 configs** (6 seeded runs + 2 smokes). Common settings in every v3 config:
`profile="jepa_v3"`, `data.mode="triples"`, `data.path=chain_general_train.jsonl`,
decoder block §5.2, `use_polar_conditioning=true` (v3) / `false` (baseline — the MLP has
no phase, so polar is meaningless and left off), `loss.nce.temperature=0.1`,
`loss.unroll.hop_weights=[1.0,0.5]`, optim/eval as the v2.1 configs. The v3 family sets
`use_polar_conditioning=true`; the baseline sets `operator_group="gated_mlp"` and
`model.gated_mlp={d_e:4,d_h:8}`.

**Smoke variants** set `data.max_chains` small (e.g. 64), `optim.epochs` small (e.g. 3),
`eval.every_epochs=1` — for the behavior/CI gate and a fast end-to-end pipeline check on
both operator families and triple mode. Out_dirs `results/jepa_v3_smoke` /
`results/jepa_v3_blackbox_smoke`.

---

# 6. Work breakdown — 3 disjoint-file tasks

Strict file ownership, **no overlaps**. Each task ships its own tests. The three tasks can
proceed in parallel against the frozen interfaces below.

### Task A — InfoNCE loss (owns `losses.py`)

**Files: `src/twm/jepa/losses.py`, `tests/jepa/test_losses.py` (extend).**

- Add `info_nce(zhat, z_target, chain_ids=None, temperature=0.1)` → scalar: L2-normalize,
  `(B,B)` cosine-sim logits / τ, optional same-chain hard-negative mask handling (§1.4),
  `cross_entropy` against the diagonal. Stop-grad on `z_target` is the caller's
  responsibility but assert it is detached (or detach defensively).
- Extend `JEPALossV2.__init__` with `w_nce: float = 0.0`, `nce_temperature: float = 0.1`,
  and `JEPALossV2.forward` to accept optional `chain_ids` and (for unroll) **lists** of
  per-hop tensors — see the unified signature below. Compute `L_nce` only when
  `w_nce > 0`. Add `L_nce` to the components dict and the weighted total. Keep `w_pred`
  path intact (recoverability).
- **Frozen forward signature for the loss** (Task B calls this; do not change after):
  ```python
  forward(logits, tgt_ids, tgt_pad, k, v_logits, p_logits, zhat, z_target, tau,
          chain_ids=None,                       # (B,) long for same-chain negatives
          nce_neg_keys=None)                    # optional (B, n_neg, dn) extra hard-neg keys (unroll cross-hop)
  ```
  For multi-hop, Task B calls the loss **once per hop** and sums with the hop weights
  *outside* the loss (keeps the loss single-hop and pure); the loss itself never knows
  about hops. `nce_neg_keys` carries the cross-hop hard negative (e.g. `z_2` when scoring
  hop 1). This keeps Task A's contract hop-agnostic.
- **Leakage test (§1.6):** a unit test feeding a decoder-memory-recording stub asserts the
  InfoNCE path never passes anything but `a*` to a decoder (the loss has no decoder
  argument — assert at the signature level) and that `z_target`/`nce_neg_keys` carry no
  gradient into the decoder.
- Tests: `info_nce` shape/finite; diagonal-positive gives near-zero loss when anchor==key;
  same-chain mask excludes the anchor's own positive; `w_nce=0` reproduces the exact v2.1
  total (regression vs a stored baseline); temperature scaling monotonicity.

### Task B — data triple-mode + train-script unroll loop + configs (owns `data.py`, `train_jepa_v2.py`, `configs/jepa/jepa_v3*.json`)

**Files: `src/twm/jepa/data.py`, `scripts/train_jepa_v2.py`, the 8 config JSONs,
`tests/jepa/test_data.py` (extend).**

- `data.py`: add `mode` arg + triple-mode tensors/keys (§2.1). Pairs mode untouched.
  `chain_ids` property works in both modes.
- `train_jepa_v2.py`:
  - **Chain-contiguous batching** (§1.5) for pairs mode: shuffle at chain granularity.
  - **Triple-mode branch:** `_get_batch` returns the `s0/s1/s2` tensors; the unroll loop
    (§2.3) runs two `model.forward`-style hops. **Important:** the unroll loop calls the
    model's per-hop primitives. To avoid Task C/B overlap, Task C exposes a **single
    model method** `forward_unroll(s0, s0_pad, s1, s1_pad, s2, s2_pad, tau, hard)` that
    returns per-hop `{logits, k/a, v_logits, p_logits, zhat, z_target}` dicts (Task C owns
    `model.py`); Task B's train loop *calls* it and assembles the per-hop loss with the
    hop weights (§2.4) and passes cross-hop `nce_neg_keys`. So **`model.py` belongs to
    C**; the **train script belongs to B** and only orchestrates.
  - Sum `L_token`/`L_nce`/`L_prior` over hops with `loss.unroll.hop_weights`. Logging adds
    per-hop CE.
  - Config parsing for the new `loss.nce`, `loss.unroll`, `data.mode`, `model.gated_mlp`
    blocks (these are dataclass additions Task C/A declare; Task B just lets `config.py`
    parse them — coordinate the dataclass additions in `__init__.py`/`config.py`: **assign
    the `__init__.py`/`config.py` dataclass edits to Task A** since it owns the loss
    dataclasses (`NCEConfig`, `UnrollConfig`, `w_nce`) and **Task C** for `GatedMLPConfig`
    + `DataConfig.mode` is owned by **Task B** with the `data` schema).
    > Ownership of the shared schema files (`__init__.py`, `config.py`): these are touched
    > by all three. To keep tasks disjoint at the *file* level, **the dataclass additions
    > land in one PR owned by Task B** (it owns the most schema surface — `data.mode`,
    > and it wires the train script that reads everything), with A and C providing their
    > field specs (`NCEConfig`/`UnrollConfig`/`w_nce` from A; `GatedMLPConfig` from C) as
    > reviewed diffs B applies. `__init__.py` + `config.py` are **Task B-owned files.**
- Write the 8 configs (§5.3).
- Tests: triple-mode `__getitem__` keys/shapes; length-<3 chains skipped + count asserted;
  `chain_ids` correctness in triple mode; chain-contiguous batching co-locates siblings;
  a tiny end-to-end smoke (`max_chains=8, epochs=1`) runs the unroll loop without error.

### Task C — `GatedMLPTransition` + model wiring (owns `operator.py` addition + `model.py`)

**Files: `src/twm/jepa/operator.py` (add the class), `src/twm/jepa/model.py`,
`tests/jepa/test_operator.py` + `tests/jepa/test_model.py` (extend), provides
`GatedMLPConfig` field spec to Task B.**

- Add `GatedMLPTransition(Operator)` to `operator.py` (§4) — `apply(theta_offset ignored)`,
  `inverse_apply`/`bake` raise, `velocity`/`integrate(T=1)` parity, near-identity init.
- `model.py`:
  - `build_jepa_model_v2` `op_cls` dict gains `"gated_mlp": GatedMLPTransition`; pass
    `d_e/d_h` from `m.gated_mlp` (kwarg-filtered by `_construct`).
  - Add `forward_unroll(...)` (the §2.3 two-hop method) returning per-hop dicts. It reuses
    `_apply_action` (which already threads polar conditioning and is operator-agnostic) and
    the per-hop posterior/prior/decoder calls. **This is the only new model method**; the
    train script (Task B) orchestrates the per-hop loss. `forward_unroll` reads
    `loss.unroll`/`w_*`? No — it is loss-free; it returns raw per-hop outputs and the
    trainer applies weights. Keeps C/B disjoint.
  - hop-2 conditioning reads `|a1|` (§2.3) — `_apply_action(a1, v2)` already computes
    `conditioner(a1)`, so no change beyond calling it on `a1`.
- Tests: `GatedMLPTransition.apply` shape/finite + `theta_offset` ignored (nonzero offset
  → identical output); `inverse_apply`/`bake` raise `NotImplementedError`; near-identity at
  init (`apply(k,v) ≈ k` within tol); param count within ~2× of the operator (assert
  512 ≤ count ≤ 1024-ish band, document the 832 figure); `forward_unroll` returns two hops
  with correct shapes and the hop-2 memory is a function of `(k0,v1,v2)` only (leakage:
  perturbing `s2` changes hop-2 memory **only** via `v2`); `operator_group="gated_mlp"`
  builds end-to-end.

**Disjoint-file guarantee:** A=`losses.py`(+its tests); B=`data.py`,
`train_jepa_v2.py`, `__init__.py`, `config.py`, the 8 configs (+their tests);
C=`operator.py`(the new class), `model.py`(+their tests). `model.py`→C, train script→B,
`losses.py`→A, exactly as required. The schema files (`__init__.py`/`config.py`) are
B-owned; A and C deliver their dataclass field specs as diffs B applies (the only
cross-task coordination point, and it is a one-directional handoff into B).

---

## Summary of resolved decisions

1. **InfoNCE:** pooled `zhat`(anchor, grad) vs stop-grad EMA `z_target`(key) — the exact
   tensors diagnostics already retrieves on. Cosine sim, `τ_nce=0.1`, `w_nce=0.25`
   replacing `w_pred` (same slot, discriminative upgrade). Negatives = in-batch +
   same-chain (guaranteed via chain-contiguous batching in pairs mode, per-example
   cross-hop in triple mode). Leakage-neutral: keys/negatives are stop-grad and never
   reach the decoder (audited §1.6).
2. **Multi-step unroll:** `data.mode="triples"` emits `(s0,s1,s2)`; per-hop posterior
   `v1=q(s0,s1)`, `v2=q(s1,s2)`; composed `a2=B_{v2}(B_{v1}k0)` via the angle-additive
   path; **H reads hop-1 modulus `|a1|` for hop 2** (correct state-dependence). Token CE
   + InfoNCE at both hops, weights **1.0/0.5**. Hop-2 `s2` reaches the decoder only
   through `v2`'s bits (audited §2.5).
3. **Profile:** `jepa_v3` = nano + d128/2L decoder + polar; **≈657K params**, InfoNCE adds
   **0** (reuses the L_pred head).
4. **Baseline:** `GatedMLPTransition` — `Operator` subclass, verb-embedding + narrow MLP +
   per-verb zero-ish gate, **832 params (1.6× the operator)**, `inverse_apply`/`bake`
   raise, `theta_offset` accepted-and-ignored, trains through the identical pipeline via
   `operator_group="gated_mlp"`.
5. **Configs:** `data.mode`, `loss.w_nce`, `loss.nce`, `loss.unroll`, `model.gated_mlp`
   added (all default to v2.1 behavior). 8 configs: `{v3, v3_baseline}×{s0,s1,s2}` +
   smoke each, out_dirs `results/jepa_v3_s{N}` / `results/jepa_v3_blackbox_s{N}`.
6. **Work breakdown:** A=`losses.py`(InfoNCE)+tests; B=`data.py` triple-mode +
   `train_jepa_v2.py` unroll loop + `__init__.py`/`config.py` schema + 8 configs +tests;
   C=`GatedMLPTransition` in `operator.py` + `model.py` wiring (`forward_unroll`,
   `op_cls` dispatch) +tests. No file overlaps; `model.py`→C, train script→B,
   `losses.py`→A.
