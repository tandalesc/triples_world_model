# JEPA v4 — Targeted Latent Actions, Diff-Weighted CE, Token Hard-Negative Contrastive, Entity-World v2

Status: **design, fully resolved.** Builds on the live v3 substrate
(`research/jepa_v3_design.md`), the entity campaign / norm budget
(`research/jepa_entity_campaign.md`), and the v2.1 polar path
(`research/jepa_v21_polar.md`). Empirical motivation:
`results/jepa_separation/separation_diag_ep50.json` (encoder linear-probe AUC = **0.532**
— the encoder cannot AIM at gold), `results/jepa_mrr_diagnosis/` (the decoder smears
discriminative clauses across the whole pool), and the four obligations restated in §0.

The v4 success criterion is **rung-0 separation AUC moving from 0.53 toward >0.7**
(diagnostic ported from the `bb01bfd` probe script into `diagnostics.py` as an
every-N-epochs metric, §C5). Everything in §1–§3 is **config-gated and DEFAULT FALSE**,
so an unmodified v3/entity config builds a **bitwise-identical** model (the v3 + entity
behavior-preservation gate still holds). Stratified non-abelian operator blocks are
**deferred to v4.1**, pending retraction-bracket results — out of scope here.

---

## 0. The four obligations (the empirical basis)

From the campaign observations (restated so a builder never loses them):

1. **The encoder cannot aim at gold.** Separation AUC 0.53 (≈ chance) on the
   `test_iid` hard pools means a density scorer has no signal to rank — the
   discriminative clause that distinguishes the gold next-state from same-chain siblings
   is not separable in the readout geometry. The contrastive fixes (§3, token hard-neg)
   must TEACH the aim — pure CE never pressures discrimination (the v3 InfoNCE pooled the
   readout, which the matrix already showed is too coarse; v4 adds a **token-level**
   margin that bites where the difference lives).
2. **The target is a SET.** A transition changes only SOME entities/slots. v3 applies the
   verb operator to ALL M slots identically (`v_onehot.unsqueeze(1).expand(B,M,V)`); the
   identity slots are forced to "no-op" implicitly. v4 makes the target-set explicit: a
   **per-slot target mask** the operator applies `B_v` to ONLY the targeted slots, exact
   identity elsewhere (§1). This makes disjoint-support commutation *architectural*, not
   merely abelian-incidental.
3. **The decoder smears discriminative clauses.** The mrr_diagnosis shows token CE spends
   its budget on boilerplate ("The dog feel(s) ...") that every candidate shares, so the
   gold-distinguishing token ("hungry" vs "fed") is under-weighted. v4 **diff-weights** the
   CE (§2): tokens in the s_t→s_{t+1} diff get weight `w_diff`, boilerplate weight 1.0.
4. **The world commutator invoice is frozen pre-targeted-actions.** The non-abelian
   stratified-block expansion (v4.1) cannot be sized until the targeted-action mask +
   retraction-bracket results are in. v4 ships the mask, the diff-weight, the token margin,
   the bigger world, and the diagnostics; v4.1 reads their results to size the operator.

---

# 1. TARGETED LATENT ACTIONS — the per-slot target mask

## 1.0 Mechanism

v3's `_apply_action` applies the SAME sequence-level verb `v` to all M slots:
`a* = B_v k` over every slot. The entity world's transitions touch only the acting
entity's slots (a `feed@0` changes entity-0's hunger/mood; entity-1 is untouched). v4
adds a **target-slot mask** `g ∈ [0,1]^(B,M)` — one gate per slot — emitted by the
posterior `TransitionEncoder` (it already sees `(s_t, s_{t+1})`, so *which slots changed*
is inferable from the pair). The operator applies `B_v` to slot `i` only to the extent
`g_i`, and is **exact identity** where `g_i = 0`:

```
a*_i = g_i · (B_v k_i) + (1 − g_i) · k_i           # per-slot gated operator apply
```

At eval, `g` is **hard-thresholded** at 0.5 with a straight-through estimator (forward =
hard 0/1, backward = sigmoid gradient), so targeted slots get the exact operator and
identity slots get the **exact** input — disjoint-support commutation becomes a structural
property: two actions on disjoint target-sets commute exactly, not merely
abelian-incidentally (the operator's existing abelian commutativity, §Operator ABC,
becomes *partitioned* by support).

Config-gated `use_targeted_actions`, **default False ⟹ v3 bitwise** (`g ≡ 1` everywhere,
which makes `a*_i = B_v k_i` exactly the v3 path — the convex combination collapses).

## 1.1 Mask architecture (TransitionEncoder addition — Task A)

The mask is a SECOND head on the existing `TransitionEncoder` trunk. The posterior already
computes `pool_t (B,d)`, `pool_t1 (B,d)`, and (with `use_delta`) their delta; it pools to
ONE sequence-level vector and emits the verb logits from it. The mask, by contrast, is
**per-slot**, so it needs a per-slot input, NOT the pooled pair vector.

**Decision: the mask head reads the per-slot DELTA between the two states' slot nouns,
which is the directly inferable "which slot's content changed" signal.** Concretely the
model (Task C) hands the TransitionEncoder the start nouns `k (B,M,dn)` and the EMA
target nouns `k_tgt (B,M,dn)` (the raw-noun encode of `s_{t+1}`, the same `pool_raw`
trunk path the InfoNCE key already uses); the mask head scores each slot from
`[k_i ; k_tgt_i ; |k_tgt_i − k_i|]`:

```python
# TransitionEncoder gains (Task A), built ONLY when use_targeted_actions=True:
#   mask_fc1 : Linear(3*dn, mask_hidden)      # per-slot, default mask_hidden=2*dn
#   mask_act : GELU
#   mask_fc2 : Linear(mask_hidden, 1)         # per-slot logit
# forward_mask(k, k_tgt) -> g_logits (B, M):
#   feat = cat([k, k_tgt, (k_tgt - k).abs()], dim=-1)   # (B, M, 3dn)
#   g_logits = mask_fc2(mask_act(mask_fc1(feat))).squeeze(-1)  # (B, M)
#   return g_logits
```

The mask head is on the noun trunk (the `SlotEncoder`'s slot nouns), NOT the text trunk —
it needs PER-SLOT granularity which the masked-mean text pool destroys. `k_tgt` is the
**stop-grad EMA** raw-noun encode of `s_{t+1}` (the existing `model.ema.pool_raw` path
exposes the trunk; Task C threads the raw `k_tgt` slots — `model._target_slots(tgt_ids,
tgt_pad)` returns `self.ema.encoder(tgt_ids, tgt_pad)[1]` detached). Using the EMA target
(detached) is what makes the mask **inferable from the pair without leaking a continuous
future channel** — see §1.4. `mask_hidden` is a config field (`model.targeted.mask_hidden`,
default `2*dn`).

**Param delta (mask head, nano dn=32):**
- `mask_fc1`: `(3·32)·64 + 64 = 6208` (with bias)
- `mask_fc2`: `64·1 + 1 = 65`
- **Total ≈ 6,273** new params, built **only** when `use_targeted_actions=True`
  (default off ⟹ 0). This is the only new trainable parameter v4 adds on the model side
  (the diff-weight and token-margin are losses with no params; §2/§3).

## 1.2 Gated operator apply (model.py `_apply_action` extension — Task A)

The convex combination is applied in `model._apply_action` AROUND the operator, so the
operator algebra is untouched (Task A owns model.py per §A; the operator file is NOT
edited for the mask — the gate lives in the model wrapper, keeping `operator.py` a pure
algebra module). The straight-through hard threshold at eval:

```python
# model._apply_action(k, v_onehot, g_logits=None):  (g_logits from the mask head)
a_op = operator.apply(k, v_slots, theta_offset=..., norm_budget=...)   # B_v k (existing)
if g_logits is None:                          # use_targeted_actions off ⟹ v3 bitwise
    return a_op                               # (or (a_op, scale_delta) under the budget)
g_soft = torch.sigmoid(g_logits)              # (B, M) in (0,1)
if self.training:
    g = g_soft                                # soft gate, full gradient
else:
    g_hard = (g_soft > 0.5).to(g_soft.dtype)  # hard 0/1
    g = g_hard + (g_soft - g_soft.detach())   # straight-through: hard fwd, soft grad
a = g.unsqueeze(-1) * a_op + (1.0 - g).unsqueeze(-1) * k     # per-slot gated apply
```

**Interaction with the norm budget + scale accumulator (load-bearing, entity §1).** When
`use_norm_budget` is on, the operator returns `(a_op, scale_delta)`. The gate must apply to
BOTH the noun and the tracked scale so that **identity slots accumulate ZERO scale**:

```python
a_op, scale_delta_op = operator.apply(k, v_slots, ..., norm_budget=True)  # (B,M,dn),(B,M)
a = g.unsqueeze(-1) * a_op + (1.0 - g) .unsqueeze(-1)* k    # gated noun
scale_delta = g * scale_delta_op                            # gated scale: identity ⟹ 0
```

This is the **critical correctness point**: the norm budget's whole pitch (entity §1.0) is
that the per-slot log-scale `s_acc` is a stable identity signal; if an identity slot
(`g_i=0`) still accumulated the operator's `scale_delta_op` (which is nonzero whenever the
verb scales), the identity slot's tracked radius would drift even though its noun did NOT
change — corrupting the identity readout and breaking the inverse. Gating `scale_delta` by
`g` makes identity slots contribute exactly `log_rho = 0` (scale 1.0), so the accumulator
`s_acc` only records drift on slots the action actually touched. The retraction-probe
inverse (entity §4.2) replays `scale_delta` per hop; the gated `scale_delta` is what gets
stored and replayed, so the round-trip stays exact: on an identity slot the stored delta is
0 and `inverse_apply` re-applies 0, while the noun was never moved (`a_i = k_i`), so
`inverse_apply(a_i, v_j, scale_delta=0)` must return `k_i`. **It does NOT** unless we also
gate the inverse — see §1.5.

**Interaction with H-conditioning.** The polar conditioner `θ_off = H(|k|)` is computed
from the operator's INPUT `k` BEFORE gating (it reads the pre-step modulus, v2.1 §3.3) and
is fed to `operator.apply` unchanged. The gate is applied to the operator's OUTPUT, so the
conditioning is untouched by the mask: `θ_off` still reads `|k|`, and at hop 2 the H input
is the **gated** output `a1` (which is `k0_i` exactly on identity slots — the correct
state-dependent input, because an untouched slot's modulus shape is preserved exactly, not
just radius-renormalized). This is *stronger* than the entity §1.2 guarantee: identity
slots feed H their EXACT unchanged modulus profile, so H's hop-2 input on those slots is
provably stationary.

## 1.3 PriorHead mask extension (Task A)

For autonomous rollout the posterior is gone, so the prior must predict the mask too.
`PriorHead` gains a per-slot mask head reading `k` (state_t slot nouns) alone:

```python
# PriorHead gains (built only when use_targeted_actions=True):
#   mask_fc1 : Linear(dn, mask_hidden)        # per-slot, from start nouns only
#   mask_act : GELU
#   mask_fc2 : Linear(mask_hidden, 1)
# forward_mask(k) -> g_prior_logits (B, M)
```

`L_mask_prior = BCE(sigmoid(g_prior_logits), sg(sigmoid(g_logits)))` — the prior mask is
distilled from the (stop-grad) posterior mask, exactly as `L_prior` distills the verb
(losses.py, Task B; weight `w_mask_prior`, default 0.1, summed into the prior term). At
rollout the model thresholds `g_prior_logits` at 0.5 and gates the operator the same way.
The prior reads only `state_t` slot nouns (no `s_{t+1}`), so it is leakage-clean by the
same argument as `PriorHead` itself (transition.py §3, "the prior never sees text_{t+1}").

## 1.4 Leakage analysis — the mask's bits add to the future→decoder channel

The v3 leakage invariant (jepa_v2 §6, jepa_v3 §1.6): the **only** path from `s_{t+1}` into
the decoder memory `a*` is the discrete verb `v` (⌈log₂V⌉ bits). The mask `g` is **also
inferred from the pair** (`forward_mask(k, k_tgt)` reads the EMA encode of `s_{t+1}`), so
its bits ALSO flow from the future into `a*`. We must quantify and bound this channel.

**Budget accounting.** The decoder memory is now `a = g⊙(B_v k) + (1−g)⊙k`. The future-
derived quantities reaching it are:
1. the verb `v`: `⌈log₂V⌉ = ⌈log₂8⌉ = 3` bits (unchanged from v3), and
2. the **hard mask** `g ∈ {0,1}^M` at eval: at most **M bits** (one bit per slot —
   "did this slot change?"), `M=8` ⟹ **≤ 8 bits**.

So the future→decoder channel grows from **3 bits** (v3) to **ceil(log₂V) + M = 3 + 8 = 11
bits** worst-case. Crucially the mask bits are a DIFFERENT KIND of information than the
verb: the verb says *what transformation*, the mask says *where it applies*. The mask
cannot encode the CONTENT of the next state (only which slots moved), because:
- `g_i ∈ {0,1}` is a single bit per slot at eval (straight-through hard threshold). A
  continuous `g_i ∈ [0,1]` would be a high-bandwidth channel (the soft-mix leak of jepa_v2
  §6 L3); the **hard threshold at eval bounds it to 1 bit/slot**, mirroring the hard-ST
  verb bound. The soft gate is train-only (gradient), the same asymmetry the verb uses.
- On an identity slot (`g_i=0`) the decoder memory is `a_i = k_i = f(s_t)` exactly — no
  `s_{t+1}` content. On a targeted slot (`g_i=1`) the memory is `B_v k_i`, whose only
  `s_{t+1}` dependence is still the discrete `v`. So the mask adds **location** bits, not
  **content** bits: the decoder learns *which* slots to render from `B_v k` vs `k`, but the
  rendered values still route through `v`.

**Extended permutation audit.** The v3 audit (jepa_v3 §1.6) asserts the decoder receives
only `a*` and no non-`v` continuous future channel. v4 extends it: the leakage test (Task A,
`test_targeted_leakage`) asserts (a) `forward_mask`'s output is the ONLY mask channel and
it is reduced to `≤M` hard bits before reaching the decoder (the test feeds a decoder stub
recording its memory arg and asserts `a` is a function of `(k, v, g_hard)` only — perturb
`s_{t+1}`'s CONTENT while holding the diff-set fixed and assert `g_hard` and hence `a` are
unchanged on identity slots); (b) the per-slot mask logit carries no gradient into the
decoder by a non-`a` route. The audit also re-runs the v3/entity bit-count assertion with
the **11-bit** ceiling logged (`ceil(log2(V)) + M`), so the budget is explicit in the test.

**Why 11 bits is acceptable.** The decoder must render an M-slot state; telling it which of
M slots changed is information it could otherwise only get by re-deriving the diff from a
high-bandwidth channel. The mask makes that derivation explicit and **hard-bounded**
(`M` bits) instead of leaking it as continuous activations. The information content of "an
M-slot diff-set" is exactly `M` bits, so the mask is the *minimal honest channel* for the
set-target obligation (§0 obligation 2) — it cannot be smaller without losing the ability
to express an arbitrary diff-set. Documented in the design and asserted by the bit-count
test.

## 1.5 Inverse exactness under gating (entity §4 retraction interaction)

The retraction probe (entity §4) requires `inverse_apply(apply(k,v,...)) == k` exactly. With
gating, the forward on slot `i` is `a_i = g_i·(B_v k_i) + (1−g_i)·k_i`. For the inverse to
recover `k_i` on BOTH targeted and identity slots, the model's `undo_latent` /
retraction-probe inverse must **gate the inverse with the SAME hard `g`** that the forward
used (stored alongside `theta_offset` and `scale_delta`):

```python
# inverse of a gated step (model.undo_latent / probe, given stored g_hard, theta_offset, scale_delta):
k_inv_op = operator.inverse_apply(a, v_slots, theta_offset=..., norm_budget=..., scale_delta=g*scale_delta_op)
# targeted slot (g=1):  a_i = B_v k_i  ⟹ inverse_apply recovers k_i exactly.
# identity slot (g=0):  a_i = k_i      ⟹ inverse_apply(k_i, v) ≠ k_i in general!
k = g_hard.unsqueeze(-1) * k_inv_op + (1.0 - g_hard).unsqueeze(-1) * a   # gate the inverse
```

The gated inverse is exact: on `g=1` slots it uses the operator inverse (which recovers
`k_i` because `a_i = B_v k_i` and the stored `scale_delta = g·scale_delta_op = scale_delta_op`);
on `g=0` slots it returns `a_i` unchanged, and `a_i = k_i`, so `k_i` is recovered. **The
stored mask must be the HARD `g`** (the eval threshold), because only the hard partition
makes the convex combination collapse to a clean operator-or-identity per slot. The
retraction probe (entity §4, Task C in the campaign — unchanged file ownership) stores
`g_hard` per hop alongside `(v, theta_offset, scale_delta)`; **Task A** exposes
`g_hard` in the `step_latent` / `forward_unroll` return so the probe can thread it (the
probe script itself is campaign-Task-C; v4 only adds the `g_hard` field to the model's
return dict, which is Task A's `model.py` surface). The round-trip test (§A tests) asserts
`undo_latent(step_latent(k, v, g), g_hard=g) ≈ k` for a constructed mixed mask.

## 1.6 Target-recovery diagnostic (Task C — diagnostics.py)

A new eval metric joins action-NMI: **target-recovery** of the inferred mask vs the oracle
entity that moved. For each labeled pair `(s_t, s_{t+1})` with oracle action
`"<verb>@<entity_idx>"`, the oracle target-set is the SLOTS that belong to entity
`entity_idx`. But the model's slots are unlabeled (slot↔entity is not given). So
target-recovery is scored two ways:

1. **NMI(g_hard cluster ; oracle moved-entity).** Treat the per-slot binary mask as a
   labeling: pool the mask over slots into a per-pair signature is too coarse — instead,
   score at the **pair level**: does the SET of targeted slots align with the moved
   entity? Build, per pair, the hard mask `g_hard (M,)`; the oracle moved-entity index is
   `int(action.split("@")[1])`. Since slot↔entity is latent, compute the **target-recovery
   F1** as: across all pairs, the fraction of slots that flip their mask consistently with
   the moved entity (a Hungarian-style best slot↔entity assignment over the dataset, then
   F1 of predicted-vs-oracle moved-slot membership). Report `ent_target_recovery_f1`.
2. **Mask-sparsity sanity.** `ent_target_mask_density` = mean fraction of slots with
   `g_hard=1`. The oracle moves ONE entity per step (entities_per_chain 1–3), so a correct
   mask should be sparse (≈ `1/n_entities`); a mask that fires on all slots
   (`density≈1.0`) means the model ignored the target-set and reverted to v3-style
   apply-all. A density near the oracle's mean moved-fraction is the health signal.
3. **Shuffle baseline** for the NMI/F1 (permute the per-pair masks), `ent_target_recovery_shuffle`.

Reported scalars: `ent_target_recovery_f1`, `ent_target_recovery_nmi`,
`ent_target_recovery_shuffle`, `ent_target_mask_density`, `ent_target_recovery_pass`
(bool: F1 comfortably above shuffle). Implemented in `diagnostics._target_recovery(...)`,
called from `eval_entity_world` ONLY when the loaded model has `use_targeted_actions=True`
(guarded `getattr(model, "use_targeted_actions", False)`), so v3/entity-v1 models skip it.
A per-pair mask↔entity contingency saved to `target_recovery_{epoch}.json`.

---

# 2. DIFF-WEIGHTED CE — token-level weights on the s_t→s_{t+1} diff

## 2.0 Mechanism

The decoder CE (`token_ce`, losses.py) weights every non-pad target token equally. The
mrr_diagnosis (§0 obligation 3) shows this wastes the budget on boilerplate every candidate
shares. v4 computes a **per-token weight** at dataset-load time: tokens in the DIFF between
the rendered `s_t` and `s_{t+1}` text (the tokens that actually changed) get weight
`w_diff` (default **4.0**); boilerplate tokens get **1.0**. The weighted CE focuses the
decoder on the discriminative clause.

Config-gated `loss.w_diff` (default **1.0 ⟹ uniform = v3 bitwise CE**); the dataset always
computes the weights when in a mode that supports it, but they only bite when `w_diff ≠ 1.0`.

## 2.1 Token-level alignment algorithm (data.py — Task B)

The diff is computed at **the tokenizer level** (post-BPE token ids), not the character
level, because the CE is over token positions. For each example (pairs: the `(src, tgt)`
pair; triples: each of the two hops `(s0→s1)`, `(s1→s2)`), align the two token-id sequences
with a standard **LCS / SequenceMatcher** alignment and mark TARGET tokens that are NOT in a
common subsequence block (i.e. the inserted/replaced tokens) as diff tokens:

```python
# data.py, per (src_ids, tgt_ids) (both list[int], pre-pad, EOS-appended):
from difflib import SequenceMatcher
def _diff_weights(src_ids, tgt_ids, w_diff, pad_id, T):
    sm = SequenceMatcher(a=src_ids, b=tgt_ids, autojunk=False)
    w = [1.0] * len(tgt_ids)                      # boilerplate weight
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):          # target tokens j1:j2 are the diff
            for j in range(j1, j2):
                w[j] = w_diff
        # "delete" (src-only) and "equal" leave target weight 1.0
    # pad/truncate the weight vector to T, pad positions get weight 0.0 (excluded anyway)
    weights = torch.zeros(T)
    for j in range(min(len(tgt_ids), T)):
        weights[j] = w[j]
    return weights                                # (T,) float
```

**Handling insertions/deletions explicitly:**
- `replace` (substituted token, e.g. "hungry"→"fed"): target side `j1:j2` is the diff →
  `w_diff`. This is the primary case (ordinal value changes).
- `insert` (target has extra tokens, e.g. an added action sentence): target side `j1:j2`
  is the diff → `w_diff`. (Chain steps prepend the action sentence to the state; the action
  clause is a genuine diff — it IS what changed — so weighting it is correct.)
- `delete` (src has tokens absent from target): no target position to weight, skip.
- `equal` (shared block): boilerplate → weight 1.0.
- **Pad positions get weight 0.0** (already excluded by `ignore_index=pad_id`; the weight
  vector zeroes them defensively so the weighted mean's denominator is the non-pad,
  weighted count).

The weights are stored as an additive `(N, T)` float tensor (`_tgt_diff_w` in pairs mode;
`_s1_diff_w`, `_s2_diff_w` per hop in triples mode), computed once at load (the alignment is
O(T²) worst case but T≤64 and runs once). `__getitem__`/`get_batch` return the matching
`*_diff_w` key. Default off path: when `w_diff==1.0` the weight tensor is all-ones (≡ v3
uniform CE) — but it is ALWAYS computed (cheap) so the trainer can flip `w_diff` without a
data rebuild; the bitwise-v3 guarantee holds because all-ones weighted CE == uniform CE.

## 2.2 Weighted CE loss (losses.py — Task B)

`token_ce` gains an optional `token_weights (B, T)` argument:

```python
def token_ce(logits, tgt_ids, pad_id=0, token_weights=None):
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(B*T, V), tgt_ids.reshape(B*T),
                         ignore_index=pad_id, reduction="none").reshape(B, T)  # (B,T) per-token
    valid = (tgt_ids != pad_id).float()                                        # (B,T)
    if token_weights is None:
        return (ce * valid).sum() / valid.sum().clamp_min(1.0)                 # == v3 mean
    w = token_weights * valid                                                   # zero pad weights
    return (ce * w).sum() / w.sum().clamp_min(1.0)                              # weighted mean
```

When `token_weights is None` (or all-ones over non-pad) this is **bitwise the v3 mean CE**
(same `ignore_index`, same denominator). `JEPALossV2.forward` gains `token_weights=None`
and passes it through; `JEPALossV2.__init__` gains nothing (the weight is per-token data, the
`w_diff` scalar lives in the dataset, not the loss — the loss just consumes the precomputed
per-token weights). The hop-summed unroll (v3 §2.4) passes each hop's `*_diff_w` to that
hop's `token_ce` call. Documented: `w_diff` shapes the WEIGHTS at load; the loss is
weight-agnostic.

> **Ownership note:** the diff-weight COMPUTATION (`_diff_weights`, the `*_diff_w` tensors,
> the dataset keys) is **Task B** (`data.py`); the `token_ce` `token_weights` argument and
> the `JEPALossV2.forward` passthrough are **Task B** too (losses.py is Task A in v4 — see
> §4 work breakdown — but the diff-weight CE is a *single feature* spanning data+loss, so
> to keep files disjoint, **the `token_ce` weighting goes in Task A's losses.py with the
> argument defaulting to None**, and Task B's data.py produces the weights and the train
> loop wires them through. The loss-signature change is Task A's; the data/wiring is Task
> B's. See §4.)

---

# 3. TOKEN-LEVEL HARD-NEGATIVE CONTRASTIVE — the encoder-AIM fix

## 3.0 Mechanism

The separation AUC of 0.53 says the encoder cannot aim: teacher-forced CE on the gold
next-state does not pressure `a*` to make the gold MORE likely than a plausible WRONG
next-state. v4 adds a **margin loss** at the TOKEN level: the teacher-forced CE of the gold
target given `a*` must beat the CE of a same-chain NEIGHBOR target given the SAME `a*`, by a
margin `m`:

```
L_margin = mean_i  max(0,  m  −  ( CE(neighbor_i | a*_i)  −  CE(gold_i | a*_i) ) )
```

i.e. decoding the WRONG next-state from `a*_i` should cost MORE (higher CE) than decoding
the right one, by at least `m`. This is a per-pair hinge on the **mean-CE difference**.
Unlike v3 InfoNCE (which pools the readout — too coarse, the matrix verdict), this bites at
the token level where the discriminative clause lives, directly fixing the AIM. Weight
`w_margin ≈ 0.25`; config-gated `loss.w_margin` (default **0.0 ⟹ off, v3 bitwise**).

## 3.1 Formulation (losses.py — Task A)

```python
def token_margin(logits_gold, gold_ids, logits_neg, neg_ids, pad_id=0, margin=0.5):
    """Per-pair hinge: CE(neighbor | a*) must exceed CE(gold | a*) by `margin`.

    logits_gold/logits_neg: (B, T, V) — decoder run on the SAME a* memory, teacher-forced
      on the gold target and on a same-chain neighbor target respectively.
    Returns scalar mean hinge over the batch.
    """
    ce_gold = _per_example_ce(logits_gold, gold_ids, pad_id)   # (B,) mean-CE per example
    ce_neg  = _per_example_ce(logits_neg,  neg_ids,  pad_id)   # (B,)
    return F.relu(margin - (ce_neg - ce_gold)).mean()          # hinge on the CE gap
```

- **Per-example mean CE** (`_per_example_ce`): the non-pad mean CE per row (reuses the
  `reduction="none"` path of the weighted `token_ce`, reduced per-row), so the margin is on
  the same scale (nats) as `L_token`.
- **Margin `m = 0.5` nats** (config `loss.margin`, default 0.5): half a nat of separation is
  a meaningful CE gap (≈ the ce_true 1.39→0.95 improvement scale from the matrix) without
  demanding the decoder make the neighbor impossible (which would fight `L_token`'s fluency).
- **Teacher-forcing details:** BOTH decoder runs use the SAME `a*_i` memory (the gold's
  operator output) but DIFFERENT teacher-forced target ids (`gold_ids` vs `neg_ids`). The
  decoder's `forward(a*, tgt_ids, tgt_pad)` is called twice — once per target. This is two
  forward passes through the decoder per batch (the negative pass adds ~1 decoder forward;
  acceptable, the decoder is the d128/2L profile). The negative pass's gradient flows into
  `a*` (push it away from the neighbor) and into the decoder (don't make the neighbor too
  easy) — both desired.

## 3.2 Neighbor selection (Task B — train loop + data)

The neighbor is a **same-chain, chain-contiguous** sibling target — the exact hard pool the
v3 InfoNCE used (jepa_v3 §1.4/§1.5), now at the token level:

- **Pairs mode:** the neighbor for pair `(s_t→s_{t+1})` is the OTHER adjacent next-state
  from the same chain (e.g. for `(t0→t1)` the neighbor target is `t2`). Chain-contiguous
  batching (jepa_v3 §1.5) guarantees the sibling co-occurs; the train loop builds, per
  anchor, the neighbor's `tgt_ids/tgt_pad` from the batch via the `chain_ids` mask (pick the
  same-chain sibling that is NOT the anchor's own positive; if a chain has only one pair in
  the batch, fall back to the **nearest in-batch off-chain** target so every anchor has a
  negative — logged `frac_anchors_with_chain_neg`).
- **Triples mode (the entity/v4 default):** the cross-hop sibling is the guaranteed
  neighbor, exactly as v3 InfoNCE cross-hop (jepa_v3 §2.4): for the **hop-1** anchor (`a1`,
  target `s1`), the neighbor target is `s2`; for the **hop-2** anchor (`a2`, target `s2`),
  the neighbor is `s1`. Both live in the SAME example, so the neighbor is ALWAYS present —
  no batch-composition dependence. The train loop runs the hop's decoder a second time on
  the cross-hop sibling ids and computes the margin.

**How many neighbors: ONE per anchor** (the cross-hop sibling in triples mode; the
same-chain sibling in pairs mode). One hard negative is the strongest, cheapest signal (a
single extra decoder forward); the margin is a hinge on ONE CE gap, not a softmax over many.
Multiple neighbors would need multiple decoder forwards (cost) and a softmax-margin
(complexity) with no evidence it beats the single hardest negative — deferred. The neighbor
ids are passed to the loss by the train loop; the loss (Task A) is neighbor-agnostic (it
just receives `logits_neg, neg_ids`).

## 3.3 Leakage note

The margin's negative pass decodes a same-chain WRONG target from `a*_i`. This does **not**
create a new future channel into the decoder: the decoder memory is STILL `a*_i` (the gold's
operator output); the negative `neg_ids` enter only as a teacher-forced AR context (the same
allowed channel as `gold_ids`). The margin shapes `a*`'s geometry (push gold-likely,
neighbor-unlikely) — it does not inject neighbor CONTENT into `a*`. Asserted at the
signature level (the loss has no memory argument other than the two logits tensors, both
produced from `a*_i`) by the §A leakage test.

---

# 4. ENTITY-WORLD v2 — generator extension (Task C, backward-compatible)

## 4.0 Scope

Extend `scripts/generate_entity_world.py` (campaign-Task-C owns this file; v4 extends it
under **backward-compatible flags** in `CONFIG`) to a larger, harder world: more types,
more entities per scene, longer chains, an optional stochastic mode with oracle-emitted
distributions, and a regenerated BPE if coverage demands. All additions are gated so the
default `CONFIG` reproduces the campaign's seed-7 data **byte-identically** (the entity
behavior-preservation gate).

## 4.1 Type library expansion (12+ train / 4 near / 4 far)

The campaign world has 7 train / 3 near / 2 far types (`TYPE_LIBRARY`). v4 expands to:
- **≥12 train types**: add (keeping the existing 7) at least 5 more across the schema
  families — e.g. `rabbit`, `goat` (living), `cactus`, `vine` (plant), `heater`, `fan`
  (device), `jar`, `crate` (container) — each a distinct response profile (not a near-OOD
  twin; genuinely new train dynamics), authored in the same `profile`/`cond` style.
- **4 near-OOD types**: keep `puppy`/`pony`/`sprout`, add 1 more (e.g. `kitten` derived from
  `cat`) — each a SMALL perturbation of a (now larger) train type's profile, `derived_from`
  + `similarity` documented.
- **4 far-OOD types**: keep `terrarium`/`robot pet`, add 2 more novel recombinations (e.g.
  `aquarium` = container+plant+living-fish-mood; `greenhouse` = device+plant) — each a novel
  schema recombination + structurally novel profile.

The split-role machinery (`_types_for_role`) already keys on `split_role`, so adding types
needs no generator-logic change — only `TYPE_LIBRARY` entries. The manifest auto-includes
them. A coverage assertion (the generator's `coverage_report`) re-checks the existing BPE;
§4.4 covers the BPE-rebuild decision.

## 4.2 More entities per scene + longer chains (backward-compatible CONFIG flags)

```python
CONFIG additions (defaults reproduce the campaign byte-for-byte):
  "entities_per_chain": (1, 2),   # campaign default — v4 entity configs use a new flag:
  "entities_per_chain_v2": (3, 5),  # NEW (3-5 entities); used ONLY when "world_version"==2
  "chain_len_min": 4, "chain_len_max": 8,        # campaign default
  "chain_len_min_v2": 6, "chain_len_max_v2": 12,  # NEW longer chains for v2
  "n_train_chains_v2": 120_000,    # ~120K train chains (campaign: 40K)
  "world_version": 1,              # 1 = campaign (bitwise); 2 = the v4 expanded world
```

`world_version` is the master switch. `world_version==1` ⟹ the generator runs EXACTLY as
the campaign (same seed-7 output — the `*_v2` keys are ignored), preserving every existing
entity result and the retraction-probe / calibration fixtures. `world_version==2` ⟹ the
expanded type library + `entities_per_chain_v2` + `chain_len_*_v2` + `n_train_chains_v2`.
`render_state` already surfaces the first 2 salient attributes per entity, so 3–5 entities
just yields longer (still bounded) state text — §4.4 checks token coverage. The
`initial_states` field (campaign §4.2) and `_labeled` twins carry over unchanged.

**Slot/entity budget interaction.** 3–5 entities per scene, each surfacing 2 salient
attributes, can approach the `n_slots=8` budget. The v4 entity configs keep `n_slots=8`
(the targeted-action mask handles which slots move); if a scene's clause count exceeds the
slot budget the slot encoder competes as designed (1-to-N is the encoder's job, Operator
ABC). The coverage report logs mean clauses/state so the slot budget can be re-sized in a
follow-up if needed — NOT changed in v4 (keeps the profile stable).

## 4.3 Optional stochastic mode (oracle emits the true distribution)

The campaign's `apply_action` is deterministic (`stochastic=False`). v4 adds a stochastic
mode where SOME conditional transitions are probabilistic, and the oracle emits the true
next-state **distribution** so diagnostics (and the retrieval ceiling, campaign §5) can
score against the real distribution, not a single sample:

```python
CONFIG additions:
  "stochastic_v2": False,          # default off (deterministic, byte-reproducible)
  "stochastic_p": 0.15,            # per-eligible-transition stochastic probability
```

When `stochastic_v2=True` (and `world_version==2`):
- A small set of transitions (authored per-type as `"stochastic": {action: [(p, effects), ...]}`)
  sample from a per-chain RNG stream (the campaign docstring already anticipates this: "a
  per-chain rng-stream is recorded so replay still matches"). The chosen branch is recorded
  so `replay_chain` reproduces it exactly (deterministic replay despite stochastic
  generation — the seed + per-chain stream IS the record).
- The `_labeled` record gains an `oracle_dist` field per step: the categorical over
  next-states the oracle would assign (the `(p, effects)` table evaluated at `s_t`),
  rendered to a list of `{"text": rendered_next, "prob": p}`. This is what the retrieval
  ceiling and a stochastic-aware MRR score against (the "true distribution" the model's
  prior should match). Deterministic transitions emit a degenerate `oracle_dist`
  (single entry, prob 1.0), so the field is always present and back-compatible.

`stochastic_v2=False` (default) ⟹ no `oracle_dist` sampling, `oracle_dist` either omitted
or degenerate — bitwise the deterministic world. The retraction probe (campaign §4) is
documented to run on the **deterministic** splits only (it needs exact replay-without-j;
stochastic transitions are excluded from the retraction split or run with the recorded
branch). The campaign eval suite's rollout-fidelity (§3c) gains nothing mandatory from
stochasticity in v4 — `oracle_dist` is emitted for a FUTURE distribution-matching metric
(v4.1), and the field is there so the data does not need regeneration to add that metric.

## 4.4 BPE coverage + regeneration

The campaign uses the existing GLUCOSE 512 BPE (`data/glucose/jepa_bpe_512.json`) or a
dedicated `data/entity_world/bpe_512.json` (campaign §2.1, `build_entity_world_bpe.py`).
v2's larger type vocabulary (new display nouns: `rabbit`, `cactus`, `heater`, `aquarium`,
...) MAY push byte-fragmentation past the `coverage_report` "POOR" threshold
(`mean_byte_frags > 4.0` or `pct_over_max > 1.0`). **Decision: the generator's existing
`coverage_report` runs on the v2 data and, if the verdict is POOR, the GPU job rebuilds the
entity BPE** (`build_entity_world_bpe.py`, pointed at the v2 `train.jsonl`) — exactly the
campaign's documented fallback. v4 entity configs point at `data/entity_world/bpe_512.json`
and `vocab_size=512` (unchanged); `max_text_tokens` STAYS 64 (3–5 entities × 2 clauses ≈ 6
short sentences; the coverage report's P95 is asserted < 64 — if it overflows, the job logs
it and the config bumps `max_text_tokens`, but the expectation is 64 holds given the
campaign's mean-26 baseline). The GPU job regenerates data + BPE before training (campaign
§2.2 job prefix), so the v2 world is byte-reproducible from seed 7.

## 4.5 Splits

Same four splits as the campaign (`train`, `test_iid`, `test_ood_near`, `test_ood_far`),
now over the expanded type library. Distinct RNG streams (`seed+1..+4`) per split keep them
reproducible and state-disjoint, exactly as the campaign `main()`. The `*_labeled` twins +
`initial_states` + (stochastic) `oracle_dist` carry over. `n_test_chains` stays 2,000 per
test split; `n_train_chains_v2=120K`. The manifest's `splits` block auto-reflects the new
type roster.

---

# 5. CONFIG SCHEMA + CONFIGS

## 5.1 Schema additions (all default to v3/entity behavior)

```python
# ModelHParams (twm/jepa/__init__.py — B-owned schema; A delivers the field spec):
use_targeted_actions: bool = False        # master switch for the per-slot target mask (§1)
@dataclass
class TargetedConfig:                      # NEW nested block, model.targeted
    mask_hidden: int = 64                  # mask-head hidden width (default 2*dn for nano)
targeted: TargetedConfig = field(default_factory=TargetedConfig)

# LossConfig:
w_diff:   float = 1.0                      # diff-CE weight on diff tokens; 1.0 ⟹ uniform (§2)
w_margin: float = 0.0                      # token hard-neg margin weight; 0.0 ⟹ off (§3)
margin:   float = 0.5                      # hinge margin in nats (§3.1)
w_mask_prior: float = 0.1                  # prior-mask distillation weight (§1.3)

# DataConfig:
#   (no new field — w_diff drives weighting; the dataset ALWAYS computes *_diff_w when in a
#    chain mode, cheap, so flipping w_diff needs no data rebuild. world_version is a
#    generator-side CONFIG, not a DataConfig field.)
```

All defaults reproduce v3/entity bitwise: `use_targeted_actions=False` (no mask head, `g≡1`,
v3 apply-all), `w_diff=1.0` (uniform CE), `w_margin=0.0` (no margin), so an unmodified v3 or
entity config builds an identical model and identical loss. `config.py` parses
`model.targeted` like the other nested blocks (`_build_model` pops it before the flat
overlay); `_build_loss` parses the new scalar `w_diff`/`w_margin`/`margin`/`w_mask_prior`
fields via `_only_known`.

## 5.2 The 12 configs — `{v4, v4_blackbox} × {s0,s1,s2}` + smokes, on world-v2

Two families × three seeds + a smoke each (8 training configs + 4 smokes = 12), all on the
**world-v2** data (`world_version=2`, regenerated by the GPU job). All inherit the entity
campaign's blocks (entity data §2.1, v3 decoder d128/2L, InfoNCE+unroll `w_nce=0.25`
`mode="triples"` `hop_weights=[1.0,0.5]`, `eval.entity_world.enabled=true`,
`use_norm_budget=true`) and ADD the v4 flags.

**Family `v4` (structured operator + targeted mask + diff-CE + margin):**

| config | seed | operator_group | use_targeted_actions | use_norm_budget | polar | w_diff | w_margin | out_dir |
|---|---:|---|---|---|---|---:|---:|---|
| `configs/jepa/jepa_v4_s0.json` | 0 | rotation_scale | true | true | true | 4.0 | 0.25 | `results/jepa_v4_s0` |
| `configs/jepa/jepa_v4_s1.json` | 1 | rotation_scale | true | true | true | 4.0 | 0.25 | `results/jepa_v4_s1` |
| `configs/jepa/jepa_v4_s2.json` | 2 | rotation_scale | true | true | true | 4.0 | 0.25 | `results/jepa_v4_s2` |
| `configs/jepa/jepa_v4_smoke.json` | 0 | rotation_scale | true | true | true | 4.0 | 0.25 | `results/jepa_v4_smoke` |

**Family `v4_blackbox` (gated_mlp + mask NO-OP + diff-CE + margin):**

The black-box has no slot-structured operator, but the targeted mask is a MODEL-level gate
(`_apply_action` wraps the operator output), so `use_targeted_actions=true` DOES apply the
convex combination `g⊙(MLP) + (1−g)⊙k` even for the gated MLP — the mask is operator-
agnostic. To honor "mask-noop" (the black-box's contrast is "no structure"), the black-box
family sets `use_targeted_actions=false` (mask OFF — apply-all, the v3 black-box behavior),
so the contrast is: structured operator WITH targeted mask vs black-box WITHOUT. The mask is
part of the structured-prior story; the black-box deliberately lacks it. The flag name in
the table is `mask-noop` = `use_targeted_actions:false`.

| config | seed | operator_group | use_targeted_actions | use_norm_budget | polar | w_diff | w_margin | out_dir |
|---|---:|---|---|---|---|---:|---:|---|
| `configs/jepa/jepa_v4_blackbox_s0.json` | 0 | gated_mlp | false | true | false | 4.0 | 0.25 | `results/jepa_v4_blackbox_s0` |
| `configs/jepa/jepa_v4_blackbox_s1.json` | 1 | gated_mlp | false | true | false | 4.0 | 0.25 | `results/jepa_v4_blackbox_s1` |
| `configs/jepa/jepa_v4_blackbox_s2.json` | 2 | gated_mlp | false | true | false | 4.0 | 0.25 | `results/jepa_v4_blackbox_s2` |
| `configs/jepa/jepa_v4_blackbox_smoke.json` | 0 | gated_mlp | false | true | false | 4.0 | 0.25 | `results/jepa_v4_blackbox_smoke` |

Smoke variants: `data.max_chains=64`, `optim.epochs=3`, `eval.every_epochs=1`,
`eval.entity_world.subsample=64`, `n_rollout_chains=16` — fast CI of both operator
families, the mask gate on/off, diff-CE, margin, and the v2 world + eval suite. Common
`model.targeted={mask_hidden:64}`, `loss.margin=0.5`, `loss.w_mask_prior=0.1`. The 12 configs
are written by **Task C** (the generator/config/diagnostics task).

## 5.3 Param deltas (nano, dn=32, M=8, V=8)

| component | params | when built |
|---|---:|---|
| v3 + entity baseline (norm budget on) | ≈ 657K + `scale_readout_proj` (1056) | always (entity) |
| TransitionEncoder mask head (§1.1) | ≈ **6,273** | `use_targeted_actions=true` only |
| PriorHead mask head (§1.3) | `dn·mh + mh + mh·1 + 1 = 32·64+64+65 = 2,177` | `use_targeted_actions=true` only |
| diff-CE (§2) | **0** (per-token data weights, no params) | — |
| token margin (§3) | **0** (loss only, reuses the decoder) | — |
| **v4 total new trainable** | **≈ 8,450** (mask heads) | targeted only; else **0** |

The mask heads are the ONLY new parameters; diff-CE and the margin are param-free. Default
configs (mask off) add **zero** params over v3/entity — the bitwise gate. The structured v4
family adds ≈8.5K params (mask heads) on top of the ≈658K entity model (≈1.3%).

---

# 6. WORK BREAKDOWN — three disjoint-file tasks

Strict file ownership, no overlaps. Each ships its own tests. The schema files
(`__init__.py`, `config.py`) are **B-owned** (most schema surface), with A and C delivering
field specs as one-directional diffs B applies — the same handoff convention as v3/entity.

### Task A — `transition.py` mask head + operator/model targeted-apply wiring + losses signatures + tests

**Files:** `src/twm/jepa/transition.py` (mask head on `TransitionEncoder` + `PriorHead`),
`src/twm/jepa/model.py` (`_apply_action` gated convex-combination + `s_acc`/inverse gating +
`g_hard` in the return dicts + `forward`/`forward_unroll`/`rollout`/`step_latent`/
`undo_latent` wiring + `build_jepa_model_v2` mask-head construction), `src/twm/jepa/losses.py`
(the `token_ce` `token_weights` arg + `token_margin` + `_per_example_ce` +
`JEPALossV2.forward` `token_weights`/margin passthrough), `tests/jepa/test_transition.py`,
`tests/jepa/test_model.py`, `tests/jepa/test_losses.py` (extend). Delivers the
`use_targeted_actions`/`TargetedConfig`/`w_diff`/`w_margin`/`margin`/`w_mask_prior` field
specs to B.

- `transition.py`: `TransitionEncoder.forward_mask(k, k_tgt) -> g_logits (B,M)` (§1.1);
  `PriorHead.forward_mask(k) -> g_prior_logits (B,M)` (§1.3); both built only when
  `use_targeted_actions=True`.
- `model.py`: `_apply_action` gains `g_logits=None`, applies the convex combination with
  straight-through hard threshold (§1.2), gates `scale_delta` by `g` (§1.2 — identity slots
  accumulate ZERO scale), and the inverse path (`undo_latent`) gates by the stored `g_hard`
  (§1.5). `forward`/`forward_unroll`/`rollout` compute `g_logits` from the posterior
  (`forward_mask`) and thread `g_hard` into the return dicts; `build_jepa_model_v2` reads
  `getattr(m, "use_targeted_actions", False)` and builds the mask heads. `model._target_slots`
  helper returns the detached EMA target slot nouns for the mask input.
- `losses.py`: `token_ce(..., token_weights=None)` weighted-mean (§2.2, None ⟹ v3 bitwise);
  `token_margin(logits_gold, gold_ids, logits_neg, neg_ids, pad_id, margin)` (§3.1);
  `_per_example_ce`; `JEPALossV2.forward` gains `token_weights=None`, `margin_logits_neg=None`,
  `margin_neg_ids=None` and computes `L_margin` only when `w_margin>0` (the train loop passes
  the neighbor logits/ids; the loss is neighbor-agnostic).
- Tests: mask head shape/finite + `use_targeted_actions=False` ⟹ no head + v3-bitwise apply
  (regression vs stored baseline); gated apply identity-slot (`g=0`) returns `k_i` exactly +
  `scale_delta` zero on identity slots; **round-trip** `undo_latent(step_latent(k,v,g),
  g_hard=g) ≈ k` on a mixed mask; straight-through eval hard / train soft; **leakage**
  (§1.4): perturb `s_{t+1}` content with diff-set fixed ⟹ `g_hard` + identity-slot memory
  unchanged, bit-count ceiling `ceil(log2 V)+M=11` logged; `token_ce(token_weights=None)`
  bitwise == v3; weighted CE focuses (constructed weights move the loss); `token_margin`
  hinge sign/zero-when-satisfied; `w_diff=1.0`+`w_margin=0.0` ⟹ exact v3 loss total.

### Task B — `losses.py` wiring is A's; `data.py` diff/weights + dataset support + train-loop wiring + schema + tests

**Files:** `src/twm/jepa/data.py` (`_diff_weights` + `*_diff_w` tensors/keys in pairs &
triples mode + neighbor-id bookkeeping), `src/twm/jepa/__init__.py` +
`src/twm/jepa/config.py` (the B-owned schema: `use_targeted_actions`, `TargetedConfig`,
`w_diff`/`w_margin`/`margin`/`w_mask_prior` — applying A's field specs), `scripts/train_jepa_v2.py`
(thread `*_diff_w` into the per-hop `token_ce`; build the same-chain/cross-hop NEIGHBOR
`tgt_ids` and run the decoder's second pass for `token_margin`; sum `L_margin`/`L_diff` over
hops with `hop_weights`; logging adds per-hop diff-CE + margin),
`tests/jepa/test_data.py` + `tests/jepa/test_config.py` (extend).

- `data.py`: `_diff_weights` (SequenceMatcher token alignment, §2.1); `_tgt_diff_w` (pairs),
  `_s1_diff_w`/`_s2_diff_w` (triples); `__getitem__`/`get_batch` return the `*_diff_w` keys.
  Always computed (cheap); all-ones when the diff is empty (degenerate ⟹ uniform). Pairs &
  triples both supported; pad positions weight 0.0.
- `train_jepa_v2.py`: pass each hop's `*_diff_w` to that hop's `token_ce`; build the cross-hop
  neighbor (triples: `s2` for hop-1, `s1` for hop-2; pairs: same-chain sibling via
  `chain_ids`), run `decoder(a*_hop, neighbor_ids, neighbor_pad)` for the margin, call
  `token_margin`, sum with hop weights. `frac_anchors_with_chain_neg` logged (pairs mode).
- schema: apply A's `ModelHParams.use_targeted_actions` + `TargetedConfig` + the four loss
  scalars; `config.py` parses `model.targeted` and the loss scalars.
- Tests: `_diff_weights` marks replace/insert as diff, equal as boilerplate, handles ins/del,
  pad weight 0; `*_diff_w` shapes/keys in both modes; all-ones when no diff; the 12 configs
  parse and build (incl. `use_targeted_actions`/`w_diff`/`w_margin`); a tiny end-to-end smoke
  (`max_chains=8, epochs=1`, `use_targeted_actions=true`, `w_diff=4`, `w_margin=0.25`,
  triples) runs the unroll + diff-CE + margin without error.

### Task C — `generate_entity_world.py` v2 extension + 12 configs + diagnostics (target-recovery + separation-AUC port) + tests

**Files:** `scripts/generate_entity_world.py` (the world-v2 `world_version`/`*_v2`/
`stochastic_v2`/`oracle_dist` extension + ≥12/4/4 type library, §4 — backward-compatible),
the 12 config JSONs (§5.2), `src/twm/jepa/diagnostics.py` (`_target_recovery` + the
`_separation_auc` port from `bb01bfd`/`scripts/jepa_separation_diag.py` as an every-N-epochs
metric, wired into `eval_entity_world`), `tests/jepa/test_generate_entity_world.py` +
`tests/jepa/test_diagnostics.py` (extend).

- `generate_entity_world.py`: `world_version` master switch (default 1 ⟹ campaign byte-
  identical); v2 type library (≥12 train / 4 near / 4 far); `entities_per_chain_v2`,
  `chain_len_*_v2`, `n_train_chains_v2`; optional `stochastic_v2` + per-type `"stochastic"`
  tables + `oracle_dist` in `_labeled` (degenerate when deterministic); `coverage_report`
  re-run + POOR ⟹ rebuild-BPE note (§4.4). `replay_chain`/`apply_action`/`initial_states`
  carry over.
- diagnostics: `_target_recovery(model, chains, ...)` (§1.6: F1/NMI of inferred mask vs
  oracle moved-entity, mask-density, shuffle baseline), called from `eval_entity_world` ONLY
  when `getattr(model,"use_targeted_actions",False)`; `_separation_auc(model, chains, ...)`
  ports the `bb01bfd` `jepa_separation_diag.py` AUC (linear-probe AUC on hard pools, the
  three encoder variants) into a per-epoch metric `ent_separation_auc` (+ `_ema`/`_online`/
  `_slot_mean`), the **v4 success criterion** (0.53→>0.7). Both write `ent_`-prefixed
  scalars + artifacts; both no-op cleanly on a black-box / mask-off model.
- Tests: world-v2 generates ≥12 train types + 3–5 entities + chains 6–12 + `oracle_dist`
  present (degenerate when deterministic); `world_version=1` byte-reproduces the campaign
  (golden-hash a tiny seeded sample); `_target_recovery` F1==1.0 on a synthetic perfect-mask
  fixture + shuffle≈0; `_separation_auc` returns the AUC scalars on a tiny labeled fixture +
  no-ops (NaN/skip) when `use_targeted_actions=False`; the 12 configs parse.

**Disjoint-file guarantee:** A=`transition.py`,`model.py`,`losses.py`(+their tests),
delivers field specs; B=`data.py`,`train_jepa_v2.py`,the schema files
(`__init__.py`/`config.py`)(+their tests); C=`generate_entity_world.py`,the 12 configs,
`diagnostics.py`(+their tests). `losses.py`→A (it owns the loss math + signatures);
`data.py`+train script→B (data weights + orchestration); `diagnostics.py`+generator+configs→C.
The schema files are B-owned; A and C hand B their dataclass field specs as reviewed diffs B
applies (the only cross-task coordination, one-directional into B).

---

## Summary of resolved decisions

1. **Targeted latent actions:** posterior `TransitionEncoder.forward_mask(k, k_tgt)` emits a
   per-slot sigmoid mask `g (B,M)` (≈6.3K params, built only when on); `model._apply_action`
   applies the convex combination `a = g⊙(B_v k) + (1−g)⊙k` with a straight-through hard
   threshold at eval (identity slots return `k_i` exactly). The norm-budget `scale_delta` is
   **gated by `g`** so identity slots accumulate ZERO scale; the inverse is gated by the
   stored hard `g` so the round-trip stays exact. Leakage: the future→decoder channel grows
   from 3 bits (verb) to **`ceil(log2 V)+M = 11` bits** (verb + hard per-slot mask); the
   mask adds LOCATION bits, not CONTENT (hard threshold bounds it to 1 bit/slot), audited +
   bit-count-asserted. PriorHead extends with a mask head distilled from the posterior
   (`w_mask_prior=0.1`). Config `use_targeted_actions`, default false ⟹ v3 bitwise.
2. **Diff-weighted CE:** `data.py` computes per-token weights via SequenceMatcher token
   alignment (replace/insert → `w_diff`, boilerplate → 1.0, pad → 0); `token_ce` gains a
   `token_weights` arg (None ⟹ v3-bitwise mean). `loss.w_diff` default 1.0 ⟹ uniform.
3. **Token hard-negative margin:** per-pair hinge `max(0, m − (CE(neighbor|a*) −
   CE(gold|a*)))`, `m=0.5` nats, ONE neighbor (cross-hop sibling in triples, same-chain in
   pairs), a second decoder pass on the SAME `a*`. The encoder-AIM fix (separation 0.53 →
   >0.7). `loss.w_margin` default 0.0 ⟹ off; leakage-clean (same `a*` memory).
4. **Entity-world v2:** `world_version` flag (default 1 ⟹ campaign byte-identical); v2 =
   ≥12 train / 4 near / 4 far types, 3–5 entities/scene, chains 6–12, ~120K train, optional
   `stochastic_v2` with oracle `oracle_dist` emission, BPE rebuilt if coverage POOR.
5. **Diagnostics:** target-recovery (F1/NMI vs oracle moved-entity + mask-density + shuffle,
   only when mask on) joins action-NMI; the `bb01bfd` separation-AUC ported into
   `diagnostics.py` as an every-N-epochs metric `ent_separation_auc` — the v4 success
   criterion (0.53 → >0.7).
6. **Work breakdown:** A = `transition.py` mask + `model.py` targeted-apply wiring +
   `losses.py` signatures (+tests); B = `data.py` diff-weights + train-loop margin/diff
   wiring + B-owned schema + 12-config parse (+tests); C = `generate_entity_world.py` v2 +
   12 configs + `diagnostics.py` (target-recovery + separation-AUC port) (+tests). No file
   overlaps. **v4.1 (deferred):** stratified non-abelian operator blocks, sized from the
   targeted-action + retraction-bracket results.
