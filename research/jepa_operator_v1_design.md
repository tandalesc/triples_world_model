# JEPA Operator World Model — Final v1 Design

**Status:** APPROVED FOR IMPLEMENTATION · **Branch:** `feature/glucose-converter` · **New code:** `src/twm/jepa/`
**Base design:** D3 (product-first) with the algebra ladder and pre-build gate from D1 and the export/efficiency path from D2.
**v1 dataset:** GLUCOSE causal chains (local jsonl).
**Operator (DECIDED by empirical fit, overrides the pure-rotation default):** block-diagonal 2×2 **ROTATION+SCALE** (`r·R(θ)` per block), structural inverse `R(θ)ᵀ/r`.

This document is implementation-complete. Every number is reconciled with the others; every flaw the three judges found is fixed inline and marked **[FIX]**; grafted ideas are marked **[GRAFT]**. No further design decisions are required before coding.

---

## 0. Thesis

World-state is a **set of M slots**, each a pair `(k_i, v_i)`:
- `k_i ∈ A` — the **free factor**, a point in a shared `d_noun`-dim noun space (continuous).
- `v_i ∈ {0..V-1}` — a **discrete verb**, an index into a codebook of operators.

The operator transforms the noun: `a*_i = B_{v_i} k_i`. We **replace every loss we can with structure**. The inverse is structural, orthogonality of the rotation part is structural, so we keep only three losses, each irreducible. We train **purely in latent space** — no token decoder — against an **EMA-of-encoder** target.

The product north star (D3): a Tamagotchi/Nintendogs-grade browser pet whose persistent per-entity state is `k_i` in JS `gameState`, mutated each tick by `B_v`, with an exact **undo** via `B_vᵀ`. The pet framing drives the API shape but **is not claimed to be realized by v1 training data** — see §0.1 and Open Question O1.

### 0.1 Honest scope of the pet framing [FIX — Judge 0 D3 flaw]

GLUCOSE chains are **third-person causal narrative** (`"Someone_A puts Something_A on Something_B."`), not user-action/entity-state pairs. The verb codebook will therefore learn **causation-type clusters** (physical / mental / social causation), **not** UI action verbs (feed/pet/play/scold). v1 delivers: the algebraic engine, the persistent-`k` + `step_latent`/`undo_latent` API, the export path, and a verb codebook over GLUCOSE causation types. Mapping UI actions to verbs requires a small action-labeled dataset or fine-tune (O1). The `step_latent`/`undo_latent`/`structural_sanity` API and the JS export are real and testable in v1; the "feed maps to verb 3" lookup is deferred.

### 0.2 Structure-replaces-loss table

| Property | Usual enforcement | Our enforcement |
|---|---|---|
| Invertibility | consistency / cycle loss | **structural**: `B⁻¹ = diag_blocks(R(θ_b)ᵀ / r_b)` |
| Rotation orthogonality | orthogonality penalty | **structural**: `cos²+sin²=1` per block |
| Bottleneck non-collapse | VAE KL / spectral / CKA | **deprecated** (CLAUDE.md "Drop the VAE"); replaced by SIGReg on nouns |
| Noun-space isotropy | — | `L_sigreg` (irreducible) |
| Verb non-triviality | — | `L_div` (irreducible) |
| Prediction signal | — | `L_pred` MSE to EMA target (irreducible) |

**Note on norm preservation:** because v1 uses rotation+**scale** (`r ≠ 1`), the operator is **not** norm-preserving. Norm preservation is no longer a structural guarantee — it becomes a *monitored quantity* (scale-drift diagnostic, §5) and motivates the noun-space normalization policy (§1.5). This is the central change the empirical fit forces.

---

## 1. The Operator Algebra (centerpiece)

### 1.1 The empirical verdict (OVERRIDES the pure-rotation default)

The pre-build operator-group-fit experiment (`scripts/operator_group_fit.py`, report `research/operator_group_fit.md`, results `results/operator_fit/`) fit candidate families on 12,000 GLUCOSE `(state_t, state_t+1)` MiniLM-embedding pairs. Held-out residual MSE (global; identity baseline **0.929**):

| Family | Global residual | Clustered residual (verb-codebook proxy, k=16) |
|---|---:|---:|
| identity | 0.929 | — |
| **U(1) pure rotation** | **0.929** (no better than identity) | 0.916 |
| full orthogonal | 0.927 | 0.997 (worse) |
| **block-diag ROTATION+SCALE** | **0.709** | **0.693** (improves under clustering) |
| general linear | 0.643 | **1.097** (blows up — O(d²) overfit) |

Singular-value analysis: cause→effect transitions **contract almost universally** (median σ ≈ 0.4; 60–88% of σ < 1). Norm-preserving operators structurally cannot express the dominant signal. **Rotation+scale is the only family robust under per-cluster fitting.** Reversibility modulates the spectrum only second-order (irreversible classes have higher σ, sometimes > 1 = expansion), so **do NOT branch the operator family on reversibility** — a single rotation+scale family covers both, with `r` free per block.

**MANDATE:** v1 operator is **block-diagonal 2×2 rotation+scale**. Pure rotation (`r=1` frozen) is retained only as a config ablation. SO(n)-via-Cayley remains an interface stub.

**Caveat carried to O3:** the fit was measured in MiniLM sentence-embedding geometry, **not** the learned bottleneck. Re-verify on learned nouns once the testbed trains (the operator-fit script runs against the trained encoder as a second pass — §7).

### 1.2 The expressivity ladder (documentation contract) [GRAFT D1]

```
U(1)^(d/2)   ⊂   C*^(d/2)        ⊂   GL(d)
pure rotation    rotation+scale       general linear
(reversible)     (can forget) ← v1    (overfits, out of scope)
```

What v1 rotation+scale **can** express: norm change (contraction/expansion via `r`), rotation, antisymmetry, abelian composition, and an **exact structural inverse**. What it **cannot**: cross-block coupling beyond `b`, non-abelian composition (enters only via the v2 `velocity(k,v)` hook), and **1-to-N branching** — handled by the encoder, not the operator (§1.4).

### 1.3 Parameterization (file: `operator.py`)

Per verb `v`, per 2×2 block `b`: a learned angle `θ_{v,b}` and an **unclamped positive scale** stored as **`log r_{v,b}`** (so `r = exp(log r) > 0` always, no clamp, gradient-stable).

```
B_v = diag_blocks( r_b · R(θ_b) ),   R(θ) = [[cosθ, -sinθ],[sinθ, cosθ]]
B_v⁻¹ = diag_blocks( R(θ_b)ᵀ / r_b )      # STRUCTURAL — exact, no consistency loss
```

- nano: `b = 2` (pure 2×2 blocks). **`apply` is RoPE-style elementwise — no matrix materialized** [GRAFT D2, Judge 1]: `(x', y') = (r·(x cosθ − y sinθ), r·(x sinθ + y cosθ))`. Expressible in the pet-sim JS primitive set (elementwise multiply + add). O(d_noun) per slot.
- mini: **`b = 2` as well** [FIX — Judge 2 D3 flaw]. The earlier `b=4`/`b=8` "export (cos,sin) pairs" claim was wrong: a 4×4 `exp(skew)` bakes to a dense 4×4 matrix (16 floats), needs `matrix_exp` in fp32, and a 4×4 matvec in JS — not RoPE-style pairs. v1 keeps `b=2` everywhere for a uniform, JS-trivial, INT8-friendly apply. Cross-block coupling (`b>2`) is deferred to v2 along with the velocity field; it is **not** needed for the rotation+scale signal the fit identified (which lives in per-block scale, not cross-block rotation).
- Param count per verb: `2 · (d_noun/2) = d_noun` (one angle + one log-scale per block). Codebook = `V · d_noun` params. (nano: `8·32 = 256`; mini: `16·32 = 512`.)
- **fp32 under `autocast(enabled=False)`** for the operator math (mirrors the VQ bf16 gotcha; cos/sin and exp are bf16-unstable for large magnitudes).
- init: `θ ~ Uniform(−π/2, π/2)` excluding `|θ|<0.1` (avoid identity); `log r ~ Normal(0, 0.1)` (near 1.0, gradient discovers contraction).

### 1.4 Architectural contracts (source-level docstrings in `operator.py`) [GRAFT D1, Judge 0]

- **1-to-N is the encoder's job, not the operator's.** `B_v k` is deterministic. Multiple outcomes from one state are routed into **different slots** by the slot encoder's competitive assignment. The operator never solves 1-to-N. *(Verbatim docstring required.)*
- **Abelian commutativity is silent.** `B_u B_v = B_v B_u` (diagonal-block scale + same-plane rotation commute). Harmless in v1 (no multiturn, no composition loss — anti-goals). Non-abelian expressiveness enters only via the v2 state-dependent `velocity(k,v)`.

### 1.5 Noun-space normalization / closure policy [FIX — required by `r ≠ 1`]

With scale, `a* = B_v k` can leave the radius band of `k`, so closure under "the same normalization as `k`" must be defined explicitly:

- **Policy (v1): raw-space closure with separate scale tracking.** Nouns `k` are produced by the noun head **without** a final L2-normalization onto the unit sphere (see §1.6 and the SIGReg fix §3). The readout (§2) consumes `a*` in raw space. `L_pred` is computed in raw noun space. We do **not** renormalize `a*` back to `‖k‖`, because that would discard exactly the contraction signal the fit says is dominant.
- **Scale-drift diagnostic (mandatory, §5):** monitor the distribution of `log r` per verb and the running mean of `‖a*‖/‖k‖`. Runaway contraction (`log r → −∞`) collapses nouns toward the origin and interacts badly with SIGReg (which wants unit variance). If `mean log r < −1.0` sustained, the trainer logs a WARN and the recommended remedy is raising `w_sigreg` or adding a soft `‖log r‖₂` penalty (off by default, config `w_scale_reg=0.0`).
- **SIGReg/scale interaction note:** SIGReg shapes the *distribution* of nouns toward isotropic Gaussian (zero-mean, unit-var per projection), not onto a sphere. Scale-driven contraction shrinks variance; SIGReg pushes back. The two are in tension by design — the scale-drift diagnostic is how we watch that tension stay healthy.

### 1.6 The Operator interface (fixed; `apply` / `inverse_apply` / `velocity` / T-step seam)

```python
class Operator(nn.Module, ABC):
    def apply(self, k, v):           # (B,M,d_noun),(B,M) -> (B,M,d_noun)   a* = B_v k
    def inverse_apply(self, a, v):   #                                       k  = B_v^{-1} a  (STRUCTURAL)
    def velocity(self, k, v):        # (B,M,d_noun),(B,M) -> generator action [v2 hook]
    def integrate(self, k, v, T=1):  # T-step seam; T hard-set to 1 in v1
    def structural_sanity(self, v):  # -> {"bbT_err": float, "inv_err": float}  [GRAFT D3, Judge 2]
    def bake(self) -> dict           # export-ready (cos, sin, r) per verb for JS / INT8
    @property
    def n_verbs(self): ...
```

**v1 `RotationScaleOperator(Operator)`** (`C*^(d/2)`): params `theta (V, d_noun//2)`, `log_r (V, d_noun//2)`. `apply` gathers `theta[v], log_r[v]`, computes RoPE-style per pair times `exp(log_r)`. `inverse_apply` negates `theta` and negates `log_r` (i.e. `R(θ)ᵀ/r`). Exact — **no consistency/invertibility loss**.

**T-step seam (v2 — SPECIFIED, NOT BUILT):**
```python
def integrate(self, k, v, T=1):          # T hard-set to 1 in v1 (config n_steps_T)
    x = k
    for _ in range(T):
        G = self.velocity(x, v)          # v1: static block generator; v2: state-dependent MLP -> skew
        x = self._integrate_step(x, G, dt=1.0 / T)   # T=1 -> single exp map == apply()
    return x
```
At `T=1`, `_integrate_step` is the single exponential map = v1 behavior **exactly**. Endpoint-only supervision (`L_pred` on final `a*`) is structurally identical to Neural-ODE endpoint loss. The loop and per-step generator hook are present and dormant.

**Stubs:** `RotationOperator` (pure rotation, `log_r=0` frozen — config ablation), `SOnCayleyOperator` (`raise NotImplementedError`).

---

## 2. Forward pass — exact shapes & reconciled param tables

Notation: `B`=batch, `T_text`=text tokens, `M`=slots, `d`=`d_model`, `dn`=`d_noun`, `V`=verbs.

```
text_ids (B,T_text) ── frozen token_emb + text_pos ──► (B,T_text,d)
   │ SlotEncoder: L_text shared self-attn layers (ALBERT-tied) → context
   │ slot extraction: M learned queries cross-attend to context, then
   │                  n_iters=3 slot self-attn coordination  [FIX — Judge 1 D2: coordination is REQUIRED]
   ▼
slots (B,M,d)
   ├── NounHead (d→dn), then standardize (NOT L2-norm) ──► k (B,M,dn)   [FIX — Judge 0/1/2 SIGReg]
   └── VerbHead (d→V) ──► verb_logits (B,M,V) ──► Gumbel-softmax ──► verb (B,M)  [FIX — VerbHead gradient]
                                                        │
Operator.apply(k, verb)  a* = B_v k  (T=1) ──► a* (B,M,dn)
   ▼
Readout (attention-pool over a*, query dim = dn) ──► zhat (B,dn)
EMA_encoder(next_text) ──► z (B,dn)  (stop-grad, raw noun-pool)
L_pred = MSE(zhat, z.detach())
```

### Encoder: ALBERT-tied self-attn [GRAFT D1, Judge 1]

`L_text` text self-attention layers share **one** weight block applied `L_text` times (ALBERT). Counted **once** in param tables. nano `L_text=2` (1 shared block); mini `L_text=3` (1 shared block, applied 3×) to hit budget [FIX — mini budget, §below].

### Slot coordination is mandatory [FIX — Judge 1 D2 flaw]

Single-pass cross-attention with no coordination gives zero slot-competition pressure → stripe collapse with no gradient signal to recover. v1 **requires** `n_iters=3` slot self-attn coordination after cross-attention (slot-attention style). This is the routing mechanism; the residual-vs-slots diagnostic (§5) verifies it works.

### nano profile (`jepa_nano`): d=64, dn=32, M=8, V=8, b=2, L_text=2 (ALBERT-tied, 1 block), heads=4, T_text=64, vocab=512 (domain BPE)

| Module | Shape out | Params | Notes |
|---|---|---:|---|
| token_emb (frozen, domain BPE) | (B,64,64) | 512·64 = 32,768 **frozen, NOT exported as dense** | quantized/shared at export |
| text_pos_emb | — | 64·64 = 4,096 | [FIX — Judge 2: pos emb was omitted by D1] |
| SlotEncoder self-attn (1 shared block, applied ×2) | (B,64,64) | self-attn 4·64²=16,384 + FFN(d_ff=128) 2·64·128=16,384 + 2 LN = 16,512+16,384+256 ≈ **33,152** | ALBERT-tied, counted once |
| slot queries + per-slot μ/σ init | (B,8,64) | M·d + 2·M·d = 512+1,024 = **1,536** | |
| slot cross-attn | (B,8,64) | 4·64² = **16,384** | per-encoder (not tied) |
| slot self-attn coordination (×3 iters, shared) | (B,8,64) | 4·64² = **16,384** | shared across iters |
| NounHead (d→dn=32) | (B,8,32) | 64·32 = **2,048** | linear; standardize after (0 params) |
| VerbHead (d→V) | (B,8,8) | 64·8 = **512** | |
| **Operator codebook** (θ + log r) | — | V·dn = 8·32 = **256** | |
| Readout attn-pool (query dim dn=32) | (B,32) | cond_query 32 + 4·32² = **4,128** | over a* (dn=32) |
| Predictor MLP (dn→dn) | (B,32) | 2·32² = **2,048** | optional, kept |
| **Trainable, non-embedding** | | **≈ 76.7K** | |
| **Exported (online, INT8 weights + shared/quantized emb)** | | **see §8** | |

**Reconciled:** nano core (excl. frozen token_emb) ≈ **77K trainable params** [FIX — all three designs under-counted; this is the honest number, matching Judge 0/1/2's recomputations]. FP32 JSON at 10.5 bytes/param ≈ 808KB — **over the 303KB envelope**, so nano ships **INT8 weight-only** (§8) ≈ **~95KB**, comfortably inside. The frozen token_emb is exported as a shared/quantized table, not dense FP32.

### mini profile (`jepa_mini`): d=128, dn=32, M=12, V=16, b=2, L_text=3 (ALBERT-tied, 1 block), heads=8, T_text=64, vocab=512

| Module | Params | Notes |
|---|---:|---|
| token_emb (frozen) | 512·128 = 65,536 | frozen |
| text_pos_emb | 64·128 = 8,192 | |
| SlotEncoder self-attn (1 shared block, applied ×3) | 4·128²=65,536 + FFN(d_ff=512) 2·128·512=131,072 + LN 512 ≈ **197,120** | ALBERT-tied, counted once |
| slot queries + init | M·d + 2·M·d = 1,536+3,072 = **4,608** | |
| slot cross-attn | 4·128² = **65,536** | |
| slot self-attn coordination (×3, shared) | 4·128² = **65,536** | |
| NounHead (d→dn=32) | 128·32 = **4,096** | |
| VerbHead (d→V) | 128·16 = **2,048** | |
| **Operator codebook** (θ + log r) | V·dn = 16·32 = **512** | [FIX — Judge 0/2: uses dn=32 NOT d; correct count] |
| Readout attn-pool (query dim dn=32) | cond_query 32 + 4·32² = **4,128** | [FIX — Judge 0/2: readout over dn=32 NOT d] |
| Predictor MLP (dn→dn) | 2·32² = **2,048** | |
| **Trainable, non-embedding** | **≈ 345K** | |

**Reconciled:** mini core ≈ **345K trainable** (excl. frozen emb). This is **below the 0.5–2M band** with `L_text=3` shared. To land **mid-band (~0.9M)**, set `L_text=3` **untied** (3 independent blocks): 3·197,120 = 591,360 for self-attn, total ≈ **740K**; or `L_text=4` untied ≈ **940K**. **DECISION:** mini ships `L_text=4` **untied** (≈0.94M) as the research rig. [FIX — Judge 1: the optional +395K TransformerDynamics coordinator is REMOVED, not left optional. Depth comes from honest untied encoder layers, not a repurposed dynamics core.] FP16 export ≈ 1.9MB (progressive-enhancement, not the primary nano pet target).

---

## 3. Losses & default weights (file: `losses.py`)

`L = w_pred·L_pred + w_sigreg·L_sigreg + w_div·L_div`, defaults `w_pred=1.0, w_sigreg=0.05, w_div=0.1`. (`w_scale_reg=0.0`, off by default — §1.5.)

### L_pred
`MSE(zhat, z.detach())`, `z = EMA-encoder(next_text)` pooled in **raw noun space** (not rotated, not L2-normed) so MSE measures operator quality, not double-rotation/normalization noise. The learning signal.

### L_sigreg [FIX — Judge 0/1/2 shared flaw: do NOT L2-normalize before the GoF test]
SIGReg on the batch of nouns `(B·M, dn)`. **Standardize, do not project to the sphere.** L2-normalization onto the unit sphere makes every random 1D projection have std `1/√dn` (≈0.18 for dn=32), so the Epps-Pulley GoF calibrated to `N(0,1)` fires constantly and provides **no gradient** [Judge 0, the decisive shared flaw]. Correct preconditioning:
- **Center and scale by batch statistics** (zero-mean, unit-variance per dim) — *not* per-vector L2-norm — before the GoF test. Equivalently, rescale projected samples by their batch std (or apply the test to standardized nouns).
- Sliced isotropic-Gaussian test: `n_slices=256` random unit directions, project, Epps-Pulley characteristic-function GoF vs `N(0,1)`, `n_knots=17` on `[0,3]`, trapezoid integration.
- **Never** applied to verbs (discrete, non-Gaussian by design).
- Precondition documented in source [GRAFT D1, Judge 2].

This fix is load-bearing: it is why we **drop the final L2-normalization from the noun head** (§2) and adopt raw-space closure (§1.5).

### L_div [FIX — VerbHead must receive a learning gradient]
Verb non-triviality, two terms over **Gumbel-softmax** verb assignment:
1. **usage entropy**: `−Σ p(v) log p(v)` over batch verb assignments — penalize unused codes.
2. **angle/scale spread**: penalize `|θ_v|→0` and `|log r_v|→0` (operator → identity) AND pairwise `‖(θ_u,log r_u) − (θ_v,log r_v)‖→0` (verbs → each other).

### VerbHead gradient fix [FIX — Judge 0/1 shared flaw]
Hard argmax verb assignment gives **zero gradient to VerbHead** from `L_pred` — only `L_div` (softmax) reaches it, so VerbHead can route all slots to one verb and `L_pred` still minimizes via operator angles → codebook collapse. **Fix: Gumbel-softmax with temperature annealing.** During training, `a*` is computed as a **soft mix** over verbs (`Σ_v softmax_v · B_v k`) so `L_pred` gradients flow into VerbHead logits. Temperature anneals `τ_g: 2.0 → 0.5` over the first 30% of steps; at eval and export, hard argmax is used (straight-through at the anneal tail). This is the single most important trainability fix.

### Explicitly absent (regression if `> 0`)
invertibility loss, consistency/cycle loss, orthogonality penalty, norm-matching loss, VAE KL, spectral loss, CKA, operator-composition loss, any token-space reconstruction loss.

---

## 4. EMA schedule [FIX — Judge 1: brief lag analysis, not just assertion]

- Fixed `τ = 0.995`, **no cosine schedule**. I-JEPA/V-JEPA cosine schedules are calibrated for 200–800-epoch ViT runs and push `τ→1` too early on short GLUCOSE runs.
- **Lag analysis (concrete):** GLUCOSE has 36,449 chains → ~72K adjacent `(t,t+1)` pairs; batch 64 → ~1,130 steps/epoch. EMA lag `1/(1−τ)=200` steps ≈ **0.18 epoch**. At `τ=0.995` the target trails the online encoder by under a fifth of an epoch — fast enough to remain a moving, informative target, slow enough not to be an adversary. Watch the held-out reconstruction diagnostic (§5): if it plateaus while train `L_pred` keeps falling, the target has stabilized prematurely and `τ` should drop to 0.99.
- EMA encoder = `copy.deepcopy(online_encoder + readout)` at step 0, `requires_grad=False`. Update `θ_ema ← τ·θ_ema + (1−τ)·θ_online` called **manually after each optimizer step** (the AdamW/CosineLR trainer has no hook). EMA params **excluded** from `clip_grad_norm_` and the trainable list.
- We keep EMA (unlike LeJEPA's no-EMA): `z` is the full-text rep of the **next** state — a genuinely different object than the operator-transformed nouns of the current state, so SIGReg alone does not make it a stationary target.

---

## 5. Diagnostics suite (`diagnostics.py`, first-class)

`eval_diagnostics(model, dataset, device, n_examples=512, out_dir)` → flat dict + PNGs, logged every N epochs and runnable standalone on any `.pt`.

| Group | Metric | Pass / alarm |
|---|---|---|
| Noun geometry | effective rank `tr(C)²/‖C‖_F²` of noun covariance; per-dim variance hist | eff_rank ≥ dn/4; alarm if `< dn/4` before epoch 10 → raise `w_sigreg` |
| **Scale drift** [FIX §1.5] | per-verb `log r` distribution; running `mean ‖a*‖/‖k‖` | WARN if `mean log r < −1.0` sustained (runaway contraction) |
| Verb non-triviality | `|θ|` hist; `|log r|` hist; codebook usage perplexity; pairwise `‖B_u−B_v‖_F` | ppl > V/2; no cluster at identity |
| Structural sanity | `operator.structural_sanity(v)` → `‖BB⁻¹−I‖`, `‖B⁻¹Bk−k‖` | ≈ 0 (else `apply` bug — NOT a loss) [GRAFT D3] |
| **Residual-vs-slots** [GRAFT D1, all judges] | `L_pred` as slots masked coarse→fine | monotone improvement → slots carry distinct info; flat → stripe collapse |
| Slot-attention entropy | per-slot attention entropy over text | peaked & distinct (catches stripe collapse — the silent failure not visible to batch-level SIGReg, Judge 1) |
| Binding | noun-cluster (KMeans) × verb-firing contingency table | structured, not uniform; catches collapsed slots [GRAFT D1] |
| Held-out reconstruction | held-out `cos(zhat, z)` / `MSE` | rising / falling — primary label-free model-selection signal |
| **Multi-step drift** [FIX — Judge 2 shared flaw] | apply `B_v` N times to held-out `k`, measure distance to nearest encoder-manifold noun (cos to nearest training noun) | bounded; alarm if drift grows unboundedly with N (the long-session product risk) |

The residual-vs-slots curve and slot-attention entropy together are the **only** signals that catch per-slot mode collapse, which batch-pooled SIGReg cannot see (Judge 1). They are first-class, not optional.

---

## 6. GLUCOSE data path (file: `data.py`)

`JEPAChainDataset` over `data/glucose/chain_general_train.jsonl` (36,449 chains, format `{"chain": [t0,t1,t2]}`).

- **v1 pairing (anti-goal: no multiturn):** emit adjacent **pairs** `(state_t, state_{t+1})`. [FIX — Judge 2 D2 flaw] The online encoder sees `state_t`; the **EMA target encoder sees `state_{t+1}`** (the prediction target). Do **not** feed the same text to both encoders — that degenerates into self-reconstruction with a moving target and supplies no cross-state JEPA signal until the encoders happen to diverge. Cross-state pairing is mandatory.
- **Vocabulary [FIX — Judge 1 critical shared flaw]:** GLUCOSE has ~32K unique surface types; a 53–64-token table is unusable and a raw GPT-2 50257 table is too large for nano export. **Build a domain BPE of vocab=512** on the GLUCOSE corpus using the existing `src/twm/domain_bpe.py :: DomainBPETokenizer.from_pretrained`. 512 tokens covers GLUCOSE's placeholder-heavy, low-entropy text (`Someone_A`, `Something_B`, common verbs) and keeps `token_emb` at 512·d. `max_text_tokens=64` (GLUCOSE steps average ~80–120 chars; at 512-vocab BPE that is comfortably < 64 subwords). A prebuild step writes `data/glucose/jepa_bpe_512.json`.
- Tokenize, pad to `max_text_tokens`, store contiguous CPU tensors `src_ids/src_pad/tgt_ids/tgt_pad` (trainer convention: no DataLoader, direct index slicing).
- `iter_text_pairs()` yields `(state_t, state_{t+1})` token tensors for the operator-fit second pass (§7).
- `nsp_train.jsonl` (key `"states"`, 4,113 rows) needs a thin `states→chain` rename adapter — **out of scope for v1**.

---

## 7. Pre-build operator-group-fit experiment (DONE + trained-encoder second pass)

**Pass 1 (COMPLETE):** `scripts/operator_group_fit.py` already ran on MiniLM embeddings and produced the §1.1 verdict (rotation+scale, residual 0.709 vs identity 0.929). Report: `research/operator_group_fit.md`; results: `results/operator_fit/`. This **gated and authorized** the rotation+scale mandate. Methodology: per-2×2-block closed-form Procrustes (det-corrected) for U(1); Umeyama optimal scalar per block for rot+scale; full Procrustes for orthogonal; ridge (λ=1e-2) for general; fit globally and per-cluster (k=16 KMeans on transition direction = verb-codebook proxy); two independent reversibility labels (GLUCOSE-dimension proxy and verb-keyword heuristic, agreement 0.657). [GRAFT D1: per-cluster SVD analysis + the physical-vs-mental causation reversibility split are the robust labeling heuristic, superior to raw string matching.]

**Pass 2 (REQUIRED before locking final scale magnitudes) [FIX — Judge 1/2: frozen embeddings ≠ trained nouns]:** after a short nano warmup (~5 epochs), re-run the fit on the **trained slot-encoder noun space** via `iter_text_pairs()`. If the trained nouns show materially *less* contraction than MiniLM (median σ → 1), the scale degrees of freedom may be re-checkable; if *more*, confirm rotation+scale and watch scale-drift. This pass is logged to `results/operator_fit/trained_pass.json` and is the resolution of O3. It does **not** re-gate the build (rotation+scale ships regardless, since reverting to pure rotation is a frozen-`log_r` config flip), but it sets `w_scale_reg` if drift is severe.

Decision rule retained as documentation [GRAFT D1]: `>80%` transitions with all σ∈[0.9,1.1] ⇒ pure rotation suffices. The data does **not** meet this (60–88% σ<1) ⇒ rotation+scale, as mandated.

---

## 8. Export & quantization (file: `export_jepa_weights.py`) [GRAFT D2, Judges 0/1]

Extends the repo's `to_list()` + JSON pattern (`demo/pet_simulation/export_weights.py`, `model_weights.json`). Exports **online encoder + operator + readout** only (EMA, predictor, SIGReg are training-only).

- **Operator INT8 bake:** `operator.bake()` returns per verb `(cosθ, sinθ, r)` → store `(cos, sin)` INT8-quantized in `[−1,1]` with scale `1/127`, and `r` as fp16 (one per block; small). nano operator = `V·(dn/2)·3` values = `8·16·3 = 384` numbers ≈ **384 INT8 + 128 fp16 bytes**, vs ~1.5KB FP32.
- **Weight-only INT8 for cross-attention** [GRAFT D2, Judge 0]: quantize cross-attn and self-attn **weights** to INT8 (per-channel, symmetric, scale=max/127); keep **activations in fp16**. Activation INT8 in cross-attention is the documented DETR collapse mode — do not quantize cross-attn activations.
- **token_emb:** export as INT8 shared table (512·d), not dense FP32.
- nano total INT8 export ≈ **~95KB** (≈77K weights at ~1 byte + small fp16 scales + emb table) — inside the 303KB pet-sim envelope.
- mini exports fp16 (~1.9MB) for the research rig; not the primary browser target.
- JS `TWM` class extended with `slot_init`, `operator` (baked cos/sin/r), `readout` sections, plus `step_latent(k, verb_idx)` and `undo_latent(a, verb_idx)` mirroring §9.

---

## 9. Pet engine API (file: `model.py`) — persistent-state loop [GRAFT D3]

```python
m = JEPAOperatorModel(cfg, token_emb)
out = m(src_ids, src_pad)             # encode current state -> {k, verb, a*, zhat}
k   = out["k"]                        # (B,M,dn) persistent points, stored in JS gameState
a   = m.step_latent(k, user_verb_idx) # one tick: a* = B_v k
k2  = m.undo_latent(a, user_verb_idx) # exact undo: k = B_v^{-1} a
san = m.operator.structural_sanity(user_verb_idx)  # runtime invariant check {bbT_err, inv_err}
```

In the browser: text→slots→`k` once per pet, then each action is RoPE-style elementwise mults + a per-block scalar `r` (baked). State persists in JS `gameState[pet]` exactly like the existing pet-sim (stateless model, external state). The v2 flow seam animates *between* ticks via `integrate(k, v, T>1)` — same operator, no retraining of the discrete path. **Caveat (§0.1):** in v1 `user_verb_idx` maps to GLUCOSE causation-type verbs, not UI actions (O1).

---

## 10. Training loop & config

`scripts/train_jepa.py configs/jepa_nano.json`. Fixed seed. AdamW + `CosineAnnealingLR` on **online params only**. Per batch: forward (Gumbel-softmax verbs) → `JEPALoss` → backward → `clip_grad_norm_(online, 1.0)` → `optimizer.step()` → `model.ema_update(τ)` → anneal `τ_g` → periodic `eval_diagnostics`. Single 24GB GPU (homelab server per CLAUDE.md; small enough for local CPU sanity checks).

### JSON config schema (`configs/jepa_nano.json`), consistent with the repo's config-driven trainer

```json
{
  "profile": "jepa_nano",
  "seed": 0,
  "data": {
    "path": "data/glucose/chain_general_train.jsonl",
    "tokenizer": "data/glucose/jepa_bpe_512.json",
    "vocab_size": 512,
    "max_text_tokens": 64,
    "pairing": "adjacent"
  },
  "model": {
    "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
    "block": 2, "n_text_layers": 2, "tie_text_layers": true,
    "n_heads": 4, "n_slot_iters": 3,
    "operator_group": "rotation_scale",
    "n_steps_T": 1
  },
  "loss": {
    "w_pred": 1.0, "w_sigreg": 0.05, "w_div": 0.1, "w_scale_reg": 0.0,
    "sigreg": {"n_slices": 256, "n_knots": 17, "knot_max": 3.0, "standardize": true},
    "verb": {"gumbel_tau_start": 2.0, "gumbel_tau_end": 0.5, "anneal_frac": 0.3}
  },
  "ema": {"tau": 0.995, "schedule": "fixed"},
  "optim": {"lr": 3e-4, "weight_decay": 0.01, "batch_size": 64, "epochs": 100, "grad_clip": 1.0, "warmup_steps": 200},
  "eval": {"every_epochs": 5, "n_examples": 512, "out_dir": "results/jepa_nano"},
  "operator_fit_pass2": {"enabled": true, "after_epoch": 5}
}
```

`jepa_mini.json` is identical with `profile=jepa_mini`, `d_model=128`, `n_slots=12`, `n_verbs=16`, `n_text_layers=4`, `tie_text_layers=false`.

`src/twm/jepa/config.py` adds `JEPA_PROFILES` (`jepa_nano`, `jepa_mini`) kept **separate** from `config.PROFILES` (so `build_model_config` is untouched) and a `JEPAConfig` dataclass with `from_json`/`from_dict` matching `training_config.py` conventions.

---

## 11. Reuse map (integrate with existing repo)

- `src/twm/domain_bpe.py` `DomainBPETokenizer.from_pretrained` — **build the 512-token GLUCOSE BPE** (resolves the vocab flaw).
- `src/twm/text_compressor.py` — `extract_queries`, `role_emb`, `triple_pos_emb`, `cross_attn`, `cross_ln`, `query_self_attn` are the **slot-encoder backbone** (cross-attn extraction + coordination).
- `src/twm/vq_layer.py` `VectorQuantizer` — fp32-under-autocast distance pattern reused for the operator fp32 gotcha; STE pattern reference for the Gumbel tail.
- `src/twm/chain_dataset.py` — adjacent-pair iteration reference for `JEPAChainDataset`.
- `demo/pet_simulation/export_weights.py` `to_list()` + JSON layout — extended by `export_jepa_weights.py`.
- `src/twm/training_config.py` — `from_json`/dataclass conventions for `JEPAConfig`.
- **Not reused as the operator:** `TransformerDynamics` (the operator IS the algebra). Its zero-init out_gate pattern informs the readout init only. The optional mini coordinator is **removed** (§2).

---

## 12. Implementation work breakdown — 6 parallelizable tasks, no two touch the same file

| # | Task | Owns (writes) | Reads (no writes) | Deliverable |
|---|---|---|---|---|
| **T1 — Operator** | `RotationScaleOperator`, interface, stubs, T-seam, `structural_sanity`, `bake` | `src/twm/jepa/operator.py` | `vq_layer.py` | apply/inverse_apply/velocity/integrate(T=1); fp32 autocast; unit tests for `‖BB⁻¹−I‖`, exact undo, scale round-trip |
| **T2 — Slot encoder** | `SlotEncoder`, `NounHead` (standardize, no L2), `VerbHead`, ALBERT tying, 3-iter coordination | `src/twm/jepa/slot_encoder.py` | `text_compressor.py` | (B,M,d) slots → k, verb_logits; param count matches §2 |
| **T3 — Losses** | `L_pred`, SIGReg (standardize precondition), `L_div`, Gumbel-softmax verb path + anneal | `src/twm/jepa/losses.py` | — | `JEPALoss(...)`; SIGReg unit test that isotropic input ≈ 0 loss and sphere-projected input is NOT used |
| **T4 — Data + tokenizer + operator-fit pass2** | `JEPAChainDataset`, `iter_text_pairs`, BPE build script, pass-2 fit hook | `src/twm/jepa/data.py`, `scripts/build_jepa_bpe.py` | `domain_bpe.py`, `chain_dataset.py`, existing `scripts/operator_group_fit.py` | cross-state `(t,t+1)` tensors; `data/glucose/jepa_bpe_512.json`; pass-2 report |
| **T5 — Model + EMA + trainer + config** | `JEPAOperatorModel` (`step_latent`/`undo_latent`), EMA, `JEPAConfig`, train loop, configs | `src/twm/jepa/model.py`, `src/twm/jepa/config.py`, `scripts/train_jepa.py`, `configs/jepa_nano.json`, `configs/jepa_mini.json` | T1–T3 interfaces, `training_config.py` | end-to-end train run; manual EMA update; Gumbel anneal wired |
| **T6 — Diagnostics + export** | `eval_diagnostics` (all §5 metrics incl. scale-drift, multi-step drift, residual-vs-slots), INT8 export | `src/twm/jepa/diagnostics.py`, `scripts/export_jepa_weights.py` | T1/T2/T5 outputs, `demo/pet_simulation/export_weights.py` | diagnostic dict+PNGs; INT8 nano export ≤ 303KB |

Interfaces (T1's `Operator`, T2's `SlotEncoder` signatures, T3's `JEPALoss` signature, T4's dataset `__getitem__` contract) are frozen in a shared `src/twm/jepa/__init__.py` **stub file written first by T5** (signatures only, no bodies) so T1–T4/T6 develop against fixed contracts. Each task owns disjoint files; `__init__.py` is T5-owned.

---

## 13. Anti-goal compliance

- ✅ No token-space decoder (latent MSE only).
- ✅ No multiturn loop (adjacent cross-state pairs only).
- ✅ T hard-set to 1 (loop + per-step hook dormant).
- ✅ SO(n) is an interface stub (`NotImplementedError`).
- ✅ No operator-composition loss; no VAE/KL/spectral/CKA/consistency/invertibility loss.
- ✅ Modular: `operator / slot_encoder / losses / diagnostics / data / model / config / export` isolated under `src/twm/jepa/`.

---

## 14. Open questions (honest)

- **O1 — Pet verbs vs GLUCOSE verbs.** GLUCOSE yields causation-type verbs, not UI actions (feed/pet/play/scold). Realizing the literal pet product needs an action-labeled dataset or fine-tune. v1 ships the engine + API + causation verbs only.
- **O2 — Multi-step latent drift.** v1 supervises only one-step transitions, but the product applies `B_v` repeatedly to stored `k`. Without a decoder there is no token-space bound on drift. The multi-step-drift diagnostic (§5) measures it but does not bound it; if drift is severe, a periodic re-encode ("the pet glances at itself") is the v1.1 mitigation.
- **O3 — Operator group in learned geometry vs MiniLM.** The rotation+scale verdict is from MiniLM embeddings. Pass-2 fit on trained nouns (§7) re-verifies; the scale magnitude / `w_scale_reg` may shift. Reverting to pure rotation is a frozen-`log_r` config flip if pass-2 contradicts.
- **O4 — SIGReg vs semantic clustering tension** (Judge 2). SIGReg pushes nouns toward isotropic Gaussian, but a readable pet state may want clustered regions (hunger cluster, mood cluster). `w_sigreg=0.05` is deliberately light; the binding contingency table (§5) watches whether useful structure survives. If clustering is suppressed, lower `w_sigreg` further or switch SIGReg to a per-slot-conditional variant (v1.1).
- **O5 — EMA stabilization on short runs.** Lag analysis (§4) suggests `τ=0.995` is fine for ~100 epochs, but if held-out reconstruction plateaus while train `L_pred` falls, drop to `τ=0.99`. Empirical, watched via diagnostics.
- **O6 — Block size.** v1 uses `b=2` everywhere for JS/INT8 triviality. Cross-block coupling (`b>2`) is deferred to v2; the fit signal lives in per-block scale, so `b=2` is expected sufficient, but this is unverified in the learned space.
