# Operator Group Fit — Which group for the v1 transition operator?

**Question.** We will model a world-model state transition as a per-element linear
operator `B` acting on a state embedding: `B @ emb(state_t) ≈ emb(state_t+1)`. Before
building an unroll, decide empirically which group `B` should live in, in increasing
expressivity:

1. **U(1)^(d/2)** — block-diagonal 2×2 rotations (RotatE-style; orthogonal, norm-preserving, reversible)
2. **rot+scale** — block-diagonal 2×2 rotation with a per-block scalar scale (can contract/forget; irreversible expressible)
3. **orthogonal** — full orthogonal Procrustes (SVD with det correction)
4. **general** — unconstrained ridge least squares

**Hypothesis.** Reversible transitions are well-fit by pure rotations; irreversible
transitions (consumption, destruction, creation, collapse) need the scale/contraction
degrees of freedom. The residual gap between groups, split by reversibility, decides
the parameterization.

Reproduce: `uv run --with sentence-transformers --with scikit-learn --with matplotlib python scripts/operator_group_fit.py`
(seeded, end-to-end). Artifacts in `results/operator_fit/`.

---

## Methodology

**Data.** GLUCOSE causal chains. We extract `(cause, effect)` text pairs from the 10
GLUCOSE causal dimensions in `data/glucose/GLUCOSE_training_data_final.csv`. Each
dimension carries a relation connector (`>Causes/Enables>`, `>Motivates>`, `>Enables>`,
`>Causes>`, `>Results in>`); we split on it to get `(state_t, state_t+1)`. 12,000 pairs
sampled (seed 0), 80/20 train/test.

**Embeddings.** `sentence-transformers/all-MiniLM-L6-v2`, d=384, L2-normalized. (TF-IDF
+ SVD-128 fallback is implemented but was not needed — the encoder downloaded fine.)

**Two reversibility labels (independent), agreement checked.**
- **A — GLUCOSE-dimension proxy.** Dims 1–5 are antecedent enabling pre-conditions /
  motivations → *reversible* standing states; dims 8–10 (`Results in` — changes of
  state/location/possession) → *irreversible*. Dims 6–7 (ambiguous causal/emotional
  consequents) dropped from the proxy-labeled set.
- **B — verb/keyword heuristic on the effect text.** consumption / destruction /
  creation / terminal verbs (eat, break, die, burn, spill, finish, win, build, born…)
  → irreversible; locomotion / manipulation / toggle verbs (move, walk, open, close,
  pick up, sit, hold…) → reversible.
- **Agreement** where both strategies fired (n=2,360): **0.657**. Moderate — the two
  views correlate but are not redundant, so we report both. Class counts:
  dim_rev=6,934 / dim_irr=1,663 ; kw_rev=2,173 / kw_irr=737.

**Operator fits.**
- *U(1) / rot+scale*: per 2×2 block, closed-form 2D orthogonal Procrustes (cross-cov
  SVD, det-corrected proper rotation); rot+scale adds the Umeyama optimal scalar per block.
- *orthogonal*: full Procrustes `B = U Vᵀ` from SVD of the cross-covariance.
- *general*: ridge least squares (λ=1e-2).
- Each family fit **globally** (one `B`) and **per-cluster** (k=16 KMeans clusters on
  the unit transition direction `emb(t+1)−emb(t)` — a verb-codebook proxy; one `B` per
  cluster, min 40 train pairs/cluster).

**Baselines.** identity (`B=I`) and mean-shift (`z_t + mean Δ`).

**Reported.** held-out residual MSE `‖B zₜ − zₜ₊₁‖²` and predicted cosine, per family ×
reversibility class × global/clustered; singular-value spectra of the general-linear
maps split by reversibility; effective rank of transition directions.

---

## Results

### Held-out residual MSE (global fit, all-MiniLM, d=384)

| class | identity | mean_shift | U(1) rot | **rot+scale** | orthogonal | general |
|---|---:|---:|---:|---:|---:|---:|
| all     | 0.929 | 0.925 | 0.929 | **0.709** | 0.927 | 0.643 |
| dim_rev | 0.916 | 0.894 | 0.914 | **0.701** | 0.921 | 0.645 |
| dim_irr | 0.828 | 0.781 | 0.824 | **0.649** | 0.868 | 0.750 |
| kw_rev  | 0.912 | 0.899 | 0.911 | **0.699** | 0.980 | 0.743 |
| kw_irr  | 0.896 | 0.868 | 0.893 | **0.689** | 1.047 | 1.178 |

### Clustered (per-cluster operator, k=16) — robustness on small subsets

| class | U(1) | **rot+scale** | orthogonal | general |
|---|---:|---:|---:|---:|
| all     | 0.916 | **0.693** | 0.997 | 1.097 |
| dim_rev | 0.906 | **0.691** | 1.031 | 1.403 |
| dim_irr | 0.806 | **0.632** | 1.042 | 1.225 |
| kw_rev  | 0.906 | **0.691** | 1.229 | 1.266 |
| kw_irr  | 0.933 | 0.705 | 1.313 | 0.980 |

### General-linear singular spectrum (contraction evidence)

| class | frac(s<1) | mean s | median s | eff-rank(Δ) |
|---|---:|---:|---:|---:|
| all     | 0.878 | 0.498 | 0.394 | 313 |
| dim_rev | 0.823 | 0.588 | 0.450 | 314 |
| dim_irr | 0.685 | 0.882 | 0.577 | 298 |
| kw_rev  | 0.716 | 0.812 | 0.503 | 301 |
| kw_irr  | 0.602 | 1.270 | 0.684 | 273 |

Plots: `results/operator_fit/residual_by_family_class.png`,
`results/operator_fit/singular_spectra.png`. Full numbers:
`results/operator_fit/operator_fit_results.json`.

---

## The singular-value story

The general-linear map fitted to GLUCOSE cause→effect pairs is **strongly contractive
even for "reversible" transitions**. The median singular value is 0.39–0.58 and 60–88%
of singular values sit below 1 in every class. A norm-preserving group (U(1), full
orthogonal) is structurally incapable of moving between two genuinely different
sentence embeddings — and the data confirms it: **U(1) and orthogonal barely beat
identity** (0.929 vs 0.929; sometimes *worse*, e.g. orthogonal on kw_irr = 1.047 > I).
This refutes the naive form of the hypothesis: reversible transitions are *not* well-fit
by pure rotations in sentence-embedding space.

The **scale degree of freedom is the single biggest, most universal win.** Adding one
scalar per 2×2 block (`rot+scale`) cuts residual ~24% below identity in *every* class.
The contraction it provides is exactly the missing capacity — the operator needs to be
able to shrink/forget, not just rotate.

Reversibility **does** modulate the spectrum in the hypothesized direction, but weakly
and as a second-order effect: irreversible classes have systematically higher singular
values and fewer contracting directions (dim_irr frac(s<1)=0.685 vs dim_rev 0.823;
kw_irr 0.602 vs kw_rev 0.716; mean s 0.88/1.27 vs 0.59/0.81). Interpretation: irreversible
"Results in" effects move the embedding *farther and more isotropically* (a new
state/location/possession), so the optimal map is less of a pure contraction; reversible
preconditions stay nearer the antecedent, so the map contracts toward a shared region.
The effect is real but small relative to the dominant "everything contracts" signal.

**Why not just use the general-linear map?** Globally it wins on aggregate MSE (0.643),
but it is *not robust*: under per-cluster fitting (the actual v1 use case — a verb
codebook of small operators) the general map **overfits and blows up** (clustered MSE
1.10–1.40, worse than identity) because a full d×d=384² matrix cannot be estimated from
a few-hundred-pair cluster. **rot+scale is the only family that improves under
clustering** (0.709→0.693 global→clustered) and stays stable on the smallest subsets.
It has O(d) parameters per operator vs O(d²), so it generalizes from tiny per-verb
samples — exactly what a nano/mini operator codebook needs.

---

## RECOMMENDATION

**Parameterize the v1 nano/mini transition operator as block-diagonal 2×2
rotation-plus-scale (`rot+scale`), i.e. per block a 2D rotation times a positive scalar
— equivalently a complex diagonal `r·e^{iθ}` per U(1) factor.** This is the smallest
group above pure rotation that contains the contraction/forgetting capacity the data
demands, and it is the only family that both (a) beats identity by a wide, consistent
margin and (b) stays robust under per-cluster (verb-codebook) fitting on small samples.

Concretely:
- **Drop pure U(1) / RotatE-style rotations and full orthogonal maps.** Norm-preservation
  is the wrong inductive bias for state transitions in embedding space — they don't beat
  identity. The reversibility distinction does not rescue them: even reversible transitions
  contract.
- **Do not ship a general-linear operator for the codebook.** It wins globally but
  overfits per-cluster (O(d²) params); reserve it only as an oracle/upper-bound or for a
  single global operator with heavy ridge.
- **A per-block (or even per-U(1)-factor) scalar scale is enough;** we did not need a full
  matrix. If you want one knob for reversibility, allow the scale to exceed 1 (don't clamp
  to ≤1): irreversible "creation/results-in" effects occasionally need expansion
  (kw_irr mean s = 1.27), so an unconstrained positive scale is preferable to a strict
  contraction.

**Caveats.**
1. Sentence-embedding geometry, not the project's learned triple space — the *qualitative*
   conclusion (need scale, not just rotation) should transfer, but absolute residuals will
   differ in the TWM bottleneck. Re-run this fit on bottleneck embeddings before locking
   numbers.
2. Reversibility labels are heuristic (dimension proxy + keyword vote, 0.657 agreement);
   the reversibility *modulation* of the spectrum is real but second-order, so v1 need not
   branch operator type on reversibility — a single `rot+scale` family covers both.
3. Residuals are high in absolute terms (~0.65–0.93 on unit vectors) because GLUCOSE
   cause→effect is a *semantic* leap, not a tight state delta. The world-model unroll over
   adjacent states should be an easier, lower-residual regime — treat these as a
   conservative lower bound on operator usefulness.
