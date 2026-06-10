# JEPA v2/v2.1 Experiment-Matrix Synthesis

Overnight matrix (2026-06-10). Five GLUCOSE chain-dynamics runs (100 epochs each,
nano d_model=64 unless noted), plus the anchor-stability experiment (the stage-0
gate for the LLM-mount interface) and the v2 probe battery. Numbers-first; honest
about what is seed-variance and what is signal. All checkpoints live server-side at
`~/triples_world_model_Glucose/results/`; staged artifacts at
`results/jepa_matrix/staged/`.

## TL;DR

- **Anchor / relative-representation thesis: MARGINAL, not viable yet.** Anchor-similarity
  Pearson 0.43 (readout) / 0.46 (slot-mean) — both in the 0.4–0.7 grey zone, neither
  above the 0.7 interface-viable bar. The Procrustes alignment ceiling is high
  (0.93 readout) so the geometry IS shared up to rotation; the relative-representation
  *encoding* of that shared geometry is just noisy at N=64 anchors.
- **Polar v2.1 ≈ v2.0 on every headline metric.** The polar gap (ce_gap) difference
  between polar and baseline is *within* the seed-to-seed spread (seed0 0.082 vs seed1
  0.134 on the same config). Polar conditioning is behaviorally neutral here — neither
  the win nor the regression the design hoped to gate.
- **The decoder arm is the one real capability win.** Doubling the decoder (d_dec
  64→128, 1→2 layers, 237K→657K params) cuts teacher-forced CE from ~1.39 to **0.95
  nats** and visibly fixes token-level degeneration (grammatical GLUCOSE sentences vs
  the nano decoder's `ashespond.ashes.ashostigned` gibberish). It does NOT fix the
  semantic/content gap — generations are fluent but wrong.
- **Three capability gaps persist across every arm**: (1) next-state retrieval barely
  beats chance, (2) the discrete action carries no recoverable semantics (NMI ≈ 0
  vs reversibility), (3) generations are generic, 0% exact-match.

---

## 1. The matrix

Final-epoch (ep100) diagnostics. **Eval-set caveat:** baseline / polar-seed1 / dn64
numbers are from their retained training logs (eval on train chains, n=512). The
decoder-arm diag was *recovered by re-running `eval_diagnostics_v2`* on its ep100
checkpoint (the live log was overwritten by the dn64 run that followed it in the same
job) — also train chains, n=512, so comparable. The polar-seed0 vs seed1 `ce_gap`
contrast is on **test** chains (a separate post-hoc pass), hence its absolute CE differs.

| Arm | params | ce_true↓ | ce_gap↑ | hard_mrr | easy_mrr | chrF | v_ppl(post) | codes | noun_rank | modulus_rank |
|-----|-------:|--------:|--------:|---------:|---------:|-----:|------------:|------:|----------:|-------------:|
| **baseline** `jepa_nano_v2` (no polar) | 236,568 | 1.394 | **0.166** | 0.071 | 0.035 | 0.384 | 7.40 | 8/8 | 11.1 | — |
| **polar seed0** `jepa_nano_v21` | 236,824 | — | (0.082)¹ | — | — | — | — | — | — | — |
| **polar seed1** `jepa_nano_v21_seed1` | 236,824 | 1.386 | 0.126 | 0.081 | 0.024 | 0.382 | 7.03 | 8/8 | 11.9 | 6.45 |
| **decoder arm** `jepa_small_v21_dec` (d_dec128/2L) | 657,304 | **0.953** | 0.135 | 0.057 | 0.018 | 0.373 | 7.84 | 8/8 | 11.4 | 6.16 |
| **dn64** `jepa_nano_v21_dn64` (d_noun 32→64) | 260,632 | 1.374 | 0.091 | 0.061 | 0.025 | 0.372 | 7.58 | 8/8 | 12.6 | 10.85 |

¹ seed0 `ce_gap` 0.082 and seed1 0.134 are the **test-chain** v-ablation contrast
(anchor experiment side-output). On test, polar seed1's gap is 63% larger than seed0's
from an *identical config* — this is the headline seed-variance finding (see §2).

### The polar `ce_gap` dynamics across epochs (one retained seed)

The retained polar log (one of the two anchor-pair seeds; wartable kept only the
second training's stdout) shows `ce_gap` is **non-monotone and noisy** across the run,
oscillating in [0.07, 0.18] with no clean trend after the τ-anneal completes (ep50):

```
ep5  0.282   ep30 0.075   ep55 0.127   ep80 0.136
ep10 0.224   ep35 0.121   ep60 0.128   ep85 0.129
ep15 0.083   ep40 0.088   ep65 0.144   ep90 0.114
ep20 0.090   ep45 0.104   ep70 0.156   ep95 0.122
ep25 0.092   ep50 0.101   ep75 0.133   ep100 0.126
```

The early-epoch high gap (0.28 @ ep5) is an artifact of high absolute CE during warmup
(a constant-verb ablation costs more when the decoder is bad), not real action causality.
The post-anneal plateau (~0.12) is the honest number. The baseline (no-polar) plateau is
~0.16 — i.e. baseline's gap is *higher*, but again inside the seed band. **Verdict: the
ce_gap "polar gap dynamics" is seed-variance-dominated; do not read a polar effect into it.**

---

## 2. Per-arm verdicts

### Decoder arm — the win, with an asterisk
- **Token-level fluency is fixed.** ce_true 1.39→**0.95 nats**. The nano decoder
  (d_dec64, 1 layer) emits degenerate BPE soup; the d_dec128/2-layer decoder emits
  well-formed GLUCOSE sentences (see §4). This is a decoder-capacity bottleneck, not a
  representation bottleneck — the slots were always carrying enough; the 1-layer head
  couldn't render them.
- **The asterisk:** chrF actually *drops* slightly (0.384→0.373) and hard_mrr drops
  (0.071→0.057). Fluency went up, content-correctness did not. The decoder learned to
  produce the GLUCOSE *register* better, not the right next *state*. 657K params bought
  grammar, not semantics.

### Polar (v2.1) — behaviorally neutral
- Modulus geometry is healthy and stable (modulus_eff_rank 6.45 > threshold 4,
  identity_persistence_pass=True, n_pure_rotation_verbs=0). The polar machinery works
  as designed and does not break anything (the §11 behavior-preservation gate holds).
- But it moves no headline metric outside the seed band. ce_gap, chrF, mrr, v_ppl are
  all baseline-equivalent. **Polar is a free, non-harmful structural prior here, not a
  capability lever** at this scale/data.

### dn64 (d_noun 32→64) — modulus rank up, nothing else
- modulus_eff_rank jumps 6.5→**10.8** (more independent identity coordinates), as
  expected from doubling the block count. noun_eff_rank also up (12.6).
- But ce_true (1.374) and chrF (0.372) are flat vs nano, and ce_gap actually *drops*
  to 0.091. More noun dimensions did not translate into better dynamics or generation.
  The bottleneck is not noun capacity.

### Seed variance — the dominant effect
- Two identical-config polar runs (seed0/seed1) give test `ce_gap` 0.082 vs 0.134
  (a 63% relative spread) and train-log plateaus that overlap the baseline's. **Any
  claim about a sub-0.05-nat effect on ce_gap in this matrix is seed-variance-dominated
  and should be treated as null.** The LOO probe (probe2) flipping v2 to "all slots
  constructive" vs v1's "non-constructive" is a real qualitative flip, but the *magnitude*
  ordering across these five arms is mostly noise.

---

## 3. Anchor-stability experiment (the headline / stage-0 gate)

**Protocol.** N=64 anchor states (first state of every 60th chain in
`chain_general_test.jsonl`) + M=512 probe states (stride-7 from offset 3, disjoint from
anchors; both seed-free deterministic slices). For each of two seed checkpoints
(`jepa_nano_v21` seed0, `jepa_nano_v21_seed1`) and three representation variants, each
probe gets an anchor-similarity vector s∈R⁶⁴ (cosine to each anchor). Headline = mean
Pearson/Spearman of s_seed0(x) vs s_seed1(x) across probes. Contrast = raw cross-seed
probe cosine (frames differ → expect ~0) vs orthogonal-Procrustes-aligned cosine
(alignment ceiling). Plus top-5 anchor-neighbor Jaccard.

Representations: **readout** = the model's native pooled-noun `readout(k)` (dn=32);
**slotmean** = slot-set mean of k (dn=32, pooling-free); **modulus** = mean over slots
of the per-block modulus profile |z_b| (nb=16, the polar "identity" factor).

| variant | anchor-sim Pearson | Spearman | raw cross-seed cos | Procrustes cos | top-5 Jaccard | frac probes ρ>0.7 | verdict |
|---------|------------------:|---------:|-------------------:|---------------:|--------------:|------------------:|---------|
| **readout** | **0.431** ± 0.17 | 0.523 | −0.090 | **0.925** | 0.275 | 5.7% | MARGINAL |
| **slotmean** | **0.456** ± 0.21 | 0.401 | −0.076 | 0.580 | 0.286 | 13.3% | MARGINAL |
| **modulus** | 0.356 ± 0.23 | 0.349 | **0.959** | 0.978 | 0.223 | 0.6% | (see note) |

### Verdicts
- **Relative-representation thesis: MARGINAL — interface NOT yet viable.** Both full-vector
  variants land at ρ≈0.43–0.46, squarely in the 0.4–0.7 grey zone, *below* the 0.7
  interface-viable bar and *above* the 0.4 thesis-in-trouble floor. The thesis is alive
  but unproven at this anchor budget.
- **The geometry IS shared up to rotation.** Raw cross-seed probe cosine is ~0
  (−0.09/−0.08) — absolute frames are uncorrelated, exactly as the relative-rep premise
  assumes. After orthogonal Procrustes fit on the anchors, readout probe cosine jumps to
  **0.925**. So the two seeds learn the same noun manifold up to an orthogonal transform;
  the anchor-similarity encoding recovers only ~half of that (ρ 0.43 vs ceiling implied
  by 0.93 Procrustes). **The bottleneck is the relative encoding's noise at N=64, not a
  missing shared structure.** An M-sweep (more anchors) is now justified and likely to
  push ρ up toward the Procrustes ceiling.
- **Modulus is a special case — strong raw stability, weak relative stability.** The
  polar identity factor has **raw cross-seed cosine 0.959** (!) — the modulus profile is
  near-seed-invariant in absolute terms, without any alignment. But its anchor-similarity
  Pearson is the *lowest* (0.356), because the modulus profile is low-rank/low-variance
  (eff_rank ~6 over 16 blocks), so cosine-to-anchors saturates and carries little
  discriminative signal. **The polar bonus hypothesis ("identity coords are MORE
  seed-stable than full vectors") is TRUE in the raw-cosine sense (0.96 vs −0.08) but
  FALSE in the anchor-similarity sense (0.36 < 0.46).** Modulus is a stable but
  information-poor identity fingerprint — good for "is this the same kind of thing"
  (a per-coordinate identity check), bad as a relative-similarity basis.
- **Retrieval consistency is weak** (top-5 Jaccard 0.22–0.29 across all variants):
  a probe's nearest anchors are only ~1.5/5 shared across seeds. Consistent with the
  marginal Pearson.

**Bottom line for the LLM-mount:** the interface is not yet a green light. The shared
geometry exists (Procrustes 0.93), but the seed-free *relative* read-out of it is too
noisy at N=64 to mount on. Next move is the M-sweep, not abandonment.

---

## 4. Sample-quality comparison (verbatim, greedy decode)

BPE decoded to text (`Ġ`→space). state_t and gold are the real chain pair; gen is the
model's greedy next-state under its prior action. **The decoder arm is the only one that
produces grammatical GLUCOSE sentences.**

**Decoder arm** (`jepa_small_v21_dec`, 657K) — fluent, structurally correct, semantically off:
```
[row6] STATE : someone b asks someone a when they can go somewhere a. someone b want(s) to go somewhere a...
       GOLD  : someone a promises someone a and someone b will go somewhere a soon. someone a promises...
       GEN   : someone a and someone b go to somewhere a (that is a place where one lives). someone a and
               someone b go to somewhere a (that is a place where one lives)...

[row0] STATE : someone a puts something a on something b.
       GOLD  : someone a did not stay in place. something a did not stay in place.
       GEN   : someone a (was) surprised by someone b. ...ed someone a (was) surprised by someone b...

[row5] STATE : someone a (who is playing catch) throws the ball. someone a throws something a.
       GOLD  : something a lands at somewhere a. something a is in the air.
       GEN   : someone a and someone b do something a. someone a and someone b do something a. ...
```

**Baseline v2** (`jepa_nano_v2`, 237K, d_dec64/1L) — token-level degeneration:
```
[row0] GOLD  : someone a did not stay in place. something a did not stay in place.
       GEN   : someone a feel(s) happy. award. ashespond. ashes. ashostigned. atespond someone a. ashes. as
[row1] GOLD  : something b blows down somewhere a.
       GEN   : someone a does something a. someone a feel(s) happy. ashespons. omeone a is at somewhere a. ashostler...
```

**Polar seed1** (`jepa_nano_v21_seed1`, 237K) — same nano decoder, same degeneration as baseline:
```
[row0] GEN   : someone a is at somewhere a. someone a is at somewhere a.eves someone a is at somewhere a.e.e is...
[row1] GEN   : someone a feel(s) happy.bocks. ther is at somewhere a. ther is a broken. ther is a new phone...
```

Takeaway: polar vs baseline are indistinguishable at the sample level (both nano
decoders). The decoder arm is a categorical jump in *form*. All three share the
content failure: repeated generic clauses, no grounding to the specific gold next-state.

---

## 5. Three capability gaps (from the probe battery, `results/jepa_v2_probe/`)

1. **Next-state retrieval barely beats chance.** Decoder-likelihood retrieval
   (`probe1_retrieval.json`): hard-pool MRR 0.117 vs chance 0.104; recall@1 0.037 vs
   0.024. The model ranks the true next-state only marginally above same-chain/NN
   distractors. The dynamics discriminate, but weakly.
2. **The discrete action is non-interpretable.** Action semantics
   (`probe3_action_semantics.json`): codebook is well-used (ppl 7.6/8, all 8 codes
   active, not degenerate) but **NMI between action id and reversible/irreversible
   keyword label is 0.004 — *below* its own shuffle baseline (0.012)**. The actions
   partition the data into 8 used buckets that align with *nothing* semantic we can
   name. This is the action-interpretability homework.
3. **Generation is generic / zero exact-match.** chrF ~0.37–0.38 and gen_exact 0.0
   across every arm; samples (§4) confirm fluent-but-wrong output. The decoder renders
   the *register* of a next-state, not the *content* of the specific transition.

Counter-evidence worth keeping: **probe2 slot-LOO flipped positive in v2** — masking any
of the 8 slots *increases* CE (all 8 constructive; v1 had every slot LOO ≤ 0). The
coarse-to-fine curve is monotone (1→8 slots: 1.86→1.46 CE). So the slot decomposition is
doing real work even though the action codes and retrieval are weak.

---

## 6. Recommended v3 recipe + next experiments

### v3 recipe (five bullets)
- **Keep the bigger decoder (d_dec 128, 2 layers).** It is the only change that bought
  a real metric (ce_true 1.39→0.95) and it fixes the BPE degeneration that makes every
  nano sample unreadable. This is non-negotiable for any demo.
- **Add an explicit InfoNCE / contrastive next-state term** alongside L_token. The
  retrieval gap (MRR ≈ chance) is the core failure; teacher-forced CE alone does not
  pressure the slots to *discriminate* the right next-state from distractors. The LOO
  flip says the slots can carry it; give them the gradient.
- **Keep polar v2.1 on (it's free and non-harmful), but stop expecting a ce_gap effect
  from it.** Demote modulus from "identity for the relative interface" to "stable
  per-coordinate kind fingerprint" — use it for the kind-head demo, not as the mount basis.
- **Drop d_noun back to 32.** dn64 bought modulus rank and nothing downstream; spend the
  param budget on the decoder and the contrastive head instead.
- **Run multi-step unroll training** (predict t+2 from t via two operator applies) to put
  real pressure on action composition — the current single-step setup lets the operator
  stay near-abelian and decorative.

### Next experiments (justified by the matrix)
- **M-sweep on the anchor experiment (N = 64 → 256 → 1024).** The Procrustes ceiling
  (0.93) minus the anchor-sim ρ (0.43) gap says the relative encoding is anchor-starved,
  not broken. This is the cheapest, highest-information next run and directly re-tests the
  mount gate. *(The LOO flip — slots now constructive — is what justifies expecting the
  extra anchors to actually add signal rather than noise.)*
- **InfoNCE ablation** (w_token only vs w_token + w_nce) measured by hard_mrr — the direct
  test of gap #1.
- **Action-interpretability homework**: condition the posterior on a held-out
  reversibility label (or cluster actions post-hoc and inspect) to see whether the 8
  codes *can* be made semantic, or whether the operator family forbids it.
- **Seed triplet for every future claim.** Given the 63% ce_gap seed spread, run ≥3 seeds
  and report the band before calling any sub-0.05-nat effect real.

---

## Artifact paths
- Anchor results: `results/jepa_matrix/staged/anchor_stability.json`
- Decoder-arm recovered diag: `results/jepa_matrix/staged/decoder_arm_diag.json`
- Param counts: `results/jepa_matrix/staged/param_counts.json`
- Samples (verbatim source): `results/jepa_matrix/staged/samples_{decoder_arm,baseline,polar_seed0,polar_seed1,dn64}.json`
- Probe battery: `results/jepa_v2_probe/probe{1_retrieval,2_slot_loo,3_action_semantics,4_samples}.json`
- Experiment scripts: `results/jepa_matrix/{anchor_stability,extract_matrix}.py`
- Source jobs: baseline `a31ca33e`, anchor-pair `f0cfcfe2`, capacity-arms `7bb602a8`,
  anchor+extract `5cc722ad` (wartable).
```
