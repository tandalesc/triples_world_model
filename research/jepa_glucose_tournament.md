# GLUCOSE Component Tournament — Structured Operator vs Gated-MLP Black Box

Formal synthesis of the 3v3 JEPA v3 component tournament on GLUCOSE `chain_general`
(2026-06-10). Two transition families, identical v3 chassis (d128/2L decoder, InfoNCE
next-state term `w_nce=0.25`, multi-step unroll `t→t+1→t+2`, polar conditioning on for
the operator / inert for the MLP), three seeds each, 100 epochs, mode=triples, 36,449
chain triples. Both jobs completed, exit 0. All ep100 numbers are parsed directly from
the persisted stdout of both jobs — log parsing was sufficient; the checkpoint-rerun
fallback was not needed.

- **Structured:** job `13fb923b-4922-4a45-b9b3-93c6cb508e2e` — configs `jepa_v3_s{0,1,2}`,
  results `results/jepa_v3_s{0,1,2}`, `operator_group="rotation_scale"`, 657,304 params.
- **Black-box:** job `84cd388c-030d-4cb0-b41a-15831cd331aa` — configs
  `jepa_v3_baseline_s{0,1,2}`, results `results/jepa_v3_blackbox_s{0,1,2}`,
  `operator_group="gated_mlp"` (832 transition params, no inverse, polar inert).

This is the GLUCOSE leg of the pre-registered component tournament. The cut rule is
explicit: **match-only means cut.** Parity on GLUCOSE moves the operator's entire
survival case to the entity campaign (NMI / far-OOD generalization) plus the retraction
probe.

---

## 1. The 3v3 final table (ep100 per seed + family mean ± spread)

Per-seed final-epoch (ep100) diagnostics, plus a codebook-stability pair (min posterior
v-usage perplexity over the whole run, and final codes used) parsed from the full curve.

| run | ce_true↓ | ce_gap↑ | hard_mrr | easy_mrr | chrF | v_usage_ppl | codes | prior_ppl | **min_ppl(run)** | **final_codes** |
|-----|---------:|--------:|---------:|---------:|-----:|------------:|------:|----------:|-----------------:|----------------:|
| **structured** s0 | 0.985 | 0.108 | 0.099 | 0.026 | 0.403 | 4.74 | 8 | 1.989 | 4.72 | 8 |
| **structured** s1 | 1.002 | 0.046 | 0.104 | 0.022 | 0.427 | 1.48 | 3 | 1.000 | 1.47 | 3 |
| **structured** s2 | 0.981 | 0.163 | 0.097 | 0.038 | 0.399 | 5.97 | 8 | 1.880 | 4.57 | 8 |
| **black-box** s0 | 1.007 | 0.178 | 0.059 | 0.009 | 0.368 | 6.66 | 8 | 4.108 | 1.93 | 8 |
| **black-box** s1 | 1.000 | 0.551 | 0.099 | 0.031 | 0.433 | 6.22 | 8 | 1.908 | 1.98 | 8 |
| **black-box** s2 | 0.974 | 0.099 | 0.097 | 0.029 | 0.391 | 5.46 | 8 | 1.910 | 1.93 | 8 |

### Family mean ± spread (min..max over 3 seeds)

| metric | structured mean | structured spread | black-box mean | black-box spread | \|Δmean\| | pooled within-family spread |
|--------|----------------:|------------------:|---------------:|-----------------:|----------:|----------------------------:|
| ce_true↓ | **0.989** | 0.981 .. 1.002 (0.021) | 0.994 | 0.974 .. 1.007 (0.033) | 0.005 | 0.033 |
| ce_gap↑ | 0.106 | 0.046 .. 0.163 (0.117) | 0.276 | 0.099 .. 0.551 (0.452) | 0.170 | 0.452 |
| hard_mrr | 0.100 | 0.097 .. 0.104 (0.006) | 0.085 | 0.059 .. 0.099 (0.040) | 0.015 | 0.040 |
| easy_mrr | 0.029 | 0.022 .. 0.038 (0.016) | 0.023 | 0.009 .. 0.031 (0.022) | 0.006 | 0.022 |
| chrF | 0.410 | 0.399 .. 0.427 (0.028) | 0.397 | 0.368 .. 0.433 (0.066) | 0.012 | 0.066 |
| v_usage_ppl | 4.07 | 1.48 .. 5.97 (4.49) | 6.12 | 5.46 .. 6.66 (1.20) | 2.05 | 4.49 |
| codes | 6.33 | 3 .. 8 (5) | 8.00 | 8 .. 8 (0) | 1.67 | 5 |
| prior_ppl | 1.62 | 1.00 .. 1.99 (0.99) | 2.64 | 1.91 .. 4.11 (2.20) | 1.02 | 2.20 |

ce_true is teacher-forced next-state cross-entropy (nats); the v3 d128/2L decoder holds
both families at the ~0.95–1.0 nats fluency floor established in the v2-era matrix
(`jepa_matrix_synthesis.md`: ce_true 1.39→0.95 was the one real decoder win). chrF and
exact-match (0.0 across all 6, omitted from the table) are unchanged from the v2 era:
fluent register, wrong content.

---

## 2. Statistical read — mean differences vs within-family seed spread

With three seeds per family the honest test is: does the family mean difference exceed
the larger of the two within-family seed spreads? **For every metric, it does not.**

| metric | \|Δmean\| | larger within-family spread | beyond seed noise? |
|--------|----------:|----------------------------:|:------------------:|
| ce_true | 0.005 | 0.033 | **no** |
| ce_gap | 0.170 | 0.452 | **no** |
| hard_mrr | 0.015 | 0.040 | **no** |
| easy_mrr | 0.006 | 0.022 | **no** |
| chrF | 0.012 | 0.066 | **no** |
| v_usage_ppl | 2.05 | 4.49 | **no** |
| codes | 1.67 | 5.00 | **no** |
| prior_ppl | 1.02 | 2.20 | **no** |

**Plainly: not a single metric shows a between-family difference that survives seed
noise.** The closest to "interesting" is ce_gap, where the black-box mean (0.276) is
nominally 2.6× the structured mean (0.106) — but the black-box spread alone is 0.452
(driven by s1's 0.551, a single high seed) and the structured spread is 0.117, so the
gap difference is entirely inside the seed band, exactly the seed-variance-dominated
ce_gap finding the v2 matrix pre-registered (63% spread on identical configs). v_usage_ppl
and codes look like a black-box win (8/8 codes every seed vs structured's one degenerate
3-code seed), but the structured spread spans 1.48→5.97, swallowing the 2.05 mean gap;
this is codebook seed-chaos, not an architecture effect. **Verdict: zero beyond-noise
differences on GLUCOSE.**

---

## 3. Formal verdict vs the component-tournament cut rule

**On GLUCOSE, the structured operator matches the gated-MLP black box.** Across all eight
reported axes the between-family mean difference is smaller than the within-family seed
spread; there is no metric on which the structured prior beats the black box beyond seed
noise, and none on which it loses. This is the parity outcome the pre-registration
expected.

**Per the component-tournament cut rule — match-only means cut — the structured operator's
case on THIS data is zero.** GLUCOSE `chain_general` is underdetermined for the structured
prior: the rotation/scale geometry buys nothing the verb-gated MLP cannot match, because
the task (3-bit action over fluent-but-generic GLUCOSE next-states) never exercises the
properties the structure exists to provide — invertibility, angle-additive composition,
modulus-as-identity. The decoder dominates either way (the matrix's standing result), and
the dynamics layer is, on this data, decorative for both families.

**The operator's survival case therefore rests entirely on the entity campaign + the
retraction probe**, exactly as pre-registered. Those are the axes where the structure is
load-bearing: cross-entity / far-OOD generalization measured by action NMI (does the
discrete code carry recoverable semantics?), and the retraction probe (does the invertible
operator support `undo` where the no-inverse black box structurally cannot?). Both are
in flight, with early structured leads. GLUCOSE does not adjudicate the operator; it
removes GLUCOSE from the operator's evidence ledger and hands the decision downstream.

**Data-side closing statement — prior-degeneracy is seed-chaotic, not unanimous.** The
substrate-era expectation was channel atrophy / prior degeneracy as a seed-chaotic
phenomenon on *both* architectures, and that is exactly what the prior-perplexity curves
show — but the strict "prior ppl ≈ 1.0" collapse is **not** unanimous across the 6 runs.
Only **2 of 6** runs ever touch prior ppl ≤ 1.05: structured **s1** collapses to a
degenerate 1.000 sustained from ep75→ep100 (and drops to a 3-code posterior), and
black-box **s2** momentarily touches 1.000 at ep55 before recovering to 1.91. The other
four runs keep a live prior (min prior ppl 1.54–1.83). The degeneracy is real but
**seed-chaotic and crosses both families** — one structured seed and one black-box seed —
which is the substrate finding confirmed: prior degeneracy is a seed lottery, not an
architecture property. The posterior↔prior agreement spread (structured 0.45–0.87,
black-box 0.30–0.45) is likewise within-family chaotic. No GLUCOSE signal separates the
families on prior health.

---

## 4. What GLUCOSE taught us

- **Channel dispensability on underdetermined data.** The structured transition channel
  (rotation/scale + polar) is dispensable here: a verb-gated MLP with no inverse matches
  it on every metric. When the task doesn't require invertibility or compositional
  structure, the structured prior is free but inert — it neither helps nor hurts, exactly
  as polar was "free, non-harmful" in the v2 matrix.
- **Seed variance dominates the gap metrics.** ce_gap, v_usage_ppl, codes, and prior_ppl
  all have within-family spreads that swallow the between-family mean differences. Any
  3v3 claim on these axes from a single seed pair would be noise. The pre-registered
  ≥3-seed discipline is what kept this honest — ce_gap looked like a 2.6× black-box win
  on means alone.
- **InfoNCE sits at chance.** The added next-state contrastive term (`w_nce=0.25`,
  τ=0.1) never separates positives from negatives: final L_nce ≈ 4.27 (structured) /
  4.89 (black-box) against a chance floor of log(B) ≈ 4.16 for the effective in-batch
  gallery. Correspondingly hard_mrr ≈ 0.10 and easy_mrr ≈ 0.03 — barely above the
  chance_hard 0.024 floor, and `easy_minus_hard_mrr` is negative across all 6 runs.
  InfoNCE did not close the retrieval gap the v3 recipe added it to close; on GLUCOSE the
  next-state is not discriminable from same-chain distractors at this scale.
- **Data properties > architecture for these axes.** Fluency (ce_true ~0.95), content
  failure (chrF ~0.40, 0% exact), weak-but-above-chance retrieval, and seed-chaotic prior
  degeneracy are all properties of GLUCOSE chain_general + the v3 chassis, invariant to
  the transition family. The architecture knob (structured vs black-box) moves nothing
  GLUCOSE can measure. To adjudicate the structured prior you need data that *requires*
  the structure — the entity campaign's far-OOD compositions and the retraction probe.

---

## Appendix — sources & parsing

- Structured stdout: job `13fb923b-4922-4a45-b9b3-93c6cb508e2e` (3 `Config:`-delimited
  seeds, 21 `diag_v2[ep]` points each, ep1..ep100).
- Black-box stdout: job `84cd388c-030d-4cb0-b41a-15831cd331aa` (same structure).
- All six seeds' ep100 diagnostics and full v-usage / codes / prior-ppl curves parsed
  from persisted logs; no checkpoint re-run required (logs complete, not truncated).
- Chance floor: effective in-batch InfoNCE gallery ≈ 64 ⇒ log(64) = 4.159.
