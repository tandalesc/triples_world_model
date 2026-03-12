# Cedric Mode Probe v2 — Findings (Micro vs Mini)

I ran a focused geometry + behavior comparison on a synthetic-but-structured assistant dataset to test whether Mini is just larger than Micro, or qualitatively different in latent dynamics.

## Dataset

- Path: `data/cedric_mode_probe_v2`
- State variables: `#mode, task, energy, focus, calendar, urgency` (+ user during generation)
- Modes: `identity, query, solve, advance`
- Splits:
  - train: 3000
  - test_comp: 450
  - test_seen: 224
  - test_context: 145

## Checkpoints

- Micro: `results/cedric_mode_probe_v2_micro`
- Mini: `results/cedric_mode_probe_v2_mini`

## Accuracy Summary

- **Micro**: comp/context F1 ~0.91, much lower exact-match on harder splits (comp ~0.44, context ~0.39)
- **Mini**: 1.00 F1 and exact-match on current v2 splits

## Key Probe Visuals (what this section is testing)

These are the primary plots used to explain the micro-vs-mini geometry difference for mode conditioning:

- **Pre-dynamics latent embeddings**
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/pre_latent_embeddings.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/pre_latent_embeddings.png`
- **Post-dynamics latent space by mode**
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/mode_conditioned_post_latent.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/mode_conditioned_post_latent.png`
- **Mode delta vectors (relative to identity)**
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/mode_delta_vectors.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/mode_delta_vectors.png`

## Interpretation

- Micro behaves like a compressed approximation with stronger mode entanglement.
- Mini exhibits clearer regime structure and more coherent mode-conditioned transport.
- In practice, Mini appears to learn a shared manifold with stronger local operators for mode transitions.

## Quantitative Mode-Delta Signal

From `mode_delta_stats.json` (delta vectors relative to identity):

- Micro cosine means:
  - query↔solve: **0.158**
  - query↔advance: **0.569**
  - solve↔advance: **0.477**
- Mini cosine means:
  - query↔solve: **0.533**
  - query↔advance: **0.697**
  - solve↔advance: **0.531**

Mini’s operators are more coherent/consistent across sampled states.

## Additional analysis notes

An exhaustive mini sweep was run separately for internal validation on CPU, but is intentionally not the centerpiece of this findings note. The core argument here is carried by the paired probe visuals above (pre-latent, post-by-mode, and mode-delta vectors) that directly test mode-operator structure in micro vs mini.

## Practical Recommendation

For this domain:
- Use **Mini** as default reasoning core.
- Keep **Micro** as ultra-light fallback where footprint matters more than nuanced mode control.
