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

## Latent Geometry Assets

- Micro:
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/latent_flow.png`
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/eigenspectrum.png`
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/pre_latent_embeddings.png`
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/mode_conditioned_post_latent.png`
  - `results/cedric_mode_probe_v2_micro/analysis_snapshots/mode_delta_vectors.png`
- Mini:
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/latent_flow.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/eigenspectrum.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/pre_latent_embeddings.png`
  - `results/cedric_mode_probe_v2_mini/analysis_snapshots/mode_conditioned_post_latent.png`
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

## Exhaustive Mini Sweep (CPU)

Using enumerated state combinations and PCA projections:

- `results/cedric_mode_probe_v2_mini/analysis_exhaustive/pre_by_task.png`
- `results/cedric_mode_probe_v2_mini/analysis_exhaustive/post_by_mode.png`

Observed:
- PRE space shows structured gradients by task.
- POST space shows substantial mode-dependent reshaping.

### Note on transition graph mapping

A full 8100-state graph over `(user, task, energy, focus, calendar, urgency, mode)` did not close directly because model outputs omit `user` in predicted next-state triples.

Reduced-state mapping over `(task, energy, focus, calendar, urgency, mode)` yields:
- states: 1620
- mapped edges: 980
- self loops: 908
- components: 1548
- largest component: 4

This suggests many locally stable regions with sparse cross-region transitions under current learned policy.

## Practical Recommendation

For this domain:
- Use **Mini** as default reasoning core.
- Keep **Micro** as ultra-light fallback where footprint matters more than nuanced mode control.
