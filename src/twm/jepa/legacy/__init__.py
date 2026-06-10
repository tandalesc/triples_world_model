"""Frozen v1 JEPA baseline (demoted, not deleted).

These modules reproduce the v1 predictive-verb negative result that motivated v2
(`results/jepa_nano_probe/`, `results/jepa_nano_viz/REPORT.md`). Nothing on the
live v2/v2.1 path imports from here. The live `diagnostics.py` borrows only the
small numeric helpers (`_effective_rank`, `_cosine_sim_matrix`, `_to_numpy`) from
`diagnostics_v1`.

  model_v1.py       — Readout/Predictor/_EncoderReadout (also lifted into the live
                      model.py) + the dead JEPAOperatorModel verb-predict path.
  losses_v1.py      — the v1 JEPALoss aggregator + L_div helpers (usage_entropy,
                      spread_penalty). DEAD on the live path; kept only to run v1.
  diagnostics_v1.py — v1 eval_diagnostics + reusable numeric helpers.
"""
