# Research index

Design docs, syntheses, and experiment logs. Read top-down: synthesis first, then designs,
then the historical sprint logs.

## Live JEPA latent-action line (current)

- **[jepa_matrix_synthesis.md](jepa_matrix_synthesis.md)** — **start here.** The latest
  numbers-first synthesis: the v2/v2.1 experiment matrix, per-arm verdicts, the anchor /
  LLM-mount gate, the three capability gaps, and the v3 recipe.
- **[jepa_v2_latent_actions.md](jepa_v2_latent_actions.md)** — v2 design: unsupervised
  latent actions + token-space grounding. Includes the probe-verified v1 failure analysis.
- **[jepa_v21_polar.md](jepa_v21_polar.md)** — v2.1 design: polar decomposition
  (modulus = identity, phase = state); the verb as per-block complex multiply.
- **[jepa_operator_v1_design.md](jepa_operator_v1_design.md)** — v1 design (the refuted
  predictive-verb path); kept for the operator algebra and the honest pet-framing scope.
- **[operator_group_fit.md](operator_group_fit.md)** — the empirical fit that chose the
  `rotation_scale` operator group (fully-local experiment; see `docs/REPRODUCING.md` §1).

## Foundations / closed-vocab lineage

- **[architecture.md](architecture.md)** — full architecture + file map + diagrams.
- **[theoretical_foundations.md](theoretical_foundations.md)** — the geometric framework.
- **[references.md](references.md)** — papers and systems referenced.

## Historical sprint logs (open-vocab / decoder experiments)

- **[sprint3_diffusion_decoder.md](sprint3_diffusion_decoder.md)** — diffusion expansion.
- **[sprint4_config_driven_training.md](sprint4_config_driven_training.md)** — the
  config-driven trainer + pet-sim dynamics analysis.
- **[sprint5_vae_bottleneck.md](sprint5_vae_bottleneck.md)** — open-vocab IO, VAE bottleneck.
- **[sprint6_chain_dynamics.md](sprint6_chain_dynamics.md)** — multi-turn chain dynamics.
- **[kaggle_titanic.md](kaggle_titanic.md)** — Kaggle structured-prediction side quest.

## How to reproduce

Every result above maps to a command in **[../docs/REPRODUCING.md](../docs/REPRODUCING.md)**.
Agent operations (jobs, probes, diagnostics) are in
**[../docs/AGENTS.md](../docs/AGENTS.md)**.
