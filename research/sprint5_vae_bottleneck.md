# Sprint 5: VAE Bottleneck, Latent Collapse & Open-Vocab Dynamics

## Goal

Add a VAE bottleneck with role-conditioned priors to the compressor, then investigate and fix latent space collapse during staged training. Evolved into: drop the VAE entirely, solve collapse with joint training + spectral penalty, push open-vocab QA as far as possible.

## Key Results

| Experiment | Architecture | IO tok_acc | IO exact | QA tok_acc | Notes |
|-----------|-------------|-----------|----------|-----------|-------|
| v21 (VAE) | d64, staged | — | — | — | Collapses to 1D by ep50 |
| v25 (VAE, clean mu) | d64, joint | 73% | 19% | 26% (v27) | First working joint training |
| v30 (no VAE) | d64, joint, t12 | 95% | 61% | — | VAE is overhead |
| v31 (+ tokfix) | d64, joint, t12 | 90%* | 83%* | 26% | *eval bug masked true perf |
| v33b (micro) | d32, joint, t12 | 83% | 34% | 8% | 312K params viable |
| v34 sweep (t16) | d32, joint, t16 | 85% | 36% | 5% | More triples = higher IO ceiling |
| **v35 (best IO)** | **d64, joint, t16** | **99.3%** | **96.9%** | **36.8%** | **Best model. QA geometry collapses** |
| v36 (frozen IO) | d64, frozen, t16 | 98.6% | 96.9% | ~8% | Preserves geometry, QA can't learn |
| v37 (joint all) | d64, joint, t16 | — | — | — | **PENDING: server died** |

## Architecture Evolution

### Phase 1: VAE + Collapse Discovery (v21-v24)

Added VAE bottleneck with role-conditioned priors. PCA visualization confirmed collapse to 1D manifold during IO training. Root cause: expander only needs 1D for identity reconstruction, KL flattens everything else.

### Phase 2: Joint Training + Spectral Fix (v24-v27)

Joint training (dynamics from epoch 1 with zero-init gate) provides back-pressure against collapse. Spectral penalty penalizes PC1 variance ratio. Together they maintain 8+ effective dimensions (spectral 0.04 vs 1.0 for staged).

**Critical bug found:** Spectral loss was measuring z (noisy VAE sample) instead of mu (structure). Noise masked collapse — reported 0.026 when real value was 1.00.

### Phase 3: Drop the VAE (v30-v31)

VAE was pure overhead. Without it: no spectral noise masking, no double-noise train/eval mismatch, no z/mu divergence. v30 hit 95% tok_acc in 150 epochs (5-10x faster than VAE).

**Eval dynamics routing bug (v31):** Joint training routes IO through dynamics with mode=0, but eval skipped dynamics for TextDataset. Length head saw different bottleneck at eval vs train → systematic N-1 length on questions. Fix: route eval through dynamics when model has `forward_dynamics`.

**Training schedule discovery (v33b):** t_min=0.5/0.3 with patience 200 (vs 0.7/0.4 with 100) dramatically accelerates learning. Smaller models need lower noise to resolve tokens. v35 hit 92% exact at ep180 where v31 needed 400+ epochs for 83%.

### Phase 4: Triples Sweep (v34)

Tested max_triples 4/8/12/16 at d_model=32. IO phase 2 results:

| Triples | Bottleneck dims | IO tok | IO exact | QA tok | Params |
|---------|----------------|--------|----------|--------|--------|
| 4       | 384d           | 57%    | 3%       | 8%     | ~220K  |
| 8       | 768d           | 74%    | 9%       | 7%     | ~260K  |
| 12      | 1152d          | 83%    | 34%      | 8%     | ~312K  |
| 16      | 1536d          | 85%    | 36%      | 5%     | ~360K  |

More triples = higher IO ceiling. Big jump at 8→12. QA at d32 is capacity-limited regardless of triples — needs d64+ with 4+ dynamics layers.

### Phase 5: v35 (Best IO) and QA Geometry Collapse

v35 combined all improvements: d64, 16 triples, v33b schedule. **99.3% tok_acc, 96.9% exact** in IO — near-perfect reconstruction through a 64d bottleneck with dynamics online.

**QA failure mode:** When dynamics training starts with all params unfrozen, compressor geometry collapses. PCA shows PC1 variance jumping 0.28→0.57 in 190 epochs. Dynamics finds cheap shortcut (mode separation via geometric collapse) instead of learning transforms within the rich space. Identity drops 98%→70%. QA plateaus at 36.8% tok with 0% exact.

### Phase 6: Addressing Geometry Collapse (v36-v37)

**v36 (frozen compressor/expander):** Preserves geometry (spectral 0.04→0.10, identity ~90%) but QA only reaches ~8%. The rigid frozen space can't support the transforms dynamics needs to learn.

**v37 (joint identity+QA from epoch 1):** The core insight: IO-first training builds a 1D+noise manifold optimized for "store and retrieve." The only cheap modification to this space that separates modes is geometric collapse. If QA is present from epoch 1, the compressor must build a space supporting both reconstruction AND transformation from the start — a fundamentally different optimization target.

Config: single stage using QA dataset (has both identity+QA mixed), joint=true, patience 300. **NOT YET RUN — server hardware failure before execution.**

## Files Changed

- `src/twm/training_eval.py` — Eval dynamics routing fix, latent snapshot system
- `src/twm/training_losses.py` — Spectral loss on mu not z, expander conditions on mu
- `src/twm/text_dynamics_model.py` — Input residual (bottleneck + delta), zero-init gate
- `src/twm/text_compressor.py` — Expose mu in vae_info
- `src/twm/domain_bpe.py` — Tokenizer fix (collapse spaces before punctuation)
- `src/twm/modules.py` — Zero-init output gate on TransformerDynamics
- `src/twm/trainer.py` — Joint training, freeze/unfreeze support
- `src/twm/training_config.py` — StageConfig.joint, freeze, unfreeze fields
- Configs: v22-v37 in `configs/`

## PCA Videos

Local copies in `results/spectral_comparison/`:
- `v28_pca_evolution.mp4` — d128 VAE (beautiful geometry, broken length)
- `v30_pca_evolution.mp4` — first no-VAE run
- `v31_pca_evolution.mp4` — tokfix run through QA
- `v35_pca_evolution.mp4` — best IO, shows QA geometry collapse

## What's Next

1. **Run v37** (`configs/v37_joint_all.json`) — joint identity+QA from epoch 1. Key hypothesis test.
2. If v37 works, scale to d_model=128 for capacity test.
3. If v37 fails, the bottleneck representation fundamentally can't support QA transforms at d64 — need architectural changes (more capacity, different bottleneck structure, or auxiliary QA signal).
