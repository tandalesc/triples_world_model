# Consolidation plan (deferred execution)

Ranked, deferred-execution plan for everything **inside the concurrent-workflow boundary**
(`src/twm/**`, `scripts/**`, `configs/**`, `tests/**`, `CLAUDE.md`). Nothing here is applied
— each item is motivation + exact moves + risk + sequencing. **Sequencing default:** apply
after the v3 workflow lands, so we don't reorganize files an in-flight branch is editing.

The repo grew a v1 → v2 → v2.1 → (v3 in flight) program in days; ~75 top-level scripts,
~18 top-level configs (+88 archived), and knowledge spread across `research/*.md`,
`results/*`, and `CLAUDE.md`. The five items below are ordered by leverage.

---

## 1. `scripts/` grouping (HIGHEST leverage)

**Motivation.** 75 top-level scripts with no grouping; the JEPA-relevant ones
(`train_jepa_v2.py`, `operator_group_fit.py`, `export_jepa_weights.py`) are buried among
~50 scripts for archived experiment families (AR decoders, flow matching, diffusion,
Kaggle converters). An agent scanning `scripts/` can't tell live from dead.

**Proposed layout** (`git mv`, preserve history):

```
scripts/
├── train/        # live trainers
│   ├── train_jepa_v2.py            # the live path
│   ├── train_chain.py              # multi-turn chain (CLAUDE.md "active")
│   └── train.py                    # closed-vocab entry
├── probes/       # diagnostics / eval that an agent reruns
│   ├── operator_group_fit.py       # (also fully-local repro headline)
│   ├── eval_chain.py
│   └── (results/jepa_*_probe/*.py stay where they are — they're owner-checkout scripts)
├── export/
│   ├── export_jepa_weights.py
│   └── export_pet_sim_figures.py
├── plotting/
│   ├── plot_3d_pca.py  plot_chain_pca.py  plot_family.py
│   ├── visualize_dynamics.py  visualize_dynamics_3d.py  visualize_architecture.py
│   ├── visualize_bottleneck.py  visualize_latent_space.py  visualize_latent_v2.py
├── data/         # converters + dataset builders + tokenizer builders
│   ├── convert_*.py (atomic, glucose, glucose_nsp, openpi, personality, propara,
│   │                 spaceship_titanic, textworld, titanic)
│   ├── build_*.py (domain_vocab, entity_world_bpe, glucose_tokenizer, jepa_bpe,
│   │               mixed_v16, pretrained_embeds)
│   ├── generate_*.py  extract_*.py  prepare_webnlg*.py  augment_webnlg_distributional.py
│   ├── download_atomic.py  merge_datasets.py  normalize_openpi_llm.py
└── archive/      # ADD the dead experiment families to the existing archive/
    ├── train_ar_autoencoder.py train_ar_dynamics.py train_ar_entity.py
    │   train_ar_frozen.py train_ar_phaseC.py train_dual_ar.py
    ├── train_flow.py train_flow_autoenc.py train_flow_dynamics.py
    │   train_diffusion_model.py infer_diffusion_model.py
    ├── eval_beam_search.py eval_denoise_steps.py eval_flow_multisample.py
    │   eval_observability.py eval_soft_predictions.py eval_v2_with_v6_metric.py
    ├── benchmark_family.py benchmark_llm.py run_mlp_baseline.py semantic_eval_all.py
    ├── check_embed_magnitudes.py check_exact_match.py compare_embeddings.py
    │   diagnose_coldstart.py collect_arc_traces.py extract_wikipedia.py
    │   generate_cedric_dataset.py generate_mode_warmup.py inference_tool.py
    └── predict_*.py (personality, spaceship_titanic, titanic) — Kaggle one-offs
```

**Risk.** Medium. Many archived scripts have relative `sys.path` / data-path assumptions and
hardcoded `scripts/<x>.py` references in `CLAUDE.md` config comments. Mitigation: grep for
each moved filename across the repo before moving; update `CLAUDE.md` script paths in the
same change (see item 5). The live trainers (`train_jepa_v2.py`, `train_chain.py`) compute
`Path(__file__).resolve().parent.parent` for the src import — moving them one level deeper
(`scripts/train/`) breaks that; bump to `parents[2]` or add a `scripts/_pathfix.py`.

**Sequencing.** After v3 lands. v3 will add `scripts/train_jepa_v3.py` (or extend v2) — do
the grouping once v3's trainer name is final, so we move it into `train/` in the same pass.

---

## 2. Config namespace cleanup

**Motivation.** Three conventions coexist: `configs/*.json` (mixed AR/flow/jepa),
`configs/jepa/*.json` (v2.1), `configs/archive/*.json` (88 historical). The JEPA v2 configs
are split across `configs/` (`jepa_nano_v2*.json`) and `configs/jepa/` (`jepa_*v21*.json`)
with no rule for which goes where. The `jepa_mini_v2.json` even carries stale v1 loss keys.

**Proposed convention.** One directory per live family; everything else archived.

```
configs/
├── jepa/         # ALL live JEPA configs (move jepa_nano_v2*.json here from configs/)
│   ├── jepa_nano_v2.json  jepa_nano_v2_smoke.json  jepa_nano_v2_m32.json
│   ├── jepa_nano_v21.json  jepa_nano_v21_smoke.json  jepa_nano_v21_seed1.json
│   ├── jepa_nano_v21_dn64.json  jepa_small_v21_dec.json
│   └── jepa_mini_v2.json   # FIX: strip v1 keys (w_div, w_scale_reg, operator_fit_pass2) or archive
├── chain/        # train_chain.py configs (currently only in archive/)
├── test_snapshot.json      # keep at root: the closed-vocab smoke
└── archive/      # ar_*.json, dual_ar_*.json, flow_*.json, arc_v1.json -> here (with the 88)
```

**Exact moves:** `git mv configs/jepa_nano_v2*.json configs/jepa/`; `git mv configs/{ar_*,
dual_ar_*,flow_*,arc_v1}.json configs/archive/`. Then update `docs/REPRODUCING.md` and any
script defaults that name `configs/jepa_nano_v2_smoke.json` (the README quickstart and the
smoke command both reference it — update in lockstep).

**Risk.** Low-medium. Config paths are passed on the CLI, so moving them only breaks
documented commands (README, REPRODUCING.md, AGENTS.md) and the `CLAUDE.md` "Active configs"
list. All are editable in the same change. Verify no script hardcodes a default config path.

**Sequencing.** Bundle with item 1 (same "reorg pass"). Low risk to do early IF the README
quickstart path is updated simultaneously — but defer to keep the quickstart stable for the
v3 author.

---

## 3. Single entry point (a thin `twm` CLI or justfile)

**Motivation.** Each train script reads its own incompatible config schema and is invoked as
`uv run python scripts/<x>.py configs/<y>.json`. There is no discoverable "what can I run."
A single entry point makes the live surface obvious to humans and agents.

**Proposed:** a `justfile` (lighter than a CLI; no new code in `src/`) at repo root:

```
# justfile
smoke:   uv run python scripts/train_jepa_v2.py configs/jepa/jepa_nano_v2_smoke.json
train cfg='configs/jepa/jepa_nano_v2.json':  uv run python scripts/train_jepa_v2.py {{cfg}}
opfit:   uv run --with sentence-transformers --with scikit-learn --with matplotlib python scripts/operator_group_fit.py
probe ckpt out:  uv run python results/jepa_v2_probe/probe_v2_battery.py --ckpt {{ckpt}} --out {{out}}
export ckpt out='':  uv run python scripts/export_jepa_weights.py {{ckpt}} {{out}}
demo:    cd demo/pet_simulation && python -m http.server 8080
test:    uv run pytest tests/jepa -q
```

Alternative: a `[project.scripts]` console entry `twm = "twm.cli:main"` with subcommands —
more work, lands inside `src/twm/`, and needs the v3 trainer to be a callable not just a
`__main__` script. The justfile is the recommended low-risk first step; promote to a real
CLI only if v3 unifies the config schemas.

**Risk.** Low (justfile is additive, breaks nothing). The CLI alternative is medium (touches
`src/`, needs schema unification).

**Sequencing.** After items 1+2 so the targets point at final paths. The justfile targets
double as living documentation of the live surface.

---

## 4. Test organization

**Motivation.** `tests/jepa/` is healthy (11 v2 tests + 4 legacy v1 tests under
`tests/jepa/legacy/`), but there is no top-level test runner convention, no closed-vocab
core tests visible, and `data/arc-prize-2026.../tests/` (vendored third-party tests) pollute
a naive `pytest` / `find test_*` discovery.

**Proposed:**
- Add `tests/__init__.py` + a `pyproject` `[tool.pytest.ini_options]` block with
  `testpaths = ["tests"]` and `norecursedirs = ["data", ".venv", "node_modules"]` so
  vendored ARC tests are never collected.
- Keep the `tests/jepa/legacy/` split — it correctly mirrors `src/twm/jepa/legacy/`.
- Add a smoke-level test that imports `build_jepa_model_v2` and runs one forward/backward on
  a 2-pair batch (guards the live contract the way the trainer's import-guards imply).
- Document `just test` (item 3) as the canonical invocation.

**Risk.** Low. Pure addition + a pytest config block. Only risk is the `norecursedirs`
glob missing a vendored test dir — verify against `find . -name test_*.py`.

**Sequencing.** Independent; can land anytime. Do the `norecursedirs` fix early (it's a
papercut for anyone running `pytest`), defer the new smoke test until the v3 contract is
settled.

---

## 5. `CLAUDE.md` revision (shrink toward pointers into `docs/`)

**Motivation.** `CLAUDE.md` carries the full architecture, the wartable submit pattern, the
VAE/VQ gotchas, the config list, and the data-file gotchas — much of which is now duplicated
or superseded by `docs/AGENTS.md`, `docs/REPRODUCING.md`, and the JEPA synthesis. It also
still says `gpu_vram_min_gb=22` (dead — see AGENTS.md) and lists Sprint-5/6 open-vocab as
"current work" though the live path is JEPA v2/v2.1. It should shrink to a stable surface +
pointers.

**Proposed revision (draft — NOT applied):**

> # Triple World Model (TWM)
>
> ## What this is
> A minimal world model that learns state dynamics over structured (entity, attribute,
> value) triples. The live research line is **JEPA latent actions** (v2/v2.1): nouns as
> points in complex latent space, a discrete verb applied as per-block complex
> multiplication, inferred unsupervised from state pairs. The older closed-vocab transformer
> core is the lineage (compositional generalization, the pet-sim demo).
>
> ## Start here
> - Human overview + quickstart: `README.md`
> - Reproduce any result (commands, data, expected numbers): `docs/REPRODUCING.md`
> - Agent operations (wartable jobs, probes, diagnostics, retention): `docs/AGENTS.md`
> - Latest synthesis (what we actually know): `research/jepa_matrix_synthesis.md`
> - Designs: `research/jepa_v2_latent_actions.md`, `research/jepa_v21_polar.md`
>
> ## Active path
> - Trainer: `scripts/train_jepa_v2.py` ; configs: `configs/jepa/*.json`
> - Data (in-repo): `data/glucose/chain_general_{train,test}.jsonl`, `jepa_bpe_512.json`
> - Branch: `feature/glucose-converter` (not `main`)
>
> ## Standing conventions (load-bearing — keep here)
> - Per-eval checkpoint retention (never overwrite earlier evals).
> - V-relative thresholds; never trust usage statistics without a shuffle/permutation baseline.
> - ≥3 seeds before any sub-0.05-nat claim (ce_gap seed spread is ~63%).
> - Device order cuda -> mps -> cpu (auto).
> - wartable: gpu_vram_min_gb=3 (the 22GB card is gone — vLLM is tensor-parallel). See AGENTS.md.
>
> ## Closed-vocab / open-vocab archive
> The VAE/VQ/diffusion/AR/flow gotchas and Sprint-5/6 results are preserved in
> `research/sprint*.md` and `docs/CONSOLIDATION_PLAN.md`. They are not the live path.

Everything dropped from `CLAUDE.md` (VAE gotchas, VQ gotchas, config tables, profile tables)
moves verbatim into `research/` docs or stays in the sprint files it came from — nothing is
deleted, only relocated and pointed to.

**Risk.** Low-medium. `CLAUDE.md` is the agent's primary context; shrinking it risks dropping
a load-bearing convention. Mitigation: the draft above explicitly *keeps* the four standing
conventions inline and only relocates reference material. Diff carefully; have one agent run
a smoke + probe using only the revised `CLAUDE.md` + `docs/` before committing.

**Sequencing.** LAST. Apply only after items 1–4 land and `docs/AGENTS.md` +
`docs/REPRODUCING.md` are battle-tested, since the revision points *into* them — the pointers
must be correct before `CLAUDE.md` delegates to them. Also update the stale
`gpu_vram_min_gb=22` and the Sprint-5/6 "current work" framing in the same pass even if the
fuller shrink is deferred (those two are actively misleading today).

---

## Apply-after-v3 sequencing summary

1. v3 trainer name finalizes →
2. `scripts/` grouping (item 1) + config namespace (item 2) in one reorg pass →
3. `justfile` entry point (item 3) pointing at the new paths →
4. pytest `norecursedirs` fix early (item 4, independent); new smoke test with v3 →
5. `CLAUDE.md` shrink (item 5) once docs are proven.

Quick wins safe to do **before** v3 (low risk, high papercut relief): the pytest
`norecursedirs` block (item 4), and the two `CLAUDE.md` factual fixes
(`gpu_vram_min_gb=22 → 3`, retire Sprint-5/6 "current work" framing) from item 5.
