# L_sep diagnosis — v5 single-variable ablations

Baseline: `jepa_v5_s0.json`. Each config below clones it and changes exactly ONE
substantive variable (the `eval.out_dir` rename is forced so runs don't clobber
each other and is not counted as the "one variable").

## jepa_v5_b256_s0.json — isolates BATCH SIZE
- Single change: `optim.batch_size` 64 -> 256.
- Hypothesis: L_sep is a SupCon over the in-batch positives/negatives. With B=64 and
  ~5-7 in-batch examples sharing a canonical-next-state id, the positive/negative
  pools are thin, so L_sep gradient is noisy / under-determined. Bigger batch = more
  same-canon positives and more hard negatives in the (B,B) similarity matrix.
- NO code change required. The batch loop already reads `optim.batch_size`
  (train_jepa_v2.py:695 `bs = o.batch_size`; :728 `for start in range(0, n_train - bs + 1, bs)`).
  Sampling is ALREADY chain-shuffled in triples mode (see sampler facts in the
  return message) — `chain_contiguous_perm` runs every epoch with a fresh
  `torch.randperm`, so transitions are NOT contiguous-frozen; there is no separate
  shuffle flag to set and none is needed.
- Caveat: B=256 needs GPU-memory headroom (see feasibility note). If it OOMs against
  the resident vLLM shard, fall back to B=128 (still a 2x positive-pool increase) or
  request a dedicated GPU.

## jepa_v5_para_s0.json — isolates the PARAPHRASE-POSITIVE substrate
- Single change: data + eval pointed at `data/entity_world_para/` instead of
  `data/entity_world/` (path, tokenizer, eval.labeled_dir — all three move together
  because they are one substrate, i.e. one variable: "which corpus").
- Hypothesis: L_sep wants paraphrase renderings of the SAME canonical next-state to
  collapse to one point. In the plain world each canonical state has ~one surface
  form, so "same-canon positives" are near-duplicate strings — the model can satisfy
  L_sep by surface matching, not by learning the invariance. The para world renders
  the same canonical state in varied surface forms, giving L_sep genuine
  surface-invariant positives to pull together.
- NO code change. `data/entity_world_para/` already exists with train/test splits,
  `*_labeled.jsonl` twins (so `attach_labels` -> canon ids work), and its own
  `bpe_512.json`.

## jepa_v5_world2_s0.json — isolates WORLD VARIETY (higher type/entity count)
- Single change: data + eval pointed at `data/entity_world_v2/` (the world_version=2
  substrate: >=12 types via `world_version_min:2`, 3-5 entities/chain, chain_len 6-12,
  120K train chains).
- Hypothesis: with only the v1 type set, distinct canonical next-states are too few /
  too separable, so L_sep has little to do (or collapses trivially). A higher-variety
  world produces many more distinct canonical states and richer near-collisions,
  stress-testing whether L_sep actually shapes the geometry.
- BLOCKED: the v2 DATA DOES NOT EXIST YET. The v2 GENERATOR exists and is complete
  (`scripts/generate_entity_world.py`, `world_version` machinery, all `*_v2` keys and
  `world_version_min:2` types), but `data/entity_world_v2/` has not been generated.
  world_version is hardcoded in the in-file `CONFIG` dict (generate_entity_world.py:80)
  and `main()` reads `cfg = CONFIG` (line 1205) — there is NO CLI flag. To produce the
  data, set in CONFIG: `world_version: 2` and `out_dir: "data/entity_world_v2"`, then
  `uv run python scripts/generate_entity_world.py`. It writes train/test + `*_labeled`
  twins + `bpe_512.json` to that dir, matching this config's paths. Generate first,
  then run this config. (Note: editing CONFIG is a generator-side data-prep edit, not a
  trainer/library code change.)
