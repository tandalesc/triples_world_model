# AGENTS.md

Universal agent guide for the Triple World Model repo. For full project context (architecture, results, gotchas, design decisions), read `CLAUDE.md` — the content there is tool-agnostic and the source of truth.

This file documents only what's most likely to bite an agent that doesn't know the project: **the training workflow is remote, not local**.

## Where to Start Next Session

If you're picking up cold, the live thread is the TextWorld chrF=49 wall investigation. Read in this order:

1. `CLAUDE.md` — "Where We Are" section at the top. Current state and proposed next steps.
2. `memory/project_v18_generation_wall.md` (Claude memory) — full experiment log, what's been tried, what's been ruled out.
3. `research/sprint6_chain_dynamics.md` — architecture context for the multi-turn chain dynamics that produced the latest results.

The active branch is `feature/glucose-converter`. The latest experiment (v4 entity-VQ) finished tied with baseline; the next experiments are cheap diagnostics, not new architecture work — see CLAUDE.md for the proposed sequence.

## TL;DR

- Active branch: `feature/glucose-converter` (not `main`)
- Training runs on a homelab GPU server via the `wartable` MCP, not locally
- Server checkout: `~/triples_world_model_Glucose`
- Pattern: edit locally → commit + push → submit wartable job → subscribe to logs
- GPU routing: `gpu_vram_min_gb=22` is required to land on the training card

## Training Workflow

### 1. Make changes locally

Edit code/config on `feature/glucose-converter`. New experiments get a config file in `configs/` (see `dual_ar_v2_contrastive.json` for shape). Train scripts are config-driven; each script reads its own schema.

### 2. Commit + push

```bash
git add <files> && git commit -m "<message>"
git push origin feature/glucose-converter
```

The server pulls from this branch — no push, no run.

### 3. Submit a wartable job

```
mcp__wartable__submit_job(
    name="<experiment-name>",
    command=(
        "cd ~/triples_world_model_Glucose && "
        "git fetch origin feature/glucose-converter && "
        "git reset --hard origin/feature/glucose-converter && "
        "rm -rf <out_dir> && "
        "uv run python scripts/<train_script>.py configs/<config>.json"
    ),
    gpu_count=1,
    gpu_vram_min_gb=22,
    tags=["<family>", "<variant>"],
)
```

The reset-hard ensures the server is at exactly the pushed commit. `rm -rf <out_dir>` clears prior checkpoints when starting fresh; omit if resuming.

### 4. Subscribe to filtered logs

```
mcp__wartable_channel__subscribe_job_logs(
    job_id=<id>,
    interval_seconds=600,
    pattern="Epoch|loss|tok_acc|chrF|contrastive|Error|Traceback|consist",
    tail_lines=30,
)
```

Filter regex matters — without it, raw log volume floods the channel. Auto-unsubscribes on job completion.

### 5. Inspect results

Job outputs land in `~/triples_world_model_Glucose/results/<out_dir>/` on the server. Pull with `mcp__wartable__download_file` if needed for local plotting.

## Local vs. Remote

| Task | Where |
|------|-------|
| Editing code, configs, docs | Local |
| Running tests / type checks | Local |
| Tiny closed-vocab CPU experiments | Local OK |
| Training any open-vocab / TextWorld / GLUCOSE config | **Remote (wartable)** |
| Inspecting checkpoints from real training runs | Remote (download via `mcp__wartable__download_file`) |

## Data File Gotcha

Configs reference data files (e.g. `data/tw_all_train.jsonl`, augmented WebNLG, dual-memory chains) that exist **only on the GPU server**. The local checkout holds the smaller datasets (`data/glucose/*`, `data/propara_*.jsonl`, `data/openpi_*.jsonl`). A config that runs on the server will fail locally at the dataset loader — that's expected, not a bug.

## What NOT to do

- Don't run training scripts locally for any config that touches TextWorld, augmented WebNLG, or chain datasets — data isn't there
- Don't push directly to `main`. Active work is on `feature/glucose-converter`
- Don't skip `gpu_vram_min_gb=22` — without it the job may land on the wrong GPU
- Don't submit a job without first pushing the commit. The server resets to `origin/feature/glucose-converter`, so unpushed local changes won't run
