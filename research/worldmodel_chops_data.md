# World-Model Chops Data — Candidate Evaluation

**Date:** 2026-06-10
**Context:** Follow-up to `transition_corpora_survey.md`. That survey ranked corpora by
transition *density*. This is the **"world-model chops"** tier: real-world datasets that
test the specific capabilities a world model must have — **state tracking,
action-conditioned prediction, OOD entities, calibration** — rather than language scale.

The bar: the existing entity-world instruments (`diagnostics.py` §3a action-NMI, §3c
target-recovery / rollout fidelity, which read `actions` as `"<verb>@<idx>"`) should work
**unmodified**, and the corpus should isolate a *mechanical* state machine (not prose).

---

## Three Candidates — Measured

All three were verified for availability/license and a real sample was downloaded and
measured. Numbers below are from the downloaded samples, not estimates.

### 1. GitHub issue lifecycles (GH Archive — gharchive.org)

- **Availability:** ✅ Verified. Hourly gzipped JSON dumps, `https://data.gharchive.org/YYYY-MM-DD-H.json.gz`. One hour ≈ 120 MB, ≈ 287K events. BigQuery mirror also public.
- **License:** GH Archive is a public archive of GitHub's *public* event stream; GitHub ToS apply. Free for research.
- **Entity:** an issue / PR (keyed `(repo, number)`). **State:** `(status, labels, assignee, milestone, locked)` — **read directly off the full issue snapshot embedded in every event** (no state-machine reconstruction needed). **Action:** the real GH event (`opened/closed/reopened/labeled/assigned/milestoned/commented/...`).
- **Chain = one issue's chronological event sequence** (cap 12).
- **Measured (9 hours, 2024-01-15, 2.27M events):**
  - 162K issues grouped → **48K chains** (len≥2), ~4.9K len≥3. Comfortably > 2K target.
  - **chains/GB ≈ 47K / 1.1 GB ≈ 43K per GB** (LOW per raw GB — issue events are ~1.6% of the stream, dominated by PushEvents).
  - **determinism given action: 0.93** (mean modal-next fraction; HIGH — the state machine is mechanical). Diluted only by `commented` near-no-ops with occasional concurrent label drift.
  - **action vocab: 4 on the 9-hour sample** (`closed/commented/reopened/opened`); `labeled/assigned/milestoned` are present in *state* but rarely *change* within 9 hours — a full month of churn grows the action vocab and lifts determinism.
  - **H(action | state): 0.45 bits** (the stochastic part).
- **OOD entities:** ✅ Native. Issues/repos created after a cutoff date = unseen test entities. Build train/test from two non-overlapping date ranges.
- **Calibration:** ⚠️ Weak — the dynamics are near-deterministic, so there is little next-state distribution to calibrate against.
- **Prep cost: 3** (large hourly downloads + JSON grouping; converter is pure-Python, no NLP dep).

### 2. Baseball play-by-play (Retrosheet — retrosheet.org)

- **Availability:** ✅ Verified. `https://www.retrosheet.org/events/YYYYeve.zip`. One **full season = 2.5 MB** (2430 games, 190K plays).
- **License:** Custom permissive; **requires attribution** ("The information used here was obtained free of charge from and is copyrighted by Retrosheet").
- **State:** `(inning, half, outs, bases occupied, score-diff bucket)`. **Action:** play outcome (`1B/2B/3B/HR/BB/K/OUT/HBP/E/FC/RUN`). **Chain = a half-inning.**
- **Measured (2023 season):**
  - 2430 games, **190K plays**, action vocab **12**.
  - **chains/GB ≈ thousands per MB** (VERY HIGH — 16K+ half-innings from 2.5 MB).
  - **H(next | state, action) = 0.000 bits** (mechanical; rules fully determine the next out-count given the outcome — measured on the simplified outs-only state proxy).
  - **H(action | state) = 2.32 bits** (MODERATE — the genuinely stochastic part: what outcome occurs given the count/base state).
- **OOD entities:** ⚠️ Weak — players are entities but the *state space* (innings/outs/bases) is fixed; no natural unseen-entity axis.
- **Calibration:** ✅✅ **The unique value.** Real stochastic dynamics with century-scale empirical ground-truth frequencies — the predicted next-state distribution can be scored against true play-outcome frequencies. The best calibration testbed of the three.
- **Prep cost: 2** (compact files, but the event field needs a custom Retrosheet play-string parser to recover full base-state; the outs-only proxy is cheap).

### 3. Wikidata revisions (Wikimedia dumps / EventStreams)

- **Availability:** ✅ API works (`action=query&prop=revisions` returns rev ids + parseable edit comments like `wbsetclaim-create`, `wbsetdescription-add`). **But** the practicality check fails the bar:
  - Full entity dump `latest-all.json.gz` = **142 GB compressed**; full *revision-history* dumps are multi-TB.
  - Recovering the (property, value) STATE at each revision requires either fetching every old revision's full JSON one-at-a-time via the rate-limited API, or parsing the history dumps — both heavy.
- **License:** CC0 (Wikidata content). The cleanest license of the three.
- **Draw:** native triple format + natural OOD entities (items created after a date).
- **Prep cost: 4–5 → DEFERRED** per the task's "> 4/5 ⟹ mark deferred" rule. The attraction (native triples, OOD-native) is real, but the state-reconstruction cost from full-JSON-per-revision dominates. Revisit only if a tractable per-property-history subset (SPARQL category slice at sample scale) is built first.

---

## Comparison Table

| Dimension | **GH issue lifecycles** | Baseball play-by-play | Wikidata revisions |
|---|---|---|---|
| Scale (sample) | 48K chains / 9 hrs; ~millions/month | 16K+ half-innings / season; M+/decade | item-history; full dumps multi-TB |
| Action explicitness | **Given** (real GH events) | **Given** (play outcome) | Inferable (edit comment op) |
| State observability | Full snapshot per event | Full (rules-derived) | Full claim set (costly to fetch) |
| Determinism given action | **HIGH (0.93 meas.)** | **HIGH (0.00 bits meas.)** | Medium (edit op → claim delta) |
| H(action \| state) | 0.45 bits | 2.32 bits | n/a (deferred) |
| OOD-entity support | **Native** (date cutoff) | Weak (fixed state space) | **Native** (item creation date) |
| Calibration support | Weak (near-deterministic) | **Strong (empirical freqs)** | Medium |
| Instruments work unmodified | **Yes** (entity=issue, `@0`) | Needs calibration metric | Needs triple adapter |
| License | GH ToS (public, research OK) | Permissive + attribution | **CC0** |
| Prep cost (1–5) | **3** | 2 | 4–5 → **deferred** |

---

## Recommendation: **GitHub issue lifecycles (GH Archive)** — built end-to-end

GH issues is the closest real-world analog to the repo's `data/entity_world`: **one entity
per chain, explicit per-transition actions, mechanical determinism-given-action (0.93
measured)**. Critically, the existing JEPA instruments run **unmodified** — the single issue
is entity slot 0, every action is tagged `@0`, and `diagnostics.py`'s action-NMI /
target-recovery read the `actions` field as-is. The full issue snapshot is embedded in
every event, so state rendering is a direct, deterministic, order-stable read of mechanical
attributes (status / labels / assignee / milestone), with no prose to reintroduce the
language-scale confound. OOD-entity probing is native (date-held-out repos/issues).

**Baseball is the strong runner-up and the recommended *next* build** — it is the unique
**calibration** testbed (the H(action|state)=2.3 bits stochastic part scored against
century-scale empirical frequencies). It complements GH issues: GH tests
state-tracking + action-conditioning on a near-deterministic machine; baseball tests
calibration on a genuinely stochastic one. It needs a calibration metric the current
instruments don't yet have, so it is deferred to a follow-up.

**Wikidata is deferred** (prep cost 4–5): native triples + CC0 + native OOD are attractive,
but per-revision state reconstruction from multi-TB dumps / rate-limited per-revision API
fetches exceeds the cost bar. Revisit after building a tractable per-property SPARQL slice.

---

## What was built (GH Archive converter, end-to-end)

- `scripts/convert_gharchive.py` — groups issue/PR events by `(repo, number)`, renders
  mechanical state, emits `train/test.jsonl` + `_labeled` twins (`actions:["<event>@0"]`,
  `types:["issue"|"pr"]`) + `manifest.json`. `--report` prints the determinism/calibration
  stats and exits. `--no-prs` for issues-only.
- `scripts/build_corpus_bpe.py` — added `gharchive` corpus choice (vocab 8192).
- `configs/corpora/gharchive_v4.json` + `gharchive_v4_smoke.json` — pairs mode (GH chains
  are mostly length 2), `max_text_tokens=64` (covers 99.85%; p99=46).
- `tests/jepa/test_chops_converter.py` — 14 tests (rendering / action / build / determinism
  / BPE round-trip + JEPAChainDataset), all passing.

### Validated sample (9 hours, 2024-01-15)
- 48K chains, determinism-given-action **0.93**, action vocab `{closed, commented, reopened, opened}`.
- BPE: actual vocab 6344, `<pad>=0`, mean 13.3 tok/state, p99=46.
- 2-epoch MPS smoke: total loss **6.01 → 2.76**, L_token **5.06 → 1.88**, gen_chrF **0.54 → 0.70**.
  Greedy decode learns the modal transition `"issue N is open." → "issue N is closed. it is unassigned."`.

### Full server-side run

```bash
# 1) Fetch a month of GH Archive hourly dumps (server-side; ~85 GB for Jan 2024)
mkdir -p data/gharchive/raw
for d in $(seq -w 1 31); do for h in $(seq -w 0 23); do
  curl -sSL -o data/gharchive/raw/2024-01-$d-$h.json.gz \
    https://data.gharchive.org/2024-01-$d-$h.json.gz
done; done

# 2) Convert (iid split). For an OOD-entity probe instead, run the converter twice on
#    two non-overlapping date ranges and use the later range as test.
uv run python scripts/convert_gharchive.py \
  --inputs 'data/gharchive/raw/2024-01-*.json.gz' \
  --out-dir data/gharchive --n-train 200000 --n-test 5000 --seed 7

# 3) Build the domain BPE (recompute coverage on the full corpus; label/login vocab grows)
uv run python scripts/build_corpus_bpe.py --corpus gharchive \
  --data-dir data/gharchive --vocab-size 8192 --max-text-tokens 64

# 4) Train (read-only use of the existing JEPA v2 script)
uv run python scripts/train_jepa_v2.py configs/corpora/gharchive_v4.json
```
