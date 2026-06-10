# Corpus Conversion Notes — wikiHow & CommitPackFT

**Date:** 2026-06-10
**Branch:** feature/glucose-converter (working tree only — not committed)
**Scope:** data pipeline for two real-world transition corpora feeding the JEPA
world-model stack. Reference: `research/transition_corpora_survey.md` (ranks 2 & 3).

These corpora are the NL/real-world scale-up of the repo's existing procedural
lineage (`data/openpi_*.jsonl`, ~430 rows): wikiHow is the ~20x-larger source of
OpenPI; CommitPackFT is the cross-domain (code) action-conditioning probe.

---

## 1. Source selection (and a survey correction)

### wikiHow
The survey nominated `tasksource/goal-step-wikihow`. On inspection **all three
of its subsets (goal / order / step) are multiple-choice QA reformattings**
(columns: `sent1`, `sent2`, `ending0..3`, `label`) — none expose a clean ordered
step list per article. It is unusable as a chain source.

**Chosen primary: `gursi26/wikihow-cleaned`** (214,293 articles, CC BY-NC-SA 3.0).
Its `summary` field is the article's ordered step **headlines**, sentence-separated
(" . "). Each headline is one imperative step → one task-progress state snapshot.
Mean ~6 steps/article → ~1.2M extractable transitions, well above the 200K-chain
target. World-state reading: each step = a state of task progress; step→step is the
implicit action (survey §2a).

**Alternative: `b-mc2/wikihow_lists`** (11,461 articles, same license). Cleaner
numbered "1. … 2. …" step boundaries but ~20x smaller. Selectable with
`--source wikihow_lists` for a higher-precision but small variant. Not used for the
200K target (insufficient scale).

License: **CC BY-NC-SA 3.0** (wikiHow content) — non-commercial, research OK, must
be flagged for any release. Recorded in each `manifest.json`.

### CommitPackFT
`bigcode/commitpackft`, `python` config (56,025 commits; already FT-filtered to
high-quality instructional messages). Fields used: `old_file`/`new_file` (single-file
filter), `old_contents`/`new_contents` (states), `subject`/`message` (action label),
`license` (permissive filter). License is per-repo permissive (MIT/Apache/BSD/ISC/…).

**Loader gotcha:** CommitPackFT ships a `commitpackft.py` loading script that the
installed `datasets` (3.x) refuses ("Dataset scripts are no longer supported"). The
converter loads the per-language jsonl directly:
`load_dataset("json", data_files="hf://datasets/bigcode/commitpackft/data/<lang>/data.jsonl", streaming=True)`.

---

## 2. Schema decisions

### wikiHow → variable-length chain
`{"chain": [step_0, step_1, …, step_k]}` — JEPAChainDataset native. `triples` mode
(length≥3) for the multi-hop unroll; chains <3 steps are skipped by the dataset.
Steps normalized OpenPI-style (lowercased already in source, whitespace-collapsed,
single trailing period), kept as free NL (open-vocab path), **not** (e,a,v) triples.

### CommitPackFT → length-2 chain
```
state_t   = "BEFORE: <old code, truncated to N tokens around the change>"
state_t+1 = "AFTER:  <new code, truncated to N tokens around the change>"
action    = cleaned commit subject (labeled twin only)
```
**Why the BEFORE:/AFTER: prefixes stay in the state text:** a shared BPE/decoder
needs to know which side of the edit it is generating, and the prefix gives the
JEPA EMA target encoder an "AFTER" surface that contrasts with the online encoder's
"BEFORE" — strengthening the cross-state signal `data.py` requires (online sees
state_t, EMA target sees state_{t+1}; same-text-to-both degenerates).

**Length-2 ⟹ pairs mode mandatory.** `data.py` triples mode requires ≥3 states and
SKIPS length-2 chains. Verified: a length-2 chain yields exactly 1 adjacent pair in
`pairs` mode (test `test_bpe_roundtrip_and_dataset_tokenization`). The commitpack
configs set `data.mode = "pairs"`.

**Change-centered truncation:** `_truncate_around_change` windows the snippet around
the first differing line so the truncated BEFORE/AFTER actually contains the diff,
not just the file's import header.

### Filters (CommitPackFT)
single-file (`old_file == new_file`), small diff (`< --max-changed-lines`, default 30,
counted from a unified diff), permissive-license subset, non-trivial (before≠after,
non-empty message). Sample retention (python, 2K rows): **86%** kept; the dominant
drop is the license filter (~10%), then big-diff (~4%).

---

## 3. Weak action labels (labeled twins)

Both corpora emit a `_labeled.jsonl` twin mirroring
`data/entity_world/train_labeled.jsonl`:
`{"chain": [...], "actions": ["<label>@0", ...]}` with `len(actions) == len(chain)-1`.
The `@0` tag matches the entity-world single-actor slot convention.

### wikiHow label = first-verb lemma of the NEXT step's headline
Heuristic (`extract_action_label`): take the first alphabetic token of the next
step, skip a small set of leading adverbs/politeness/articles, then lemmatize via a
tiny irregular map + suffix rules (no spaCy/nltk dependency → deterministic, runs
anywhere `tokenizers` does). wikiHow headlines are overwhelmingly imperative so the
first token is usually the bare verb already; lemmatization mostly fixes the
occasional gerund/3rd-person leak.

**Measured quality (1500-chain sample, 8,579 transitions):**
- distinct verb labels: 1,108; `none` (no word token): 0.01%
- top labels: dont(232), look(229), keep(203), get(180), use(166), ask(156),
  check(154), take(150), give(148), consider(129), place(125), avoid(119)…
- ~34% of labels fall in a hand-list of ~45 common action verbs (rough
  imperative-coverage proxy).

**Noise modes (documented):**
- **Negated imperatives** → `dont` is the single most frequent label (232). "Don't
  take advantage of…" yields `dont`, not the real verb `take`. This is the largest
  systematic error; a future fix could skip `dont/don't` and take the following verb.
- **Non-imperative headlines** ("Your phone will restart.") → a noun/pronoun token,
  not a true action (~5-10% of steps; e.g. `phone`).
- **Phrasal verbs** lose their particle ("turn off" → `turn`).
This is a WEAK supervisory signal for action-recovery eval, NOT ground truth.
Treat recovered-action-vs-label agreement as a soft metric.

### CommitPackFT label = cleaned commit message
`clean_message` prefers `subject`, strips trailing `(#123)` / `[skip ci]` / signoffs,
collapses whitespace. **Much higher quality than wikiHow's verb heuristic** — these
are human-written instructional messages.

**Measured (1500-transition sample):** mean 7.0 words, p50=6, p95=12. Top first-words
are clean imperatives: add(334), fix(246), update(109), remove(98), use(98),
make(72), change(69)… i.e. the action is explicit and reliable.

**Label-quality verdict:** CommitPackFT actions are *given* (clean NL imperatives);
wikiHow actions are *inferred* (noisy first-verb lemma, ~10-15% effective error from
negation + non-imperative headlines). For action-recovery eval, weight CommitPackFT
agreement higher and treat wikiHow as a noisy upper-bound signal.

---

## 4. BPE + coverage verdicts

`scripts/build_corpus_bpe.py` trains a ByteLevel BPE (same special-token order so
`<pad>=0` matches `DomainBPETokenizer.PAD_ID`), reports tokens-per-state distribution
+ overflow vs a candidate `max_text_tokens`, and recommends a max (p99 → next mult of 16).

### Why ~8192 (not 512)
The 512 domain BPEs cover tiny low-entropy vocabularies (GLUCOSE placeholders, a
fixed entity pool). wikiHow is open-domain prose and CommitPackFT is source code —
both have a far larger type vocabulary, which a 512 BPE shatters into byte fragments.
8192 (~16x) keeps states compact. **Param impact (noted via config `_comment`):** at
`d_model=96`, token_emb = 8192·96 ≈ 0.79M params → total model ~1.5-2M (M-scale), up
from the 512-vocab nano (~29-80K). The configs intentionally exceed the 250K nano
budget (the train script warns; expected).

### Measured coverage (samples; full run fills vocab further)

| corpus | vocab built (sample) | mean tok/state | p95 | p99 | recommended max_text_tokens |
|--------|---------------------:|---------------:|----:|----:|----------------------------:|
| wikiHow | 6,242 (2K-article sample; fills toward 8192 at scale) | 8.6 | 19 | 33 | **64** (covers >99.9%; survey's "96-128" was for full step *paragraphs*, but the cleaned headlines are short) |
| CommitPackFT (`--max-code-tokens 48`) | 8,192 | 138 | 216 | 257 | **192** (covers ~p90; truncation is change-centered so the diff stays in-window) |

CommitPackFT note: 48 *whitespace* tokens/side expands to ~138 *subword* tokens
because code BPE fragments paths/punctuation heavily (`Ġ#!/ usr / bin / python`).
A larger `max_code_tokens` blows past any tractable state length; 48 + `max_text_tokens=192`
is the chosen balance (drops ~11% at 192, ~0% at 272 if a longer state is acceptable).

### Shared vs per-corpus: **per-corpus wins**
A shared BPE over both corpora was measured: wikiHow states stay ~10 tok but
CommitPackFT states stay ~140 tok and the shared vocab is split between two
radically different surface distributions (prose vs code), giving CommitPackFT no
fragmentation benefit while diluting wikiHow's coverage. The corpora share almost no
subword structure. **Recommendation: train a dedicated 8192 BPE per corpus**
(`data/wikihow/bpe_8192.json`, `data/commitpack/bpe_8192.json`).

---

## 5. Validation summary (local, on samples)

End-to-end on the 2K-article / 2K-row samples (mps):
- **convert → BPE → 2-epoch smoke train** runs clean for both corpora via the
  existing `scripts/train_jepa_v2.py` (read-only use).
- wikiHow smoke: total loss 15.11 → 13.28, L_token 13.36 → 11.52 (2 epochs).
- commitpack smoke: total loss 9.17 → 7.53, L_token 8.27 → 6.80 (2 epochs, pairs mode).
- Generation samples are well-formed (real `text_t` / `gold_t1` pairs, BPE round-trip
  through `Ġ`-prefixed tokens); text content is still noise at 2 epochs / 1K chains /
  4.3M params, as expected — the smoke test validates plumbing, not convergence.
- Determinism: same seed → byte-identical `train/test/_labeled` jsonl (verified).
- Tests: `tests/jepa/test_corpus_converters.py` — 17 pass (determinism, chain
  well-formedness, label alignment, diff filters, BPE round-trip + dataset tokenize).

---

## 6. Server-side commands (full datasets)

The full download+convert runs on the GPU server (per CLAUDE.md workflow). `datasets`
was added to `pyproject.toml`; the server `uv run` will install it.

```bash
# wikiHow — 200K train + 5K iid test chains (drop --limit-articles to stream ALL 214K)
uv run python scripts/convert_wikihow.py \
    --source wikihow_cleaned --n-train 200000 --n-test 5000 \
    --out-dir data/wikihow --seed 7

# wikiHow BPE (8192) + coverage report
uv run python scripts/build_corpus_bpe.py --corpus wikihow \
    --data-dir data/wikihow --vocab-size 8192 --max-text-tokens 64

# CommitPackFT — 100K target needs multiple languages (python alone ~40-50K post-filter)
uv run python scripts/convert_commitpack.py \
    --langs python,javascript,java,go,ruby \
    --n-train 100000 --n-test 5000 --max-code-tokens 48 \
    --out-dir data/commitpack --seed 7

# CommitPackFT BPE (8192) + coverage report
uv run python scripts/build_corpus_bpe.py --corpus commitpack \
    --data-dir data/commitpack --vocab-size 8192 --max-text-tokens 192
```

Then train with `configs/corpora/wikihow_v3.json` / `commitpack_v3.json` via
`scripts/train_jepa_v2.py` (set `vocab_size` to the BPE's actual built size if it
differs from 8192; vocab_size ≥ actual is safe, < actual indexes out of range).
```
