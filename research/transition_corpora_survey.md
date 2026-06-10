# Transition-Structured Text Corpora Survey

**Date:** 2026-06-10  
**Context:** M-scale (5–20M param) action-conditioned world model training. The project confirmed that transition-poor narrative data (GLUCOSE) starves latent-action discovery, while deterministic entity worlds with real action→state structure yield unsupervised action recovery within one epoch. We want corpora where (state_t, action, state_t+1) structure is intrinsic — either explicit or tightly inferable — at 100K–10M transition scale.

---

## Scoring Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| **Transitions** | Estimated (state_t, state_t+1) pair count |
| **Action explicitness** | Given (labeled) / Inferable (recoverable from text) / Absent |
| **State observability** | Full (complete world snapshot) / Partial (only changed attributes) |
| **License** | Open reuse status |
| **Prep cost** | 1 (raw download) → 5 (heavy custom parsing) |
| **Transition density** | How tightly state_t constrains state_t+1 given the action; this is the critical property GLUCOSE lacked |

---

## 1. Wikipedia Edit Histories

### 1a. WikiAtomicEdits
- **URL:** https://github.com/google-research-datasets/wiki-atomic-edits
- **Transitions:** ~43 million (25.7M insertions + 17.2M deletions across 8 languages; ~18M English)
- **Action explicitness:** Absent — the edit type (insert/delete) is labeled, but *why* the edit was made is not. There is no action that caused the state change; the edit itself is the diff.
- **State observability:** Partial — only the changed sentence is provided, not the full article state. No entity-level structure.
- **License:** CC BY-SA 4.0
- **Prep cost:** 2 — files are pre-packaged TSV (original sentence | change | edited sentence); no custom parsing.
- **Transition density:** Low for our purpose. The before/after pair is tight — a single phrase insertion or deletion is highly constrained. But there is no action conditioning: the model cannot learn *why* (or be asked to predict *what would change if action X were taken*). Without an action signal, this reduces to a denoising/editing corpus, not a world model corpus. The edit signal is also stylistic/factual, not causal-physical.
- **Notes:** Archived May 2023, read-only. Good for text-editing tasks; unsuitable as a transition-structured world model corpus because it has no action variable.

### 1b. WikiHist.html (Full Revision History)
- **URL:** https://zenodo.org/records/3605388
- **Transitions:** 580M revisions of 5.8M articles (18 years, 2001–2019); however, most revisions are minor text edits, vandalism reversions, or formatting changes — not state-change events.
- **Action explicitness:** Absent — edit summaries exist but are free-form and sparse.
- **State observability:** Full article text per revision, but not in entity/triple form.
- **License:** Derived from Wikipedia; CC BY-SA
- **Prep cost:** 5 — 7 TB compressed; extracting diff-as-transition tuples requires custom tooling to align consecutive revisions, filter meaningful edits, and extract entity states.
- **Transition density:** Very low. Most consecutive revisions differ in formatting, categories, or single-word corrections. True causal state changes (e.g., a historical event article being updated) are buried in noise. No action variable.
- **Assessment:** Too large to filter efficiently; no natural action structure. Not recommended for this phase.

---

## 2. Procedural / Recipe Corpora

### 2a. wikiHow Steps (goal-step-wikihow)
- **URL:** https://huggingface.co/datasets/tasksource/goal-step-wikihow
- **Transitions:** ~1.4M step-pair examples; underlying wikiHow corpus has ~230K articles with ~19 steps/article avg → ~4.4M step-to-step transitions extractable
- **Action explicitness:** Inferable — each step is an imperative action sentence (e.g., "Place the pan on the burner"). The action and its effect are fused in the step text; with minor parsing, (action, resulting state) can be separated. Steps are temporally ordered, giving a natural chain.
- **State observability:** Partial — only the described change is recorded, not a full world state. Entity states are implicit in step text.
- **License:** MIT (HuggingFace card); original wikiHow content is CC BY-NC-SA 3.0 — non-commercial restriction applies.
- **Prep cost:** 2 — steps are pre-extracted. Building (state_t, action, state_t+1) triples requires an extraction step (e.g., an LLM pass to identify affected entities per step).
- **Transition density:** Medium-high for narrow task domains (cooking, repair, cleaning). Each step tightly constrains what must have changed. However, wikiHow spans 19 categories including abstract social tasks ("How to Be Confident") where transitions are diffuse. Filtering to physical/procedural tasks improves density substantially.
- **Notes:** wikiHow was the source for OpenPI (the project's current procedural corpus). The full corpus is ~20× larger than OpenPI and has richer task diversity. The non-commercial license is a practical constraint for any release.

### 2b. RecipeNLG / Recipe1M
- **URL:** https://huggingface.co/datasets/mbien/recipe_nlg
- **Transitions:** ~2.2M recipes; avg ~10 steps/recipe → ~20M step-to-step transitions
- **Action explicitness:** Inferable — cooking verbs (chop, boil, fold) are explicit actions; ingredient/tool states change predictably. Physical causality is tight.
- **State observability:** Partial — only described ingredients and tools; no full world state.
- **License:** RecipeNLG is freely available for research; Recipe1M requires registration.
- **Prep cost:** 3 — steps are structured but require ingredient-state extraction (e.g., "raw egg" → "cooked egg" after "boil for 10 min"). A recent annotated dataset (ingredient-states, 2025) exists but is small.
- **Transition density:** High within the cooking domain. Physical transformations (state, temperature, location of food items) are tightly constrained by cooking actions. Domain is narrow but deterministic.
- **Assessment:** Excellent transition density within a narrow domain; actionable as a first open-domain real-world corpus if extraction is feasible.

---

## 3. Game Logs

### 3a. Lichess PGN Database (Chess)
- **URL:** https://database.lichess.org/
- **Transitions:** 7.86 billion games; average chess game ≈ 40 half-moves → ~315 billion (position, move, position) transitions. Even a 1-month slice (≈20M games compressed) yields ~800M transitions.
- **Action explicitness:** Given — every move (action) is explicitly labeled in algebraic/UCI notation; the resulting board state is fully deterministic.
- **State observability:** Full — the complete board state (64 squares + castling rights + en passant + clocks) is deterministic and reconstructable from the PGN prefix.
- **License:** CC0 (public domain); code MIT.
- **Prep cost:** 2 — python-chess library parses PGN directly; FEN-encoded states are standard. The Chess-World-Model benchmark (2025) provides a ready 10M-game split with 75-label state vectors.
- **Transition density:** Maximum — chess is a deterministic finite automaton. Given (state, action), state_t+1 is uniquely determined. This is the strongest possible transition density of any corpus surveyed here.
- **Notes:** The Chess-World-Model benchmark (arxiv 2605.30100, CC BY 4.0) provides an off-the-shelf 10M-game dataset with aligned (move-sequence, board-state) labels. Language models trained on PGN strings develop emergent world-state representations (Karvonen 2024, COLM 2024). Downside: the state space is fully symbolic (no natural language); this is ideal for latent action discovery benchmarking but not for grounded NL world models. Can serve as a high-signal synthetic scaffold before scaling to NL corpora.

### 3b. JerichoWorld (Text-Adventure Transitions)
- **URL:** https://arxiv.org/abs/2106.09578
- **Transitions:** 24,198 (state, action, state_t+1) tuples across 27 training games + 7,836 test instances across 9 games. Scale: ~32K labeled transitions total.
- **Action explicitness:** Given — each tuple explicitly contains the natural language action (e.g., "take lamp", "go north") that caused the state change.
- **State observability:** Partial — states are knowledge graphs (subject, relation, object) reflecting the observable world map, not the full game state. Typically 5–15 KG triples per state.
- **License:** Research; see Jericho repo (MIT-adjacent, game-specific licenses vary)
- **Prep cost:** 1 — pre-packaged dataset with structured tuples.
- **Transition density:** High within each game. KG-encoded states tightly constrain valid next states for a given action. However, the corpus is very small (~32K transitions) — well below the 100K lower bound for M-scale training.
- **Notes:** The ClubFloyd corpus (426 transcripts, 223K context-action pairs) is larger but lacks KG state labels and has data-leakage concerns on test games. ByteSized32-SP (76K transitions) is similarly structured but derived from synthetic text games with JSON state objects.

### 3c. ByteSized32-SP (Text Game State Prediction)
- **URL:** https://aclanthology.org/2023.emnlp-main.830/
- **Transitions:** 76,369 (context, state, action, state_t+1) tuples from 31 synthetic text games
- **Action explicitness:** Given — natural language action strings with structured JSON state objects
- **State observability:** Full JSON objects capturing all object properties (temp, location, dirty, etc.); avg 10.4 objects per state
- **License:** Research (EMNLP 2023)
- **Prep cost:** 1 — pre-packaged
- **Transition density:** High — synthetic game rules are deterministic; JSON states are complete. GPT-4 achieves only 59.9% accuracy on non-trivial transitions, confirming these are genuinely hard.
- **Notes:** Just at the 100K lower bound; real scale is borderline. The synthetic nature means domain coverage is narrow. Good as an evaluation set or scaffold; not sufficient alone for M-scale training.

---

## 4. Software Commit Histories

### 4a. CommitPack / CommitPackFT
- **URL:** https://huggingface.co/datasets/bigcode/commitpack
- **Transitions:** 57.7M commits; each commit is a (code_before, commit_message, code_after) triple — approximately 57M (state_t, action, state_t+1) transitions
- **Action explicitness:** Inferable-to-given — commit messages describe the intended action ("fix null pointer in login handler", "add rate limiting to API"). Quality varies: CommitPackFT filters to ~277K high-quality instructional messages, reducing scale by ~1000×.
- **State observability:** Partial — only changed files are stored per commit, not the full codebase state. Multi-file changes create partial-observability issues.
- **License:** Permissive per-repo licenses (MIT, Apache 2.0, BSD, ISC); dataset license is research-use.
- **Prep cost:** 3 — 3.8 TB total; processing requires language filtering, diff extraction, and alignment of before/after states. CommitPackFT is already filtered (2 GB, 277K samples) but highly reduced in scale.
- **Transition density:** Medium — commit messages constrain what changed, but code is complex; many commits touch unrelated subsystems simultaneously. The action (commit message) is high-level; the state change (diff) is low-level. Alignment is semantic rather than syntactic.
- **Notes:** This is a genuine (state, action, state_t+1) corpus at massive scale, but the state space (code files) is very different from triple-structured NL states. Preprocessing to extract entity-level state changes from diffs requires substantial NLP work. The filtered CommitPackFT is at the right scale (277K) but the domain may be too narrow (code only).

---

## 5. Task-Oriented Dialogue State Tracking

### 5a. MultiWOZ 2.2–2.4
- **URL:** https://github.com/budzianowski/multiwoz
- **Transitions:** 10,438 dialogues × 13.46 turns/dialogue avg = ~140K turn-to-turn transitions. Each turn has an explicit (domain, slot, value) state dict.
- **Action explicitness:** Inferable — user utterances are the actions; system responses update the state. Slot-filling actions (book hotel, find restaurant) are implicit in utterances.
- **State observability:** Full within domain — the belief state is a complete (domain, slot, value) dictionary at each turn. 30 (domain, slot) pairs across 8 domains.
- **License:** Apache 2.0
- **Prep cost:** 1 — pre-packaged with structured state annotations.
- **Transition density:** High within the constrained slot-filling ontology. Turn t+1 belief state is tightly constrained by turn t state + utterance. However, the state space is closed-vocab (finite slot values in most domains).
- **Notes:** Small scale (~140K transitions) but very clean structure. Excellent for evaluation and fine-tuning but insufficient alone for M-scale pretraining. Schema-guided: transitions are limited to the 30 (domain, slot) pairs — no open-world compositional generalization. ABCD, SGD (Schema-Guided Dialogue) are larger successors with 16K/20K dialogues respectively.

---

## 6. Process/Procedure Datasets (Repo Lineage)

### 6a. ProPara (local: data/propara_*.jsonl)
- **Local size:** 769 total rows (738 train / 14 dev / 17 test)
- **Transitions:** ~2–5 state changes per sentence; estimated ~3,000 entity-state transitions total
- **Action explicitness:** Inferable — sentences describe scientific processes; entity state changes (created/destroyed/moved) are labeled but the action is the sentence itself.
- **State observability:** Partial — only tracked entity states (location, existence)
- **License:** CC BY 4.0
- **Transition density:** High within its domain (biological/chemical/physical processes). Each sentence has tight entity-state semantics.
- **Assessment:** Way too small (~3K transitions). Useful as eval only.

### 6b. OpenPI / OpenPI2.0 (local: data/openpi_*.jsonl)
- **Local size:** 434 total rows (heavily subsampled from the full 29,928 state changes)
- **Full OpenPI:** 29,928 state changes over 4,050 sentences from 810 WikiHow paragraphs
- **OpenPI2.0 (2024):** Same scale but canonicalized entities/attributes; adds entity salience annotations
- **Action explicitness:** Inferable — the procedural sentence is both action and context; (entity, attribute, before, after) annotations are explicit.
- **State observability:** Partial — only salient entity-attribute changes per step
- **License:** CC BY 4.0
- **Prep cost:** 1 — pre-structured
- **Transition density:** High for physical procedures; each step has tightly determined attribute changes (location, state, temperature)
- **Assessment:** ~30K transitions is still below target for M-scale training. The full wikiHow corpus (source of OpenPI) is the right scale-up path.

### 6c. GLUCOSE (local: data/glucose/)
- **Local size:** 65,521 training annotations + 36,449 chain training examples
- **Structure:** 10-dimensional causal explanations of narrative sentences in ROCStories; chains are 3-sentence causal sequences
- **Action explicitness:** Absent — causal relationships are annotated post-hoc; no explicit action variable conditions the state change
- **Transition density:** Low — confirmed experimentally in this project. GLUCOSE narrative transitions are underspecified: the same story sentence can follow from many prior states, and the 3-sentence chains do not form closed-world dynamics. This is precisely the failure mode identified.

---

## 7. Additional Candidates Found During Research

### 7a. ATOMIC / ATOMIC20 (Commonsense If-Then)
- **URL:** https://arxiv.org/abs/1811.00146
- **Transitions:** 877K (event, relation, effect) tuples; 710K training
- **Action explicitness:** Given — 9 typed relations (xEffect, xReact, xWant, oEffect, Causes, etc.) make the causal structure explicit
- **State observability:** Partial — mental/social states rather than physical world states; effects are free-text phrases
- **License:** CC BY 4.0
- **Transition density:** Medium — social/emotional causality is looser than physical state causality. The same event can have many plausible effects (crowdsourced).
- **Notes:** Already used in the project (ATOMIC 10K). Rich causal structure but social rather than physical; transition density is lower than deterministic environments.

### 7b. Chess-World-Model Benchmark (2025)
- **URL:** https://arxiv.org/html/2605.30100
- **Transitions:** 10M games × ~40 moves = ~400M (position, move, position) triples; benchmark provides aligned state vectors
- **Action explicitness:** Given — UCI move notation is the action label
- **State observability:** Full — 75-label FEN-style state representation
- **License:** CC BY 4.0 (benchmark); CC0 (Lichess data)
- **Transition density:** Maximum (deterministic finite automaton)
- **Notes:** Purpose-built for state-tracking benchmarking. Symbolic state space. Ideal as a high-signal training scaffold or evaluation regime for the transition-learning claim, even if not directly transferable to NL world models.

---

## Comparative Table

| Corpus | Scale (transitions) | Action explicit | State observability | License | Prep cost | Transition density |
|--------|--------------------:|----------------|--------------------:|---------|:---------:|:-----------------:|
| WikiAtomicEdits | 43M | Absent | Partial | CC BY-SA 4.0 | 2 | Low — no action var |
| WikiHist.html | 580M revisions | Absent | Full text | CC BY-SA | 5 | Very low |
| wikiHow steps | ~4.4M | Inferable | Partial | CC BY-NC-SA | 2 | Med-high (physical tasks) |
| RecipeNLG | ~20M | Inferable | Partial | Research | 3 | High (cooking domain) |
| **Lichess PGN** | **~315B (subsettable)** | **Given** | **Full** | **CC0** | **2** | **Maximum** |
| JerichoWorld | 32K | Given | Partial KG | Research | 1 | High — too small |
| ByteSized32-SP | 76K | Given | Full JSON | Research | 1 | High — borderline scale |
| CommitPack | 57M | Inferable | Partial | Permissive | 3 | Medium |
| CommitPackFT | 277K | Given (filtered) | Partial | Permissive | 1 | Medium |
| MultiWOZ 2.x | ~140K | Inferable | Full (slot) | Apache 2.0 | 1 | High (closed-vocab) |
| ProPara (local) | ~3K | Inferable | Partial | CC BY 4.0 | 1 | High — too small |
| OpenPI / 2.0 | 30K | Inferable | Partial | CC BY 4.0 | 1 | High — too small |
| GLUCOSE (local) | 65K chains | Absent | Partial | CC BY 4.0 | 1 | Low (confirmed) |
| ATOMIC | 877K | Given (typed) | Partial | CC BY 4.0 | 1 | Medium (social) |
| Chess-World-Model | 10M games (400M pos) | Given | Full | CC BY 4.0 | 2 | Maximum |

---

## Top 3 Ranked Candidates

### Rank 1: Lichess PGN / Chess-World-Model Benchmark

**Rationale:** The only corpus where (state, action, state_t+1) is exact, complete, and lossless — a deterministic finite automaton with ~400M transitions freely subsettable to any target scale. Action labels (moves) are given; states are fully observable (board position). Transition density is maximum by construction. The Chess-World-Model benchmark (2025, CC BY 4.0) packages this as 10M games with aligned 75-label state vectors, lowering prep cost to ~2. The primary limitation is that the state space is symbolic, not natural language, so this corpus validates the latent-action learning mechanism on perfect data but does not transfer directly to NL entity triples. Recommended use: high-signal training scaffold and ablation ground-truth for the M-scale transition-learning claim; establish that the architecture recovers latent actions when transitions are perfect before testing on noisier NL corpora.

### Rank 2: wikiHow Full Corpus (source for OpenPI scale-up)

**Rationale:** ~4.4M extractable step-to-step transitions from ~230K procedural articles, with action sentences fused into step text and physical causality tight enough to support entity-state extraction. This is a natural M-scale continuation of OpenPI (the project already uses OpenPI successfully). The full wikiHow corpus is 20× larger than OpenPI and covers the same action-conditioned procedural domain where transition density is confirmed to be high. License (CC BY-NC-SA 3.0) restricts commercial use but is fine for research. Prep cost is moderate (2): a one-time pass with an extraction model or rule-based NLP to generate (entity, attribute, before, after) tuples per step, producing a triple-structured transition dataset in the project's native format. OpenPI2.0 canonicalization work (2024) provides a template.

### Rank 3: CommitPackFT (Filtered Software Commits)

**Rationale:** 277K high-quality (code_before, commit_message, code_after) tuples with natural-language action descriptions that read as instructions ("fix X", "add Y", "remove Z"). Commit messages function as explicit action labels; diffs are deterministic state changes. At 277K transitions, this is at the right scale for M-scale training. Prep cost is low (1 — already filtered and packaged). The domain (code) differs from NL triples, but the structure maps well: entities = functions/variables, attributes = their values/types/existence, actions = commit operations. Limitation: converting code diffs to entity-attribute triple format requires non-trivial extraction. However, if the model is extended to handle code-diff states (which are naturally structured), this corpus provides the best action-label quality among NL-native corpora at the right scale. License is permissive (MIT/Apache per-repo). Recommended as a secondary corpus to test whether action-conditioning generalizes beyond physical-process domains.

---

## Recommendations for M-Scale Phase

1. **Start with Chess-World-Model** to validate the architecture's latent-action discovery capability on a noise-free, perfectly-structured corpus. This is a 2-hour experiment that will confirm or deny the mechanism before investing in NL data prep.

2. **Scale-up wikiHow → OpenPI format** as the primary NL training corpus. Extract (entity, attribute, before, after) triples from the full ~230K wikiHow articles using the OpenPI2.0 annotation schema as a template. Filter to physical/procedural categories (food, home repair, gardening) for maximum transition density. Target 1–3M transitions.

3. **CommitPackFT as a cross-domain probe** to test whether action-conditioning transfers to code-edit dynamics. Requires defining a triple format for code entities (file, function, type/signature/value).

**Explicitly not recommended for this phase:**
- WikiAtomicEdits / WikiHist.html: no action variable; editing ≠ causal state change
- GLUCOSE chains: low transition density, confirmed experimentally
- JerichoWorld / ByteSized32-SP: high quality but too small (30K–76K) for M-scale
- MultiWOZ: closed-vocab, too small, wrong domain for compositional generalization testing

---

## References

- WikiAtomicEdits: [Faruqui et al., EMNLP 2018](https://aclanthology.org/D18-1028/)
- WikiHist.html: [Mitrevski et al., ICWSM 2020](https://ojs.aaai.org/index.php/ICWSM/article/view/7353)
- wikiHow goal-step dataset: [tasksource/goal-step-wikihow](https://huggingface.co/datasets/tasksource/goal-step-wikihow)
- RecipeNLG: [Bien et al., INLG 2020](https://aclanthology.org/2020.inlg-1.4.pdf)
- OpenPI: [Tandon et al., EMNLP 2020](https://arxiv.org/pdf/2011.08092)
- OpenPI2.0: [Zhang et al., EACL 2024](https://aclanthology.org/2024.eacl-long.10/)
- Lichess Open Database: [database.lichess.org](https://database.lichess.org/)
- Chess-World-Model Benchmark: [arxiv 2605.30100](https://arxiv.org/html/2605.30100)
- JerichoWorld: [Ammanabrolu et al., NeurIPS 2021](https://arxiv.org/abs/2106.09578)
- ByteSized32 / ByteSized32-SP: [Dalvi et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.830/)
- CommitPack / OctoPack: [Muennighoff et al., 2023](https://arxiv.org/pdf/2308.07124)
- MultiWOZ 2.2: [Zang et al., 2020](https://research.google/pubs/multiwoz-22-a-dialogue-dataset-with-additional-annotation-corrections-and-state-tracking-baselines/)
- ATOMIC: [Sap et al., AAAI 2019](https://arxiv.org/abs/1811.00146)
- CALM/ClubFloyd: [Yao et al., EMNLP 2020](https://arxiv.org/pdf/2010.02903)
- Karvonen chess world model interventions: [COLM 2024](https://arxiv.org/pdf/2403.15498v2)
