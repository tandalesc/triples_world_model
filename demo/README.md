# Pet Simulator demo

A fully client-side browser demo of the Triple World Model: a 29K-parameter closed-vocab
dynamics core runs **entirely in JavaScript** (no server, no WASM, no backend) and drives a
little Tamagotchi-style pet sim. Every game tick, the current per-pet state triples are fed
through the transformer and the predicted next state updates the UI.

## Serve it locally

```bash
cd demo/pet_simulation
python -m http.server 8080
# open http://localhost:8080
```

That's it — `index.html` fetches `model_weights.json` and runs inference in-browser. No
build step. The `resources/` folder holds the pet sprites and UI art.

## What the current model does

- **Architecture:** closed-vocab triple dynamics core — `d_model=32`, 2 layers, 2 heads,
  `d_ff=128`, vocab 53, up to 16 triples (the `config` block in `model_weights.json`).
- **Task:** 5 pets (Buddy, Luna, Max, Daisy, Rocky), each with 6 attributes
  (hunger, energy, mood, cleanliness, boredom, anger). You apply an action (feed, play, …),
  the model predicts the next state of every pet's attributes — including conditional
  cross-pet effects.
- **Quality:** 29K params, **98.9% exact match**, 303 KB of weights JSON, runs at interactive
  speed client-side.
- **How weights are produced:** `demo/pet_simulation/export_weights.py` loads a trained
  closed-vocab checkpoint (`results/pet_sim_v3/`) via `twm.serve.WorldModel`, dumps the
  state dict to nested JSON (rounded to 6 decimals), and prints the size. Run once:
  `cd demo/pet_simulation && python export_weights.py`.

This is the **closed-vocab / fixed-token-set** path — the lineage the JEPA latent-action
line grew out of, not the JEPA model itself.

## Upgrade path to a v2/v3-based pet

The interesting future demo is a pet driven by the **JEPA latent-action** model: persistent
per-entity identity (the modulus profile), discrete verbs as the actions, and an **exact undo**
via operator inverse (complex division). The pieces that exist vs. what's missing:

**Exists:**
- `scripts/export_jepa_weights.py` — INT8/fp16 weight-only export for the JEPA nano model,
  asserting the nano export fits ≤303 KB (the same browser budget as this demo). It bakes the
  operator codebook to `(cos, sin)` per verb, so `step_latent`/`undo_latent` are exportable.
- A trained nano/decoder-arm checkpoint (see `docs/REPRODUCING.md` §4).

**Missing (the real work):**
1. **Action-labeled training data.** GLUCOSE verbs are *causation-type clusters* from
   third-person narrative, not UI actions (feed/pet/play/scold). Mapping a UI button to a
   verb index needs a small action-labeled dataset or a fine-tune — this is the deferred
   "feed → verb 3" lookup from the v1 design (`research/jepa_operator_v1_design.md` §0.1).
   Until then the codebook also carries no recoverable semantics (synthesis §5, gap #2), so
   even a correct export wouldn't give nameable buttons yet.
2. **UI wiring.** This demo's `index.html` speaks the closed-vocab triple format
   (`stateTriples`, `gameState`); a JEPA pet needs new JS that loads the
   `export_jepa_weights.py` JSON, runs the slot encoder + operator + token decoder, and
   renders the decoded next-state. The persistent-`k` + `step_latent`/`undo_latent` API
   shape is designed (v1 design §0) but not yet wired to a frontend.

Recommended sequence: (a) collect/synthesize a tiny action-labeled pet dataset so verbs map
to UI actions, (b) fine-tune a nano JEPA checkpoint on it, (c) export via
`scripts/export_jepa_weights.py`, (d) fork this demo's `index.html` into a JEPA inference
shell. Steps (a)/(b) are the blockers; (c)/(d) are mechanical.

Do not break this demo's existing inference code when adding the JEPA path — keep them as
separate HTML shells loading separate weight JSONs.
