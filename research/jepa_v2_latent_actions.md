# JEPA v2 — Unsupervised Latent Actions + Token-Space Grounding

**Status:** APPROVED FOR IMPLEMENTATION · **Branch:** `feature/glucose-converter` · **Code:** `src/twm/jepa/` (new v2 files alongside v1)
**Supersedes (training objective only):** `research/jepa_operator_v1_design.md`. The v1 **operator algebra, SlotEncoder noun path, SIGReg, and diagnostics philosophy are KEPT.** The v1 *prediction objective* (latent-MSE-only, per-slot verbs, `L_div`) is **replaced.**
**v2 dataset:** GLUCOSE causal chains, local jsonl (`data/glucose/chain_general_{train,test}.jsonl`), tokenizer `data/glucose/jepa_bpe_512.json` (vocab 512; `<pad>=0 <mask>=1 <unk>=2 <bos>=3 <eos>=4`).

This document fixes every open design point. Builders need no further decisions.

---

## 0. Why v2 exists (probe-verified v1 failure)

v1 was trained (95 epochs, `results/jepa_nano_smoke/model_latest.pt`) and probed (`results/jepa_nano_probe/`). It failed three independent diagnostics, and the failures share **one root cause: predicting the next state's verb from `state_t` alone is ill-posed on narrative data.** The next causal step is underdetermined by the present, so the model took the only degenerate route that minimizes the losses:

| Probe | Result | Reading |
|---|---|---|
| **Retrieval hard-neg** (`probe1_retrieval.json`) | `easy_minus_hard_mrr = −0.041` (both EMA & online). Hard-pool MRR `0.070` vs **chance `0.104`** — *below* chance on same-chain hard negatives | `L_pred` was satisfied by **topic-prior similarity** to the EMA target, not by modeling the transition. Predictions are identity-biased: they rank the wrong same-chain state *above* the right next-state. |
| **Verb–label MI** (`probe2_verb_mi.json`) | seq-level `nmi_minus_shuffle ≈ 0.0002`; slot-level `≈ −0.001` (at/below shuffle). Usage ppl `6.2/8` (looks healthy!) | The codebook carries **no transition semantics.** Usage entropy was high because verbs were assigned **per slot position**, not per transition — gaming `L_div` (batch-pooled usage entropy is satisfied by positional assignment). |
| **Slot purity** (`probe3_slot_purity.json`) | every slot's `loo_residual_increase < 0` (masking a slot *helps* `L_pred`); per-slot verb histograms near-deterministic (slot 2→verb 7 always; slot 4→verb 1 always; slot 1→verb 7 999/1000) | The encoder **hardwired one verb per slot position.** Slots carry no distinct predictive info; the operator/verb path is decorative. |

**The three v1 design pillars that enabled the cheat, all now removed/replaced:**

1. **Verb predicted from `state_t` only** → ill-posed → identity-biased `L_pred`. **v2 fix:** a training-time **posterior** (TransitionEncoder) that sees *both* `state_t` and `state_{t+1}` and emits the latent action. The action `v` is the *only* path carrying next-state information; the operator now has a well-posed target.
2. **Per-slot verbs** → positional assignment games usage entropy. **v2 fix:** **ONE sequence-level discrete action** `v` per pair, applied to all slots through `B_v`.
3. **`L_div` as the informativeness driver** → gameable regularizer. **v2 fix:** **DELETE `L_div`.** Verb informativeness now comes from **necessity** — a token decoder must reconstruct `text_{t+1}` and the only conditioning signal that distinguishes futures is `v`'s bits. Token CE is the primary loss. Codebook usage is a **diagnostic only.**

The v1 anti-goal "no token decoder" is **explicitly revoked by the user.** Token-space grounding is the central v2 mechanism.

---

## 1. Architecture overview

```
                       ┌──────────────── TRAINING ONLY ────────────────┐
  text_t  ─► SlotEncoder (KEEP v1) ─► slots_t ─► NounHead ─► k (B,M,dn)
                  │  (shared trunk)                  └─► [VerbHead DELETED in v2]
                  │
  text_t ─┐       │
  text_t+1┴─► TransitionEncoder (posterior q(v | t, t+1)) ─► v_logits (B,V)
                  │                          │ Gumbel-ST over V codes
                  │                          ▼
                  │                       v (B,)  ◄── ONE discrete action per pair
                  │                          │
   k (B,M,dn) ────┼──────────► Operator.apply(k, v) ─► a* (B,M,dn)   [KEEP v1 RotationScaleOperator]
                  │                                        │
  text_t ─► PriorHead p(v | pool(slots_t)) ─► p_logits     │  L_prior = KL(stopgrad q ‖ p)
                  │                                        ▼
                  │                            ┌─► Readout(a*) ─► zhat (B,dn) ── L_pred (aux, EMA)
                  │                            └─► TokenDecoder(memory = a*) ──► logits_t+1 ── L_token (PRIMARY)
                  ▼
  text_t+1 ─► EMA(SlotEncoder+Readout) ─► z (B,dn) stop-grad     (KEEP v1 EMA target)
```

**Inference / autonomous rollout:** the posterior is *gone*. Sample `v ~ PriorHead(pool(slots_t))` (or set `v` directly = user action), then `a* = B_v k`, then `TokenDecoder.generate(a*)` → `text_t+1`. Persistent-`k` pet loop unchanged from v1 §9; the only difference is the verb now comes from the prior (or a UI action), and the *quality artifact* is generated text, not a latent vector.

**What carries next-state info, and the leakage rule (the load-bearing invariant):** the TokenDecoder may see **ONLY** `{a*_i}` as cross-attention memory, plus its own teacher-forced AR context over `text_{t+1}` tokens. The decoder must **never** see raw `text_{t+1}` encodings, the posterior features, `k` un-transformed, or `slots_t`. Since `a* = B_v k` and `k` is a deterministic function of `text_t` (no `t+1` info), the **only** path from `text_{t+1}` into the decoder's conditioning is through the **discrete `v`** (`log2 V = 3` bits for nano). This is the information bottleneck that forces `v` to carry the causal-step identity. Full enumeration of bypass paths in §6.

---

## 2. TransitionEncoder (posterior q(v | text_t, text_{t+1}))

**File:** `src/twm/jepa/transition.py` (T-task A). Training-only module; not exported, not used at inference.

### 2.1 How the pair is consumed (DECIDED: shared trunk, pooled, concatenated)

We **reuse the SlotEncoder's text trunk** (the ALBERT-tied `encode_text` + a mean-pool) for *both* texts rather than building a separate trunk. Rationale (param budget): a separate trunk would duplicate the ~33K-param text self-attention block; the posterior is training-only overhead we want minimal. Sharing the trunk costs **zero new attention params** and keeps the posterior's view of text in the same representation space the nouns are built from (which is exactly the space the operator must explain).

Concretely the TransitionEncoder is given a **callable `encode_text(ids, pad) -> (B,T,d)`** (the SlotEncoder's bound method) at construction. It does NOT own the trunk weights; it owns only a small head:

```
pool_t   = masked_mean( encode_text(text_t,   pad_t),   pad_t )   # (B, d)   shared trunk, no new params
pool_t1  = masked_mean( encode_text(text_t+1, pad_t1),  pad_t1)   # (B, d)   shared trunk
pair     = concat([pool_t, pool_t1, pool_t1 - pool_t], dim=-1)    # (B, 3d)  delta channel makes the transition explicit
v_logits = MLP(pair)                                             # (B, V)
```

- The **delta channel `pool_t1 - pool_t`** is included deliberately: the action is a *transition*, so making the difference an explicit input biases the MLP toward encoding "what changed" rather than "what the next state is" (a small inductive lever against the identity-prior failure mode of v1).
- `masked_mean` over non-pad positions (pad mask is `True` at pad, as in v1 data).
- **Gradient into the shared trunk:** posterior gradients **do** flow back into the SlotEncoder text trunk (it is the same `encode_text`). This is intended — it lets the trunk shape a representation that supports both noun extraction and transition discrimination. It is *not* a leak: the trunk only ever sees `text_t` for the noun path; the `text_t+1` pass exists solely inside the posterior and its output (`v`) is bottlenecked to 3 bits before reaching the decoder.

### 2.2 MLP head

```
MLP:  Linear(3d → h) → GELU → LayerNorm(h) → Linear(h → V)
      nano: d=64, h=128, V=8.
```
Params (nano): `3·64·128 + 128` (in) `+ 128` (LN) `+ 128·8 + 8` (out) `= 24576 + 128 + 128 + 1024 + 8 = 25,864`. (≈25.9K; rounded **26K** in the table.)

### 2.3 Gumbel-Softmax / straight-through over V codes

- **Forward (train):** `v_onehot = gumbel_softmax(v_logits, tau, hard=True)` — straight-through hard one-hot `(B, V)`. Hard one-hot is used so the operator sees a *single* code (matching the inference-time discrete action), while the ST gradient flows through the soft sample into `v_logits` and the shared trunk. This is **different from v1**, which used a *soft mix* `Σ_v p_v B_v k`. v2 uses **hard ST** because the leakage bound (§6) requires `v` to be genuinely discrete (a soft mix leaks `> log2 V` bits — the continuous mixing weights themselves carry next-state info). The ST estimator preserves trainability.
- The operator's existing soft-path code (`_gather_blocks` accepting `(B,M,V)` float) is reused by broadcasting the per-pair one-hot `(B, V)` to `(B, M, V)`; with a hard one-hot this is exactly the hard-index path numerically.

### 2.4 Temperature schedule (gentler than v1; v1's floor 0.5 caused collapse oscillation)

v1 annealed `τ: 2.0 → 0.5` over the first 30% of steps and **held at 0.5**. The probe shows codebook usage was *nominally* healthy but semantically empty, and the v1 smoke logs show the loss oscillating in the anneal tail — a too-low floor sharpens the Gumbel sample before the decoder has taught `v` to mean anything, locking in whatever positional assignment exists early.

**v2 schedule (DECIDED):**
- `τ_start = 3.0`, `τ_end = 1.0` (floor **raised** to 1.0, not 0.5), **linear** anneal over the **first 50%** of steps, then held at `1.0` for the rest of training.
- Justification: at the new dataset/budget the action codebook must be shaped by the *decoder*, which is itself learning to read `a*`; a higher, longer-held floor keeps `q(v|·)` soft enough that all codes keep receiving CE gradient until the decoder can disambiguate them, preventing the early lock-in. `τ=1.0` is still informative (the Gumbel sample is meaningfully peaked at convergence given well-separated logits) without the collapse-oscillation of `0.5`.
- **Eval / rollout / export:** `hard=True` argmax of `v_logits` (posterior) or `p_logits` (prior). No Gumbel noise at eval.
- Config keys `gumbel_tau_start / gumbel_tau_end / anneal_frac` reused; defaults change to `3.0 / 1.0 / 0.5`.

---

## 3. PriorHead p(v | text_t)

**File:** `src/twm/jepa/transition.py` (same file as TransitionEncoder; both are the "action heads", T-task A).

The prior predicts the action from `state_t` alone — this is what makes autonomous rollout possible (no `text_{t+1}` at inference).

```
prior_pool = masked_mean( encode_text(text_t, pad_t), pad_t )   # (B, d)  shared trunk (SAME pool_t as posterior)
                                                                #         reuse pool_t — do not recompute
p_logits   = MLP_prior(prior_pool)                              # (B, V)
MLP_prior: Linear(d → h) → GELU → Linear(h → V)
           nano: d=64, h=64, V=8.
```
Params (nano): `64·64 + 64 + 64·8 + 8 = 4096 + 64 + 512 + 8 = 4,680` (≈**4.7K**).

- **Why no LayerNorm here:** the prior is small and reads the already-LN'd trunk pool; keep it minimal.
- **Multi-modality note (honest):** the true `p(v | state_t)` is genuinely multi-modal on narrative data (this is *why* v1's single-point prediction failed). A categorical prior over `V` codes represents that multi-modality natively — sampling `v ~ Categorical(softmax(p_logits))` at rollout yields *different* plausible futures, which is the correct behavior. Greedy argmax gives the single most-likely future. Both are exposed (diagnostics §8 logs samples at temperature).

### 3.1 L_prior (KL with stop-grad posterior)

```
L_prior = KL( stopgrad(softmax(v_logits / τ_post)) ‖ softmax(p_logits) )
```
- The **posterior is the target** (stop-grad): the prior learns to imitate the posterior's action distribution from `text_t` only. This is the standard amortized-inference / "distill the posterior into the prior" pattern (VAE-style, but discrete-categorical).
- `τ_post` = the *current annealed* posterior temperature, so prior and posterior are compared at the same sharpness.
- KL direction is `KL(q ‖ p)` (forward KL from the detached posterior): mode-covering, so the prior keeps mass on all actions the posterior uses — appropriate for multi-modal rollout.
- Weight `w_prior = 0.1` (§5). Kept small: the prior must *follow* the posterior, never *steer* it (steering would let the prior collapse the posterior to an easy-to-predict-from-`t` action, re-introducing the v1 identity bias).

---

## 4. Token decoder (THE grounding loss)

**File:** `src/twm/jepa/decoder.py` (T-task B). A thin **adapter** around the repo's `ARDecoder` (`src/twm/ar_decoder.py`).

### 4.1 Decoder choice (DECIDED: adapt `ARDecoder`, NOT `dual_ar_decoder`)

| Candidate | Fit | Verdict |
|---|---|---|
| `ARDecoder` (single memory, cross-attends one latent set, no pos-enc on memory) | Exactly one memory channel = `{a*_i}`. Set-invariant cross-attn (correct: `a*` is a slot *set*). Smallest. | **CHOSEN** |
| `DualARDecoder` (dense + sparse channels) | The dense channel is the *leak* — it exists to ground entities from the compressor output, which here would be raw `text_t`/`k` info bypassing `v`. Adding it violates §6. Also ~1.5× params. | Rejected (leakage + budget) |
| New minimal decoder | `ARDecoder` is already minimal and matches the contract; writing a new one duplicates it. | Rejected (reuse) |

The v2 decoder is a **single-memory** AR decoder: memory = projected `a*` (B,M,dn)→(B,M,d_dec), **no positional encoding on memory** (slots are a permutation-invariant set — `ARDecoder` already enforces this), causal self-attention over `text_{t+1}` tokens **with** positional encoding, cross-attention to memory, vocab projection. Teacher forcing on `text_{t+1}` is standard AR and is **not** a leak (the model only ever sees ground-truth *prefix* tokens; the loss is next-token CE).

### 4.2 Adapter responsibilities (what `decoder.py` adds over `ARDecoder`)

`ARDecoder` is already correct; the adapter is thin:
1. Construct `ARDecoder(vocab_size=512, d_model=d_dec, n_heads, n_layers, d_ff, max_text_tokens, bottleneck_dim=dn, pad_id=0)`. Memory is `a*` of width `dn=32`; `ARDecoder.memory_proj` maps `dn → d_dec` — so set `bottleneck_dim=dn`.
2. Expose `forward(a_star, tgt_ids, tgt_pad) -> logits (B, T, V)` calling `ARDecoder.forward(bottleneck=a_star, target_ids=tgt_ids, target_pad_mask=tgt_pad)`.
3. Expose `generate(a_star, max_tokens, temperature) -> (B,T) ids` (greedy and temperature, for the diagnostics text samples §8).
4. **Nano sizing knob:** `d_dec` is the decoder's own width, **decoupled from `d_model=64`** so we can hit the param budget. **DECIDED: `d_dec = 64, n_layers = 2, n_heads = 4, d_ff = 128`** (matches encoder width/ff for uniformity). `max_text_tokens = 64`.

### 4.3 L_token (primary loss)

```
logits = decoder(a*, tgt_ids, tgt_pad)              # (B, T, V)
L_token = cross_entropy(logits[:, :-1].reshape(-1,V), tgt_ids[:, 1:].reshape(-1), ignore_index=pad_id)
```
- **Target = `text_{t+1}` tokens** (the next state's text), shifted for next-token prediction. Weight `w_token = 1.0` (primary).
- **EOS requirement (data change, §7):** the BPE `encode` does **not** currently append `<eos>` (id 4). The v2 dataset MUST append `<eos>` to each tokenized state before padding, so the decoder learns to stop. `ignore_index = pad_id (0)` masks pad positions; `<eos>` is a real predicted token.
- The decoder's own BOS comes from `ARDecoder.bos_emb` (learned), so `tgt_ids` is the target *without* a leading BOS; `ARDecoder.forward` prepends `bos_emb` internally and predicts `tgt_ids[:, 1:]` from `[bos, tgt[:-1]]`. (Adapter passes `tgt_ids` directly; the CE shift above matches `ARDecoder`'s internal `[bos]+tgt[:-1]` construction — i.e. position `t` predicts `tgt_ids[t]`. **Implementation note for T-B:** `ARDecoder.forward` returns logits aligned to `tgt_ids` positions `0..T-1`; the loss is `CE(logits.reshape(-1,V), tgt_ids.reshape(-1), ignore_index=0)` — no manual shift, the shift is baked into `ARDecoder`'s `[bos]+tgt[:-1]` input. Use this form; the `[:, :-1]/[:, 1:]` form above is the conceptual equivalent. T-B's test asserts logits shape `(B,T,V)` and that CE is finite.)

---

## 5. Losses (file: `src/twm/jepa/losses_v2.py`)

```
L = w_token·L_token  +  w_prior·L_prior  +  w_sigreg·L_sigreg  +  w_pred·L_pred
```

| Term | Weight | Source | Role |
|---|---:|---|---|
| **L_token** | **1.0** | new (§4.3) | **PRIMARY.** CE of `text_{t+1}` given `a*`. The grounding loss; makes `v` necessary. |
| **L_prior** | **0.1** | new (§3.1) | KL(stopgrad q ‖ p). Enables autonomous rollout. |
| **L_sigreg** | **0.05** | REUSE v1 `losses.sigreg_loss` | Keeps noun space isotropic (unchanged; standardize, never L2). |
| **L_pred** | **0.25** | REUSE v1 (`MSE(zhat, z.detach())`, EMA target) | **OPTIONAL aux**, ON by default. Keeps the JEPA latent objective alive as a regularizer on `a*`. |
| ~~L_div~~ | **DELETED** | — | Verb informativeness now from necessity, not a gameable regularizer. |

- `losses_v2.py` **imports** `sigreg_loss`, `anneal_tau` from v1 `losses.py` (no duplication, no edit to v1 file). It adds `token_ce`, `prior_kl`, and a `JEPALossV2(nn.Module)` aggregator whose `forward` signature is frozen in §9.
- `w_pred = 0.25` (down from v1's 1.0) because L_token is now the driver; L_pred is a stabilizer. Setting `w_pred = 0.0` is a supported ablation (pure token-grounding).
- **Codebook usage is NOT a loss.** Monitored in diagnostics only (§8).
- Operator `theta`/`log_r` get gradient from **L_pred and L_token** (both flow through `a* = B_v k`), so the operator still learns even with `L_div` gone.

---

## 6. Leakage analysis (every path next-state info could bypass `v`)

The invariant: **the only `text_{t+1}` → decoder path is the 3-bit discrete `v`.** Enumerated bypasses and how each is blocked:

| # | Potential bypass | Blocked by |
|---|---|---|
| L1 | Decoder cross-attends raw `text_{t+1}` encodings | Decoder memory is **`a*` only**. No `text_{t+1}` tensor is ever passed to `decoder.forward` except as the **teacher-forced target prefix** (standard AR; the model sees only ground-truth *previous* tokens to predict the *next* — it cannot see future tokens past the causal mask, and at generation time it sees none). |
| L2 | Decoder reads posterior features (`pool_t1`, `v_logits`) directly | Decoder signature accepts `(a*, tgt_ids, tgt_pad)` **only**. `transition.py` outputs `v` (one-hot) — the model passes `v` to the **operator**, never to the decoder. Asserted by T-B's contract test (decoder constructor has no posterior arg). |
| L3 | **Soft Gumbel mix leaks > 3 bits** (v1's mode) | v2 uses **hard ST one-hot** (`hard=True`). The continuous mixing weights of a soft mix are themselves a high-bandwidth `text_{t+1}` channel; hard one-hot collapses the channel to exactly `⌈log2 V⌉` bits. This is the single most important change vs v1. |
| L4 | `k` carries `t+1` info | `k` is a function of `text_t` only (SlotEncoder sees `src_ids`). The posterior's `text_{t+1}` pass is isolated in `transition.py` and its sole output is `v`. |
| L5 | Shared-trunk gradient leaks `t+1` into `k` at the **next** step | Gradient ≠ activation. The trunk *weights* are shaped by both passes, but at forward time the noun path only ever *runs* `encode_text(text_t)`. No `t+1` activation reaches `k`. (This is the same reasoning as weight-tying in VAEs.) |
| L6 | Readout/predictor (`zhat`/`L_pred`) leaks via EMA target | The EMA target `z` is stop-grad and only supervises `zhat` (an MSE scalar), not the decoder. `zhat` is not decoder memory. |
| L7 | Prior sees `t+1` | Prior reads `pool_t` (`text_t`) only. KL target is stop-grad posterior. |
| L8 | Positional encoding on `a*` memory reconstructs slot→token alignment | `ARDecoder` adds **no** pos-enc to memory (set invariance, already enforced). |

**Net:** with L3 fixed (hard ST), the decoder's conditioning bandwidth from the future is `⌈log2 8⌉ = 3` bits per pair (nano). If `L_token` drops materially below the `v`-free baseline (a decoder conditioned on `k` with `v` ablated to a constant), those 3 bits are doing causal work — the **`v`-ablation CE gap** is the headline v2 success metric (§8).

---

## 7. Data path (file: `src/twm/jepa/data.py` — v2 ADDITIONS, additive)

v1 `JEPAChainDataset` already yields `{src_ids, src_pad, tgt_ids, tgt_pad}` cross-state pairs. v2 needs **`<eos>` appended to targets** (and srcs, for symmetry) so the decoder learns to stop. Two clean options; **DECIDED: option (a)** to keep file ownership disjoint:

- **(a) `JEPAChainDatasetV2(JEPAChainDataset)`** subclass in `data.py` (owned by the **model task**, T-task C, which already owns `data.py`? — NO: `data.py` is v1-owned). **Resolution:** the v2 dataset lives in the **model/config task's** new file is not allowed to touch `data.py`. Therefore **append-eos is implemented as a flag on the existing `JEPAChainDataset.__init__(..., append_eos: bool = False)`**, and `data.py` ownership for *this one additive kwarg* belongs to **T-task C** (model/scripts/config), which is explicitly permitted to add the `append_eos` path to `data.py` as its sole v1-file edit. The flag defaults `False` (v1 behavior preserved); v2 config sets it `True`.
  - When `append_eos=True`: after `tokenizer.encode(text, max_length=T)`, if the sequence has room before the first pad, insert `<eos>=4` at the first pad position (or at `T-1` if full). Pad mask recomputed so `<eos>` is **not** masked (it is a real target token) and positions after it are.
- `tgt_ids` now ends in `<eos>` then pad; `tgt_pad` is `True` only on the trailing pad (post-eos). `src` likewise (harmless; encoder ignores it via pad mask and `<eos>` is just another token to the trunk).

No new tokenizer build — `jepa_bpe_512.json` already has `<eos>=4`.

---

## 8. Diagnostics (file: `src/twm/jepa/diagnostics_v2.py` — NEW; v1 `diagnostics.py` untouched)

`eval_diagnostics_v2(model, dataset, device, n_examples=512, out_dir=None) -> dict` + PNGs, same calling convention as v1 (§ v1 diagnostics.py signature). v2 **adds** (and re-imports the v1 noun-geometry / scale-drift / slot-entropy helpers it can reuse):

| Group | Metric | Pass / alarm | Notes |
|---|---|---|---|
| **Generated text (THE quality artifact)** | greedy + temperature(0.7,1.0) `text_{t+1}` samples for N held-out pairs, logged as a table `{text_t, gold_t+1, v_posterior, v_prior, gen_greedy, gen_temp}` | qualitative; written to `out_dir/samples_epoch{e}.json` and as `chrF`/exact-match vs gold | The replacement for v1's latent-only "quality" — now human-readable. |
| **`v`-ablation CE gap (headline)** | `L_token` with true posterior `v` vs `L_token` with `v` forced to a **single constant code** | gap **> 0.1 nats** sustained ⇒ `v` carries causal info; gap ≈ 0 ⇒ regression to v1 (verb decorative) | The direct test that §6's 3 bits do work. |
| **Latent-action usage ppl** (diagnostic only) | perplexity of posterior `argmax v` over held set; prior `argmax v`; posterior↔prior agreement rate | ppl healthy ≈ `V/2..V`; **but high ppl is NOT sufficient** (v1 had 6.2/8 and was empty) — interpret jointly with the CE gap | Carries v1's lesson: usage ≠ semantics. |
| **Emergent action semantics probe** | cluster held-out pairs by `argmax v`; emit per-`v` example table (3–5 `(text_t → text_t+1)` pairs per code) to `out_dir/action_semantics_epoch{e}.json` | human-readable; do codes correspond to causation types (physical/mental/social)? | The "what did the codebook learn" artifact. |
| **Retrieval hard-negative MRR (REGRESSION metric)** | port `probe1_retrieval.py` logic: `easy_pool` and `hard_pool` (same-chain negatives) MRR for `zhat` vs EMA `z` | **`easy_minus_hard_mrr` must become ≥ 0** (v1 was −0.041, below chance) | Direct regression guard against the v1 identity-bias failure. |
| Verb–action MI (regression) | port `probe2_verb_mi` seq-level NMI of `v` vs keyword-reversibility labels | `nmi_minus_shuffle > 0` (v1 ≈ 0) | Optional; runs if labels available. |
| Noun geometry / scale-drift / slot-entropy | reuse v1 helpers (eff_rank, `log r` hist, attn entropy) | unchanged v1 thresholds | Imported from `diagnostics.py`, not reimplemented. |

The **generated-text samples** and the **`v`-ablation CE gap** are first-class and run every diag epoch.

---

## 9. Param budget — nano-v2 (≤ ~250K target)

Encoder is the v1 SlotEncoder **minus VerbHead** (`64·8+8 = 520` params removed; negligible). All counts trainable, exclude frozen `token_emb` (512·64 shared/frozen).

| Module | File | Params | Notes |
|---|---|---:|---|
| text_pos_emb | slot_encoder (v1) | 4,096 | |
| text self-attn (ALBERT-tied ×2) | slot_encoder (v1) | 33,152 | shared block |
| slot queries + μ/σ init | slot_encoder (v1) | 1,536 | |
| slot cross-attn | slot_encoder (v1) | 16,384 | |
| slot coordination (×3 shared) | slot_encoder (v1) | 16,384 | |
| NounHead (64→32) | slot_encoder (v1) | 2,048 | |
| ~~VerbHead~~ | — | **0** | **deleted in v2** |
| **Encoder subtotal** | | **≈ 73.6K** | (v1 was 77K incl. VerbHead) |
| Operator codebook (θ + log r) | operator (v1) | 256 | V·dn = 8·32 |
| Readout attn-pool (q dim dn=32) | model (v1) | 4,128 | |
| Predictor MLP (32→32) | model (v1) | 2,048 | kept (L_pred aux) |
| **TransitionEncoder MLP** | transition.py (NEW) | **25,864** | §2.2; shared trunk = 0 new attn |
| **PriorHead MLP** | transition.py (NEW) | **4,680** | §3 |
| **TokenDecoder** (`ARDecoder`, d_dec=64, L=2, h=4, ff=128) | decoder.py (NEW, adapter) | **≈ 132.5K** | breakdown below |
| **nano-v2 trainable, non-embedding TOTAL** | | **≈ 243.5K** | **≤ 250K ✓** |

**TokenDecoder breakdown** (`ARDecoder`, vocab=512, d=64, n_layers=2, n_heads=4, d_ff=128, bottleneck_dim=32):
- token_emb (decoder's own, NOT shared) `512·64 = 32,768`
- pos_emb `(64+17)·64 = 5,184`
- memory_proj `Linear(32→64)=2,112 + LN 128 = 2,240`
- 2× `TransformerDecoderLayer` (norm_first): per layer self-attn `4·64²=16,384` + cross-attn `4·64²=16,384` + FFN `2·64·128=16,384` + 3 LN `384` ≈ `49,536`; ×2 = `99,072`
- ln_f `128` + output_proj `64·512+512 = 33,280`
- **decoder subtotal ≈ 172.7K** ⚠️ — *over the §9 line item.*

**RECONCILIATION (DECIDED):** the decoder's own `token_emb` (32.8K) and `output_proj` (33.3K) dominate and scale with vocab. To land the **total at ≤ 250K**, set decoder **`n_layers = 1`** for nano-v2 (drops one 49.5K layer): decoder subtotal `≈ 123.2K`. Revised total:

| | Params |
|---|---:|
| Encoder subtotal | 73.6K |
| Operator + Readout + Predictor | 6.4K |
| TransitionEncoder + PriorHead | 30.5K |
| **TokenDecoder (d_dec=64, n_layers=1)** | **123.2K** |
| **nano-v2 TOTAL** | **≈ 233.7K ≤ 250K ✓** |

**Locked nano-v2 decoder config: `d_dec=64, n_layers=1, n_heads=4, d_ff=128, max_text_tokens=64`.** If 1 layer underfits in the smoke run, the first lever is `d_ff→256` (+8.4K, still under budget) before adding a layer. The 2-layer / `d_dec=128` decoder is a **mini-v2** concern, out of scope here.

---

## 10. Config schema (`configs/jepa_nano_v2.json`) + config additions

Extends the v1 schema (`JEPA_PROFILES`, dataclasses in `__init__.py`). New profile `jepa_nano_v2` and three new dataclasses; **the model/config task (T-C) owns these additions to `config.py` and the `__init__.py` registry**, per §11.

```json
{
  "profile": "jepa_nano_v2",
  "seed": 0,
  "data": {
    "path": "data/glucose/chain_general_train.jsonl",
    "tokenizer": "data/glucose/jepa_bpe_512.json",
    "vocab_size": 512,
    "max_text_tokens": 64,
    "pairing": "adjacent",
    "append_eos": true
  },
  "model": {
    "d_model": 64, "d_noun": 32, "n_slots": 8, "n_verbs": 8,
    "block": 2, "n_text_layers": 2, "tie_text_layers": true,
    "n_heads": 4, "n_slot_iters": 3,
    "operator_group": "rotation_scale", "n_steps_T": 1,
    "transition": { "mlp_hidden": 128, "use_delta": true },
    "prior": { "mlp_hidden": 64 },
    "decoder": { "d_dec": 64, "n_layers": 1, "n_heads": 4, "d_ff": 128 }
  },
  "loss": {
    "w_token": 1.0, "w_prior": 0.1, "w_sigreg": 0.05, "w_pred": 0.25,
    "sigreg": { "n_slices": 256, "n_knots": 17, "knot_max": 3.0, "standardize": true },
    "verb": { "gumbel_tau_start": 3.0, "gumbel_tau_end": 1.0, "anneal_frac": 0.5 }
  },
  "ema": { "tau": 0.995, "schedule": "fixed" },
  "optim": { "lr": 3e-4, "weight_decay": 0.01, "batch_size": 64, "epochs": 100, "grad_clip": 1.0, "warmup_steps": 200 },
  "eval": { "every_epochs": 5, "n_examples": 512, "out_dir": "results/jepa_nano_v2", "n_text_samples": 16, "temperatures": [0.7, 1.0] }
}
```

**New dataclasses** (added to `__init__.py` registry by T-C, parsed by `config.py`):
```python
@dataclass
class TransitionConfig:   # model.transition
    mlp_hidden: int = 128
    use_delta: bool = True

@dataclass
class PriorConfig:        # model.prior
    mlp_hidden: int = 64

@dataclass
class DecoderConfig:      # model.decoder
    d_dec: int = 64
    n_layers: int = 1
    n_heads: int = 4
    d_ff: int = 128
```
- `ModelHParams` gains `transition: TransitionConfig`, `prior: PriorConfig`, `decoder: DecoderConfig` (default_factory). `config.py::from_dict` parses the nested `model.{transition,prior,decoder}` blocks with the existing `_only_known` helper.
- `LossConfig` gains `w_token: float = 1.0`, `w_prior: float = 0.1`, **drops** `w_div`/`w_scale_reg` from the *v2 schema* (leave them on the dataclass with defaults so v1 configs still parse; v2 loss ignores them).
- `DataConfig` gains `append_eos: bool = False` (set `true` in v2 config; default preserves v1).
- `EvalConfig` gains `n_text_samples: int = 16`, `temperatures: list[float] = [0.7, 1.0]`.
- `JEPA_PROFILES["jepa_nano_v2"]` = the v1 `jepa_nano` dict + `{"decoder_n_layers": 1, ...}` (nested blocks parsed from JSON, not the flat profile — keep the profile minimal, let the JSON `model` block carry the nested configs).

---

## 11. Work breakdown — 4 parallel tasks, disjoint file ownership

Interfaces are frozen below (signatures). Tasks A/B develop against these; Task C wires them; Task D consumes outputs. Each task owns **disjoint files**; the only shared v1-file edits are **explicitly assigned to Task C** (`config.py` parsing, `__init__.py` registry, the single `append_eos` kwarg on `data.py`). **No task edits v1 `operator.py`, `slot_encoder.py`, `losses.py`, `diagnostics.py`, or `model.py`.** Tests use **new filenames only** under `tests/jepa/`.

| Task | Owns (writes) | Reads (no writes) | Deliverable + frozen interface |
|---|---|---|---|
| **A — Action heads** | `src/twm/jepa/transition.py`; `tests/jepa/test_transition.py` | `slot_encoder.py` (`encode_text` signature), `losses.py` (`gumbel_softmax_sample`, `anneal_tau`) | `TransitionEncoder(encode_text_fn, d_model, n_verbs, mlp_hidden, use_delta)` with `.forward(src_ids,src_pad,tgt_ids,tgt_pad,tau,hard)->(v_onehot (B,V), v_logits (B,V), pool_t (B,d))`. `PriorHead(d_model, n_verbs, mlp_hidden)` with `.forward(pool_t)->p_logits (B,V)`. fp32 under autocast for the Gumbel sample (mirror operator gotcha). Test: hard one-hot sums to 1, gradient reaches `v_logits`, prior shape, delta channel on/off. |
| **B — Decoder adapter** | `src/twm/jepa/decoder.py`; `tests/jepa/test_decoder_v2.py` | `ar_decoder.py` (`ARDecoder`) | `TokenDecoder(vocab_size, d_dec, n_layers, n_heads, d_ff, d_noun, max_text_tokens, pad_id=0)` wrapping `ARDecoder` (set `bottleneck_dim=d_noun`). `.forward(a_star (B,M,dn), tgt_ids, tgt_pad)->logits (B,T,V)`; `.generate(a_star, max_tokens, temperature)->(B,T)`. Constructor has **no posterior/text_t+1 arg** (leakage contract). Test: logits shape, finite CE, generate runs, constructor signature has no `t+1` channel. |
| **C — Model + config + script** | `src/twm/jepa/model_v2.py`; `scripts/train_jepa_v2.py`; `configs/jepa_nano_v2.json`; **+ registry/parse edits** to `src/twm/jepa/__init__.py` (3 dataclasses, `jepa_nano_v2` profile, `ModelHParams`/`LossConfig`/`DataConfig`/`EvalConfig` field additions) and `src/twm/jepa/config.py` (nested parse) and the single `append_eos` kwarg on `src/twm/jepa/data.py`; `tests/jepa/test_model_v2.py` | A's `TransitionEncoder`/`PriorHead`, B's `TokenDecoder`, v1 `SlotEncoder`/`RotationScaleOperator`/`Readout`/`Predictor`/`JEPAChainDataset`, D's `JEPALossV2` | `JEPAOperatorModelV2`: composes v1 SlotEncoder (no VerbHead use) + Operator + Readout + Predictor + EMA (reuse v1 pattern) + TransitionEncoder + PriorHead + TokenDecoder. `forward(src_ids,src_pad,tgt_ids,tgt_pad,tau,hard)->{k,a,v,v_logits,p_logits,zhat,z_target,logits}`. Inference `rollout(src_ids,src_pad, sample=True\|verb_idx)->{v, a, gen_ids}`. Trainer: AdamW + CosineLR on online params, manual EMA update, τ anneal (§2.4), periodic `eval_diagnostics_v2`. Test: end-to-end forward, param count ≤ 250K, leakage assert (decoder memory id is `a`, not `tgt` encodings), EMA excluded from optimizer. |
| **D — Losses + diagnostics** | `src/twm/jepa/losses_v2.py`; `src/twm/jepa/diagnostics_v2.py`; `tests/jepa/test_losses_v2.py`; `tests/jepa/test_diagnostics_v2.py` | v1 `losses.py` (`sigreg_loss`, `anneal_tau`), `diagnostics.py` (noun/scale/entropy helpers), `results/jepa_nano_probe/probe1_retrieval.py`, `probe2_verb_mi.py` (port logic) | `JEPALossV2(operator, w_token=1.0, w_prior=0.1, w_sigreg=0.05, w_pred=0.25)` with `.forward(logits,tgt_ids,tgt_pad, k, v_logits,p_logits, zhat,z_target, tau)->(total, components)`. `token_ce`, `prior_kl(q_logits, p_logits, tau)` (stopgrad q). `eval_diagnostics_v2(model, dataset, device, n_examples, out_dir, n_text_samples, temperatures)->dict` with v-ablation CE gap, gen-text samples, action-semantics table, hard-neg MRR (port), usage ppl. Test: token CE finite/masks pad, KL ≥ 0 and 0 at q==p, v-ablation gap computed, MRR port returns same keys as probe1 json. |

**Dependency order (for the shared stub):** Task C writes the frozen signatures for A/B/D into a short docstring block at the top of `model_v2.py` first (mirrors v1's `__init__.py`-as-stub pattern), then A/B/D develop against them in parallel; C integrates last.

---

## 12. Anti-goal compliance (v2)

- ✅ **Token decoder is now REQUIRED** (v1 anti-goal explicitly revoked by user).
- ✅ One **sequence-level** discrete action per pair (NOT per-slot) — kills the positional cheat.
- ✅ `L_div` **deleted**; informativeness from necessity (token CE), usage monitored as diagnostic only.
- ✅ Hard ST Gumbel (NOT soft mix) — bounds the future→decoder channel to `log2 V` bits.
- ✅ KEPT from v1: RotationScaleOperator (+ structural inverse), SlotEncoder noun path, SIGReg (standardize), EMA target, L_pred (as aux), diagnostics philosophy.
- ✅ T hard-set to 1 (operator `integrate` seam dormant). SO(n) still a stub.
- ✅ No new v1-file edits except the assigned C-task registry/parse/`append_eos` additions; all tests new filenames.
- ✅ Leakage invariant enforced and tested (§6, Task B/C contract tests).

---

## 13. Open questions (honest)

- **O1 — 3 bits enough?** nano `V=8` ⇒ 3 bits per transition. GLUCOSE causal steps may need more action granularity; if the `v`-ablation CE gap saturates low, bump `V` (16/32) — cheap (operator codebook is `V·dn`). Watched via the CE gap (§8).
- **O2 — Posterior collapse to identity action.** If the decoder can reconstruct `text_{t+1}` from `k` alone (high topic overlap between `t` and `t+1` on GLUCOSE), `v` becomes vestigial (CE gap ≈ 0) — the v1 failure in a new disguise. Mitigation lever: raise `w_prior` is wrong (steers posterior); instead **dropout the `a*` memory's `k`-dependence** is not available — the real lever is **information-penalizing the posterior** (KL to a uniform prior with a small weight) if collapse appears. Left OFF by default; flagged for the smoke run.
- **O3 — Multi-modality vs greedy rollout.** The categorical prior represents multi-modal futures, but greedy argmax picks one mode. Temperature sampling (§8) exposes the others; whether the *modes are coherent* is an empirical question for the action-semantics probe.
- **O4 — Shared-trunk gradient interference.** The posterior's `text_{t+1}` gradient reshapes the shared trunk; if this degrades noun geometry (SIGReg eff_rank drops), give the posterior its own tiny trunk (the §2.1 separate-trunk path, +33K — would push nano-v2 to ~267K, so this is a mini-v2 fallback).
- **O5 — Decoder 1-layer capacity.** nano-v2 ships a 1-layer decoder for budget. If text samples are incoherent, `d_ff→256` first, then mini-v2 (2 layers, d_dec=128).
