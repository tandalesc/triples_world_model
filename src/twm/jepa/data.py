"""JEPA chain dataset over GLUCOSE adjacent (state_t, state_{t+1}) pairs.

Spec reference: jepa_operator_v1_design.md §6 and T4 row in §12.

Cross-state pairing is mandatory: online encoder sees state_t; EMA target encoder
sees state_{t+1}. Same text to both encoders degenerates into self-reconstruction
with a moving target — no cross-state JEPA signal (spec §6 FIX, Judge 2 D2 flaw).

Storage: contiguous CPU tensors, direct index slicing (no DataLoader).
"""

import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from ..domain_bpe import DomainBPETokenizer


def _diff_mask(
    src_ids: list[int],
    tgt_ids: list[int],
    pad_id: int,
    T: int,
) -> Tensor:
    """Per-token boolean diff mask for masked-diff prediction (v4.2, trad-JEPA masking).

    Aligns the two token-id sequences with the SAME SequenceMatcher (LCS) alignment
    `_diff_weights` uses, and marks TRUE the TARGET positions that are part of the
    s_t→s_{t+1} diff (the `replace`/`insert` opcodes — the CHANGED span). `equal`
    (boilerplate) and `delete` (source-only) positions stay FALSE. Pad positions
    (>= len(tgt) real tokens) stay FALSE.

    This is the CHANGED span the masked-diff objective focuses on: the decoder's
    teacher-forcing INPUT is corrupted (replaced with the mask id) at these positions,
    and the CE is computed ONLY here — 100% discriminative tokens, no boilerplate.

    ALL-EQUAL edge case (identity-ish pairs, no `replace`/`insert` op): the mask is
    all-FALSE. The masked-diff loss for such a pair contributes ZERO (no masked
    positions ⟹ the per-pair CE denominator is empty ⟹ excluded). This is the
    documented choice: SKIP the masked-diff loss for an all-equal pair rather than
    fabricate a random span — a genuinely-unchanged pair has no causal diff to predict,
    so masking a random boilerplate token would teach surface infilling, exactly the
    failure mode v4.2 is built to avoid. (The full-reconstruction L_token still trains
    on these pairs, so generation health is unaffected.)

    Args:
        src_ids: state_t token ids (post-encode, may include trailing pad).
        tgt_ids: state_{t+1} token ids (the CE target; its positions are masked/scored).
        pad_id:  padding id (trailing pad in either list is stripped before alignment).
        T:       output length (== max_text_tokens).

    Returns:
        (T,) bool mask, TRUE at the changed target span, FALSE on boilerplate/pad.
    """
    n_tgt = len(tgt_ids)
    while n_tgt > 0 and tgt_ids[n_tgt - 1] == pad_id:
        n_tgt -= 1
    n_src = len(src_ids)
    while n_src > 0 and src_ids[n_src - 1] == pad_id:
        n_src -= 1
    src_real = src_ids[:n_src]
    tgt_real = tgt_ids[:n_tgt]

    mask = torch.zeros(T, dtype=torch.bool)
    sm = SequenceMatcher(a=src_real, b=tgt_real, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):          # target tokens j1:j2 ARE the diff
            hi = min(j2, T)
            for j in range(j1, hi):
                mask[j] = True
        # "delete" (src-only) has no target position; "equal" stays False.
    return mask


def _random_span_mask(
    tgt_ids: list[int],
    pad_id: int,
    T: int,
    rng: random.Random,
    min_spans: int = 1,
    max_spans: int = 3,
    min_cov: float = 0.20,
    max_cov: float = 0.35,
) -> Tensor:
    """Per-token boolean span mask for v4.4 random-span masked reconstruction.

    Surface-diversity reframing (Wikipedia, v4.4): there is NO s_t→s_{t+1} diff to focus
    on (a chain is just adjacent prose sentences, not a state transition). Instead of the
    causal diff span, we mask 1-3 CONTIGUOUS spans covering 20-35% of the target's REAL
    (non-pad) tokens — the standard span-corruption masked-LM recipe (SpanBERT/T5 style),
    re-used through the SAME <mask>-input-corruption + masked-positions-only CE path as the
    diff objective.

    Sampling (per call, driven by `rng`):
      1. n_real = count of non-pad tokens (pad is stripped — pad is never masked).
      2. coverage fraction c ~ U[min_cov, max_cov]; n_mask = round(c · n_real),
         clamped to [1, n_real] (a 1-token target masks its single token).
      3. n_spans ~ randint(min_spans, max_spans), capped at n_mask (cannot have more
         spans than masked tokens) and at the count that fits without overlap.
      4. Partition n_mask masked tokens into `n_spans` span LENGTHS (each >= 1), then
         place the spans left-to-right into the n_real positions with the leftover gap
         budget distributed BEFORE/BETWEEN/AFTER the spans (seeded), so the spans are
         contiguous, non-overlapping, and stay within the real (non-pad) region.

    Determinism: WHEN this is computed (per-load, see JEPAChainDataset) the spans are
    sampled once with a seeded `rng` — NOT resampled per epoch. This is the documented
    choice: the trainer precomputes all per-example mask tensors at load (no DataLoader,
    direct index slicing — data.py module docstring), so a per-epoch resample would mean
    rebuilding (N,T) bool tensors every epoch. A fixed seeded mask per example is the
    cheap, reproducible option; coverage is still drawn per-example so the corpus sees a
    spread of mask rates. Pad positions stay FALSE; an empty/all-pad target ⟹ all-FALSE
    (the masked-diff CE skips it, exactly the diff-mode all-equal edge case).

    Args:
        tgt_ids: state token ids (the CE target; its positions are masked/scored).
        pad_id:  padding id (trailing pad is stripped; pad is never masked).
        T:       output length (== max_text_tokens).
        rng:     seeded random.Random for reproducible span sampling.
        min_spans/max_spans: number of contiguous spans (default 1-3).
        min_cov/max_cov:     fraction of real tokens to mask (default 0.20-0.35).

    Returns:
        (T,) bool mask, TRUE at the masked spans, FALSE on unmasked/pad positions.
    """
    n_real = len(tgt_ids)
    while n_real > 0 and tgt_ids[n_real - 1] == pad_id:
        n_real -= 1

    mask = torch.zeros(T, dtype=torch.bool)
    if n_real == 0:
        return mask

    cov = rng.uniform(min_cov, max_cov)
    n_mask = max(1, min(n_real, round(cov * n_real)))
    n_spans = rng.randint(min_spans, max_spans)
    n_spans = max(1, min(n_spans, n_mask))

    # Partition n_mask into n_spans positive integer lengths (each >= 1).
    # Start at all-ones, distribute the remaining (n_mask - n_spans) one at a time.
    span_lens = [1] * n_spans
    for _ in range(n_mask - n_spans):
        span_lens[rng.randrange(n_spans)] += 1

    # Gap budget = real positions not masked, distributed into n_spans+1 gaps
    # (before / between / after). Each "between" gap is >= 1 so spans don't merge
    # into one contiguous block (keeps them distinct spans); before/after may be 0.
    total_gap = n_real - n_mask
    n_gaps = n_spans + 1
    # Reserve one slot per internal gap so adjacent spans are separated (when budget
    # allows); if budget is too tight, spans may abut (acceptable — still <= max_spans).
    internal = n_spans - 1
    reserved = min(internal, total_gap)
    free_gap = total_gap - reserved
    gaps = [0] * n_gaps
    for g in range(1, n_spans):  # internal gaps get their reserved 1
        if reserved > 0:
            gaps[g] = 1
            reserved -= 1
    for _ in range(free_gap):
        gaps[rng.randrange(n_gaps)] += 1

    # Lay spans out left to right.
    pos = gaps[0]
    for s in range(n_spans):
        hi = min(pos + span_lens[s], n_real, T)
        for j in range(pos, hi):
            mask[j] = True
        pos = hi + gaps[s + 1]
    return mask


def _diff_weights(
    src_ids: list[int],
    tgt_ids: list[int],
    w_diff: float,
    pad_id: int,
    T: int,
) -> Tensor:
    """Per-token diff weights for diff-weighted CE (design v4 §2.1).

    Aligns the two token-id sequences with a SequenceMatcher (LCS) and marks the TARGET
    tokens that are NOT in a shared block (the inserted/replaced tokens — the s_t→s_{t+1}
    diff) with weight `w_diff`; shared (`equal`) boilerplate tokens get 1.0. Source-only
    (`delete`) tokens have no target position and are skipped. The weight vector is padded
    / truncated to length `T`; pad positions (beyond `len(tgt_ids)`) get weight 0.0 (they
    are excluded from the weighted-CE denominator anyway).

    When `w_diff == 1.0` (default) every weight is 1.0 over the non-pad target positions
    (pad ⟹ 0.0), so the weighted CE is BITWISE the v3 uniform mean CE. The diff is always
    computed at the BPE-token level (the CE is over token positions), matching the loss.

    Args:
        src_ids: state_t token ids (post-encode, pre-pad-stripped list[int]).
        tgt_ids: state_{t+1} token ids (the CE target; its positions are weighted).
        w_diff:  weight for diff (replace/insert) target tokens.
        pad_id:  padding id (the source/target lists may include trailing pad ids; pad
                 positions in the OUTPUT vector are zeroed regardless).
        T:       output length (== max_text_tokens).

    Returns:
        (T,) float weight tensor aligned to tgt_ids positions.
    """
    # Strip trailing pad so the alignment matches on the real content tokens; pad
    # positions in the target get weight 0.0 below regardless of the diff.
    n_tgt = len(tgt_ids)
    while n_tgt > 0 and tgt_ids[n_tgt - 1] == pad_id:
        n_tgt -= 1
    n_src = len(src_ids)
    while n_src > 0 and src_ids[n_src - 1] == pad_id:
        n_src -= 1
    src_real = src_ids[:n_src]
    tgt_real = tgt_ids[:n_tgt]

    w = [1.0] * n_tgt                              # boilerplate weight
    sm = SequenceMatcher(a=src_real, b=tgt_real, autojunk=False)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):          # target tokens j1:j2 ARE the diff
            for j in range(j1, j2):
                w[j] = w_diff
        # "delete" (src-only) has no target position; "equal" keeps weight 1.0.

    weights = torch.zeros(T, dtype=torch.float32)
    for j in range(min(n_tgt, T)):
        weights[j] = w[j]                          # pad positions (>= n_tgt) stay 0.0
    return weights


class JEPAChainDataset:
    """Adjacent cross-state pairs extracted from GLUCOSE chains.

    Each chain has 3 states [t0, t1, t2], yielding 2 pairs:
        (t0, t1) and (t1, t2).

    For N chains with chain_length=3, __len__() == 2·N.

    Tensors are stored contiguously on CPU (no DataLoader; the trainer does
    direct index slicing per the repo convention, matching chain_dataset.py).

    Args:
        path: path to chain_general_train.jsonl (or any GLUCOSE chain JSONL).
        tokenizer: a DomainBPETokenizer loaded from jepa_bpe_512.json.
        max_text_tokens: pad/truncate each state to this many tokens (spec: 64).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: DomainBPETokenizer,
        max_text_tokens: int = 64,
        append_eos: bool = False,
        mode: str = "pairs",
        w_diff: float = 1.0,
        compute_diff_mask: bool = False,
        mask_mode: str = "diff",
        mask_seed: int = 0,
        attach_labels: bool = False,
    ) -> None:
        if mode not in ("pairs", "triples"):
            raise ValueError(f"mode must be 'pairs' or 'triples', got {mode!r}")
        if mask_mode not in ("diff", "random_span"):
            raise ValueError(
                f"mask_mode must be 'diff' or 'random_span', got {mask_mode!r}"
            )
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.append_eos = append_eos
        self.mode = mode
        # v5 Step-1a oracle labels (entity-world only). When True AND a `<stem>_labeled.jsonl`
        # twin exists, each transition additionally carries oracle_verb_id (int, the GT verb
        # mapped through entity_labels.ORACLE_VERBS; -1 if absent) and canonical_next_state_id
        # (int, the oracle canonical next-state via apply_action replay). DEFAULT FALSE ⟹ the
        # labeled twin is never read, NO label tensors are built, and the __getitem__/get_batch
        # keys are absent — so GLUCOSE/unlabeled configs are bitwise-unchanged. Resolved to the
        # actual availability in `self.has_labels` after the build (False if the twin is missing).
        self.attach_labels = attach_labels
        self.has_labels = False
        # v4.4 masked-reconstruction mode selector. "diff" (default) = the v4.2 causal
        # diff span (bitwise-unchanged). "random_span" = 1-3 contiguous spans covering
        # 20-35% of the target's non-pad tokens, sampled once per example at load with a
        # seeded RNG (see _random_span_mask for the per-load-vs-per-epoch rationale).
        self.mask_mode = mask_mode
        # Per-load RNG for reproducible random-span masks. Only consumed when
        # mask_mode == "random_span" AND compute_diff_mask is set.
        self._mask_rng = random.Random(mask_seed)
        # Diff-weighted CE (v4 §2): per-token weights on the s_t→s_{t+1} token diff.
        # ALWAYS computed (cheap; runs once at load), so the trainer can flip w_diff
        # without a data rebuild. w_diff=1.0 (default) ⟹ all-ones over non-pad ⟹ v3
        # bitwise uniform CE. The weights live in _*_diff_w tensors (built below).
        self.w_diff = w_diff
        # Masked-diff prediction (v4.2 §1): per-target boolean diff masks (the CHANGED
        # span). Computed ONLY when enabled (the extra (N,T) bool tensors cost memory and
        # an O(T²) alignment per pair; gated so v3/v4.0/v4.1 runs pay nothing). The trainer
        # sets compute_diff_mask=True iff loss.w_masked_diff>0.
        self.compute_diff_mask = compute_diff_mask
        pad_id = tokenizer.pad_token_id  # 0 per domain_bpe.py convention
        # <eos>=4 per the GLUCOSE BPE artifact (design v2 §7). Only used when
        # append_eos=True so the AR token decoder learns to stop.
        eos_id = 4
        T = max_text_tokens

        def _insert_eos(ids: list[int]) -> list[int]:
            """Insert <eos> at the first pad slot (or overwrite T-1 if full) so the
            decoder learns to stop. <eos> stays a real (unmasked) target token;
            positions after it remain pad. List length is unchanged (== T)."""
            ids = list(ids)
            if pad_id in ids:
                ids[ids.index(pad_id)] = eos_id
            else:
                ids[-1] = eos_id  # sequence fills T — overwrite the last token
            return ids

        def _encode(text: str) -> tuple[Tensor, Tensor]:
            """Tokenize one state -> (ids (T,) long, pad (T,) bool). Reuses the shared
            _insert_eos / pad-mask logic (design §2.1: "Reuse the existing _insert_eos")."""
            ids = tokenizer.encode(text, max_length=T)
            if append_eos:
                ids = _insert_eos(ids)
            ids_t = torch.tensor(ids, dtype=torch.long)
            return ids_t, (ids_t == pad_id)

        # v5 Step-1a: load the labeled twin (oracle verbs + states) if requested AND present.
        # `_labeled` is a list aligned 1:1 with the plain chain file (same generation order).
        # The per-state canonical-next-state registry is shared across the whole dataset so
        # positives (same canonical state) get the SAME id everywhere (the L_sep label).
        self._labeled = None
        self._canon_registry = None
        if attach_labels:
            from .entity_labels import (
                load_labeled_records,
                labeled_path_for,
                CanonicalStateRegistry,
            )
            self._labeled = load_labeled_records(labeled_path_for(path))
            if self._labeled is not None:
                self._canon_registry = CanonicalStateRegistry()
                self.has_labels = True

        if mode == "pairs":
            self._build_pairs(path, _encode, T)
        else:
            self._build_triples(path, _encode, T)

    def _make_mask(self, src_ids: list[int], tgt_ids: list[int], pad_id: int, T: int) -> Tensor:
        """Build the per-target masked-reconstruction mask for the active `mask_mode`.

        - "diff" (v4.2): the causal s_t→s_{t+1} changed span (_diff_mask). Bitwise-unchanged.
        - "random_span" (v4.4): 1-3 contiguous spans over 20-35% of the non-pad target
          tokens (_random_span_mask), independent of `src_ids` — there is no causal diff on
          adjacent prose, so the source is ignored and a seeded random span is masked.
        """
        if self.mask_mode == "random_span":
            return _random_span_mask(tgt_ids, pad_id, T, self._mask_rng)
        return _diff_mask(src_ids, tgt_ids, pad_id, T)

    # ------------------------------------------------------------------
    # v5 Step-1a oracle-label helper (entity-world only)
    # ------------------------------------------------------------------

    def _chain_labels(self, chain_no: int, n_states: int):
        """Per-transition oracle labels for the chain at `chain_no` (v5 Step-1a).

        Returns (verb_ids, canon_ids) where:
          - verb_ids[i]  = oracle verb id of transition state_i -> state_{i+1}
                           (entity_labels.ORACLE_VERBS index; -1 if no/unknown label).
          - canon_ids[i] = canonical id of the oracle next-state (state_{i+1}),
                           shared across the dataset so paraphrase renderings of the same
                           underlying state collapse to one id; -1 if unavailable.
        Both lists have length n_states-1 (one per transition). When labels are absent
        (no twin, or this record lacks the fields) every entry is -1 — the loss treats
        -1 as ignore (verb CE ignore_index) / excludes it from L_sep positives.
        """
        from .entity_labels import verb_to_id, replay_canonical_states

        n_trans = max(0, n_states - 1)
        verb_ids = [-1] * n_trans
        canon_ids = [-1] * n_trans
        if self._labeled is None or chain_no >= len(self._labeled):
            return verb_ids, canon_ids
        rec = self._labeled[chain_no]
        actions = rec.get("actions", []) or []
        types = rec.get("types", []) or []
        inits = rec.get("initial_states", []) or []
        # Verb ids straight from the action labels.
        for i in range(min(n_trans, len(actions))):
            from .entity_labels import parse_action_label
            verb, _ = parse_action_label(actions[i])
            verb_ids[i] = verb_to_id(verb)
        # Canonical next-state ids from an oracle replay (one canonical string per state).
        canon_strs = replay_canonical_states(types, inits, actions)
        if canon_strs is not None:
            for i in range(n_trans):
                # transition i targets state_{i+1} -> canon string at index i+1.
                ci = i + 1
                if ci < len(canon_strs):
                    canon_ids[i] = self._canon_registry.get(canon_strs[ci])
        return verb_ids, canon_ids

    # ------------------------------------------------------------------
    # Build helpers (mode-specific; one populates _src/_tgt, the other _s0/_s1/_s2)
    # ------------------------------------------------------------------

    def _build_pairs(self, path, _encode, T: int) -> None:
        """v2 pairs mode (unchanged behavior). A chain of length L yields L-1 adjacent
        (state_t, state_{t+1}) pairs. The two pairs of a length-3 chain SHARE a chain_id
        so the hard-negative MRR builds true same-chain negative pools (design §8.2)."""
        src_texts: list[str] = []
        tgt_texts: list[str] = []
        chain_ids: list[int] = []
        # v5 Step-1a per-pair oracle labels (parallel to src/tgt; all -1 when no twin).
        verb_lbls: list[int] = []
        canon_lbls: list[int] = []

        with open(path) as f:
            for chain_no, line in enumerate(f):
                data = json.loads(line)
                chain: list[str] = data["chain"]
                v_ids, c_ids = (
                    self._chain_labels(chain_no, len(chain))
                    if self.has_labels else ([-1] * (len(chain) - 1), [-1] * (len(chain) - 1))
                )
                for i in range(len(chain) - 1):
                    src_texts.append(chain[i])
                    tgt_texts.append(chain[i + 1])
                    chain_ids.append(chain_no)
                    verb_lbls.append(v_ids[i] if i < len(v_ids) else -1)
                    canon_lbls.append(c_ids[i] if i < len(c_ids) else -1)

        n = len(src_texts)
        self._src_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._src_pad: Tensor = torch.ones((n, T), dtype=torch.bool)
        self._tgt_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._tgt_pad: Tensor = torch.ones((n, T), dtype=torch.bool)

        # Diff-weighted CE (§2.1): per-token weights on the (src→tgt) token diff. The CE
        # target is tgt, so the weights are aligned to tgt positions. Always computed;
        # all-ones (over non-pad) when w_diff==1.0 ⟹ v3-bitwise uniform CE.
        self._tgt_diff_w: Tensor = torch.zeros((n, T), dtype=torch.float32)

        # Masked-diff prediction (v4.2 §1): boolean mask over the changed tgt span. Only
        # built when compute_diff_mask is set (else stays a None placeholder so the
        # __getitem__/get_batch keys are absent and the trainer skips the masked pass).
        self._tgt_diff_mask: Tensor | None = (
            torch.zeros((n, T), dtype=torch.bool) if self.compute_diff_mask else None
        )

        pad_id = self.tokenizer.pad_token_id
        for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
            self._src_ids[i], self._src_pad[i] = _encode(src)
            self._tgt_ids[i], self._tgt_pad[i] = _encode(tgt)
            self._tgt_diff_w[i] = _diff_weights(
                self._src_ids[i].tolist(), self._tgt_ids[i].tolist(),
                self.w_diff, pad_id, T,
            )
            if self._tgt_diff_mask is not None:
                self._tgt_diff_mask[i] = self._make_mask(
                    self._src_ids[i].tolist(), self._tgt_ids[i].tolist(), pad_id, T,
                )

        # Keep the raw texts for iter_text_pairs() (operator-fit pass-2 §7).
        self._src_texts: list[str] = src_texts
        self._tgt_texts: list[str] = tgt_texts
        # Per-pair originating chain id (design §8.2). len == len(dataset).
        self._chain_ids: list[int] = chain_ids
        # v5 Step-1a per-pair oracle labels (None when no twin; else (N,) long tensors).
        if self.has_labels:
            self._verb_id: Tensor | None = torch.tensor(verb_lbls, dtype=torch.long)
            self._canon_id: Tensor | None = torch.tensor(canon_lbls, dtype=torch.long)
        else:
            self._verb_id = None
            self._canon_id = None

    def _build_triples(self, path, _encode, T: int) -> None:
        """v3 triples mode (design §2.1). Emit ONE example per length-3 chain holding all
        three states (s0 start, s1 hop-1 target, s2 hop-2 target). Chains of length < 3
        are SKIPPED (GLUCOSE chain_general is uniformly length 3 — none should drop;
        the skipped count is logged via self.n_skipped). chain_ids: one id per chain so
        same-chain InfoNCE negatives still work; the hard negative in triple mode comes
        from the cross-hop targets within the SAME example (s1 vs s2)."""
        s0_texts: list[str] = []
        s1_texts: list[str] = []
        s2_texts: list[str] = []
        chain_ids: list[int] = []
        n_skipped = 0
        # v5 Step-1a per-hop oracle labels (one row per chain, two hops). All -1 w/o twin.
        verb_h1: list[int] = []
        verb_h2: list[int] = []
        canon_h1: list[int] = []
        canon_h2: list[int] = []

        with open(path) as f:
            for chain_no, line in enumerate(f):
                data = json.loads(line)
                chain: list[str] = data["chain"]
                if len(chain) < 3:
                    n_skipped += 1
                    continue
                # Use the first three states (chains are uniformly length 3; if longer,
                # take the leading triple — the unroll trains two adjacent hops from s0).
                s0_texts.append(chain[0])
                s1_texts.append(chain[1])
                s2_texts.append(chain[2])
                chain_ids.append(chain_no)
                # Hop-1 = transition 0 (s0->s1), hop-2 = transition 1 (s1->s2).
                v_ids, c_ids = (
                    self._chain_labels(chain_no, len(chain))
                    if self.has_labels else ([-1, -1], [-1, -1])
                )
                verb_h1.append(v_ids[0] if len(v_ids) > 0 else -1)
                verb_h2.append(v_ids[1] if len(v_ids) > 1 else -1)
                canon_h1.append(c_ids[0] if len(c_ids) > 0 else -1)
                canon_h2.append(c_ids[1] if len(c_ids) > 1 else -1)

        n = len(s0_texts)
        self.n_skipped = n_skipped
        # Triple-mode tensors are ADDITIVE; the pairs-mode _src/_tgt attrs are NOT set
        # in triple mode (design §2.1 back-compat: each mode populates only its own attrs).
        self._s0_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._s0_pad: Tensor = torch.ones((n, T), dtype=torch.bool)
        self._s1_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._s1_pad: Tensor = torch.ones((n, T), dtype=torch.bool)
        self._s2_ids: Tensor = torch.zeros((n, T), dtype=torch.long)
        self._s2_pad: Tensor = torch.ones((n, T), dtype=torch.bool)

        # Diff-weighted CE (§2.1), per hop: _s1_diff_w weights the s0→s1 diff (hop-1 CE
        # target s1), _s2_diff_w weights the s1→s2 diff (hop-2 CE target s2). Always
        # computed; all-ones over non-pad when w_diff==1.0 ⟹ v3-bitwise uniform CE.
        self._s1_diff_w: Tensor = torch.zeros((n, T), dtype=torch.float32)
        self._s2_diff_w: Tensor = torch.zeros((n, T), dtype=torch.float32)

        # Masked-diff prediction (v4.2 §1), per hop: _s1_diff_mask marks the changed s0→s1
        # span (hop-1 target s1), _s2_diff_mask the s1→s2 span (hop-2 target s2). Only built
        # when compute_diff_mask is set (else None placeholders ⟹ keys absent, masked pass
        # skipped — bitwise-neutral when w_masked_diff=0).
        self._s1_diff_mask: Tensor | None = (
            torch.zeros((n, T), dtype=torch.bool) if self.compute_diff_mask else None
        )
        self._s2_diff_mask: Tensor | None = (
            torch.zeros((n, T), dtype=torch.bool) if self.compute_diff_mask else None
        )

        pad_id = self.tokenizer.pad_token_id
        for i in range(n):
            self._s0_ids[i], self._s0_pad[i] = _encode(s0_texts[i])
            self._s1_ids[i], self._s1_pad[i] = _encode(s1_texts[i])
            self._s2_ids[i], self._s2_pad[i] = _encode(s2_texts[i])
            self._s1_diff_w[i] = _diff_weights(
                self._s0_ids[i].tolist(), self._s1_ids[i].tolist(),
                self.w_diff, pad_id, T,
            )
            self._s2_diff_w[i] = _diff_weights(
                self._s1_ids[i].tolist(), self._s2_ids[i].tolist(),
                self.w_diff, pad_id, T,
            )
            if self._s1_diff_mask is not None:
                self._s1_diff_mask[i] = self._make_mask(
                    self._s0_ids[i].tolist(), self._s1_ids[i].tolist(), pad_id, T,
                )
                self._s2_diff_mask[i] = self._make_mask(
                    self._s1_ids[i].tolist(), self._s2_ids[i].tolist(), pad_id, T,
                )

        self._s0_texts: list[str] = s0_texts
        self._s1_texts: list[str] = s1_texts
        self._s2_texts: list[str] = s2_texts
        # One id per chain (== example index's originating chain).
        self._chain_ids: list[int] = chain_ids
        # v5 Step-1a per-hop oracle labels (None when no twin; else (N,) long tensors).
        if self.has_labels:
            self._verb_id_h1: Tensor | None = torch.tensor(verb_h1, dtype=torch.long)
            self._verb_id_h2: Tensor | None = torch.tensor(verb_h2, dtype=torch.long)
            self._canon_id_h1: Tensor | None = torch.tensor(canon_h1, dtype=torch.long)
            self._canon_id_h2: Tensor | None = torch.tensor(canon_h2, dtype=torch.long)
        else:
            self._verb_id_h1 = self._verb_id_h2 = None
            self._canon_id_h1 = self._canon_id_h2 = None

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------

    @property
    def chain_ids(self) -> list[int]:
        """Per-pair originating chain id (design §8.2), len == len(self).

        Adjacent pairs of one chain share an id, so the diagnostics hard-negative MRR
        builds true same-chain negative pools. Stays aligned with the `max_chains` cap
        path (train_jepa_v2.py slices `_chain_ids` alongside the tensors).
        """
        return self._chain_ids

    def __len__(self) -> int:
        if self.mode == "triples":
            return self._s0_ids.shape[0]
        return self._src_ids.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return a single example.

        pairs mode — an adjacent (state_t, state_{t+1}) tensor pair:
            src_ids/src_pad: (T_text,) — tokenized state_t, padded.
            tgt_ids/tgt_pad: (T_text,) — tokenized state_{t+1}, padded.
        Cross-state pairing guarantee: src_ids and tgt_ids encode DIFFERENT states.

        triples mode — one (s0, s1, s2) chain example (design §2.1):
            s0_ids/s0_pad, s1_ids/s1_pad, s2_ids/s2_pad: (T_text,) for each state,
            plus chain_id (the originating chain index).
        """
        if self.mode == "triples":
            out = {
                "s0_ids": self._s0_ids[idx],
                "s0_pad": self._s0_pad[idx],
                "s1_ids": self._s1_ids[idx],
                "s1_pad": self._s1_pad[idx],
                "s2_ids": self._s2_ids[idx],
                "s2_pad": self._s2_pad[idx],
                "s1_diff_w": self._s1_diff_w[idx],
                "s2_diff_w": self._s2_diff_w[idx],
                "chain_id": self._chain_ids[idx],
            }
            if self._s1_diff_mask is not None:
                out["s1_diff_mask"] = self._s1_diff_mask[idx]
                out["s2_diff_mask"] = self._s2_diff_mask[idx]
            if self.has_labels:
                out["verb_id_h1"] = self._verb_id_h1[idx]
                out["verb_id_h2"] = self._verb_id_h2[idx]
                out["canon_id_h1"] = self._canon_id_h1[idx]
                out["canon_id_h2"] = self._canon_id_h2[idx]
            return out
        out = {
            "src_ids": self._src_ids[idx],
            "src_pad": self._src_pad[idx],
            "tgt_ids": self._tgt_ids[idx],
            "tgt_pad": self._tgt_pad[idx],
            "tgt_diff_w": self._tgt_diff_w[idx],
        }
        if self._tgt_diff_mask is not None:
            out["tgt_diff_mask"] = self._tgt_diff_mask[idx]
        if self.has_labels:
            out["verb_id"] = self._verb_id[idx]
            out["canon_id"] = self._canon_id[idx]
        return out

    def get_batch(self, indices) -> dict:
        """Return a batch of items for the given indices (list or Tensor).

        Convenience method for the no-DataLoader direct-slicing trainer convention.
        Returns tensors of shape (B, T_text) for each id/pad key; in triples mode also
        returns chain_id: (B,) long.
        """
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        if self.mode == "triples":
            out = {
                "s0_ids": self._s0_ids[indices],
                "s0_pad": self._s0_pad[indices],
                "s1_ids": self._s1_ids[indices],
                "s1_pad": self._s1_pad[indices],
                "s2_ids": self._s2_ids[indices],
                "s2_pad": self._s2_pad[indices],
                "s1_diff_w": self._s1_diff_w[indices],
                "s2_diff_w": self._s2_diff_w[indices],
                "chain_id": torch.tensor(
                    [self._chain_ids[i] for i in indices], dtype=torch.long
                ),
            }
            if self._s1_diff_mask is not None:
                out["s1_diff_mask"] = self._s1_diff_mask[indices]
                out["s2_diff_mask"] = self._s2_diff_mask[indices]
            if self.has_labels:
                out["verb_id_h1"] = self._verb_id_h1[indices]
                out["verb_id_h2"] = self._verb_id_h2[indices]
                out["canon_id_h1"] = self._canon_id_h1[indices]
                out["canon_id_h2"] = self._canon_id_h2[indices]
            return out
        out = {
            "src_ids": self._src_ids[indices],
            "src_pad": self._src_pad[indices],
            "tgt_ids": self._tgt_ids[indices],
            "tgt_pad": self._tgt_pad[indices],
            "tgt_diff_w": self._tgt_diff_w[indices],
        }
        if self._tgt_diff_mask is not None:
            out["tgt_diff_mask"] = self._tgt_diff_mask[indices]
        if self.has_labels:
            out["verb_id"] = self._verb_id[indices]
            out["canon_id"] = self._canon_id[indices]
        return out

    # ------------------------------------------------------------------
    # Operator-fit pass-2 interface (spec §7)
    # ------------------------------------------------------------------

    def iter_text_pairs(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield (src_ids, tgt_ids) token tensors for the operator-fit pass-2.

        Yields one (state_t, state_{t+1}) pair at a time as (T_text,) long tensors.
        Used by scripts/operator_group_fit.py pass-2 to re-verify the operator
        family choice in the trained noun space (spec §7).

        Only valid in pairs mode (the operator-fit pass-2 is a v2 IO probe). In triple
        mode the _src/_tgt tensors are not populated, so this raises.
        """
        if self.mode == "triples":
            raise RuntimeError(
                "iter_text_pairs() is pairs-mode only; triple mode does not populate "
                "_src_ids/_tgt_ids (design §2.1 back-compat)."
            )
        for i in range(len(self)):
            yield self._src_ids[i], self._tgt_ids[i]
