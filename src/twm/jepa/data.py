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
        lam_augment: bool = False,
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
        # v6 §B surface-augmentation invariance (research/jepa_v6_unsupervised_design.md,
        # arXiv:2506.15691). When True AND the chain JSONL carries a `chain_aug` field (a
        # SECOND independent surface frame φ' of the SAME chain states, emitted by the data
        # generator with a fresh chain_template_seed), each transition additionally carries:
        #   posterior frame φ : the existing `chain` rendering (posterior + noun path)
        #   decoder target φ' : the `chain_aug` rendering of the SAME s_{t+1} (the CE target)
        # The loss/trainer only ever see RENDERED TEXT — never an oracle state/label. The
        # underlying-state access that produced the two frames is the data generator's, not
        # the trainer's (the augmentation interface is pluggable: a text-level paraphraser
        # could supply `chain_aug` instead of the renderer — see the design doc). DEFAULT
        # FALSE ⟹ no second-frame tensors are built and the dataset is bitwise v4. Resolved
        # to actual availability in `self.lam_augment` after the build (False if no field).
        self._lam_augment_requested = lam_augment
        self.lam_augment = False
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

        self._encode_fn = _encode  # bound for the v6 §B paired-frame builders

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
    # Build helpers (mode-specific; one populates _src/_tgt, the other _s0/_s1/_s2)
    # ------------------------------------------------------------------

    def _build_pairs(self, path, _encode, T: int) -> None:
        """v2 pairs mode (unchanged behavior at lam_augment off). A chain of length L yields
        L-1 adjacent (state_t, state_{t+1}) pairs. The two pairs of a length-3 chain SHARE a
        chain_id so the hard-negative MRR builds true same-chain negative pools (design §8.2).

        v6 §B: when lam_augment is requested AND the record has a `chain_aug` field (a second
        independent surface frame φ' of the SAME chain states), the φ' renderings of s_t and
        s_{t+1} are captured per pair for the surface-invariance forward (posterior frame φ =
        `chain`, decoder target frame φ' = `chain_aug`). LABEL-FREE — text only."""
        src_texts: list[str] = []
        tgt_texts: list[str] = []
        chain_ids: list[int] = []
        # v6 §B: φ' (chain_aug) renderings of the same s_t / s_{t+1} (empty when lam off/absent).
        aug_src_texts: list[str] = []
        aug_tgt_texts: list[str] = []
        saw_aug = False

        with open(path) as f:
            for chain_no, line in enumerate(f):
                data = json.loads(line)
                chain: list[str] = data["chain"]
                chain_aug = data.get("chain_aug") if self._lam_augment_requested else None
                if chain_aug is not None and len(chain_aug) == len(chain):
                    saw_aug = True
                for i in range(len(chain) - 1):
                    src_texts.append(chain[i])
                    tgt_texts.append(chain[i + 1])
                    chain_ids.append(chain_no)
                    if chain_aug is not None and len(chain_aug) == len(chain):
                        aug_src_texts.append(chain_aug[i])
                        aug_tgt_texts.append(chain_aug[i + 1])

        self.lam_augment = bool(self._lam_augment_requested and saw_aug
                                and len(aug_src_texts) == len(src_texts))
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

        # v6 §B: per-pair φ-frame posterior tensors (== src/tgt, the `chain` frame) and the
        # φ'-frame decoder target tensors (the `chain_aug` rendering of s_{t+1}). The posterior
        # sees frame φ of BOTH inputs; the decoder reconstructs φ' of s_{t+1}. The posterior φ
        # frame == the existing src/tgt tensors (no new tensors), so we only build the φ'
        # decoder target (+ the φ s_t/s_{t+1} aliases for an explicit, swappable interface).
        if self.lam_augment:
            self._lam_post_src_ids = self._src_ids
            self._lam_post_src_pad = self._src_pad
            self._lam_post_tgt_ids = self._tgt_ids
            self._lam_post_tgt_pad = self._tgt_pad
            dec_ids = torch.zeros((n, T), dtype=torch.long)
            dec_pad = torch.ones((n, T), dtype=torch.bool)
            for i, atxt in enumerate(aug_tgt_texts):
                dec_ids[i], dec_pad[i] = _encode(atxt)
            self._lam_dec_tgt_ids = dec_ids
            self._lam_dec_tgt_pad = dec_pad

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
        # v6 §B: φ' (chain_aug) renderings of s0/s1/s2 (empty when lam off/absent).
        aug_s0: list[str] = []
        aug_s1: list[str] = []
        aug_s2: list[str] = []
        saw_aug = False

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
                chain_aug = data.get("chain_aug") if self._lam_augment_requested else None
                if chain_aug is not None and len(chain_aug) >= 3:
                    saw_aug = True
                    aug_s0.append(chain_aug[0])
                    aug_s1.append(chain_aug[1])
                    aug_s2.append(chain_aug[2])

        n = len(s0_texts)
        self.n_skipped = n_skipped
        self.lam_augment = bool(self._lam_augment_requested and saw_aug
                                and len(aug_s0) == n)
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

        # v6 §B: per-hop two-frame tensors. For each hop h (0: s0→s1, 1: s1→s2) the posterior
        # frame φ is the existing s{h}/s{h+1} `chain` tensors; the decoder target frame φ' is
        # the `chain_aug` rendering of the hop's target state (s1 for hop-1, s2 for hop-2).
        # The lists are indexed by hop in the trainer (`_lam_*_hop[h]`). LABEL-FREE.
        if self.lam_augment:
            # Posterior φ pairs per hop (aliases of the existing `chain` tensors).
            self._lam_post_src_ids_hop = [self._s0_ids, self._s1_ids]
            self._lam_post_src_pad_hop = [self._s0_pad, self._s1_pad]
            self._lam_post_tgt_ids_hop = [self._s1_ids, self._s2_ids]
            self._lam_post_tgt_pad_hop = [self._s1_pad, self._s2_pad]
            # Decoder φ' targets per hop (chain_aug renderings of the hop's target state).
            d1_ids = torch.zeros((n, T), dtype=torch.long)
            d1_pad = torch.ones((n, T), dtype=torch.bool)
            d2_ids = torch.zeros((n, T), dtype=torch.long)
            d2_pad = torch.ones((n, T), dtype=torch.bool)
            for i in range(n):
                d1_ids[i], d1_pad[i] = _encode(aug_s1[i])
                d2_ids[i], d2_pad[i] = _encode(aug_s2[i])
            self._lam_dec_tgt_ids_hop = [d1_ids, d2_ids]
            self._lam_dec_tgt_pad_hop = [d1_pad, d2_pad]

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
            if self.lam_augment:
                out["lam_dec_tgt_ids_h1"] = self._lam_dec_tgt_ids_hop[0][idx]
                out["lam_dec_tgt_ids_h2"] = self._lam_dec_tgt_ids_hop[1][idx]
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
        if self.lam_augment:
            out["lam_dec_tgt_ids"] = self._lam_dec_tgt_ids[idx]
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
            if self.lam_augment:
                out["lam_dec_tgt_ids_h1"] = self._lam_dec_tgt_ids_hop[0][indices]
                out["lam_dec_tgt_ids_h2"] = self._lam_dec_tgt_ids_hop[1][indices]
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
        if self.lam_augment:
            out["lam_dec_tgt_ids"] = self._lam_dec_tgt_ids[indices]
        return out

    # ------------------------------------------------------------------
    # v6 §B max_chains-cap slicers for the paired-frame tensors
    # ------------------------------------------------------------------

    def _slice_lam_pairs(self, cap: int) -> None:
        """Truncate the pairs-mode φ' decoder-target tensors to `cap` (max_chains path).

        The φ posterior tensors are aliases of self._src_ids/_tgt_ids (sliced by the trainer
        already), so re-point them after slicing the φ' targets."""
        self._lam_dec_tgt_ids = self._lam_dec_tgt_ids[:cap].contiguous()
        self._lam_dec_tgt_pad = self._lam_dec_tgt_pad[:cap].contiguous()
        self._lam_post_src_ids = self._src_ids
        self._lam_post_src_pad = self._src_pad
        self._lam_post_tgt_ids = self._tgt_ids
        self._lam_post_tgt_pad = self._tgt_pad

    def _slice_lam_triples(self, cap: int) -> None:
        """Truncate the triples-mode per-hop φ' decoder-target tensors to `cap`.

        The φ posterior tensors are aliases of self._s{0,1,2}_ids (sliced by the trainer
        already), so re-point the per-hop posterior aliases after slicing the φ' targets."""
        self._lam_dec_tgt_ids_hop = [t[:cap].contiguous() for t in self._lam_dec_tgt_ids_hop]
        self._lam_dec_tgt_pad_hop = [t[:cap].contiguous() for t in self._lam_dec_tgt_pad_hop]
        self._lam_post_src_ids_hop = [self._s0_ids, self._s1_ids]
        self._lam_post_src_pad_hop = [self._s0_pad, self._s1_pad]
        self._lam_post_tgt_ids_hop = [self._s1_ids, self._s2_ids]
        self._lam_post_tgt_pad_hop = [self._s1_pad, self._s2_pad]

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
