"""JEPA chain dataset over GLUCOSE adjacent (state_t, state_{t+1}) pairs.

Spec reference: jepa_operator_v1_design.md §6 and T4 row in §12.

Cross-state pairing is mandatory: online encoder sees state_t; EMA target encoder
sees state_{t+1}. Same text to both encoders degenerates into self-reconstruction
with a moving target — no cross-state JEPA signal (spec §6 FIX, Judge 2 D2 flaw).

Storage: contiguous CPU tensors, direct index slicing (no DataLoader).
"""

import json
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
    ) -> None:
        if mode not in ("pairs", "triples"):
            raise ValueError(f"mode must be 'pairs' or 'triples', got {mode!r}")
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        self.append_eos = append_eos
        self.mode = mode
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

        if mode == "pairs":
            self._build_pairs(path, _encode, T)
        else:
            self._build_triples(path, _encode, T)

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

        with open(path) as f:
            for chain_no, line in enumerate(f):
                data = json.loads(line)
                chain: list[str] = data["chain"]
                for i in range(len(chain) - 1):
                    src_texts.append(chain[i])
                    tgt_texts.append(chain[i + 1])
                    chain_ids.append(chain_no)

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
                self._tgt_diff_mask[i] = _diff_mask(
                    self._src_ids[i].tolist(), self._tgt_ids[i].tolist(), pad_id, T,
                )

        # Keep the raw texts for iter_text_pairs() (operator-fit pass-2 §7).
        self._src_texts: list[str] = src_texts
        self._tgt_texts: list[str] = tgt_texts
        # Per-pair originating chain id (design §8.2). len == len(dataset).
        self._chain_ids: list[int] = chain_ids

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
                self._s1_diff_mask[i] = _diff_mask(
                    self._s0_ids[i].tolist(), self._s1_ids[i].tolist(), pad_id, T,
                )
                self._s2_diff_mask[i] = _diff_mask(
                    self._s1_ids[i].tolist(), self._s2_ids[i].tolist(), pad_id, T,
                )

        self._s0_texts: list[str] = s0_texts
        self._s1_texts: list[str] = s1_texts
        self._s2_texts: list[str] = s2_texts
        # One id per chain (== example index's originating chain).
        self._chain_ids: list[int] = chain_ids

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
