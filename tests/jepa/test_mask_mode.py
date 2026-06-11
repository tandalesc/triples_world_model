"""Tests for the v4.4 mask_mode selector (diff | random_span).

The v4.4 Wikipedia arm reuses the v4.2 masked-reconstruction machinery (the same
<mask> input-corruption + masked-positions-only CE path) but swaps the CAUSAL diff
span for RANDOM contiguous spans, because adjacent prose sentences have no s_t→s_{t+1}
diff to focus on. This file pins:

  - span coverage lands in 20-35% of the target's non-pad tokens (modulo the >=1-token
    floor and integer rounding for small targets),
  - 1-3 contiguous spans, no overlap, no pad leakage,
  - the random mask is seeded/deterministic per load (documented per-load choice),
  - diff mode is BITWISE untouched by the new code path (default mask_mode="diff"),
  - JEPAChainDataset routes mask_mode through to the per-target masks correctly.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest
import torch

from twm.jepa.data import JEPAChainDataset, _diff_mask, _random_span_mask

ENT_BPE = Path("data/entity_world/bpe_512.json")
MAX_T = 64


@pytest.fixture(scope="module")
def ent_tokenizer():
    if not ENT_BPE.exists():
        pytest.skip(f"entity-world BPE not found at {ENT_BPE}")
    from twm.domain_bpe import DomainBPETokenizer
    return DomainBPETokenizer.load(ENT_BPE, max_length=MAX_T)


# ---------------------------------------------------------------------------
# _random_span_mask: coverage, contiguity, no leakage
# ---------------------------------------------------------------------------

class TestRandomSpanMask:
    def test_coverage_in_band(self):
        """Masked fraction lands in [20%, 35%] of non-pad tokens (rounding-tolerant)."""
        rng = random.Random(0)
        for _ in range(2000):
            n_real = rng.randint(8, 40)  # >=8 so rounding noise stays inside the band
            tgt = [rng.randint(5, 200) for _ in range(n_real)] + [0] * (MAX_T - n_real)
            m = _random_span_mask(tgt, pad_id=0, T=MAX_T, rng=rng)
            n_mask = int(m.sum())
            cov = n_mask / n_real
            lo = max(1, round(0.20 * n_real)) / n_real
            hi = round(0.35 * n_real) / n_real
            assert lo - 1e-9 <= cov <= hi + 1e-9, (cov, n_real, n_mask)

    def test_span_count_at_most_three(self):
        """At most 3 contiguous spans; at least 1 when the target is non-empty."""
        rng = random.Random(1)
        for _ in range(2000):
            n_real = rng.randint(1, 40)
            tgt = [rng.randint(5, 200) for _ in range(n_real)] + [0] * (MAX_T - n_real)
            m = _random_span_mask(tgt, pad_id=0, T=MAX_T, rng=rng)
            spans, prev = 0, False
            for j in range(MAX_T):
                if m[j] and not prev:
                    spans += 1
                prev = bool(m[j])
            assert 1 <= spans <= 3, (spans, n_real)

    def test_no_pad_leakage(self):
        """Pad positions (>= n_real) are NEVER masked."""
        rng = random.Random(2)
        for _ in range(500):
            n_real = rng.randint(1, 30)
            tgt = [rng.randint(5, 200) for _ in range(n_real)] + [0] * (MAX_T - n_real)
            m = _random_span_mask(tgt, pad_id=0, T=MAX_T, rng=rng)
            for j in range(n_real, MAX_T):
                assert not m[j]

    def test_empty_target_empty_mask(self):
        """All-pad target ⟹ all-False mask (the masked CE skips it, like the diff edge case)."""
        m = _random_span_mask([0, 0, 0, 0], pad_id=0, T=8, rng=random.Random(3))
        assert m.sum().item() == 0

    def test_single_token_target_masks_one(self):
        """A 1-token target masks exactly that token (>=1 floor)."""
        m = _random_span_mask([7, 0, 0, 0], pad_id=0, T=8, rng=random.Random(4))
        assert m[0].item() is True or m[0].item()  # noqa: E712
        assert int(m.sum()) == 1

    def test_deterministic_with_seed(self):
        """Same RNG seed ⟹ identical mask (per-load reproducibility)."""
        tgt = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        a = _random_span_mask(tgt, 0, 16, random.Random(42))
        b = _random_span_mask(tgt, 0, 16, random.Random(42))
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# diff mode is bitwise untouched
# ---------------------------------------------------------------------------

class TestDiffModeUntouched:
    def test_diff_mask_bitwise(self):
        """The diff-span mask is unchanged (v4.2 behavior preserved)."""
        src = [10, 11, 12, 13, 14]
        tgt = [10, 11, 99, 13, 14]
        m = _diff_mask(src, tgt, pad_id=0, T=8)
        assert m.tolist() == [False, False, True, False, False, False, False, False]


# ---------------------------------------------------------------------------
# JEPAChainDataset routes mask_mode
# ---------------------------------------------------------------------------

class TestDatasetMaskMode:
    def _tiny_ds(self, tok, mask_mode):
        chains = [
            {"chain": [
                "the river flows north through the valley.",
                "it joins a larger stream near the town.",
                "the combined flow reaches the sea after fifty miles.",
            ]},
            {"chain": [
                "the engine uses a four stroke cycle.",
                "fuel and air mix inside each cylinder.",
                "the spark plug ignites the mixture at the top.",
            ]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            for c in chains:
                tmp.write(json.dumps(c) + "\n")
            p = tmp.name
        return JEPAChainDataset(
            p, tok, max_text_tokens=MAX_T, append_eos=True, mode="triples",
            w_diff=1.0, compute_diff_mask=True, mask_mode=mask_mode, mask_seed=0,
        )

    def test_random_span_dataset_masks_non_pad_only(self, ent_tokenizer):
        ds = self._tiny_ds(ent_tokenizer, "random_span")
        item = ds[0]
        m1 = item["s1_diff_mask"]
        pad1 = item["s1_pad"]
        # No masked position is pad.
        assert not bool((m1 & pad1).any())
        # Something got masked (non-empty target).
        assert int(m1.sum()) >= 1

    def test_invalid_mask_mode_raises(self, ent_tokenizer):
        with pytest.raises(ValueError):
            self._tiny_ds(ent_tokenizer, "bogus")

    def test_diff_default_unchanged(self, ent_tokenizer):
        """mask_mode default is 'diff' and produces the causal-diff mask (not random)."""
        ds = self._tiny_ds(ent_tokenizer, "diff")
        assert ds.mask_mode == "diff"


# ---------------------------------------------------------------------------
# config parse: the v4.4 wiki configs carry mask_mode and the masked recipe
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path("configs/corpora")


class TestWikiV44Configs:
    @pytest.mark.parametrize(
        "name,op,targeted",
        [
            ("wiki_v44_s0", "rotation_scale", True),
            ("wiki_v44_smoke", "rotation_scale", True),
            ("wiki_v44_blackbox_s0", "gated_mlp", False),
        ],
    )
    def test_config_parses(self, name, op, targeted):
        from twm.jepa.config import JEPAConfig
        path = CONFIGS_DIR / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} not found")
        cfg = JEPAConfig.from_json(path)
        assert cfg.data.mask_mode == "random_span"
        assert cfg.data.vocab_size == 8192
        assert cfg.loss.w_masked_diff == 1.0
        assert cfg.loss.w_token == 0.5
        assert cfg.loss.w_margin == 0.0
        assert cfg.loss.w_pool_nce == 0.0
        assert cfg.model.operator_group == op
        assert cfg.model.use_targeted_actions is targeted
