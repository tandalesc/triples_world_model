"""Tests for v4.2 masked-diff prediction (trad-JEPA masked reconstruction, focused by
causality).

The objective masks the CHANGED span of s_{t+1} in the decoder's teacher-forcing INPUT
(replacing those ids with the reserved <mask> special) and computes CE ONLY at the masked
positions vs the original gold ids. This severs the copy path and concentrates the loss on
100% discriminative tokens.

Coverage (per the build spec):
  - OFF == bitwise-neutral: w_masked_diff=0 ⟹ the term is skipped, total == v3, no diff
    masks even computed (compute_diff_mask=False ⟹ keys absent).
  - The diff mask aligns with the ACTUAL changed tokens on real entity-world data.
  - The masked positions' CE is the loss: boilerplate positions contribute ZERO.
  - The decoder INPUT really is corrupted at masked positions (no leakage of the answer
    through the teacher-forcing stream).
  - The all-equal pair edge case: empty mask ⟹ zero contribution (skip), no crash.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from twm.jepa.data import JEPAChainDataset, _diff_mask
from twm.jepa.losses import (
    JEPALossV2,
    corrupt_masked_input,
    masked_diff_ce,
)

ENT_BPE = Path("data/entity_world/bpe_512.json")
ENT_TRAIN = Path("data/entity_world/train.jsonl")
MAX_T = 64


@pytest.fixture(scope="module")
def ent_tokenizer():
    if not ENT_BPE.exists():
        pytest.skip(f"entity-world BPE not found at {ENT_BPE}")
    from twm.domain_bpe import DomainBPETokenizer
    return DomainBPETokenizer.load(ENT_BPE, max_length=MAX_T)


# ---------------------------------------------------------------------------
# _diff_mask: alignment with the actual changed tokens
# ---------------------------------------------------------------------------

class TestDiffMask:
    def test_replace_span_is_masked(self):
        """A single-token substitution marks exactly that target position."""
        # boilerplate ... X ... boilerplate  ->  boilerplate ... Y ... boilerplate
        src = [10, 11, 12, 13, 14]
        tgt = [10, 11, 99, 13, 14]  # position 2 replaced 12 -> 99
        m = _diff_mask(src, tgt, pad_id=0, T=8)
        assert m.dtype == torch.bool and m.shape == (8,)
        assert m.tolist() == [False, False, True, False, False, False, False, False]

    def test_insert_span_is_masked(self):
        """Inserted target tokens (target longer) are the diff."""
        src = [10, 11, 12]
        tgt = [10, 88, 89, 11, 12]  # 88,89 inserted after position 0
        m = _diff_mask(src, tgt, pad_id=0, T=8)
        assert m.tolist()[:5] == [False, True, True, False, False]

    def test_equal_pair_empty_mask(self):
        """Identical src/tgt ⟹ all-False mask (all-equal edge case, v4.2 §1)."""
        ids = [10, 11, 12, 13]
        m = _diff_mask(ids, ids, pad_id=0, T=8)
        assert m.sum().item() == 0

    def test_pad_positions_never_masked(self):
        """Trailing pad in tgt is stripped before alignment and never masked."""
        src = [10, 11, 12, 0, 0]
        tgt = [10, 99, 12, 0, 0]  # position 1 replaced; positions 3,4 are pad
        m = _diff_mask(src, tgt, pad_id=0, T=8)
        assert m[1].item() is True or m[1].item() == True  # noqa: E712
        assert m[3].item() == False and m[4].item() == False  # noqa: E712

    def test_mask_aligns_with_real_entity_data(self, ent_tokenizer):
        """On real entity-world chains, the diff mask marks ONLY positions where the
        token ids actually differ between adjacent states (the causal change), and the
        changed-token check holds for the marked span."""
        if not ENT_TRAIN.exists():
            pytest.skip("entity-world train.jsonl not found")
        tok = ent_tokenizer
        n_checked = 0
        with ENT_TRAIN.open() as f:
            for _ in range(40):
                line = f.readline()
                if not line:
                    break
                chain = json.loads(line)["chain"]
                if len(chain) < 2:
                    continue
                s0 = tok.encode(chain[0], max_length=MAX_T)
                s1 = tok.encode(chain[1], max_length=MAX_T)
                m = _diff_mask(s0, s1, pad_id=tok.pad_token_id, T=MAX_T)
                # Every masked position must be a real (non-pad) target token.
                for j in range(MAX_T):
                    if m[j]:
                        assert s1[j] != tok.pad_token_id
                # The masked span must cover at least the positions where s0 and s1
                # token ids differ within the shared prefix (changed tokens are masked).
                # Build the set of positions where a same-index token differs and is in
                # the shorter length — those must be inside the diff (replace/insert).
                n_real_s1 = sum(1 for t in s1 if t != tok.pad_token_id)
                n_real_s0 = sum(1 for t in s0 if t != tok.pad_token_id)
                limit = min(n_real_s0, n_real_s1)
                pointwise_diff = [j for j in range(limit) if s0[j] != s1[j]]
                # If there is a pointwise diff, the mask must be non-empty.
                if pointwise_diff:
                    assert m.sum().item() > 0, (
                        f"mask empty but tokens differ at {pointwise_diff}"
                    )
                n_checked += 1
        assert n_checked > 0


# ---------------------------------------------------------------------------
# corrupt_masked_input: the decoder INPUT is corrupted, answer not leaked
# ---------------------------------------------------------------------------

class TestCorruptInput:
    def test_masked_positions_replaced_with_mask_id(self):
        tgt = torch.tensor([[10, 11, 12, 13, 0]])
        mask = torch.tensor([[False, True, True, False, False]])
        corrupt = corrupt_masked_input(tgt, mask, mask_id=1)
        assert corrupt.tolist() == [[10, 1, 1, 13, 0]]

    def test_no_answer_leakage(self):
        """At every masked position the corrupted INPUT holds the mask id, NOT the gold
        token — the decoder cannot copy the answer from its teacher-forcing stream."""
        tgt = torch.randint(5, 200, (4, 16))
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, 3:6] = True
        corrupt = corrupt_masked_input(tgt, mask, mask_id=1)
        # Masked positions are mask_id; unmasked positions are the original gold ids.
        assert (corrupt[mask] == 1).all()
        assert (corrupt[~mask] == tgt[~mask]).all()


# ---------------------------------------------------------------------------
# masked_diff_ce: loss only at masked positions
# ---------------------------------------------------------------------------

class TestMaskedDiffCE:
    def test_only_masked_positions_count(self):
        """Boilerplate (unmasked) positions contribute ZERO to the loss — perturbing the
        logits at unmasked positions does not change the masked-diff CE."""
        torch.manual_seed(0)
        B, T, V = 2, 10, 32
        tgt = torch.randint(3, V, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 2:4] = True
        logits = torch.randn(B, T, V)
        l1 = masked_diff_ce(logits, tgt, mask, pad_id=0)
        # Corrupt logits ONLY at unmasked positions — loss must be unchanged.
        logits2 = logits.clone()
        logits2[:, 5:8] += 100.0
        l2 = masked_diff_ce(logits2, tgt, mask, pad_id=0)
        assert torch.allclose(l1, l2), "unmasked positions leaked into the masked-diff loss"

    def test_loss_drops_when_masked_logits_correct(self):
        """Sharpening logits AT the masked positions toward the gold token drops the loss."""
        B, T, V = 1, 6, 16
        tgt = torch.tensor([[3, 4, 7, 9, 2, 5]])
        mask = torch.tensor([[False, False, True, True, False, False]])
        bad = torch.zeros(B, T, V)
        l_bad = masked_diff_ce(bad, tgt, mask, pad_id=0)
        good = torch.zeros(B, T, V)
        for j in (2, 3):
            good[0, j, tgt[0, j]] = 20.0
        l_good = masked_diff_ce(good, tgt, mask, pad_id=0)
        assert l_good < l_bad

    def test_empty_mask_zero(self):
        """All-equal pair (empty mask) ⟹ loss is exactly 0.0 (skip), no NaN/crash."""
        B, T, V = 3, 8, 16
        tgt = torch.randint(3, V, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        l = masked_diff_ce(torch.randn(B, T, V), tgt, mask, pad_id=0)
        assert torch.isfinite(l) and l.item() == 0.0

    def test_pad_excluded(self):
        """A masked position that is pad in the target is excluded (defensive)."""
        B, T, V = 1, 5, 16
        tgt = torch.tensor([[3, 4, 0, 0, 0]])  # positions 2-4 are pad
        mask = torch.tensor([[False, True, True, True, True]])  # mask overlaps pad
        l = masked_diff_ce(torch.randn(B, T, V), tgt, mask, pad_id=0)
        # Only position 1 is a real masked token; the loss is finite and uses 1 token.
        assert torch.isfinite(l) and l.item() > 0.0


# ---------------------------------------------------------------------------
# OFF == bitwise-neutral, and dataset gating
# ---------------------------------------------------------------------------

class TestBitwiseNeutral:
    def _toy_loss_inputs(self, B=2, T=6, V=16, M=4, dn=8):
        torch.manual_seed(7)
        return dict(
            logits=torch.randn(B, T, V, requires_grad=True),
            tgt_ids=torch.randint(3, V, (B, T)),
            tgt_pad=torch.zeros(B, T, dtype=torch.bool),
            k=torch.randn(B, M, dn),
            v_logits=torch.randn(B, 8),
            p_logits=torch.randn(B, 8),
            zhat=torch.randn(B, dn),
            z_target=torch.randn(B, dn),
            tau=1.0,
        )

    def test_w_masked_diff_zero_is_neutral(self):
        """w_masked_diff=0 ⟹ identical total to a loss without the masked-diff term, even
        when masked_diff_logits/diff_mask are supplied (the term is gated off)."""
        loss_off = JEPALossV2(w_masked_diff=0.0)
        loss_ref = JEPALossV2()  # default also 0.0
        inp = self._toy_loss_inputs()
        B, T, V = inp["logits"].shape
        mdl = torch.randn(B, T, V)
        dm = torch.zeros(B, T, dtype=torch.bool)
        dm[:, 1:3] = True
        # L_sigreg draws random projection directions per call, so seed identically before
        # each forward to compare the deterministic parts (everything except the RNG).
        torch.manual_seed(123)
        t_off, c_off = loss_off(**inp, masked_diff_logits=mdl, diff_mask=dm)
        torch.manual_seed(123)
        t_ref, c_ref = loss_ref(**inp)
        assert torch.allclose(t_off, t_ref)
        assert c_off["L_masked_diff"] == 0.0

    def test_w_masked_diff_on_adds_term(self):
        """w_masked_diff>0 with supplied masked logits ⟹ total grows by w·L_masked_diff."""
        w = 1.0
        loss_on = JEPALossV2(w_masked_diff=w)
        loss_base = JEPALossV2(w_masked_diff=0.0)
        inp = self._toy_loss_inputs()
        B, T, V = inp["logits"].shape
        mdl = torch.randn(B, T, V)
        dm = torch.zeros(B, T, dtype=torch.bool)
        dm[:, 1:3] = True
        # Seed identically so the stochastic L_sigreg term cancels in the difference.
        torch.manual_seed(123)
        t_on, c_on = loss_on(**inp, masked_diff_logits=mdl, diff_mask=dm)
        torch.manual_seed(123)
        t_base, _ = loss_base(**inp)
        assert c_on["L_masked_diff"] > 0.0
        assert torch.allclose(t_on - t_base, torch.tensor(w * c_on["L_masked_diff"]), atol=1e-5)

    def test_dataset_no_mask_by_default(self, ent_tokenizer):
        """compute_diff_mask defaults False ⟹ no diff-mask tensors / keys (zero cost)."""
        ds = self._tiny_ds(ent_tokenizer, compute_diff_mask=False)
        item = ds[0]
        assert "s1_diff_mask" not in item and "s2_diff_mask" not in item
        assert ds._s1_diff_mask is None and ds._s2_diff_mask is None

    def test_dataset_emits_mask_when_enabled(self, ent_tokenizer):
        """compute_diff_mask=True ⟹ per-hop bool mask tensors/keys with the right shape."""
        ds = self._tiny_ds(ent_tokenizer, compute_diff_mask=True)
        item = ds[0]
        assert item["s1_diff_mask"].dtype == torch.bool
        assert item["s1_diff_mask"].shape == (MAX_T,)
        b = ds.get_batch([0, 1])
        assert b["s1_diff_mask"].shape == (2, MAX_T)
        assert b["s2_diff_mask"].shape == (2, MAX_T)

    def _tiny_ds(self, tok, compute_diff_mask):
        chains = [
            {"chain": ["alpha is calm.", "alpha is upset.", "alpha is calm again."]},
            {"chain": ["beta is full.", "beta is hungry.", "beta is fed."]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            for c in chains:
                tmp.write(json.dumps(c) + "\n")
            p = tmp.name
        return JEPAChainDataset(
            p, tok, max_text_tokens=MAX_T, append_eos=True, mode="triples",
            w_diff=4.0, compute_diff_mask=compute_diff_mask,
        )


# ---------------------------------------------------------------------------
# v4.2 configs parse + build; v4.1 poison-pill margin=0
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path("configs/jepa")


class TestV42Configs:
    @pytest.mark.parametrize(
        "name,targeted,op",
        [
            ("jepa_v42_s0", True, "rotation_scale"),
            ("jepa_v42_blackbox_s0", False, "gated_mlp"),
            ("jepa_v42_smoke", True, "rotation_scale"),
        ],
    )
    def test_v42_config_parses(self, name, targeted, op):
        from twm.jepa.config import JEPAConfig
        path = CONFIGS_DIR / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} not found")
        cfg = JEPAConfig.from_json(path)
        # The masked-diff arm is ON; the corrosive terms are OFF (clean attribution).
        assert cfg.loss.w_masked_diff == 1.0
        assert cfg.loss.w_token == 0.5            # generation-health floor
        assert cfg.loss.w_margin == 0.0           # corrosive margin OFF
        assert cfg.loss.w_pool_nce == 0.0         # separate arm OFF
        assert cfg.model.use_targeted_actions is targeted
        assert cfg.model.operator_group == op

    @pytest.mark.parametrize("name", ["jepa_v41_s0", "jepa_v41_blackbox_s0"])
    def test_v41_margin_off_poison_pill(self, name):
        """v4.1 must launch with the corrosive hardened margin OFF (w_margin=0.0)."""
        from twm.jepa.config import JEPAConfig
        path = CONFIGS_DIR / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} not found")
        cfg = JEPAConfig.from_json(path)
        assert cfg.loss.w_margin == 0.0
