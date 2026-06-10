"""Tests for v3 triple-mode dataset + unroll training loop (Task B).

Design ref: research/jepa_v3_design.md §2.1 (triple mode), §1.5 (chain-contiguous
batching), §6 Task B.

Coverage:
  - Triple alignment: s0/s1/s2 tensors recover the chain's three states in order, and
    the hop-1 (s0->s1) / hop-2 (s1->s2) decomposition matches the equivalent pairs-mode
    pairs token-for-token.
  - chain_ids consistency: one id per chain in triple mode; aligned with examples.
  - mode default ("pairs") reproduces v2 pair behavior bitwise (the n_unroll_steps=1 ==
    current-behavior gate — triple mode is the >1-hop path; pairs mode is the 1-hop path).
  - Length-<3 chains are skipped in triple mode and the count is reported.
  - Chain-contiguous batching co-locates a chain's sibling examples in one batch.
  - A tiny end-to-end smoke runs the triple-mode unroll training loop without error.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

BPE_PATH = REPO / "data/glucose/jepa_bpe_512.json"
TRAIN_PATH = REPO / "data/glucose/chain_general_train.jsonl"
MAX_TEXT_TOKENS = 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tokenizer():
    if not BPE_PATH.exists():
        pytest.skip(f"BPE artifact not found at {BPE_PATH}")
    from twm.domain_bpe import DomainBPETokenizer
    return DomainBPETokenizer.load(BPE_PATH, max_length=MAX_TEXT_TOKENS)


# Deterministic 5-chain fixture, all length 3, distinct content per state.
_CHAINS = [
    ["State A is active.", "State B follows A.", "State C is final."],
    ["Someone_A runs fast.", "Someone_A arrives tired.", "Someone_A rests."],
    ["Something_A falls down.", "Something_A hits Something_B.", "Something_B breaks."],
    ["It is raining hard.", "The ground becomes wet.", "The grass grows green."],
    ["Someone_B calls Someone_A.", "Someone_A answers the phone.", "They talk a while."],
]


@pytest.fixture(scope="session")
def tiny_jsonl(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "tiny_triples.jsonl"
    with open(p, "w") as f:
        for c in _CHAINS:
            f.write(json.dumps({"chain": c}) + "\n")
    return p


@pytest.fixture(scope="session")
def mixed_len_jsonl(tmp_path_factory):
    """Fixture mixing length-3 chains with shorter ones to exercise the skip path."""
    chains = [
        {"chain": ["a one.", "a two.", "a three."]},   # len 3 -> kept
        {"chain": ["b one.", "b two."]},               # len 2 -> skipped
        {"chain": ["c one."]},                          # len 1 -> skipped
        {"chain": ["d one.", "d two.", "d three."]},   # len 3 -> kept
    ]
    p = tmp_path_factory.mktemp("data") / "mixed_len.jsonl"
    with open(p, "w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")
    return p


@pytest.fixture(scope="session")
def pairs_ds(tiny_jsonl, tokenizer):
    from twm.jepa.data import JEPAChainDataset
    return JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="pairs")


@pytest.fixture(scope="session")
def triples_ds(tiny_jsonl, tokenizer):
    from twm.jepa.data import JEPAChainDataset
    return JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="triples")


# ---------------------------------------------------------------------------
# Triple alignment
# ---------------------------------------------------------------------------

class TestTripleAlignment:
    def test_one_example_per_chain(self, triples_ds):
        """Triple mode emits exactly ONE example per length-3 chain (design §2.1)."""
        assert len(triples_ds) == len(_CHAINS)

    def test_getitem_keys_and_shapes(self, triples_ds):
        item = triples_ds[0]
        for key in ("s0_ids", "s0_pad", "s1_ids", "s1_pad", "s2_ids", "s2_pad"):
            assert key in item, f"missing key {key}"
        assert "chain_id" in item
        for k in ("s0_ids", "s1_ids", "s2_ids"):
            assert item[k].shape == (MAX_TEXT_TOKENS,)
            assert item[k].dtype == torch.long
        for k in ("s0_pad", "s1_pad", "s2_pad"):
            assert item[k].shape == (MAX_TEXT_TOKENS,)
            assert item[k].dtype == torch.bool

    def test_states_are_distinct(self, triples_ds):
        """s0, s1, s2 encode DIFFERENT states (no degenerate self-recon)."""
        for i in range(len(triples_ds)):
            it = triples_ds[i]
            assert not torch.equal(it["s0_ids"], it["s1_ids"])
            assert not torch.equal(it["s1_ids"], it["s2_ids"])

    def test_triple_matches_pairs_tokenization(self, triples_ds, pairs_ds):
        """The triple's (s0,s1) is exactly the pairs-mode hop-1 pair, and (s1,s2) the
        hop-2 pair, token-for-token. This is the triple<->pairs alignment guarantee:
        chain c -> pairs[2c]=(t0,t1), pairs[2c+1]=(t1,t2)."""
        for c in range(len(_CHAINS)):
            tri = triples_ds[c]
            p0 = pairs_ds[2 * c]      # (t0, t1)
            p1 = pairs_ds[2 * c + 1]  # (t1, t2)
            assert torch.equal(tri["s0_ids"], p0["src_ids"])
            assert torch.equal(tri["s1_ids"], p0["tgt_ids"])
            assert torch.equal(tri["s1_ids"], p1["src_ids"])
            assert torch.equal(tri["s2_ids"], p1["tgt_ids"])
            # pad masks must align too
            assert torch.equal(tri["s0_pad"], p0["src_pad"])
            assert torch.equal(tri["s2_pad"], p1["tgt_pad"])

    def test_s1_is_shared_pivot(self, triples_ds, pairs_ds):
        """s1 is simultaneously hop-1's target and hop-2's source — one tensor, no drift."""
        for c in range(len(_CHAINS)):
            tri = triples_ds[c]
            # within the same example s1 is a single tensor used at both hops
            assert torch.equal(tri["s1_ids"], pairs_ds[2 * c]["tgt_ids"])
            assert torch.equal(tri["s1_ids"], pairs_ds[2 * c + 1]["src_ids"])

    def test_eos_appended_when_requested(self, tiny_jsonl, tokenizer):
        """append_eos path is reused in triple mode (design §2.1: reuse _insert_eos)."""
        from twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(
            tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS,
            mode="triples", append_eos=True,
        )
        it = ds[0]
        # eos id = 4 must appear in every state's token stream
        for k in ("s0_ids", "s1_ids", "s2_ids"):
            assert (it[k] == 4).any(), f"{k} missing <eos>=4 under append_eos"


# ---------------------------------------------------------------------------
# chain_ids consistency
# ---------------------------------------------------------------------------

class TestChainIds:
    def test_one_id_per_chain(self, triples_ds):
        """Triple mode: one chain_id per example, len == #examples (design §2.1)."""
        cids = triples_ds.chain_ids
        assert len(cids) == len(triples_ds)
        # all chains are length 3 and distinct -> ids are the 0..N-1 chain indices
        assert cids == list(range(len(_CHAINS)))

    def test_getitem_chain_id_matches_property(self, triples_ds):
        for i in range(len(triples_ds)):
            assert triples_ds[i]["chain_id"] == triples_ds.chain_ids[i]

    def test_get_batch_chain_ids_aligned(self, triples_ds):
        idx = [0, 2, 4]
        batch = triples_ds.get_batch(idx)
        assert "chain_id" in batch
        assert batch["chain_id"].tolist() == [triples_ds.chain_ids[i] for i in idx]
        assert batch["s0_ids"].shape == (3, MAX_TEXT_TOKENS)

    def test_pairs_mode_chain_ids_unchanged(self, pairs_ds):
        """Pairs mode keeps the v2.1 semantics: 2 pairs per chain SHARE an id."""
        cids = pairs_ds.chain_ids
        assert len(cids) == 2 * len(_CHAINS)
        # adjacent pairs of chain c both carry id c
        for c in range(len(_CHAINS)):
            assert cids[2 * c] == c
            assert cids[2 * c + 1] == c


# ---------------------------------------------------------------------------
# Behavior preservation: default mode == pairs == v2 (the n_unroll_steps=1 gate)
# ---------------------------------------------------------------------------

class TestPairsBehaviorPreserved:
    def test_default_mode_is_pairs(self, tiny_jsonl, tokenizer):
        from twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
        assert ds.mode == "pairs"

    def test_default_construction_bitwise_identical(self, tiny_jsonl, tokenizer):
        """Default construction (no mode arg) reproduces the explicit pairs-mode dataset
        tensor-for-tensor — the v2 behavior-preservation gate (design §0/§2.1)."""
        from twm.jepa.data import JEPAChainDataset
        ds_default = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
        ds_pairs = JEPAChainDataset(
            tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="pairs"
        )
        assert len(ds_default) == len(ds_pairs)
        for a in ("_src_ids", "_src_pad", "_tgt_ids", "_tgt_pad"):
            assert torch.equal(getattr(ds_default, a), getattr(ds_pairs, a))
        assert ds_default.chain_ids == ds_pairs.chain_ids

    def test_pairs_mode_has_no_triple_attrs(self, pairs_ds):
        """Pairs mode does NOT populate the triple tensors (design §2.1 additivity)."""
        assert not hasattr(pairs_ds, "_s0_ids")

    def test_triples_mode_has_no_pair_attrs(self, triples_ds):
        """Triple mode does NOT populate the pair tensors (design §2.1 additivity)."""
        assert not hasattr(triples_ds, "_src_ids")

    def test_iter_text_pairs_raises_in_triple_mode(self, triples_ds):
        with pytest.raises(RuntimeError):
            next(triples_ds.iter_text_pairs())

    def test_invalid_mode_raises(self, tiny_jsonl, tokenizer):
        from twm.jepa.data import JEPAChainDataset
        with pytest.raises(ValueError):
            JEPAChainDataset(tiny_jsonl, tokenizer, mode="quadruples")


# ---------------------------------------------------------------------------
# Length-<3 chains are skipped (triple mode)
# ---------------------------------------------------------------------------

class TestShortChainSkip:
    def test_short_chains_skipped(self, mixed_len_jsonl, tokenizer):
        from twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(
            mixed_len_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="triples"
        )
        # 2 kept (the two length-3 chains), 2 skipped
        assert len(ds) == 2
        assert ds.n_skipped == 2

    def test_kept_chains_have_original_ids(self, mixed_len_jsonl, tokenizer):
        """Skipped chains do not shift the kept chains' originating chain ids."""
        from twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(
            mixed_len_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="triples"
        )
        # chain 0 (kept) and chain 3 (kept); 1,2 skipped
        assert ds.chain_ids == [0, 3]

    def test_full_glucose_no_skips(self, tokenizer):
        """GLUCOSE chain_general is uniformly length 3 — triple mode drops nothing."""
        if not TRAIN_PATH.exists():
            pytest.skip("GLUCOSE train data not present")
        from twm.jepa.data import JEPAChainDataset
        with TRAIN_PATH.open() as f:
            lines = [f.readline() for _ in range(200)]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp.writelines(lines)
            path = tmp.name
        ds = JEPAChainDataset(path, tokenizer, max_text_tokens=MAX_TEXT_TOKENS, mode="triples")
        assert ds.n_skipped == 0
        assert len(ds) == 200


# ---------------------------------------------------------------------------
# Chain-contiguous batching (design §1.5)
# ---------------------------------------------------------------------------

def _load_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_jepa_v2_mod", str(REPO / "scripts" / "train_jepa_v2.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestChainContiguousBatching:
    def test_siblings_adjacent(self):
        """chain_contiguous_perm groups a chain's sibling examples adjacently so they
        co-occur in one batch (the in-batch hard negative for InfoNCE, design §1.5)."""
        mod = _load_train_script()
        # 4 chains, each with 2 pairs -> chain_ids = [0,0,1,1,2,2,3,3]
        chain_ids = [0, 0, 1, 1, 2, 2, 3, 3]
        torch.manual_seed(0)
        perm = mod.chain_contiguous_perm(chain_ids)
        assert sorted(perm.tolist()) == list(range(8))  # a true permutation
        permuted_cids = [chain_ids[i] for i in perm.tolist()]
        # adjacent pairs in the permutation must share a chain id (siblings together)
        for s in range(0, 8, 2):
            assert permuted_cids[s] == permuted_cids[s + 1], (
                f"siblings split at {s}: {permuted_cids}"
            )

    def test_is_permutation_of_indices(self):
        mod = _load_train_script()
        chain_ids = [0, 0, 1, 1, 2, 2]
        perm = mod.chain_contiguous_perm(chain_ids)
        assert sorted(perm.tolist()) == [0, 1, 2, 3, 4, 5]

    def test_triple_mode_one_example_per_chain(self):
        """In triple mode each chain has ONE example; the perm is still a valid shuffle."""
        mod = _load_train_script()
        chain_ids = [0, 1, 2, 3, 4]  # one example per chain
        perm = mod.chain_contiguous_perm(chain_ids)
        assert sorted(perm.tolist()) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# End-to-end unroll training smoke (design §6 Task B)
# ---------------------------------------------------------------------------

class TestUnrollSmoke:
    def test_triple_unroll_step_runs(self, triples_ds, tokenizer):
        """A single triple-mode unroll step (forward_unroll -> per-hop weighted loss)
        runs end to end and produces a finite, backpropagating loss."""
        # Skip gracefully if Task C's model/loss are not yet importable.
        try:
            from twm.jepa.config import JEPAConfig
        except Exception as e:  # pragma: no cover
            pytest.skip(f"config import failed: {e}")
        mod = _load_train_script()

        cfg = JEPAConfig.from_json(str(REPO / "configs/jepa/jepa_v3_smoke.json"))
        device = torch.device("cpu")

        try:
            from twm.jepa.model import build_jepa_model_v2
            token_emb = mod.build_token_emb(cfg.data.vocab_size, cfg.model.d_model)
            model = build_jepa_model_v2(cfg, token_emb).to(device)
            loss_fn = mod.build_loss_v2(cfg, model.operator).to(device)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"model/loss build unavailable (Task C mid-build?): {e}")

        if not hasattr(model, "forward_unroll"):
            pytest.skip("model.forward_unroll not present yet (Task C mid-build)")

        model.train()
        idx = torch.arange(min(4, len(triples_ds)))
        hop_weights = list(cfg.loss.unroll.hop_weights)
        loss, comps = mod._unroll_step(
            model, loss_fn, triples_ds, idx, device, tau=1.0, hop_weights=hop_weights
        )
        assert torch.isfinite(loss), f"non-finite unroll loss: {loss}"
        loss.backward()  # gradients flow without error
        # per-hop CE is logged
        assert "L_token_h1" in comps and "L_token_h2" in comps

    def test_pair_step_matches_inline_v2_when_w_nce_zero(self, pairs_ds):
        """_pair_step with w_nce==0 reproduces the inline v2 step output: a finite loss
        with the v2 component keys and NO InfoNCE contribution (the behavior gate)."""
        try:
            from twm.jepa.config import JEPAConfig
            from twm.jepa.model import build_jepa_model_v2
        except Exception as e:  # pragma: no cover
            pytest.skip(f"model import unavailable: {e}")
        mod = _load_train_script()
        cfg = JEPAConfig.from_json(str(REPO / "configs/jepa/jepa_nano_v21_smoke.json"))
        device = torch.device("cpu")
        try:
            token_emb = mod.build_token_emb(cfg.data.vocab_size, cfg.model.d_model)
            model = build_jepa_model_v2(cfg, token_emb).to(device)
            loss_fn = mod.build_loss_v2(cfg, model.operator).to(device)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"model build unavailable: {e}")
        model.train()
        idx = torch.arange(min(4, len(pairs_ds)))
        loss, comps = mod._pair_step(model, loss_fn, pairs_ds, idx, device, tau=1.0, w_nce=0.0)
        assert torch.isfinite(loss)
        assert comps.get("L_nce", 0.0) == 0.0  # InfoNCE off -> no contribution
        assert "L_token" in comps
