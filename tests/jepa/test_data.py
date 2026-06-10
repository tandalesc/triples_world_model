"""Tests for JEPAChainDataset and the jepa_bpe_512.json tokenizer artifact.

Spec ref: jepa_operator_v1_design.md §6, T4 row in §12.

Coverage:
  - Cross-state pairing: src text != tgt text for every pair.
  - Output shapes and dtypes (T_text,) long / bool.
  - Padding masks: True at pad positions, False elsewhere.
  - Pair count: len(dataset) == 2 * n_chains for chains of length 3.
  - get_batch returns (B, T_text) tensors.
  - iter_text_pairs yields matching (src, tgt) tensors.
  - Round-trip detokenization sanity: decoded token sequence contains
    recognisable substrings of the original state text.
  - BPE artifact exists and has vocab_size == 512.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BPE_PATH = Path("data/glucose/jepa_bpe_512.json")
TRAIN_PATH = Path("data/glucose/chain_general_train.jsonl")
MAX_TEXT_TOKENS = 64


@pytest.fixture(scope="session")
def tokenizer():
    """Load the pre-built 512-vocab GLUCOSE BPE tokenizer."""
    if not BPE_PATH.exists():
        pytest.skip(f"BPE artifact not found at {BPE_PATH} — run scripts/build_jepa_bpe.py first")
    from src.twm.domain_bpe import DomainBPETokenizer
    return DomainBPETokenizer.load(BPE_PATH, max_length=MAX_TEXT_TOKENS)


@pytest.fixture(scope="session")
def tiny_jsonl(tmp_path_factory):
    """Write a tiny 5-chain JSONL fixture with deterministic content."""
    chains = [
        {"chain": ["State A is active.", "State B follows A.", "State C is final."]},
        {"chain": ["Someone_A runs fast.", "Someone_A arrives tired.", "Someone_A rests."]},
        {"chain": ["Something_A falls down.", "Something_A hits Something_B.", "Something_B breaks."]},
        {"chain": ["It is raining hard.", "The ground becomes wet.", "The grass grows green."]},
        {"chain": ["Someone_B calls Someone_A.", "Someone_A answers the phone.", "They talk for a while."]},
    ]
    p = tmp_path_factory.mktemp("data") / "tiny_chains.jsonl"
    with open(p, "w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")
    return p


@pytest.fixture(scope="session")
def tiny_dataset(tiny_jsonl, tokenizer):
    from src.twm.jepa.data import JEPAChainDataset
    return JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)


@pytest.fixture(scope="session")
def real_dataset(tokenizer):
    """Load a slice of the real GLUCOSE dataset (first 200 items only)."""
    if not TRAIN_PATH.exists():
        pytest.skip(f"Training data not found at {TRAIN_PATH}")
    from src.twm.jepa.data import JEPAChainDataset
    # Build a tiny JSONL with the first 100 chains to keep the fixture fast.
    with TRAIN_PATH.open() as f:
        lines = [f.readline() for _ in range(100)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    return JEPAChainDataset(tmp_path, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)


# ---------------------------------------------------------------------------
# BPE artifact tests
# ---------------------------------------------------------------------------

class TestBPEArtifact:
    def test_artifact_exists(self):
        """jepa_bpe_512.json must exist (produced by scripts/build_jepa_bpe.py)."""
        assert BPE_PATH.exists(), (
            f"BPE artifact missing at {BPE_PATH}. "
            "Run: uv run python scripts/build_jepa_bpe.py"
        )

    def test_vocab_size_512(self, tokenizer):
        assert tokenizer.vocab_size == 512, (
            f"Expected vocab_size=512, got {tokenizer.vocab_size}"
        )

    def test_special_token_ids(self, tokenizer):
        """PAD=0, MASK=1, UNK=2 per domain_bpe.py convention."""
        assert tokenizer.pad_token_id == 0, f"Expected pad_id=0, got {tokenizer.pad_token_id}"
        assert tokenizer.mask_token_id == 1, f"Expected mask_id=1, got {tokenizer.mask_token_id}"
        assert tokenizer.unk_token_id == 2, f"Expected unk_id=2, got {tokenizer.unk_token_id}"

    def test_encode_decode_roundtrip(self, tokenizer):
        """Decoded text should contain recognisable substrings of input.

        ByteLevel BPE with normalize=True lowercases input and the decoder
        produces raw byte-level tokens (with Ġ/space separators). We strip
        whitespace and check for key character sequences (case-insensitive).
        """
        text = "Someone_A puts Something_A on Something_B."
        ids = tokenizer.encode(text, max_length=64)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # Strip non-alpha noise and collapse spaces for comparison
        decoded_norm = decoded.replace("Ġ", "").replace(" ", "").lower()
        assert "someone" in decoded_norm, (
            f"'someone' not found in normalised decoded '{decoded_norm[:80]}...'"
        )
        assert "something" in decoded_norm, (
            f"'something' not found in normalised decoded '{decoded_norm[:80]}...'"
        )

    def test_encode_length_is_capped(self, tokenizer):
        """Encoding with max_length=64 must return exactly 64 ids."""
        long_text = "Someone_A " * 50  # deliberately long
        ids = tokenizer.encode(long_text, max_length=64)
        assert len(ids) == 64, f"Expected 64 ids, got {len(ids)}"

    def test_encode_short_is_padded(self, tokenizer):
        short_text = "OK."
        ids = tokenizer.encode(short_text, max_length=64)
        assert len(ids) == 64, f"Expected 64 ids (padded), got {len(ids)}"
        # All trailing tokens should be pad_id
        non_pad = sum(1 for i in ids if i != tokenizer.pad_token_id)
        assert non_pad > 0, "Expected at least one non-pad token"
        assert non_pad < 64, "Expected some padding for a short string"


# ---------------------------------------------------------------------------
# Dataset construction tests
# ---------------------------------------------------------------------------

class TestJEPAChainDatasetConstruction:
    def test_pair_count(self, tiny_dataset):
        """5 chains of length 3 each yield 2 pairs/chain = 10 pairs total."""
        assert len(tiny_dataset) == 10, f"Expected 10 pairs, got {len(tiny_dataset)}"

    def test_shapes(self, tiny_dataset):
        """Each item must have (T_text,) tensors for all four keys."""
        item = tiny_dataset[0]
        T = MAX_TEXT_TOKENS
        for key in ("src_ids", "src_pad", "tgt_ids", "tgt_pad"):
            assert key in item, f"Missing key '{key}'"
            assert item[key].shape == (T,), (
                f"key='{key}': expected shape ({T},), got {item[key].shape}"
            )

    def test_dtypes(self, tiny_dataset):
        """src_ids/tgt_ids must be long (int64); src_pad/tgt_pad must be bool."""
        item = tiny_dataset[0]
        assert item["src_ids"].dtype == torch.long, (
            f"src_ids dtype: expected torch.long, got {item['src_ids'].dtype}"
        )
        assert item["tgt_ids"].dtype == torch.long, (
            f"tgt_ids dtype: expected torch.long, got {item['tgt_ids'].dtype}"
        )
        assert item["src_pad"].dtype == torch.bool, (
            f"src_pad dtype: expected torch.bool, got {item['src_pad'].dtype}"
        )
        assert item["tgt_pad"].dtype == torch.bool, (
            f"tgt_pad dtype: expected torch.bool, got {item['tgt_pad'].dtype}"
        )


# ---------------------------------------------------------------------------
# Cross-state pairing tests (mandatory per spec §6)
# ---------------------------------------------------------------------------

class TestCrossStatePairing:
    def test_src_tgt_differ_for_all_pairs(self, tiny_dataset):
        """CRITICAL: src_ids must differ from tgt_ids for every pair.

        Same text to both encoders degenerates into self-reconstruction.
        Cross-state pairing is mandatory (spec §6, Judge 2 D2 flaw fix).
        """
        n_same = 0
        for i in range(len(tiny_dataset)):
            item = tiny_dataset[i]
            if torch.equal(item["src_ids"], item["tgt_ids"]):
                n_same += 1
        assert n_same == 0, (
            f"{n_same}/{len(tiny_dataset)} pairs have identical src_ids and tgt_ids. "
            "This breaks cross-state JEPA pairing — online and EMA encoders would see "
            "the same text, degenerating into self-reconstruction."
        )

    def test_src_is_t_not_t_plus_1(self, tiny_jsonl, tokenizer):
        """Verify src encodes state_t and tgt encodes state_{t+1} (not swapped)."""
        from src.twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)

        # Pair index 0 should be (chain[0][0], chain[0][1])
        item = ds[0]
        src_decoded = tokenizer.decode(item["src_ids"], skip_special_tokens=True)
        tgt_decoded = tokenizer.decode(item["tgt_ids"], skip_special_tokens=True)
        # The first chain is ["State A is active.", "State B follows A.", ...]
        # src should encode "State A is active." and tgt "State B follows A."
        # ByteLevel BPE may insert spaces; normalise by stripping all whitespace.
        src_clean = src_decoded.replace("Ġ", "").replace(" ", "").lower()
        tgt_clean = tgt_decoded.replace("Ġ", "").replace(" ", "").lower()
        assert "statea" in src_clean or "active" in src_clean, (
            f"src for pair 0 should encode chain[0][0], got: '{src_clean}'"
        )
        assert "stateb" in tgt_clean or "follows" in tgt_clean, (
            f"tgt for pair 0 should encode chain[0][1], got: '{tgt_clean}'"
        )

    def test_pair_ordering_within_chain(self, tiny_jsonl, tokenizer):
        """Pairs from the same chain should be sequentially ordered.

        Chain [t0, t1, t2] yields pairs at indices 2*i and 2*i+1.
        """
        from src.twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
        # For chain index k, pair k*2 is (t0, t1), pair k*2+1 is (t1, t2).
        # The tgt of pair 2*k should match the src of pair 2*k+1 (they share state t1).
        for k in range(5):  # 5 chains
            pair_0 = ds[k * 2]
            pair_1 = ds[k * 2 + 1]
            assert torch.equal(pair_0["tgt_ids"], pair_1["src_ids"]), (
                f"Chain {k}: tgt of first pair should equal src of second pair "
                "(they share the same intermediate state). Pairs are out of order."
            )


# ---------------------------------------------------------------------------
# Padding mask tests
# ---------------------------------------------------------------------------

class TestPaddingMasks:
    def test_pad_mask_true_at_pad_positions(self, tiny_dataset):
        """src_pad[i] is True exactly where src_ids[i] == pad_token_id."""
        pad_id = tiny_dataset.tokenizer.pad_token_id
        for i in range(min(len(tiny_dataset), 20)):
            item = tiny_dataset[i]
            expected_src_pad = (item["src_ids"] == pad_id)
            expected_tgt_pad = (item["tgt_ids"] == pad_id)
            assert torch.equal(item["src_pad"], expected_src_pad), (
                f"pair {i}: src_pad does not match (src_ids == pad_id)"
            )
            assert torch.equal(item["tgt_pad"], expected_tgt_pad), (
                f"pair {i}: tgt_pad does not match (tgt_ids == pad_id)"
            )

    def test_nonempty_states_have_some_nonpad_tokens(self, tiny_dataset):
        """Non-empty state texts should produce at least one non-padding token."""
        for i in range(len(tiny_dataset)):
            item = tiny_dataset[i]
            n_src_nonpad = (~item["src_pad"]).sum().item()
            n_tgt_nonpad = (~item["tgt_pad"]).sum().item()
            assert n_src_nonpad > 0, f"pair {i}: src has no non-pad tokens"
            assert n_tgt_nonpad > 0, f"pair {i}: tgt has no non-pad tokens"

    def test_pad_is_contiguous_suffix(self, tiny_dataset):
        """Padding tokens should form a contiguous suffix (no gaps in the middle).

        The tokenizer pads to max_length, so once we hit a pad token the remainder
        should all be pad tokens (they are at the end, never in the middle).
        """
        pad_id = tiny_dataset.tokenizer.pad_token_id
        T = MAX_TEXT_TOKENS
        for i in range(min(len(tiny_dataset), 20)):
            item = tiny_dataset[i]
            for key_ids in ("src_ids", "tgt_ids"):
                ids = item[key_ids].tolist()
                pad_seen = False
                for tok in ids:
                    if tok == pad_id:
                        pad_seen = True
                    elif pad_seen:
                        # Non-pad token after a pad token → not a clean suffix
                        pytest.fail(
                            f"pair {i}, key={key_ids}: non-pad token after pad token "
                            f"(padding not a contiguous suffix). ids={ids}"
                        )


# ---------------------------------------------------------------------------
# Batch interface test
# ---------------------------------------------------------------------------

class TestGetBatch:
    def test_get_batch_shape(self, tiny_dataset):
        """get_batch should return (B, T_text) tensors."""
        indices = [0, 1, 2, 3]
        batch = tiny_dataset.get_batch(indices)
        B, T = len(indices), MAX_TEXT_TOKENS
        for key in ("src_ids", "src_pad", "tgt_ids", "tgt_pad"):
            assert batch[key].shape == (B, T), (
                f"get_batch key='{key}': expected ({B},{T}), got {batch[key].shape}"
            )

    def test_get_batch_single_index(self, tiny_dataset):
        """get_batch([i]) should return (1, T) tensors matching __getitem__(i)."""
        i = 3
        batch = tiny_dataset.get_batch([i])
        item = tiny_dataset[i]
        assert torch.equal(batch["src_ids"][0], item["src_ids"])
        assert torch.equal(batch["tgt_ids"][0], item["tgt_ids"])


# ---------------------------------------------------------------------------
# iter_text_pairs test
# ---------------------------------------------------------------------------

class TestIterTextPairs:
    def test_iter_count_matches_len(self, tiny_dataset):
        pairs = list(tiny_dataset.iter_text_pairs())
        assert len(pairs) == len(tiny_dataset), (
            f"iter_text_pairs yielded {len(pairs)} pairs, expected {len(tiny_dataset)}"
        )

    def test_iter_yields_tensor_pairs(self, tiny_dataset):
        for i, (src, tgt) in enumerate(tiny_dataset.iter_text_pairs()):
            assert isinstance(src, torch.Tensor), f"pair {i}: src is not a Tensor"
            assert isinstance(tgt, torch.Tensor), f"pair {i}: tgt is not a Tensor"
            assert src.shape == (MAX_TEXT_TOKENS,), (
                f"pair {i}: src shape {src.shape}, expected ({MAX_TEXT_TOKENS},)"
            )
            assert tgt.shape == (MAX_TEXT_TOKENS,), (
                f"pair {i}: tgt shape {tgt.shape}, expected ({MAX_TEXT_TOKENS},)"
            )
            if i >= 20:
                break  # spot-check only

    def test_iter_matches_getitem(self, tiny_dataset):
        """iter_text_pairs() tensors should match __getitem__ src_ids / tgt_ids."""
        for i, (src, tgt) in enumerate(tiny_dataset.iter_text_pairs()):
            item = tiny_dataset[i]
            assert torch.equal(src, item["src_ids"]), f"pair {i}: iter src != getitem src_ids"
            assert torch.equal(tgt, item["tgt_ids"]), f"pair {i}: iter tgt != getitem tgt_ids"

    def test_iter_cross_state(self, tiny_dataset):
        """iter_text_pairs must yield different tensors for src and tgt."""
        n_same = sum(
            1 for src, tgt in tiny_dataset.iter_text_pairs()
            if torch.equal(src, tgt)
        )
        assert n_same == 0, (
            f"{n_same} iter_text_pairs items have identical src and tgt tensors."
        )


# ---------------------------------------------------------------------------
# Detokenization round-trip sanity (real GLUCOSE data)
# ---------------------------------------------------------------------------

class TestDetokenizationSanity:
    def test_roundtrip_preserves_placeholders(self, real_dataset):
        """Decoded tokens for GLUCOSE states should contain GLUCOSE placeholder patterns.

        GLUCOSE uses placeholders like Someone_A, Something_B, Somewhere_A.
        These should survive BPE encode → decode. ByteLevel BPE with
        normalize=True lowercases input; we check case-insensitively after
        stripping byte-level whitespace markers.
        """
        tokenizer = real_dataset.tokenizer
        found_placeholder = 0
        n_checked = min(len(real_dataset), 50)
        for i in range(n_checked):
            item = real_dataset[i]
            for key in ("src_ids", "tgt_ids"):
                decoded = tokenizer.decode(item[key], skip_special_tokens=True)
                # Strip byte-level markers and all spaces, then lowercase
                decoded_clean = decoded.replace("Ġ", "").replace("Ċ", "").replace(" ", "").lower()
                # Check for GLUCOSE placeholder stems (lowercased + no underscore due to normalize)
                if any(p in decoded_clean for p in ("someone", "something", "somewhere", "feel", "put")):
                    found_placeholder += 1
                    break  # one match per pair is enough

        # At least 40% of checked pairs should decode to recognisable GLUCOSE text
        assert found_placeholder >= n_checked * 0.4, (
            f"Only {found_placeholder}/{n_checked} decoded states contained GLUCOSE "
            f"placeholder patterns. BPE coverage may be too low."
        )


# ---------------------------------------------------------------------------
# Contiguous tensor layout test
# ---------------------------------------------------------------------------

class TestContiguousLayout:
    def test_tensors_are_contiguous(self, tiny_dataset):
        """Tensors must be contiguous for efficient direct slicing (repo convention)."""
        assert tiny_dataset._src_ids.is_contiguous(), "_src_ids is not contiguous"
        assert tiny_dataset._src_pad.is_contiguous(), "_src_pad is not contiguous"
        assert tiny_dataset._tgt_ids.is_contiguous(), "_tgt_ids is not contiguous"
        assert tiny_dataset._tgt_pad.is_contiguous(), "_tgt_pad is not contiguous"

    def test_tensors_on_cpu(self, tiny_dataset):
        """All stored tensors must live on CPU (trainer moves batches to device)."""
        assert tiny_dataset._src_ids.device.type == "cpu"
        assert tiny_dataset._tgt_ids.device.type == "cpu"


# ---------------------------------------------------------------------------
# chain_ids — same-chain hard-negative MRR wiring (v2.1, design §8.2)
# ---------------------------------------------------------------------------

class TestChainIds:
    def test_chain_ids_length_matches_dataset(self, tiny_dataset):
        """chain_ids is a public list with one entry per pair (len == len(dataset))."""
        assert hasattr(tiny_dataset, "chain_ids")
        assert isinstance(tiny_dataset.chain_ids, list)
        assert len(tiny_dataset.chain_ids) == len(tiny_dataset)

    def test_n_distinct_chains(self, tiny_dataset):
        """5 chains in the tiny fixture -> exactly 5 distinct chain ids."""
        assert len(set(tiny_dataset.chain_ids)) == 5

    def test_adjacent_pairs_share_chain_id(self, tiny_dataset):
        """A length-3 chain yields 2 adjacent pairs that MUST share one chain id.

        Without this, the diagnostics same-chain pool degenerates to `cid = idx` and
        easy_minus_hard_mrr passes vacuously (the integrator-flagged gap)."""
        cids = tiny_dataset.chain_ids
        # pairs are emitted in chain order: (chain0 p0, chain0 p1, chain1 p0, ...)
        for chain_no in range(5):
            assert cids[2 * chain_no] == cids[2 * chain_no + 1] == chain_no

    def test_chain_ids_align_after_cap(self, tiny_jsonl, tokenizer):
        """The max_chains cap path (train_jepa_v2.py) slices _chain_ids alongside the
        tensors; after truncation, chain_ids stays aligned (len == len(dataset))."""
        from src.twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
        cap = 3 * 2  # keep 3 chains -> 6 pairs (mirror trainer: max_chains*2)
        ds._src_ids = ds._src_ids[:cap].contiguous()
        ds._src_pad = ds._src_pad[:cap].contiguous()
        ds._tgt_ids = ds._tgt_ids[:cap].contiguous()
        ds._tgt_pad = ds._tgt_pad[:cap].contiguous()
        ds._src_texts = ds._src_texts[:cap]
        ds._tgt_texts = ds._tgt_texts[:cap]
        ds._chain_ids = ds._chain_ids[:cap]
        assert len(ds) == cap
        assert len(ds.chain_ids) == cap
        assert set(ds.chain_ids) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Diff-weighted CE token weights (v4 §2.1): _diff_weights + *_diff_w tensors
# ---------------------------------------------------------------------------

ENTITY_BPE_PATH = Path("data/glucose/jepa_bpe_512.json")
ENTITY_TRAIN_PATH = Path("data/entity_world/train.jsonl")


class TestDiffWeightsUnit:
    """Direct unit tests of _diff_weights (no tokenizer — synthetic id sequences)."""

    def test_replace_marks_target_diff(self):
        from src.twm.jepa.data import _diff_weights
        # src and tgt share [10, 11] then differ at the last token (replace 12 -> 99).
        src = [10, 11, 12]
        tgt = [10, 11, 99]
        w = _diff_weights(src, tgt, w_diff=4.0, pad_id=0, T=8)
        assert w.shape == (8,)
        assert w[0].item() == 1.0 and w[1].item() == 1.0   # shared boilerplate
        assert w[2].item() == 4.0                          # replaced token is the diff
        assert w[3:].sum().item() == 0.0                   # pad positions zeroed

    def test_insert_marks_target_diff(self):
        from src.twm.jepa.data import _diff_weights
        # tgt has an extra token (44) inserted at the front (an added action clause).
        src = [10, 11, 12]
        tgt = [44, 10, 11, 12]
        w = _diff_weights(src, tgt, w_diff=3.0, pad_id=0, T=8)
        assert w[0].item() == 3.0                          # inserted token is the diff
        assert w[1].item() == 1.0 and w[2].item() == 1.0 and w[3].item() == 1.0

    def test_equal_sequences_all_boilerplate(self):
        from src.twm.jepa.data import _diff_weights
        src = [10, 11, 12, 13]
        tgt = [10, 11, 12, 13]
        w = _diff_weights(src, tgt, w_diff=4.0, pad_id=0, T=8)
        # No diff ⟹ every non-pad target weight is 1.0 even with w_diff=4.
        assert w[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert w[4:].sum().item() == 0.0

    def test_w_diff_one_is_all_ones_over_nonpad(self):
        from src.twm.jepa.data import _diff_weights
        src = [10, 11, 12]
        tgt = [10, 77, 88]  # two diffs, but w_diff=1.0 ⟹ all ones
        w = _diff_weights(src, tgt, w_diff=1.0, pad_id=0, T=6)
        assert w[:3].tolist() == [1.0, 1.0, 1.0]
        assert w[3:].sum().item() == 0.0

    def test_pad_positions_zero(self):
        from src.twm.jepa.data import _diff_weights
        src = [10, 11, 0, 0]    # trailing pad
        tgt = [10, 99, 0, 0]
        w = _diff_weights(src, tgt, w_diff=4.0, pad_id=0, T=4)
        # Token 1 replaced -> 4.0; the two pad positions -> 0.0.
        assert w[0].item() == 1.0
        assert w[1].item() == 4.0
        assert w[2].item() == 0.0 and w[3].item() == 0.0

    def test_delete_does_not_index_out_of_range(self):
        from src.twm.jepa.data import _diff_weights
        # src has an extra token that is deleted in tgt — no target position to weight.
        src = [10, 11, 12, 13]
        tgt = [10, 12, 13]
        w = _diff_weights(src, tgt, w_diff=4.0, pad_id=0, T=8)
        assert w.shape == (8,)
        # 11 deleted; tgt is [10, 12, 13]. SequenceMatcher sees a delete (no tgt weight).
        # The remaining shared tokens stay boilerplate.
        assert torch.isfinite(w).all()


@pytest.fixture(scope="session")
def entity_tokenizer():
    if not ENTITY_BPE_PATH.exists():
        pytest.skip(f"BPE artifact not found at {ENTITY_BPE_PATH}")
    from src.twm.domain_bpe import DomainBPETokenizer
    return DomainBPETokenizer.load(ENTITY_BPE_PATH, max_length=MAX_TEXT_TOKENS)


@pytest.fixture(scope="session")
def entity_triples_dataset(entity_tokenizer):
    """A small slice of the real entity-world train data in triples mode."""
    if not ENTITY_TRAIN_PATH.exists():
        pytest.skip(f"Entity data not found at {ENTITY_TRAIN_PATH}")
    from src.twm.jepa.data import JEPAChainDataset
    with ENTITY_TRAIN_PATH.open() as f:
        lines = [f.readline() for _ in range(80)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    return JEPAChainDataset(
        tmp_path, entity_tokenizer, max_text_tokens=MAX_TEXT_TOKENS,
        mode="triples", w_diff=4.0,
    )


class TestDiffWeightTensorsTriples:
    def test_keys_present(self, entity_triples_dataset):
        item = entity_triples_dataset[0]
        assert "s1_diff_w" in item and "s2_diff_w" in item
        assert item["s1_diff_w"].shape == (MAX_TEXT_TOKENS,)
        assert item["s2_diff_w"].shape == (MAX_TEXT_TOKENS,)
        assert item["s1_diff_w"].dtype == torch.float32

    def test_get_batch_keys(self, entity_triples_dataset):
        batch = entity_triples_dataset.get_batch([0, 1, 2])
        assert batch["s1_diff_w"].shape == (3, MAX_TEXT_TOKENS)
        assert batch["s2_diff_w"].shape == (3, MAX_TEXT_TOKENS)

    def test_pad_positions_zero(self, entity_triples_dataset):
        """diff weight is 0 exactly where the target is pad (real entity data)."""
        for i in range(min(len(entity_triples_dataset), 20)):
            item = entity_triples_dataset[i]
            s1_pad = item["s1_pad"]
            assert (item["s1_diff_w"][s1_pad] == 0.0).all(), (
                f"example {i}: s1_diff_w must be 0 at pad positions"
            )
            s2_pad = item["s2_pad"]
            assert (item["s2_diff_w"][s2_pad] == 0.0).all()

    def test_diff_weights_match_actual_changed_tokens(self, entity_triples_dataset):
        """On real entity data the diff weight must mark exactly the tokens that differ
        between consecutive states (vs the prior state), and ONLY those.

        We recompute the expected diff independently via SequenceMatcher on the stored
        id tensors and assert the dataset's stored weights agree position-by-position."""
        from difflib import SequenceMatcher
        ds = entity_triples_dataset
        pad_id = ds.tokenizer.pad_token_id
        n_checked = 0
        n_with_diff = 0
        for i in range(min(len(ds), 30)):
            item = ds[i]
            for src_key, tgt_key, w_key in (
                ("s0_ids", "s1_ids", "s1_diff_w"),
                ("s1_ids", "s2_ids", "s2_diff_w"),
            ):
                src = item[src_key].tolist()
                tgt = item[tgt_key].tolist()
                # strip pad like _diff_weights does
                ns = len(src)
                while ns > 0 and src[ns - 1] == pad_id:
                    ns -= 1
                nt = len(tgt)
                while nt > 0 and tgt[nt - 1] == pad_id:
                    nt -= 1
                expected = torch.zeros(MAX_TEXT_TOKENS)
                expected[:nt] = 1.0
                sm = SequenceMatcher(a=src[:ns], b=tgt[:nt], autojunk=False)
                for tag, _a, _b, j1, j2 in sm.get_opcodes():
                    if tag in ("replace", "insert"):
                        for j in range(j1, j2):
                            expected[j] = 4.0
                # pad positions are 0.0
                expected[nt:] = 0.0
                assert torch.equal(item[w_key], expected), (
                    f"example {i} {w_key}: diff weights do not match the actual token diff"
                )
                n_checked += 1
                if (expected == 4.0).any():
                    n_with_diff += 1
        assert n_checked > 0
        # Entity-world consecutive states differ (an action changes some clause), so at
        # least some examples must carry a real diff (else the test is vacuous).
        assert n_with_diff > 0, "expected some real diffs in entity-world consecutive states"

    def test_w_diff_one_yields_all_ones(self, entity_tokenizer):
        """With w_diff=1.0 (default) the stored weights are all-ones over non-pad ⟹
        bitwise-v3 uniform CE when consumed by token_ce."""
        if not ENTITY_TRAIN_PATH.exists():
            pytest.skip("entity data missing")
        from src.twm.jepa.data import JEPAChainDataset
        with ENTITY_TRAIN_PATH.open() as f:
            lines = [f.readline() for _ in range(20)]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
            tmp.writelines(lines)
            p = tmp.name
        ds = JEPAChainDataset(
            p, entity_tokenizer, max_text_tokens=MAX_TEXT_TOKENS,
            mode="triples", w_diff=1.0,
        )
        for i in range(len(ds)):
            item = ds[i]
            for w_key, pad_key in (("s1_diff_w", "s1_pad"), ("s2_diff_w", "s2_pad")):
                w = item[w_key]
                pad = item[pad_key]
                assert (w[~pad] == 1.0).all(), f"{w_key} must be all-ones over non-pad with w_diff=1"
                assert (w[pad] == 0.0).all()


class TestDiffWeightTensorsPairs:
    @pytest.fixture(scope="class")
    def pairs_dataset(self, tiny_jsonl, tokenizer):
        from src.twm.jepa.data import JEPAChainDataset
        return JEPAChainDataset(
            tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS,
            mode="pairs", w_diff=4.0,
        )

    def test_key_present(self, pairs_dataset):
        item = pairs_dataset[0]
        assert "tgt_diff_w" in item
        assert item["tgt_diff_w"].shape == (MAX_TEXT_TOKENS,)
        assert item["tgt_diff_w"].dtype == torch.float32

    def test_get_batch_key(self, pairs_dataset):
        batch = pairs_dataset.get_batch([0, 1])
        assert batch["tgt_diff_w"].shape == (2, MAX_TEXT_TOKENS)

    def test_pad_positions_zero(self, pairs_dataset):
        for i in range(len(pairs_dataset)):
            item = pairs_dataset[i]
            assert (item["tgt_diff_w"][item["tgt_pad"]] == 0.0).all()

    def test_default_w_diff_is_one(self, tiny_jsonl, tokenizer):
        """Default constructor (no w_diff) ⟹ w_diff=1.0 ⟹ all-ones over non-pad."""
        from src.twm.jepa.data import JEPAChainDataset
        ds = JEPAChainDataset(tiny_jsonl, tokenizer, max_text_tokens=MAX_TEXT_TOKENS)
        assert ds.w_diff == 1.0
        item = ds[0]
        assert (item["tgt_diff_w"][~item["tgt_pad"]] == 1.0).all()
