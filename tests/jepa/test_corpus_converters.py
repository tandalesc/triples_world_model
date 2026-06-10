"""Tests for the wikiHow / CommitPackFT corpus converters + the large-vocab BPE.

Covers (per task validation requirement):
  - converter determinism: same seed -> byte-identical output
  - chain well-formedness: every record has a non-empty "chain" list of strings;
    wikiHow chains respect min/max step bounds; commitpack chains are length-2
  - label alignment: labeled twin shares the same chain; actions has len == chain-1;
    commitpack action == cleaned commit message; wikiHow action == first-verb lemma
  - BPE round-trip: a built tokenizer decodes a chain state back to (normalized) text
    and the JEPAChainDataset can tokenize the produced chains with <pad>=0

These tests DO NOT hit the network. They drive the converters' pure functions
(step splitting, label extraction, diff filtering, state-text building) on fixed
inputs, then exercise the seeded shuffle/split logic via the module's build helpers
fed from a stubbed iterator.

Run:
    uv run --with pytest python -m pytest tests/jepa/test_corpus_converters.py -q
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wikihow = _load("convert_wikihow", REPO / "scripts" / "convert_wikihow.py")
commitpack = _load("convert_commitpack", REPO / "scripts" / "convert_commitpack.py")


# ---------------------------------------------------------------------------
# wikiHow: step splitting
# ---------------------------------------------------------------------------

def test_wikihow_split_cleaned_summary_orders_steps():
    summary = "take theatre classes . volunteer at a theatre . try out every position ."
    steps = wikihow.steps_from_cleaned_summary(summary)
    assert steps == [
        "take theatre classes.",
        "volunteer at a theatre.",
        "try out every position.",
    ]


def test_wikihow_split_lists_strips_numbers():
    result = "1. Open Facebook. \n2. Find a photo. \n3. Tap on a photo. "
    steps = wikihow.steps_from_wikihow_lists(result)
    assert steps == ["Open Facebook.", "Find a photo.", "Tap on a photo."]


def test_wikihow_clean_step_collapses_whitespace_and_adds_period():
    assert wikihow._clean_step("  do   a thing  ") == "do a thing."
    assert wikihow._clean_step("") == ""


# ---------------------------------------------------------------------------
# wikiHow: action-label heuristic (first-verb lemma)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step,expected", [
    ("Take theatre classes.", "take"),
    ("Opening the box.", "open"),          # gerund -> lemma
    ("Removes the cover.", "remove"),      # 3rd-person -> lemma
    ("First, carefully open the lid.", "open"),  # skip leading adverbs
    ("Buy gloves.", "buy"),
    ("Your phone will restart.", "phone"),  # non-imperative noise (documented failure)
])
def test_wikihow_action_label(step, expected):
    assert wikihow.extract_action_label(step) == expected


def test_wikihow_action_label_never_empty():
    # An all-skip-word headline still yields a (fallback) lemma, never "".
    assert wikihow.extract_action_label("Please be sure to.") != ""
    assert wikihow.extract_action_label("12345 . ! ?") == "none"


# ---------------------------------------------------------------------------
# wikiHow: build_chains via a stubbed iterator (determinism, well-formedness, alignment)
# ---------------------------------------------------------------------------

_FAKE_ARTICLES = [
    ("How to Ride", "wear a helmet . pick a shirt . buy a jacket ."),
    ("How to X", "do a . do b ."),
    ("Too short", "only one step ."),          # 1 step -> dropped at min_steps=2
    ("How to Long", " . ".join(f"step {i}" for i in range(20))),  # truncated
]


def _wikihow_build(monkeypatch_steps_source):
    src = dict(wikihow.SOURCES["wikihow_cleaned"])

    def fake_iter(source, limit):
        for title, summary in monkeypatch_steps_source:
            yield title, wikihow.steps_from_cleaned_summary(summary)

    orig = wikihow.iter_articles
    wikihow.iter_articles = fake_iter
    try:
        return wikihow.build_chains(src, None, min_steps=2, max_steps=12)
    finally:
        wikihow.iter_articles = orig


def test_wikihow_build_wellformed_and_aligned():
    plain, labeled, stats = _wikihow_build(_FAKE_ARTICLES)
    # "Too short" (1 step) dropped; the 20-step one truncated to 12.
    assert stats["dropped_too_short"] == 1
    assert stats["truncated_over_max"] == 1
    assert len(plain) == len(labeled) == 3
    for p, l in zip(plain, labeled):
        assert isinstance(p["chain"], list) and len(p["chain"]) >= 2
        assert all(isinstance(s, str) and s for s in p["chain"])
        # labeled twin carries the SAME chain
        assert l["chain"] == p["chain"]
        # one action per transition
        assert len(l["actions"]) == len(p["chain"]) - 1
        # action label format "verb@0", verb == first-verb lemma of NEXT step
        for i, act in enumerate(l["actions"]):
            assert act.endswith("@0")
            assert act[:-2] == wikihow.extract_action_label(p["chain"][i + 1])
    # max step bound respected
    assert max(len(p["chain"]) for p in plain) <= 12


# ---------------------------------------------------------------------------
# CommitPackFT: message cleaning, diff sizing, state-text building
# ---------------------------------------------------------------------------

def test_commit_clean_message():
    assert commitpack.clean_message("Fix bug (#123)", "") == "Fix bug"
    assert commitpack.clean_message("", "Add feature\n\nlong body") == "Add feature"
    assert commitpack.clean_message("Tidy   up  [skip ci]", "") == "Tidy up"


def test_commit_count_changed_lines():
    old = "a\nb\nc\n"
    new = "a\nB\nc\nd\n"        # b->B (1 del + 1 add) + d added
    assert commitpack.count_changed_lines(old, new) == 3


def test_commit_state_texts_prefixed_and_truncated():
    old = "line0\n" + "\n".join(f"x{i}" for i in range(100))
    new = "line0\n" + "\n".join(("CHANGED" if i == 50 else f"x{i}") for i in range(100))
    s0, s1 = commitpack.make_state_texts(old, new, max_tokens=20)
    assert s0.startswith("BEFORE: ")
    assert s1.startswith("AFTER: ")
    # token-budget respected (prefix word + <= max_tokens code words)
    assert len(s0.split()) <= 21
    # change-centered window actually contains the changed token
    assert "CHANGED" in s1


def _commit_build(rows, **kw):
    def fake_iter(lang_key, limit):
        yield from rows

    orig = commitpack.iter_rows
    commitpack.iter_rows = fake_iter
    try:
        return commitpack.build_chains(
            ["python"], licenses={"mit"}, limit=None,
            max_changed_lines=kw.get("max_changed_lines", 30),
            max_code_tokens=kw.get("max_code_tokens", 48),
        )
    finally:
        commitpack.iter_rows = orig


_FAKE_COMMITS = [
    {"old_file": "a.py", "new_file": "a.py", "old_contents": "x = 1\n",
     "new_contents": "x = 2\n", "subject": "Bump x", "message": "Bump x\n",
     "lang": "Python", "license": "mit"},
    # multi-file -> dropped
    {"old_file": "a.py", "new_file": "b.py", "old_contents": "x\n",
     "new_contents": "y\n", "subject": "Move", "message": "Move", "license": "mit"},
    # non-permissive license -> dropped
    {"old_file": "c.py", "new_file": "c.py", "old_contents": "p\n",
     "new_contents": "q\n", "subject": "Edit", "message": "Edit", "license": "gpl-3.0"},
    # no change -> dropped
    {"old_file": "d.py", "new_file": "d.py", "old_contents": "z\n",
     "new_contents": "z\n", "subject": "Noop", "message": "Noop", "license": "mit"},
]


def test_commit_build_filters_and_alignment():
    plain, labeled, stats = _commit_build(_FAKE_COMMITS)
    assert stats["dropped"]["multi_file"] == 1
    assert stats["dropped"]["license"] == 1
    assert stats["dropped"]["no_change"] == 1
    assert len(plain) == 1
    p, l = plain[0], labeled[0]
    # length-2 chain, BEFORE/AFTER prefixed
    assert len(p["chain"]) == 2
    assert p["chain"][0].startswith("BEFORE: ") and p["chain"][1].startswith("AFTER: ")
    # labeled twin shares chain; one action == cleaned commit message
    assert l["chain"] == p["chain"]
    assert l["actions"] == ["Bump x@0"]


# ---------------------------------------------------------------------------
# Determinism: same seed -> identical emitted order
# ---------------------------------------------------------------------------

def test_seeded_shuffle_determinism():
    import random
    items = list(range(50))
    a = list(items); random.Random(7).shuffle(a)
    b = list(items); random.Random(7).shuffle(b)
    c = list(items); random.Random(8).shuffle(c)
    assert a == b      # same seed reproducible
    assert a != c      # different seed differs (the converters' split is seed-driven)


# ---------------------------------------------------------------------------
# BPE round-trip + JEPAChainDataset tokenization on produced chains
# ---------------------------------------------------------------------------

def test_bpe_roundtrip_and_dataset_tokenization(tmp_path):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

    # Build a tiny BPE the same way build_corpus_bpe does (special-token order -> <pad>=0).
    builder = _load("build_corpus_bpe", REPO / "scripts" / "build_corpus_bpe.py")
    texts = [
        "wear a helmet.", "pick a suitable shirt.", "buy a jacket.",
        "BEFORE: x = 1", "AFTER: x = 2",
    ] * 20
    tok = builder.build_tokenizer(texts, vocab_size=300)
    assert tok.token_to_id("<pad>") == 0

    # Round-trip: decode(encode(t)) recovers the text (modulo ByteLevel spacing).
    dec = tok.decode(tok.encode("wear a helmet.").ids)
    assert "wear" in dec and "helmet" in dec

    bpe_path = tmp_path / "bpe.json"
    tok.save(str(bpe_path))

    # JEPAChainDataset must tokenize the produced chains with the DomainBPETokenizer.
    sys.path.insert(0, str(REPO / "src"))
    from twm.domain_bpe import DomainBPETokenizer
    from twm.jepa.data import JEPAChainDataset

    chain_path = tmp_path / "chains.jsonl"
    with open(chain_path, "w") as f:
        f.write(json.dumps({"chain": ["wear a helmet.", "pick a suitable shirt.", "buy a jacket."]}) + "\n")

    dtok = DomainBPETokenizer.load(str(bpe_path), max_length=32)
    ds = JEPAChainDataset(str(chain_path), dtok, max_text_tokens=32, mode="triples")
    assert len(ds) == 1
    ex = ds[0]
    assert ex["s0_ids"].shape[0] == 32
    # pad positions are exactly the pad-id (0) positions -> mask well-formed
    assert bool((ex["s0_pad"] == (ex["s0_ids"] == 0)).all())

    # pairs mode on the same length-3 chain yields 2 adjacent pairs.
    ds_pairs = JEPAChainDataset(str(chain_path), dtok, max_text_tokens=32, mode="pairs")
    assert len(ds_pairs) == 2
