"""Tests for the GH Archive issue-lifecycle converter (the "world-model chops" tier).

Pattern-style twin of tests/jepa/test_corpus_converters.py. Covers (per task validation):
  - state rendering: deterministic, order-stable (labels sorted), mechanical-state-only
  - action extraction: GH event -> coarse action token; comment -> "commented"
  - chain build via a stubbed grouped-events dict: well-formedness, label alignment
    (actions[i] is the action of the event that produced state_{i+1}; len == chain-1),
    types slot, min/max-event bounds
  - determinism report: a mechanical state machine yields HIGH determinism-given-action
  - BPE round-trip + JEPAChainDataset tokenization on produced chains (pairs mode)

No network: the converter's pure functions (render_state, extract_action, build_chains,
determinism_report) are driven on fixed inputs / a stubbed grouped dict.

Run:
    uv run --with pytest python -m pytest tests/jepa/test_chops_converter.py -q
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


gha = _load("convert_gharchive", REPO / "scripts" / "convert_gharchive.py")


# ---------------------------------------------------------------------------
# state rendering
# ---------------------------------------------------------------------------

def test_render_state_basic_open_unassigned():
    iss = {"number": 7, "state": "open", "labels": [], "assignee": None}
    assert gha.render_state(iss) == "issue 7 is open. it is unassigned."


def test_render_state_labels_sorted_order_stable():
    # Label order must NOT carry signal: two orderings render identically.
    a = {"number": 1, "state": "open", "labels": [{"name": "Bug"}, {"name": "help wanted"}]}
    b = {"number": 1, "state": "open", "labels": [{"name": "help wanted"}, {"name": "Bug"}]}
    assert gha.render_state(a) == gha.render_state(b)
    s = gha.render_state(a)
    assert "it has label bug." in s and "it has label help wanted." in s
    # bug sorts before help wanted (stable)
    assert s.index("bug") < s.index("help wanted")


def test_render_state_assignee_milestone_locked():
    iss = {
        "number": 42, "state": "closed",
        "labels": [{"name": "p1"}],
        "assignees": [{"login": "Alice"}],
        "milestone": {"title": "v2.0"},
        "locked": True,
    }
    s = gha.render_state(iss)
    assert s.startswith("issue 42 is closed.")
    assert "it is assigned to alice." in s
    assert "it has milestone v2.0." in s
    assert "it is locked." in s
    assert "unassigned" not in s


# ---------------------------------------------------------------------------
# action extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("etype,payload,expected", [
    ("IssuesEvent", {"action": "opened"}, "opened"),
    ("IssuesEvent", {"action": "closed"}, "closed"),
    ("IssuesEvent", {"action": "labeled"}, "labeled"),
    ("IssuesEvent", {"action": "assigned"}, "assigned"),
    ("IssueCommentEvent", {"action": "created"}, "commented"),
    ("PullRequestEvent", {"action": "synchronize"}, "edited"),
    ("PullRequestEvent", {"action": "reopened"}, "reopened"),
])
def test_extract_action(etype, payload, expected):
    assert gha.extract_action(etype, payload) == expected


# ---------------------------------------------------------------------------
# build_chains via a stubbed grouped-events dict
# ---------------------------------------------------------------------------

def _grouped():
    """A stubbed {(repo,num,kind): [event_record,...]} dict (the collect_issue_events output)."""
    def ev(ts, action, state, labels):
        return {
            "ts": ts, "type": "IssuesEvent", "action": action,
            "issue": {"number": 5, "state": state,
                      "labels": [{"name": l} for l in labels], "assignee": None},
        }
    return {
        # a real 3-event lifecycle: open -> labeled bug -> closed (out of ts order to test sort)
        ("o/r", 5, "issue"): [
            ev("2024-01-15T12:02:00Z", "labeled", "open", ["bug"]),
            ev("2024-01-15T12:00:00Z", "opened", "open", []),
            ev("2024-01-15T12:05:00Z", "closed", "closed", ["bug"]),
        ],
        # singleton -> dropped at min_events=2
        ("o/r", 6, "issue"): [ev("2024-01-15T12:00:00Z", "opened", "open", [])],
    }


def test_build_chains_wellformed_and_aligned():
    plain, labeled, stats = gha.build_chains(_grouped(), min_events=2, max_events=12)
    assert stats["dropped_too_short"] == 1          # the singleton
    assert len(plain) == len(labeled) == 1
    p, l = plain[0], labeled[0]
    # chronological (ts-sorted): open -> labeled -> closed
    assert p["chain"][0] == "issue 5 is open. it is unassigned."
    assert "it has label bug." in p["chain"][1]
    assert p["chain"][2].startswith("issue 5 is closed.")
    # labeled twin shares the chain; one action per transition
    assert l["chain"] == p["chain"]
    assert len(l["actions"]) == len(p["chain"]) - 1
    # action[i] is the action of the event that produced state_{i+1}
    assert l["actions"] == ["labeled@0", "closed@0"]
    assert l["types"] == ["issue"]


def test_build_chains_truncates_to_max_events():
    def ev(i):
        return {"ts": f"2024-01-15T12:{i:02d}:00Z", "type": "IssuesEvent",
                "action": "edited", "issue": {"number": 9, "state": "open", "labels": []}}
    grouped = {("o/r", 9, "issue"): [ev(i) for i in range(20)]}
    plain, labeled, stats = gha.build_chains(grouped, min_events=2, max_events=12)
    assert stats["truncated_over_max"] == 1
    assert len(plain[0]["chain"]) == 12
    assert len(labeled[0]["actions"]) == 11


# ---------------------------------------------------------------------------
# determinism report: a mechanical state machine => HIGH determinism-given-action
# ---------------------------------------------------------------------------

def test_determinism_report_high_for_mechanical():
    # 10 identical "open --closed--> closed" transitions: given (state, closed) the
    # next state is unique => determinism 1.0; action vocab = {closed}.
    s0 = "issue 1 is open. it is unassigned."
    s1 = "issue 1 is closed. it is unassigned."
    labeled = [{"chain": [s0, s1], "actions": ["closed@0"]} for _ in range(10)]
    rep = gha.determinism_report(labeled)
    assert rep["action_vocab_size"] == 1
    assert rep["det_given_action_mean"] == 1.0
    # only one action ever follows this state -> zero conditional entropy
    assert rep["H_action_given_state_mean_bits"] == 0.0


# ---------------------------------------------------------------------------
# BPE round-trip + JEPAChainDataset tokenization on produced chains (pairs mode)
# ---------------------------------------------------------------------------

def test_bpe_roundtrip_and_dataset_tokenization(tmp_path):
    pytest.importorskip("tokenizers")
    builder = _load("build_corpus_bpe", REPO / "scripts" / "build_corpus_bpe.py")
    texts = [
        "issue 1 is open. it is unassigned.",
        "issue 1 is closed. it has label bug. it is unassigned.",
        "issue 2 is open. it is assigned to alice.",
    ] * 20
    tok = builder.build_tokenizer(texts, vocab_size=300)
    assert tok.token_to_id("<pad>") == 0
    dec = tok.decode(tok.encode("issue 1 is open. it is unassigned.").ids)
    assert "issue" in dec and "open" in dec

    bpe_path = tmp_path / "bpe.json"
    tok.save(str(bpe_path))

    sys.path.insert(0, str(REPO / "src"))
    from twm.domain_bpe import DomainBPETokenizer
    from twm.jepa.data import JEPAChainDataset

    chain_path = tmp_path / "chains.jsonl"
    with open(chain_path, "w") as f:
        f.write(json.dumps({"chain": [
            "issue 1 is open. it is unassigned.",
            "issue 1 is closed. it is unassigned.",
        ]}) + "\n")

    dtok = DomainBPETokenizer.load(str(bpe_path), max_length=32)
    # GH chains are mostly length-2 -> pairs mode (one transition per chain).
    ds = JEPAChainDataset(str(chain_path), dtok, max_text_tokens=32, mode="pairs")
    assert len(ds) == 1
    ex = ds[0]
    assert ex["src_ids"].shape[0] == 32
    # pad mask well-formed: pad positions are exactly the pad-id (0) positions
    assert bool((ex["src_pad"] == (ex["src_ids"] == 0)).all())
    # cross-state: src (open) and tgt (closed) differ
    assert not bool((ex["src_ids"] == ex["tgt_ids"]).all())
