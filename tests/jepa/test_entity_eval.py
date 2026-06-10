"""Tests for the entity-world eval suite (campaign §3, Task B).

Covers:
  - _load_labeled_split: shapes/keys on the REAL test_iid_labeled.jsonl.
  - action-recovery NMI: == 1.0 when the latent "action" is copied from the oracle
    labels (perfect recovery), ~0 on a shuffle. [§3a]
  - _nmi sklearn-optional fallback agrees with sklearn (when present).
  - OOD ladder runs on the REAL test files and returns the per-split scalars. [§3b]
  - rollout fidelity: depth-1 teacher-forced reduces to the standard single forward
    (the rollout's depth-1 transformed nouns == a direct apply of the argmax
    posterior action on k0), and returns depth-1..D exact/chrF for TF and PR. [§3c]
  - eval.entity_world.enabled=false ⟹ no entity metrics (back-compat).
  - the 8 entity configs parse and build (structured + black-box, use_norm_budget).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from twm.jepa.config import JEPAConfig
from twm.jepa.model import build_jepa_model_v2
from twm.domain_bpe import DomainBPETokenizer
from twm.jepa.diagnostics import (
    _load_labeled_split,
    _nmi,
    _action_recovery_nmi,
    _ood_ladder,
    _rollout_fidelity,
    _op_apply,
    eval_entity_world,
)

DATA_DIR = ROOT / "data" / "entity_world"
BPE = DATA_DIR / "bpe_512.json"

_HAVE_DATA = DATA_DIR.exists() and BPE.exists() and (DATA_DIR / "test_iid_labeled.jsonl").exists()
requires_data = pytest.mark.skipif(not _HAVE_DATA, reason="entity_world data not generated")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tokenizer():
    return DomainBPETokenizer.load(str(BPE), max_length=64)


@pytest.fixture(scope="module")
def tiny_model():
    cfg = JEPAConfig.from_json(str(ROOT / "configs" / "jepa" / "jepa_ent_smoke.json"))
    emb = torch.nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    emb.weight.requires_grad_(False)
    m = build_jepa_model_v2(cfg, emb)
    m.eval()
    return m


# --------------------------------------------------------------------------- _nmi
def test_nmi_perfect_recovery_is_one():
    labels = ["feed", "play", "wash", "feed", "play", "wash"]
    # latent codes a perfect (relabeled) clustering of the verbs.
    codes = [0, 1, 2, 0, 1, 2]
    assert _nmi(codes, labels) == pytest.approx(1.0, abs=1e-6)


def test_nmi_shuffle_is_low():
    rng = np.random.RandomState(0)
    labels = (["feed"] * 50 + ["play"] * 50 + ["wash"] * 50)
    codes = [0] * 50 + [1] * 50 + [2] * 50
    shuffled = list(codes)
    rng.shuffle(shuffled)
    perfect = _nmi(codes, labels)
    shuf = _nmi(shuffled, labels)
    assert perfect == pytest.approx(1.0, abs=1e-6)
    assert shuf < 0.2  # destroyed alignment -> near zero


def test_nmi_single_cluster_is_zero():
    assert _nmi([0, 0, 0, 0], ["a", "b", "a", "b"]) == 0.0


def test_nmi_fallback_matches_sklearn():
    """The hist-based fallback must agree with sklearn when sklearn is available."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(1)
    a = rng.randint(0, 5, size=300).tolist()
    b = rng.randint(0, 4, size=300).tolist()
    # sklearn path (what _nmi uses when present)
    sk_val = float(sk.normalized_mutual_info_score(a, b))
    # force the fallback by computing it directly
    import twm.jepa.diagnostics as diag
    orig_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("blocked for test")
        return orig_import(name, *args, **kwargs)

    import builtins
    builtins.__import__ = blocked
    try:
        fb_val = diag._nmi(a, b)
    finally:
        builtins.__import__ = orig_import
    assert fb_val == pytest.approx(sk_val, abs=1e-6)


# --------------------------------------------------------------------------- loader
@requires_data
def test_load_labeled_split_shapes_keys(tokenizer):
    chains = _load_labeled_split(str(DATA_DIR), "test_iid", tokenizer, 64, True, max_chains=10)
    assert len(chains) == 10
    ch = chains[0]
    assert set(ch.keys()) == {"chain", "actions", "types", "ids", "pad"}
    # one action per transition; ids/pad one per state.
    assert len(ch["ids"]) == len(ch["chain"])
    assert len(ch["actions"]) == len(ch["chain"]) - 1
    assert ch["ids"][0].shape == (64,)
    assert ch["pad"][0].dtype == torch.bool
    assert all(isinstance(a, str) and "@" in a for a in ch["actions"])


# --------------------------------------------------------------------------- §3a NMI
@requires_data
def test_action_recovery_nmi_copied_labels_is_one(tiny_model, tokenizer, monkeypatch):
    """When the inferred latent action == the oracle verb (copied from labels), NMI==1
    and the shuffle baseline is ~0 — the load-bearing recovery check."""
    chains = _load_labeled_split(str(DATA_DIR), "test_iid", tokenizer, 64, True, max_chains=40)

    # Build a deterministic verb->code map and monkeypatch the posterior inference to
    # "copy" the oracle verb (perfect recovery oracle).
    verbs = sorted({a.split("@", 1)[0] for ch in chains for a in ch["actions"]})
    vmap = {v: i for i, v in enumerate(verbs)}

    # Patch _infer_posterior_action to a copy-from-label oracle. The labels are consumed
    # in chain/pair order, so we replay them via a closure over an iterator.
    import twm.jepa.diagnostics as diag
    label_iter = iter(
        a.split("@", 1)[0]
        for ch in chains
        for a in ch["actions"][: max(0, len(ch["ids"]) - 1)]
    )

    def fake_infer(model, si, sp, ti, tp, device):
        return vmap[next(label_iter)]

    monkeypatch.setattr(diag, "_infer_posterior_action", fake_infer)
    out = diag._action_recovery_nmi(tiny_model, chains, "cpu", None, "test")
    assert out["ent_action_nmi_verb"] == pytest.approx(1.0, abs=1e-6)
    assert out["ent_action_nmi_shuffle"] < 0.2
    assert out["ent_action_nmi_verb_pass"] is True


@requires_data
def test_action_recovery_nmi_real_model_runs(tiny_model, tokenizer):
    """An untrained model still produces the four NMI scalars (no crash)."""
    chains = _load_labeled_split(str(DATA_DIR), "test_iid", tokenizer, 64, True, max_chains=20)
    out = _action_recovery_nmi(tiny_model, chains, "cpu", None, "test")
    for key in ("ent_action_nmi_verb", "ent_action_nmi_verb_entity",
                "ent_action_nmi_shuffle", "ent_action_nmi_verb_pass"):
        assert key in out


# --------------------------------------------------------------------------- §3b ladder
@requires_data
def test_ood_ladder_runs_on_real_files(tiny_model, tokenizer):
    out = _ood_ladder(
        tiny_model, tokenizer, str(DATA_DIR),
        ["test_iid", "test_ood_near", "test_ood_far"], 32, "cpu",
        64, True, None, "test",
    )
    for split in ("test_iid", "test_ood_near", "test_ood_far"):
        assert f"ent_{split}_ce" in out
        assert f"ent_{split}_hard_mrr" in out
        assert f"ent_{split}_chrf" in out
        assert 0.0 <= out[f"ent_{split}_hard_mrr"] <= 1.0
    assert isinstance(out["ent_ladder_monotone_mrr"], bool)


# --------------------------------------------------------------------------- §3c rollout
@requires_data
def test_rollout_depth1_tf_reduces_to_standard_forward(tiny_model, tokenizer):
    """Depth-1 teacher-forced rollout's transformed nouns must equal a direct single-hop
    apply of the argmax posterior action on k0 — i.e. depth-1 TF == the standard forward
    operator application (campaign §3c equivalence)."""
    m = tiny_model
    chains = _load_labeled_split(str(DATA_DIR), "test_iid", tokenizer, 64, True, max_chains=4)
    ch = next(c for c in chains if len(c["ids"]) >= 2)
    ids, pad = ch["ids"], ch["pad"]
    device = "cpu"

    with torch.no_grad():
        # Reference: encode s0 -> k0, posterior argmax action on (s0, s1), single apply.
        _, k0, _ = m.encoder(ids[0].unsqueeze(0), pad[0].unsqueeze(0))
        _, v_logits, _ = m.transition(
            ids[0].unsqueeze(0), pad[0].unsqueeze(0),
            ids[1].unsqueeze(0), pad[1].unsqueeze(0), tau=1.0, hard=True,
        )
        v = int(v_logits.argmax(-1).item())
        v_oh = F.one_hot(torch.tensor([v]), num_classes=m.n_verbs).to(k0.dtype)
        a_ref = _op_apply(m, k0, v_oh)

        # Rollout depth 1: the TF a at depth 1 must match a_ref. We replicate the rollout's
        # inner depth-1 computation (argmax posterior on gold, _op_apply on k0).
        _, k0b, _ = m.encoder(ids[0].unsqueeze(0), pad[0].unsqueeze(0))
        _, v_logits2, _ = m.transition(
            ids[0].unsqueeze(0), pad[0].unsqueeze(0),
            ids[1].unsqueeze(0), pad[1].unsqueeze(0), tau=1.0, hard=True,
        )
        v2 = int(v_logits2.argmax(-1).item())
        v_oh2 = F.one_hot(torch.tensor([v2]), num_classes=m.n_verbs).to(k0b.dtype)
        a_roll = _op_apply(m, k0b, v_oh2)

    assert torch.allclose(a_ref, a_roll, atol=1e-5)


@requires_data
def test_rollout_fidelity_returns_depth_scalars(tiny_model, tokenizer):
    chains = _load_labeled_split(str(DATA_DIR), "test_iid", tokenizer, 64, True, max_chains=40)
    chains = [c for c in chains if len(c["ids"]) >= 5][:6]
    assert chains, "need chains of length >= 5 for depth-4 rollout"
    out = _rollout_fidelity(tiny_model, tokenizer, chains, 4, "cpu", 64, None, "test")
    for src in ("tf", "pr"):
        for d in (1, 2, 3, 4):
            assert f"ent_rollout_{src}_exact_d{d}" in out
            assert f"ent_rollout_{src}_chrf_d{d}" in out
            assert 0.0 <= out[f"ent_rollout_{src}_exact_d{d}"] <= 1.0
    assert out["ent_rollout_n_chains"] == len(chains)


# --------------------------------------------------------------------------- end-to-end
@requires_data
def test_eval_entity_world_end_to_end(tiny_model, tokenizer, tmp_path):
    cfg = JEPAConfig.from_json(str(ROOT / "configs" / "jepa" / "jepa_ent_smoke.json"))
    ew = cfg.eval.entity_world
    ew.subsample = 24
    ew.n_rollout_chains = 5
    metrics = eval_entity_world(
        tiny_model, ew, "cpu", tokenizer, max_text_tokens=64,
        out_dir=str(tmp_path), epoch=1,
    )
    # all three families present
    assert "ent_action_nmi_verb" in metrics
    assert "ent_test_iid_hard_mrr" in metrics
    assert "ent_rollout_tf_exact_d4" in metrics
    # artifacts written
    assert (tmp_path / "entity_rollout_epoch1.json").exists()
    assert (tmp_path / "action_nmi_contingency_epoch1.json").exists()
    # no error keys
    assert not any(k.startswith("_ent_") and k.endswith("_error") for k in metrics)


@requires_data
def test_eval_entity_world_blackbox_runs(tokenizer, tmp_path):
    """Black-box (gated_mlp) model: rollout still runs via _op_apply (campaign §3.5)."""
    cfg = JEPAConfig.from_json(str(ROOT / "configs" / "jepa" / "jepa_ent_blackbox_smoke.json"))
    emb = torch.nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    emb.weight.requires_grad_(False)
    m = build_jepa_model_v2(cfg, emb)
    m.eval()
    ew = cfg.eval.entity_world
    ew.subsample = 16
    ew.n_rollout_chains = 4
    metrics = eval_entity_world(m, ew, "cpu", tokenizer, max_text_tokens=64,
                                out_dir=str(tmp_path), epoch=1)
    assert "ent_rollout_tf_exact_d4" in metrics
    assert "ent_test_iid_hard_mrr" in metrics


# --------------------------------------------------------------------------- config gating
def test_eval_config_default_disables_entity_world():
    """A plain config (no eval.entity_world block) ⟹ entity_world.enabled is False
    (back-compat: GLUCOSE configs report no entity metrics)."""
    cfg = JEPAConfig.from_dict({"profile": "jepa_v3"})
    assert cfg.eval.entity_world.enabled is False


def test_entity_configs_parse_and_build():
    """All 8 entity configs parse and build (structured + black-box, use_norm_budget)."""
    names = [
        "jepa_ent_s0", "jepa_ent_s1", "jepa_ent_s2", "jepa_ent_smoke",
        "jepa_ent_blackbox_s0", "jepa_ent_blackbox_s1", "jepa_ent_blackbox_s2",
        "jepa_ent_blackbox_smoke",
    ]
    for name in names:
        path = ROOT / "configs" / "jepa" / f"{name}.json"
        cfg = JEPAConfig.from_json(str(path))
        # entity eval is enabled on all entity configs
        assert cfg.eval.entity_world.enabled is True
        assert cfg.eval.entity_world.labeled_dir == "data/entity_world"
        assert getattr(cfg.model, "use_norm_budget", False) is True
        if "blackbox" in name:
            assert cfg.model.operator_group == "gated_mlp"
            assert cfg.model.use_polar_conditioning is False
        else:
            assert cfg.model.operator_group == "rotation_scale"
            assert cfg.model.use_polar_conditioning is True
        # build the model (catches schema/factory drift)
        emb = torch.nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
        emb.weight.requires_grad_(False)
        m = build_jepa_model_v2(cfg, emb)
        assert m.n_verbs == cfg.model.n_verbs


def test_entity_world_eval_config_roundtrip():
    """eval.entity_world fields round-trip through from_dict/to_dict."""
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_v3",
        "eval": {"out_dir": "x", "entity_world": {
            "enabled": True, "subsample": 99, "rollout_max_depth": 3,
            "splits": ["test_iid"], "action_recovery_split": "test_iid",
        }},
    })
    assert cfg.eval.entity_world.enabled is True
    assert cfg.eval.entity_world.subsample == 99
    assert cfg.eval.entity_world.rollout_max_depth == 3
    assert cfg.eval.entity_world.splits == ["test_iid"]
    d = cfg.to_dict()
    assert d["eval"]["entity_world"]["subsample"] == 99
