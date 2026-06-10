"""Unit-smoke tests for T5 (model + EMA + config + trainer helpers).

These run WITHOUT the concurrently-developed sibling modules (operator/slot_encoder/
losses/data/diagnostics) by substituting minimal mocks that satisfy the frozen
contracts in twm.jepa.__init__. They verify T5's own composition logic:
    - JEPAOperatorModel forward shapes, step_latent/undo_latent, EMA lifecycle
    - JEPAConfig.from_dict / from_json parsing against the §10 schema
    - train_jepa helper math (gumbel anneal, warmup+cosine lr)
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "legacy"))


# --------------------------------------------------------------------------- mocks
class MockOperator(nn.Module):
    """Diagonal rotation+scale stand-in satisfying the apply/inverse_apply contract.

    Per verb v: angle theta[v] (applied to each 2x2 block), scale r[v]. apply rotates
    + scales each (x,y) pair; inverse_apply rotates back / divides by r. Exact inverse
    so step_latent/undo_latent round-trips test the model wiring, not a real operator.
    """

    def __init__(self, n_verbs=8, d_noun=32):
        super().__init__()
        self._n_verbs = n_verbs
        self.d_noun = d_noun
        self.theta = nn.Parameter(torch.linspace(0.2, 1.0, n_verbs))
        self.log_r = nn.Parameter(torch.zeros(n_verbs) + 0.05)

    def _rot(self, k, v, inverse):
        # k: (B,M,dn), v: (B,M) long
        B, M, dn = k.shape
        th = self.theta[v]          # (B,M)
        r = torch.exp(self.log_r[v])  # (B,M)
        if inverse:
            th = -th
            r = 1.0 / r
        x = k[..., 0::2]
        y = k[..., 1::2]
        c = torch.cos(th).unsqueeze(-1)
        s = torch.sin(th).unsqueeze(-1)
        rr = r.unsqueeze(-1)
        xo = rr * (x * c - y * s)
        yo = rr * (x * s + y * c)
        out = torch.empty_like(k)
        out[..., 0::2] = xo
        out[..., 1::2] = yo
        return out

    def apply(self, k, v):
        return self._rot(k, v, inverse=False)

    def inverse_apply(self, a, v):
        return self._rot(a, v, inverse=True)

    @property
    def n_verbs(self):
        return self._n_verbs


class MockEncoder(nn.Module):
    """Minimal slot encoder: embed -> mean-context -> per-slot linear heads.

    Returns (slots (B,M,d), k (B,M,dn) standardized, verb_logits (B,M,V)).
    """

    def __init__(self, vocab=512, d_model=64, d_noun=32, n_slots=8, n_verbs=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.slot_q = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)
        self.noun_head = nn.Linear(d_model, d_noun)
        self.verb_head = nn.Linear(d_model, n_verbs)
        self.n_slots = n_slots

    def forward(self, text_ids, text_pad):
        x = self.emb(text_ids)                # (B,T,d)
        ctx = x.mean(dim=1, keepdim=True)     # (B,1,d)
        slots = self.proj(self.slot_q).unsqueeze(0) + ctx  # (B,M,d)
        k = self.noun_head(slots)
        # standardize per dim across (B,M) — matches the "standardize, not L2" policy
        flat = k.reshape(-1, k.shape[-1])
        k = ((k - flat.mean(0)) / (flat.std(0) + 1e-5))
        verb_logits = self.verb_head(slots)
        return slots, k, verb_logits


def _make_model(d_model=64, d_noun=32, n_slots=8, n_verbs=8, n_heads=4):
    from twm.jepa.legacy.model_v1 import JEPAOperatorModel
    enc = MockEncoder(d_model=d_model, d_noun=d_noun, n_slots=n_slots, n_verbs=n_verbs)
    op = MockOperator(n_verbs=n_verbs, d_noun=d_noun)
    return JEPAOperatorModel(enc, op, d_noun=d_noun, n_verbs=n_verbs, n_heads=n_heads)


# --------------------------------------------------------------------------- config
def test_config_from_dict_defaults_and_overrides():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano",
        "seed": 7,
        "model": {"d_noun": 32},
        "loss": {"w_pred": 1.0, "sigreg": {"n_slices": 256}, "verb": {"anneal_frac": 0.3}},
        "optim": {"epochs": 3, "batch_size": 4},
    })
    # profile overlay populated model fields not in JSON
    assert cfg.model.d_model == 64
    assert cfg.model.n_slots == 8
    assert cfg.model.n_verbs == 8
    assert cfg.model.tie_text_layers is True
    assert cfg.seed == 7
    assert cfg.loss.sigreg.standardize is True  # default preserved
    assert cfg.loss.verb.gumbel_tau_start == 2.0
    assert cfg.optim.epochs == 3


def test_config_from_json_roundtrip(tmp_path):
    from twm.jepa.config import JEPAConfig
    nano = ROOT / "configs" / "archive" / "jepa_nano.json"
    mini = ROOT / "configs" / "archive" / "jepa_mini.json"
    c_nano = JEPAConfig.from_json(nano)
    c_mini = JEPAConfig.from_json(mini)
    assert c_nano.model.d_model == 64 and c_nano.model.n_verbs == 8
    assert c_mini.model.d_model == 128 and c_mini.model.n_verbs == 16
    assert c_mini.model.tie_text_layers is False
    assert c_nano.ema.tau == 0.995
    assert c_nano.eval.out_dir == "results/jepa_nano"
    # save -> reload stability
    out = tmp_path / "rt.json"
    c_nano.save(out)
    c2 = JEPAConfig.from_json(out)
    assert c2.model.d_model == c_nano.model.d_model
    assert c2.loss.w_sigreg == c_nano.loss.w_sigreg


def test_jepa_profiles_separate_from_model_profiles():
    # JEPA_PROFILES must not leak into config.PROFILES (build_model_config untouched)
    from twm.jepa import JEPA_PROFILES
    from twm.config import PROFILES as MODEL_PROFILES
    assert "jepa_nano" in JEPA_PROFILES
    assert "jepa_nano" not in MODEL_PROFILES


# --------------------------------------------------------------------------- model
def test_forward_shapes():
    torch.manual_seed(0)
    m = _make_model()
    B, T, M, dn = 4, 16, 8, 32
    src = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.randint(0, 512, (B, T))
    out = m(src, pad, tgt, pad, gumbel_tau=1.0, hard=False)
    assert out["k"].shape == (B, M, dn)
    assert out["a"].shape == (B, M, dn)
    assert out["zhat"].shape == (B, dn)
    assert out["z_target"].shape == (B, dn)
    assert out["verb"].shape == (B, M)
    assert out["verb_logits"].shape == (B, M, 8)


def test_forward_without_target_skips_ema():
    m = _make_model()
    src = torch.randint(0, 512, (2, 8))
    pad = torch.zeros(2, 8, dtype=torch.bool)
    out = m(src, pad)
    assert "z_target" not in out


def test_z_target_is_detached():
    m = _make_model()
    src = torch.randint(0, 512, (2, 8))
    pad = torch.zeros(2, 8, dtype=torch.bool)
    out = m(src, pad, src, pad)
    assert not out["z_target"].requires_grad


def test_step_undo_roundtrip():
    m = _make_model()
    k = torch.randn(3, 8, 32)
    a = m.step_latent(k, 2)
    k2 = m.undo_latent(a, 2)
    assert torch.allclose(k, k2, atol=1e-4), (k - k2).abs().max().item()


def test_step_latent_accepts_scalar_and_tensor_verbs():
    m = _make_model()
    k = torch.randn(2, 8, 32)
    a_scalar = m.step_latent(k, 3)
    v = torch.full((2, 8), 3, dtype=torch.long)
    a_tensor = m.step_latent(k, v)
    assert torch.allclose(a_scalar, a_tensor)


# --------------------------------------------------------------------------- EMA
def test_ema_deepcopy_requires_grad_false():
    m = _make_model()
    assert all(not p.requires_grad for p in m.ema.parameters())
    assert any(p.requires_grad for p in m.encoder.parameters())


def test_ema_initially_matches_online():
    m = _make_model()
    online = dict(m._online_bundle.named_parameters())
    for name, ep in m.ema.named_parameters():
        assert torch.allclose(ep, online[name]), name


def test_ema_update_moves_toward_online():
    m = _make_model()
    # perturb online encoder so ema and online diverge
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(torch.randn_like(p))
    before = {n: p.clone() for n, p in m.ema.named_parameters()}
    m.ema_update(tau=0.9)
    online = dict(m._online_bundle.named_parameters())
    for name, ep in m.ema.named_parameters():
        moved = (ep - before[name]).norm().item()
        gap = (online[name] - before[name]).norm().item()
        if gap > 1e-6:
            assert moved > 0, name
            # tau=0.9 -> moved 10% of the gap
            assert moved < gap, name


def test_online_parameters_excludes_ema():
    m = _make_model()
    ema_ids = {id(p) for p in m.ema.parameters()}
    online_ids = {id(p) for p in m.online_parameters()}
    assert ema_ids.isdisjoint(online_ids)
    # predictor (online-only) is present
    pred_ids = {id(p) for p in m.predictor.parameters()}
    assert pred_ids.issubset(online_ids)


def test_no_param_duplicates_in_module_tree():
    m = _make_model()
    ids = [id(p) for p in m.parameters()]
    assert len(ids) == len(set(ids)), "parameters() yielded duplicates"


# --------------------------------------------------------------------------- trainer helpers
def test_gumbel_anneal_monotone():
    import train_jepa as T
    from twm.jepa.config import JEPAConfig
    vc = JEPAConfig.from_dict({}).loss.verb
    total = 1000
    taus = [T.gumbel_tau_at(s, total, vc) for s in range(0, total, 50)]
    assert taus[0] == pytest.approx(vc.gumbel_tau_start, abs=1e-6)
    assert taus[-1] == pytest.approx(vc.gumbel_tau_end, abs=1e-6)
    # monotone non-increasing then flat
    assert all(b <= a + 1e-9 for a, b in zip(taus, taus[1:]))
    # held at end past the anneal window
    assert T.gumbel_tau_at(total, total, vc) == pytest.approx(vc.gumbel_tau_end)


def test_lr_factor_warmup_then_cosine():
    import train_jepa as T
    warmup, total = 100, 1000
    assert T.lr_factor_at(0, warmup, total) == pytest.approx(1 / warmup)
    assert T.lr_factor_at(warmup - 1, warmup, total) == pytest.approx(1.0, abs=1e-6)
    assert T.lr_factor_at(warmup, warmup, total) == pytest.approx(1.0, abs=1e-3)
    end = T.lr_factor_at(total, warmup, total)
    assert end == pytest.approx(0.0, abs=1e-6)


def test_device_resolver_returns_device():
    import train_jepa as T
    d = T.resolve_device()
    assert isinstance(d, torch.device)
    assert T.resolve_device("cpu") == torch.device("cpu")


def test_build_token_emb_frozen():
    import train_jepa as T
    emb = T.build_token_emb(512, 64)
    assert emb.weight.shape == (512, 64)
    assert not emb.weight.requires_grad


# --------------------------------------------------------------------------- backward
def test_backward_flows_to_online_not_ema():
    torch.manual_seed(0)
    m = _make_model()
    src = torch.randint(0, 512, (4, 12))
    pad = torch.zeros(4, 12, dtype=torch.bool)
    out = m(src, pad, src, pad, gumbel_tau=1.0, hard=False)
    # simple surrogate loss on zhat vs z_target
    loss = ((out["zhat"] - out["z_target"]) ** 2).mean()
    loss.backward()
    # online encoder gets grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.parameters())
    # ema never gets grad
    assert all(p.grad is None for p in m.ema.parameters())
