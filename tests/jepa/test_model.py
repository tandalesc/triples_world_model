"""Unit-smoke tests for Task C (JEPAOperatorModelV2 + v2 config + train_jepa_v2).

Run WITHOUT the concurrently-developed sibling modules (transition/decoder/losses/
diagnostics) by substituting minimal mocks that satisfy the FROZEN interfaces from
research/jepa_v2_latent_actions.md §11. These verify Task C's own composition logic:

  - JEPAOperatorModelV2 forward shapes + leakage invariant (decoder memory IS a*)
  - param budget <= 250K (nano-v2)
  - EMA excluded from optimizer / online_parameters
  - rollout (prior-sampled action + user-set action)
  - JEPAConfig.from_dict/from_json parses the nested transition/prior/decoder blocks
  - append_eos data path inserts <eos>=4 as a real (unmasked) token
  - train_jepa_v2 helper math (tau anneal 3->1 over 50%, warmup+cosine lr)
"""

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


# =========================================================================== mocks
class MockOperator(nn.Module):
    """Rotation+scale stand-in. apply accepts v as soft (B,M,V) float OR (B,M) long
    (the v2 model broadcasts a hard one-hot (B,V) -> (B,M,V))."""

    def __init__(self, n_verbs=8, d_noun=32, block=2):
        super().__init__()
        self._n_verbs = n_verbs
        self.d_noun = d_noun
        self.n_blocks = d_noun // 2
        self.theta = nn.Parameter(torch.linspace(0.2, 1.0, n_verbs).unsqueeze(-1).repeat(1, self.n_blocks))
        self.log_r = nn.Parameter(torch.zeros(n_verbs, self.n_blocks) + 0.05)

    def _coeffs(self, v):
        rcos = (torch.exp(self.log_r) * torch.cos(self.theta))  # (V, nb)
        rsin = (torch.exp(self.log_r) * torch.sin(self.theta))
        if torch.is_floating_point(v):  # (B,M,V)
            a = v @ rcos
            b = v @ rsin
        else:  # (B,M) long
            a = F.embedding(v.long(), rcos)
            b = F.embedding(v.long(), rsin)
        return a, b

    def _apply(self, k, a, b):
        xp = k.reshape(*k.shape[:-1], self.n_blocks, 2)
        x, y = xp[..., 0], xp[..., 1]
        return torch.stack([a * x - b * y, b * x + a * y], dim=-1).reshape(k.shape)

    def apply(self, k, v):
        a, b = self._coeffs(v)
        return self._apply(k, a, b)

    def inverse_apply(self, a_in, v):
        rcos = (torch.exp(-self.log_r) * torch.cos(self.theta))
        rsin = (torch.exp(-self.log_r) * torch.sin(self.theta))
        if torch.is_floating_point(v):
            ac = v @ rcos
            bc = -(v @ rsin)
        else:
            ac = F.embedding(v.long(), rcos)
            bc = -F.embedding(v.long(), rsin)
        return self._apply(a_in, ac, bc)

    @property
    def n_verbs(self):
        return self._n_verbs


class MockEncoder(nn.Module):
    """Minimal SlotEncoder: exposes forward -> (slots,k,verb_logits) AND a bound
    encode_text(ids, pad) -> (B,T,d) (the shared trunk the posterior/prior reuse)."""

    def __init__(self, vocab=512, d_model=64, d_noun=32, n_slots=8, n_verbs=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.token_emb.weight.requires_grad_(False)  # frozen, like the real encoder
        self.trunk = nn.Linear(d_model, d_model)
        self.slot_q = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.proj = nn.Linear(d_model, d_model)
        self.noun_head = nn.Linear(d_model, d_noun)
        self.verb_head = nn.Linear(d_model, n_verbs)
        self.n_slots = n_slots

    def encode_text(self, text_ids, text_pad):
        return self.trunk(self.token_emb(text_ids))  # (B,T,d)

    def forward(self, text_ids, text_pad):
        ctx = self.encode_text(text_ids, text_pad).mean(dim=1, keepdim=True)
        slots = self.proj(self.slot_q).unsqueeze(0) + ctx
        k = self.noun_head(slots)
        flat = k.reshape(-1, k.shape[-1])
        k = (k - flat.mean(0)) / (flat.std(0) + 1e-5)
        return slots, k, self.verb_head(slots)


class MockTransition(nn.Module):
    """Posterior q(v|t,t+1). Frozen sig §11: forward(src,src_pad,tgt,tgt_pad,tau,hard)
    -> (v_onehot (B,V), v_logits (B,V), pool_t (B,d)). Sees BOTH texts via the shared
    trunk; emits ONE hard ST one-hot per pair."""

    def __init__(self, encode_text_fn, d_model=64, n_verbs=8, mlp_hidden=128, use_delta=True):
        super().__init__()
        self.encode_text_fn = encode_text_fn  # NOT a submodule (shared trunk, no new params)
        self.use_delta = use_delta
        self.n_verbs = n_verbs
        in_dim = 3 * d_model if use_delta else 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.GELU(),
            nn.LayerNorm(mlp_hidden), nn.Linear(mlp_hidden, n_verbs),
        )

    @staticmethod
    def _pool(ctx, pad):
        if pad is None:
            return ctx.mean(dim=1)
        m = (~pad.bool()).unsqueeze(-1).to(ctx.dtype)
        return (ctx * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, src_ids, src_pad, tgt_ids, tgt_pad, tau, hard):
        pool_t = self._pool(self.encode_text_fn(src_ids, src_pad), src_pad)
        pool_t1 = self._pool(self.encode_text_fn(tgt_ids, tgt_pad), tgt_pad)
        if self.use_delta:
            pair = torch.cat([pool_t, pool_t1, pool_t1 - pool_t], dim=-1)
        else:
            pair = torch.cat([pool_t, pool_t1], dim=-1)
        v_logits = self.mlp(pair)  # (B,V)
        v_onehot = F.gumbel_softmax(v_logits, tau=tau, hard=hard, dim=-1)
        return v_onehot, v_logits, pool_t


class MockPrior(nn.Module):
    """Prior p(v|pool_t). Frozen sig §11: forward(pool_t) -> p_logits (B,V)."""

    def __init__(self, d_model=64, n_verbs=8, mlp_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, n_verbs)
        )

    def forward(self, pool_t):
        return self.mlp(pool_t)


class SpyDecoder(nn.Module):
    """TokenDecoder mock. Frozen sig §11: constructor has NO posterior/text_t+1 arg.
    forward(a_star, tgt_ids, tgt_pad) -> logits (B,T,V). Records the id() of the memory
    tensor it received so the leakage test can assert memory IS a* (not tgt encodings)."""

    def __init__(self, vocab_size=512, d_dec=64, n_layers=1, n_heads=4, d_ff=128,
                 d_noun=32, max_text_tokens=64, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_dec)  # decoder's OWN embedding
        self.mem_proj = nn.Linear(d_noun, d_dec)
        self.out = nn.Linear(d_dec, vocab_size)
        self.last_memory_id = None  # leakage spy

    def forward(self, a_star, tgt_ids, tgt_pad):
        self.last_memory_id = id(a_star)
        B, T = tgt_ids.shape
        mem = self.mem_proj(a_star).mean(dim=1, keepdim=True)  # (B,1,d)
        tok = self.token_emb(tgt_ids)  # (B,T,d)
        return self.out(tok + mem)  # (B,T,V)

    @torch.no_grad()
    def generate(self, a_star, max_tokens=None, temperature=0.0):
        B = a_star.shape[0]
        T = max_tokens or 8
        return torch.zeros(B, T, dtype=torch.long, device=a_star.device)


def _make_model(d_model=64, d_noun=32, n_slots=8, n_verbs=8, n_heads=4, use_pred=True):
    from twm.jepa.model import JEPAOperatorModelV2
    enc = MockEncoder(d_model=d_model, d_noun=d_noun, n_slots=n_slots, n_verbs=n_verbs)
    op = MockOperator(n_verbs=n_verbs, d_noun=d_noun)
    trans = MockTransition(enc.encode_text, d_model=d_model, n_verbs=n_verbs)
    prior = MockPrior(d_model=d_model, n_verbs=n_verbs)
    dec = SpyDecoder(d_dec=d_model, d_noun=d_noun, n_heads=n_heads)
    return JEPAOperatorModelV2(
        enc, op, trans, prior, dec, d_noun=d_noun, n_verbs=n_verbs,
        n_heads=n_heads, use_pred=use_pred,
    )


# =========================================================================== forward
def test_forward_shapes():
    torch.manual_seed(0)
    m = _make_model()
    B, T, M, dn, V = 4, 16, 8, 32, 8
    src = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.randint(0, 512, (B, T))
    out = m(src, pad, tgt, pad, tau=1.0, hard=True)
    assert out["k"].shape == (B, M, dn)
    assert out["a"].shape == (B, M, dn)
    assert out["v"].shape == (B,)
    assert out["v_onehot"].shape == (B, V)
    assert out["v_logits"].shape == (B, V)
    assert out["p_logits"].shape == (B, V)
    assert out["zhat"].shape == (B, dn)
    assert out["z_target"].shape == (B, dn)
    # decoder logits are (B, T, vocab); only batch+time are contract-fixed here.
    assert out["logits"].shape[:2] == (B, T)
    assert out["logits"].shape[-1] == 512  # SpyDecoder vocab


def test_action_is_one_hot_per_pair():
    m = _make_model()
    src = torch.randint(0, 512, (5, 10))
    pad = torch.zeros(5, 10, dtype=torch.bool)
    out = m(src, pad, src, pad, tau=1.0, hard=True)
    # hard ST one-hot: each row sums to 1 (design §2.3 single discrete action).
    assert torch.allclose(out["v_onehot"].sum(-1), torch.ones(5), atol=1e-5)


def test_z_target_detached():
    m = _make_model()
    src = torch.randint(0, 512, (3, 8))
    pad = torch.zeros(3, 8, dtype=torch.bool)
    out = m(src, pad, src, pad)
    assert not out["z_target"].requires_grad


def test_use_pred_false_skips_aux():
    m = _make_model(use_pred=False)
    src = torch.randint(0, 512, (3, 8))
    pad = torch.zeros(3, 8, dtype=torch.bool)
    out = m(src, pad, src, pad)
    assert out["zhat"] is None and out["z_target"] is None
    # token logits still produced (primary loss survives the ablation)
    assert out["logits"].shape[:2] == (3, 8)


# =========================================================================== leakage (§6)
def test_leakage_decoder_memory_is_a_star():
    """The decoder's cross-attn memory MUST be a* (operator output), NOT tgt encodings
    or posterior features (design §6 L1/L2). Assert via the spy's recorded memory id."""
    m = _make_model()
    src = torch.randint(0, 512, (2, 8))
    pad = torch.zeros(2, 8, dtype=torch.bool)
    tgt = torch.randint(0, 512, (2, 8))
    out = m(src, pad, tgt, pad)
    assert m.decoder.last_memory_id == id(out["a"]), "decoder memory must be a*"


def test_leakage_decoder_constructor_has_no_posterior_channel():
    """Frozen contract (design §6 L2): TokenDecoder constructor exposes no posterior /
    text_t+1 argument. The model can therefore never wire t+1 features into the decoder."""
    from twm.jepa.model import JEPAOperatorModelV2
    # Inspect the decoder the model actually holds.
    m = _make_model()
    params = set(inspect.signature(type(m.decoder).__init__).parameters)
    forbidden = {"posterior", "v_logits", "tgt_encoding", "pool_t1", "text_t1", "next_state"}
    assert params.isdisjoint(forbidden)
    # forward takes exactly (a_star, tgt_ids, tgt_pad) — no extra leakage channel.
    fwd = set(inspect.signature(type(m.decoder).forward).parameters) - {"self"}
    assert fwd == {"a_star", "tgt_ids", "tgt_pad"}


def test_token_decoder_gradient_flows_into_posterior():
    """L_token through a* must reach v_logits (the only future->decoder path is v).
    Confirms the ST estimator keeps the posterior trainable (design §2.3 / §6 L3)."""
    torch.manual_seed(0)
    m = _make_model()
    src = torch.randint(0, 512, (4, 10))
    pad = torch.zeros(4, 10, dtype=torch.bool)
    tgt = torch.randint(0, 512, (4, 10))
    out = m(src, pad, tgt, pad, tau=1.0, hard=True)
    loss = out["logits"].float().pow(2).mean()  # surrogate token loss
    loss.backward()
    posterior_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.transition.mlp.parameters()
    )
    assert posterior_grad, "L_token did not reach the posterior MLP through the ST action"


# =========================================================================== budget / EMA
def test_param_budget_under_250k_with_real_modules():
    """Build the REAL nano-v2 (sibling modules, if present) and assert <= 250K trainable.
    Skips gracefully if a sibling module is not yet written."""
    from twm.jepa.config import JEPAConfig
    try:
        from twm.jepa.model import build_jepa_model_v2
        cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
        emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
        emb.weight.requires_grad_(False)
        model = build_jepa_model_v2(cfg, emb)
    except Exception as e:
        pytest.skip(f"sibling v2 module not ready: {e}")
    n = model.trainable_param_count()
    assert n <= 250_000, f"nano-v2 trainable params {n:,} exceed 250K budget"


def test_ema_requires_grad_false_and_excluded():
    m = _make_model()
    assert all(not p.requires_grad for p in m.ema.parameters())
    ema_ids = {id(p) for p in m.ema.parameters()}
    online_ids = {id(p) for p in m.online_parameters()}
    assert ema_ids.isdisjoint(online_ids)


def test_ema_update_moves_toward_online():
    m = _make_model()
    with torch.no_grad():
        for p in m.encoder.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    before = {n: p.clone() for n, p in m.ema.named_parameters()}
    m.ema_update(tau=0.9)
    online = dict(m._online_bundle.named_parameters())
    for name, ep in m.ema.named_parameters():
        # The frozen token_emb is the SAME tensor on both sides (re-pointed in __init__)
        # so its "EMA update" is a self-update — exclude it from the move-toward check.
        if name.endswith("token_emb.weight"):
            continue
        gap = (online[name] - before[name]).norm().item()
        if gap > 1e-6:
            moved = (ep - before[name]).norm().item()
            assert 0 < moved < gap, name


def test_no_param_duplicates():
    m = _make_model()
    ids = [id(p) for p in m.parameters()]
    assert len(ids) == len(set(ids))


def test_decoder_token_emb_not_shared_with_encoder():
    """Decoder owns its OWN token embedding (design §9): generating tokens != encoding."""
    m = _make_model()
    assert m.decoder.token_emb.weight.data_ptr() != m.encoder.token_emb.weight.data_ptr()


# =========================================================================== rollout (§1)
def test_rollout_prior_sampled():
    m = _make_model()
    src = torch.randint(0, 512, (3, 8))
    pad = torch.zeros(3, 8, dtype=torch.bool)
    r = m.rollout(src, pad, sample=True, max_tokens=6)
    assert r["v"].shape == (3,)
    assert r["a"].shape == (3, 8, 32)
    assert r["gen_ids"].shape == (3, 6)


def test_rollout_user_action_override():
    m = _make_model()
    src = torch.randint(0, 512, (3, 8))
    pad = torch.zeros(3, 8, dtype=torch.bool)
    r = m.rollout(src, pad, verb_idx=2, max_tokens=4)
    assert torch.equal(r["v"], torch.full((3,), 2))


def test_rollout_greedy_argmax_deterministic():
    torch.manual_seed(0)
    m = _make_model()
    src = torch.randint(0, 512, (3, 8))
    pad = torch.zeros(3, 8, dtype=torch.bool)
    r1 = m.rollout(src, pad, sample=False, max_tokens=4)
    r2 = m.rollout(src, pad, sample=False, max_tokens=4)
    assert torch.equal(r1["v"], r2["v"])  # argmax of prior is deterministic


def test_step_undo_roundtrip():
    m = _make_model()
    k = torch.randn(3, 8, 32)
    a = m.step_latent(k, 2)
    k2 = m.undo_latent(a, 2)
    assert torch.allclose(k, k2, atol=1e-4), (k - k2).abs().max().item()


# =========================================================================== config (§10)
def test_config_parses_nested_v2_blocks():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    assert cfg.profile == "jepa_nano_v2"
    assert cfg.data.append_eos is True
    assert cfg.model.transition.mlp_hidden == 128
    assert cfg.model.transition.use_delta is True
    assert cfg.model.prior.mlp_hidden == 64
    assert cfg.model.decoder.d_dec == 64
    assert cfg.model.decoder.n_layers == 1
    assert cfg.model.decoder.n_heads == 4
    assert cfg.model.decoder.d_ff == 128
    assert cfg.loss.w_token == 1.0
    assert cfg.loss.w_prior == 0.1
    assert cfg.loss.verb.gumbel_tau_start == 3.0
    assert cfg.loss.verb.gumbel_tau_end == 1.0
    assert cfg.loss.verb.anneal_frac == 0.5
    assert cfg.eval.n_text_samples == 16
    assert cfg.eval.temperatures == [0.7, 1.0]


def test_config_v2_defaults_when_blocks_absent():
    """A config without the nested blocks still constructs valid dataclasses (v1 compat)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({"profile": "jepa_nano"})
    assert cfg.model.transition.mlp_hidden == 128
    assert cfg.model.decoder.n_layers == 1
    assert cfg.data.append_eos is False  # v1 default preserved


def test_config_roundtrip_save_reload(tmp_path):
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    out = tmp_path / "rt.json"
    cfg.save(out)
    c2 = JEPAConfig.from_json(out)
    assert c2.model.decoder.d_dec == cfg.model.decoder.d_dec
    assert c2.data.append_eos == cfg.data.append_eos
    assert c2.loss.w_token == cfg.loss.w_token


# ====================================================== v2.1 polar conditioning (§3/§10/§11)
def test_config_v2_defaults_polar_flags_off():
    """v2.0 configs default the v2.1 flags to False ⟹ identical model (the §11 gate)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    assert cfg.model.use_polar_conditioning is False
    assert cfg.model.use_kind_head is False
    assert cfg.model.kind_codebook_size == 16


def test_config_v21_parses_polar_flag():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    assert cfg.model.use_polar_conditioning is True
    assert cfg.model.use_kind_head is False


def _build_real(cfg):
    """Build a real v2.1/v2.0 model (real operator + conditioner) from a JEPAConfig."""
    from twm.jepa.model import build_jepa_model_v2
    emb = nn.Embedding(cfg.data.vocab_size, cfg.model.d_model)
    emb.weight.requires_grad_(False)
    return build_jepa_model_v2(cfg, emb)


def test_v20_model_has_no_conditioner():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    model = _build_real(cfg)
    assert model.conditioner is None
    assert model.kind_head is None


def test_v21_model_has_zero_init_conditioner():
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    model = _build_real(cfg)
    assert model.conditioner is not None
    # zero-init H (the v2.1==v2.0-at-init guarantee).
    assert torch.equal(
        model.conditioner.H.weight, torch.zeros_like(model.conditioner.H.weight)
    )


def test_v21_param_delta_is_nb_squared():
    """v2.1 adds exactly nb² params (the H matrix) over v2.0 — budget unaffected (§6)."""
    from twm.jepa.config import JEPAConfig
    cfg20 = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    cfg21 = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    m20 = _build_real(cfg20)
    m21 = _build_real(cfg21)
    nb = cfg20.model.d_noun // 2
    assert m21.trainable_param_count() - m20.trainable_param_count() == nb * nb


def test_v21_equals_v20_at_init():
    """Zero-init H ⟹ the v2.1 forward output (k, a, logits) is bitwise-identical to the
    v2.0 twin at init — the load-bearing behavior-preservation guarantee (§11)."""
    from twm.jepa.config import JEPAConfig

    cfg20 = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    cfg21 = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")

    torch.manual_seed(123)
    m20 = _build_real(cfg20)
    torch.manual_seed(123)
    m21 = _build_real(cfg21)
    m20.eval(); m21.eval()

    # Copy v2.0's trainable weights into v2.1 so the two models are parameter-identical
    # apart from the zero-init H (which the conditioner build shifted later-module RNG
    # for). This isolates the guarantee under test: a zero offset leaves the forward
    # output untouched, regardless of how the surrounding params were seeded.
    sd20 = m20.state_dict()
    missing, unexpected = m21.load_state_dict(sd20, strict=False)
    # the only param v2.1 has that v2.0 lacks is the zero-init H.
    assert set(missing) <= {"conditioner.H.weight"}, missing
    assert not unexpected, unexpected

    B, T = 4, 16
    src = torch.randint(5, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.randint(5, 512, (B, T))

    with torch.no_grad():
        torch.manual_seed(999)  # same Gumbel noise for both forwards
        o20 = m20(src, pad, tgt, pad, tau=1.0, hard=True)
        torch.manual_seed(999)
        o21 = m21(src, pad, tgt, pad, tau=1.0, hard=True)

    # The forward at init must match: a* (conditioned with a zero offset) == a* (plain).
    assert torch.equal(o20["a"], o21["a"]), (o20["a"] - o21["a"]).abs().max().item()
    assert torch.equal(o20["k"], o21["k"])
    assert torch.equal(o20["logits"], o21["logits"])


def test_v21_kind_head_emits_kind_ids():
    """When use_kind_head=True, the forward surfaces kind_ids (diagnostic; never routes)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_dict({
        "profile": "jepa_nano_v2",
        "model": {"use_polar_conditioning": True, "use_kind_head": True,
                  "kind_codebook_size": 8},
    })
    model = _build_real(cfg)
    assert model.kind_head is not None
    B, T = 3, 12
    src = torch.randint(5, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    tgt = torch.randint(5, 512, (B, T))
    out = model(src, pad, tgt, pad, tau=1.0, hard=True)
    assert "kind_ids" in out
    assert out["kind_ids"].shape == (B, cfg.model.n_slots)
    # kind head does NOT change the routed action / a* (it is a microscope, not a gear).
    assert out["a"].shape == (B, cfg.model.n_slots, cfg.model.d_noun)


def test_v21_step_undo_roundtrip_with_offset():
    """Pet/demo API: step_latent returns (a, offset); undo_latent with that offset is
    an exact inverse even under a conditioned (and scaling) verb (§4.2)."""
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa" / "jepa_nano_v21.json")
    model = _build_real(cfg)
    # make H nonzero so the offset is genuinely state-dependent.
    model.conditioner.H.weight.data.normal_(0, 0.3)
    torch.manual_seed(7)
    k = torch.randn(3, cfg.model.n_slots, cfg.model.d_noun)
    a, offset = model.step_latent(k, 2)
    k_rt = model.undo_latent(a, 2, theta_offset=offset)
    assert torch.allclose(k, k_rt, atol=1e-4), (k - k_rt).abs().max().item()


def test_jepa_nano_v2_profile_registered():
    from twm.jepa import JEPA_PROFILES
    from twm.config import PROFILES as MODEL_PROFILES
    assert "jepa_nano_v2" in JEPA_PROFILES
    assert "jepa_nano_v2" not in MODEL_PROFILES  # must not leak into model profiles


# =========================================================================== data append_eos
def test_append_eos_inserts_real_eos_token():
    """append_eos=True inserts <eos>=4 at the first pad slot; it must NOT be masked."""
    from twm.jepa.data import JEPAChainDataset
    from twm.domain_bpe import DomainBPETokenizer
    tok = DomainBPETokenizer.load(
        str(ROOT / "data" / "glucose" / "jepa_bpe_512.json"), max_length=64
    )
    ds = JEPAChainDataset(
        path=str(ROOT / "data" / "glucose" / "chain_general_train.jsonl"),
        tokenizer=tok, max_text_tokens=64, append_eos=True,
    )
    item = ds[0]
    ids, pad = item["tgt_ids"], item["tgt_pad"]
    assert (ids == 4).any(), "expected an <eos>=4 token in the target"
    eos_pos = (ids == 4).nonzero()[0, 0].item()
    assert not pad[eos_pos].item(), "<eos> position must NOT be masked"
    # everything strictly after the (first) eos is pad
    if eos_pos + 1 < ids.shape[0]:
        assert pad[eos_pos + 1:].all(), "post-eos positions must be pad"


def test_append_eos_default_false_preserves_v1():
    from twm.jepa.data import JEPAChainDataset
    from twm.domain_bpe import DomainBPETokenizer
    tok = DomainBPETokenizer.load(
        str(ROOT / "data" / "glucose" / "jepa_bpe_512.json"), max_length=64
    )
    ds = JEPAChainDataset(
        path=str(ROOT / "data" / "glucose" / "chain_general_train.jsonl"),
        tokenizer=tok, max_text_tokens=64,
    )
    # v1 default: no eos appended.
    assert ds.append_eos is False


# =========================================================================== trainer helpers
def test_tau_anneal_3_to_1_over_half():
    import train_jepa_v2 as T
    from twm.jepa.config import JEPAConfig
    vc = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json").loss.verb
    total = 1000
    taus = [T.gumbel_tau_at(s, total, vc) for s in range(0, total, 50)]
    assert taus[0] == pytest.approx(3.0, abs=1e-6)
    assert taus[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(b <= a + 1e-9 for a, b in zip(taus, taus[1:]))  # monotone non-increasing
    # held at 1.0 past the 50% anneal window
    assert T.gumbel_tau_at(600, total, vc) == pytest.approx(1.0)
    # at exactly 50% it has reached the floor
    assert T.gumbel_tau_at(500, total, vc) == pytest.approx(1.0)


def test_lr_factor_warmup_then_cosine():
    import train_jepa_v2 as T
    warmup, total = 100, 1000
    assert T.lr_factor_at(0, warmup, total) == pytest.approx(1 / warmup)
    assert T.lr_factor_at(warmup - 1, warmup, total) == pytest.approx(1.0, abs=1e-6)
    assert T.lr_factor_at(total, warmup, total) == pytest.approx(0.0, abs=1e-6)


def test_device_resolver():
    import train_jepa_v2 as T
    assert isinstance(T.resolve_device(), torch.device)
    assert T.resolve_device("cpu") == torch.device("cpu")


def test_build_token_emb_frozen():
    import train_jepa_v2 as T
    emb = T.build_token_emb(512, 64)
    assert emb.weight.shape == (512, 64)
    assert not emb.weight.requires_grad


def test_save_checkpoint_per_eval_retention(tmp_path):
    """Per-eval retention (design §11): tag='ep{N}' writes a DISTINCT file per eval, so
    an earlier checkpoint is never clobbered by 'latest'."""
    import train_jepa_v2 as T
    from twm.jepa.config import JEPAConfig
    cfg = JEPAConfig.from_json(ROOT / "configs" / "jepa_nano_v2.json")
    m = _make_model()
    T.save_checkpoint(m, cfg, step=10, epoch=5, out_dir=tmp_path, tag="ep5")
    T.save_checkpoint(m, cfg, step=20, epoch=10, out_dir=tmp_path, tag="ep10")
    T.save_checkpoint(m, cfg, step=20, epoch=10, out_dir=tmp_path, tag="latest")
    assert (tmp_path / "model_ep5.pt").exists()
    assert (tmp_path / "model_ep10.pt").exists()
    assert (tmp_path / "model_latest.pt").exists()


# =========================================================================== backward
def test_backward_flows_to_online_not_ema():
    torch.manual_seed(0)
    m = _make_model()
    src = torch.randint(0, 512, (4, 12))
    pad = torch.zeros(4, 12, dtype=torch.bool)
    tgt = torch.randint(0, 512, (4, 12))
    out = m(src, pad, tgt, pad, tau=1.0, hard=True)
    loss = out["logits"].float().pow(2).mean() + (out["zhat"] - out["z_target"]).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.decoder.parameters())
    assert all(p.grad is None for p in m.ema.parameters())
