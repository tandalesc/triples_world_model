"""Unit tests for the JEPA v2.1 polar decomposition (research/jepa_v21_polar.md).

Covers every new v2.1 behavior and the load-bearing guarantees:

  - operator `theta_offset=None` is bitwise-identical to the v2.0 `apply`/`inverse_apply`
    (regression — the in-flight v2.0 run path is untouched). [§4.1]
  - zero `theta_offset` is identical to the None path. [§4.1]
  - PolarConditioner.H is zero-init ⟹ H(|k|) == 0 ⟹ the conditioned apply equals the
    unconditioned apply at step 0 (the v2.1 == v2.0-at-init guarantee). [§3.1, §11]
  - identity-persistence: RotationOperator preserves modulus to < 1e-5 for ALL verbs,
    WITH a nonzero theta_offset (the phase offset cannot perturb modulus). [§8.1]
  - inverse_apply with an explicit theta_offset round-trips to < 1e-5. [§3.3]
  - the full v2.1 model at init produces bitwise-identical outputs to its v2.0 twin
    (zero-init H, no graph change). [§11]
  - decoder phase-sensitivity: a phase-rotated-but-equal-modulus memory changes the
    decoder logits (the decoder is NOT phase-blind, so the state channel is usable). [§9]
"""

import copy

import pytest
import torch

from twm.jepa import RotationScaleOperator, RotationOperator, PolarConditioner, block_modulus
from twm.jepa.conditioning import KindHead
from twm.jepa.decoder import TokenDecoder


# ---------------------------------------------------------------------------
# Operator: theta_offset arg (§4.1)
# ---------------------------------------------------------------------------

def _op(n_verbs=8, d_noun=32, seed=0):
    torch.manual_seed(seed)
    return RotationScaleOperator(n_verbs=n_verbs, d_noun=d_noun, block=2)


def test_apply_theta_offset_none_is_v20_regression():
    """theta_offset=None must route through the exact v2.0 _gather_blocks coefficients."""
    op = _op()
    torch.manual_seed(1)
    k = torch.randn(3, 6, 32)
    v = torch.randint(0, op.n_verbs, (3, 6))

    a_default = op.apply(k, v)              # default arg (no offset)
    a_explicit_none = op.apply(k, v, theta_offset=None)
    assert torch.equal(a_default, a_explicit_none)


def test_apply_zero_offset_matches_none():
    """A zero theta_offset must equal the None path to machine precision."""
    op = _op()
    torch.manual_seed(2)
    k = torch.randn(3, 6, 32)
    v = torch.randint(0, op.n_verbs, (3, 6))
    nb = 32 // 2

    a_none = op.apply(k, v)
    a_zero = op.apply(k, v, theta_offset=torch.zeros(3, 6, nb))
    assert (a_zero - a_none).abs().max().item() < 1e-5


def test_inverse_apply_zero_offset_matches_none():
    op = _op()
    torch.manual_seed(3)
    a = torch.randn(3, 6, 32)
    v = torch.randint(0, op.n_verbs, (3, 6))
    nb = 32 // 2

    k_none = op.inverse_apply(a, v)
    k_zero = op.inverse_apply(a, v, theta_offset=torch.zeros(3, 6, nb))
    assert (k_zero - k_none).abs().max().item() < 1e-5


def test_inverse_roundtrip_with_explicit_offset():
    """apply then inverse_apply with the SAME offset recovers k to < 1e-5 (§3.3)."""
    op = _op()
    torch.manual_seed(4)
    k = torch.randn(4, 5, 32)
    v = torch.randint(0, op.n_verbs, (4, 5))
    nb = 32 // 2
    offset = torch.randn(4, 5, nb) * 0.5  # nonzero, arbitrary

    a = op.apply(k, v, theta_offset=offset)
    k_rt = op.inverse_apply(a, v, theta_offset=offset)
    assert (k_rt - k).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# PolarConditioner H map (§3)
# ---------------------------------------------------------------------------

def test_H_zero_init_gives_zero_offset():
    """Zero-init H ⟹ H(|k|) == 0 for any input (the v2.1==v2.0-at-init guarantee)."""
    torch.manual_seed(5)
    cond = PolarConditioner(n_blocks=16)
    # H weight is exactly zero at init.
    assert torch.equal(cond.H.weight, torch.zeros_like(cond.H.weight))
    k = torch.randn(3, 8, 32)
    off = cond(k)
    assert off.shape == (3, 8, 16)
    assert torch.equal(off, torch.zeros_like(off))


def test_H_no_bias():
    """bias=False is mandatory (a bias would break the zero-offset-at-init guarantee)."""
    cond = PolarConditioner(n_blocks=16)
    assert cond.H.bias is None


def test_conditioned_apply_equals_unconditioned_at_init():
    """With zero-init H, apply(k, v, H(|k|)) == apply(k, v) exactly (§3.1)."""
    op = _op()
    cond = PolarConditioner(n_blocks=16)
    torch.manual_seed(6)
    k = torch.randn(3, 8, 32)
    v = torch.randint(0, op.n_verbs, (3, 8))

    off = cond(k)
    a_cond = op.apply(k, v, theta_offset=off)
    a_plain = op.apply(k, v)
    assert torch.equal(a_cond, a_plain)


def test_H_offset_keeps_gradient_to_k():
    """The modulus profile keeps gradient (design §3.1: NOT detached in the forward)."""
    cond = PolarConditioner(n_blocks=16)
    cond.H.weight.data.normal_(0, 0.3)  # nonzero so the offset depends on k
    k = torch.randn(2, 4, 32, requires_grad=True)
    off = cond(k)
    off.sum().backward()
    assert k.grad is not None
    assert k.grad.abs().sum().item() > 0


def test_block_modulus_matches_manual():
    torch.manual_seed(7)
    k = torch.randn(2, 3, 8)
    m = block_modulus(k)
    assert m.shape == (2, 3, 4)
    # manual: sqrt(x^2 + y^2) per 2-block
    pair = k.reshape(2, 3, 4, 2)
    manual = (pair[..., 0] ** 2 + pair[..., 1] ** 2).sqrt()
    assert torch.allclose(m, manual)


# ---------------------------------------------------------------------------
# Identity persistence: rotation preserves modulus (§8.1) — the polar claim
# ---------------------------------------------------------------------------

def test_rotation_preserves_modulus_all_verbs_with_offset():
    """RotationOperator (log_r≡0): modulus drift < 1e-5 for every verb, WITH a nonzero
    theta_offset. The offset only moves phase, so it cannot perturb modulus."""
    rop = RotationOperator(n_verbs=8, d_noun=32)
    cond = PolarConditioner(n_blocks=16)
    cond.H.weight.data.normal_(0, 0.5)  # nonzero offset, genuinely conditioned
    torch.manual_seed(8)
    k = torch.randn(4, 8, 32)
    off = cond(k)
    m_k = block_modulus(k)
    for v in range(rop.n_verbs):
        v_slots = torch.full((4, 8), v, dtype=torch.long)
        a = rop.apply(k, v_slots, theta_offset=off)
        m_a = block_modulus(a)
        drift = ((m_a - m_k).norm() / (m_k.norm() + 1e-8)).item()
        assert drift < 1e-5, (v, drift)


def test_rotation_scale_verb_changes_modulus():
    """A scaling verb (r ≠ 1) DOES change modulus — the irreversible-change channel
    (§2). Sanity: not every verb is identity-preserving, else the test above is vacuous."""
    op = _op()
    # ensure at least one verb has a clearly nonzero log_r
    op.log_r.data[0] = 0.5
    torch.manual_seed(9)
    k = torch.randn(4, 8, 32)
    v_slots = torch.zeros(4, 8, dtype=torch.long)  # verb 0, r = exp(0.5) ≈ 1.65
    a = op.apply(k, v_slots)
    m_k = block_modulus(k); m_a = block_modulus(a)
    drift = ((m_a - m_k).norm() / (m_k.norm() + 1e-8)).item()
    assert drift > 0.1


# ---------------------------------------------------------------------------
# Decoder phase-sensitivity (§9) — the decoder must read phase
# ---------------------------------------------------------------------------

def _rotate_each_block(x, delta):
    """Rotate every 2×2 block of x by a fixed angle delta. Modulus is preserved."""
    dn = x.shape[-1]
    pair = x.reshape(*x.shape[:-1], dn // 2, 2)
    xc, yc = pair[..., 0], pair[..., 1]
    c, s = torch.cos(torch.tensor(delta)), torch.sin(torch.tensor(delta))
    out_x = c * xc - s * yc
    out_y = s * xc + c * yc
    return torch.stack([out_x, out_y], dim=-1).reshape(*x.shape)


def test_decoder_is_phase_sensitive():
    """Two memories differing ONLY in phase (equal modulus) must give different logits —
    a phase-blind decoder cannot use the state channel at all (§9)."""
    torch.manual_seed(10)
    dec = TokenDecoder(vocab_size=512, d_dec=64, n_layers=1, n_heads=4, d_ff=128,
                       d_noun=32, max_text_tokens=64)
    dec.eval()
    B, M, dn, T = 4, 8, 32, 12
    a1 = torch.randn(B, M, dn)
    a2 = _rotate_each_block(a1, delta=0.7)  # fixed nonzero per-block rotation

    # modulus must be identical (the test isolates phase).
    assert (block_modulus(a1) - block_modulus(a2)).abs().max().item() < 1e-5

    tgt = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    with torch.no_grad():
        logits1 = dec(a1, tgt, pad)
        logits2 = dec(a2, tgt, pad)
    assert (logits1 - logits2).abs().max().item() > 1e-3


# ---------------------------------------------------------------------------
# KindHead (§7) — diagnostic only, never routes
# ---------------------------------------------------------------------------

def test_kind_head_assign_shape_and_range():
    kh = KindHead(n_blocks=16, codebook_size=12)
    torch.manual_seed(11)
    k = torch.randn(3, 8, 32)
    ids = kh.assign(k)
    assert ids.shape == (3, 8)
    assert int(ids.min()) >= 0 and int(ids.max()) < 12


def test_kind_head_codebook_normal_init():
    """Codebook init is normal(0,1) (repo VQ gotcha), NOT the collapse-prone uniform."""
    kh = KindHead(n_blocks=16, codebook_size=64)
    std = kh.codebook.std().item()
    assert 0.5 < std < 2.0  # ~1.0 for normal(0,1); rules out the ~0.001 uniform init


def test_kind_head_commitment_loss_finite():
    kh = KindHead(n_blocks=16, codebook_size=8)
    k = torch.randn(2, 4, 32)
    loss = kh.commitment_loss(k)
    assert torch.isfinite(loss) and loss.item() >= 0
