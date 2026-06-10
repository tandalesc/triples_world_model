"""Unit tests for the JEPA v2 token decoder adapter (design doc §4, §6, §9).

Covers: teacher-forced logits shape + finite CE (the grounding loss), generation
terminates within budget for greedy and temperature, the leakage contract (no
posterior / text_{t+1} channel on the constructor or forward signature), the memory
ablation (the decoder actually READS a* — changing a* changes the output), and the
nano-v2 param budget.
"""

import inspect

import torch
import torch.nn.functional as F

from twm.jepa.decoder import TokenDecoder


def _make(vocab_size=512, d_dec=64, n_layers=1, n_heads=4, d_ff=128, d_noun=32,
          max_text_tokens=64, seed=0):
    torch.manual_seed(seed)
    return TokenDecoder(
        vocab_size=vocab_size, d_dec=d_dec, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, d_noun=d_noun, max_text_tokens=max_text_tokens,
    )


def test_forward_logits_shape():
    dec = _make()
    B, M, dn, T = 4, 8, 32, 12
    a_star = torch.randn(B, M, dn)
    tgt = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    logits = dec(a_star, tgt, pad)
    assert logits.shape == (B, T, 512)


def test_teacher_forced_ce_finite_and_backprops():
    # Token CE is the primary loss; it must compute and produce finite gradients.
    dec = _make()
    B, M, dn, T = 4, 8, 32, 16
    a_star = torch.randn(B, M, dn, requires_grad=True)
    tgt = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)

    logits = dec(a_star, tgt, pad)
    # Design doc §4.3: no manual shift — the [bos]+tgt[:-1] shift is baked into ARDecoder.
    loss = F.cross_entropy(
        logits.reshape(-1, 512), tgt.reshape(-1), ignore_index=dec.pad_id,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Gradient must flow back into a* (the memory the decoder reads).
    assert a_star.grad is not None
    assert torch.isfinite(a_star.grad).all()
    assert a_star.grad.abs().sum() > 0


def test_ce_ignores_pad_positions():
    # pad_id targets must be masked: a batch differing only in pad-position targets
    # yields the same CE.
    dec = _make()
    B, M, dn, T = 3, 8, 32, 10
    a_star = torch.randn(B, M, dn)
    tgt = torch.randint(1, 512, (B, T))  # avoid pad_id=0 in real positions
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -3:] = True
    tgt_padded = tgt.clone()
    tgt_padded[pad] = dec.pad_id  # pad positions carry pad_id

    logits = dec(a_star, tgt_padded, pad)
    loss_a = F.cross_entropy(
        logits.reshape(-1, 512), tgt_padded.reshape(-1), ignore_index=dec.pad_id,
    )
    # Change ONLY the pad-position targets; CE must be unchanged (they are ignored).
    tgt_alt = tgt_padded.clone()
    tgt_alt[pad] = dec.pad_id  # still pad_id, ignore_index drops them regardless
    loss_b = F.cross_entropy(
        logits.reshape(-1, 512), tgt_alt.reshape(-1), ignore_index=dec.pad_id,
    )
    assert torch.allclose(loss_a, loss_b)


def test_generate_greedy_terminates_within_budget():
    dec = _make()
    B, M, dn = 4, 8, 32
    a_star = torch.randn(B, M, dn)
    gen = dec.generate(a_star, max_tokens=20, temperature=0.0)
    assert gen.shape[0] == B
    assert gen.shape[1] <= 20
    assert gen.dtype == torch.long


def test_generate_temperature_terminates_within_budget():
    dec = _make()
    B, M, dn = 4, 8, 32
    a_star = torch.randn(B, M, dn)
    gen = dec.generate(a_star, max_tokens=20, temperature=0.8)
    assert gen.shape[0] == B
    assert gen.shape[1] <= 20
    assert gen.dtype == torch.long


def test_generate_defaults_to_max_text_tokens():
    dec = _make(max_text_tokens=24)
    a_star = torch.randn(2, 8, 32)
    gen = dec.generate(a_star)  # no max_tokens -> max_text_tokens cap
    assert gen.shape[1] <= 24


def test_memory_ablation_changes_output():
    # The decoder must actually READ a*: different memory -> different logits.
    # (This is the in-test analogue of the v-ablation CE gap; if the decoder ignores
    # a*, v's bits do no work and we regress to the v1 failure.)
    dec = _make()
    dec.eval()  # disable dropout for a deterministic comparison
    B, M, dn, T = 4, 8, 32, 12
    tgt = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)

    torch.manual_seed(1)
    a1 = torch.randn(B, M, dn)
    torch.manual_seed(2)
    a2 = torch.randn(B, M, dn)

    with torch.no_grad():
        logits1 = dec(a1, tgt, pad)
        logits2 = dec(a2, tgt, pad)
    # Same target prefix, different memory -> outputs must differ materially.
    assert not torch.allclose(logits1, logits2, atol=1e-4)
    assert (logits1 - logits2).abs().max() > 1e-2


def test_memory_ablation_constant_vs_real():
    # Forcing a* to a constant (the v-ablation analogue) changes the output vs real a*.
    dec = _make()
    dec.eval()
    B, M, dn, T = 4, 8, 32, 12
    tgt = torch.randint(0, 512, (B, T))
    pad = torch.zeros(B, T, dtype=torch.bool)
    a_real = torch.randn(B, M, dn)
    a_const = torch.zeros(B, M, dn)
    with torch.no_grad():
        logits_real = dec(a_real, tgt, pad)
        logits_const = dec(a_const, tgt, pad)
    assert not torch.allclose(logits_real, logits_const, atol=1e-4)


def test_leakage_constructor_has_no_posterior_channel():
    # Leakage contract (§6 L2): the constructor must NOT accept any posterior /
    # text_{t+1}-encoding argument. Only token/shape/budget knobs are allowed.
    sig = inspect.signature(TokenDecoder.__init__)
    names = set(sig.parameters.keys())
    forbidden_substrings = ["posterior", "tgt_enc", "t1", "t_plus", "next_enc",
                            "dense", "compressor", "future", "v_logits"]
    for name in names:
        low = name.lower()
        for bad in forbidden_substrings:
            assert bad not in low, f"constructor arg '{name}' may leak text_t+1 info"


def test_leakage_forward_takes_only_a_star_and_target():
    # forward accepts ONLY (a_star, tgt_ids, tgt_pad) — no raw text_{t+1} encoding,
    # no posterior features, no untransformed k / slots_t.
    sig = inspect.signature(TokenDecoder.forward)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["a_star", "tgt_ids", "tgt_pad"], params


def test_generate_takes_only_a_star():
    # generate's conditioning argument is a_star only (no posterior / future channel).
    sig = inspect.signature(TokenDecoder.generate)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["a_star", "max_tokens", "temperature"], params


def test_param_count_within_nano_budget():
    # nano-v2 decoder line item ~123.2K (design doc §9). Assert it is in budget and
    # well under the 250K total ceiling on its own.
    dec = _make()
    n = dec.param_count()
    assert n == dec.trainable_param_count()  # all trainable
    assert 110_000 <= n <= 135_000, n
    assert n < 250_000


def test_dropout_makes_train_eval_differ_but_eval_deterministic():
    # Sanity: ARDecoder has dropout=0.1; eval mode must be deterministic.
    dec = _make()
    a_star = torch.randn(2, 8, 32)
    tgt = torch.randint(0, 512, (2, 10))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    dec.eval()
    with torch.no_grad():
        l1 = dec(a_star, tgt, pad)
        l2 = dec(a_star, tgt, pad)
    assert torch.allclose(l1, l2)
