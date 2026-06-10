"""Tests for SlotEncoder (T2). Spec: research/jepa_operator_v1_design.md §2, §12 row T2.

Covers:
  * forward shapes: slots (B,M,d), k (B,M,dn), verb_logits (B,M,V)
  * NounHead standardizes (zero-mean / ~unit-var per dim), NOT L2-normalized
  * ALBERT tying actually shares one block (param count drops vs untied)
  * slot coordination is wired (3 shared iters)
  * frozen token_emb is excluded from the trainable budget
  * nano param-count lands within 5% of the §2 table SlotEncoder share (~74.1K),
    with a printed per-module breakdown
"""

import torch
import torch.nn as nn

from twm.jepa.slot_encoder import SlotEncoder, NounHead, VerbHead


# nano profile (spec §2): d=64, dn=32, M=8, V=8, L_text=2 tied, heads=4,
# d_ff=128, vocab=512, T_text=64.
NANO = dict(
    d_model=64,
    d_noun=32,
    n_slots=8,
    n_verbs=8,
    n_text_layers=2,
    tie_text_layers=True,
    n_heads=4,
    d_ff=128,
    n_slot_iters=3,
    max_text_tokens=64,
)
VOCAB = 512

# Section-2 table, SlotEncoder share (trainable, non-embedding):
#   text_pos_emb        4,096
#   text self-attn     33,152  (ALBERT-tied, counted once)
#   slot queries+init   1,536
#   cross-attn         16,384
#   slot coordination  16,384
#   NounHead            2,048
#   VerbHead              512
#   ----------------- -------
#   total              74,112
# (The remaining ~2.6K of the ~76.7K nano total — operator codebook 256,
#  readout 4,128, predictor 2,048 — are owned by T1/T5, not SlotEncoder.)
SECTION2_SLOT_ENCODER_SHARE = 74_112


def _make(profile=NANO, freeze_emb=True, **overrides):
    cfg = {**profile, **overrides}
    emb = nn.Embedding(VOCAB, cfg["d_model"])
    if freeze_emb:
        emb.weight.requires_grad_(False)
    return SlotEncoder(emb, **cfg)


def test_forward_shapes():
    enc = _make()
    B, T = 4, NANO["max_text_tokens"]
    text_ids = torch.randint(0, VOCAB, (B, T))
    text_pad = torch.zeros(B, T, dtype=torch.bool)
    text_pad[:, T // 2:] = True  # second half is padding

    slots, k, verb_logits = enc(text_ids, text_pad)
    assert slots.shape == (B, NANO["n_slots"], NANO["d_model"])
    assert k.shape == (B, NANO["n_slots"], NANO["d_noun"])
    assert verb_logits.shape == (B, NANO["n_slots"], NANO["n_verbs"])
    assert torch.isfinite(slots).all()
    assert torch.isfinite(k).all()
    assert torch.isfinite(verb_logits).all()


def test_noun_head_standardizes_not_l2():
    """k should be ~zero-mean / unit-var per dim over the batch, and NOT unit-norm."""
    enc = _make()
    B, T = 32, NANO["max_text_tokens"]
    text_ids = torch.randint(0, VOCAB, (B, T))
    text_pad = torch.zeros(B, T, dtype=torch.bool)
    _, k, _ = enc(text_ids, text_pad)

    flat = k.reshape(-1, k.shape[-1])  # (B*M, dn)
    # Standardized: per-dim mean ~0, per-dim std ~1.
    assert flat.mean(dim=0).abs().max().item() < 1e-3
    assert (flat.std(dim=0, unbiased=False) - 1.0).abs().max().item() < 1e-2

    # NOT L2-normalized: per-vector norms are not all ~1.
    norms = flat.norm(dim=-1)
    assert (norms - 1.0).abs().mean().item() > 0.1


def test_token_emb_frozen_excluded():
    enc = _make(freeze_emb=True)
    # token_emb params must not be in the trainable budget.
    assert not enc.token_emb.weight.requires_grad
    with_emb = enc.trainable_param_count(include_embedding=True)
    without_emb = enc.trainable_param_count(include_embedding=False)
    # token_emb is frozen, so include_embedding=True still excludes it via requires_grad,
    # but the name-filter path must also exclude it.
    assert with_emb == without_emb  # frozen emb contributes 0 either way


def test_albert_tying_shares_block():
    """Tied L_text=2 must have FEWER params than untied L_text=2."""
    tied = _make(tie_text_layers=True)
    untied = _make(tie_text_layers=False)
    t = sum(p.numel() for b in tied.text_blocks for p in b.parameters())
    u = sum(p.numel() for b in untied.text_blocks for p in b.parameters())
    assert len(tied.text_blocks) == 1
    assert len(untied.text_blocks) == NANO["n_text_layers"]
    assert u == NANO["n_text_layers"] * t


def test_slot_coordination_wired():
    enc = _make()
    assert enc.n_slot_iters == 3
    # One shared coordination block reused across iters.
    assert hasattr(enc, "coord_block")


def test_nano_param_count_within_5pct_and_print_breakdown():
    enc = _make()
    breakdown = enc.param_breakdown(include_embedding=True)
    total_nonemb = enc.trainable_param_count(include_embedding=False)

    print("\n=== SlotEncoder nano per-module trainable param breakdown ===")
    width = max(len(k) for k in breakdown)
    for name, n in breakdown.items():
        print(f"  {name:<{width}}  {n:>8,}")
    print(f"  {'-' * width}  {'-' * 8}")
    print(f"  {'TOTAL (non-embedding)':<{width}}  {total_nonemb:>8,}")
    print(f"  {'§2 table SlotEncoder share':<{width}}  {SECTION2_SLOT_ENCODER_SHARE:>8,}")
    rel = abs(total_nonemb - SECTION2_SLOT_ENCODER_SHARE) / SECTION2_SLOT_ENCODER_SHARE
    print(f"  relative error vs table: {rel * 100:.2f}%")

    # Sanity: breakdown sums to the non-embedding total.
    summed = sum(v for kk, v in breakdown.items() if "frozen" not in kk)
    assert summed == total_nonemb

    assert rel <= 0.05, (
        f"SlotEncoder nano param count {total_nonemb:,} is "
        f"{rel * 100:.2f}% off the §2 table share {SECTION2_SLOT_ENCODER_SHARE:,} "
        f"(>5%)."
    )


def test_heads_standalone():
    nh = NounHead(64, 32)
    vh = VerbHead(64, 8)
    slots = torch.randn(3, 8, 64)
    k = nh(slots)
    vl = vh(slots)
    assert k.shape == (3, 8, 32)
    assert vl.shape == (3, 8, 8)
    # NounHead has no L2 normalization parameter and standardize is parameter-free.
    assert sum(p.numel() for p in nh.parameters()) == 64 * 32 + 32  # weight + bias
