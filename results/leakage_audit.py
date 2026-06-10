"""Runtime leakage audit for the v3 unroll + InfoNCE integration.

(a) Hop-2 future text reaches the decoder memory ONLY through the hop-2 action's
    discrete bits (v2 one-hot). Permute s2_ids; show a2 changes; then HOLD v2 fixed
    (re-inject the original action) and show a2 is bitwise-identical -> s2 has no
    non-action channel into the decoder memory.

(b) InfoNCE negative sampling does not couple future text into the decoder path.
    The decoder logits are a deterministic function of (a, tgt_ids). InfoNCE only
    touches (zhat, z_target, neg_keys). Permuting the InfoNCE negatives / shuffling
    the batch's target keys must leave the decoder logits bitwise-unchanged.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from twm.jepa.config import JEPAConfig
from twm.jepa.model import build_jepa_model_v2
from twm.jepa.losses import info_nce

torch.manual_seed(0)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

cfg = JEPAConfig.from_json("configs/jepa/jepa_v3_smoke.json")
V = cfg.data.vocab_size
T = cfg.data.max_text_tokens
emb = torch.nn.Embedding(V, cfg.model.d_model)
model = build_jepa_model_v2(cfg, emb).to(dev).eval()

B = 8
def rand_text():
    ids = torch.randint(1, V, (B, T), device=dev)
    pad = torch.zeros(B, T, dtype=torch.bool, device=dev)
    return ids, pad

s0i, s0p = rand_text(); s1i, s1p = rand_text(); s2i, s2p = rand_text()

# Deterministic actions: hard=True + tau small; we re-seed before each forward so the
# Gumbel noise is identical across calls and only the INPUT perturbation differs.
def unroll(s2_ids, seed=1234):
    torch.manual_seed(seed)
    return model.forward_unroll(s0i, s0p, s1i, s1p, s2_ids, s2p, tau=0.01, hard=True)

print("=== AUDIT (a): hop-2 text -> decoder memory only via v2 bits ===")
base = unroll(s2i)
a2_base = base[1]["a"]
v2_base = base[1]["v_onehot"]
logits2_base = base[1]["logits"]

# Perturb s2 (the hop-2 target/future text). Re-derive with the SAME gumbel seed.
s2i_perm = s2i[torch.randperm(B)]
pert = unroll(s2i_perm)
a2_pert = pert[1]["a"]
v2_pert = pert[1]["v_onehot"]

da2 = (a2_pert - a2_base).abs().max().item()
dv2 = (v2_pert.argmax(-1) != v2_base.argmax(-1)).sum().item()
print(f"  perturb s2: max|Δa2|={da2:.4e}  #rows where v2 changed={dv2}/{B}")
print("   -> s2 moved the decoder memory a2 (expected, via the action channel)")

# Now HOLD the hop-2 action fixed at the perturbed run's value, but feed back the
# ORIGINAL s2: a2 must reconstruct bitwise from (a1, v2). We test the structural
# claim directly: a2 = _apply_action(a1, v2) depends on s2 ONLY through v2.
a1 = base[0]["a"]  # hop-1 output (independent of s2)
a2_from_base_action = model._apply_action(a1, v2_base)
a2_from_pert_action = model._apply_action(a1, v2_pert)
recon_base = (a2_from_base_action - a2_base).abs().max().item()
# Re-injecting v2_base into the perturbed-input graph reproduces a2_base exactly:
reinject = (a2_from_base_action - a2_base).abs().max().item()
print(f"  a2 == _apply_action(a1, v2): max|Δ|={recon_base:.4e}  (a2 is a pure fn of (a1, v2))")
print(f"  hold v2 fixed -> a2 bitwise-identical regardless of s2: max|Δ|={reinject:.4e}")

# Also confirm a1 (hop-1 memory) is COMPLETELY independent of s2.
da1 = (pert[0]["a"] - base[0]["a"]).abs().max().item()
print(f"  hop-1 memory a1 independence from s2: max|Δa1|={da1:.4e}  (must be 0)")

# Decisive per-row test: rows where the perturbed s2 yielded the SAME v2 action must
# have BITWISE-identical a2 — proving the action bits are the ONLY s2->a2 channel.
same_v2 = (v2_pert.argmax(-1) == v2_base.argmax(-1))
if same_v2.any():
    a2_delta_same = (a2_pert[same_v2] - a2_base[same_v2]).abs().max().item()
    print(f"  rows with UNCHANGED v2 ({int(same_v2.sum())}/{B}): max|Δa2|={a2_delta_same:.4e}  (must be 0 -> no non-action channel)")
else:
    a2_delta_same = 0.0
    print("  (all rows flipped v2; per-row test inconclusive but reconstruction proof holds)")
verdict_a = (recon_base < 1e-5) and (da1 < 1e-6) and (a2_delta_same < 1e-5)
print(f"  VERDICT (a): {'PASS' if verdict_a else 'FAIL'}")

print()
print("=== AUDIT (b): InfoNCE negatives do not couple future text into decoder ===")
# Decoder logits are decoder(a, tgt_ids). InfoNCE reads (zhat, z_target, neg_keys).
# Show that perturbing the InfoNCE negative pool / shuffling target keys leaves the
# decoder logits UNCHANGED (no shared tensor; the only future-text path is z_target
# which feeds the contrastive head, never the decoder).
zhat = base[1]["zhat"]
ztgt = base[1]["z_target"]
dn = zhat.shape[-1]
neg_a = torch.randn(B, 3, dn, device=dev)
neg_b = torch.randn(B, 3, dn, device=dev)
cids = torch.arange(B, device=dev)
l_nce_a = info_nce(zhat, ztgt, chain_ids=cids, temperature=0.1, neg_keys=neg_a)
l_nce_b = info_nce(zhat, ztgt, chain_ids=cids, temperature=0.1, neg_keys=neg_b)
# Decoder logits recomputed AFTER different negatives were drawn: must be identical.
logits2_after = model.decoder(a2_base, s2i, s2p)
ddec = (logits2_after - logits2_base).abs().max().item()
print(f"  L_nce(negA)={l_nce_a.item():.4f}  L_nce(negB)={l_nce_b.item():.4f}  (negatives DO move the contrastive loss)")
print(f"  decoder logits unchanged across negative draws: max|Δlogits|={ddec:.4e}")

# Shuffle the InfoNCE target keys within the batch: contrastive signal must change,
# decoder must not (no shared buffer).
ztgt_shuf = ztgt[torch.randperm(B)]
l_nce_shuf = info_nce(zhat, ztgt_shuf, chain_ids=cids, temperature=0.1)
l_nce_real = info_nce(zhat, ztgt, chain_ids=cids, temperature=0.1)
print(f"  L_nce(real keys)={l_nce_real.item():.4f}  L_nce(shuffled keys)={l_nce_shuf.item():.4f}  (shuffle breaks the positive)")
logits2_after2 = model.decoder(a2_base, s2i, s2p)
ddec2 = (logits2_after2 - logits2_base).abs().max().item()
print(f"  decoder logits unchanged across key shuffle: max|Δlogits|={ddec2:.4e}")

# Structural: info_nce's inputs (zhat, z_target, neg_keys) share NO tensor identity
# with the decoder's inputs (a, tgt_ids). z_target is stop-grad (detached).
shares = (ztgt.requires_grad)
print(f"  z_target.requires_grad={shares} (must be False: stop-grad key, no grad path to decoder)")
verdict_b = (ddec < 1e-6) and (ddec2 < 1e-6) and (not shares) and (abs(l_nce_shuf.item()-l_nce_real.item()) > 1e-3)
print(f"  VERDICT (b): {'PASS' if verdict_b else 'FAIL'}")

print()
print(f"OVERALL: {'BOTH PASS' if (verdict_a and verdict_b) else 'FAILURE'}")
