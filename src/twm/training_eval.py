"""Unified evaluation and diagnostics for text models.

Auto-detects IO vs dynamics mode from dataset type.
"""

from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .training_losses import _clean
from .text_dataset import TextDataset
from .text_pair_dataset import TextPairDataset


def _is_pair_dataset(ds) -> bool:
    return isinstance(ds, TextPairDataset)


@torch.no_grad()
def _generate(model, dataset, device, n, n_steps):
    """Run compression + generation, returning all intermediates.

    Returns (target_ids, gen_ids, pred_lens, mode_ids, input_ids).
    """
    if _is_pair_dataset(dataset):
        input_ids = dataset._input_token_ids[:n].to(device)
        input_pad = dataset._input_pad_mask[:n].to(device)
        output_ids = dataset._output_token_ids[:n].to(device)
        mode_ids = dataset._modes[:n].to(device)

        compress_out = model.compress(input_ids, input_pad)
        bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)
        target_ids = output_ids
    else:
        input_ids = dataset._text_token_ids[:n].to(device)
        input_pad = dataset._text_pad_mask[:n].to(device)

        compress_out = model.compress(input_ids, input_pad)
        bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out
        # Joint IO training routes identity data through dynamics with mode=0.
        # Eval must match: if the model has dynamics, run it so the length head
        # and expander see the same post-dynamics bottleneck they trained on.
        if hasattr(model, "forward_dynamics"):
            mode_ids = torch.zeros(n, dtype=torch.long, device=device)
            bottleneck = model.forward_dynamics(bottleneck, mode_ids)
        else:
            mode_ids = None
        target_ids = input_ids

    gen_ids = model.generate(bottleneck, n_steps=n_steps)
    pred_lens = model.forward_length(bottleneck)
    pred_lens = pred_lens.round().long().clamp(1, gen_ids.shape[-1])

    return target_ids, gen_ids, pred_lens, mode_ids, input_ids


@torch.no_grad()
def assess(model, dataset, device, tokenizer, n_examples=64, n_steps=10) -> dict:
    """Evaluate model on dataset, auto-detecting IO vs dynamics mode.

    Returns dict with: tok_acc, exact, len_acc, and optionally tok_id, tok_qa.
    Also stores generation results in returned dict under '_gen' for reuse
    by print_samples.
    """
    model.eval()
    n = min(n_examples, len(dataset))
    pad_id = tokenizer.pad_token_id

    target_ids, gen_ids, pred_lens, mode_ids, input_ids = _generate(
        model, dataset, device, n, n_steps
    )

    tok_match = total_tok = exact_count = len_match = 0
    id_tok = id_total = qa_tok = qa_total = rev_tok = rev_total = 0

    for i in range(n):
        tgt = [x for x in target_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        tgt_len = len(tgt)

        # Length match
        if pl == tgt_len:
            len_match += 1

        # Compare at the length the model committed to (predicted length)
        # Tokens beyond pl are denoising noise — ignore them.
        # Tokens beyond tgt_len don't exist — can't match.
        cmp_len = min(pl, tgt_len)
        pred = gen_ids[i].cpu().tolist()[:cmp_len]
        matches = sum(1 for a, b in zip(pred, tgt[:cmp_len]) if a == b)
        tok_match += matches
        total_tok += tgt_len

        if mode_ids is not None:
            mode = mode_ids[i].item()
            if mode == 0:
                id_tok += matches
                id_total += tgt_len
            elif mode == 1:
                qa_tok += matches
                qa_total += tgt_len
            elif mode == 2:
                rev_tok += matches
                rev_total += tgt_len

        # Exact match requires correct length AND all tokens match
        if pl == tgt_len and pred == tgt:
            exact_count += 1

    result = {
        "tok_acc": tok_match / max(total_tok, 1),
        "exact": exact_count / n,
        "len_acc": len_match / n,
    }
    if id_total > 0:
        result["tok_id"] = id_tok / id_total
    if qa_total > 0:
        result["tok_qa"] = qa_tok / qa_total
    if rev_total > 0:
        result["tok_rev"] = rev_tok / rev_total

    # Stash generation results for print_samples to reuse
    result["_gen"] = (target_ids, gen_ids, pred_lens, mode_ids, input_ids)
    return result


@torch.no_grad()
def print_samples(model, dataset, device, tokenizer, n=5, n_steps=10,
                  gen_cache=None):
    """Print pred vs target samples, auto-detecting IO vs dynamics mode.

    Args:
        gen_cache: optional tuple from assess()'s '_gen' key to reuse
                   the same generation instead of re-generating with new noise.
    """
    model.eval()
    n = min(n, len(dataset))
    pad_id = tokenizer.pad_token_id

    if gen_cache is not None:
        target_ids, gen_ids, pred_lens, mode_ids, input_ids = gen_cache
        # Slice to n samples
        target_ids = target_ids[:n]
        gen_ids = gen_ids[:n]
        pred_lens = pred_lens[:n]
        if mode_ids is not None:
            mode_ids = mode_ids[:n]
        input_ids = input_ids[:n]
    else:
        target_ids, gen_ids, pred_lens, mode_ids, input_ids = _generate(
            model, dataset, device, n, n_steps
        )

    mode_names = {0: "ID", 1: "QA", 2: "REV"}
    print(f"\n{'='*70}")
    for i in range(n):
        tgt_ids_list = [x for x in target_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        tgt_len = len(tgt_ids_list)

        # Truncate prediction to length head's commitment
        cmp_len = min(pl, tgt_len)
        pred_ids_list = gen_ids[i].cpu().tolist()[:cmp_len]
        match = "Y" if pl == tgt_len and pred_ids_list == tgt_ids_list else "N"
        # Display only tokens within predicted length — anything beyond is noise
        pred_text = _clean(tokenizer.decode(gen_ids[i, :pl].cpu(), skip_special_tokens=True))
        len_info = f"len {pl}/{tgt_len}" + ("" if pl == tgt_len else " !")

        # Find first mismatch position
        mismatch = ""
        if match == "N":
            if pl != tgt_len:
                mismatch = f" (len mismatch)"
            else:
                for j in range(cmp_len):
                    if gen_ids[i, j].item() != tgt_ids_list[j]:
                        wrong = _clean(tokenizer.decode([gen_ids[i, j].item()]))
                        right = _clean(tokenizer.decode([tgt_ids_list[j]]))
                        mismatch = f" (pos {j}: '{wrong}'!='{right}')"
                        break

        if mode_ids is not None:
            mode = mode_names.get(mode_ids[i].item(), "??")
            inp = _clean(tokenizer.decode(input_ids[i].cpu(), skip_special_tokens=True))
            tgt = _clean(tokenizer.decode(target_ids[i].cpu(), skip_special_tokens=True))
            print(f"  [{i}|{mode}] inp:  {inp}")
            print(f"         tgt:  {tgt}")
            print(f"         pred: {pred_text}  [{match}] {len_info}{mismatch}")
        else:
            tgt = _clean(tokenizer.decode(target_ids[i].cpu(), skip_special_tokens=True))
            print(f"  [{i}] tgt:  {tgt}")
            print(f"       pred: {pred_text}  [{match}] {len_info}{mismatch}")
    print(f"{'='*70}", flush=True)


@torch.no_grad()
def diagnose_mode_attention(model, dataset, device, n_examples=64):
    """Analyze dynamics core attention to mode triple, split by mode.

    Reports per-layer, per-head mean attention from data positions to mode
    positions, separately for each mode. Mode-reading circuitry shows as
    mode-dependent attention patterns across heads.

    The mode triple occupies positions 0-2 in the dynamics input.
    Data triples start at position 3.
    """
    model.eval()
    n = min(n_examples, len(dataset))

    input_ids = dataset._input_token_ids[:n].to(device)
    input_pad = dataset._input_pad_mask[:n].to(device)
    mode_ids = dataset._modes[:n].to(device)

    compress_out = model.compress(input_ids, input_pad)
    bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out

    # Build dynamics input with mode triple (same as forward_dynamics)
    mode_triple = model._build_mode_triple(mode_ids)  # (B, 3, d)
    x = torch.cat([mode_triple, bottleneck], dim=1)  # (B, 3+N*3, d)

    # Extract per-layer attention weights
    attn_weights = model.dynamics.extract_attention_weights(x)
    # Each: (B, n_heads, T, T) — already on CPU

    unique_modes = mode_ids.unique().cpu()
    mode_labels = {0: "identity", 1: "qa", 2: "reverse"}

    print(f"\n{'='*60}")
    print("Mode-Attention Diagnostic (data→mode attention)")
    print(f"{'='*60}")

    for li, layer_attn in enumerate(attn_weights):
        # layer_attn: (B, n_heads, T, T)
        # Attention from data positions (rows 3+) to mode positions (cols 0:3)
        data_to_mode = layer_attn[:, :, 3:, :3].mean(dim=(2, 3))  # (B, n_heads)

        print(f"\nLayer {li}:")
        mode_means = {}
        for mode_val in unique_modes:
            mask = (mode_ids.cpu() == mode_val)
            mode_name = mode_labels.get(mode_val.item(), f"mode_{mode_val}")
            mode_attn = data_to_mode[mask].mean(dim=0)  # (n_heads,)
            mode_means[mode_val.item()] = mode_attn
            head_strs = " ".join(f"h{h}={v:.4f}" for h, v in enumerate(mode_attn))
            print(f"  {mode_name:>10}: {head_strs}")

        # Differential between modes (if exactly 2 modes)
        if len(unique_modes) == 2:
            m0, m1 = unique_modes.tolist()
            diff = mode_means[m1] - mode_means[m0]
            diff_str = " ".join(f"h{h}={v:+.4f}" for h, v in enumerate(diff))
            max_diff = diff.abs().max().item()
            print(f"  {'diff':>10}: {diff_str}  (max |diff|={max_diff:.4f})")

    print(f"{'='*60}\n", flush=True)


@torch.no_grad()
def save_latent_snapshot(model, dataset, device, epoch, stage_name, out_dir,
                         pca_basis=None):
    """Save a PCA scatter plot of bottleneck geometry for video assembly.

    Args:
        pca_basis: PCA object from a previous call. If None, fits a new one.

    Returns:
        The PCA basis (reuse for consistent axes across frames).
    """
    model.eval()
    is_pair = _is_pair_dataset(dataset)
    n = min(128, len(dataset))

    # Compress examples to get bottleneck vectors
    if is_pair:
        input_ids = dataset._input_token_ids[:n].to(device)
        input_pad = dataset._input_pad_mask[:n].to(device)
        mode_ids = dataset._modes[:n]
    else:
        input_ids = dataset._text_token_ids[:n].to(device)
        input_pad = dataset._text_pad_mask[:n].to(device)
        mode_ids = None

    compress_out = model.compress(input_ids, input_pad)
    bottleneck = compress_out[0] if isinstance(compress_out, tuple) else compress_out
    # bottleneck: (B, max_triples*3, d_model)

    B, T, D = bottleneck.shape
    bn_np = bottleneck.cpu().numpy()

    # Mean-pool across position dim -> (B, d_model)
    pooled = bn_np.mean(axis=1)

    # Fit or reuse PCA basis
    if pca_basis is None:
        pca_basis = PCA(n_components=2)
        pca_basis.fit(pooled)
    coords = pca_basis.transform(pooled)  # (B, 2)

    # Per-role variance: separate PCA on entity/attr/value slot subsets
    # Slots repeat as [E, A, V, E, A, V, ...] across the T positions
    role_var = {}
    for role_idx, role_name in enumerate(["E", "A", "V"]):
        slot_data = bn_np[:, role_idx::3, :]  # (B, n_triples, D)
        slot_pooled = slot_data.mean(axis=1)   # (B, D)
        if slot_pooled.shape[0] > 1:
            role_pca = PCA(n_components=min(2, slot_pooled.shape[1]))
            role_pca.fit(slot_pooled)
            role_var[role_name] = role_pca.explained_variance_ratio_[0]
        else:
            role_var[role_name] = 0.0

    # Build role colors per sample: assign based on mean of each role's slots
    # For the right subplot, we plot each sample 3 times (once per role)
    role_coords = {}
    for role_idx, role_name in enumerate(["E", "A", "V"]):
        slot_pooled = bn_np[:, role_idx::3, :].mean(axis=1)  # (B, D)
        role_coords[role_name] = pca_basis.transform(slot_pooled)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: colored by mode
    mode_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    mode_labels = {0: "identity", 1: "qa", 2: "reverse"}
    if is_pair and mode_ids is not None:
        modes_np = mode_ids.numpy()
        for m_val in sorted(set(modes_np)):
            mask = modes_np == m_val
            ax1.scatter(coords[mask, 0], coords[mask, 1], c=mode_colors.get(m_val, "gray"),
                        label=mode_labels.get(m_val, f"mode_{m_val}"), s=12, alpha=0.6)
        ax1.legend(fontsize=8, loc="upper right")
    else:
        ax1.scatter(coords[:, 0], coords[:, 1], c="#1f77b4", s=12, alpha=0.6)
    ax1.set_title("Bottleneck by Mode", fontsize=10)
    ax1.set_xlabel("PC1", fontsize=8)
    ax1.set_ylabel("PC2", fontsize=8)
    ax1.text(0.02, 0.02, f"Stage: {stage_name}\nEpoch: {epoch:04d}",
             transform=ax1.transAxes, fontsize=7, verticalalignment="bottom")

    # Right: colored by role position
    role_colors_map = {"E": "#e41a1c", "A": "#377eb8", "V": "#4daf4a"}
    for role_name, rc in role_coords.items():
        ax2.scatter(rc[:, 0], rc[:, 1], c=role_colors_map[role_name],
                    label=role_name, s=12, alpha=0.6)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("Bottleneck by Role", fontsize=10)
    ax2.set_xlabel("PC1", fontsize=8)
    ax2.set_ylabel("PC2", fontsize=8)
    var_text = " ".join(f"{k}={v:.2f}" for k, v in role_var.items())
    ax2.text(0.02, 0.02, f"PC1 var: {var_text}",
             transform=ax2.transAxes, fontsize=7, verticalalignment="bottom")

    fig.tight_layout()

    # Save frame
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"{stage_name}_{epoch:04d}.png"
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)

    return pca_basis


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a log string."""
    parts = [f"tok={metrics['tok_acc']:.3f}", f"exact={metrics['exact']:.3f}",
             f"len={metrics.get('len_acc', 0):.3f}"]
    if "tok_id" in metrics:
        parts.append(f"id={metrics['tok_id']:.3f}")
    if "tok_qa" in metrics:
        parts.append(f"qa={metrics['tok_qa']:.3f}")
    if "tok_rev" in metrics:
        parts.append(f"rev={metrics['tok_rev']:.3f}")
    return " ".join(parts)
