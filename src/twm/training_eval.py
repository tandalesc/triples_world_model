"""Unified evaluation and diagnostics for text models.

Auto-detects IO vs dynamics mode from dataset type.
"""

import torch

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

        bottleneck = model.compress(input_ids, input_pad)
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)
        target_ids = output_ids
    else:
        input_ids = dataset._text_token_ids[:n].to(device)
        input_pad = dataset._text_pad_mask[:n].to(device)

        bottleneck = model.compress(input_ids, input_pad)
        target_ids = input_ids
        mode_ids = None

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
    id_tok = id_total = qa_tok = qa_total = 0

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
            else:
                qa_tok += matches
                qa_total += tgt_len

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

    mode_names = {0: "ID", 1: "QA"}
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


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a log string."""
    parts = [f"tok={metrics['tok_acc']:.3f}", f"exact={metrics['exact']:.3f}",
             f"len={metrics.get('len_acc', 0):.3f}"]
    if "tok_id" in metrics:
        parts.append(f"id={metrics['tok_id']:.3f}")
    if "tok_qa" in metrics:
        parts.append(f"qa={metrics['tok_qa']:.3f}")
    return " ".join(parts)
