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
def assess(model, dataset, device, tokenizer, n_examples=64, n_steps=10) -> dict:
    """Evaluate model on dataset, auto-detecting IO vs dynamics mode.

    Returns dict with: tok_acc, exact, len_acc, and optionally tok_id, tok_qa.
    """
    model.eval()
    n = min(n_examples, len(dataset))
    pad_id = tokenizer.pad_token_id

    if _is_pair_dataset(dataset):
        input_ids = dataset._input_token_ids[:n].to(device)
        input_pad = dataset._input_pad_mask[:n].to(device)
        output_ids = dataset._output_token_ids[:n].to(device)
        mode_ids = dataset._modes[:n].to(device)

        bottleneck = model.compress(input_ids, input_pad)
        bottleneck = model.forward_dynamics(bottleneck, mode_ids)
        target_ids = output_ids
    else:
        text_ids = dataset._text_token_ids[:n].to(device)
        text_pad = dataset._text_pad_mask[:n].to(device)

        bottleneck = model.compress(text_ids, text_pad)
        target_ids = text_ids
        mode_ids = None

    gen_ids = model.generate(bottleneck, n_steps=n_steps)
    pred_lens = model.forward_length(bottleneck)
    pred_lens = pred_lens.round().long().clamp(1, gen_ids.shape[-1])

    tok_match = total_tok = exact_count = len_match = 0
    id_tok = id_total = qa_tok = qa_total = 0

    for i in range(n):
        tgt = [x for x in target_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        pred = gen_ids[i].cpu().tolist()[:pl]

        if pl == len(tgt):
            len_match += 1
        cmp_len = min(pl, len(tgt))
        matches = sum(1 for a, b in zip(pred[:cmp_len], tgt[:cmp_len]) if a == b)
        tok_match += matches
        total_tok += len(tgt)

        if mode_ids is not None:
            mode = mode_ids[i].item()
            if mode == 0:
                id_tok += matches
                id_total += len(tgt)
            else:
                qa_tok += matches
                qa_total += len(tgt)

        if pred == tgt:
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
    return result


@torch.no_grad()
def print_samples(model, dataset, device, tokenizer, n=5, n_steps=10):
    """Print pred vs target samples, auto-detecting IO vs dynamics mode."""
    model.eval()
    n = min(n, len(dataset))
    pad_id = tokenizer.pad_token_id

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

    mode_names = {0: "ID", 1: "QA"}
    print(f"\n{'='*70}")
    for i in range(n):
        tgt_ids_list = [x for x in target_ids[i].tolist() if x != pad_id]
        pl = pred_lens[i].item()
        pred_text = _clean(tokenizer.decode(gen_ids[i, :pl].cpu(), skip_special_tokens=True))
        match = "Y" if gen_ids[i].cpu().tolist()[:pl] == tgt_ids_list else "N"
        len_info = f"len {pl}/{len(tgt_ids_list)}" + ("" if pl == len(tgt_ids_list) else " !")

        if mode_ids is not None:
            mode = mode_names.get(mode_ids[i].item(), "??")
            inp = _clean(tokenizer.decode(input_ids[i].cpu(), skip_special_tokens=True))
            tgt = _clean(tokenizer.decode(target_ids[i].cpu(), skip_special_tokens=True))
            print(f"  [{i}|{mode}] inp:  {inp}")
            print(f"         tgt:  {tgt}")
            print(f"         pred: {pred_text}  [{match}] {len_info}")
        else:
            tgt = _clean(tokenizer.decode(target_ids[i].cpu(), skip_special_tokens=True))
            print(f"  [{i}] tgt:  {tgt}")
            print(f"       pred: {pred_text}  [{match}] {len_info}")
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
