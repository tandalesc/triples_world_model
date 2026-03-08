"""Set-based assessment metrics for Triple World Model."""

import argparse
import json
from pathlib import Path

import torch

from .vocab import Vocabulary
from .dataset import TripleTransitionDataset, collate_fn, _sort_triples
from .config import ModelConfig
from .model import TripleWorldModel


def _set_match(pred_triples: list[list[str]], tgt_triples: list[list[str]]) -> dict:
    """Compare predicted and target triple sets, order-independent.

    Returns per-example metrics:
        correct_triples: number of exact triple matches
        total_triples:   number of target triples
        correct_tokens:  token matches (via sorted alignment)
        total_tokens:    total non-pad tokens in target
        exact_match:     bool, predicted set == target set
        length_match:    bool, same number of triples
    """
    pred_set = set(tuple(t) for t in pred_triples)
    tgt_set = set(tuple(t) for t in tgt_triples)

    correct_triples = len(pred_set & tgt_set)
    exact_match = pred_set == tgt_set
    length_match = len(pred_triples) == len(tgt_triples)

    # Token accuracy via sorted alignment (best effort for partial credit)
    ps = sorted(pred_triples)
    ts = sorted(tgt_triples)
    correct_tokens = 0
    total_tokens = len(tgt_triples) * 3
    for i in range(min(len(ps), len(ts))):
        for k in range(3):
            if ps[i][k] == ts[i][k]:
                correct_tokens += 1
    # Unmatched triples contribute 0 correct tokens but add to total
    total_tokens += max(0, len(pred_triples) - len(tgt_triples)) * 3

    return {
        "correct_triples": correct_triples,
        "total_triples": len(tgt_triples),
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "exact_match": exact_match,
        "length_match": length_match,
    }


def compute_metrics(
    model: TripleWorldModel,
    dataset: TripleTransitionDataset,
    vocab: Vocabulary,
    device: torch.device,
    split_vocab: bool = False,
) -> dict[str, float]:
    """Compute set-based assessment metrics on a dataset.

    Returns dict with:
        precision:    fraction of predicted triples that are correct
        recall:       fraction of target triples that were predicted
        f1:           harmonic mean of precision and recall
        token_acc:    token-level accuracy (sorted alignment)
        exact_match:  fraction of examples with perfect output
        length_acc:   fraction of examples with correct triple count
    """
    was_training = model.training
    model.train(False)

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    total_correct_tokens = 0
    total_tokens = 0
    total_exact = 0
    total_length = 0
    n = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            pred_ids = model.predict(ex["input_ids"].unsqueeze(0).to(device))[0].cpu().tolist()
            tgt_ids = ex["target_ids"].tolist()

            decode = vocab.decode_triples_split if split_vocab else vocab.decode_triples
            pred_triples = decode(pred_ids)
            tgt_triples = decode(tgt_ids)

            m = _set_match(pred_triples, tgt_triples)

            # Precision / recall / F1
            pred_set = set(tuple(t) for t in pred_triples)
            tgt_set = set(tuple(t) for t in tgt_triples)
            correct = len(pred_set & tgt_set)
            prec = correct / max(len(pred_set), 1)
            rec = correct / max(len(tgt_set), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            sum_precision += prec
            sum_recall += rec
            sum_f1 += f1
            total_correct_tokens += m["correct_tokens"]
            total_tokens += m["total_tokens"]
            total_exact += int(m["exact_match"])
            total_length += int(m["length_match"])
            n += 1

    if was_training:
        model.train(True)

    return {
        "precision": sum_precision / max(n, 1),
        "recall": sum_recall / max(n, 1),
        "f1": sum_f1 / max(n, 1),
        "token_acc": total_correct_tokens / max(total_tokens, 1),
        "exact_match": total_exact / max(n, 1),
        "length_acc": total_length / max(n, 1),
    }


def compute_delta_metrics(
    model: TripleWorldModel,
    dataset: TripleTransitionDataset,
    vocab: Vocabulary,
    device: torch.device,
    split_vocab: bool = False,
) -> dict[str, float]:
    """Compute metrics on CHANGED triples only.

    Strips out triples identical between input and output, then measures
    accuracy on the remainder. This isolates the model's ability to predict
    actual state changes vs just copying.
    """
    was_training = model.training
    model.train(False)

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    total_exact = 0
    n = 0
    n_with_changes = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            ex = dataset[i]
            input_triples_raw, output_triples_raw = dataset.examples[i]

            # Find which triples changed
            input_set = set(tuple(t) for t in input_triples_raw)
            output_set = set(tuple(t) for t in output_triples_raw)
            persisted = input_set & output_set

            # Target delta: triples in output that weren't in input
            tgt_delta = output_set - persisted
            # Also count triples that disappeared (in input but not output)
            disappeared = input_set - output_set

            if not tgt_delta and not disappeared:
                continue  # no changes in this example

            n_with_changes += 1

            # Get model prediction
            pred_ids = model.predict(ex["input_ids"].unsqueeze(0).to(device))[0].cpu().tolist()
            decode = vocab.decode_triples_split if split_vocab else vocab.decode_triples
            pred_triples = decode(pred_ids)
            pred_set = set(tuple(t) for t in pred_triples)

            # Predicted delta: triples model outputs that weren't in input
            pred_delta = pred_set - persisted

            correct = len(pred_delta & tgt_delta)
            prec = correct / max(len(pred_delta), 1)
            rec = correct / max(len(tgt_delta), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            # Also check: did model correctly NOT output disappeared triples?
            false_persist = len(pred_set & disappeared)

            sum_precision += prec
            sum_recall += rec
            sum_f1 += f1
            total_exact += int(pred_delta == tgt_delta and false_persist == 0)
            n += 1

    if was_training:
        model.train(True)

    return {
        "delta_precision": sum_precision / max(n, 1),
        "delta_recall": sum_recall / max(n, 1),
        "delta_f1": sum_f1 / max(n, 1),
        "delta_exact": total_exact / max(n, 1),
        "n_with_changes": n_with_changes,
    }


def copy_baseline(dataset: TripleTransitionDataset) -> dict[str, float]:
    """Compute metrics for the trivial baseline: predict output = input."""
    sum_f1 = 0.0
    total_exact = 0
    n = 0

    for input_triples, output_triples in dataset.examples:
        pred_set = set(tuple(t) for t in input_triples)
        tgt_set = set(tuple(t) for t in output_triples)
        correct = len(pred_set & tgt_set)
        prec = correct / max(len(pred_set), 1)
        rec = correct / max(len(tgt_set), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        sum_f1 += f1
        total_exact += int(pred_set == tgt_set)
        n += 1

    return {
        "f1": sum_f1 / max(n, 1),
        "exact_match": total_exact / max(n, 1),
    }


def extract_attention_weights(
    model: TripleWorldModel,
    input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Extract attention weights from all layers.

    Returns list of (n_heads, T, T) tensors, one per layer.
    """
    model.train(False)

    latent, _raw_emb = model.triple_encoder(input_ids)
    pad_mask = input_ids == 0
    return model.dynamics.extract_attention_weights(latent, pad_mask)


def run_assessment():
    parser = argparse.ArgumentParser(description="Assess Triple World Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to run directory")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test_comp", "test_seen", "test_context"])

    args = parser.parse_args()

    run_dir = Path(args.checkpoint)
    data_dir = Path(args.data_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    vocab = Vocabulary.load(run_dir / "vocab.json")
    config = ModelConfig.load(run_dir / "config.json")
    model = TripleWorldModel(config).to(device)

    ckpt = run_dir / "model_best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "model_final.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.train(False)

    print(f"Loaded model from {ckpt} ({model.param_count():,} params)")
    print(f"Device: {device}\n")

    split_vocab = config.use_split_embeddings

    splits = ["train", "test_comp", "test_seen", "test_context"] if args.split == "all" else [args.split]

    for name in splits:
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        ds = TripleTransitionDataset(
            path,
            vocab,
            max_triples=config.max_triples,
            split_vocab=split_vocab,
        )
        m = compute_metrics(model, ds, vocab, device, split_vocab=split_vocab)

        print(f"--- {name} ({len(ds)} examples) ---")
        for k, v in m.items():
            print(f"  {k:15s}: {v:.4f}")
        print()

    train_path = data_dir / "train.jsonl"
    if train_path.exists():
        train_ds = TripleTransitionDataset(train_path, vocab, max_triples=config.max_triples)
        cb = copy_baseline(train_ds)
        print(f"--- copy baseline (train) ---")
        for k, v in cb.items():
            print(f"  {k:15s}: {v:.4f}")


if __name__ == "__main__":
    run_assessment()
