#!/usr/bin/env python3
"""Train the seq2seq Triple World Model on ATOMIC or other free-text data.

Supports multiple loss strategies:
  --loss ce          Cross-entropy only (baseline)
  --loss roundtrip   Cross-entropy + contrastive round-trip consistency
  --loss combined    Weighted CE + round-trip (default)

Usage:
    uv run python scripts/train_seq2seq_model.py \
        --data-dir data/atomic_small \
        --out-dir results/seq2seq_test \
        --config base --epochs 50

    # With round-trip contrastive loss
    uv run python scripts/train_seq2seq_model.py \
        --data-dir data/atomic_2000 \
        --out-dir results/seq2seq_roundtrip \
        --loss combined --roundtrip-weight 0.5 \
        --config base --epochs 200
"""

import argparse
import json
from pathlib import Path

import torch

from twm.config import ModelConfig, PROFILES
from twm.phrase_vocab import PhraseVocab
from twm.seq2seq_model import Seq2SeqTripleWorldModel
from twm.seq2seq_dataset import Seq2SeqTripleDataset
from twm.losses import Seq2SeqCrossEntropyLoss, RoundTripContrastiveLoss, CombinedLoss


def make_encode_fn(model_name: str, device: torch.device, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name, device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    def encode(phrases: list[str]) -> torch.Tensor:
        return st_model.encode(
            phrases,
            batch_size=batch_size,
            show_progress_bar=len(phrases) > 1000,
            convert_to_tensor=True,
            device=str(device),
        )

    return encode, st_dim


def build_eval(
    model: Seq2SeqTripleWorldModel,
    dataset: Seq2SeqTripleDataset,
    loss_fn,
    device: torch.device,
    max_examples: int = 200,
) -> dict[str, float]:
    """Compute loss and per-role accuracy."""
    model.eval()
    n = min(len(dataset), max_examples)

    inp = dataset._all_inputs[:n].to(device)
    inp_pad = dataset._all_input_pad_masks[:n].to(device)
    tgt_e = dataset._all_target_entity[:n].to(device)
    tgt_a = dataset._all_target_attr[:n].to(device)
    tgt_v = dataset._all_target_value[:n].to(device)
    tgt_pad = dataset._all_target_pad_masks[:n].to(device)

    with torch.no_grad():
        logits = model(inp, inp_pad)
        targets = {"entity": tgt_e, "attr": tgt_a, "value": tgt_v}
        # Add target embeds if the dataset has them (for round-trip eval)
        if hasattr(dataset, '_all_target_entity_embeds'):
            targets["entity_embeds"] = dataset._all_target_entity_embeds[:n].to(device)
            targets["attr_embeds"] = dataset._all_target_attr_embeds[:n].to(device)
            targets["value_embeds"] = dataset._all_target_value_embeds[:n].to(device)
        loss, metrics = loss_fn(logits, targets, pad_mask=tgt_pad)

    # Triple-level recall
    n_decode = min(n, 50)
    ids = model.predict_ids(inp[:n_decode], inp_pad[:n_decode])
    total_correct = 0
    total_triples = 0

    for i in range(n_decode):
        pred_set = set()
        tgt_set = set()
        M = model.config.max_triples
        for m in range(M):
            if not tgt_pad[i, m]:
                tgt_set.add((tgt_e[i, m].item(), tgt_a[i, m].item(), tgt_v[i, m].item()))
            pe = ids[i, m, 0].item()
            pa = ids[i, m, 1].item()
            pv = ids[i, m, 2].item()
            if pe != 0:
                pred_set.add((pe, pa, pv))
        total_correct += len(pred_set & tgt_set)
        total_triples += len(tgt_set)

    metrics["triple_recall"] = total_correct / max(total_triples, 1)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train seq2seq TWM")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="base",
                    choices=list(PROFILES.keys()))
    ap.add_argument("--st-model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--pretrained-dynamics", type=str, default=None)
    ap.add_argument("--pretrained-sentence-model", type=str, default=None,
                    help="Load encoder weights from trained SentenceTripleWorldModel")
    ap.add_argument("--freeze-dynamics", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--loss", type=str, default="ce",
                    choices=["ce", "roundtrip", "combined"],
                    help="Loss strategy: ce, roundtrip, or combined")
    ap.add_argument("--roundtrip-weight", type=float, default=0.5,
                    help="Weight for round-trip loss in combined mode")
    ap.add_argument("--roundtrip-temp", type=float, default=1.0,
                    help="Softmax temperature for round-trip soft lookup")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Sentence-transformer: {args.st_model}")

    # Build encoding function
    print("Loading sentence-transformer...")
    encode_fn, st_dim = make_encode_fn(args.st_model, device)
    print(f"  Embedding dim: {st_dim}")

    # Load config
    config = ModelConfig.from_profile(args.config)
    print(f"  Config: {args.config} (d_model={config.d_model}, {config.n_layers}L, {config.n_heads}H)")

    # Build phrase vocabulary
    print("Building phrase vocabulary...")
    train_path = data_dir / "train.jsonl"
    train_examples = []
    with open(train_path) as f:
        for line in f:
            train_examples.append(json.loads(line))

    vocab = PhraseVocab()
    vocab.build(train_examples)
    for role, size in vocab.vocab_sizes.items():
        print(f"  {role}: {size} phrases")
    vocab.save(out_dir / "phrase_vocab.json")

    # Build datasets
    print("Building datasets...")
    train_ds = Seq2SeqTripleDataset(train_path, encode_fn, vocab, max_triples=config.max_triples)
    print(f"  Train: {len(train_ds)} examples")

    test_ds = None
    test_path = data_dir / "test.jsonl"
    if test_path.exists():
        test_ds = Seq2SeqTripleDataset(test_path, encode_fn, vocab, max_triples=config.max_triples)
        print(f"  Test: {len(test_ds)} examples")

    # Build model
    model = Seq2SeqTripleWorldModel(config, st_dim, vocab).to(device)

    if args.pretrained_dynamics:
        print(f"Loading dynamics from {args.pretrained_dynamics}...")
        model.load_dynamics_from_checkpoint(args.pretrained_dynamics)

    if args.pretrained_sentence_model:
        print(f"Loading encoder from {args.pretrained_sentence_model}...")
        model.load_encoder_from_sentence_model(args.pretrained_sentence_model)

    if args.freeze_dynamics:
        model.freeze_dynamics()
    if args.freeze_encoder:
        model.freeze_encoder()

    print(f"Model: {model.param_count():,} total, {model.trainable_param_count():,} trainable", flush=True)

    # Save config
    config.save(out_dir / "config.json")
    with open(out_dir / "st_config.json", "w") as f:
        json.dump({"st_model": args.st_model, "st_dim": st_dim}, f, indent=2)

    # Build loss function
    ce_loss = Seq2SeqCrossEntropyLoss(pad_id=0, label_smoothing=args.label_smoothing)

    if args.loss == "ce":
        loss_fn = ce_loss
        print(f"  Loss: cross-entropy (label_smoothing={args.label_smoothing})")
    else:
        # Build phrase embeddings for round-trip loss
        print("  Building phrase embeddings for round-trip loss...")
        phrase_embeddings = vocab.build_embeddings(encode_fn)
        rt_loss = RoundTripContrastiveLoss(phrase_embeddings, temperature=args.roundtrip_temp)

        if args.loss == "roundtrip":
            loss_fn = rt_loss
            print(f"  Loss: round-trip contrastive (temp={args.roundtrip_temp})")
        else:
            loss_fn = CombinedLoss([
                (ce_loss, 1.0),
                (rt_loss, args.roundtrip_weight),
            ])
            print(f"  Loss: combined (CE + {args.roundtrip_weight}x round-trip, temp={args.roundtrip_temp})")

    # Optimizer with differential LR
    param_groups = []
    decoder_params = list(model.decoder.parameters())
    param_groups.append({"params": decoder_params, "lr": args.lr})

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        enc_lr = args.lr * 0.1 if args.pretrained_sentence_model else args.lr
        param_groups.append({"params": encoder_params, "lr": enc_lr})

    dynamics_params = [p for p in model.dynamics.parameters() if p.requires_grad]
    if dynamics_params:
        dyn_lr = args.lr * 0.1 if args.pretrained_dynamics else args.lr
        param_groups.append({"params": dynamics_params, "lr": dyn_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Move dataset to device
    all_inputs = train_ds._all_inputs.to(device)
    all_input_pads = train_ds._all_input_pad_masks.to(device)
    all_tgt_e = train_ds._all_target_entity.to(device)
    all_tgt_a = train_ds._all_target_attr.to(device)
    all_tgt_v = train_ds._all_target_value.to(device)
    all_tgt_pads = train_ds._all_target_pad_masks.to(device)
    # Target ST embeddings for round-trip loss
    all_tgt_e_emb = train_ds._all_target_entity_embeds.to(device)
    all_tgt_a_emb = train_ds._all_target_attr_embeds.to(device)
    all_tgt_v_emb = train_ds._all_target_value_embeds.to(device)
    n_train = all_inputs.shape[0]

    best_test_loss = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = perm[start:start + args.batch_size]
            inp = all_inputs[idx]
            inp_pad = all_input_pads[idx]
            tgt_pad = all_tgt_pads[idx]

            logits = model(inp, inp_pad)
            targets = {
                "entity": all_tgt_e[idx],
                "attr": all_tgt_a[idx],
                "value": all_tgt_v[idx],
                "entity_embeds": all_tgt_e_emb[idx],
                "attr_embeds": all_tgt_a_emb[idx],
                "value_embeds": all_tgt_v_emb[idx],
            }
            loss, _ = loss_fn(logits, targets, pad_mask=tgt_pad)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == 1:
            train_metrics = build_eval(model, train_ds, loss_fn, device)
            log = (
                f"Epoch {epoch:4d} | loss {avg_loss:.4f}"
                f" | acc_e {train_metrics.get('acc_entity', 0):.3f}"
                f" | acc_a {train_metrics.get('acc_attr', 0):.3f}"
                f" | acc_v {train_metrics.get('acc_value', 0):.3f}"
                f" | recall {train_metrics['triple_recall']:.3f}"
            )

            test_metrics = None
            if test_ds is not None:
                test_metrics = build_eval(model, test_ds, loss_fn, device)
                log += (
                    f" || test_recall {test_metrics['triple_recall']:.3f}"
                    f" | test_loss {test_metrics['loss_total']:.3f}"
                )
                if test_metrics["loss_total"] < best_test_loss:
                    best_test_loss = test_metrics["loss_total"]
                    torch.save(model.state_dict(), out_dir / "model_best.pt")

            print(log, flush=True)
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_recall": train_metrics["triple_recall"],
                **({f"train_{k}": v for k, v in train_metrics.items() if k.startswith("acc_")}),
                **({f"test_{k}": v for k, v in test_metrics.items()} if test_metrics else {}),
            })

    # Save final
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Saved to {out_dir}/")
    if best_test_loss < float("inf"):
        print(f"Best test loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    main()
