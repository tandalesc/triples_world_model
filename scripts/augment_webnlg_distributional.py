#!/usr/bin/env python3
"""Augment WebNLG training data by swapping spans with distributional neighbors.

Uses distributional embeddings to find similar entities/values, generates new
triples by swapping, and produces identity + QA pairs using the same templates
as generate_qa_dataset.py.

Usage:
    uv run python scripts/augment_webnlg_distributional.py \
        --data-dir data/webnlg_multi \
        --k 5 --sim-threshold 0.5
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# Reuse templates from generate_qa_dataset
from generate_qa_dataset import (
    make_question,
    make_answer_sentence,
    TEMPLATES,
    ANSWER_TEMPLATES,
)


def build_neighbor_index(span_embeddings, attr_values, k=5, sim_threshold=0.5):
    """Build K-nearest-neighbor index per role using cosine similarity.

    For values, neighbors are constrained to spans that share at least one
    attribute — this prevents swapping cities with numbers, etc.

    Args:
        span_embeddings: {"entity": {span: tensor(384,)}, ...}
        attr_values: {attribute: set(values)} — which values appear with each attribute
        k: number of neighbors per span
        sim_threshold: minimum cosine similarity to include

    Returns:
        entity_neighbors: {span: [(neighbor_span, sim), ...]}
        value_neighbors: {(attribute, value): [(neighbor_value, sim), ...]}
    """
    # Entity neighbors: global (entities are interchangeable across attributes)
    entity_spans = list(span_embeddings["entity"].keys())
    entity_neighbors = {}
    if len(entity_spans) >= 2:
        embs = torch.stack([span_embeddings["entity"][s] for s in entity_spans])
        embs_norm = F.normalize(embs, dim=-1)
        sim_matrix = embs_norm @ embs_norm.T

        for i, span in enumerate(entity_spans):
            sims = sim_matrix[i].clone()
            sims[i] = -1
            topk_vals, topk_idx = sims.topk(min(k, len(entity_spans) - 1))
            nn = [(entity_spans[idx.item()], val.item())
                  for val, idx in zip(topk_vals, topk_idx)
                  if val.item() >= sim_threshold]
            entity_neighbors[span] = nn

        n_with = sum(1 for v in entity_neighbors.values() if v)
        avg_nn = sum(len(v) for v in entity_neighbors.values()) / max(len(entity_neighbors), 1)
        print(f"  entity: {n_with}/{len(entity_spans)} spans have neighbors (avg {avg_nn:.1f})")

    # Value neighbors: per-attribute (only swap with values seen with same attribute)
    value_neighbors = {}
    value_embs = span_embeddings["value"]
    total_pairs = 0
    total_with = 0
    for attr, values in attr_values.items():
        # Filter to values that have embeddings
        valid = [v for v in values if v in value_embs]
        if len(valid) < 2:
            continue
        spans = list(valid)
        embs = torch.stack([value_embs[s] for s in spans])
        embs_norm = F.normalize(embs, dim=-1)
        sim_matrix = embs_norm @ embs_norm.T

        for i, span in enumerate(spans):
            sims = sim_matrix[i].clone()
            sims[i] = -1
            topk_vals, topk_idx = sims.topk(min(k, len(spans) - 1))
            nn = [(spans[idx.item()], val.item())
                  for val, idx in zip(topk_vals, topk_idx)
                  if val.item() >= sim_threshold]
            value_neighbors[(attr, span)] = nn
            total_pairs += 1
            if nn:
                total_with += 1

    avg_nn = sum(len(v) for v in value_neighbors.values()) / max(len(value_neighbors), 1)
    print(f"  value (per-attr): {total_with}/{total_pairs} (attr,value) pairs have neighbors (avg {avg_nn:.1f})")

    return entity_neighbors, value_neighbors


def augment_triples(original_triples, entity_neighbors, value_neighbors, existing_triple_keys):
    """Generate augmented triple sets by swapping entities and values.

    Args:
        original_triples: list of [e, a, v] for one example
        entity_neighbors: {entity: [(nn, sim), ...]}
        value_neighbors: {(attr, value): [(nn_value, sim), ...]}
        existing_triple_keys: set of (e, a, v) tuples for dedup

    Returns:
        list of augmented triple sets (each is a list of [e, a, v])
    """
    augmented = []

    for triple in original_triples:
        e, a, v = triple
        e_nns = entity_neighbors.get(e, [])
        v_nns = value_neighbors.get((a, v), [])

        # Swap entity only
        for e_nn, _ in e_nns:
            key = (e_nn, a, v)
            if key not in existing_triple_keys:
                augmented.append([list(key)])
                existing_triple_keys.add(key)

        # Swap value only
        for v_nn, _ in v_nns:
            key = (e, a, v_nn)
            if key not in existing_triple_keys:
                augmented.append([list(key)])
                existing_triple_keys.add(key)

        # Swap both
        for e_nn, _ in e_nns:
            for v_nn, _ in v_nns:
                key = (e_nn, a, v_nn)
                if key not in existing_triple_keys:
                    augmented.append([list(key)])
                    existing_triple_keys.add(key)

    return augmented


def triples_to_text(triples):
    """Generate identity text from triples using answer templates."""
    sentences = []
    for e, a, v in triples:
        sentences.append(make_answer_sentence(e, a, v))
    return " ".join(sentences)


def triples_to_qa_pairs(triples):
    """Generate QA pairs from triples using templates."""
    pairs = []
    for e, a, v in triples:
        question, answer = make_question(e, a, v)
        pairs.append({"mode": "qa", "input_text": question, "output_text": answer})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/webnlg_multi")
    parser.add_argument("--k", type=int, default=5, help="Neighbors per span")
    parser.add_argument("--sim-threshold", type=float, default=0.5,
                        help="Min cosine similarity for neighbor swap")
    parser.add_argument("--max-augmented-per-example", type=int, default=20,
                        help="Cap augmented triples per original example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    # Load distributional lookup
    print("Loading distributional lookup...")
    lookup = torch.load(data_dir / "distributional_lookup.pt", weights_only=False)
    span_embeddings = lookup["span_embeddings"]

    # Load original data and build attribute→values index
    print("Loading original data...")
    original_examples = []
    attr_values = defaultdict(set)
    with open(data_dir / "train.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            original_examples.append(ex)
            for e, a, v in ex["triples"]:
                attr_values[a].add(v)
    print(f"  {len(original_examples)} original examples")
    print(f"  {len(attr_values)} unique attributes")
    print(f"  Top attrs by value count: {', '.join(f'{a}={len(vs)}' for a, vs in sorted(attr_values.items(), key=lambda x: -len(x[1]))[:5])}")

    # Build neighbor index (attribute-constrained for values)
    print("Building neighbor index...")
    entity_neighbors, value_neighbors = build_neighbor_index(
        span_embeddings, attr_values, k=args.k, sim_threshold=args.sim_threshold
    )

    # Track existing triples for dedup (individual triples, not sets)
    existing_keys = set()
    for ex in original_examples:
        for t in ex["triples"]:
            existing_keys.add(tuple(t))
    print(f"  {len(existing_keys)} unique original triples")

    # Generate augmented data
    print("Generating augmented triples...")
    augmented_triple_sets = []
    for ex in original_examples:
        aug = augment_triples(ex["triples"], entity_neighbors, value_neighbors, existing_keys)
        if args.max_augmented_per_example > 0:
            random.shuffle(aug)
            aug = aug[:args.max_augmented_per_example]
        augmented_triple_sets.extend(aug)

    print(f"  {len(augmented_triple_sets)} augmented triple sets generated")

    # Generate texts and QA pairs
    print("Generating texts and QA pairs...")
    all_texts = set()
    qa_pairs = []

    # Original data (identity pairs + QA)
    for ex in original_examples:
        text = ex["text"]
        all_texts.add(text)
        qa_pairs.append({"mode": "identity", "input_text": text, "output_text": text})
        for triple in ex["triples"]:
            q, a = make_question(*triple)
            all_texts.add(q)
            all_texts.add(a)
            qa_pairs.append({"mode": "qa", "input_text": q, "output_text": a})

    # Augmented data
    augmented_train = []
    for triple_set in augmented_triple_sets:
        text = triples_to_text(triple_set)
        all_texts.add(text)
        augmented_train.append({"triples": triple_set, "text": text, "alt_texts": []})
        # Identity pair
        qa_pairs.append({"mode": "identity", "input_text": text, "output_text": text})
        # QA pairs
        qa_pairs.extend(triples_to_qa_pairs(triple_set))
        for triple in triple_set:
            q, a = make_question(*triple)
            all_texts.add(q)
            all_texts.add(a)

    # Write augmented train.jsonl (original + augmented, for distributional rebuild)
    aug_train_path = data_dir / "augmented_train.jsonl"
    with open(aug_train_path, "w") as f:
        for ex in original_examples:
            f.write(json.dumps(ex) + "\n")
        for ex in augmented_train:
            f.write(json.dumps(ex) + "\n")
    print(f"  augmented_train.jsonl: {len(original_examples) + len(augmented_train)} examples")

    # Write identity dataset
    identity_path = data_dir / "augmented_identity_train.jsonl"
    unique_texts = sorted(all_texts)
    with open(identity_path, "w") as f:
        for text in unique_texts:
            f.write(json.dumps({"text": text}) + "\n")
    print(f"  augmented_identity_train.jsonl: {len(unique_texts)} unique texts")

    # Write QA dataset
    random.shuffle(qa_pairs)
    qa_path = data_dir / "augmented_qa_train.jsonl"
    with open(qa_path, "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")

    n_id = sum(1 for p in qa_pairs if p["mode"] == "identity")
    n_qa = sum(1 for p in qa_pairs if p["mode"] == "qa")
    print(f"  augmented_qa_train.jsonl: {len(qa_pairs)} pairs ({n_id} identity, {n_qa} QA)")

    # Also write a balanced version name hint (the trainer balances on load)
    # The trainer looks for {dataset}_train.jsonl, so "augmented_qa_balanced" will
    # try "augmented_qa_balanced_train.jsonl" first, then fall back.
    # Easiest: symlink or just copy as the balanced name
    balanced_path = data_dir / "augmented_qa_balanced_train.jsonl"
    import shutil
    shutil.copy2(qa_path, balanced_path)
    print(f"  augmented_qa_balanced_train.jsonl: (copy, balanced on load)")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Original: {len(original_examples)} examples, {len(existing_keys)} unique triples")
    print(f"Augmented: {len(augmented_triple_sets)} new triple sets")
    print(f"Total QA pairs: {len(qa_pairs)} ({n_id} id + {n_qa} qa)")
    print(f"Augmentation ratio: {len(augmented_triple_sets) / max(len(existing_keys), 1):.1f}x")


if __name__ == "__main__":
    main()
