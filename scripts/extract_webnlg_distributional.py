#!/usr/bin/env python3
"""Extract distributional role embeddings from WebNLG triples for CKA alignment.

Produces two artifacts:
1. distributional_lookup.pt — text-to-triples index + per-span 384d embeddings
2. distributional_spectra.pt — per-role normalized eigenvalue spectra (spectral target)

Usage:
    uv run python scripts/extract_webnlg_distributional.py \
        --data-dir data/webnlg_multi \
        --encoder all-MiniLM-L6-v2 \
        --top-k 32
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_triples(data_dir: Path):
    """Load triples and text from train.jsonl.

    Returns:
        text_to_triples: {text: [[e, a, v], ...]}
        role_spans: {"entity": {span: [sentence_indices]}, ...}
        sentences: list of all text strings
    """
    text_to_triples = {}
    role_spans = {"entity": defaultdict(list), "attribute": defaultdict(list), "value": defaultdict(list)}
    sentences = []

    path = data_dir / "train.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            text = rec["text"]
            triples = rec["triples"]
            text_to_triples[text] = triples
            sentences.append(text)

            for triple in triples:
                e, a, v = triple
                role_spans["entity"][e].append(i)
                role_spans["attribute"][a].append(i)
                role_spans["value"][v].append(i)

    return text_to_triples, role_spans, sentences


def compute_span_embeddings(role_spans, sentences, encoder, batch_size=256):
    """Compute distributional centroid embeddings per span per role.

    For each unique span in a role, the embedding is the mean of the sentence
    embeddings for all sentences containing that span (distributional centroid).

    Returns:
        span_embeddings: {"entity": {span: tensor(384,)}, ...}
    """
    print(f"Encoding {len(sentences)} sentences...")
    sent_embs = encoder.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    sent_embs = torch.from_numpy(sent_embs).float()

    span_embeddings = {}
    for role in ["entity", "attribute", "value"]:
        spans = role_spans[role]
        role_embs = {}
        for span, indices in spans.items():
            centroid = sent_embs[indices].mean(dim=0)
            role_embs[span] = centroid
        span_embeddings[role] = role_embs
        print(f"  {role}: {len(role_embs)} unique spans")

    return span_embeddings


def compute_spectral_targets(span_embeddings, top_k=32):
    """Compute per-role normalized eigenvalue spectra from distributional embeddings.

    Returns:
        spectra: {"entity": tensor(K,), "attribute": tensor(K,), "value": tensor(K,)}
    """
    spectra = {}
    for role in ["entity", "attribute", "value"]:
        vecs = torch.stack(list(span_embeddings[role].values()))
        centered = vecs - vecs.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (vecs.shape[0] - 1)
        eigvals = torch.linalg.eigvalsh(cov).flip(0)  # descending
        eigvals = eigvals.clamp(min=1e-8)

        K = min(top_k, len(eigvals))
        spectrum = eigvals[:K] / eigvals.sum()
        spectra[role] = spectrum

        print(f"  {role}: top-{K} spectrum, PC1={spectrum[0]:.3f}, "
              f"effective dims={1.0 / (spectrum ** 2).sum():.1f}")

    return spectra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/webnlg_multi")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load triples
    print("Loading triples from train.jsonl...")
    text_to_triples, role_spans, sentences = load_triples(data_dir)
    print(f"  {len(sentences)} examples, {len(text_to_triples)} unique texts")
    for role in ["entity", "attribute", "value"]:
        print(f"  {role}: {len(role_spans[role])} unique spans")

    # Encode sentences and compute span centroids
    print(f"\nLoading encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)

    print("Computing distributional span embeddings...")
    span_embeddings = compute_span_embeddings(
        role_spans, sentences, encoder, batch_size=args.batch_size
    )

    # Compute spectral targets
    print("\nComputing spectral targets...")
    spectra = compute_spectral_targets(span_embeddings, top_k=args.top_k)

    # Save lookup
    lookup_path = data_dir / "distributional_lookup.pt"
    torch.save({
        "text_to_triples": text_to_triples,
        "span_embeddings": span_embeddings,
    }, lookup_path)
    print(f"\nSaved lookup: {lookup_path}")

    # Save spectra
    spectra_path = data_dir / "distributional_spectra.pt"
    torch.save(spectra, spectra_path)
    print(f"Saved spectra: {spectra_path}")

    # Summary
    d = encoder.get_sentence_embedding_dimension()
    n_spans = sum(len(v) for v in span_embeddings.values())
    print(f"\nDone: {n_spans} total span embeddings ({d}d), "
          f"spectra top-{args.top_k} per role")


if __name__ == "__main__":
    main()
