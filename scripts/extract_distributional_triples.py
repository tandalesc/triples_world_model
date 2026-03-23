#!/usr/bin/env python3
"""Extract distributional role embeddings from text using spaCy dependency parses.

Pipeline:
1. Parse sentences with spaCy → dependency trees
2. Extract (subject_span, predicate_span, object_span) triples
3. Embed each span with a frozen sentence encoder
4. Accumulate per-role distributional centroids

Usage:
    uv run python scripts/extract_distributional_triples.py \
        --corpus data/sample_corpus.txt \
        --out data/distributional_triples/ \
        --max-sentences 100000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer


def extract_triples_from_doc(doc):
    """Extract (subject, predicate, object) triples from a spaCy Doc.

    Walks the dependency tree to find nsubj→ROOT→dobj/pobj patterns.
    Returns list of (subj_text, pred_text, obj_text, full_sentence) tuples.
    """
    triples = []
    for token in doc:
        # Find ROOT verbs
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            subjects = []
            objects = []

            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Get the full subtree span for the subject
                    subjects.append(_subtree_span(child))
                elif child.dep_ in ("dobj", "attr", "oprd"):
                    objects.append(_subtree_span(child))
                elif child.dep_ == "prep":
                    # Prepositional objects: "born in Nottingham"
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            # Include the preposition in the predicate
                            pred_text = token.text + " " + child.text
                            for subj in subjects:
                                triples.append((
                                    subj,
                                    pred_text,
                                    _subtree_span(pobj),
                                    doc.text
                                ))

            # Direct verb-object triples
            pred_text = token.lemma_
            for subj in subjects:
                for obj in objects:
                    triples.append((subj, pred_text, obj, doc.text))

    return triples


def _subtree_span(token):
    """Get the text of a token's full subtree (compound nouns, modifiers, etc.)."""
    subtree = sorted(token.subtree, key=lambda t: t.i)
    return " ".join(t.text for t in subtree)


def compute_distributional_embeddings(triples, encoder, batch_size=256):
    """Compute distributional role embeddings from extracted triples.

    For each unique span in each role, accumulate sentence embeddings
    to compute a centroid = the distributional role embedding.

    Returns:
        role_embeddings: dict[role][span] = np.array (centroid)
        role_counts: dict[role][span] = int (number of occurrences)
        triple_records: list of dicts with span texts and sentence
    """
    # Collect all unique sentences for batch encoding
    sentences = list(set(t[3] for t in triples))
    print(f"  Encoding {len(sentences)} unique sentences...")

    # Batch encode
    sent_to_idx = {s: i for i, s in enumerate(sentences)}
    embeddings = encoder.encode(sentences, batch_size=batch_size, show_progress_bar=True)

    # Accumulate per-role centroids
    role_names = ["subject", "predicate", "object"]
    accumulators = {r: defaultdict(list) for r in role_names}

    for subj, pred, obj, sent in triples:
        sent_emb = embeddings[sent_to_idx[sent]]
        accumulators["subject"][subj].append(sent_emb)
        accumulators["predicate"][pred].append(sent_emb)
        accumulators["object"][obj].append(sent_emb)

    # Compute centroids
    role_embeddings = {}
    role_counts = {}
    for role in role_names:
        role_embeddings[role] = {}
        role_counts[role] = {}
        for span, embs in accumulators[role].items():
            role_embeddings[role][span] = np.mean(embs, axis=0)
            role_counts[role][span] = len(embs)

    # Build triple records
    triple_records = []
    for subj, pred, obj, sent in triples:
        triple_records.append({
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "sentence": sent,
        })

    return role_embeddings, role_counts, triple_records


def analyze_geometry(role_embeddings, role_counts, top_k=20):
    """Print geometric analysis of the distributional embeddings."""
    print("\n=== Distributional Embedding Geometry ===\n")

    for role in ["subject", "predicate", "object"]:
        embs = role_embeddings[role]
        counts = role_counts[role]

        # Filter to spans with enough occurrences
        frequent = {s: e for s, e in embs.items() if counts[s] >= 5}
        if len(frequent) < 2:
            print(f"  {role}: too few frequent spans ({len(frequent)})")
            continue

        spans = list(frequent.keys())
        vecs = np.stack([frequent[s] for s in spans])

        # Compute pairwise cosine similarities
        norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norms @ norms.T

        # PCA variance
        centered = vecs - vecs.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained = S ** 2 / (S ** 2).sum()

        print(f"  {role.upper()}: {len(frequent)} unique spans (≥5 occurrences)")
        print(f"    PCA variance: PC1={var_explained[0]:.3f} PC2={var_explained[1]:.3f} PC3={var_explained[2]:.3f}")
        print(f"    Mean pairwise cosine sim: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.3f}")

        # Show nearest neighbors for a few common spans
        top_spans = sorted(spans, key=lambda s: counts[s], reverse=True)[:top_k]
        print(f"    Top spans by frequency:")
        for s in top_spans[:5]:
            # Find nearest neighbor
            s_vec = norms[spans.index(s)]
            sims = norms @ s_vec
            sims[spans.index(s)] = -1  # exclude self
            nn_idx = sims.argmax()
            nn_span = spans[nn_idx]
            print(f"      '{s}' (n={counts[s]}) → nearest: '{nn_span}' (sim={sims[nn_idx]:.3f})")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Input text file (one sentence per line)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--max-sentences", type=int, default=100000)
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2", help="Sentence encoder model")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    print(f"Loading sentence encoder: {args.encoder}")
    encoder = SentenceTransformer(args.encoder)

    # Read corpus
    print(f"Reading corpus: {args.corpus}")
    with open(args.corpus) as f:
        sentences = [line.strip() for line in f if line.strip()]
    sentences = sentences[:args.max_sentences]
    print(f"  {len(sentences)} sentences")

    # Extract triples
    print("Extracting dependency triples...")
    all_triples = []
    n_parsed = 0
    for doc in nlp.pipe(sentences, batch_size=1000, n_process=1):
        triples = extract_triples_from_doc(doc)
        all_triples.extend(triples)
        n_parsed += 1
        if n_parsed % 10000 == 0:
            print(f"  Parsed {n_parsed}/{len(sentences)} sentences, {len(all_triples)} triples so far")

    print(f"  Total: {len(all_triples)} triples from {n_parsed} sentences")
    print(f"  Yield: {len(all_triples) / n_parsed:.2f} triples/sentence")

    if not all_triples:
        print("No triples extracted! Check corpus format.")
        return

    # Compute distributional embeddings
    print("Computing distributional embeddings...")
    role_embeddings, role_counts, triple_records = compute_distributional_embeddings(
        all_triples, encoder, batch_size=args.batch_size
    )

    # Analyze geometry
    analyze_geometry(role_embeddings, role_counts)

    # Save
    print(f"Saving to {out_dir}/")

    # Save embeddings as numpy
    for role in ["subject", "predicate", "object"]:
        spans = list(role_embeddings[role].keys())
        vecs = np.stack([role_embeddings[role][s] for s in spans])
        np.save(out_dir / f"{role}_embeddings.npy", vecs)
        with open(out_dir / f"{role}_spans.json", "w") as f:
            json.dump({"spans": spans, "counts": [role_counts[role][s] for s in spans]}, f)

    # Save triple records
    with open(out_dir / "triples.jsonl", "w") as f:
        for rec in triple_records:
            f.write(json.dumps(rec) + "\n")

    # Summary stats
    stats = {
        "n_sentences": n_parsed,
        "n_triples": len(all_triples),
        "triples_per_sentence": len(all_triples) / n_parsed,
        "unique_subjects": len(role_embeddings["subject"]),
        "unique_predicates": len(role_embeddings["predicate"]),
        "unique_objects": len(role_embeddings["object"]),
        "encoder": args.encoder,
        "embedding_dim": encoder.get_sentence_embedding_dimension(),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! {stats['n_triples']} triples, "
          f"{stats['unique_subjects']} subjects, "
          f"{stats['unique_predicates']} predicates, "
          f"{stats['unique_objects']} objects")


if __name__ == "__main__":
    main()
