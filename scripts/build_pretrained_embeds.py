"""Build pretrained embedding matrix for TWM vocabulary.

Uses GloVe vectors (via gensim) to initialize token embeddings.
For compound tokens like 'tiny_parts_of_rocks', averages subword vectors.

Usage:
    python scripts/build_pretrained_embeds.py --vocab data/combined/vocab.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_glove(dim: int = 300):
    """Download and load GloVe vectors via gensim."""
    import gensim.downloader as api
    name = f"glove-wiki-gigaword-{dim}"
    print(f"Loading {name}...")
    return api.load(name)


def embed_token(token: str, glove, dim: int) -> np.ndarray | None:
    """Get embedding for a token, handling compound words.

    For 'tiny_parts_of_rocks':
      1. Try exact match
      2. Split on '_', average subword vectors
      3. Return None if no subwords found
    """
    if token in glove:
        return glove[token]

    # Split compound token and average found subwords
    parts = token.split("_")
    vectors = [glove[p] for p in parts if p in glove]
    if vectors:
        return np.mean(vectors, axis=0)

    return None


def build_embedding_matrix(vocab_path: Path, glove_dim: int = 300) -> tuple[torch.Tensor, dict]:
    """Build embedding matrix aligned with vocabulary.

    Returns:
        (vocab_size, glove_dim) tensor
        stats dict with coverage info
    """
    with open(vocab_path) as f:
        data = json.load(f)
    token2id = data["token2id"]
    vocab_size = data["next_id"]

    glove = load_glove(glove_dim)

    matrix = np.zeros((vocab_size, glove_dim), dtype=np.float32)
    found = 0
    missing = []

    for token, idx in token2id.items():
        if token == "<pad>":
            continue  # keep pad as zeros
        vec = embed_token(token, glove, glove_dim)
        if vec is not None:
            matrix[idx] = vec
            found += 1
        else:
            # Random init for missing tokens (small magnitude)
            matrix[idx] = np.random.randn(glove_dim) * 0.02
            missing.append(token)

    real_tokens = vocab_size - 1  # exclude <pad>
    stats = {
        "vocab_size": vocab_size,
        "glove_dim": glove_dim,
        "found": found,
        "missing": len(missing),
        "coverage": found / max(real_tokens, 1),
        "missing_tokens": missing,
    }

    return torch.from_numpy(matrix), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--glove-dim", type=int, default=300)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    output_path = Path(args.output) if args.output else vocab_path.parent / "pretrained_embeds.pt"

    matrix, stats = build_embedding_matrix(vocab_path, args.glove_dim)

    torch.save(matrix, output_path)

    print(f"\nCoverage: {stats['found']}/{stats['found'] + stats['missing']} "
          f"({stats['coverage']:.1%})")
    if stats['missing_tokens']:
        print(f"Missing ({len(stats['missing_tokens'])}): {stats['missing_tokens'][:30]}")
        if len(stats['missing_tokens']) > 30:
            print(f"  ... and {len(stats['missing_tokens']) - 30} more")
    print(f"\nSaved {matrix.shape} to {output_path}")


if __name__ == "__main__":
    main()
