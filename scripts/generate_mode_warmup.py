"""Generate mode warmup data from identity training data.

Creates paired examples with two modes:
  - identity: input_text = output_text (passthrough)
  - reverse: output_text has sentences in reversed order

This forces the dynamics core to develop mode-reading circuitry before
encountering QA. The core must attend to the mode triple to decide
passthrough vs sentence reordering.

Usage:
    python scripts/generate_mode_warmup.py data/webnlg_multi/identity_train.jsonl \
        --output data/webnlg_multi/mode_warmup_train.jsonl
"""

import argparse
import json
import re
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ' boundaries.

    Handles common WebNLG patterns like abbreviations and decimals.
    """
    # Split on ". " followed by a lowercase or uppercase letter
    # This avoids splitting on "d.c." or "$10,264,000,000"
    parts = re.split(r'(?<=\.)\s+(?=[a-z])', text, flags=re.IGNORECASE)
    # Filter empty
    return [p.strip() for p in parts if p.strip()]


def reverse_sentences(text: str) -> str | None:
    """Reverse sentence order. Returns None if single sentence."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return None
    return " ".join(reversed(sents))


def main():
    parser = argparse.ArgumentParser(description="Generate mode warmup data")
    parser.add_argument("input", type=Path, help="identity_train.jsonl path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max total examples (0=unlimited)")
    args = parser.parse_args()

    identity_pairs = []
    reverse_pairs = []

    with open(args.input) as f:
        for line in f:
            ex = json.loads(line)
            text = ex.get("text") or ex.get("input_text", "")
            if not text:
                continue

            # Identity pair
            identity_pairs.append({
                "mode": "identity",
                "input_text": text,
                "output_text": text,
            })

            # Reverse pair (only if multi-sentence)
            rev = reverse_sentences(text)
            if rev is not None:
                reverse_pairs.append({
                    "mode": "reverse",
                    "input_text": text,
                    "output_text": rev,
                })

    # Balance: use min(identity, reverse) of each for 50/50 split
    n_rev = len(reverse_pairs)
    n_id = min(len(identity_pairs), n_rev)
    pairs = identity_pairs[:n_id] + reverse_pairs[:n_rev]

    if args.max_examples > 0:
        pairs = pairs[:args.max_examples]

    # Shuffle deterministically (interleave identity/reverse)
    import random
    random.seed(42)
    random.shuffle(pairs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    n_id_final = sum(1 for p in pairs if p["mode"] == "identity")
    n_rev_final = sum(1 for p in pairs if p["mode"] == "reverse")
    print(f"Generated {len(pairs)} pairs ({n_id_final} identity, {n_rev_final} reverse)")
    print(f"  from {len(identity_pairs)} identity texts ({n_rev} multi-sentence)")
    print(f"  → {args.output}")


if __name__ == "__main__":
    main()
