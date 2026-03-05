"""Convert OpenPI dataset to TWM triple transition format.

OpenPI tracks entity state changes across steps in wikiHow procedures.
Each step has multiple (entity, attribute, before, after) annotations.

We convert each step to:
  state_t:   [(entity, attribute, before_value), ...]
  state_t+1: [(entity, attribute, after_value), ...]

Values are normalized to clean single/compound tokens.

Usage:
    python scripts/convert_openpi.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# LLM-generated normalization mapping (if available)
_llm_normalizations: dict[str, str] = {}
_llm_norm_path = Path("data/openpi_raw/value_normalizations.json")
if _llm_norm_path.exists():
    with open(_llm_norm_path) as f:
        _llm_normalizations = json.load(f)


def normalize_token(text: str) -> str:
    """Normalize a natural language value/entity/attribute to a clean token."""
    text_lower = text.strip().lower()

    # Use LLM normalization if available
    if text_lower in _llm_normalizations:
        return _llm_normalizations[text_lower]

    # Fallback: rule-based
    text = text_lower
    text = re.sub(r"^(now |now changed to |changed to |becomes? |became )", "", text)
    text = re.sub(r"[.!?]+$", "", text)
    text = re.sub(r"[''`]s\b", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = text.strip("_")
    return text or "unknown"


def is_clean_value(val: str) -> bool:
    """Check if a normalized value is usable (not too long, not empty)."""
    return 1 <= len(val) <= 30 and val != "unknown"


def convert_step(answers_metadata: list[dict]) -> dict | None:
    """Convert one OpenPI step's annotations to a triple transition.

    Returns {"state_t": [...], "state_t+1": [...]} or None if unusable.
    """
    triples_t = []
    triples_t1 = []

    for ann in answers_metadata:
        entity = normalize_token(ann["entity"])
        attr = normalize_token(ann["attr"])
        before = normalize_token(ann["before"])
        after = normalize_token(ann["after"])

        # Skip if any component is unusable
        if not all(is_clean_value(v) for v in [entity, attr, before, after]):
            continue
        # Skip if no actual change
        if before == after:
            continue

        triples_t.append([entity, attr, before])
        triples_t1.append([entity, attr, after])

    if not triples_t or not triples_t1:
        return None

    return {"state_t": triples_t, "state_t+1": triples_t1}


def convert_file(input_path: Path, max_triples: int = 8) -> list[dict]:
    """Convert an OpenPI JSONL file to TWM format."""
    examples = []
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            result = convert_step(d["answers_metadata"])
            if result is None:
                continue
            # Filter: skip examples with too many triples
            if len(result["state_t"]) > max_triples:
                continue
            examples.append(result)
    return examples


def dedup_conflicts(examples: list[dict]) -> list[dict]:
    """Remove examples with same input but different outputs."""
    input_to_outputs = defaultdict(list)
    for ex in examples:
        key = tuple(tuple(t) for t in sorted(ex["state_t"]))
        out = tuple(tuple(t) for t in sorted(ex["state_t+1"]))
        input_to_outputs[key].append((out, ex))

    clean = []
    n_conflicts = 0
    for key, entries in input_to_outputs.items():
        unique_outputs = set(out for out, _ in entries)
        if len(unique_outputs) == 1:
            clean.append(entries[0][1])
        else:
            n_conflicts += len(entries)

    print(f"  Dedup: removed {n_conflicts} conflicting examples ({len(clean)} remain)")
    return clean


def filter_by_vocab(examples: list[dict], allowed_tokens: set[str]) -> list[dict]:
    """Keep only examples where ALL tokens are in the allowed set."""
    filtered = []
    for ex in examples:
        all_tokens = set()
        for triple in ex["state_t"] + ex["state_t+1"]:
            all_tokens.update(triple)
        if all_tokens <= allowed_tokens:
            filtered.append(ex)
    return filtered


def main():
    gold_dir = Path("data/openpi_raw/data/gold")
    out_dir = Path("data")
    min_freq = 2

    # Convert all splits
    train_raw = convert_file(gold_dir / "train" / "id_answers_metadata.jsonl")
    dev_raw = convert_file(gold_dir / "dev" / "id_answers_metadata.jsonl")
    test_raw = convert_file(gold_dir / "test" / "id_answers_metadata.jsonl")
    print(f"Raw examples — train: {len(train_raw)}, dev: {len(dev_raw)}, test: {len(test_raw)}")

    # Token frequency analysis on train
    token_counts: Counter[str] = Counter()
    for ex in train_raw:
        for triple in ex["state_t"] + ex["state_t+1"]:
            token_counts.update(triple)

    allowed = {tok for tok, c in token_counts.items() if c >= min_freq}
    print(f"\nVocab: {len(token_counts)} total tokens, {len(allowed)} with freq >= {min_freq}")

    # Frequency distribution
    freq_buckets = Counter()
    for c in token_counts.values():
        if c == 1: freq_buckets["1"] += 1
        elif c <= 3: freq_buckets["2-3"] += 1
        elif c <= 10: freq_buckets["4-10"] += 1
        elif c <= 50: freq_buckets["11-50"] += 1
        else: freq_buckets["50+"] += 1
    print(f"Frequency distribution: {dict(sorted(freq_buckets.items()))}")

    # Filter and dedup
    train = dedup_conflicts(filter_by_vocab(train_raw, allowed))
    train_vocab = set()
    for ex in train:
        for triple in ex["state_t"] + ex["state_t+1"]:
            train_vocab.update(triple)
    dev = dedup_conflicts(filter_by_vocab(dev_raw, train_vocab))
    test = dedup_conflicts(filter_by_vocab(test_raw, train_vocab))
    print(f"\nAfter filter + dedup — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    # Write output
    for name, data in [("openpi_train", train), ("openpi_dev", dev), ("openpi_test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {name}.jsonl: {len(data)} examples")

    # Final stats
    all_tokens = set()
    for ex in train + dev + test:
        for triple in ex["state_t"] + ex["state_t+1"]:
            all_tokens.update(triple)
    print(f"\nFinal unique tokens: {len(all_tokens)}")

    # Samples
    print("\nSample transitions:")
    for ex in train[:8]:
        print(f"  in:  {ex['state_t']}")
        print(f"  out: {ex['state_t+1']}")
        print()

    # Top tokens
    attr_counts: Counter[str] = Counter()
    for ex in train:
        for triple in ex["state_t"] + ex["state_t+1"]:
            attr_counts[triple[1]] += 1
    print(f"Top attributes: {attr_counts.most_common(20)}")


if __name__ == "__main__":
    main()
