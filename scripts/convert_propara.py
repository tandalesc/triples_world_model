"""Convert ProPara dataset to TWM triple transition format.

ProPara tracks entity locations across timesteps in process descriptions.
Each consecutive timestep pair becomes one training example:
  state_t:   [(entity, "location", place), ...] for all entities with known locations
  state_t+1: [(entity, "location", place), ...] for all entities with known locations

Entities with state "-" (doesn't exist) are omitted.
Entities with state "?" (unknown) cause the transition to be skipped if too many.

Usage:
    python scripts/convert_propara.py
"""

import json
import re
from pathlib import Path


def normalize_location(loc: str) -> str:
    """Normalize location strings to clean tokens."""
    loc = loc.strip().lower()
    # Replace spaces/punctuation with underscores for single-token locations
    loc = re.sub(r"[''`]s\b", "", loc)  # remove possessives
    loc = re.sub(r"\s+", "_", loc)
    loc = re.sub(r"[^a-z0-9_]", "", loc)
    loc = loc.strip("_")
    return loc or "unknown"


def normalize_entity(entity: str) -> str:
    """Normalize entity names."""
    # Handle "a; b; c" format (multiple names for same entity)
    entity = entity.split(";")[0].strip()
    entity = entity.strip().lower()
    entity = re.sub(r"[''`]s\b", "", entity)
    entity = re.sub(r"\s+", "_", entity)
    entity = re.sub(r"[^a-z0-9_]", "", entity)
    return entity.strip("_") or "unknown"


def convert_paragraph(paragraph: dict, max_unknowns_pct: float = 0.5) -> list[dict]:
    """Convert one ProPara paragraph to a list of triple transitions.

    Args:
        paragraph: ProPara format dict with participants, states, sentence_texts
        max_unknowns_pct: skip transitions where more than this fraction of entities are unknown

    Returns:
        list of {"state_t": [...], "state_t+1": [...]} dicts
    """
    participants = [normalize_entity(p) for p in paragraph["participants"]]
    states = paragraph["states"]
    n_steps = len(states[0])
    n_participants = len(participants)

    examples = []

    for t in range(n_steps - 1):
        triples_t = []
        triples_t1 = []
        n_unknown = 0

        for p_idx in range(n_participants):
            entity = participants[p_idx]
            s_t = states[p_idx][t]
            s_t1 = states[p_idx][t + 1]

            if s_t == "?" or s_t1 == "?":
                n_unknown += 1
                continue

            # Build triples for known, existing entities
            if s_t != "-":
                triples_t.append([entity, "location", normalize_location(s_t)])
            if s_t1 != "-":
                triples_t1.append([entity, "location", normalize_location(s_t1)])

        # Skip if too many unknowns or empty states
        if n_participants > 0 and n_unknown / n_participants > max_unknowns_pct:
            continue
        if not triples_t and not triples_t1:
            continue

        # Need at least one triple on EACH side for a meaningful transition
        if not triples_t or not triples_t1:
            continue

        examples.append({
            "state_t": triples_t,
            "state_t+1": triples_t1,
        })

    return examples


def collect_all_examples(input_path: Path, max_triples: int = 8) -> list[dict]:
    """Collect all valid examples from a ProPara JSONL file."""
    examples = []
    with open(input_path) as f:
        for line in f:
            paragraph = json.loads(line)
            transitions = convert_paragraph(paragraph)
            for ex in transitions:
                if len(ex["state_t"]) > max_triples or len(ex["state_t+1"]) > max_triples:
                    continue
                if set(tuple(t) for t in ex["state_t"]) == set(tuple(t) for t in ex["state_t+1"]):
                    continue
                examples.append(ex)
    return examples


def get_token_counts(examples: list[dict]) -> dict[str, int]:
    """Count frequency of each token across all examples."""
    from collections import Counter
    counts: Counter[str] = Counter()
    for ex in examples:
        for triple in ex["state_t"] + ex["state_t+1"]:
            counts.update(triple)
    return dict(counts)


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


def dedup_conflicts(examples: list[dict]) -> list[dict]:
    """Remove examples that share an input but have different outputs.

    These are ambiguous — the model can't learn contradictory mappings.
    """
    from collections import defaultdict

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
            # All same output — keep one copy
            clean.append(entries[0][1])
        else:
            n_conflicts += len(entries)

    print(f"  Dedup: removed {n_conflicts} conflicting examples ({len(clean)} remain)")
    return clean


def write_examples(examples: list[dict], output_path: Path) -> int:
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def main():
    from collections import Counter

    raw_dir = Path("data/propara_raw")
    out_dir = Path("data")
    min_freq = 2  # tokens must appear at least this many times across train

    # Collect all examples (unfiltered)
    train_raw = collect_all_examples(raw_dir / "train.json")
    dev_raw = collect_all_examples(raw_dir / "dev.json")
    test_raw = collect_all_examples(raw_dir / "test.json")
    print(f"Raw examples — train: {len(train_raw)}, dev: {len(dev_raw)}, test: {len(test_raw)}")

    # Build vocab from train set with frequency filter
    train_counts = get_token_counts(train_raw)
    # "location" is the relation for all triples — always keep it
    allowed = {tok for tok, c in train_counts.items() if c >= min_freq}
    allowed.add("location")
    print(f"\nVocab: {len(train_counts)} total tokens, {len(allowed)} with freq >= {min_freq}")

    # Show frequency distribution
    freq_buckets = Counter()
    for c in train_counts.values():
        if c == 1: freq_buckets["1"] += 1
        elif c <= 3: freq_buckets["2-3"] += 1
        elif c <= 10: freq_buckets["4-10"] += 1
        elif c <= 50: freq_buckets["11-50"] += 1
        else: freq_buckets["50+"] += 1
    print(f"Frequency distribution: {dict(sorted(freq_buckets.items()))}")

    # Filter train by allowed vocab, then deduplicate conflicts
    train = filter_by_vocab(train_raw, allowed)
    train = dedup_conflicts(train)
    # Dev/test: only require that ALL tokens appeared somewhere in filtered train
    train_vocab = set()
    for ex in train:
        for triple in ex["state_t"] + ex["state_t+1"]:
            train_vocab.update(triple)
    dev = dedup_conflicts(filter_by_vocab(dev_raw, train_vocab))
    test = dedup_conflicts(filter_by_vocab(test_raw, train_vocab))
    print(f"\nAfter vocab filter + dedup — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

    # Write output
    write_examples(train, out_dir / "propara_train.jsonl")
    write_examples(dev, out_dir / "propara_dev.jsonl")
    write_examples(test, out_dir / "propara_test.jsonl")

    # Final vocab stats
    all_examples = train + dev + test
    final_tokens = set()
    for ex in all_examples:
        for triple in ex["state_t"] + ex["state_t+1"]:
            final_tokens.update(triple)
    print(f"Final unique tokens: {len(final_tokens)}")

    # Show sample
    print("\nSample transitions:")
    for ex in train[:5]:
        print(f"  in:  {ex['state_t']}")
        print(f"  out: {ex['state_t+1']}")
        print()

    # Show top entities/locations
    entity_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    for ex in train:
        for triple in ex["state_t"] + ex["state_t+1"]:
            entity_counts[triple[0]] += 1
            location_counts[triple[2]] += 1
    print("Top 20 entities:", entity_counts.most_common(20))
    print("Top 20 locations:", location_counts.most_common(20))


if __name__ == "__main__":
    main()
