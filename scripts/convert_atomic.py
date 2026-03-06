#!/usr/bin/env python3
"""Convert ATOMIC 2020 commonsense tuples to TWM triple-transition format.

Uses an LLM (Anthropic API or local vLLM) to decompose free-text ATOMIC tuples
into structured (entity, attribute, value) triples.

ATOMIC 2020 physical-entity relations (initial focus):
  ObjectUse, AtLocation, MadeUpOf, HasProperty, CapableOf, Desires, NotDesires

Usage:
  # Using Anthropic API
  uv run python scripts/convert_atomic.py \
    --atomic-path data/atomic2020/train.tsv \
    --out-dir data/atomic \
    --api anthropic \
    --max-examples 500

  # Using local vLLM endpoint
  uv run python scripts/convert_atomic.py \
    --atomic-path data/atomic2020/train.tsv \
    --out-dir data/atomic \
    --api vllm \
    --vllm-url http://localhost:8000/v1

  # Using pre-converted cache (no LLM needed)
  uv run python scripts/convert_atomic.py \
    --atomic-path data/atomic2020/train.tsv \
    --out-dir data/atomic \
    --api cache \
    --cache-path data/atomic/conversion_cache.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path


PHYSICAL_RELATIONS = {
    "ObjectUse", "AtLocation", "MadeUpOf", "HasProperty", "CapableOf",
    "Desires", "NotDesires",
}

SOCIAL_RELATIONS = {
    "xIntent", "xReact", "oReact", "xWant", "oWant",
    "xNeed", "xEffect", "oEffect", "xAttr",
}

EVENT_RELATIONS = {
    "isAfter", "isBefore", "HinderedBy", "Causes", "xReason",
    "isFilledBy", "HasSubEvent",
}

# Map ATOMIC relations to TWM-friendly attribute names
RELATION_MAP = {
    "ObjectUse": "use",
    "AtLocation": "location",
    "MadeUpOf": "component",
    "HasProperty": "property",
    "CapableOf": "capability",
    "Desires": "desire",
    "NotDesires": "aversion",
    "xIntent": "intent",
    "xReact": "reaction",
    "oReact": "other_reaction",
    "xWant": "want",
    "oWant": "other_want",
    "xNeed": "need",
    "xEffect": "effect",
    "oEffect": "other_effect",
    "xAttr": "attribute",
    "Causes": "causes",
    "HinderedBy": "hindered_by",
    "isAfter": "after",
    "isBefore": "before",
    "xReason": "reason",
    "isFilledBy": "filled_by",
    "HasSubEvent": "subevent",
}


def load_atomic_tsv(path: str, relations: set[str] | None = None) -> list[dict]:
    """Load ATOMIC 2020 TSV file, optionally filtering by relation."""
    examples = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            head, relation, tail = parts[0], parts[1], parts[2]
            if relations and relation not in relations:
                continue
            if tail.lower() in ("none", ""):
                continue
            examples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
    return examples


def normalize_token(text: str) -> str:
    """Normalize a free-text phrase into a TWM-compatible token."""
    t = text.strip().lower()
    t = re.sub(r'[^a-z0-9\s_]', '', t)
    t = re.sub(r'\s+', '_', t)
    # Truncate very long tokens
    if len(t) > 30:
        t = t[:30]
    return t or "unknown"


def simple_decompose(example: dict) -> dict | None:
    """Rule-based decomposition for physical-entity relations.

    Converts ATOMIC tuples directly without LLM, using structural rules.
    This handles the common case where the mapping is straightforward.
    """
    head = example["head"]
    relation = example["relation"]
    tail = example["tail"]

    entity = normalize_token(head)
    attr = RELATION_MAP.get(relation, normalize_token(relation))
    value = normalize_token(tail)

    if not entity or not value:
        return None

    # For physical relations, state_t has entity with unknown/default state,
    # state_t+1 reveals the property/capability/location.
    state_t = [[entity, attr, "unknown"]]
    state_t1 = [[entity, attr, value]]

    return {"state_t": state_t, "state_t+1": state_t1}


DECOMPOSE_PROMPT = """You are converting ATOMIC 2020 commonsense knowledge into structured triples.

Given an ATOMIC tuple (head, relation, tail), decompose it into TWM state transitions:
- state_t: the "before" state as a list of [entity, attribute, value] triples
- state_t+1: the "after" state as a list of [entity, attribute, value] triples

Rules:
1. Each token must be a single lowercase word or underscore-separated phrase (e.g., "go_store", "buy_groceries")
2. Use 1-4 triples per state
3. Capture the causal/temporal relationship: state_t is the precondition, state_t+1 is the consequence
4. Entity names should be generic (person, object, location) unless specific
5. Values should be descriptive single tokens or short phrases

Example input: head="PersonX goes to the store", relation="xIntent", tail="to buy groceries"
Example output:
{{"state_t": [["person", "action", "go_store"], ["person", "intent", "buy_groceries"]], "state_t+1": [["person", "location", "store"], ["person", "has", "groceries"]]}}

Now convert this ATOMIC tuple:
head="{head}", relation="{relation}", tail="{tail}"

Output ONLY the JSON object, nothing else."""


def llm_decompose_anthropic(examples: list[dict], api_key: str, batch_size: int = 10) -> list[dict | None]:
    """Use Anthropic API to decompose ATOMIC tuples."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: uv pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for i, ex in enumerate(examples):
        prompt = DECOMPOSE_PROMPT.format(**ex)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Normalize all tokens
                for key in ("state_t", "state_t+1"):
                    data[key] = [[normalize_token(t) for t in triple] for triple in data[key]]
                results.append(data)
            else:
                results.append(None)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            results.append(None)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(examples)}")
        time.sleep(0.1)  # Rate limiting

    return results


def llm_decompose_vllm(examples: list[dict], base_url: str, model: str = "default") -> list[dict | None]:
    """Use local vLLM OpenAI-compatible endpoint."""
    try:
        import openai
    except ImportError:
        print("Error: openai package not installed. Run: uv pip install openai")
        sys.exit(1)

    client = openai.OpenAI(base_url=base_url, api_key="dummy")
    results = []

    # Get available models if not specified
    if model == "default":
        models = client.models.list()
        model = models.data[0].id if models.data else "default"

    for i, ex in enumerate(examples):
        prompt = DECOMPOSE_PROMPT.format(**ex)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                for key in ("state_t", "state_t+1"):
                    data[key] = [[normalize_token(t) for t in triple] for triple in data[key]]
                results.append(data)
            else:
                results.append(None)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            results.append(None)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(examples)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert ATOMIC 2020 to TWM format")
    parser.add_argument("--atomic-path", required=True, help="Path to ATOMIC 2020 TSV file")
    parser.add_argument("--out-dir", default="data/atomic", help="Output directory")
    parser.add_argument("--api", choices=["anthropic", "vllm", "simple", "cache"],
                        default="simple",
                        help="Decomposition method (simple=rule-based, no LLM needed)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="default")
    parser.add_argument("--cache-path", default=None, help="Path to conversion cache")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--relations", default="physical",
                        choices=["physical", "social", "event", "all"],
                        help="Which ATOMIC relation categories to convert")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # Select relations
    rel_sets = {
        "physical": PHYSICAL_RELATIONS,
        "social": SOCIAL_RELATIONS,
        "event": EVENT_RELATIONS,
        "all": PHYSICAL_RELATIONS | SOCIAL_RELATIONS | EVENT_RELATIONS,
    }
    target_relations = rel_sets[args.relations]

    # Load ATOMIC data
    print(f"Loading ATOMIC data from {args.atomic_path}...")
    raw = load_atomic_tsv(args.atomic_path, target_relations)
    print(f"  Found {len(raw)} tuples for {args.relations} relations")

    # Sample if needed
    if len(raw) > args.max_examples:
        raw = random.sample(raw, args.max_examples)
        print(f"  Sampled {args.max_examples} examples")

    # Convert
    print(f"Converting with method: {args.api}")
    if args.api == "simple":
        converted = [simple_decompose(ex) for ex in raw]
    elif args.api == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)
        converted = llm_decompose_anthropic(raw, api_key)
    elif args.api == "vllm":
        converted = llm_decompose_vllm(raw, args.vllm_url, args.vllm_model)
    elif args.api == "cache":
        if not args.cache_path:
            print("Error: --cache-path required with --api cache")
            sys.exit(1)
        with open(args.cache_path) as f:
            converted = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown API: {args.api}")

    # Filter out failures
    valid = [c for c in converted if c is not None]
    print(f"  Successfully converted: {len(valid)}/{len(converted)}")

    # Validate triple structure
    cleaned = []
    for ex in valid:
        ok = True
        for key in ("state_t", "state_t+1"):
            if key not in ex or not isinstance(ex[key], list):
                ok = False
                break
            for triple in ex[key]:
                if not (isinstance(triple, list) and len(triple) == 3):
                    ok = False
                    break
                if any(not t or t == "<pad>" for t in triple):
                    ok = False
                    break
        if ok:
            cleaned.append(ex)
    print(f"  After validation: {len(cleaned)} examples")

    # Split into train/dev/test
    random.shuffle(cleaned)
    n_test = max(1, int(len(cleaned) * args.test_frac))
    n_dev = max(1, int(len(cleaned) * args.dev_frac))
    n_train = len(cleaned) - n_test - n_dev

    train_data = cleaned[:n_train]
    dev_data = cleaned[n_train:n_train + n_dev]
    test_data = cleaned[n_train + n_dev:]

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_data), ("dev", dev_data), ("test", test_data)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(data)} examples to {path}")

    # Save conversion cache
    cache_path = out_dir / "conversion_cache.jsonl"
    with open(cache_path, "w") as f:
        for ex in cleaned:
            f.write(json.dumps(ex) + "\n")

    # Print vocab stats
    tokens = set()
    for ex in cleaned:
        for triple in ex["state_t"] + ex["state_t+1"]:
            tokens.update(triple)
    print(f"\n  Total unique tokens: {len(tokens)}")
    print(f"  Sample tokens: {sorted(list(tokens))[:20]}")

    # Print sample conversions
    print("\n  Sample conversions:")
    for ex in cleaned[:5]:
        print(f"    {ex['state_t']} -> {ex['state_t+1']}")


if __name__ == "__main__":
    main()
