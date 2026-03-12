#!/usr/bin/env python3
"""Generate datasets for staged v18 training from WebNLG triples.

Outputs:
  1. Identity-all dataset: every unique text (original, questions, answers)
     paired with itself for compressor/expander training.
     Format: {"text": "..."}

  2. QA paired dataset: question → full-sentence answer, plus identity pairs.
     Format: {"mode": "identity"|"qa", "input_text": "...", "output_text": "..."}

Usage:
    uv run python scripts/generate_qa_dataset.py \
        --input data/webnlg_multi/train.jsonl \
        --output-dir data/webnlg_multi

    Produces:
      data/webnlg_multi/identity_train.jsonl   (stage 1: compressor/expander)
      data/webnlg_multi/qa_train.jsonl         (stage 2: dynamics)
"""

import argparse
import json
import random
from pathlib import Path

# ── Question templates by attribute ──────────────────────────────────

TEMPLATES = {
    "country": "What country is {entity} in?",
    "leader": "Who is the leader of {entity}?",
    "leader name": "Who is the leader of {entity}?",
    "leader title": "What is the leader title in {entity}?",
    "location": "Where is {entity} located?",
    "birth place": "Where was {entity} born?",
    "death place": "Where did {entity} die?",
    "capital": "What is the capital of {entity}?",
    "genre": "What genre is {entity}?",
    "language": "What language is associated with {entity}?",
    "ingredient": "What is an ingredient of {entity}?",
    "nationality": "What is the nationality of {entity}?",
    "alma mater": "Where did {entity} study?",
    "occupation": "What is the occupation of {entity}?",
    "birth date": "When was {entity} born?",
    "death date": "When did {entity} die?",
    "elevation": "What is the elevation of {entity}?",
    "population": "What is the population of {entity}?",
    "area": "What is the area of {entity}?",
    "currency": "What currency is used in {entity}?",
    "ethnic group": "What ethnic group is associated with {entity}?",
    "club": "What club does {entity} play for?",
    "manager": "Who is the manager of {entity}?",
    "ground": "What is the ground of {entity}?",
    "region": "What region is {entity} in?",
    "city served": "What city does {entity} serve?",
    "runway length": "What is the runway length of {entity}?",
    "runway name": "What is the runway name of {entity}?",
    "is part of": "What is {entity} part of?",
    "status": "What is the status of {entity}?",
    "selected by nasa": "When was {entity} selected by NASA?",
    "time in space": "How much time did {entity} spend in space?",
    "operating organisation": "What organisation operates {entity}?",
}

DEFAULT_TEMPLATE = "What is the {attribute} of {entity}?"

ANSWER_TEMPLATES = {
    "country": "the country of {entity} is {value}.",
    "leader": "the leader of {entity} is {value}.",
    "leader name": "the leader of {entity} is {value}.",
    "leader title": "the leader title in {entity} is {value}.",
    "location": "{entity} is located in {value}.",
    "birth place": "{entity} was born in {value}.",
    "death place": "{entity} died in {value}.",
    "capital": "the capital of {entity} is {value}.",
    "genre": "the genre of {entity} is {value}.",
    "language": "the language associated with {entity} is {value}.",
    "ingredient": "an ingredient of {entity} is {value}.",
    "nationality": "the nationality of {entity} is {value}.",
    "alma mater": "{entity} studied at {value}.",
    "occupation": "the occupation of {entity} is {value}.",
    "birth date": "{entity} was born on {value}.",
    "death date": "{entity} died on {value}.",
    "elevation": "the elevation of {entity} is {value}.",
    "population": "the population of {entity} is {value}.",
    "area": "the area of {entity} is {value}.",
    "currency": "the currency used in {entity} is {value}.",
    "ethnic group": "the ethnic group associated with {entity} is {value}.",
    "club": "{entity} plays for {value}.",
    "manager": "the manager of {entity} is {value}.",
    "ground": "the ground of {entity} is {value}.",
    "region": "{entity} is in the region of {value}.",
    "city served": "{entity} serves the city of {value}.",
    "runway length": "the runway length of {entity} is {value}.",
    "runway name": "the runway name of {entity} is {value}.",
    "is part of": "{entity} is part of {value}.",
    "status": "the status of {entity} is {value}.",
    "selected by nasa": "{entity} was selected by nasa in {value}.",
    "time in space": "{entity} spent {value} in space.",
    "operating organisation": "{entity} is operated by {value}.",
}

DEFAULT_ANSWER_TEMPLATE = "the {attribute} of {entity} is {value}."


def make_question(entity: str, attribute: str, value: str) -> tuple[str, str]:
    attr_lower = attribute.lower().replace("_", " ")
    template = TEMPLATES.get(attr_lower, DEFAULT_TEMPLATE)
    question = template.format(entity=entity, attribute=attr_lower)
    ans_template = ANSWER_TEMPLATES.get(attr_lower, DEFAULT_ANSWER_TEMPLATE)
    answer = ans_template.format(entity=entity, attribute=attr_lower, value=value)
    return question, answer


def make_answer_sentence(entity: str, attribute: str, value: str) -> str:
    """Generate just the answer sentence for a triple."""
    attr_lower = attribute.lower().replace("_", " ")
    ans_template = ANSWER_TEMPLATES.get(attr_lower, DEFAULT_ANSWER_TEMPLATE)
    return ans_template.format(entity=entity, attribute=attr_lower, value=value)


def generate_multi_triple_qa(triples: list[list[str]]) -> list[tuple[str, str]]:
    """Generate multi-triple Q&A pairs from an entry's triples.

    Returns list of (question, answer) tuples for:
      1. 2-hop chains: value of triple A = entity of triple B
      2. Multi-attribute: same entity with 2+ attributes → "tell me about" style
      3. Multi-fact summary: question about the full entry → combined answer
    """
    pairs = []

    # Index triples by entity (lowercased for matching)
    by_entity: dict[str, list[list[str]]] = {}
    for t in triples:
        key = t[0].lower()
        by_entity.setdefault(key, []).append(t)

    # ── 2-hop chains ──────────────────────────────────────────────────
    # If triple A has value V, and triple B has entity V, generate:
    #   Q: "What is the {B.attr} of the {A.attr} of {A.entity}?"
    #   A: "the {A.attr} of {A.entity} is {A.value}. {B answer sentence}"
    for t_a in triples:
        e_a, attr_a, val_a = t_a
        val_key = val_a.lower()
        if val_key in by_entity:
            for t_b in by_entity[val_key]:
                e_b, attr_b, val_b = t_b
                attr_a_lower = attr_a.lower().replace("_", " ")
                attr_b_lower = attr_b.lower().replace("_", " ")
                question = f"What is the {attr_b_lower} of the {attr_a_lower} of {e_a}?"
                ans_a = make_answer_sentence(e_a, attr_a, val_a)
                ans_b = make_answer_sentence(e_b, attr_b, val_b)
                answer = f"{ans_a} {ans_b}"
                pairs.append((question, answer))

    # ── Multi-attribute (same entity, 2+ facts) ──────────────────────
    # Q: "What do you know about {entity}?"
    # A: combined answer sentences
    for entity_key, entity_triples in by_entity.items():
        if len(entity_triples) >= 2:
            entity_name = entity_triples[0][0]  # use original casing
            question = f"What do you know about {entity_name}?"
            answer_parts = []
            for t in entity_triples:
                answer_parts.append(make_answer_sentence(t[0], t[1], t[2]))
            answer = " ".join(answer_parts)
            pairs.append((question, answer))

            # Also generate 2-attribute subset questions for variety
            if len(entity_triples) >= 3:
                subset = random.sample(entity_triples, 2)
                attr1 = subset[0][1].lower().replace("_", " ")
                attr2 = subset[1][1].lower().replace("_", " ")
                question = f"What are the {attr1} and {attr2} of {entity_name}?"
                ans1 = make_answer_sentence(subset[0][0], subset[0][1], subset[0][2])
                ans2 = make_answer_sentence(subset[1][0], subset[1][1], subset[1][2])
                answer = f"{ans1} {ans2}"
                pairs.append((question, answer))

    # ── Full entry summary ────────────────────────────────────────────
    # Q: original text (paragraph) → A: original text (identity with context)
    # Already handled by identity pairs, so skip here.

    return pairs


def generate(input_path: str, output_dir: str, split_name: str = "train",
             max_examples: int = 0) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_texts = set()
    qa_pairs = []

    with open(input_path) as f:
        for line in f:
            ex = json.loads(line)

            # Collect original text
            all_texts.add(ex["text"])

            # Identity pair for QA dataset
            qa_pairs.append({
                "mode": "identity",
                "input_text": ex["text"],
                "output_text": ex["text"],
            })

            # Single-triple Q&A pairs
            for triple in ex["triples"]:
                entity, attribute, value = triple
                question, answer = make_question(entity, attribute, value)
                all_texts.add(question)
                all_texts.add(answer)
                qa_pairs.append({
                    "mode": "qa",
                    "input_text": question,
                    "output_text": answer,
                })

            # Multi-triple Q&A pairs
            if len(ex["triples"]) >= 2:
                multi_pairs = generate_multi_triple_qa(ex["triples"])
                for question, answer in multi_pairs:
                    all_texts.add(question)
                    all_texts.add(answer)
                    qa_pairs.append({
                        "mode": "qa",
                        "input_text": question,
                        "output_text": answer,
                    })

    # ── Identity-all dataset (stage 1: compressor/expander) ──────────
    # Every unique text paired with itself
    identity_examples = [{"text": t} for t in all_texts]
    random.shuffle(identity_examples)
    if max_examples > 0:
        identity_examples = identity_examples[:max_examples]

    identity_path = out_dir / f"identity_{split_name}.jsonl"
    with open(identity_path, "w") as f:
        for ex in identity_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Identity-all: {len(identity_examples)} unique texts → {identity_path}")

    # ── QA paired dataset (stage 2: dynamics) ────────────────────────
    random.shuffle(qa_pairs)
    if max_examples > 0:
        qa_pairs = qa_pairs[:max_examples]

    qa_path = out_dir / f"qa_{split_name}.jsonl"
    with open(qa_path, "w") as f:
        for ex in qa_pairs:
            f.write(json.dumps(ex) + "\n")
    n_id = sum(1 for e in qa_pairs if e["mode"] == "identity")
    n_qa = sum(1 for e in qa_pairs if e["mode"] == "qa")
    print(f"QA paired: {len(qa_pairs)} examples ({n_id} identity, {n_qa} Q&A) → {qa_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate datasets for staged v18 training")
    ap.add_argument("--input", type=str, required=True,
                    help="Input WebNLG JSONL")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Output directory")
    ap.add_argument("--split", type=str, default="train",
                    help="Split name (train/test)")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    generate(args.input, args.output_dir, split_name=args.split,
             max_examples=args.max_examples)


if __name__ == "__main__":
    main()
