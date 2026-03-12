#!/usr/bin/env python3
"""Prepare WebNLG data for multimodal (triple ↔ text) experiments.

Downloads WebNLG parquet files, extracts paired (triples, text) data,
and writes JSONL with both structured triples and free text references.

Also trains a shared BPE tokenizer over ALL text (triple strings + free text)
for a unified vocabulary.

Usage:
    uv run python scripts/prepare_webnlg_multimodal.py --out-dir data/webnlg_multi
"""

import argparse
import json
import re
from pathlib import Path
from urllib.request import urlretrieve


def clean_phrase(text: str) -> str:
    """Normalize WebNLG phrase: underscores→spaces, camelCase split, lowercase."""
    text = text.strip().strip('"').strip("'")
    text = text.replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.lower().strip()


def clean_text(text: str) -> str:
    """Normalize free text: lowercase, strip extra whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_triple_str(triple_str: str) -> list[str] | None:
    """Parse 'Subject | Predicate | Object' into [entity, attr, value]."""
    parts = triple_str.split("|")
    if len(parts) != 3:
        return None
    entity = clean_phrase(parts[0])
    attr = clean_phrase(parts[1])
    value = clean_phrase(parts[2])
    if entity and attr and value:
        return [entity, attr, value]
    return None


PARQUET_URLS = {
    "train": "https://huggingface.co/api/datasets/GEM/web_nlg/parquet/en/train/0.parquet",
    "validation": "https://huggingface.co/api/datasets/GEM/web_nlg/parquet/en/validation/0.parquet",
    "test": "https://huggingface.co/api/datasets/GEM/web_nlg/parquet/en/test/0.parquet",
}


def download_parquet(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-2] + "_" + url.split("/")[-1]
    cached = cache_dir / fname
    if not cached.exists():
        print(f"  Downloading {url}...")
        urlretrieve(url, cached)
    else:
        print(f"  Using cached {cached}")
    return cached


def train_bpe(all_texts: list[str], vocab_size: int, out_path: Path):
    """Train a BPE tokenizer on all texts and save."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    special_tokens = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]
    # Ensure common punctuation is always in the vocab — questions use '?'
    # which doesn't appear in the declarative WebNLG training texts.
    initial_alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789.,;:!?'-()/ ")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(str(out_path))
    print(f"  BPE tokenizer saved: {out_path} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def main():
    ap = argparse.ArgumentParser(description="Prepare WebNLG for multimodal experiment")
    ap.add_argument("--out-dir", type=str, default="data/webnlg_multi")
    ap.add_argument("--max-triples", type=int, default=8)
    ap.add_argument("--bpe-vocab-size", type=int, default=4000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"

    import pandas as pd

    split_map = {"train": "train", "validation": "test"}
    all_texts_for_bpe = []  # Collect ALL text for BPE training

    for parquet_split, out_name in split_map.items():
        url = PARQUET_URLS.get(parquet_split)
        if not url:
            continue

        print(f"\nProcessing {parquet_split} → {out_name}...")
        parquet_path = download_parquet(url, cache_dir)
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

        examples = []
        for _, row in df.iterrows():
            input_triples = row.get("input", [])

            if input_triples is None or len(input_triples) == 0:
                continue

            triples = []
            for triple_str in input_triples:
                if not isinstance(triple_str, str):
                    continue
                parsed = parse_triple_str(triple_str)
                if parsed:
                    triples.append(parsed)

            if not triples:
                continue

            triples = triples[:args.max_triples]

            # Get text: prefer references (array), fall back to target (string)
            texts = []
            references = row.get("references", [])
            if references is not None and hasattr(references, '__len__') and len(references) > 0:
                for ref in references:
                    if isinstance(ref, str) and ref.strip():
                        cleaned = clean_text(ref)
                        if cleaned:
                            texts.append(cleaned)

            # Fall back to target column
            if not texts:
                target = row.get("target", "")
                if isinstance(target, str) and target.strip():
                    cleaned = clean_text(target)
                    if cleaned:
                        texts.append(cleaned)

            if not texts:
                continue

            # Use first text as primary
            example = {
                "triples": triples,
                "text": texts[0],
                "alt_texts": texts[1:] if len(texts) > 1 else [],
            }
            examples.append(example)

            # Collect texts for BPE
            for t in triples:
                all_texts_for_bpe.extend(t)  # entity, attr, value strings
            all_texts_for_bpe.extend(texts)

        out_path = out_dir / f"{out_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} examples to {out_path}")

        if examples:
            n_triples = [len(ex["triples"]) for ex in examples]
            text_lens = [len(ex["text"].split()) for ex in examples]
            all_entities = set()
            all_attrs = set()
            all_values = set()
            for ex in examples:
                for t in ex["triples"]:
                    all_entities.add(t[0])
                    all_attrs.add(t[1])
                    all_values.add(t[2])
            print(f"  Triples/example: avg={sum(n_triples)/len(n_triples):.1f}, "
                  f"max={max(n_triples)}")
            print(f"  Text words/example: avg={sum(text_lens)/len(text_lens):.1f}, "
                  f"max={max(text_lens)}")
            print(f"  Unique entities: {len(all_entities)}, "
                  f"attrs: {len(all_attrs)}, values: {len(all_values)}")

    # Train shared BPE tokenizer
    print(f"\nTraining shared BPE tokenizer ({args.bpe_vocab_size} tokens)...")
    print(f"  Training on {len(all_texts_for_bpe)} text segments")
    bpe_path = out_dir / "shared_bpe_tokenizer.json"
    train_bpe(all_texts_for_bpe, args.bpe_vocab_size, bpe_path)

    print(f"\nDone. Data written to {out_dir}/")


if __name__ == "__main__":
    main()
