#!/usr/bin/env python3
"""Prepare WebNLG data for identity advance experiment.

Downloads WebNLG parquet files directly from HuggingFace (bypasses loading
scripts), extracts triples, and writes TWM-format JSONL with state_t ==
state_t+1 (identity advance — reconstruct input).

WebNLG triples: "Subject | Predicate | Object" → TWM: (entity, attr, value)
Underscores → spaces, camelCase split, lowercased, stripped.

Usage:
    uv run python scripts/prepare_webnlg.py --out-dir data/webnlg
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
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase → camel Case
    return text.lower().strip()


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
    """Download parquet file if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-2] + "_" + url.split("/")[-1]
    cached = cache_dir / fname
    if not cached.exists():
        print(f"  Downloading {url}...")
        urlretrieve(url, cached)
    else:
        print(f"  Using cached {cached}")
    return cached


def main():
    ap = argparse.ArgumentParser(description="Prepare WebNLG for identity advance experiment")
    ap.add_argument("--out-dir", type=str, default="data/webnlg")
    ap.add_argument("--max-triples", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"

    import pandas as pd

    # Map: parquet split → output filename
    split_map = {
        "train": "train",
        "validation": "test",
    }

    for parquet_split, out_name in split_map.items():
        url = PARQUET_URLS.get(parquet_split)
        if not url:
            continue

        print(f"\nProcessing {parquet_split} → {out_name}...")
        parquet_path = download_parquet(url, cache_dir)
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

        if len(df) > 0:
            print(f"  Example input: {df.iloc[0]['input'][:3]}")

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
            examples.append({
                "state_t": triples,
                "state_t+1": triples,
            })

        out_path = out_dir / f"{out_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Wrote {len(examples)} examples to {out_path}")

        if examples:
            n_triples = [len(ex["state_t"]) for ex in examples]
            all_entities = set()
            all_attrs = set()
            all_values = set()
            for ex in examples:
                for t in ex["state_t"]:
                    all_entities.add(t[0])
                    all_attrs.add(t[1])
                    all_values.add(t[2])
            print(f"  Triples/example: avg={sum(n_triples)/len(n_triples):.1f}, "
                  f"max={max(n_triples)}")
            print(f"  Unique entities: {len(all_entities)}, "
                  f"attrs: {len(all_attrs)}, values: {len(all_values)}")

    print(f"\nDone. Data written to {out_dir}/")


if __name__ == "__main__":
    main()
