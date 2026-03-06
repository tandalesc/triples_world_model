#!/usr/bin/env python3
"""Download ATOMIC 2020 dataset from HuggingFace.

ATOMIC 2020 is a commonsense knowledge graph with 1.33M tuples across 23 relation types.
Original paper: Hwang et al., "(Comet-)Atomic 2020: On Symbolic and Neural Commonsense
Knowledge Graphs", AAAI 2021.

The dataset is hosted on HuggingFace: allenai/atomic2020_data

Usage:
    uv run python scripts/download_atomic.py
    uv run python scripts/download_atomic.py --out-dir data/atomic_raw
"""

import argparse
import csv
import gzip
import os
from pathlib import Path

import requests


HF_BASE = "https://huggingface.co/datasets/allenai/atomic2020_data/resolve/main"
FILES = {
    "train.tsv.gz": f"{HF_BASE}/train.tsv.gz",
    "dev.tsv.gz": f"{HF_BASE}/dev.tsv.gz",
    "test.tsv.gz": f"{HF_BASE}/test.tsv.gz",
}


def _get_hf_token() -> str | None:
    """Get HuggingFace token from env or cached login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    # Check cached token from `huggingface-cli login`
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def download_file(url: str, dest: Path, token: str | None = None) -> None:
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url} ...")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code == 401:
        print("  ERROR: 401 Unauthorized. This is a gated dataset.")
        print("  1. Go to https://huggingface.co/datasets/allenai/atomic2020_data")
        print("     and accept the terms")
        print("  2. Set HF_TOKEN env var or run `huggingface-cli login`")
        resp.raise_for_status()
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"  Saved: {dest} ({size_mb:.1f} MB)")


def decompress_and_stats(gz_path: Path, tsv_path: Path) -> dict:
    """Decompress .tsv.gz and return basic stats."""
    if tsv_path.exists():
        print(f"  Already decompressed: {tsv_path}")
    else:
        print(f"  Decompressing {gz_path} ...")
        with gzip.open(gz_path, "rt") as fin, open(tsv_path, "w") as fout:
            fout.write(fin.read())

    # Count lines and relations
    relations = set()
    n_lines = 0
    with open(tsv_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                relations.add(parts[1])
                n_lines += 1

    return {"lines": n_lines, "relations": sorted(relations)}


def main():
    parser = argparse.ArgumentParser(description="Download ATOMIC 2020 dataset")
    parser.add_argument("--out-dir", default="data/atomic_raw", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ATOMIC 2020 from HuggingFace...")
    for fname, url in FILES.items():
        download_file(url, out_dir / fname)

    print("\nDecompressing and analyzing...")
    total_tuples = 0
    all_relations = set()
    for fname in FILES:
        gz_path = out_dir / fname
        tsv_path = out_dir / fname.replace(".gz", "")
        stats = decompress_and_stats(gz_path, tsv_path)
        total_tuples += stats["lines"]
        all_relations.update(stats["relations"])
        print(f"  {tsv_path.name}: {stats['lines']:,} tuples")

    print(f"\nTotal: {total_tuples:,} tuples across {len(all_relations)} relations")
    print(f"Relations: {', '.join(sorted(all_relations))}")

    # Write a quick summary
    summary_path = out_dir / "README.md"
    with open(summary_path, "w") as f:
        f.write(f"# ATOMIC 2020\n\n")
        f.write(f"Downloaded from: {HF_BASE}\n\n")
        f.write(f"Total tuples: {total_tuples:,}\n")
        f.write(f"Relations ({len(all_relations)}): {', '.join(sorted(all_relations))}\n")

    print(f"\nDone. Data saved to {out_dir}/")
    print(f"Next: run `uv run python scripts/convert_atomic.py --atomic-path {out_dir}/train.tsv --out-dir data/atomic`")


if __name__ == "__main__":
    main()
