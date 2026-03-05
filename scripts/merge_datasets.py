"""Merge handwritten and ProPara training data into combined files."""

import json
from pathlib import Path


def merge(output_path: Path, *input_paths: Path) -> int:
    """Merge multiple JSONL files, keeping only state_t and state_t+1 fields."""
    examples = []
    for p in input_paths:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        n = 0
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                examples.append({
                    "state_t": d["state_t"],
                    "state_t+1": d["state_t+1"],
                })
                n += 1
        print(f"  {p.name}: {n} examples")

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def main():
    data = Path("data")
    combined = data / "combined"
    combined.mkdir(exist_ok=True)

    print("Merging train:")
    n = merge(
        combined / "train.jsonl",
        data / "train.jsonl",
        data / "propara_train.jsonl",
        data / "openpi_train.jsonl",
        data / "context_dependent_train.jsonl",
    )
    print(f"  TOTAL: {n}\n")

    # Keep original test sets separate — they test different things
    # Handwritten test_comp/test_seen test compositional generalization
    # ProPara dev/test have different processes
    print("Copying test sets:")
    for src_name, dst_name in [("test_comp_v2.jsonl", "test_comp.jsonl"),
                                ("test_seen_v2.jsonl", "test_seen.jsonl"),
                                ("test_context.jsonl", "test_context.jsonl")]:
        src = data / src_name
        name = dst_name
        dst = combined / name
        if src.exists():
            with open(src) as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)
            n = content.strip().count("\n") + 1
            print(f"  {name}: {n} examples")

    # Additional test sets from ProPara and OpenPI
    for name in ["propara_dev.jsonl", "openpi_dev.jsonl"]:
        src = data / name
        dst = combined / name
        if src.exists():
            with open(src) as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)
            n = content.strip().count("\n") + 1 if content.strip() else 0
            print(f"  {name}: {n} examples")

    # Vocab stats
    tokens = set()
    for f in combined.glob("*.jsonl"):
        for line in open(f):
            d = json.loads(line)
            for triple in d["state_t"] + d["state_t+1"]:
                tokens.update(triple)
    print(f"\nTotal unique tokens: {len(tokens)}")


if __name__ == "__main__":
    main()
