"""Use a local LLM to normalize OpenPI values into cleaner tokens.

Sends parallel batches of raw values to the LLM and asks it to normalize them
to short, consistent tokens suitable for a world model vocabulary.

Usage:
    python scripts/normalize_openpi_llm.py
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


LLM_URL = "http://192.168.1.194:8001/v1/chat/completions"
MODEL = "ramblerun/Multimodal-AI"
PARALLEL = 6
BATCH_SIZE = 50
TIMEOUT = 60


def normalize_batch(values: list[str], batch_id: int, retries: int = 2) -> tuple[int, dict[str, str]]:
    """Ask the LLM to normalize a batch of OpenPI values.

    Returns (batch_id, mapping from original value -> normalized token).
    """
    prompt = """You are normalizing state values for a structured knowledge base.
For each value below, output a short normalized token (1-3 words, lowercase, underscores for spaces).
Rules:
- Remove filler words (now, very, quite, somewhat, being, getting)
- Simplify to the core concept: "now changed to fighting position" → "fighting_position"
- Use common antonym pairs consistently: hot/cold, wet/dry, full/empty, on/off, open/closed
- "in X" locations become "in_X": "in the bowl" → "in_bowl"
- "on X" locations become "on_X": "on the table" → "on_table"
- Keep it short: max 3 words/tokens

Output ONLY a JSON object mapping input → normalized output. No explanation.

Values to normalize:
"""
    prompt += json.dumps(values)

    for attempt in range(retries):
        try:
            resp = requests.post(LLM_URL, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4096,
            }, timeout=TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Extract JSON from response (might have markdown fences)
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            # Handle trailing commas before closing brace
            content = re.sub(r",\s*}", "}", content)
            result = json.loads(content)

            # Validate and clean the output
            mapping = {}
            for orig, norm in result.items():
                if isinstance(norm, str):
                    norm = norm.strip().lower()
                    norm = re.sub(r"\s+", "_", norm)
                    norm = re.sub(r"[^a-z0-9_]", "", norm)
                    norm = norm.strip("_")
                    if norm:
                        mapping[orig] = norm
            return batch_id, mapping
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == retries - 1:
                print(f"  Batch {batch_id} FAILED after {retries} attempts: {e}")

    return batch_id, {}


def main():
    # Collect all unique values from OpenPI that need normalization
    gold_dir = Path("data/openpi_raw/data/gold")
    all_values = set()

    for split in ["train", "dev", "test"]:
        path = gold_dir / split / "id_answers_metadata.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                for ann in d["answers_metadata"]:
                    for field in ["entity", "attr", "before", "after"]:
                        val = ann[field].strip().lower()
                        if val and len(val) <= 50:
                            all_values.add(val)

    print(f"Total unique values to normalize: {len(all_values)}")

    # Check if LLM is reachable
    try:
        resp = requests.get("http://192.168.1.194:8001/v1/models", timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json()["data"]]
        print(f"LLM available: {models}")
    except requests.RequestException as e:
        print(f"LLM not reachable: {e}")
        return

    # Build batches
    values_list = sorted(all_values)
    batches = []
    for i in range(0, len(values_list), BATCH_SIZE):
        batches.append(values_list[i:i + BATCH_SIZE])

    total_batches = len(batches)
    print(f"Processing {len(values_list)} values in {total_batches} batches "
          f"({PARALLEL} parallel, batch_size={BATCH_SIZE})")

    # Process in parallel
    mapping = {}
    done = 0

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = {
            pool.submit(normalize_batch, batch, idx): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_id, batch_mapping = future.result()
            mapping.update(batch_mapping)
            done += 1
            if done % 10 == 0 or done == total_batches:
                print(f"  Progress: {done}/{total_batches} batches, "
                      f"{len(mapping)} values normalized")

    print(f"\nNormalized {len(mapping)}/{len(all_values)} values")

    # Save mapping
    output_path = Path("data/openpi_raw/value_normalizations.json")
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Saved to {output_path}")

    # Stats
    normalized_tokens = set(mapping.values())
    print(f"Unique normalized tokens: {len(normalized_tokens)} "
          f"(compression: {len(all_values)} → {len(normalized_tokens)})")

    # Show samples
    print("\nSample normalizations:")
    samples = [(o, n) for o, n in sorted(mapping.items()) if o != n][:30]
    for orig, norm in samples:
        print(f"  '{orig}' → '{norm}'")


if __name__ == "__main__":
    main()
