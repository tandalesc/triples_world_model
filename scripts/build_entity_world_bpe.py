#!/usr/bin/env python3
"""Build a 512-token domain BPE on entity-world chain text -> data/entity_world/bpe_512.json.

Adapts scripts/build_jepa_bpe.py (same ByteLevel BPE recipe, same special-token order so
<pad>=0 matches DomainBPETokenizer.PAD_ID). The existing GLUCOSE 512 BPE fragments the
entity-world vocabulary (kettle, lamp, "worn out", terrarium, ...) into byte-level pieces
(~13 single-char fragments/state, ~40 tokens/state) because it never saw those surface
forms. This dedicated BPE is trained on the entity-world train + OOD splits so device /
container / plant vocabulary is covered.

Train types are in `train.jsonl`; OOD splits are included at BUILD time only so OOD surface
forms (puppy / robot pet / terrarium) tokenize cleanly — this is vocabulary coverage, not
label leakage (the BPE sees no dynamics, only character statistics).

Usage (from repo root, AFTER generate_entity_world.py):
    uv run python scripts/build_entity_world_bpe.py
"""

import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

DATA_DIR = Path("data/entity_world")
OUT_PATH = DATA_DIR / "bpe_512.json"
VOCAB_SIZE = 512
MAX_TEXT_TOKENS = 64
SPECIAL_TOKENS = ["<pad>", "<mask>", "<unk>", "<bos>", "<eos>"]  # <pad>=0


def collect_texts(data_dir: Path) -> list[str]:
    texts: list[str] = []
    # All splits for full vocabulary coverage (character stats only, no dynamics).
    for fname in ["train.jsonl", "test_iid.jsonl", "test_ood_near.jsonl", "test_ood_far.jsonl"]:
        path = data_dir / fname
        if not path.exists():
            continue
        n_before = len(texts)
        with open(path) as f:
            for line in f:
                for step_text in json.loads(line)["chain"]:
                    texts.append(step_text)
        print(f"  Read {fname}: +{len(texts) - n_before} state texts")
    return texts


def build_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    initial_alphabet = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,;:!?'\"()-/_@# "
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def coverage_report(tokenizer: Tokenizer, texts: list[str], max_tokens: int) -> None:
    lengths = [len(tokenizer.encode(t).ids) for t in texts]
    n = len(lengths)
    n_over = sum(1 for l in lengths if l > max_tokens)
    sorted_lens = sorted(lengths)
    print(f"\n=== Coverage (vocab={tokenizer.get_vocab_size()}, max_tokens={max_tokens}) ===")
    print(f"  States   : {n:,}")
    print(f"  Mean tok : {statistics.mean(lengths):.1f}")
    print(f"  P95 tok  : {sorted_lens[int(n * 0.95)]}")
    print(f"  Max tok  : {max(lengths)}")
    print(f"  > {max_tokens} tok : {n_over:,} ({100 * n_over / n:.2f}%)")


def main() -> None:
    print(f"Building entity-world domain BPE (vocab={VOCAB_SIZE}) -> {OUT_PATH}")
    texts = collect_texts(DATA_DIR)
    if not texts:
        raise RuntimeError(
            f"No texts under {DATA_DIR}. Run scripts/generate_entity_world.py first.")
    print(f"  Total state texts: {len(texts):,}")

    tokenizer = build_tokenizer(texts, VOCAB_SIZE)
    pad_id = tokenizer.token_to_id("<pad>")
    print(f"\nTrained. vocab={tokenizer.get_vocab_size()}  <pad>={pad_id}")
    if pad_id != 0:
        print("  WARNING: <pad> != 0; DomainBPETokenizer.PAD_ID=0 expects id 0.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_PATH))
    print(f"Saved: {OUT_PATH}")

    test_text = "Someone_A feeds the dog. The dog feel(s) hungry. The dog is messy."
    enc = tokenizer.encode(test_text)
    print(f"\nRound-trip:\n  in : {test_text}\n  out: {tokenizer.decode(enc.ids).strip()}")

    coverage_report(tokenizer, texts, MAX_TEXT_TOKENS)
    print("\nDone.")


if __name__ == "__main__":
    main()
