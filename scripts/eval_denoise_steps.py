"""Assess a checkpoint with varying denoise steps.

Usage:
    uv run python scripts/eval_denoise_steps.py results/v27_joint_qa_lenfix/dynamics_phase1/model_best.pt
"""

import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from twm.text_dynamics_model import TextDynamicsModel
from twm.config import ModelConfig
from twm.domain_bpe import DomainBPETokenizer
from twm.text_pair_dataset import TextPairDataset
from twm.training_eval import assess, format_metrics, print_samples


def main():
    ckpt_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/v27_joint_qa_lenfix.json"

    with open(config_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DomainBPETokenizer.load(config["tokenizer_path"], max_length=config["max_text_tokens"])

    model_config = ModelConfig.from_profile(config["profile"])
    model_config = ModelConfig(
        d_model=config.get("d_model", model_config.d_model),
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        d_ff=config.get("d_model", model_config.d_model) * 4,
        max_triples=config.get("max_triples", model_config.max_triples),
    )

    model = TextDynamicsModel(
        config=model_config, domain_tokenizer=tokenizer,
        text_compressor_layers=config["text_compressor_layers"],
        text_expander_layers=config["text_expander_layers"],
        dynamics_layers=config.get("dynamics_layers", model_config.n_layers),
        max_text_tokens=config["max_text_tokens"],
        dropout=0.0, alpha_min=config["alpha_min"],
        vae=config.get("vae", False),
    )
    model.init_embeddings()

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.train(False)

    ds = TextPairDataset(
        Path(config["data_dir"]) / "qa_test.jsonl",
        tokenizer, max_text_tokens=config["max_text_tokens"],
    )

    for n_steps in [5, 10, 20, 50, 100]:
        m = assess(model, ds, device, tokenizer, n_examples=64, n_steps=n_steps)
        gen_cache = m.pop("_gen", None)
        print(f"\n=== denoise_steps={n_steps} ===")
        print(format_metrics(m))
        if n_steps in [10, 50]:
            print_samples(model, ds, device, tokenizer, n=5, n_steps=n_steps, gen_cache=gen_cache)


if __name__ == "__main__":
    main()
