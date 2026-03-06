"""Inference wrapper for Triple World Model.

Usage as a library:
    from twm.serve import WorldModel
    wm = WorldModel("results/run")
    next_state = wm.advance([["glass", "state", "full"], ["person_a", "state", "thirsty"]])
    # -> [["glass", "state", "empty"], ["person_a", "state", "satisfied"]]

Usage as CLI:
    python -m twm.serve --checkpoint results/run --interactive
"""

import argparse
import json
from pathlib import Path

import torch

from .vocab import Vocabulary
from .dataset import _sort_triples, _pad_triples, _flatten_triples
from .config import ModelConfig
from .model import TripleWorldModel


class WorldModel:
    def __init__(self, checkpoint_dir: str | Path, device: str | None = None):
        self.run_dir = Path(checkpoint_dir)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.vocab = Vocabulary.load(self.run_dir / "vocab.json")
        self.config = ModelConfig.load(self.run_dir / "config.json")
        self.model = TripleWorldModel(self.config).to(self.device)

        ckpt = self.run_dir / "model_best.pt"
        if not ckpt.exists():
            ckpt = self.run_dir / "model_final.pt"
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
        self.model.train(False)

    def advance(self, state: list[list[str]]) -> list[list[str]]:
        """Predict the next state given current state triples.

        Args:
            state: list of [entity, relation, value] triples

        Returns:
            predicted next-state triples (variable length, <pad> stripped)
        """
        padded = _pad_triples(state, self.config.max_triples)
        ids = _flatten_triples(padded, self.vocab)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_ids = self.model.predict(input_ids)[0].cpu().tolist()

        return self.vocab.decode_triples(pred_ids)

    def advance_n(self, state: list[list[str]], n_steps: int) -> list[list[list[str]]]:
        """Multi-step prediction: advance state n times.

        Returns list of n states (each a list of triples).
        """
        trajectory = []
        current = state
        for _ in range(n_steps):
            current = self.advance(current)
            trajectory.append(current)
        return trajectory


def main():
    parser = argparse.ArgumentParser(description="Triple World Model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to run directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--input", type=str, help="JSON string of input triples")
    parser.add_argument("--steps", type=int, default=1, help="Number of advance steps")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    wm = WorldModel(args.checkpoint, device=args.device)
    print(f"Loaded model ({wm.model.param_count():,} params) on {wm.device}")

    if args.interactive:
        print("\nInteractive mode. Enter triples as JSON, e.g.:")
        print('  [["glass", "state", "full"], ["person_a", "state", "thirsty"]]')
        print("  Type 'quit' to exit.\n")

        while True:
            try:
                line = input("state> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("quit", "exit", "q"):
                break
            if not line:
                continue
            try:
                state = json.loads(line)
                if args.steps == 1:
                    result = wm.advance(state)
                    print(f"  -> {json.dumps(result)}")
                else:
                    trajectory = wm.advance_n(state, args.steps)
                    for i, s in enumerate(trajectory, 1):
                        print(f"  t+{i}: {json.dumps(s)}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Error: {e}")

    elif args.input:
        state = json.loads(args.input)
        if args.steps == 1:
            result = wm.advance(state)
            print(json.dumps(result, indent=2))
        else:
            trajectory = wm.advance_n(state, args.steps)
            for i, s in enumerate(trajectory, 1):
                print(f"t+{i}: {json.dumps(s)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
