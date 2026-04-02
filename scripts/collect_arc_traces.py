#!/usr/bin/env python3
"""Collect random agent traces from ARC-AGI-3 games.

Runs random actions on each game environment and records
(frame, action, frame) transitions as chain data for TWM training.

Frames are downsampled from 64x64 to 32x32 (2x2 blocks are ~95% uniform).
Each frame is encoded as a hex string (0-f per pixel, 1024 chars).

Usage:
  uv run python scripts/collect_arc_traces.py --episodes 100
  uv run python scripts/collect_arc_traces.py --episodes 100 --games ls20,ar25
"""

import argparse
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
from arcengine import ARCBaseGame, ActionInput, GameAction


ACTIONS = [
    GameAction.ACTION1,  # up
    GameAction.ACTION2,  # down
    GameAction.ACTION3,  # left
    GameAction.ACTION4,  # right
    GameAction.ACTION5,  # perform action
]

ACTION_NAMES = {
    GameAction.RESET: "reset",
    GameAction.ACTION1: "up",
    GameAction.ACTION2: "down",
    GameAction.ACTION3: "left",
    GameAction.ACTION4: "right",
    GameAction.ACTION5: "action",
    GameAction.ACTION6: "click",
    GameAction.ACTION7: "undo",
}


def load_game(game_dir: str) -> ARCBaseGame | None:
    for f in os.listdir(game_dir):
        if f.endswith(".py"):
            spec = importlib.util.spec_from_file_location(f[:-3], os.path.join(game_dir, f))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
                    return obj()
    return None


def get_frame(game: ARCBaseGame, downsample: int = 2) -> np.ndarray:
    pixels = np.array(game.get_pixels(0, 0, 64, 64), dtype=np.uint8)
    if downsample > 1:
        pixels = pixels[::downsample, ::downsample]
    return pixels


def frame_to_hex(frame: np.ndarray) -> str:
    return "".join(format(v, "x") for v in frame.flatten())


def step(game: ARCBaseGame, action: GameAction) -> np.ndarray:
    game.perform_action(ActionInput(id=action))
    return get_frame(game)


def collect_traces(
    game: ARCBaseGame,
    game_id: str,
    n_episodes: int = 100,
    max_steps: int = 20,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    chains = []

    for ep in range(n_episodes):
        step(game, GameAction.RESET)
        frames = [get_frame(game)]
        actions = []

        for t in range(max_steps):
            a = rng.choice(ACTIONS)
            f = step(game, a)

            # Only record if something changed
            if (frames[-1] != f).any():
                actions.append(ACTION_NAMES[a])
                frames.append(f)

        if len(frames) < 2:
            continue

        # Convert to chain format: each step is "action:hex_frame"
        chain = [frame_to_hex(frames[0])]
        for i, a in enumerate(actions):
            chain.append(f"{a}:{frame_to_hex(frames[i + 1])}")

        chains.append({
            "chain": chain,
            "mode": 0,  # advance
            "source": game_id,
        })

        # Also create sub-chains (length 2, 3, ..., N)
        for end in range(2, len(chain) + 1):
            chains.append({
                "chain": chain[:end],
                "mode": 0,
                "source": game_id,
            })

        # Identity chain from first frame
        hex0 = frame_to_hex(frames[0])
        chains.append({
            "chain": [hex0, hex0, hex0],
            "mode": 2,  # identity
            "source": game_id,
        })

    return chains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", default="data/arc-prize-2026-arc-agi-3/environment_files")
    parser.add_argument("--out-dir", default="data/arc_agi_3")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--games", default=None, help="Comma-separated game IDs (default: all)")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env_dir = Path(args.env_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all games
    if args.games:
        game_ids = args.games.split(",")
    else:
        game_ids = sorted([d for d in os.listdir(env_dir) if os.path.isdir(env_dir / d)])

    all_chains = []

    for game_id in game_ids:
        game_path = env_dir / game_id
        subdirs = [d for d in os.listdir(game_path) if os.path.isdir(game_path / d)]
        if not subdirs:
            print(f"  SKIP {game_id}: no subdirs")
            continue

        game_dir = str(game_path / subdirs[0])

        try:
            game = load_game(game_dir)
        except Exception as e:
            print(f"  SKIP {game_id}: {e}")
            continue

        if game is None:
            print(f"  SKIP {game_id}: no game class found")
            continue

        try:
            chains = collect_traces(
                game, game_id,
                n_episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            all_chains.extend(chains)
            print(f"  {game_id}: {len(chains)} chains")
        except Exception as e:
            print(f"  SKIP {game_id}: {e}")
            continue

    print(f"\nTotal: {len(all_chains)} chains from {len(game_ids)} games")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(all_chains)
    n_test = int(len(all_chains) * args.test_frac)
    test = all_chains[:n_test]
    train = all_chains[n_test:]

    # Stats
    mode_counts = {}
    for c in all_chains:
        m = c.get("mode", "?")
        mode_counts[m] = mode_counts.get(m, 0) + 1
    for m, cnt in sorted(mode_counts.items()):
        print(f"  mode {m}: {cnt}")

    chain_lens = [len(c["chain"]) for c in all_chains]
    print(f"  chain lengths: min={min(chain_lens)}, max={max(chain_lens)}, avg={sum(chain_lens)/len(chain_lens):.1f}")
    print(f"  chars per step: {len(all_chains[0]['chain'][0])}")

    for name, data in [("train", train), ("test", test)]:
        path = out_dir / f"arc_{name}.jsonl"
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        print(f"Wrote {path}: {len(data)}")


if __name__ == "__main__":
    main()
