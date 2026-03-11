#!/usr/bin/env python3
"""Config-driven training entry point.

Usage:
    uv run python scripts/train.py configs/v18_mini64.json
"""

import sys

from twm.training_config import TrainingConfig
from twm.trainer import Trainer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config.json>")
        sys.exit(1)

    config = TrainingConfig.load(sys.argv[1])
    trainer = Trainer(config)
    trainer.run()
