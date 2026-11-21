"""Training entry point and experiment runner for Bank Marketing neural models.

Part of Role 1 (Tech Lead) code deliverables.
Allows training Model A, B, or C using a YAML config and seeds.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Local imports
from src.models import build_model_a, build_model_b


def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Bank Marketing MLP models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML experiment config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed in config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    print(f"[*] Running Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"[*] Random Seed: {seed}")
    print(f"[*] Config file: {args.config}")
    print(f"[*] Ready for full data pipeline integration (src/data.py)")


if __name__ == "__main__":
    main()
