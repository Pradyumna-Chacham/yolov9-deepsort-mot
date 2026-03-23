from __future__ import annotations

import argparse

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("Benchmark scaffold ready.")
    print(f"Using detector={cfg.detector['name']} tracker={cfg.tracker['name']}")


if __name__ == "__main__":
    main()