"""Fine-tune HunyuanVideo with LoRA on NitroGen gameplay data.

Usage (from repo root)::

    python3.12 -m scripts.finetune --config configs/finetune.example.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.train.config_io import load_videogen_config
from src.train.hf_finetune import run_hunyuanvideo_lora_finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune HunyuanVideo with LoRA on NitroGen data.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to finetune JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_videogen_config(args.config)
    summary = run_hunyuanvideo_lora_finetune(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
