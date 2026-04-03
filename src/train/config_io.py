"""Shared JSON config helpers for HunyuanVideo fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.data.schema import SplitPolicy
from src.model.action_encoder import GamepadActionEncoder


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at: {path}")
    return raw


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter configuration."""

    rank: int = 128
    alpha: int = 128
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")


@dataclass(frozen=True)
class VideoGenConfig:
    """Full configuration for HunyuanVideo fine-tuning."""

    dataset_path: str
    output_dir: str
    model_id: str

    training_type: str = "lora"
    lora: LoRAConfig = LoRAConfig()

    resolution: tuple[int, int] = (480, 720)
    num_frames: int = 49

    learning_rate: float = 2e-5
    num_train_steps: int = 5000
    gradient_accumulation_steps: int = 4
    batch_size: int = 1
    gradient_checkpointing: bool = True
    quantization: str = "nf4"
    mixed_precision: str = "bf16"

    action_encoding: str = "text"
    prompt_template: str = "Gameplay video of {game}."
    joystick_deadzone: float = 0.2

    split_seed: int = 42
    split_policy: SplitPolicy = SplitPolicy(train=0.8, val=0.1, test=0.1)

    game_filter: Optional[str] = None
    max_train_chunks: Optional[int] = None
    max_val_chunks: Optional[int] = None
    use_processed_actions: bool = True

    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42


def load_videogen_config(path: Path) -> VideoGenConfig:
    """Parse a finetune JSON config into a typed ``VideoGenConfig``."""
    raw = load_json_object(path)
    required = ("dataset_path", "output_dir", "model_id")
    for key in required:
        if key not in raw:
            raise ValueError(f"config missing required key: {key}")

    lora_cfg = LoRAConfig(
        rank=int(raw.get("lora_rank", 128)),
        alpha=int(raw.get("lora_alpha", 128)),
        target_modules=tuple(raw.get("target_modules", ["to_q", "to_k", "to_v", "to_out.0"])),
    )

    sp_raw = raw.get("split_policy", {})
    split_policy = SplitPolicy(
        train=float(sp_raw.get("train", 0.8)),
        val=float(sp_raw.get("val", 0.1)),
        test=float(sp_raw.get("test", 0.1)),
    )

    res_raw = raw.get("resolution", [480, 720])
    resolution = (int(res_raw[0]), int(res_raw[1]))

    return VideoGenConfig(
        dataset_path=str(raw["dataset_path"]),
        output_dir=str(raw["output_dir"]),
        model_id=str(raw["model_id"]),
        training_type=str(raw.get("training_type", "lora")),
        lora=lora_cfg,
        resolution=resolution,
        num_frames=int(raw.get("num_frames", 49)),
        learning_rate=float(raw.get("learning_rate", 2e-5)),
        num_train_steps=int(raw.get("num_train_steps", 5000)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 4)),
        batch_size=int(raw.get("batch_size", 1)),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", True)),
        quantization=str(raw.get("quantization", "nf4")),
        mixed_precision=str(raw.get("mixed_precision", "bf16")),
        action_encoding=str(raw.get("action_encoding", "text")),
        prompt_template=str(raw.get("prompt_template", "Gameplay video of {game}.")),
        joystick_deadzone=float(raw.get("joystick_deadzone", 0.2)),
        split_seed=int(raw.get("split_seed", 42)),
        split_policy=split_policy,
        game_filter=raw.get("game_filter"),
        max_train_chunks=int(raw["max_train_chunks"]) if raw.get("max_train_chunks") is not None else None,
        max_val_chunks=int(raw["max_val_chunks"]) if raw.get("max_val_chunks") is not None else None,
        use_processed_actions=bool(raw.get("use_processed_actions", True)),
        logging_steps=int(raw.get("logging_steps", 10)),
        save_steps=int(raw.get("save_steps", 500)),
        seed=int(raw.get("seed", 42)),
    )


def build_action_encoder(config: VideoGenConfig) -> GamepadActionEncoder:
    """Construct a ``GamepadActionEncoder`` from config."""
    return GamepadActionEncoder(joystick_deadzone=config.joystick_deadzone)
