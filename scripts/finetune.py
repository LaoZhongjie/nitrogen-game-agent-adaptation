"""Fine-tune a Hugging Face image classifier on manifest frames (frame → action id)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.schema import SplitName
from src.train.config_io import build_vocab_aligner_from_finetune_config, load_json_object
from src.train.hf_finetune import (
    HFFinetuneParams,
    build_label2id,
    collect_aligned_frame_rows,
    run_image_classification_finetune,
)


@dataclass(slots=True, frozen=True)
class FineTuneConfig:
    """JSON-driven fine-tuning configuration."""

    manifest_path: str
    output_dir: str
    model_id: str
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    logging_steps: int = 10
    seed: int = 42
    max_train_clips: int | None = None
    max_val_clips: int | None = None
    run_validation: bool = True


def load_finetune_config(path: Path) -> FineTuneConfig:
    """Parse finetune JSON into a typed config."""
    raw = load_json_object(path)
    required = ("manifest_path", "output_dir", "model_id")
    for key in required:
        if key not in raw:
            raise ValueError(f"config missing required key: {key}")
    return FineTuneConfig(
        manifest_path=str(raw["manifest_path"]),
        output_dir=str(raw["output_dir"]),
        model_id=str(raw["model_id"]),
        num_train_epochs=float(raw.get("num_train_epochs", 3.0)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(raw.get("per_device_eval_batch_size", 4)),
        learning_rate=float(raw.get("learning_rate", 5e-5)),
        logging_steps=int(raw.get("logging_steps", 10)),
        seed=int(raw.get("seed", 42)),
        max_train_clips=int(raw["max_train_clips"]) if raw.get("max_train_clips") is not None else None,
        max_val_clips=int(raw["max_val_clips"]) if raw.get("max_val_clips") is not None else None,
        run_validation=bool(raw.get("run_validation", True)),
    )


def run_finetune(config: FineTuneConfig, raw_config: dict[str, Any]) -> dict[str, Any]:
    """Load data, train, and write artifacts under ``output_dir``."""
    manifest_path = Path(config.manifest_path)
    aligner = build_vocab_aligner_from_finetune_config(raw_config)
    unknown_id = aligner.unknown_action_id

    train_rows = collect_aligned_frame_rows(
        manifest_path=manifest_path,
        split=SplitName.TRAIN,
        aligner=aligner,
        max_samples=config.max_train_clips,
    )
    if len(train_rows) == 0:
        raise ValueError("no training frame rows after alignment; check manifest train split.")

    label2id = build_label2id([r.action_id for r in train_rows], unknown_id)

    eval_rows = None
    if config.run_validation:
        eval_rows = collect_aligned_frame_rows(
            manifest_path=manifest_path,
            split=SplitName.VAL,
            aligner=aligner,
            max_samples=config.max_val_clips,
        )
        if len(eval_rows) == 0:
            eval_rows = None

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params = HFFinetuneParams(
        model_id=config.model_id,
        output_dir=out,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
        logging_steps=config.logging_steps,
    )
    trainer = run_image_classification_finetune(
        train_rows=train_rows,
        eval_rows=eval_rows,
        label2id=label2id,
        unknown_action_id=unknown_id,
        params=params,
    )

    finetune_config_path = out / "finetune_config.json"
    with finetune_config_path.open("w", encoding="utf-8") as fp:
        json.dump(raw_config, fp, indent=2)
        fp.write("\n")

    metrics: dict[str, Any] = {
        "train_frame_count": len(train_rows),
        "eval_frame_count": 0 if eval_rows is None else len(eval_rows),
        "num_labels": len(label2id),
        "output_dir": str(out.resolve()),
        "hf_model_dir": str((out / "hf_model").resolve()),
    }
    if eval_rows is not None and trainer.state.log_history:
        last_eval = next(
            (h for h in reversed(trainer.state.log_history) if "eval_accuracy" in h),
            None,
        )
        if last_eval is not None:
            metrics["final_eval_accuracy"] = last_eval["eval_accuracy"]

    metrics_path = out / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
        fp.write("\n")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face image classifier on clip-manifest frames.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to finetune JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_json_object(args.config)
    cfg = load_finetune_config(args.config)
    summary = run_finetune(cfg, raw)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
