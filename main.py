"""Run the full pipeline: build manifest → fine-tune → predict → evaluate.

Edit the ``PipelinePaths`` values below, then from the repo root:

    python3.12 main.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Edit paths and hyperparameters below for your data and hardware.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    """Filesystem locations for one end-to-end run."""

    raw_episodes_dir: Path
    manifest_path: Path
    run_output_dir: Path
    finetune_config_template: Path


PATHS = PipelinePaths(
    raw_episodes_dir=Path("data/raw"),
    manifest_path=Path("data/processed/manifest.json"),
    run_output_dir=Path("outputs/pipeline_run"),
    finetune_config_template=Path("configs/finetune.example.json"),
)

# Split used for inference and evaluation (must match manifest clip splits).
EVAL_SPLIT: str = "val"

# Manifest construction: sliding window and train/val/test episode ratios.
DATASET_SEED: int = 0
CLIP_LENGTH: int = 16
STRIDE: int = 16
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

# Inference batch size in frames.
PREDICT_BATCH_SIZE: int = 8


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def run_pipeline() -> None:
    """Execute build → train → predict → report."""
    root = _repo_root()
    sys.path.insert(0, str(root))

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from scripts.finetune import load_finetune_config, run_finetune
    from src.data.loader import ManifestDataset
    from src.data.schema import SplitName, SplitPolicy
    from src.eval.pipeline import build_evaluation_records_from_manifest_predictions, load_prediction_records
    from src.eval.report import build_evaluation_report, write_evaluation_report
    from src.train.config_io import build_vocab_aligner_from_finetune_config, load_json_object
    from src.train.hf_finetune import manifest_input_root, predict_clip_actions

    paths = PipelinePaths(
        raw_episodes_dir=(root / PATHS.raw_episodes_dir).resolve(),
        manifest_path=(root / PATHS.manifest_path).resolve(),
        run_output_dir=(root / PATHS.run_output_dir).resolve(),
        finetune_config_template=(root / PATHS.finetune_config_template).resolve(),
    )

    if not paths.finetune_config_template.is_file():
        raise FileNotFoundError(f"missing finetune template: {paths.finetune_config_template}")

    # 1) Manifest
    print("[1/4] Building dataset manifest...")
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ds_cfg = BuildDatasetConfig(
        input_root=str(paths.raw_episodes_dir),
        output_manifest_path=str(paths.manifest_path),
        seed=DATASET_SEED,
        split_policy=SplitPolicy(train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO),
        clip_length=CLIP_LENGTH,
        stride=STRIDE,
    )
    manifest_obj = build_manifest(ds_cfg)
    with paths.manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest_obj, fp, indent=2)
        fp.write("\n")
    print(f"      Wrote {paths.manifest_path}")

    # 2) Fine-tune (effective config lives under run dir)
    print("[2/4] Fine-tuning...")
    paths.run_output_dir.mkdir(parents=True, exist_ok=True)
    raw_ft = dict(load_json_object(paths.finetune_config_template))
    raw_ft["manifest_path"] = str(paths.manifest_path)
    raw_ft["output_dir"] = str(paths.run_output_dir)
    ft_cfg_path = paths.run_output_dir / "finetune_config.json"
    with ft_cfg_path.open("w", encoding="utf-8") as fp:
        json.dump(raw_ft, fp, indent=2)
        fp.write("\n")

    cfg = load_finetune_config(ft_cfg_path)
    train_summary = run_finetune(cfg, raw_ft)
    print(json.dumps(train_summary, indent=2))

    # 3) Predict on EVAL_SPLIT
    print(f"[3/4] Predicting (split={EVAL_SPLIT})...")
    split = SplitName(EVAL_SPLIT)
    aligner = build_vocab_aligner_from_finetune_config(raw_ft)
    samples = list(ManifestDataset(paths.manifest_path, split=split))
    if len(samples) == 0:
        raise ValueError(f"no clips for split={EVAL_SPLIT}; check manifest splits.")
    input_root = manifest_input_root(paths.manifest_path)
    preds = predict_clip_actions(
        model_dir=paths.run_output_dir,
        samples=samples,
        input_root=input_root,
        aligner=aligner,
        batch_size=PREDICT_BATCH_SIZE,
    )
    predictions_path = paths.run_output_dir / f"predictions_{EVAL_SPLIT}.json"
    with predictions_path.open("w", encoding="utf-8") as fp:
        json.dump(preds, fp, indent=2)
        fp.write("\n")
    print(f"      Wrote {predictions_path} ({len(preds)} clips)")

    # 4) Evaluate
    print("[4/4] Evaluation report...")
    predictions = load_prediction_records(predictions_path)
    records = build_evaluation_records_from_manifest_predictions(
        manifest_path=paths.manifest_path,
        predictions=predictions,
        split=split,
        require_all_clips=True,
        target_aligner=aligner,
    )
    report_path = paths.run_output_dir / f"report_{EVAL_SPLIT}.json"
    report = build_evaluation_report(
        records=records,
        input_records_path=str(predictions_path),
        output_report_path=str(report_path),
        split=split,
    )
    write_evaluation_report(report)
    print(
        json.dumps(
            {
                "output_report_path": str(report_path),
                "action_accuracy": report.summary.action_accuracy,
                "mean_temporal_consistency": report.summary.mean_temporal_consistency,
                "total_clip_count": report.summary.total_clip_count,
                "total_action_count": report.summary.total_action_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_pipeline()
