"""Offline evaluation CLI for Stage 4 report generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both:
# - python -m scripts.evaluate (recommended)
# - python scripts/evaluate.py (common)
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.schema import SplitName
from src.eval.pipeline import build_evaluation_records_from_manifest_predictions, load_prediction_records
from src.eval.report import build_evaluation_report, evaluate_records_file, write_evaluation_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation report generation."""
    parser = argparse.ArgumentParser(description="Generate offline evaluation report from prediction records.")
    parser.add_argument("--input", help="Path to evaluation records JSON array.")
    parser.add_argument(
        "--manifest",
        help="Path to dataset manifest JSON. Required when using --predictions mode.",
    )
    parser.add_argument(
        "--predictions",
        help="Path to prediction records JSON array. Requires --manifest.",
    )
    parser.add_argument("--output", required=True, help="Path to output evaluation report JSON.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Optional split filter.")
    parser.add_argument(
        "--allow-missing-clips",
        action="store_true",
        help="Allow manifest clips without predictions in --manifest/--predictions mode.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    split = None if args.split is None else SplitName(args.split)
    use_manifest_prediction_mode = args.manifest is not None or args.predictions is not None
    if use_manifest_prediction_mode:
        if args.manifest is None or args.predictions is None:
            raise ValueError("both --manifest and --predictions are required together.")
        predictions = load_prediction_records(args.predictions)
        records = build_evaluation_records_from_manifest_predictions(
            manifest_path=args.manifest,
            predictions=predictions,
            split=split,
            require_all_clips=not bool(args.allow_missing_clips),
        )
        report = build_evaluation_report(
            records=records,
            input_records_path=str(args.predictions),
            output_report_path=str(args.output),
            split=split,
        )
        write_evaluation_report(report)
    else:
        if args.input is None:
            raise ValueError("either --input or (--manifest and --predictions) must be provided.")
        report = evaluate_records_file(
            input_records_path=str(args.input),
            output_report_path=str(args.output),
            split=split,
        )
    payload = {
        "schema_version": report.schema_version,
        "evaluated_split": report.evaluated_split,
        "input_records_path": report.input_records_path,
        "output_report_path": report.output_report_path,
        "summary": {
            "total_clip_count": report.summary.total_clip_count,
            "total_action_count": report.summary.total_action_count,
            "total_correct_action_count": report.summary.total_correct_action_count,
            "action_accuracy": report.summary.action_accuracy,
            "mean_temporal_consistency": report.summary.mean_temporal_consistency,
            "split_metrics": {
                key: {
                    "split": metric.split.value,
                    "clip_count": metric.clip_count,
                    "action_count": metric.action_count,
                    "correct_action_count": metric.correct_action_count,
                    "action_accuracy": metric.action_accuracy,
                    "mean_temporal_consistency": metric.mean_temporal_consistency,
                }
                for key, metric in report.summary.split_metrics.items()
            },
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
