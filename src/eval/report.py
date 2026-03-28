"""Evaluation report artifact contracts and JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import SplitName
from src.eval.metrics import EvaluationRecord, EvaluationSummary, SplitEvaluationMetrics, evaluate_records


@dataclass(slots=True, frozen=True)
class EvaluationReport:
    """Serializable offline evaluation report artifact."""

    schema_version: str
    evaluated_split: str | None
    input_records_path: str
    output_report_path: str
    summary: EvaluationSummary

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty.")
        if not self.input_records_path.strip():
            raise ValueError("input_records_path must be non-empty.")
        if not self.output_report_path.strip():
            raise ValueError("output_report_path must be non-empty.")
        if self.evaluated_split is not None and self.evaluated_split not in {
            SplitName.TRAIN.value,
            SplitName.VAL.value,
            SplitName.TEST.value,
        }:
            raise ValueError("evaluated_split must be one of: train, val, test, or None.")


def _split_metrics_to_dict(metrics: SplitEvaluationMetrics) -> dict[str, Any]:
    return {
        "split": metrics.split.value,
        "clip_count": metrics.clip_count,
        "action_count": metrics.action_count,
        "correct_action_count": metrics.correct_action_count,
        "action_accuracy": metrics.action_accuracy,
        "mean_temporal_consistency": metrics.mean_temporal_consistency,
    }


def _summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    return {
        "total_clip_count": summary.total_clip_count,
        "total_action_count": summary.total_action_count,
        "total_correct_action_count": summary.total_correct_action_count,
        "action_accuracy": summary.action_accuracy,
        "mean_temporal_consistency": summary.mean_temporal_consistency,
        "split_metrics": {
            split_name: _split_metrics_to_dict(metrics)
            for split_name, metrics in summary.split_metrics.items()
        },
    }


def load_evaluation_records(path: str | Path, split: SplitName | None = None) -> tuple[EvaluationRecord, ...]:
    """Load clip-level evaluation records from JSON."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"evaluation records not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, list):
        raise ValueError("evaluation records JSON root must be an array.")

    records: list[EvaluationRecord] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each evaluation record must be an object.")
        record_split = SplitName(str(entry["split"]))
        record = EvaluationRecord(
            episode_id=str(entry["episode_id"]),
            clip_id=str(entry["clip_id"]),
            split=record_split,
            target_action_ids=tuple(str(v) for v in entry["target_action_ids"]),
            predicted_action_ids=tuple(str(v) for v in entry["predicted_action_ids"]),
        )
        if split is None or record.split is split:
            records.append(record)
    return tuple(records)


def build_evaluation_report(
    *,
    records: Sequence[EvaluationRecord],
    input_records_path: str,
    output_report_path: str,
    split: SplitName | None = None,
) -> EvaluationReport:
    """Build a typed offline evaluation report from clip-level records."""
    summary = evaluate_records(records)
    return EvaluationReport(
        schema_version="v1_offline_evaluation_report",
        evaluated_split=None if split is None else split.value,
        input_records_path=input_records_path,
        output_report_path=output_report_path,
        summary=summary,
    )


def write_evaluation_report(report: EvaluationReport) -> None:
    """Persist evaluation report as JSON artifact."""
    output_path = Path(report.output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": report.schema_version,
        "evaluated_split": report.evaluated_split,
        "input_records_path": report.input_records_path,
        "output_report_path": report.output_report_path,
        "summary": _summary_to_dict(report.summary),
    }
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def evaluate_records_file(
    *,
    input_records_path: str,
    output_report_path: str,
    split: SplitName | None = None,
) -> EvaluationReport:
    """Load records from disk, compute metrics, and write report."""
    records = load_evaluation_records(path=input_records_path, split=split)
    report = build_evaluation_report(
        records=records,
        input_records_path=input_records_path,
        output_report_path=output_report_path,
        split=split,
    )
    write_evaluation_report(report)
    return report
