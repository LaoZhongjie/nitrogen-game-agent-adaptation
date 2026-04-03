"""Video generation evaluation report serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.data.schema import SplitName
from src.eval.metrics import VideoEvalRecord, VideoEvalSummary, evaluate_records


@dataclass(slots=True, frozen=True)
class VideoEvalReport:
    """Serializable evaluation report for video generation."""

    schema_version: str
    evaluated_split: str | None
    generation_manifest_path: str
    output_report_path: str
    summary: VideoEvalSummary
    per_video: tuple[VideoEvalRecord, ...]


def _record_to_dict(record: VideoEvalRecord) -> dict[str, Any]:
    return {
        "chunk_id": record.chunk_id,
        "game": record.game,
        "num_frames_generated": record.num_frames_generated,
        "num_frames_reference": record.num_frames_reference,
        "fid_per_frame": record.fid_per_frame,
        "lpips_mean": record.lpips_mean,
        "temporal_consistency": record.temporal_consistency,
        "psnr_mean": record.psnr_mean,
        "ssim_mean": record.ssim_mean,
    }


def _summary_to_dict(summary: VideoEvalSummary) -> dict[str, Any]:
    return {
        "total_videos": summary.total_videos,
        "mean_fid": summary.mean_fid,
        "mean_lpips": summary.mean_lpips,
        "mean_temporal_consistency": summary.mean_temporal_consistency,
        "mean_psnr": summary.mean_psnr,
        "mean_ssim": summary.mean_ssim,
        "fvd": summary.fvd,
    }


def build_evaluation_report(
    *,
    records: Sequence[VideoEvalRecord],
    generation_manifest_path: str,
    output_report_path: str,
    split: SplitName | None = None,
) -> VideoEvalReport:
    """Build a typed video generation evaluation report."""
    summary = evaluate_records(records)
    return VideoEvalReport(
        schema_version="v1_video_generation_evaluation",
        evaluated_split=None if split is None else split.value,
        generation_manifest_path=generation_manifest_path,
        output_report_path=output_report_path,
        summary=summary,
        per_video=tuple(records),
    )


def write_evaluation_report(report: VideoEvalReport) -> None:
    """Persist evaluation report as JSON artifact."""
    output_path = Path(report.output_report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": report.schema_version,
        "evaluated_split": report.evaluated_split,
        "generation_manifest_path": report.generation_manifest_path,
        "output_report_path": report.output_report_path,
        "summary": _summary_to_dict(report.summary),
        "per_video": [_record_to_dict(r) for r in report.per_video],
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
