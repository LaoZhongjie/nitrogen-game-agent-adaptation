"""Video generation evaluation report serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.data.schema import SplitName
from src.eval.metrics import VideoEvalRecord, VideoEvalSummary, evaluate_records


@dataclass(frozen=True)
class VideoEvalReport:
    """Serializable evaluation report for video generation."""

    schema_version: str
    evaluated_split: Optional[str]
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
        "mean_mae": record.mean_mae,
        "reference_temporal_consistency": record.reference_temporal_consistency,
        "temporal_error_vs_reference": record.temporal_error_vs_reference,
    }


def _summary_to_dict(summary: VideoEvalSummary) -> dict[str, Any]:
    return {
        "total_videos": summary.total_videos,
        "total_frames_with_reference": summary.total_frames_with_reference,
        "mean_fid": summary.mean_fid,
        "inception_feature_mean_l2": summary.inception_feature_mean_l2,
        "mean_lpips": summary.mean_lpips,
        "mean_temporal_consistency": summary.mean_temporal_consistency,
        "mean_psnr": summary.mean_psnr,
        "mean_ssim": summary.mean_ssim,
        "mean_mae": summary.mean_mae,
        "mean_reference_temporal_consistency": summary.mean_reference_temporal_consistency,
        "mean_temporal_error_vs_reference": summary.mean_temporal_error_vs_reference,
        "fvd": summary.fvd,
    }


def build_evaluation_report(
    *,
    records: Sequence[VideoEvalRecord],
    generation_manifest_path: str,
    output_report_path: str,
    split: Optional[SplitName] = None,
    pooled_gen_features: Optional[np.ndarray] = None,
    pooled_ref_features: Optional[np.ndarray] = None,
) -> VideoEvalReport:
    """Build a typed video generation evaluation report."""
    summary = evaluate_records(
        records,
        pooled_gen_features=pooled_gen_features,
        pooled_ref_features=pooled_ref_features,
    )
    return VideoEvalReport(
        schema_version="v2_video_generation_evaluation",
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
