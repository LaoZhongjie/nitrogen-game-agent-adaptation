"""Offline evaluation pipeline from manifest + prediction records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.data.loader import ManifestDataset, ManifestSample
from src.data.schema import SplitName
from src.eval.metrics import EvaluationRecord


@dataclass(slots=True, frozen=True)
class PredictionRecord:
    """Predicted actions for a single manifest clip."""

    clip_id: str
    predicted_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.clip_id.strip():
            raise ValueError("clip_id must be non-empty.")
        if len(self.predicted_action_ids) == 0:
            raise ValueError("predicted_action_ids must be non-empty.")


def load_prediction_records(path: str | Path) -> tuple[PredictionRecord, ...]:
    """Load prediction records keyed by clip ID."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"prediction records not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, list):
        raise ValueError("prediction records JSON root must be an array.")

    records: list[PredictionRecord] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each prediction record must be an object.")
        predicted_raw = entry["predicted_action_ids"]
        if not isinstance(predicted_raw, list):
            raise ValueError("predicted_action_ids must be an array.")
        records.append(
            PredictionRecord(
                clip_id=str(entry["clip_id"]),
                predicted_action_ids=tuple(str(v) for v in predicted_raw),
            )
        )
    return tuple(records)


def _manifest_samples(
    *,
    manifest_path: str | Path,
    split: SplitName | None,
) -> tuple[ManifestSample, ...]:
    dataset = ManifestDataset(manifest_path, split=split)
    return tuple(dataset)


def _prediction_map(predictions: Sequence[PredictionRecord]) -> dict[str, PredictionRecord]:
    by_clip: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.clip_id in by_clip:
            raise ValueError(f"duplicate prediction clip_id: {prediction.clip_id}")
        by_clip[prediction.clip_id] = prediction
    return by_clip


def build_evaluation_records_from_manifest_predictions(
    *,
    manifest_path: str | Path,
    predictions: Sequence[PredictionRecord],
    split: SplitName | None = None,
    require_all_clips: bool = True,
) -> tuple[EvaluationRecord, ...]:
    """Join manifest targets with predicted actions into evaluation records."""
    if len(predictions) == 0:
        raise ValueError("predictions must be non-empty.")

    manifest_samples = _manifest_samples(manifest_path=manifest_path, split=split)
    if len(manifest_samples) == 0:
        raise ValueError("no manifest samples available for evaluation.")
    samples_by_clip = {sample.clip_id: sample for sample in manifest_samples}
    prediction_by_clip = _prediction_map(predictions)

    unknown_clip_ids = sorted(set(prediction_by_clip.keys()).difference(samples_by_clip.keys()))
    if unknown_clip_ids:
        unknown_preview = ", ".join(unknown_clip_ids[:5])
        raise ValueError(f"prediction clip_id not found in manifest split: {unknown_preview}")

    if require_all_clips:
        missing_clip_ids = sorted(set(samples_by_clip.keys()).difference(prediction_by_clip.keys()))
        if missing_clip_ids:
            missing_preview = ", ".join(missing_clip_ids[:5])
            raise ValueError(f"missing predictions for manifest clips: {missing_preview}")

    records: list[EvaluationRecord] = []
    for sample in manifest_samples:
        prediction = prediction_by_clip.get(sample.clip_id)
        if prediction is None:
            continue
        if len(prediction.predicted_action_ids) != len(sample.action_labels):
            raise ValueError(
                f"predicted_action_ids length mismatch for clip {prediction.clip_id}: "
                f"expected {len(sample.action_labels)}, got {len(prediction.predicted_action_ids)}"
            )
        records.append(
            EvaluationRecord(
                episode_id=sample.episode_id,
                clip_id=sample.clip_id,
                split=sample.split,
                target_action_ids=sample.action_labels,
                predicted_action_ids=prediction.predicted_action_ids,
            )
        )
    if len(records) == 0:
        raise ValueError("predictions did not match any manifest clips.")
    return tuple(records)

