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


def _manifest_samples_by_clip(
    *,
    manifest_path: str | Path,
    split: SplitName | None,
) -> dict[str, ManifestSample]:
    dataset = ManifestDataset(manifest_path, split=split)
    by_clip: dict[str, ManifestSample] = {}
    for sample in dataset:
        by_clip[sample.clip_id] = sample
    return by_clip


def build_evaluation_records_from_manifest_predictions(
    *,
    manifest_path: str | Path,
    predictions: Sequence[PredictionRecord],
    split: SplitName | None = None,
) -> tuple[EvaluationRecord, ...]:
    """Join manifest targets with predicted actions into evaluation records."""
    samples_by_clip = _manifest_samples_by_clip(manifest_path=manifest_path, split=split)
    if len(samples_by_clip) == 0:
        raise ValueError("no manifest samples available for evaluation.")

    records: list[EvaluationRecord] = []
    for prediction in predictions:
        sample = samples_by_clip.get(prediction.clip_id)
        if sample is None:
            raise ValueError(f"prediction clip_id not found in manifest split: {prediction.clip_id}")
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
        raise ValueError("predictions must be non-empty.")
    return tuple(records)

