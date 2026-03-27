"""Offline evaluation metrics for post-training adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.data.schema import SplitName


@dataclass(slots=True, frozen=True)
class EvaluationRecord:
    """One evaluated clip with target and predicted actions."""

    episode_id: str
    clip_id: str
    split: SplitName
    target_action_ids: tuple[str, ...]
    predicted_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty.")
        if not self.clip_id.strip():
            raise ValueError("clip_id must be non-empty.")
        if len(self.target_action_ids) == 0:
            raise ValueError("target_action_ids must be non-empty.")
        if len(self.target_action_ids) != len(self.predicted_action_ids):
            raise ValueError("target_action_ids and predicted_action_ids must have equal length.")


@dataclass(slots=True, frozen=True)
class SplitEvaluationMetrics:
    """Aggregated metrics for one data split."""

    split: SplitName
    clip_count: int
    action_count: int
    correct_action_count: int
    action_accuracy: float
    mean_temporal_consistency: float

    def __post_init__(self) -> None:
        if self.clip_count < 0:
            raise ValueError("clip_count must be >= 0.")
        if self.action_count < 0:
            raise ValueError("action_count must be >= 0.")
        if self.correct_action_count < 0:
            raise ValueError("correct_action_count must be >= 0.")
        if self.correct_action_count > self.action_count:
            raise ValueError("correct_action_count must be <= action_count.")
        if not 0.0 <= self.action_accuracy <= 1.0:
            raise ValueError("action_accuracy must be in range [0.0, 1.0].")
        if not 0.0 <= self.mean_temporal_consistency <= 1.0:
            raise ValueError("mean_temporal_consistency must be in range [0.0, 1.0].")


@dataclass(slots=True, frozen=True)
class EvaluationSummary:
    """Overall and split-wise evaluation outputs."""

    total_clip_count: int
    total_action_count: int
    total_correct_action_count: int
    action_accuracy: float
    mean_temporal_consistency: float
    split_metrics: Mapping[str, SplitEvaluationMetrics]

    def __post_init__(self) -> None:
        if self.total_clip_count < 0:
            raise ValueError("total_clip_count must be >= 0.")
        if self.total_action_count < 0:
            raise ValueError("total_action_count must be >= 0.")
        if self.total_correct_action_count < 0:
            raise ValueError("total_correct_action_count must be >= 0.")
        if self.total_correct_action_count > self.total_action_count:
            raise ValueError("total_correct_action_count must be <= total_action_count.")
        if not 0.0 <= self.action_accuracy <= 1.0:
            raise ValueError("action_accuracy must be in range [0.0, 1.0].")
        if not 0.0 <= self.mean_temporal_consistency <= 1.0:
            raise ValueError("mean_temporal_consistency must be in range [0.0, 1.0].")


def compute_action_accuracy(target_action_ids: Sequence[str], predicted_action_ids: Sequence[str]) -> float:
    """Compute exact-match action accuracy for one aligned sequence."""
    if len(target_action_ids) != len(predicted_action_ids):
        raise ValueError("target_action_ids and predicted_action_ids must have equal length.")
    if len(target_action_ids) == 0:
        return 0.0
    correct = sum(1 for target, predicted in zip(target_action_ids, predicted_action_ids, strict=True) if target == predicted)
    return correct / float(len(target_action_ids))


def compute_temporal_consistency(action_ids: Sequence[str]) -> float:
    """Compute adjacent-step action consistency in range [0.0, 1.0]."""
    if len(action_ids) <= 1:
        return 1.0
    stable_pairs = sum(1 for idx in range(len(action_ids) - 1) if action_ids[idx] == action_ids[idx + 1])
    return stable_pairs / float(len(action_ids) - 1)


def evaluate_records(records: Sequence[EvaluationRecord]) -> EvaluationSummary:
    """Aggregate overall and split-wise offline metrics from clip records."""
    if len(records) == 0:
        raise ValueError("records must be non-empty.")

    split_buckets: dict[SplitName, list[EvaluationRecord]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }
    for record in records:
        split_buckets[record.split].append(record)

    split_metrics: dict[str, SplitEvaluationMetrics] = {}
    all_temporal_scores: list[float] = []
    total_action_count = 0
    total_correct_action_count = 0

    for split_name, split_records in split_buckets.items():
        if len(split_records) == 0:
            continue

        split_action_count = sum(len(record.target_action_ids) for record in split_records)
        split_correct_count = sum(
            sum(
                1
                for target, predicted in zip(record.target_action_ids, record.predicted_action_ids, strict=True)
                if target == predicted
            )
            for record in split_records
        )
        split_temporal_scores = [compute_temporal_consistency(record.predicted_action_ids) for record in split_records]
        split_mean_temporal = sum(split_temporal_scores) / float(len(split_temporal_scores))
        split_accuracy = split_correct_count / float(split_action_count) if split_action_count > 0 else 0.0

        split_metrics[split_name.value] = SplitEvaluationMetrics(
            split=split_name,
            clip_count=len(split_records),
            action_count=split_action_count,
            correct_action_count=split_correct_count,
            action_accuracy=split_accuracy,
            mean_temporal_consistency=split_mean_temporal,
        )

        all_temporal_scores.extend(split_temporal_scores)
        total_action_count += split_action_count
        total_correct_action_count += split_correct_count

    mean_temporal_consistency = sum(all_temporal_scores) / float(len(all_temporal_scores))
    action_accuracy = total_correct_action_count / float(total_action_count) if total_action_count > 0 else 0.0

    return EvaluationSummary(
        total_clip_count=len(records),
        total_action_count=total_action_count,
        total_correct_action_count=total_correct_action_count,
        action_accuracy=action_accuracy,
        mean_temporal_consistency=mean_temporal_consistency,
        split_metrics=split_metrics,
    )
