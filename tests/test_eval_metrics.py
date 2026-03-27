"""Unit tests for Stage 4 offline evaluation metrics."""

from __future__ import annotations

import pytest

from src.data.schema import SplitName
from src.eval.metrics import (
    EvaluationRecord,
    compute_action_accuracy,
    compute_temporal_consistency,
    evaluate_records,
)


def test_compute_action_accuracy_returns_expected_ratio() -> None:
    accuracy = compute_action_accuracy(
        target_action_ids=("left", "jump", "left", "fire"),
        predicted_action_ids=("left", "slide", "left", "fire"),
    )
    assert accuracy == 0.75


def test_compute_action_accuracy_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        compute_action_accuracy(target_action_ids=("left",), predicted_action_ids=("left", "jump"))


def test_compute_temporal_consistency_matches_adjacent_stability() -> None:
    consistency = compute_temporal_consistency(("left", "left", "jump", "jump", "jump", "left"))
    # stable pairs: 0-1, 2-3, 3-4 => 3 / 5
    assert consistency == 0.6


def test_compute_temporal_consistency_singleton_is_one() -> None:
    assert compute_temporal_consistency(("left",)) == 1.0


def test_evaluate_records_aggregates_overall_and_split_metrics() -> None:
    records = (
        EvaluationRecord(
            episode_id="ep_train",
            clip_id="clip_train",
            split=SplitName.TRAIN,
            target_action_ids=("left", "jump", "jump", "left"),
            predicted_action_ids=("left", "jump", "slide", "left"),
        ),
        EvaluationRecord(
            episode_id="ep_val",
            clip_id="clip_val",
            split=SplitName.VAL,
            target_action_ids=("a", "a", "b"),
            predicted_action_ids=("a", "b", "b"),
        ),
        EvaluationRecord(
            episode_id="ep_test",
            clip_id="clip_test",
            split=SplitName.TEST,
            target_action_ids=("x", "y"),
            predicted_action_ids=("x", "y"),
        ),
    )

    summary = evaluate_records(records)

    # Overall: correct = 3 + 2 + 2 = 7, total = 4 + 3 + 2 = 9
    assert summary.total_clip_count == 3
    assert summary.total_action_count == 9
    assert summary.total_correct_action_count == 7
    assert summary.action_accuracy == pytest.approx(7.0 / 9.0)

    assert set(summary.split_metrics.keys()) == {"train", "val", "test"}
    assert summary.split_metrics["train"].action_accuracy == 0.75
    assert summary.split_metrics["val"].action_accuracy == pytest.approx(2.0 / 3.0)
    assert summary.split_metrics["test"].action_accuracy == 1.0


def test_evaluate_records_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_records(())
