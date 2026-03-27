from __future__ import annotations

import pytest

from src.train.checkpoint import build_checkpoint_metadata
from src.train.metadata import (
    TRAINING_METADATA_SCHEMA,
    build_training_metadata,
    validate_training_metadata,
)
from src.train.runner import TrainingStepResult


def _build_step_metrics() -> list[TrainingStepResult]:
    return [
        TrainingStepResult(step=1, loss=1.0, known_ratio=0.75, samples_seen=3),
        TrainingStepResult(step=2, loss=0.7, known_ratio=0.75, samples_seen=3),
    ]


def test_build_training_metadata_generates_schema_and_steps() -> None:
    checkpoint_metadata = build_checkpoint_metadata(
        checkpoint_version="v1",
        backend="train_stub",
        step_count=2,
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
    )
    payload = build_training_metadata(
        mode="train_stub",
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
        unknown_ratio=0.25,
        train_steps=2,
        step_metrics=_build_step_metrics(),
        runner_backend="train_stub",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_metadata=checkpoint_metadata,
        note="ok",
    )

    assert payload["schema"] == TRAINING_METADATA_SCHEMA
    assert len(payload["step_metrics"]) == 2
    assert payload["step_metrics"][0]["step"] == 1
    assert payload["step_metrics"][1]["step"] == 2


def test_validate_training_metadata_rejects_step_length_mismatch() -> None:
    checkpoint_metadata = build_checkpoint_metadata(
        checkpoint_version="v1",
        backend="train_stub",
        step_count=2,
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
    )
    payload = build_training_metadata(
        mode="train_stub",
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
        unknown_ratio=0.25,
        train_steps=2,
        step_metrics=_build_step_metrics(),
        runner_backend="train_stub",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_metadata=checkpoint_metadata,
        note="ok",
    )
    payload["step_metrics"] = payload["step_metrics"][:1]

    with pytest.raises(ValueError, match="step_metrics length must equal train_steps"):
        validate_training_metadata(payload)


def test_validate_training_metadata_rejects_runner_mismatch() -> None:
    checkpoint_metadata = build_checkpoint_metadata(
        checkpoint_version="v1",
        backend="train_stub",
        step_count=2,
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
    )
    payload = build_training_metadata(
        mode="train_stub",
        seed=4,
        processed_samples=3,
        total_action_labels=8,
        unknown_action_labels=2,
        unknown_ratio=0.25,
        train_steps=2,
        step_metrics=_build_step_metrics(),
        runner_backend="train_stub",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_metadata=checkpoint_metadata,
        note="ok",
    )
    payload["runner_backend"] = "train_noop"

    with pytest.raises(ValueError, match="runner_backend must match mode"):
        validate_training_metadata(payload)
