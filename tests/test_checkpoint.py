from __future__ import annotations

import pytest

from src.train.checkpoint import (
    build_checkpoint_metadata,
    validate_checkpoint_metadata,
)


def test_build_checkpoint_metadata_generates_expected_shape() -> None:
    metadata = build_checkpoint_metadata(
        checkpoint_version="v1",
        backend="train_stub",
        step_count=3,
        seed=7,
        processed_samples=11,
        total_action_labels=22,
        unknown_action_labels=5,
    )

    assert metadata["checkpoint_version"] == "v1"
    assert metadata["backend"] == "train_stub"
    assert metadata["step_count"] == 3
    assert isinstance(metadata["state_digest"], str)
    assert len(metadata["state_digest"]) == 64


def test_validate_checkpoint_metadata_rejects_missing_required_key() -> None:
    with pytest.raises(ValueError, match="missing checkpoint metadata key"):
        validate_checkpoint_metadata(
            {
                "checkpoint_version": "v1",
                "backend": "train_stub",
                "state_digest": "a" * 64,
            }
        )


def test_validate_checkpoint_metadata_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="state_digest"):
        validate_checkpoint_metadata(
            {
                "checkpoint_version": "v1",
                "backend": "train_stub",
                "step_count": 3,
                "state_digest": "not-a-sha256",
            }
        )

