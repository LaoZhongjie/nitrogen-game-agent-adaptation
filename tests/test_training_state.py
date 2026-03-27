from __future__ import annotations

import pytest

from src.train.state import TRAINING_STATE_SCHEMA, TrainingState, validate_training_state


def test_training_state_roundtrip_serialization() -> None:
    state = TrainingState(
        backend="train_mock",
        train_steps=4,
        latest_step=4,
        latest_loss=0.75,
        known_ratio=0.8,
        samples_seen=12,
        mock_learning_rate=0.2,
    )

    payload = state.to_dict()
    restored = TrainingState.from_dict(payload)

    assert payload["schema"] == TRAINING_STATE_SCHEMA
    assert restored == state


def test_training_state_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported training state schema"):
        TrainingState.from_dict(
            {
                "schema": "training_state_v0",
                "backend": "train_mock",
                "train_steps": 3,
                "latest_step": 3,
                "latest_loss": 1.0,
                "known_ratio": 0.7,
                "samples_seen": 5,
                "mock_learning_rate": 0.1,
            }
        )


def test_validate_training_state_rejects_incomplete_payload() -> None:
    with pytest.raises(ValueError, match="missing training state key"):
        validate_training_state(
            {
                "schema": TRAINING_STATE_SCHEMA,
                "backend": "train_mock",
            }
        )
