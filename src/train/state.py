"""Serializable training-state contract for checkpoint payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

TRAINING_STATE_SCHEMA = "training_state_v1"


def _expect(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
) -> Any:
    """Read and type-check a required field."""
    if key not in mapping:
        raise ValueError(f"missing training state key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_name = " or ".join(t.__name__ for t in expected_type)
        else:
            type_name = expected_type.__name__
        raise ValueError(f"training state field {key} must be {type_name}.")
    return value


def _expect_floatish(mapping: Mapping[str, Any], key: str) -> float:
    """Read a required numeric field as float."""
    value = _expect(mapping, key, (int, float))
    return float(value)


@dataclass(slots=True, frozen=True)
class TrainingState:
    """Versioned, JSON-friendly snapshot of backend training state."""

    backend: str
    train_steps: int
    latest_step: int
    latest_loss: float
    known_ratio: float
    samples_seen: int
    mock_learning_rate: float

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend must be non-empty.")
        if self.train_steps <= 0:
            raise ValueError("train_steps must be > 0.")
        if self.latest_step <= 0:
            raise ValueError("latest_step must be > 0.")
        if self.latest_step > self.train_steps:
            raise ValueError("latest_step must be <= train_steps.")
        if self.latest_loss < 0.0:
            raise ValueError("latest_loss must be >= 0.0.")
        if not 0.0 <= self.known_ratio <= 1.0:
            raise ValueError("known_ratio must be in range [0.0, 1.0].")
        if self.samples_seen < 0:
            raise ValueError("samples_seen must be >= 0.")
        if self.mock_learning_rate <= 0.0:
            raise ValueError("mock_learning_rate must be > 0.0.")

    def to_dict(self) -> dict[str, str | int | float]:
        """Serialize to a stable, versioned dictionary."""
        return {
            "schema": TRAINING_STATE_SCHEMA,
            "backend": self.backend,
            "train_steps": self.train_steps,
            "latest_step": self.latest_step,
            "latest_loss": self.latest_loss,
            "known_ratio": self.known_ratio,
            "samples_seen": self.samples_seen,
            "mock_learning_rate": self.mock_learning_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingState:
        """Parse and validate a versioned training-state dictionary."""
        schema = _expect(payload, "schema", str)
        if schema != TRAINING_STATE_SCHEMA:
            raise ValueError(f"unsupported training state schema: {schema}")
        return cls(
            backend=_expect(payload, "backend", str),
            train_steps=_expect(payload, "train_steps", int),
            latest_step=_expect(payload, "latest_step", int),
            latest_loss=_expect_floatish(payload, "latest_loss"),
            known_ratio=_expect_floatish(payload, "known_ratio"),
            samples_seen=_expect(payload, "samples_seen", int),
            mock_learning_rate=_expect_floatish(payload, "mock_learning_rate"),
        )


def validate_training_state(payload: Mapping[str, Any]) -> None:
    """Validate a serialized training-state payload."""
    _ = TrainingState.from_dict(payload)
