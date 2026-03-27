"""Training metadata schema helpers for Stage 3 fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.train.checkpoint import validate_checkpoint_metadata
from src.train.runner import TrainingStepResult

TRAINING_METADATA_SCHEMA = "training_metadata_v1"


def _expect(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
) -> Any:
    """Fetch and type-check a required field."""
    if key not in mapping:
        raise ValueError(f"missing training metadata key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected = " or ".join(t.__name__ for t in expected_type)
        else:
            expected = expected_type.__name__
        raise ValueError(f"training metadata field {key} must be {expected}.")
    return value


def build_training_metadata(
    *,
    mode: str,
    seed: int,
    processed_samples: int,
    total_action_labels: int,
    unknown_action_labels: int,
    unknown_ratio: float,
    train_steps: int,
    step_metrics: Sequence[TrainingStepResult],
    runner_backend: str,
    train_backend_metadata: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    note: str,
) -> dict[str, Any]:
    """Build a stable training metadata payload for artifact output."""
    return {
        "schema": TRAINING_METADATA_SCHEMA,
        "mode": mode,
        "seed": seed,
        "processed_samples": processed_samples,
        "total_action_labels": total_action_labels,
        "unknown_action_labels": unknown_action_labels,
        "unknown_ratio": unknown_ratio,
        "train_steps": train_steps,
        "step_metrics": [result.to_dict() for result in step_metrics],
        "runner_backend": runner_backend,
        "train_backend_metadata": dict(train_backend_metadata),
        "checkpoint_metadata": dict(checkpoint_metadata),
        "note": note,
    }


def validate_training_metadata(payload: Mapping[str, Any]) -> None:
    """Validate the training metadata payload contract."""
    schema = _expect(payload, "schema", str)
    mode = _expect(payload, "mode", str)
    seed = _expect(payload, "seed", int)
    processed_samples = _expect(payload, "processed_samples", int)
    total_action_labels = _expect(payload, "total_action_labels", int)
    unknown_action_labels = _expect(payload, "unknown_action_labels", int)
    unknown_ratio = float(_expect(payload, "unknown_ratio", (int, float)))
    train_steps = _expect(payload, "train_steps", int)
    step_metrics = _expect(payload, "step_metrics", list)
    runner_backend = _expect(payload, "runner_backend", str)
    train_backend_metadata = _expect(payload, "train_backend_metadata", dict)
    checkpoint_metadata = _expect(payload, "checkpoint_metadata", dict)
    _expect(payload, "note", str)

    if schema != TRAINING_METADATA_SCHEMA:
        raise ValueError(f"unsupported training metadata schema: {schema}")
    if mode not in {"train_stub", "train_noop", "train_mock"}:
        raise ValueError(f"unsupported training metadata mode: {mode}")
    if seed < 0:
        raise ValueError("training metadata seed must be >= 0.")
    if processed_samples < 0:
        raise ValueError("training metadata processed_samples must be >= 0.")
    if total_action_labels < 0:
        raise ValueError("training metadata total_action_labels must be >= 0.")
    if unknown_action_labels < 0:
        raise ValueError("training metadata unknown_action_labels must be >= 0.")
    if unknown_action_labels > total_action_labels:
        raise ValueError("training metadata unknown_action_labels cannot exceed total_action_labels.")
    if not 0.0 <= unknown_ratio <= 1.0:
        raise ValueError("training metadata unknown_ratio must be in range [0.0, 1.0].")
    if train_steps <= 0:
        raise ValueError("training metadata train_steps must be > 0.")
    if len(step_metrics) != train_steps:
        raise ValueError("training metadata step_metrics length must equal train_steps.")
    if runner_backend != mode:
        raise ValueError("training metadata runner_backend must match mode.")
    if train_backend_metadata.get("runner_backend") != runner_backend:
        raise ValueError("training metadata train_backend_metadata.runner_backend mismatch.")

    validate_checkpoint_metadata(checkpoint_metadata)

    for index, item in enumerate(step_metrics):
        if not isinstance(item, dict):
            raise ValueError("training metadata step_metrics entries must be objects.")
        step_result = TrainingStepResult(
            step=int(_expect(item, "step", int)),
            loss=float(_expect(item, "loss", (int, float))),
            known_ratio=float(_expect(item, "known_ratio", (int, float))),
            samples_seen=int(_expect(item, "samples_seen", int)),
        )
        expected_step = index + 1
        if step_result.step != expected_step:
            raise ValueError("training metadata step_metrics must be contiguous and 1-indexed.")
