"""Summary artifact schema helpers for Stage 3 fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping

SUMMARY_SCHEMA = "summary_v1"


def _expect(
    mapping: Mapping[str, Any],
    key: str,
    expected_type: type[Any] | tuple[type[Any], ...],
) -> Any:
    """Fetch and type-check a required field."""
    if key not in mapping:
        raise ValueError(f"missing summary key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected = " or ".join(t.__name__ for t in expected_type)
        else:
            expected = expected_type.__name__
        raise ValueError(f"summary field {key} must be {expected}.")
    return value


def build_summary_payload(
    *,
    mode: str,
    runner_backend: str,
    train_backend_metadata: Mapping[str, Any],
    checkpoint_version: str,
    manifest_path: str,
    output_dir: str,
    split: str | None,
    train_steps: int,
    mock_learning_rate: float,
    processed_samples: int,
    total_action_labels: int,
    unknown_action_labels: int,
    unknown_ratio: float,
    checkpoint_path: str,
    metrics_path: str,
    training_metadata_path: str,
    seed: int,
) -> dict[str, Any]:
    """Build a stable summary payload for latest_metrics.json."""
    return {
        "schema": SUMMARY_SCHEMA,
        "mode": mode,
        "runner_backend": runner_backend,
        "train_backend_metadata": dict(train_backend_metadata),
        "checkpoint_version": checkpoint_version,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "split": split,
        "train_steps": train_steps,
        "mock_learning_rate": mock_learning_rate,
        "processed_samples": processed_samples,
        "total_action_labels": total_action_labels,
        "unknown_action_labels": unknown_action_labels,
        "unknown_ratio": unknown_ratio,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "training_metadata_path": training_metadata_path,
        "seed": seed,
    }


def validate_summary_payload(payload: Mapping[str, Any]) -> None:
    """Validate the latest-metrics summary payload contract."""
    schema = _expect(payload, "schema", str)
    mode = _expect(payload, "mode", str)
    runner_backend = _expect(payload, "runner_backend", str)
    backend_metadata = _expect(payload, "train_backend_metadata", dict)
    _expect(payload, "checkpoint_version", str)
    _expect(payload, "manifest_path", str)
    _expect(payload, "output_dir", str)
    split = payload.get("split")
    if split is not None and not isinstance(split, str):
        raise ValueError("summary field split must be str or null.")
    train_steps = _expect(payload, "train_steps", int)
    mock_learning_rate = float(_expect(payload, "mock_learning_rate", (int, float)))
    processed_samples = _expect(payload, "processed_samples", int)
    total_action_labels = _expect(payload, "total_action_labels", int)
    unknown_action_labels = _expect(payload, "unknown_action_labels", int)
    unknown_ratio = float(_expect(payload, "unknown_ratio", (int, float)))
    _expect(payload, "checkpoint_path", str)
    _expect(payload, "metrics_path", str)
    _expect(payload, "training_metadata_path", str)
    seed = _expect(payload, "seed", int)

    if schema != SUMMARY_SCHEMA:
        raise ValueError(f"unsupported summary schema: {schema}")
    if mode not in {"dry_run", "train_stub", "train_noop", "train_mock"}:
        raise ValueError(f"unsupported summary mode: {mode}")
    if runner_backend not in {"train_stub", "train_noop", "train_mock"}:
        raise ValueError(f"unsupported summary runner_backend: {runner_backend}")
    if mode != "dry_run" and mode != runner_backend:
        raise ValueError("summary mode must match runner_backend when not dry_run.")
    if backend_metadata.get("runner_backend") != runner_backend:
        raise ValueError("summary train_backend_metadata.runner_backend mismatch.")
    if runner_backend == "train_mock" and "mock_learning_rate" not in backend_metadata:
        raise ValueError("summary train_backend_metadata missing mock_learning_rate for train_mock.")
    if runner_backend != "train_mock" and "mock_learning_rate" in backend_metadata:
        raise ValueError("summary train_backend_metadata.mock_learning_rate only allowed for train_mock.")
    if train_steps <= 0:
        raise ValueError("summary train_steps must be > 0.")
    if mock_learning_rate <= 0.0:
        raise ValueError("summary mock_learning_rate must be > 0.0.")
    if processed_samples < 0:
        raise ValueError("summary processed_samples must be >= 0.")
    if total_action_labels < 0:
        raise ValueError("summary total_action_labels must be >= 0.")
    if unknown_action_labels < 0:
        raise ValueError("summary unknown_action_labels must be >= 0.")
    if unknown_action_labels > total_action_labels:
        raise ValueError("summary unknown_action_labels cannot exceed total_action_labels.")
    if not 0.0 <= unknown_ratio <= 1.0:
        raise ValueError("summary unknown_ratio must be in range [0.0, 1.0].")
    if seed < 0:
        raise ValueError("summary seed must be >= 0.")
