"""Checkpoint metadata schema helpers for Stage 3 fine-tuning."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

CHECKPOINT_VERSION = "v1"


def build_checkpoint_metadata(
    *,
    checkpoint_version: str,
    backend: str,
    step_count: int,
    seed: int,
    processed_samples: int,
    total_action_labels: int,
    unknown_action_labels: int,
) -> dict[str, Any]:
    """Build deterministic checkpoint metadata payload."""
    digest_input = (
        f"{checkpoint_version}|{backend}|{step_count}|{seed}|"
        f"{processed_samples}|{total_action_labels}|{unknown_action_labels}"
    )
    state_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return {
        "checkpoint_version": checkpoint_version,
        "backend": backend,
        "step_count": step_count,
        "state_digest": state_digest,
    }


def _expect(mapping: Mapping[str, Any], key: str, expected_type: type[Any]) -> Any:
    """Fetch and type-check a required key."""
    if key not in mapping:
        raise ValueError(f"missing checkpoint metadata key: {key}")
    value = mapping[key]
    if not isinstance(value, expected_type):
        raise ValueError(f"checkpoint metadata field {key} must be {expected_type.__name__}.")
    return value


def validate_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate checkpoint metadata contract for artifact integrity."""
    version = _expect(metadata, "checkpoint_version", str)
    backend = _expect(metadata, "backend", str)
    step_count = _expect(metadata, "step_count", int)
    state_digest = _expect(metadata, "state_digest", str)

    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint_version: {version}")
    if backend not in {"train_stub", "train_noop", "train_mock"}:
        raise ValueError(f"unsupported backend in checkpoint metadata: {backend}")
    if step_count <= 0:
        raise ValueError("checkpoint metadata step_count must be > 0.")
    if len(state_digest) != 64:
        raise ValueError("checkpoint metadata state_digest must be 64 hex chars.")
    if any(ch not in "0123456789abcdef" for ch in state_digest):
        raise ValueError("checkpoint metadata state_digest must be lowercase hex.")

