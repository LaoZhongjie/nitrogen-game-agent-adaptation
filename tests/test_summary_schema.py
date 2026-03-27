from __future__ import annotations

import pytest

from src.train.summary import SUMMARY_SCHEMA, build_summary_payload, validate_summary_payload


def test_build_summary_emits_versioned_schema_payload() -> None:
    payload = build_summary_payload(
        mode="dry_run",
        runner_backend="train_stub",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_version="v1",
        manifest_path="data/processed/manifest.json",
        output_dir="outputs/run_001",
        split="train",
        train_steps=3,
        mock_learning_rate=0.05,
        processed_samples=2,
        total_action_labels=8,
        unknown_action_labels=1,
        unknown_ratio=0.125,
        checkpoint_path="outputs/run_001/checkpoints/latest.ckpt",
        metrics_path="outputs/run_001/metrics/latest_metrics.json",
        training_metadata_path="outputs/run_001/metrics/training_metadata.json",
        seed=0,
    )

    assert payload["schema"] == SUMMARY_SCHEMA
    assert payload["mode"] == "dry_run"
    assert payload["runner_backend"] == "train_stub"


def test_validate_summary_rejects_backend_metadata_mismatch() -> None:
    payload = build_summary_payload(
        mode="train_noop",
        runner_backend="train_noop",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_version="v1",
        manifest_path="data/processed/manifest.json",
        output_dir="outputs/run_001",
        split=None,
        train_steps=5,
        mock_learning_rate=0.05,
        processed_samples=1,
        total_action_labels=2,
        unknown_action_labels=0,
        unknown_ratio=0.0,
        checkpoint_path="outputs/run_001/checkpoints/latest.ckpt",
        metrics_path="outputs/run_001/metrics/latest_metrics.json",
        training_metadata_path="outputs/run_001/metrics/training_metadata.json",
        seed=1,
    )
    with pytest.raises(ValueError, match="train_backend_metadata.runner_backend mismatch"):
        validate_summary_payload(payload)


def test_validate_summary_rejects_invalid_mode() -> None:
    payload = build_summary_payload(
        mode="dry_run",
        runner_backend="train_stub",
        train_backend_metadata={"runner_backend": "train_stub"},
        checkpoint_version="v1",
        manifest_path="data/processed/manifest.json",
        output_dir="outputs/run_001",
        split="train",
        train_steps=1,
        mock_learning_rate=0.05,
        processed_samples=1,
        total_action_labels=1,
        unknown_action_labels=0,
        unknown_ratio=0.0,
        checkpoint_path="outputs/run_001/checkpoints/latest.ckpt",
        metrics_path="outputs/run_001/metrics/latest_metrics.json",
        training_metadata_path="outputs/run_001/metrics/training_metadata.json",
        seed=0,
    )
    payload["mode"] = "invalid_mode"
    with pytest.raises(ValueError, match="unsupported summary mode"):
        validate_summary_payload(payload)
