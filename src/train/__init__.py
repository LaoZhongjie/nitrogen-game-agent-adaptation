"""Training utilities and contracts."""

from src.train.runner import (
    RunnerBackend,
    TrainingRunContext,
    TrainingRunner,
    TrainingStepResult,
    resolve_runner_backend,
    resolve_training_runner,
    run_train_backend_steps,
    run_mock_runner_steps,
    run_noop_runner_steps,
    run_train_noop_steps,
    run_train_stub_steps,
)
from src.train.checkpoint import build_checkpoint_metadata, validate_checkpoint_metadata
from src.train.metadata import (
    TRAINING_METADATA_SCHEMA,
    build_training_metadata,
    validate_training_metadata,
)
from src.train.summary import SUMMARY_SCHEMA, build_summary_payload, validate_summary_payload
from src.train.state import TRAINING_STATE_SCHEMA, TrainingState, validate_training_state

__all__ = [
    "RunnerBackend",
    "TrainingRunContext",
    "TrainingRunner",
    "TrainingStepResult",
    "resolve_runner_backend",
    "resolve_training_runner",
    "run_train_backend_steps",
    "run_mock_runner_steps",
    "run_noop_runner_steps",
    "run_train_noop_steps",
    "run_train_stub_steps",
    "build_checkpoint_metadata",
    "validate_checkpoint_metadata",
    "TRAINING_METADATA_SCHEMA",
    "build_training_metadata",
    "validate_training_metadata",
    "SUMMARY_SCHEMA",
    "build_summary_payload",
    "validate_summary_payload",
    "TRAINING_STATE_SCHEMA",
    "TrainingState",
    "validate_training_state",
]

