"""Training runner contracts for Stage 3 fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunnerBackend = Literal["train_stub", "train_noop"]


@dataclass(slots=True, frozen=True)
class TrainingStepResult:
    """Deterministic per-step metrics emitted by the train stub."""

    step: int
    loss: float
    known_ratio: float
    samples_seen: int

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be > 0.")
        if self.loss < 0.0:
            raise ValueError("loss must be >= 0.0.")
        if not 0.0 <= self.known_ratio <= 1.0:
            raise ValueError("known_ratio must be in range [0.0, 1.0].")
        if self.samples_seen < 0:
            raise ValueError("samples_seen must be >= 0.")

    def to_dict(self) -> dict[str, float | int]:
        """Serialize step result to a JSON-friendly dictionary."""
        return {
            "step": self.step,
            "loss": self.loss,
            "known_ratio": self.known_ratio,
            "samples_seen": self.samples_seen,
        }


def run_train_stub_steps(
    *,
    train_steps: int,
    known_ratio: float,
    unknown_ratio: float,
    processed_samples: int,
) -> list[TrainingStepResult]:
    """Emit deterministic train-stub step metrics."""
    if train_steps <= 0:
        raise ValueError("train_steps must be > 0.")
    if not 0.0 <= known_ratio <= 1.0:
        raise ValueError("known_ratio must be in range [0.0, 1.0].")
    if not 0.0 <= unknown_ratio <= 1.0:
        raise ValueError("unknown_ratio must be in range [0.0, 1.0].")
    if abs((known_ratio + unknown_ratio) - 1.0) > 1e-6:
        raise ValueError("known_ratio + unknown_ratio must sum to 1.0.")
    if processed_samples < 0:
        raise ValueError("processed_samples must be >= 0.")

    base_loss = 1.0 + unknown_ratio
    return [
        TrainingStepResult(
            step=step_idx + 1,
            loss=base_loss / float(step_idx + 1),
            known_ratio=known_ratio,
            samples_seen=processed_samples,
        )
        for step_idx in range(train_steps)
    ]


def run_train_noop_steps(*, train_steps: int) -> list[TrainingStepResult]:
    """Emit no-op training steps with deterministic zeroed metrics."""
    if train_steps <= 0:
        raise ValueError("train_steps must be > 0.")
    return [
        TrainingStepResult(
            step=step_idx + 1,
            loss=0.0,
            known_ratio=0.0,
            samples_seen=0,
        )
        for step_idx in range(train_steps)
    ]


def run_noop_runner_steps(
    *,
    train_steps: int,
    known_ratio: float,
    processed_samples: int,
) -> list[TrainingStepResult]:
    """Emit deterministic no-op runner metrics."""
    if train_steps <= 0:
        raise ValueError("train_steps must be > 0.")
    if not 0.0 <= known_ratio <= 1.0:
        raise ValueError("known_ratio must be in range [0.0, 1.0].")
    if processed_samples < 0:
        raise ValueError("processed_samples must be >= 0.")
    return [
        TrainingStepResult(
            step=step_idx + 1,
            loss=0.0,
            known_ratio=known_ratio,
            samples_seen=processed_samples,
        )
        for step_idx in range(train_steps)
    ]


def resolve_runner_backend(backend: str) -> RunnerBackend:
    """Validate and normalize runner backend name."""
    if backend == "train_stub":
        return "train_stub"
    if backend == "train_noop":
        return "train_noop"
    raise ValueError(f"unsupported runner backend: {backend}")


def run_train_backend_steps(
    *,
    backend: str,
    train_steps: int,
    known_ratio: float,
    unknown_ratio: float,
    processed_samples: int,
) -> list[TrainingStepResult]:
    """Dispatch to the configured training runner backend."""
    resolved = resolve_runner_backend(backend)
    if resolved == "train_stub":
        return run_train_stub_steps(
            train_steps=train_steps,
            known_ratio=known_ratio,
            unknown_ratio=unknown_ratio,
            processed_samples=processed_samples,
        )
    return run_noop_runner_steps(
        train_steps=train_steps,
        known_ratio=known_ratio,
        processed_samples=processed_samples,
    )

