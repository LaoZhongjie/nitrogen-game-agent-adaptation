"""Training runner contracts for Stage 3 fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

RunnerBackend = Literal["train_stub", "train_noop", "train_mock"]


@dataclass(slots=True, frozen=True)
class TrainingStepResult:
    """Deterministic per-step metrics emitted by train backends."""

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


@dataclass(slots=True, frozen=True)
class TrainingRunContext:
    """Shared input context for all train backend runners."""

    train_steps: int
    known_ratio: float
    unknown_ratio: float
    processed_samples: int
    mock_learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.train_steps <= 0:
            raise ValueError("train_steps must be > 0.")
        if not 0.0 <= self.known_ratio <= 1.0:
            raise ValueError("known_ratio must be in range [0.0, 1.0].")
        if not 0.0 <= self.unknown_ratio <= 1.0:
            raise ValueError("unknown_ratio must be in range [0.0, 1.0].")
        if abs((self.known_ratio + self.unknown_ratio) - 1.0) > 1e-6:
            raise ValueError("known_ratio + unknown_ratio must sum to 1.0.")
        if self.processed_samples < 0:
            raise ValueError("processed_samples must be >= 0.")
        if self.mock_learning_rate <= 0.0:
            raise ValueError("mock_learning_rate must be > 0.0.")


class TrainingRunner(Protocol):
    """Backend contract for train-step metric generation."""

    def run(self, context: TrainingRunContext) -> list[TrainingStepResult]:
        """Run backend-specific deterministic step generation."""


def run_train_stub_steps(
    *,
    train_steps: int,
    known_ratio: float,
    unknown_ratio: float,
    processed_samples: int,
) -> list[TrainingStepResult]:
    """Emit deterministic train-stub step metrics."""
    context = TrainingRunContext(
        train_steps=train_steps,
        known_ratio=known_ratio,
        unknown_ratio=unknown_ratio,
        processed_samples=processed_samples,
    )
    return TrainStubRunner().run(context)


def run_train_noop_steps(*, train_steps: int) -> list[TrainingStepResult]:
    """Emit no-op training steps with deterministic zeroed metrics."""
    context = TrainingRunContext(
        train_steps=train_steps,
        known_ratio=0.0,
        unknown_ratio=1.0,
        processed_samples=0,
    )
    return TrainNoopRunner().run(context)


def run_noop_runner_steps(
    *,
    train_steps: int,
    known_ratio: float,
    processed_samples: int,
) -> list[TrainingStepResult]:
    """Emit deterministic no-op runner metrics preserving known ratio."""
    context = TrainingRunContext(
        train_steps=train_steps,
        known_ratio=known_ratio,
        unknown_ratio=1.0 - known_ratio,
        processed_samples=processed_samples,
    )
    return TrainNoopRunner().run(context)


def run_mock_runner_steps(
    *,
    train_steps: int,
    known_ratio: float,
    unknown_ratio: float,
    processed_samples: int,
    mock_learning_rate: float,
) -> list[TrainingStepResult]:
    """Emit deterministic non-trivial mock runner metrics."""
    context = TrainingRunContext(
        train_steps=train_steps,
        known_ratio=known_ratio,
        unknown_ratio=unknown_ratio,
        processed_samples=processed_samples,
        mock_learning_rate=mock_learning_rate,
    )
    return TrainMockRunner().run(context)


@dataclass(slots=True, frozen=True)
class TrainStubRunner:
    """Deterministic baseline runner used for train-stub mode."""

    def run(self, context: TrainingRunContext) -> list[TrainingStepResult]:
        base_loss = 1.0 + context.unknown_ratio
        return [
            TrainingStepResult(
                step=step_idx + 1,
                loss=base_loss / float(step_idx + 1),
                known_ratio=context.known_ratio,
                samples_seen=context.processed_samples,
            )
            for step_idx in range(context.train_steps)
        ]


@dataclass(slots=True, frozen=True)
class TrainNoopRunner:
    """No-op runner that emits zero-loss deterministic steps."""

    def run(self, context: TrainingRunContext) -> list[TrainingStepResult]:
        return [
            TrainingStepResult(
                step=step_idx + 1,
                loss=0.0,
                known_ratio=context.known_ratio,
                samples_seen=context.processed_samples,
            )
            for step_idx in range(context.train_steps)
        ]


@dataclass(slots=True, frozen=True)
class TrainMockRunner:
    """Mock runner with deterministic, non-trivial learning curve."""

    def run(self, context: TrainingRunContext) -> list[TrainingStepResult]:
        base_loss = 1.0 + context.unknown_ratio
        return [
            TrainingStepResult(
                step=step_idx + 1,
                loss=base_loss / float(1.0 + context.mock_learning_rate * float(step_idx + 1)),
                known_ratio=context.known_ratio,
                samples_seen=context.processed_samples,
            )
            for step_idx in range(context.train_steps)
        ]


def resolve_runner_backend(backend: str) -> RunnerBackend:
    """Validate and normalize runner backend name."""
    if backend == "train_stub":
        return "train_stub"
    if backend == "train_noop":
        return "train_noop"
    if backend == "train_mock":
        return "train_mock"
    raise ValueError(f"unsupported runner backend: {backend}")


def resolve_training_runner(backend: str) -> TrainingRunner:
    """Resolve backend name to a concrete training runner."""
    resolved = resolve_runner_backend(backend)
    if resolved == "train_stub":
        return TrainStubRunner()
    if resolved == "train_noop":
        return TrainNoopRunner()
    return TrainMockRunner()


def run_train_backend_steps(
    *,
    backend: str,
    train_steps: int,
    known_ratio: float,
    unknown_ratio: float,
    processed_samples: int,
    mock_learning_rate: float = 0.05,
) -> list[TrainingStepResult]:
    """Dispatch to the configured training runner backend."""
    context = TrainingRunContext(
        train_steps=train_steps,
        known_ratio=known_ratio,
        unknown_ratio=unknown_ratio,
        processed_samples=processed_samples,
        mock_learning_rate=mock_learning_rate,
    )
    runner = resolve_training_runner(backend)
    return runner.run(context)

