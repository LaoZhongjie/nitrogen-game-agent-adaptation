"""Short-horizon rollout validation harness for Stage 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(slots=True, frozen=True)
class RolloutStep:
    """One transition in a short-horizon rollout."""

    step_idx: int
    action_id: str
    reward: float
    success: bool
    done: bool
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.step_idx < 0:
            raise ValueError("step_idx must be >= 0.")
        if not self.action_id.strip():
            raise ValueError("action_id must be non-empty.")
        for key in self.diagnostics.keys():
            if not str(key).strip():
                raise ValueError("diagnostics keys must be non-empty.")


@dataclass(slots=True, frozen=True)
class RolloutTrace:
    """Trace data for one rollout episode."""

    rollout_id: str
    max_horizon: int
    steps: tuple[RolloutStep, ...]

    def __post_init__(self) -> None:
        if not self.rollout_id.strip():
            raise ValueError("rollout_id must be non-empty.")
        if self.max_horizon <= 0:
            raise ValueError("max_horizon must be > 0.")
        if len(self.steps) == 0:
            raise ValueError("steps must be non-empty.")
        if len(self.steps) > self.max_horizon:
            raise ValueError("steps length must be <= max_horizon.")


@dataclass(slots=True, frozen=True)
class RolloutSummary:
    """Aggregate short-horizon success and behavior diagnostics."""

    rollout_count: int
    step_count: int
    success_count: int
    failure_count: int
    success_rate: float
    mean_reward: float
    mean_episode_length: float
    action_switch_rate: float
    diagnostics_mean: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.rollout_count < 0:
            raise ValueError("rollout_count must be >= 0.")
        if self.step_count < 0:
            raise ValueError("step_count must be >= 0.")
        if self.success_count < 0:
            raise ValueError("success_count must be >= 0.")
        if self.failure_count < 0:
            raise ValueError("failure_count must be >= 0.")
        if self.success_count + self.failure_count != self.rollout_count:
            raise ValueError("success_count + failure_count must equal rollout_count.")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be in range [0.0, 1.0].")
        if not 0.0 <= self.action_switch_rate <= 1.0:
            raise ValueError("action_switch_rate must be in range [0.0, 1.0].")
        for key in self.diagnostics_mean.keys():
            if not str(key).strip():
                raise ValueError("diagnostics_mean keys must be non-empty.")


class RolloutPolicy(Protocol):
    """Policy contract used by rollout harness."""

    def action_for_step(self, step_idx: int, rollout_id: str) -> str:
        """Return action ID for a given rollout step."""


class RolloutEnvironment(Protocol):
    """Environment contract used by rollout harness."""

    def reset(self, rollout_id: str) -> None:
        """Reset environment state for a new rollout."""

    def step(self, action_id: str, step_idx: int, rollout_id: str) -> RolloutStep:
        """Advance one step and return transition info."""


@dataclass(slots=True, frozen=True)
class MockRolloutPolicy:
    """Deterministic cyclic policy for harness testing."""

    action_cycle: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.action_cycle) == 0:
            raise ValueError("action_cycle must be non-empty.")
        for action_id in self.action_cycle:
            if not action_id.strip():
                raise ValueError("action_cycle actions must be non-empty.")

    def action_for_step(self, step_idx: int, rollout_id: str) -> str:
        if step_idx < 0:
            raise ValueError("step_idx must be >= 0.")
        return self.action_cycle[step_idx % len(self.action_cycle)]


@dataclass(slots=True, frozen=True)
class MockRolloutEnvironment:
    """Deterministic mock environment for short-horizon validation."""

    success_on_step: int
    terminal_on_step: int
    reward_on_success: float = 1.0
    reward_on_failure: float = 0.0

    def __post_init__(self) -> None:
        if self.success_on_step < 0:
            raise ValueError("success_on_step must be >= 0.")
        if self.terminal_on_step < 0:
            raise ValueError("terminal_on_step must be >= 0.")
        if self.terminal_on_step < self.success_on_step:
            raise ValueError("terminal_on_step must be >= success_on_step.")

    def reset(self, rollout_id: str) -> None:
        if not rollout_id.strip():
            raise ValueError("rollout_id must be non-empty.")

    def step(self, action_id: str, step_idx: int, rollout_id: str) -> RolloutStep:
        success = step_idx >= self.success_on_step
        done = step_idx >= self.terminal_on_step
        reward = self.reward_on_success if success else self.reward_on_failure
        diagnostics = {
            "progress": min(1.0, float(step_idx + 1) / float(self.terminal_on_step + 1)),
            "stability": 1.0 if action_id.strip() else 0.0,
        }
        return RolloutStep(
            step_idx=step_idx,
            action_id=action_id,
            reward=reward,
            success=success,
            done=done,
            diagnostics=diagnostics,
        )


def run_short_horizon_rollouts(
    *,
    rollout_count: int,
    max_horizon: int,
    policy: RolloutPolicy,
    environment: RolloutEnvironment,
) -> tuple[RolloutTrace, ...]:
    """Execute deterministic short-horizon rollouts."""
    if rollout_count <= 0:
        raise ValueError("rollout_count must be > 0.")
    if max_horizon <= 0:
        raise ValueError("max_horizon must be > 0.")

    traces: list[RolloutTrace] = []
    for rollout_idx in range(rollout_count):
        rollout_id = f"rollout_{rollout_idx:06d}"
        environment.reset(rollout_id)
        steps: list[RolloutStep] = []
        for step_idx in range(max_horizon):
            action_id = policy.action_for_step(step_idx=step_idx, rollout_id=rollout_id)
            step = environment.step(action_id=action_id, step_idx=step_idx, rollout_id=rollout_id)
            steps.append(step)
            if step.done:
                break
        traces.append(
            RolloutTrace(
                rollout_id=rollout_id,
                max_horizon=max_horizon,
                steps=tuple(steps),
            )
        )
    return tuple(traces)


def summarize_rollouts(traces: Sequence[RolloutTrace]) -> RolloutSummary:
    """Compute success/failure and behavior diagnostics across rollouts."""
    if len(traces) == 0:
        raise ValueError("traces must be non-empty.")

    rollout_count = len(traces)
    step_count = sum(len(trace.steps) for trace in traces)
    success_count = sum(1 for trace in traces if any(step.success for step in trace.steps))
    failure_count = rollout_count - success_count
    success_rate = success_count / float(rollout_count)
    mean_reward = sum(step.reward for trace in traces for step in trace.steps) / float(step_count)
    mean_episode_length = step_count / float(rollout_count)

    transitions = 0
    switches = 0
    for trace in traces:
        for idx in range(1, len(trace.steps)):
            transitions += 1
            if trace.steps[idx - 1].action_id != trace.steps[idx].action_id:
                switches += 1
    action_switch_rate = (switches / float(transitions)) if transitions > 0 else 0.0

    diagnostic_keys = {
        key
        for trace in traces
        for step in trace.steps
        for key in step.diagnostics.keys()
    }
    diagnostics_mean: dict[str, float] = {}
    for key in sorted(diagnostic_keys):
        values = [step.diagnostics[key] for trace in traces for step in trace.steps if key in step.diagnostics]
        diagnostics_mean[key] = sum(values) / float(len(values))

    return RolloutSummary(
        rollout_count=rollout_count,
        step_count=step_count,
        success_count=success_count,
        failure_count=failure_count,
        success_rate=success_rate,
        mean_reward=mean_reward,
        mean_episode_length=mean_episode_length,
        action_switch_rate=action_switch_rate,
        diagnostics_mean=diagnostics_mean,
    )

