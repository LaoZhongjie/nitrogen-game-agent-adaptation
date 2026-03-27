"""Short-horizon rollout validation CLI for Stage 5."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Support both:
# - python -m scripts.rollout_validate (recommended)
# - python scripts/rollout_validate.py (common)
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.eval.rollout import (
    MockRolloutEnvironment,
    MockRolloutPolicy,
    RolloutSummary,
    RolloutTrace,
    run_short_horizon_rollouts,
    summarize_rollouts,
)


@dataclass(slots=True, frozen=True)
class RolloutValidationConfig:
    """Config for deterministic short-horizon rollout validation."""

    output_path: str
    rollout_count: int = 4
    max_horizon: int = 8
    action_cycle: tuple[str, ...] = ("move_left", "jump")
    success_on_step: int = 3
    terminal_on_step: int = 5
    reward_on_success: float = 1.0
    reward_on_failure: float = 0.0

    def __post_init__(self) -> None:
        if not self.output_path.strip():
            raise ValueError("output_path must be non-empty.")
        if self.rollout_count <= 0:
            raise ValueError("rollout_count must be > 0.")
        if self.max_horizon <= 0:
            raise ValueError("max_horizon must be > 0.")
        if len(self.action_cycle) == 0:
            raise ValueError("action_cycle must be non-empty.")
        if self.success_on_step < 0:
            raise ValueError("success_on_step must be >= 0.")
        if self.terminal_on_step < self.success_on_step:
            raise ValueError("terminal_on_step must be >= success_on_step.")


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at: {path}")
    return raw


def load_config(config_path: Path) -> RolloutValidationConfig:
    """Load rollout validation config from JSON."""
    raw = _load_json_object(config_path)
    action_cycle_raw = raw.get("action_cycle", ["move_left", "jump"])
    if not isinstance(action_cycle_raw, list):
        raise ValueError("action_cycle must be an array.")
    return RolloutValidationConfig(
        output_path=str(raw["output_path"]),
        rollout_count=int(raw.get("rollout_count", 4)),
        max_horizon=int(raw.get("max_horizon", 8)),
        action_cycle=tuple(str(action) for action in action_cycle_raw),
        success_on_step=int(raw.get("success_on_step", 3)),
        terminal_on_step=int(raw.get("terminal_on_step", 5)),
        reward_on_success=float(raw.get("reward_on_success", 1.0)),
        reward_on_failure=float(raw.get("reward_on_failure", 0.0)),
    )


def _trace_to_dict(trace: RolloutTrace) -> dict[str, Any]:
    return {
        "rollout_id": trace.rollout_id,
        "max_horizon": trace.max_horizon,
        "steps": [
            {
                "step_idx": step.step_idx,
                "action_id": step.action_id,
                "reward": step.reward,
                "success": step.success,
                "done": step.done,
                "diagnostics": dict(step.diagnostics),
            }
            for step in trace.steps
        ],
    }


def _summary_to_dict(summary: RolloutSummary) -> dict[str, Any]:
    return {
        "rollout_count": summary.rollout_count,
        "step_count": summary.step_count,
        "success_count": summary.success_count,
        "failure_count": summary.failure_count,
        "success_rate": summary.success_rate,
        "mean_reward": summary.mean_reward,
        "mean_episode_length": summary.mean_episode_length,
        "action_switch_rate": summary.action_switch_rate,
        "diagnostics_mean": dict(summary.diagnostics_mean),
    }


def run_rollout_validation(config: RolloutValidationConfig) -> dict[str, Any]:
    """Run deterministic rollout harness and emit report payload."""
    policy = MockRolloutPolicy(action_cycle=config.action_cycle)
    environment = MockRolloutEnvironment(
        success_on_step=config.success_on_step,
        terminal_on_step=config.terminal_on_step,
        reward_on_success=config.reward_on_success,
        reward_on_failure=config.reward_on_failure,
    )
    traces = run_short_horizon_rollouts(
        rollout_count=config.rollout_count,
        max_horizon=config.max_horizon,
        policy=policy,
        environment=environment,
    )
    summary = summarize_rollouts(traces)
    payload = {
        "schema_version": "v1_rollout_validation_report",
        "config": {
            "rollout_count": config.rollout_count,
            "max_horizon": config.max_horizon,
            "action_cycle": list(config.action_cycle),
            "success_on_step": config.success_on_step,
            "terminal_on_step": config.terminal_on_step,
            "reward_on_success": config.reward_on_success,
            "reward_on_failure": config.reward_on_failure,
        },
        "summary": _summary_to_dict(summary),
        "traces": [_trace_to_dict(trace) for trace in traces],
    }
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI args for rollout validation."""
    parser = argparse.ArgumentParser(description="Run short-horizon rollout validation.")
    parser.add_argument("--config", required=True, help="Path to rollout validation config JSON.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(Path(args.config))
    payload = run_rollout_validation(config)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
