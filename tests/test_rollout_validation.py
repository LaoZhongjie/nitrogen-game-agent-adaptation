from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rollout_validate import RolloutValidationConfig, main as rollout_main, run_rollout_validation
from src.eval.rollout import (
    MockRolloutEnvironment,
    MockRolloutPolicy,
    RolloutSummary,
    run_short_horizon_rollouts,
    summarize_rollouts,
)


def test_run_short_horizon_rollouts_respects_horizon_and_done() -> None:
    policy = MockRolloutPolicy(action_cycle=("left", "jump"))
    env = MockRolloutEnvironment(success_on_step=1, terminal_on_step=2)

    traces = run_short_horizon_rollouts(
        rollout_count=2,
        max_horizon=5,
        policy=policy,
        environment=env,
    )

    assert len(traces) == 2
    assert all(len(trace.steps) == 3 for trace in traces)  # steps 0,1,2 then done
    assert traces[0].steps[-1].done is True


def test_summarize_rollouts_emits_success_failure_and_diagnostics() -> None:
    policy = MockRolloutPolicy(action_cycle=("left", "jump"))
    env = MockRolloutEnvironment(success_on_step=1, terminal_on_step=2, reward_on_success=2.0, reward_on_failure=-1.0)
    traces = run_short_horizon_rollouts(
        rollout_count=3,
        max_horizon=5,
        policy=policy,
        environment=env,
    )

    summary = summarize_rollouts(traces)

    assert isinstance(summary, RolloutSummary)
    assert summary.rollout_count == 3
    assert summary.success_count == 3
    assert summary.failure_count == 0
    assert summary.success_rate == 1.0
    assert summary.step_count == 9
    assert summary.mean_episode_length == 3.0
    assert 0.0 <= summary.action_switch_rate <= 1.0
    assert set(summary.diagnostics_mean.keys()) == {"progress", "stability"}


def test_run_rollout_validation_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "rollout_validation.json"
    cfg = RolloutValidationConfig(
        output_path=str(output_path),
        rollout_count=2,
        max_horizon=4,
        action_cycle=("left", "jump"),
        success_on_step=1,
        terminal_on_step=2,
        reward_on_success=1.5,
        reward_on_failure=0.0,
    )

    payload = run_rollout_validation(cfg)

    assert output_path.exists()
    disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1_rollout_validation_report"
    assert disk_payload["schema_version"] == "v1_rollout_validation_report"
    assert payload["summary"]["rollout_count"] == 2
    assert payload["summary"]["success_count"] == 2
    assert len(payload["traces"]) == 2


def test_rollout_config_rejects_invalid_terminal_order() -> None:
    with pytest.raises(ValueError, match="terminal_on_step must be >= success_on_step"):
        RolloutValidationConfig(
            output_path="out.json",
            success_on_step=3,
            terminal_on_step=1,
        )


def test_rollout_validate_cli_main(tmp_path: Path, monkeypatch: object, capsys: object) -> None:
    output_path = tmp_path / "reports" / "rollout_validation.json"
    config_path = tmp_path / "rollout_config.json"
    config_path.write_text(
        json.dumps(
            {
                "output_path": str(output_path),
                "rollout_count": 2,
                "max_horizon": 4,
                "action_cycle": ["left", "jump"],
                "success_on_step": 1,
                "terminal_on_step": 2,
                "reward_on_success": 1.5,
                "reward_on_failure": 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["rollout_validate.py", "--config", str(config_path)])
    rollout_main()
    printed = json.loads(capsys.readouterr().out)

    assert printed["rollout_count"] == 2
    assert printed["success_count"] == 2
    assert output_path.exists()
