from __future__ import annotations

from src.train.runner import (
    resolve_runner_backend,
    run_mock_runner_steps,
    run_noop_runner_steps,
    run_train_stub_steps,
)


def test_run_train_stub_steps_emits_expected_step_shape() -> None:
    metrics = run_train_stub_steps(
        train_steps=3,
        known_ratio=0.75,
        unknown_ratio=0.25,
        processed_samples=5,
    )

    assert len(metrics) == 3
    assert [entry.step for entry in metrics] == [1, 2, 3]
    assert metrics[0].samples_seen == 5
    assert metrics[-1].samples_seen == 5
    assert metrics[0].known_ratio == 0.75


def test_run_train_stub_steps_loss_is_monotonic_non_increasing() -> None:
    metrics = run_train_stub_steps(
        train_steps=4,
        known_ratio=0.6,
        unknown_ratio=0.4,
        processed_samples=2,
    )
    losses = [entry.loss for entry in metrics]

    assert losses == sorted(losses, reverse=True)


def test_run_noop_runner_steps_returns_zero_loss() -> None:
    metrics = run_noop_runner_steps(train_steps=3, known_ratio=0.9, processed_samples=2)

    assert len(metrics) == 3
    assert all(entry.loss == 0.0 for entry in metrics)
    assert all(entry.known_ratio == 0.9 for entry in metrics)
    assert all(entry.samples_seen == 2 for entry in metrics)


def test_resolve_runner_backend_rejects_unknown_backend() -> None:
    try:
        resolve_runner_backend("not-real-backend")
    except ValueError as exc:
        assert "unsupported runner backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown backend.")


def test_run_mock_runner_steps_emits_nontrivial_loss_curve() -> None:
    metrics = run_mock_runner_steps(
        train_steps=3,
        known_ratio=0.7,
        unknown_ratio=0.3,
        processed_samples=4,
        mock_learning_rate=0.1,
    )

    assert len(metrics) == 3
    assert metrics[0].loss > metrics[1].loss > metrics[2].loss
    assert all(entry.samples_seen == 4 for entry in metrics)
