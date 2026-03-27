"""Offline evaluation utilities."""

from src.eval.metrics import (
    EvaluationRecord,
    EvaluationSummary,
    SplitEvaluationMetrics,
    compute_action_accuracy,
    compute_temporal_consistency,
    evaluate_records,
)
from src.eval.report import (
    EvaluationReport,
    build_evaluation_report,
    evaluate_records_file,
    load_evaluation_records,
    write_evaluation_report,
)
from src.eval.pipeline import (
    PredictionRecord,
    build_evaluation_records_from_manifest_predictions,
    load_prediction_records,
)
from src.eval.rollout import (
    MockRolloutEnvironment,
    MockRolloutPolicy,
    RolloutEnvironment,
    RolloutPolicy,
    RolloutStep,
    RolloutSummary,
    RolloutTrace,
    run_short_horizon_rollouts,
    summarize_rollouts,
)

__all__ = [
    "EvaluationRecord",
    "EvaluationSummary",
    "SplitEvaluationMetrics",
    "compute_action_accuracy",
    "compute_temporal_consistency",
    "evaluate_records",
    "EvaluationReport",
    "build_evaluation_report",
    "evaluate_records_file",
    "load_evaluation_records",
    "write_evaluation_report",
    "PredictionRecord",
    "build_evaluation_records_from_manifest_predictions",
    "load_prediction_records",
    "RolloutStep",
    "RolloutTrace",
    "RolloutSummary",
    "RolloutPolicy",
    "RolloutEnvironment",
    "MockRolloutPolicy",
    "MockRolloutEnvironment",
    "run_short_horizon_rollouts",
    "summarize_rollouts",
]
