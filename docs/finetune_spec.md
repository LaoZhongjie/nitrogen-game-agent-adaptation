# Fine-Tuning Entry Spec (Stage 3 Starter)

This document describes the current `scripts/finetune.py` contract.

## Scope

The Stage 3 starter currently supports:

- config-driven execution from a JSON file
- dataset loading via `ManifestDataset`
- action alignment integration via `VocabularyActionAligner`
- two execution modes:
  - `dry_run`: validation-only summary path
  - `train_stub`: placeholder non-dry-run artifact path (no optimization loop yet)

## Config schema

Required:

- `manifest_path` (`str`): path to built manifest JSON
- `output_dir` (`str`): root output directory

Optional:

- `split` (`"train" | "val" | "test"`): dataset split filter
- `max_samples` (`int`): max number of samples to process
- `dry_run` (`bool`, default `true`)
- `save_summary` (`bool`, default `true`)
- `action_mapping` (`dict[str, str]`): inline mapping from raw action text to canonical action ID
- `action_mapping_path` (`str`): JSON object path for mapping (mutually exclusive with `action_mapping`)
- `action_aliases` (`dict[str, str]`): inline alias map
- `action_aliases_path` (`str`): JSON object path for aliases (mutually exclusive with `action_aliases`)
- `unknown_action_id` (`str`, default `"unknown"`)
- `confidence_floor` (`float`, default `0.0`)
- `seed` (`int`, default `0`)
- `train_steps` (`int`, default `1`): pseudo training steps in `train_stub` mode
- `runner_backend` (`"train_stub" | "train_noop" | "train_mock"`, default `"train_stub"`): runner backend used when `dry_run=false`
- `mock_learning_rate` (`float`, default `0.05`): deterministic rate used only when `runner_backend="train_mock"`

Normalization behavior:

- mapping and alias keys are lowercased and stripped
- mapping and alias values are stringified and stripped

## Outputs

Returned summary fields include:

- `mode` (`"dry_run"` or `"train_stub"`)
- `processed_samples`
- `total_action_labels`
- `unknown_action_labels`
- `unknown_ratio`
- `checkpoint_path`
- `metrics_path`
- `training_metadata_path`
- `seed`
- `runner_backend`
- `mock_learning_rate`
- `train_backend_metadata` (`dict`): backend-specific parameter snapshot for reproducibility

When `save_summary=true`, summary is written to:

- `output_dir/metrics/latest_metrics.json`

In non-dry-run backend modes (`dry_run=false`), placeholder artifacts are also written:

- `output_dir/checkpoints/latest.ckpt`
- `output_dir/metrics/training_metadata.json`

`latest.ckpt` is a JSON checkpoint metadata record with a stable schema:

- `checkpoint_version` (`str`, current `"v1"`)
- `backend` (`str`)
- `step_count` (`int`)
- `state_digest` (`str`): deterministic digest-like token for backend state snapshot
- `seed` (`int`)
- `processed_samples` (`int`)

`training_metadata.json` includes deterministic `step_metrics` entries and backend metadata:

- `step` (1-indexed integer)
- `samples_seen` (constant per step from dry-run pass)
- `known_ratio` (constant per step from dry-run pass)
- `loss` (deterministic value derived from unknown ratio and step index)
- `train_backend_metadata` (`dict`) mirrors the backend snapshot in summary

If `runner_backend="train_noop"`, `step_metrics` contains deterministic zero-loss entries.

If `runner_backend="train_mock"`, `step_metrics` contains deterministic non-zero loss values shaped by:

- unknown ratio from aligned labels
- step index
- `mock_learning_rate`

## Integration extension point

`run_finetune(config, runner_factory=...)` supports injecting a custom backend resolver for tests and staged integration.
If omitted, built-in backend dispatch is used.

## Current limitation

Current backends do not perform real model optimization; they validate execution contracts and emit deterministic artifacts.
