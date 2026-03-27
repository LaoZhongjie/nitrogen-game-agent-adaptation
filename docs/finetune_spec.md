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

When `save_summary=true`, summary is written to:

- `output_dir/metrics/latest_metrics.json`

In `train_stub` mode (`dry_run=false`), placeholder artifacts are also written:

- `output_dir/checkpoints/latest.ckpt`
- `output_dir/metrics/training_metadata.json`

`training_metadata.json` includes deterministic `steps` entries:

- `step` (1-indexed integer)
- `samples_seen` (constant per step from dry-run pass)
- `unknown_ratio` (constant per step from dry-run pass)
- `pseudo_loss` (deterministic value: `1.0 / (step + seed)` )

## Current limitation

`train_stub` does not run model optimization yet; it only validates integration and emits deterministic artifacts.
