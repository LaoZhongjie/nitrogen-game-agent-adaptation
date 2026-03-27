# NitroGen Post-Training for New Game Skill Adaptation

This repository implements a staged, reproducible post-training pipeline for adapting a pretrained NitroGen gaming agent to a new game domain using small amounts of demonstration data.

The project prioritizes:

- stable data and artifact contracts
- deterministic behavior where practical
- small, testable modules
- config-driven execution (no hardcoded paths)

The project intentionally does not include large-scale distributed training infrastructure or a full world model.

## Stage Status

All planned stages in `docs/project_plan.md` are implemented in this repository:

- Stage 1: dataset contracts and dataset builder
- Stage 2: action alignment module
- Stage 3: fine-tuning entry contract and artifact schemas
- Stage 4: offline evaluation pipeline and report format
- Stage 5: short-horizon rollout validation harness and metrics

Note on Stage 3:

- `scripts.finetune` currently provides a deterministic training contract with `train_stub`, `train_noop`, and `train_mock` backends.
- It is a robust integration skeleton (data flow + artifact contracts), not a production-grade real optimizer loop yet.

## Repository Layout

- `docs/`
  - `project_plan.md`: staged roadmap and goals
  - `dataset_spec.md`: canonical dataset contract
  - `finetune_spec.md`: fine-tuning entry contract
- `scripts/`
  - `build_dataset.py`: build validated clip manifest from raw episodes
  - `finetune.py`: config-driven fine-tuning entry and artifact writer
  - `evaluate.py`: offline evaluation and report generation
  - `rollout_validate.py`: short-horizon rollout validation
- `src/data/`
  - schema types, split assignment, manifest loader
- `src/model/`
  - action alignment interfaces and vocabulary mapper
- `src/train/`
  - runner backends and Stage 3 artifact schema helpers
- `src/eval/`
  - offline metrics, report contracts, prediction-join pipeline, rollout harness
- `tests/`
  - unit and contract tests for all stages

## Requirements

- Python `3.12`
- `pytest`

Install and verify:

```bash
python3.12 -m pip install -U pip pytest
python3.12 -m pytest -q
```

Run all commands from repo root.

## End-to-End Workflow

### 1) Prepare raw episodes

Expected episode layout:

- `<episodes_root>/<episode_id>/frames/`
- `<episodes_root>/<episode_id>/actions.json` or `actions.csv`

Rules:

- `actions.json` and `actions.csv` are mutually exclusive per episode.
- every frame file must have an action label.

### 2) Build a dataset manifest

```bash
python3.12 -m scripts.build_dataset \
  --input data/raw \
  --output data/processed/manifest.json \
  --seed 7 \
  --clip-length 16 \
  --stride 16 \
  --train 0.8 \
  --val 0.1 \
  --test 0.1
```

Output:

- `manifest.json` with `schema_version`, `split_policy`, `episode_splits`, and clip records.

### 3) Run fine-tuning entry (Stage 3 contract)

Create a config JSON, then run:

```bash
python3.12 -m scripts.finetune --config configs/finetune.json
```

Important config fields:

- required: `manifest_path`, `output_dir`
- common: `split`, `train_steps`, `runner_backend`, `dry_run`
- alignment: `action_mapping`, `action_aliases`, `unknown_action_id`, `confidence_floor`

Backends:

- `train_stub`: deterministic decreasing placeholder loss
- `train_noop`: deterministic zero-loss steps
- `train_mock`: deterministic non-trivial loss curve

Primary outputs:

- `output_dir/metrics/latest_metrics.json`
- `output_dir/checkpoints/latest.ckpt` (non-dry-run)
- `output_dir/metrics/training_metadata.json` (non-dry-run)

### 4) Run offline evaluation (Stage 4)

You can evaluate in two modes.

Mode A: prebuilt evaluation records:

```bash
python3.12 -m scripts.evaluate \
  --input data/eval/records.json \
  --output data/eval/offline_report.json \
  --split train
```

Mode B: manifest + predictions join:

```bash
python3.12 -m scripts.evaluate \
  --manifest data/processed/manifest.json \
  --predictions data/eval/predictions.json \
  --output data/eval/offline_report.json \
  --split train
```

Optional in mode B:

- `--allow-missing-clips` to allow partial prediction coverage

Evaluation report schema:

- `v1_offline_evaluation_report`

### 5) Run rollout validation (Stage 5)

Create `configs/rollout_validation.json`, then run:

```bash
python3.12 -m scripts.rollout_validate --config configs/rollout_validation.json
```

Example config keys:

- required: `output_path`
- rollout control: `rollout_count`, `max_horizon`
- deterministic policy/env knobs:
  - `action_cycle`
  - `success_on_step`
  - `terminal_on_step`
  - `reward_on_success`
  - `reward_on_failure`

Rollout report schema:

- `v1_rollout_validation_report`

## Minimal Config Examples

### `configs/finetune.json`

```json
{
  "manifest_path": "data/processed/manifest.json",
  "output_dir": "outputs/finetune_run_001",
  "split": "train",
  "dry_run": false,
  "runner_backend": "train_mock",
  "train_steps": 5,
  "mock_learning_rate": 0.1,
  "action_mapping": {
    "left": "move_left",
    "jump": "jump"
  },
  "action_aliases": {
    "move left": "left"
  },
  "unknown_action_id": "unknown",
  "confidence_floor": 0.0,
  "seed": 7
}
```

### `configs/rollout_validation.json`

```json
{
  "output_path": "outputs/rollout_validation/report.json",
  "rollout_count": 4,
  "max_horizon": 8,
  "action_cycle": ["move_left", "jump"],
  "success_on_step": 3,
  "terminal_on_step": 5,
  "reward_on_success": 1.0,
  "reward_on_failure": 0.0
}
```

## Testing

Run all tests:

```bash
python3.12 -m pytest -q
```

Targeted examples:

```bash
python3.12 -m pytest -q tests/test_finetune.py
python3.12 -m pytest -q tests/test_evaluate_cli.py tests/test_eval_pipeline.py tests/test_eval_metrics.py
python3.12 -m pytest -q tests/test_rollout_validation.py
```

## Common Issues

- `manifest not found`: verify `manifest_path` and run `build_dataset` first.
- `missing action for frame ...`: ensure every frame file has an action label entry.
- `missing predictions for manifest clips`: either provide complete predictions or use `--allow-missing-clips`.
- import errors after branch switches: ensure `__init__.py` files in `src/`, `src/data/`, `scripts/`, and `tests/` exist.

## Design Principles

- keep modules simple and composable
- keep interfaces stable and versioned
- prefer explicit validation over implicit assumptions
- keep scripts runnable from repo root
