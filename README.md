# NitroGen Post-Training for New Game Skill Adaptation

This repository provides a practical, research-style post-training pipeline for adapting a pretrained NitroGen gaming agent to a new game domain with limited video-action demonstrations.

The scope is intentionally narrow: define a clean data contract and reproducible pipeline stages for dataset construction, action alignment, fine-tuning, and evaluation. We do **not** build a full world model.

## Current Stage

Stage 5 short-horizon rollout validation is in place with deterministic harness contracts:

- Short-horizon rollout harness (`src/eval/rollout.py`) with typed trace/summary contracts
- Success/failure and behavior diagnostics (`success_rate`, `mean_reward`, `action_switch_rate`, diagnostics means)
- Config-driven rollout validation CLI (`scripts/rollout_validate.py`) that writes report artifacts

## Project Structure

- `docs/` - planning and data specifications
- `scripts/` - CLI scripts runnable from repo root
- `src/data/` - data contracts and validation schemas
- `src/model/` - action alignment and mapping modules
- `src/train/` - training runner abstractions
- `src/eval/` - offline metrics, reports, and rollout validation
- `tests/` - unit tests

## Quick Start

Use Python 3.12.

```bash
python3.12 -m pip install -U pip pytest
python3.12 -m pytest -q
```

### Build a Dataset Manifest

The dataset builder scans an input root where each episode is:

- `data/raw/<episode_id>/frames/` (frame image files)
- `data/raw/<episode_id>/actions.json` or `actions.csv` (frame filename → action label)

```bash
python3.12 scripts/build_dataset.py --input data/raw --output data/processed/manifest.json
```

This creates a normalized manifest JSON file containing validated episode records and split information.

### Run Rollout Validation

```bash
python3.12 -m scripts.rollout_validate --config configs/rollout_validation.json
```

Example config fields:

- `output_path` (required)
- `rollout_count`, `max_horizon`
- `action_cycle`
- `success_on_step`, `terminal_on_step`
- `reward_on_success`, `reward_on_failure`

## Guiding Principles

- Keep modules small and testable
- Use explicit schemas and type hints
- Avoid hardcoded paths
- Keep configs separate from code
- Build incrementally in stages
