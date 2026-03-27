# NitroGen Post-Training for New Game Skill Adaptation

This repository provides a practical, research-style post-training pipeline for adapting a pretrained NitroGen gaming agent to a new game domain with limited video-action demonstrations.

The scope is intentionally narrow: define a clean data contract and reproducible pipeline stages for dataset construction, action alignment, fine-tuning, and evaluation. We do **not** build a full world model.

## Current Stage

Stage 3 fine-tuning entry has started with config-driven dry-run and pluggable train backends:

- Manifest loading and action-alignment integration
- Deterministic dry-run / `train_stub` / `train_noop` / `train_mock` artifact contracts
- Train-step runner abstraction (`src/train/runner.py`) with injectable runner-factory path for staged training-loop evolution
- Versioned mock-backend checkpoint state snapshots via `src/train/state.py` (`TrainingState`)

## Project Structure

- `docs/` - planning and data specifications
- `scripts/` - CLI scripts runnable from repo root
- `src/data/` - data contracts and validation schemas
- `src/model/` - action alignment and mapping modules
- `src/train/` - training runner abstractions
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

## Guiding Principles

- Keep modules small and testable
- Use explicit schemas and type hints
- Avoid hardcoded paths
- Keep configs separate from code
- Build incrementally in stages
