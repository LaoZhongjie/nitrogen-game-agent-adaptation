# Project Plan: NitroGen Post-Training for New Game Skill Adaptation

## Objective

Adapt a pretrained NitroGen gaming agent to a new game domain using a small amount of video-action demonstration data, with an emphasis on reproducibility and practical utility.

## Non-Goals

- Building a full world model
- Solving online RL from scratch
- Optimizing large-scale distributed training infrastructure in this repo

## Principles

- Modular, testable components
- Stable data contracts between stages
- Config-driven execution (no hardcoded paths)
- Deterministic behavior where feasible

## Staged Roadmap

### Stage 1 (Current): Dataset Contracts and Builder

Deliverables:

- `README.md`
- `docs/project_plan.md`
- `docs/dataset_spec.md`
- `src/data/schema.py`
- `scripts/build_dataset.py`
- `tests/test_schema.py`

Exit criteria:

- Dataset schema can represent episode/clip/frame/action data
- Build script can validate inputs and write a manifest
- Tests validate schema behavior and split policy invariants

### Stage 2: Action Alignment Module

Planned outputs:

- Alignment interface and baseline implementation
- Action vocabulary mapping and confidence metadata
- Unit tests for alignment edge cases

### Stage 3: Fine-Tuning Entry Point

Planned outputs:

- Config-driven fine-tuning script for pretrained NitroGen policy head/adapters
- Dataset loader integration
- Checkpoint and metrics logging contracts

### Stage 4: Offline Evaluation

Planned outputs:

- Offline metrics pipeline (action accuracy, temporal consistency, split-wise metrics)
- Evaluation report artifact format

### Stage 5: Short-Horizon Rollout Validation

Planned outputs:

- Short-horizon rollout harness in target game environment
- Basic success/failure and behavior diagnostic metrics

## Risks and Mitigations

- **Data ambiguity:** enforce strict schema validation and required metadata.
- **Small data overfitting:** include clear split policy and deterministic split assignment.
- **Pipeline drift:** lock interfaces in docs and unit tests before expanding stages.
