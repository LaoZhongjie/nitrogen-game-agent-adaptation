# Project plan

Single pipeline:

1. **Dataset** — Raw episodes → validated clip manifest (`scripts/build_dataset`, `docs/dataset_spec.md`).
2. **Alignment** — Map raw action strings to canonical `action_id` for training and eval (`src/model/alignment.py`).
3. **Fine-tuning** — HF image classifier on frames (`scripts/finetune`, `src/train/hf_finetune.py`).
4. **Prediction + metrics** — `scripts/predict` then `scripts/evaluate` for accuracy and temporal consistency.

Non-goals for this repository: game environment rollouts, non-HF training stacks, and hosting inference services.
