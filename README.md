# Game agent adaptation (manifest → HF fine-tune → evaluate)

End-to-end pipeline: build a clip manifest from raw episodes, fine-tune a **Hugging Face image classifier** on frames with aligned action IDs, run inference to produce prediction JSON, and report offline metrics (action accuracy, temporal consistency).

The project prioritizes stable data and artifact contracts, deterministic behavior where practical, small testable modules, and config-driven execution (no hardcoded paths). It does not include large-scale distributed training, a full world model, or environment rollouts.

Use **Python 3.12** from the repository root.

## Install

```bash
python3.12 -m pip install -r requirements.txt
```

Run all commands from the repo root.

## 1. Prepare raw episodes

Expected layout under `--input`:

- `<episodes_root>/<episode_id>/frames/` — image files  
- `<episodes_root>/<episode_id>/actions.json` or `actions.csv` — columns `frame`, `action`

Rules:

- `actions.json` and `actions.csv` are mutually exclusive per episode.
- Every frame file must have an action label.

## 2. Build the dataset manifest

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

Output includes `schema_version`, `split_policy`, `episode_splits`, and clip records. Field definitions: `docs/dataset_spec.md`.

## 3. Fine-tune (Hugging Face)

JSON config must include `manifest_path`, `output_dir`, `model_id`, and action vocabulary fields (`action_mapping` and/or path-based fields supported by `src.train.config_io`). See `configs/finetune.example.json`.

Uses `transformers.AutoModelForImageClassification` + `AutoImageProcessor`. Set `model_id` to any HF image-classification checkpoint whose head can be resized (for example `google/vit-base-patch16-224`).

```bash
python3.12 -m scripts.finetune --config configs/finetune.example.json
```

Artifacts:

- `output_dir/hf_model/` — weights + processor  
- `output_dir/label2id.json` — class map  
- `output_dir/finetune_config.json` — copy of config  
- `output_dir/train_metrics.json` — run summary  

## 4. Predict on a split

```bash
python3.12 -m scripts.predict \
  --model-dir outputs/run1 \
  --manifest data/processed/manifest.json \
  --split val \
  --finetune-config outputs/run1/finetune_config.json \
  --output outputs/run1/predictions_val.json
```

## 5. Evaluation report

Offline evaluation is done via `src.eval` (join manifest + predictions, write report JSON). The one-shot driver writes `report_<split>.json` next to your run outputs.

## One-shot pipeline

From the repo root, after editing paths in `main.py` and your finetune template:

```bash
python3.12 main.py
```

This runs manifest build → fine-tune → predict → report (see `main.py` for defaults).

## Layout

- `docs/project_plan.md` — pipeline overview  
- `docs/dataset_spec.md` — manifest field definitions  
- `scripts/` — `build_dataset`, `finetune`, `predict`  
- `src/data/` — schema, splits, manifest loader  
- `src/model/` — action alignment  
- `src/train/` — HF fine-tuning and prediction helpers  
- `src/eval/` — metrics, joining manifest + predictions, report JSON  

## Common issues

- **Manifest not found:** verify `manifest_path` and run `build_dataset` first.  
- **Missing action for frame:** ensure every frame file has an action label entry.  
- **Import errors after branch switches:** ensure `__init__.py` exists under `src/`, `src/data/`, and `scripts/` where used.  

## Design principles

- Keep modules simple and composable.  
- Keep interfaces stable and versioned where practical.  
- Prefer explicit validation over implicit assumptions.  
