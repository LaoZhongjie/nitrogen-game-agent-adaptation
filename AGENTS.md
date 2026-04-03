# AGENTS.md

Python 3.12 project: NitroGen dataset download, HunyuanVideo LoRA fine-tuning for action-conditioned video generation, and offline evaluation.

## Dependencies

Install from the repo root:

```bash
python3.12 -m pip install -r requirements.txt
```

Requires GPU with >= 24GB VRAM for inference, >= 80GB for training.

## One-shot run

```bash
python3.12 main.py
```

Edit `PATHS` and related constants in `main.py` first. This runs download -> manifest -> fine-tune -> generate -> evaluate.

## CLIs (run from repo root)

Download NitroGen data:

```bash
python3.12 -m scripts.download_data --output data/nitrogen --shards 0 1 2 --download-videos
```

Build manifest:

```bash
python3.12 -m scripts.build_dataset --input data/nitrogen --output data/processed/manifest.json
```

Fine-tune:

```bash
python3.12 -m scripts.finetune --config configs/finetune.example.json
```

Generate videos:

```bash
python3.12 -m scripts.predict --model-dir outputs/run1 --dataset-path data/nitrogen \
  --split val --output outputs/run1/generated_videos --max-samples 10
```

## Imports

The package lives under `src/`. Use `from src....` and run modules as `python3.12 -m scripts.<name>`. Ensure `src/`, `src/data/`, `scripts/`, and `tests/` contain `__init__.py` where present.
