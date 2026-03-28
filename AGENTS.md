# AGENTS.md

Python 3.12 project: dataset manifest, Hugging Face image-classification fine-tuning, prediction export, and offline evaluation.

## Dependencies

Install from the repo root:

```bash
python3.12 -m pip install -r requirements.txt
```

## One-shot run

```bash
python3.12 main.py
```

Edit `PATHS` and related constants in `main.py` first. This runs build → train → predict → report.

## CLIs (run from repo root)

Build manifest:

```bash
python3.12 -m scripts.build_dataset --input <episodes_root> --output <manifest.json>
```

Fine-tune:

```bash
python3.12 -m scripts.finetune --config <config.json>
```

Predict (optional; `main.py` already does this):

```bash
python3.12 -m scripts.predict --model-dir <output_dir> --manifest <manifest.json> \
  --split val --finetune-config <config.json> --output <predictions.json>
```

## Imports

The package lives under `src/`. Use `from src....` and run modules as `python3.12 -m scripts.<name>`. Ensure `src/`, `src/data/`, `scripts/`, and `tests/` contain `__init__.py` where present.
