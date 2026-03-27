# AGENTS.md

## Cursor Cloud specific instructions

This is a pure Python 3.12 project with no external service dependencies. The sole third-party runtime dependency is `pytest`.

### Running tests

```bash
python3.12 -m pytest -q
```

All scripts and tests must be run from the repo root (`/workspace`).

### Running the CLI

The dataset builder is invoked as a Python module:

```bash
python3.12 -m scripts.build_dataset --input <episodes_root> --output <manifest.json> [--seed N] [--clip-length N] [--stride N] [--train F] [--val F] [--test F]
```

### Package structure gotcha

The project uses `src/` as a top-level Python package (not `src`-layout with a `pyproject.toml`). Imports look like `from src.data.schema import ...` and `from scripts.build_dataset import ...`. The `__init__.py` files in `src/`, `src/data/`, `scripts/`, and `tests/` are required for these imports to work. If imports break after pulling, verify these files exist.

### No linter configured

There is no linter (ruff, flake8, mypy, pyright) configured in the repo. The `.gitignore` anticipates `.mypy_cache/`, `.ruff_cache/`, etc., but no config files or dependencies exist yet.
