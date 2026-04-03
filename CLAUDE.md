# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

End-to-end ML pipeline that fine-tunes HunyuanVideo (8.3B diffusion-based video generation model) with LoRA on the NitroGen gameplay dataset. Builds an action-conditioned game world model: given gamepad action sequences, generates corresponding gameplay video.

## Setup

```bash
python3.12 -m pip install -r requirements.txt
```

All scripts must be run from the repo root using `-m` syntax so `src/` imports resolve correctly.

GPU requirements: 1x A100 80GB minimum for training (LoRA + nf4 quantization).

## Commands

```bash
# Stage 1: Download NitroGen data
python3.12 -m scripts.download_data \
  --output data/nitrogen --shards 0 1 2 \
  --download-videos --max-chunks-per-shard 50

# Stage 2: Build dataset manifest
python3.12 -m scripts.build_dataset \
  --input data/nitrogen --output data/processed/manifest.json \
  --seed 42 --train 0.8 --val 0.1 --test 0.1

# Stage 3: Fine-tune HunyuanVideo with LoRA
python3.12 -m scripts.finetune --config configs/finetune.example.json

# Stage 4: Generate videos from actions
python3.12 -m scripts.predict \
  --model-dir outputs/run1 --dataset-path data/nitrogen \
  --split val --output outputs/run1/generated_videos --max-samples 10

# Full pipeline (edit PipelinePaths in main.py first)
python3.12 main.py

# Tests
pytest tests/
```

## Architecture

The pipeline has five sequential stages:

1. **Download** (`scripts/download_data.py`) — Fetches NitroGen action annotation shards (parquet files + metadata) from HuggingFace (`nvidia/NitroGen`), optionally downloads source gameplay videos from URLs in metadata.

2. **Build manifest** (`scripts/build_dataset.py`) — Scans NitroGen data directory `SHARD_XXXX/<video_id>/<chunk_id>/`, parses parquet action files, assigns train/val/test splits via SHA-256 hashing, writes `manifest.json`.

3. **Fine-tuning** (`src/train/hf_finetune.py`) — Loads `VideoActionDataset` from NitroGen chunks, applies LoRA adapters to HunyuanVideo transformer, trains with diffusion (flow matching) loss. Gamepad actions are encoded as text prompts (MVP) for conditioning. Saves LoRA weights, config, and metrics under `output_dir`.

4. **Generation** (`scripts/predict.py`) — Loads fine-tuned LoRA weights, encodes action sequences as conditioning prompts, generates videos via the HunyuanVideo pipeline. Outputs individual frames and mp4 files.

5. **Evaluation** (`src/eval/`) — Computes temporal consistency, PSNR/SSIM, LPIPS perceptual distance, and optional FVD. Writes `report_<split>.json`.

## Key Module Map

| Path | Role |
|------|------|
| `src/data/schema.py` | Core data contracts: `GamepadAction`, `VideoChunk`, `TrainingSample`, `ChunkMetadata` |
| `src/data/split.py` | `VideoSplitAssigner` — deterministic split via SHA-256 |
| `src/data/loader.py` | `NitroGenDataset` — parquet reading, chunk discovery, video loading |
| `src/model/action_encoder.py` | `GamepadActionEncoder` — text and vector action encoding |
| `src/train/config_io.py` | `VideoGenConfig`, `LoRAConfig` — typed config loading |
| `src/train/hf_finetune.py` | `VideoActionDataset`, `run_hunyuanvideo_lora_finetune`, `generate_video` |
| `src/eval/metrics.py` | `compute_temporal_consistency`, PSNR, SSIM, LPIPS, FID |
| `src/eval/pipeline.py` | Join generated + reference videos for evaluation |
| `src/eval/report.py` | `VideoEvalReport` serialization |
| `main.py` | `PipelinePaths` dataclass + `run_pipeline()` orchestrator |
| `configs/finetune.example.json` | Template for fine-tuning config |

## Data Contracts

All major structs are frozen dataclasses. Key types:

- `GamepadAction`: 17 boolean buttons + 4 joystick floats per frame
- `VideoChunk`: chunk_id, video_id, shard_id, game, video_path, actions, metadata, split
- `TrainingSample`: chunk_id, video_path, actions, prompt, num_frames, resolution, split
- `ChunkMetadata`: uuid, video URL, resolution, timestamps, game info, bounding boxes

NitroGen data layout:
```
data/nitrogen/SHARD_XXXX/<video_id>/<chunk_id>/
  actions_raw.parquet
  actions_processed.parquet
  metadata.json
  video.mp4  (if downloaded)
```

## Configuration

`configs/finetune.example.json` controls fine-tuning. Critical fields:
- `dataset_path`, `output_dir`, `model_id` — required
- `training_type` — `lora` or `full`
- `lora_rank`, `lora_alpha`, `target_modules` — LoRA hyperparameters
- `resolution`, `num_frames` — video generation parameters
- `quantization` — `nf4` for 4-bit quantization
- `action_encoding` — `text` (MVP) or `vector` (future)
- `game_filter` — optional game name filter
- `split_policy` — train/val/test ratios

## Action Encoding

Two approaches (switchable via `action_encoding` config):
- **text**: Converts gamepad states to natural language ("buttons: A; left stick: up") and uses HunyuanVideo's text conditioning channel
- **vector**: Packs actions into (T, 21) float tensor for direct conditioning injection (GameCraft-style, not yet implemented)
