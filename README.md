# NitroGen + HunyuanVideo World Model

End-to-end pipeline that fine-tunes **HunyuanVideo** (8.3B video generation model) with LoRA on the **NitroGen** gameplay dataset to build an action-conditioned game world model. Given a sequence of gamepad actions, the model generates corresponding gameplay video.

The project uses **Python 3.12** and should be run from the repository root.

## Install

```bash
python3.12 -m pip install -r requirements.txt
```

GPU requirements (LoRA + nf4 quantization):

- Training: 1x A100 80GB minimum, 2-4x A100 recommended
- Inference: 1x A100 or RTX 4090 24GB with quantization

## Pipeline stages

### 1. Download NitroGen data

```bash
python3.12 -m scripts.download_data \
  --output data/nitrogen \
  --shards 0 1 2 \
  --download-videos \
  --max-chunks-per-shard 50
```

The NitroGen dataset (`nvidia/NitroGen` on HuggingFace) contains per-frame gamepad action annotations for 40,000 hours of gameplay across 1,000+ games. The downloader fetches action parquet files and optionally downloads source videos.

### 2. Build dataset manifest

```bash
python3.12 -m scripts.build_dataset \
  --input data/nitrogen \
  --output data/processed/manifest.json \
  --seed 42 \
  --train 0.8 --val 0.1 --test 0.1
```

### 3. Fine-tune HunyuanVideo with LoRA

```bash
python3.12 -m scripts.finetune --config configs/finetune.example.json
```

Key config fields in `configs/finetune.example.json`:

| Field | Description |
|-------|-------------|
| `model_id` | HuggingFace model ID (default: `tencent/HunyuanVideo`) |
| `training_type` | `lora` or `full` |
| `lora_rank` / `lora_alpha` | LoRA hyperparameters |
| `resolution` | Target video resolution `[H, W]` |
| `num_frames` | Target frame count (must satisfy 4k or 4k+1) |
| `quantization` | `nf4` for 4-bit quantization |
| `action_encoding` | `text` (MVP) or `vector` (future) |
| `game_filter` | Optional game name filter |

### 4. Generate videos from actions

```bash
python3.12 -m scripts.predict \
  --model-dir outputs/run1 \
  --dataset-path data/nitrogen \
  --split val \
  --output outputs/run1/generated_videos \
  --max-samples 10
```

### 5. Evaluate generated videos

Evaluation is run as part of the full pipeline. Metrics include:

- **Temporal consistency** — smoothness between adjacent generated frames
- **PSNR / SSIM** — per-frame quality vs reference (when available)
- **LPIPS** — perceptual similarity
- **FVD** — Frechet Video Distance (distribution-level)

## One-shot pipeline

Edit constants in `main.py`, then:

```bash
python3.12 main.py
```

Runs: download -> manifest -> fine-tune -> generate -> evaluate.

## Project layout

| Path | Role |
|------|------|
| `src/data/schema.py` | Data contracts: `GamepadAction`, `VideoChunk`, `TrainingSample` |
| `src/data/split.py` | `VideoSplitAssigner` — deterministic split via SHA-256 |
| `src/data/loader.py` | `NitroGenDataset` — parquet + video loading |
| `src/model/action_encoder.py` | `GamepadActionEncoder` — text and vector action encoding |
| `src/train/config_io.py` | `VideoGenConfig` — typed config loading |
| `src/train/hf_finetune.py` | `VideoActionDataset`, LoRA training loop, video generation |
| `src/eval/metrics.py` | PSNR, SSIM, LPIPS, temporal consistency |
| `src/eval/pipeline.py` | Join generated + reference videos for evaluation |
| `src/eval/report.py` | Evaluation report serialization |
| `scripts/download_data.py` | NitroGen HuggingFace downloader |
| `scripts/build_dataset.py` | Manifest builder |
| `scripts/finetune.py` | Fine-tuning CLI |
| `scripts/predict.py` | Video generation CLI |
| `main.py` | Full pipeline orchestrator |
| `configs/finetune.example.json` | Fine-tuning config template |

## NitroGen action format

Each frame has 17 boolean button states and 2 joystick (x, y) positions:

- Buttons: `dpad_down`, `dpad_left`, `dpad_right`, `dpad_up`, `left_shoulder`, `left_thumb`, `left_trigger`, `right_shoulder`, `right_thumb`, `right_trigger`, `south` (A), `west` (X), `east` (B), `north` (Y), `back`, `start`, `guide`
- Joysticks: `j_left` and `j_right`, each (x, y) in [-1, 1]

## Action conditioning approaches

1. **Text encoding (MVP)**: Gamepad states are converted to natural-language descriptions (e.g., "buttons: A; left stick: up") and fed through HunyuanVideo's text conditioning channel.

2. **Vector encoding (future)**: Actions packed into dense float tensors and injected as additional conditioning signals, following the Hunyuan-GameCraft approach with Plucker embeddings.

## References

- [NitroGen paper](https://arxiv.org/abs/2601.02427) — dataset and behavior cloning model
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo/) — base video generation model
- [Hunyuan-GameCraft](https://hunyuan-gamecraft.github.io/) — action-conditioned game world model on HunyuanVideo
