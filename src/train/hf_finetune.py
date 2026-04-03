"""HunyuanVideo LoRA fine-tuning: action-conditioned video generation.

Implements the training loop for fine-tuning HunyuanVideo with LoRA adapters
on NitroGen gameplay data. Text-based action conditioning is used as the
MVP approach (actions are encoded into text prompts).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.loader import NitroGenDataset, chunk_to_training_sample
from src.data.mp4_validate import is_probably_valid_mp4
from src.data.schema import SplitName, SplitPolicy, TrainingSample
from src.model.action_encoder import GamepadActionEncoder
from src.train.config_io import VideoGenConfig

logger = logging.getLogger(__name__)

_decord_missing_logged = False


def _load_video_frames_decord(
    video_path: str,
    num_frames: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    import cv2
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()

    h_target, w_target = resolution
    if frames.shape[1] != h_target or frames.shape[2] != w_target:
        frames = np.stack(
            [cv2.resize(f, (w_target, h_target), interpolation=cv2.INTER_LINEAR) for f in frames]
        )
    return frames


def _load_video_frames_opencv(
    video_path: str,
    num_frames: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Decode video with OpenCV (BGR→RGB). Loads all frames then subsamples — fine for short NitroGen chunks."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    rgb_frames: list[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not rgb_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    total = len(rgb_frames)
    if total <= num_frames:
        chosen = rgb_frames
    else:
        idxs = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        chosen = [rgb_frames[i] for i in idxs]
    frames = np.stack(chosen, axis=0)

    h_target, w_target = resolution
    if frames.shape[1] != h_target or frames.shape[2] != w_target:
        frames = np.stack(
            [cv2.resize(f, (w_target, h_target), interpolation=cv2.INTER_LINEAR) for f in frames]
        )
    return frames


def _load_video_frames(
    video_path: str,
    num_frames: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Load and preprocess video frames (decord if available, else OpenCV).

    Returns an array of shape ``(T, H, W, 3)`` with uint8 pixel values.
    ``resolution`` is ``(height, width)``.
    """
    global _decord_missing_logged
    try:
        return _load_video_frames_decord(video_path, num_frames, resolution)
    except ImportError:
        if not _decord_missing_logged:
            logger.warning(
                "decord not installed — using OpenCV for video frames. "
                "Install decord for faster IO: pip install decord"
            )
            _decord_missing_logged = True
        return _load_video_frames_opencv(video_path, num_frames, resolution)


class VideoActionDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset yielding video latents and text prompts for diffusion training."""

    def __init__(
        self,
        samples: Sequence[TrainingSample],
        action_encoder: GamepadActionEncoder,
    ) -> None:
        self._samples = list(samples)
        self._encoder = action_encoder

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one training item; skip corrupt / incomplete ``video.mp4`` (try other chunks)."""
        n = len(self._samples)
        if n == 0:
            raise RuntimeError("VideoActionDataset has no samples.")

        last_exc: BaseException | None = None
        for offset in range(n):
            i = (index + offset) % n
            sample = self._samples[i]
            try:
                prompt = self._encoder.encode_conditioning_prompt(
                    actions=sample.actions,
                    game_name="",
                    base_prompt=sample.prompt,
                )
                frames = _load_video_frames(
                    video_path=sample.video_path,
                    num_frames=sample.num_frames,
                    resolution=sample.resolution,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Unreadable video (using another chunk for this step) chunk=%s path=%s: %s",
                    sample.chunk_id,
                    sample.video_path,
                    exc,
                )
                continue

            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            return {
                "pixel_values": pixel_values,
                "prompt": prompt,
                "chunk_id": sample.chunk_id,
            }

        raise RuntimeError(
            f"No readable video after cycling all {n} samples (start index {index}). "
            f"Delete bad mp4 files and re-download, or run download with videos again. Last error: {last_exc}"
        )


def _collect_training_samples(
    config: VideoGenConfig,
    split: SplitName,
    action_encoder: GamepadActionEncoder,
) -> list[TrainingSample]:
    """Load NitroGen chunks and convert to training samples for the given split."""
    max_chunks = config.max_train_chunks if split is SplitName.TRAIN else config.max_val_chunks

    dataset = NitroGenDataset(
        data_dir=config.dataset_path,
        split=split,
        split_policy=config.split_policy,
        seed=config.split_seed,
        game_filter=config.game_filter,
        max_chunks=max_chunks,
        use_processed_actions=config.use_processed_actions,
        split_granularity=config.split_granularity,
    )

    samples: list[TrainingSample] = []
    skipped_invalid = 0
    for chunk in dataset:
        vp = Path(chunk.video_path) if chunk.video_path else None
        if vp is None or not vp.is_file():
            skipped_invalid += 1
            continue
        if not is_probably_valid_mp4(vp):
            logger.warning(
                "Skipping chunk (invalid or incomplete MP4 — re-run download with "
                "--download-videos): chunk=%s path=%s",
                chunk.chunk_id,
                vp,
            )
            skipped_invalid += 1
            continue
        sample = chunk_to_training_sample(
            chunk=chunk,
            target_resolution=config.resolution,
            max_frames=config.num_frames,
            prompt_template=config.prompt_template,
        )
        if sample is not None:
            samples.append(sample)

    if skipped_invalid:
        logger.info("Skipped %d chunks with missing or invalid video.mp4", skipped_invalid)

    return samples


def _build_pipeline_load_kwargs(config: VideoGenConfig) -> dict[str, Any]:
    """Kwargs for ``HunyuanVideoPipeline.from_pretrained`` (dtype + optional NF4)."""
    load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}

    if config.quantization != "nf4":
        return load_kwargs

    if not torch.cuda.is_available():
        logger.warning(
            "quantization is 'nf4' but no CUDA GPU is visible — bitsandbytes 4-bit needs a GPU. "
            "Loading without NF4 (full fp/bf16 weights; HunyuanVideo still needs very large VRAM or "
            "CPU RAM and may OOM). Use a GPU-enabled container (e.g. --gpus all) for real training."
        )
        return load_kwargs

    try:
        from diffusers.quantizers import PipelineQuantizationConfig

        load_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize="transformer",
        )
        return load_kwargs
    except ImportError:
        pass

    # Older diffusers: transformers BitsAndBytesConfig (may error on very new diffusers).
    try:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.warning(
            "Using transformers.BitsAndBytesConfig for nf4; prefer diffusers with "
            "PipelineQuantizationConfig for HunyuanVideo."
        )
    except ImportError:
        logger.warning("bitsandbytes/transformers not available; loading without nf4 quantization.")
        load_kwargs.pop("quantization_config", None)

    return load_kwargs


def _setup_model_and_tokenizer(config: VideoGenConfig) -> tuple[Any, Any, Any]:
    """Load HunyuanVideo pipeline components with optional quantization.

    Returns ``(pipeline, text_encoder, vae)`` — or stubs if the model is not
    locally available (allowing the training script structure to be validated
    without the full 8.3B checkpoint).
    """
    try:
        from diffusers import HunyuanVideoPipeline
    except ImportError:
        logger.warning(
            "diffusers.HunyuanVideoPipeline not available. "
            "Install diffusers>=0.30.0 for HunyuanVideo support."
        )
        return None, None, None

    load_kwargs = _build_pipeline_load_kwargs(config)

    try:
        pipe = HunyuanVideoPipeline.from_pretrained(config.model_id, **load_kwargs)
    except Exception as exc:
        logger.error("Failed to load HunyuanVideo pipeline: %s", exc)
        return None, None, None

    return pipe, getattr(pipe, "text_encoder", None), getattr(pipe, "vae", None)


def _apply_lora(model: Any, config: VideoGenConfig) -> Any:
    """Wrap the transformer/UNet with LoRA adapters via PEFT."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: %d / %d trainable parameters (%.2f%%)", trainable, total, 100 * trainable / total)
    return model


def _run_validation(
    model: Any,
    val_loader: Optional[DataLoader],
    device: torch.device,
) -> Optional[float]:
    """Compute mean diffusion loss on the validation set."""
    if val_loader is None or len(val_loader) == 0:
        return None

    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)
            noisy = pixel_values + noise * (timesteps.float() / 1000.0).view(-1, 1, 1, 1, 1)
            pred = model(noisy, timesteps)
            if hasattr(pred, "sample"):
                pred = pred.sample
            loss = torch.nn.functional.mse_loss(pred, noise)
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / max(count, 1)


def run_hunyuanvideo_lora_finetune(config: VideoGenConfig) -> dict[str, Any]:
    """Run the full HunyuanVideo LoRA fine-tuning pipeline.

    Returns a summary dict with training metrics.
    """
    from src.train.config_io import build_action_encoder

    action_encoder = build_action_encoder(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting training samples from %s ...", config.dataset_path)
    train_samples = _collect_training_samples(config, SplitName.TRAIN, action_encoder)
    if not train_samples:
        raise ValueError(
            "No training samples found. Check dataset_path, game_filter, and that "
            "video.mp4 files exist and are valid (incomplete downloads cause "
            "'moov atom not found'). Re-run: python -m scripts.download_data "
            "--output <dir> --shards ... --download-videos"
        )
    logger.info("Training samples: %d", len(train_samples))

    val_samples = _collect_training_samples(config, SplitName.VAL, action_encoder)
    logger.info("Validation samples: %d", len(val_samples))

    pipe, text_encoder, vae = _setup_model_and_tokenizer(config)
    if pipe is None:
        logger.warning(
            "Pipeline not available — writing sample manifest only. "
            "Full training requires the HunyuanVideo model checkpoint."
        )
        manifest = _write_sample_manifest(train_samples, val_samples, output_dir)
        return manifest

    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if transformer is None:
        raise RuntimeError("Could not locate transformer/unet in pipeline.")

    if config.training_type == "lora":
        transformer = _apply_lora(transformer, config)

    if config.gradient_checkpointing:
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()

    train_dataset = VideoActionDataset(samples=train_samples, action_encoder=action_encoder)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader] = None
    if val_samples:
        val_dataset = VideoActionDataset(samples=val_samples, action_encoder=action_encoder)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    num_epochs = max(1, math.ceil(config.num_train_steps / max(len(train_loader), 1)))
    global_step = 0
    best_loss = float("inf")
    step_log: list[dict[str, Any]] = []
    epoch_log: list[dict[str, Any]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)

    step_csv_path = output_dir / "train_log_steps.csv"
    epoch_csv_path = output_dir / "train_log_epochs.csv"

    with step_csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "epoch", "loss", "lr", "elapsed_sec"])
    with epoch_csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "best_loss",
            "steps_in_epoch", "lr", "epoch_duration_sec", "total_elapsed_sec",
        ])

    train_start = time.time()
    logger.info(
        "Starting training: %d steps, %d epochs, %d train samples, %d val samples",
        config.num_train_steps, num_epochs, len(train_samples), len(val_samples),
    )

    for epoch in range(num_epochs):
        transformer.train()
        epoch_start = time.time()
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= config.num_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device)

            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device)

            noisy = pixel_values + noise * (timesteps.float() / 1000.0).view(-1, 1, 1, 1, 1)
            pred = transformer(noisy, timesteps)
            if hasattr(pred, "sample"):
                pred = pred.sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            raw_loss = loss.item()

            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss_sum += raw_loss
            epoch_step_count += 1
            global_step += 1

            if global_step % config.logging_steps == 0:
                avg_loss = epoch_loss_sum / epoch_step_count
                elapsed = time.time() - train_start
                current_lr = optimizer.param_groups[0]["lr"]
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(avg_loss, 6),
                    "lr": current_lr,
                    "elapsed_sec": round(elapsed, 1),
                }
                step_log.append(entry)
                logger.info(
                    "Step %d | epoch %d | loss=%.6f | lr=%.2e | elapsed=%.0fs",
                    global_step, epoch, avg_loss, current_lr, elapsed,
                )
                with step_csv_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        global_step, epoch, round(avg_loss, 6), current_lr, round(elapsed, 1),
                    ])

            if global_step % config.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                _save_checkpoint(transformer, ckpt_dir, config)

        epoch_duration = time.time() - epoch_start
        avg_train_loss = epoch_loss_sum / max(epoch_step_count, 1)

        val_loss = _run_validation(transformer, val_loader, device) if val_loader else None

        is_best = avg_train_loss < best_loss
        if is_best:
            best_loss = avg_train_loss
            _save_checkpoint(transformer, output_dir / "best_model", config)

        current_lr = optimizer.param_groups[0]["lr"]
        total_elapsed = time.time() - train_start
        epoch_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "best_loss": round(best_loss, 6),
            "steps_in_epoch": epoch_step_count,
            "lr": current_lr,
            "epoch_duration_sec": round(epoch_duration, 1),
            "total_elapsed_sec": round(total_elapsed, 1),
            "is_best": is_best,
        }
        epoch_log.append(epoch_entry)

        val_str = f"val_loss={val_loss:.6f}" if val_loss is not None else "val_loss=N/A"
        logger.info(
            "Epoch %d/%d complete | train_loss=%.6f | %s | best=%.6f | %.0fs",
            epoch + 1, num_epochs, avg_train_loss, val_str, best_loss, epoch_duration,
        )

        with epoch_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, round(avg_train_loss, 6),
                round(val_loss, 6) if val_loss is not None else "",
                round(best_loss, 6), epoch_step_count, current_lr,
                round(epoch_duration, 1), round(total_elapsed, 1),
            ])

    _save_checkpoint(transformer, output_dir / "final_model", config)

    total_duration = time.time() - train_start
    metrics: dict[str, Any] = {
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "total_steps": global_step,
        "total_epochs": num_epochs,
        "final_train_loss": round(best_loss, 6),
        "final_val_loss": round(epoch_log[-1]["val_loss"], 6) if epoch_log and epoch_log[-1]["val_loss"] is not None else None,
        "best_train_loss": round(best_loss, 6),
        "total_duration_sec": round(total_duration, 1),
        "output_dir": str(output_dir.resolve()),
    }

    # Save all training history
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
        fp.write("\n")

    history_path = output_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as fp:
        json.dump({"step_log": step_log, "epoch_log": epoch_log}, fp, indent=2)
        fp.write("\n")

    config_path = output_dir / "finetune_config.json"
    with config_path.open("w", encoding="utf-8") as fp:
        json.dump(_config_to_dict(config), fp, indent=2)
        fp.write("\n")

    # Auto-generate training plots
    try:
        from scripts.plot_training import generate_plots
        plots_dir = generate_plots(output_dir)
        logger.info("Training plots saved to %s", plots_dir)
    except Exception as exc:
        logger.warning("Could not generate training plots: %s", exc)

    logger.info(
        "Training complete: %d steps, %d epochs, %.0f seconds. Artifacts in %s",
        global_step, num_epochs, total_duration, output_dir,
    )

    return metrics


def _save_checkpoint(model: Any, save_dir: Path, config: VideoGenConfig) -> None:
    """Save LoRA adapter weights or full model checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if config.training_type == "lora" and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_dir))
    else:
        torch.save(model.state_dict(), str(save_dir / "model.pt"))
    logger.info("Saved checkpoint to %s", save_dir)


def _write_sample_manifest(
    train_samples: list[TrainingSample],
    val_samples: list[TrainingSample],
    output_dir: Path,
) -> dict[str, Any]:
    """Write a JSON manifest of collected samples (fallback when model is unavailable)."""
    manifest: dict[str, Any] = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_chunks": [s.chunk_id for s in train_samples[:20]],
        "val_chunks": [s.chunk_id for s in val_samples[:20]],
        "output_dir": str(output_dir.resolve()),
        "status": "manifest_only",
    }
    with (output_dir / "sample_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")
    return manifest


def _config_to_dict(config: VideoGenConfig) -> dict[str, Any]:
    """Serialize a ``VideoGenConfig`` to a JSON-safe dict."""
    return {
        "dataset_path": config.dataset_path,
        "output_dir": config.output_dir,
        "model_id": config.model_id,
        "training_type": config.training_type,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "target_modules": list(config.lora.target_modules),
        "resolution": list(config.resolution),
        "num_frames": config.num_frames,
        "learning_rate": config.learning_rate,
        "num_train_steps": config.num_train_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "batch_size": config.batch_size,
        "gradient_checkpointing": config.gradient_checkpointing,
        "quantization": config.quantization,
        "mixed_precision": config.mixed_precision,
        "action_encoding": config.action_encoding,
        "prompt_template": config.prompt_template,
        "joystick_deadzone": config.joystick_deadzone,
        "split_seed": config.split_seed,
        "split_policy": {
            "train": config.split_policy.train,
            "val": config.split_policy.val,
            "test": config.split_policy.test,
        },
        "split_granularity": config.split_granularity,
        "game_filter": config.game_filter,
        "max_train_chunks": config.max_train_chunks,
        "max_val_chunks": config.max_val_chunks,
        "use_processed_actions": config.use_processed_actions,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "seed": config.seed,
    }


def _default_hunyuan_model_id() -> str:
    """HF repo id with Diffusers layout (``model_index.json`` at repo root).

    ``tencent/HunyuanVideo`` is the upstream release; weights live under subfolders
    and are not a Diffusers single-folder checkpoint. Use the community mirror.
    """
    return "hunyuanvideo-community/HunyuanVideo"


def generate_video(
    model_dir: Union[str, Path],
    prompt: str,
    num_frames: int = 49,
    resolution: tuple[int, int] = (480, 720),
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Generate a video from a text prompt using a fine-tuned HunyuanVideo model.

    Returns frames as ``(T, H, W, 3)`` uint8 array, or ``None`` if generation fails.
    """
    try:
        from diffusers import HunyuanVideoPipeline
    except ImportError:
        logger.error("diffusers not available for video generation.")
        return None

    model_path = Path(model_dir)
    lora_dir = model_path / "best_model"
    if not lora_dir.exists():
        lora_dir = model_path / "final_model"

    config_path = model_path / "finetune_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fp:
            raw_config = json.load(fp)
        base_model_id = raw_config.get("model_id", _default_hunyuan_model_id())
    else:
        base_model_id = _default_hunyuan_model_id()

    try:
        pipe = HunyuanVideoPipeline.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
        if lora_dir.exists():
            pipe.load_lora_weights(str(lora_dir))
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:
        logger.error("Failed to load pipeline for generation: %s", exc)
        return None

    generator = torch.Generator().manual_seed(seed)
    h, w = resolution
    output = pipe(
        prompt=prompt,
        height=h,
        width=w,
        num_frames=num_frames,
        generator=generator,
    )

    if hasattr(output, "frames") and output.frames is not None:
        frames = output.frames[0]
        if isinstance(frames, list):
            frames_np = np.stack([np.array(f) for f in frames])
        else:
            frames_np = np.array(frames)
        return frames_np

    return None
