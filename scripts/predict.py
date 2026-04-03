"""Generate videos from action sequences using a fine-tuned HunyuanVideo model.

Usage (from repo root)::

    python3.12 -m scripts.predict \\
        --model-dir outputs/run1 \\
        --dataset-path data/nitrogen \\
        --split val \\
        --output outputs/run1/generated_videos \\
        --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.loader import NitroGenDataset, chunk_to_training_sample
from src.data.schema import SplitName, SplitPolicy
from src.model.action_encoder import GamepadActionEncoder
from src.train.config_io import load_json_object, load_videogen_config
from src.train.hf_finetune import generate_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _save_video_frames(frames: np.ndarray, output_dir: Path, chunk_id: str) -> Path:
    """Save generated video frames as individual PNGs under ``output_dir/chunk_id/``."""
    from PIL import Image

    frame_dir = output_dir / chunk_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(str(frame_dir / f"frame_{i:04d}.png"))

    return frame_dir


def _try_save_mp4(frames: np.ndarray, output_path: Path, fps: int = 24) -> bool:
    """Attempt to save frames as an mp4 video. Returns success flag."""
    try:
        import cv2

        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()
        return True
    except ImportError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos from action sequences using fine-tuned HunyuanVideo.",
    )
    parser.add_argument("--model-dir", required=True, type=Path, help="Fine-tuned model directory.")
    parser.add_argument("--dataset-path", required=True, type=Path, help="NitroGen dataset directory.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for generated videos.")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--config", type=Path, default=None, help="Optional finetune config for encoder settings.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config or (args.model_dir / "finetune_config.json")
    if config_path.exists():
        config = load_videogen_config(config_path)
        resolution = config.resolution
        num_frames = config.num_frames
        prompt_template = config.prompt_template
        split_policy = config.split_policy
        split_seed = config.split_seed
    else:
        resolution = (480, 720)
        num_frames = 49
        prompt_template = "Gameplay video of {game}."
        split_policy = SplitPolicy(train=0.8, val=0.1, test=0.1)
        split_seed = 42

    encoder = GamepadActionEncoder()
    split = SplitName(args.split)

    dataset = NitroGenDataset(
        data_dir=args.dataset_path,
        split=split,
        split_policy=split_policy,
        seed=split_seed,
        max_chunks=args.max_samples,
    )

    results: list[dict[str, str]] = []
    for idx, chunk in enumerate(dataset):
        sample = chunk_to_training_sample(
            chunk=chunk,
            target_resolution=resolution,
            max_frames=num_frames,
            prompt_template=prompt_template,
        )
        if sample is None:
            continue

        prompt = encoder.encode_conditioning_prompt(
            actions=sample.actions,
            game_name=chunk.game,
            base_prompt=sample.prompt,
        )

        logger.info("Generating video %d/%d for chunk %s ...", idx + 1, len(dataset), chunk.chunk_id)

        frames = generate_video(
            model_dir=args.model_dir,
            prompt=prompt,
            num_frames=sample.num_frames,
            resolution=resolution,
            seed=args.seed,
        )

        if frames is not None:
            frame_dir = _save_video_frames(frames, output_dir, chunk.chunk_id)
            mp4_path = output_dir / f"{chunk.chunk_id}.mp4"
            _try_save_mp4(frames, mp4_path)
            results.append({
                "chunk_id": chunk.chunk_id,
                "game": chunk.game,
                "prompt": prompt,
                "frames_dir": str(frame_dir),
                "num_frames": len(frames),
            })
        else:
            logger.warning("Generation returned None for chunk %s", chunk.chunk_id)
            results.append({
                "chunk_id": chunk.chunk_id,
                "game": chunk.game,
                "prompt": prompt,
                "frames_dir": "",
                "num_frames": 0,
            })

    manifest_path = output_dir / "generation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
        fp.write("\n")

    logger.info("Generated %d videos, manifest at %s", len(results), manifest_path)
    print(json.dumps({"total": len(results), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
