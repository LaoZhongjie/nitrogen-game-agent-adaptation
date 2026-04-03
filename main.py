"""Run the full pipeline: download data -> fine-tune -> generate -> evaluate.

Edit the ``PipelinePaths`` values below, then from the repo root:

    python3.12 main.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelinePaths:
    """Filesystem locations for one end-to-end run."""

    nitrogen_data_dir: Path
    manifest_path: Path
    run_output_dir: Path
    finetune_config_template: Path


PATHS = PipelinePaths(
    nitrogen_data_dir=Path("data/nitrogen"),
    manifest_path=Path("data/processed/manifest.json"),
    run_output_dir=Path("outputs/pipeline_run"),
    finetune_config_template=Path("configs/finetune.example.json"),
)

EVAL_SPLIT: str = "val"
DATASET_SEED: int = 42
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

DOWNLOAD_SHARDS: list[int] = [0]
DOWNLOAD_VIDEOS: bool = True
MAX_CHUNKS_PER_SHARD: Optional[int] = 50

GENERATE_MAX_SAMPLES: int = 5

# When downloading few chunks from few videos, video-level splits can put all data in
# one split (e.g. train=0). Use "chunk" so train/val/test are spread across chunks.
DATASET_SPLIT_GRANULARITY: str = "chunk"

# If True, skip stages that are already satisfied on disk (faster re-runs).
# Set False to force re-download and/or rebuild manifest (e.g. after changing shards/seed).
SKIP_DOWNLOAD_IF_PRESENT: bool = True
SKIP_MANIFEST_IF_PRESENT: bool = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _nitrogen_has_chunk_data(data_dir: Path) -> bool:
    """Return True if ``data_dir`` already contains at least one NitroGen chunk (metadata.json)."""
    if not data_dir.is_dir():
        return False
    for shard in sorted(data_dir.glob("SHARD_*")):
        if not shard.is_dir():
            continue
        for video_dir in shard.iterdir():
            if not video_dir.is_dir():
                continue
            for chunk_dir in video_dir.iterdir():
                if chunk_dir.is_dir() and (chunk_dir / "metadata.json").is_file():
                    return True
    return False


def run_pipeline() -> None:
    """Execute download -> build manifest -> fine-tune -> generate -> evaluate."""
    root = _repo_root()
    sys.path.insert(0, str(root))

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from scripts.download_data import download_nitrogen
    from src.data.schema import SplitName, SplitPolicy
    from src.eval.pipeline import build_evaluation_records
    from src.eval.report import build_evaluation_report, write_evaluation_report
    from src.train.config_io import load_json_object, load_videogen_config
    from src.train.hf_finetune import run_hunyuanvideo_lora_finetune

    paths = PipelinePaths(
        nitrogen_data_dir=(root / PATHS.nitrogen_data_dir).resolve(),
        manifest_path=(root / PATHS.manifest_path).resolve(),
        run_output_dir=(root / PATHS.run_output_dir).resolve(),
        finetune_config_template=(root / PATHS.finetune_config_template).resolve(),
    )

    if not paths.finetune_config_template.is_file():
        raise FileNotFoundError(f"missing finetune template: {paths.finetune_config_template}")

    # --- Stage 1: Download NitroGen data ---
    paths.nitrogen_data_dir.mkdir(parents=True, exist_ok=True)
    has_chunks = _nitrogen_has_chunk_data(paths.nitrogen_data_dir)
    skip_dl = SKIP_DOWNLOAD_IF_PRESENT and has_chunks
    if skip_dl:
        logger.info(
            "[1/5] Skipping download — NitroGen chunk data already under %s "
            "(set SKIP_DOWNLOAD_IF_PRESENT=False to force).",
            paths.nitrogen_data_dir,
        )
        download_stats = {"skipped": True, "reason": "existing_chunk_data"}
    else:
        shard_dirs = [p for p in paths.nitrogen_data_dir.glob("SHARD_*") if p.is_dir()]
        logger.info(
            "[1/5] Downloading NitroGen data (shards=%s)... "
            "SKIP_DOWNLOAD_IF_PRESENT=%s, existing_SHARD_dirs=%d, has_chunk_data=%s",
            DOWNLOAD_SHARDS,
            SKIP_DOWNLOAD_IF_PRESENT,
            len(shard_dirs),
            has_chunks,
        )
        download_stats = download_nitrogen(
            output_dir=paths.nitrogen_data_dir,
            shard_indices=DOWNLOAD_SHARDS,
            download_videos=DOWNLOAD_VIDEOS,
            max_chunks_per_shard=MAX_CHUNKS_PER_SHARD,
        )
        try:
            marker = paths.nitrogen_data_dir / ".nitrogen_prepared"
            marker.write_text(
                json.dumps(
                    {
                        "shards": DOWNLOAD_SHARDS,
                        "max_chunks_per_shard": MAX_CHUNKS_PER_SHARD,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
    logger.info("Download stats: %s", json.dumps(download_stats, indent=2))

    # --- Stage 2: Build manifest ---
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    skip_mf = SKIP_MANIFEST_IF_PRESENT and paths.manifest_path.is_file()
    if skip_mf:
        logger.info(
            "[2/5] Skipping manifest build — using existing %s "
            "(set SKIP_MANIFEST_IF_PRESENT=False to rebuild).",
            paths.manifest_path,
        )
        with paths.manifest_path.open("r", encoding="utf-8") as fp:
            manifest_obj = json.load(fp)
        if not isinstance(manifest_obj, dict) or "total_chunks" not in manifest_obj:
            raise ValueError(
                f"invalid manifest at {paths.manifest_path}; delete it or set SKIP_MANIFEST_IF_PRESENT=False"
            )
    else:
        logger.info("[2/5] Building dataset manifest...")
        ds_cfg = BuildDatasetConfig(
            input_root=str(paths.nitrogen_data_dir),
            output_manifest_path=str(paths.manifest_path),
            seed=DATASET_SEED,
            split_policy=SplitPolicy(train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO),
            split_granularity=DATASET_SPLIT_GRANULARITY,
        )
        manifest_obj = build_manifest(ds_cfg)
        with paths.manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest_obj, fp, indent=2)
            fp.write("\n")
    logger.info("Manifest: %d chunks, splits=%s", manifest_obj["total_chunks"], manifest_obj["split_counts"])

    # --- Stage 3: Fine-tune HunyuanVideo with LoRA ---
    logger.info("[3/5] Fine-tuning HunyuanVideo...")
    paths.run_output_dir.mkdir(parents=True, exist_ok=True)

    raw_ft = dict(load_json_object(paths.finetune_config_template))
    raw_ft["dataset_path"] = str(paths.nitrogen_data_dir)
    raw_ft["output_dir"] = str(paths.run_output_dir)
    raw_ft["split_seed"] = DATASET_SEED
    raw_ft["split_granularity"] = DATASET_SPLIT_GRANULARITY

    ft_cfg_path = paths.run_output_dir / "finetune_config.json"
    with ft_cfg_path.open("w", encoding="utf-8") as fp:
        json.dump(raw_ft, fp, indent=2)
        fp.write("\n")

    config = load_videogen_config(ft_cfg_path)
    train_summary = run_hunyuanvideo_lora_finetune(config)
    logger.info("Training summary: %s", json.dumps(train_summary, indent=2))

    # --- Stage 4: Generate videos ---
    logger.info("[4/5] Generating videos (split=%s, max=%d)...", EVAL_SPLIT, GENERATE_MAX_SAMPLES)

    from src.data.loader import NitroGenDataset, chunk_to_training_sample
    from src.model.action_encoder import GamepadActionEncoder
    from src.train.hf_finetune import generate_video

    split = SplitName(EVAL_SPLIT)
    encoder = GamepadActionEncoder()

    eval_dataset = NitroGenDataset(
        data_dir=paths.nitrogen_data_dir,
        split=split,
        split_policy=config.split_policy,
        seed=config.split_seed,
        max_chunks=GENERATE_MAX_SAMPLES,
        split_granularity=config.split_granularity,
    )

    gen_output_dir = paths.run_output_dir / "generated_videos"
    gen_output_dir.mkdir(parents=True, exist_ok=True)

    gen_results: list[dict] = []
    for chunk in eval_dataset:
        sample = chunk_to_training_sample(
            chunk=chunk,
            target_resolution=config.resolution,
            max_frames=config.num_frames,
            prompt_template=config.prompt_template,
        )
        if sample is None:
            continue

        prompt = encoder.encode_conditioning_prompt(
            actions=sample.actions,
            game_name=chunk.game,
            base_prompt=sample.prompt,
        )

        frames = generate_video(
            model_dir=paths.run_output_dir,
            prompt=prompt,
            num_frames=sample.num_frames,
            resolution=config.resolution,
            seed=config.seed,
        )

        gen_entry: dict = {
            "chunk_id": chunk.chunk_id,
            "game": chunk.game,
            "prompt": prompt,
            "frames_dir": "",
            "num_frames": 0,
        }

        if frames is not None:
            import numpy as np
            from PIL import Image

            frame_dir = gen_output_dir / chunk.chunk_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame.astype(np.uint8)).save(str(frame_dir / f"frame_{i:04d}.png"))
            gen_entry["frames_dir"] = str(frame_dir)
            gen_entry["num_frames"] = len(frames)

        gen_results.append(gen_entry)

    gen_manifest_path = paths.run_output_dir / "generation_manifest.json"
    with gen_manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(gen_results, fp, indent=2)
        fp.write("\n")
    logger.info("Generated %d videos", len(gen_results))

    # --- Stage 5: Evaluate ---
    logger.info("[5/5] Evaluation report...")
    records = build_evaluation_records(
        generation_manifest_path=gen_manifest_path,
        reference_dataset_path=paths.nitrogen_data_dir,
    )

    if records:
        report_path = paths.run_output_dir / f"report_{EVAL_SPLIT}.json"
        report = build_evaluation_report(
            records=records,
            generation_manifest_path=str(gen_manifest_path),
            output_report_path=str(report_path),
            split=split,
        )
        write_evaluation_report(report)
        logger.info(
            "Report: %s | videos=%d | temporal_consistency=%s | lpips=%s",
            report_path,
            report.summary.total_videos,
            report.summary.mean_temporal_consistency,
            report.summary.mean_lpips,
        )
    else:
        logger.warning("No evaluation records generated (no valid generated videos).")


if __name__ == "__main__":
    run_pipeline()
