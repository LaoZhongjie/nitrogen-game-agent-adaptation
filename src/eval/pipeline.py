"""Evaluation pipeline: load generated videos and compute quality metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from src.eval.metrics import VideoEvalRecord, evaluate_video_pair

logger = logging.getLogger(__name__)


def load_generation_manifest(path: Union[str, Path]) -> List[dict]:
    """Load a generation manifest (output of ``scripts.predict``)."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"generation manifest not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, list):
        raise ValueError("generation manifest must be a JSON array.")
    return raw


def _load_frames_from_dir(frame_dir: Union[str, Path]) -> Optional[np.ndarray]:
    """Load PNG frames from a directory into a ``(T, H, W, 3)`` uint8 array."""
    from PIL import Image

    path = Path(frame_dir)
    if not path.is_dir():
        return None

    frame_files = sorted(path.glob("frame_*.png"))
    if not frame_files:
        return None

    frames = [np.array(Image.open(f).convert("RGB")) for f in frame_files]
    return np.stack(frames)


def _load_reference_frames(
    video_path: str,
    num_frames: int,
    resolution: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Load reference frames from a source video file."""
    if not video_path or not Path(video_path).exists():
        return None

    import cv2

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()
    except ImportError:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning("Could not open reference video %s", video_path)
                return None
            rgb: list[np.ndarray] = []
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            if not rgb:
                return None
            total = len(rgb)
            idxs = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
            frames = np.stack([rgb[i] for i in idxs])
        except Exception as exc:
            logger.warning("Could not load reference video %s (opencv fallback): %s", video_path, exc)
            return None
    except Exception as exc:
        logger.warning("Could not load reference video %s: %s", video_path, exc)
        return None

    if resolution:
        h, w = resolution
        frames = np.stack([
            cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames
        ])

    return frames


def build_evaluation_records(
    generation_manifest_path: Union[str, Path],
    reference_dataset_path: Optional[Union[str, Path]] = None,
) -> List[VideoEvalRecord]:
    """Build evaluation records by comparing generated videos against references.

    If ``reference_dataset_path`` is provided, attempts to load original videos
    from the NitroGen data directory for per-frame comparison. Otherwise, only
    temporal consistency is computed.
    """
    manifest = load_generation_manifest(generation_manifest_path)
    records: list[VideoEvalRecord] = []

    for entry in manifest:
        chunk_id = str(entry.get("chunk_id", ""))
        game = str(entry.get("game", ""))
        frames_dir = str(entry.get("frames_dir", ""))
        num_frames = int(entry.get("num_frames", 0))

        if not frames_dir or num_frames == 0:
            continue

        gen_frames = _load_frames_from_dir(frames_dir)
        if gen_frames is None:
            continue

        ref_frames = None
        if reference_dataset_path:
            video_path = entry.get("source_video_path", "")
            if video_path:
                ref_frames = _load_reference_frames(
                    video_path, num_frames=len(gen_frames)
                )

        record = evaluate_video_pair(
            chunk_id=chunk_id,
            game=game,
            generated_frames=gen_frames,
            reference_frames=ref_frames,
        )
        records.append(record)

    return records
