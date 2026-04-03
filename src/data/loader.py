"""Dataset loader for NitroGen video chunks (parquet + video files)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.schema import (
    BUTTON_COLUMNS,
    JOYSTICK_COLUMNS,
    ChunkMetadata,
    GamepadAction,
    SplitName,
    SplitPolicy,
    TrainingSample,
    VideoChunk,
)
from src.data.split import VideoSplitAssigner


def _parse_metadata(meta_path: Path) -> ChunkMetadata:
    """Parse a NitroGen ``metadata.json`` into a typed ``ChunkMetadata``."""
    with meta_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    orig = raw.get("original_video", {})
    resolution = tuple(orig.get("resolution", [0, 0]))
    bbox_game = raw.get("bbox_game_area")
    bbox_ctrl_raw = raw.get("bbox_controller_overlay")
    bbox_ctrl = tuple(bbox_ctrl_raw) if bbox_ctrl_raw else None

    return ChunkMetadata(
        uuid=str(raw.get("uuid", "")),
        chunk_id=str(raw.get("chunk_id", "")),
        chunk_size=int(raw.get("chunk_size", 0)),
        video_id=str(orig.get("video_id", "")),
        game=str(raw.get("game", "unknown")),
        controller_type=str(raw.get("controller_type", "")),
        resolution=(int(resolution[0]), int(resolution[1])),
        source=str(orig.get("source", "")),
        url=str(orig.get("url", "")),
        start_time=float(orig.get("start_time", 0.0)),
        end_time=float(orig.get("end_time", 0.0)),
        duration=float(orig.get("duration", 0.0)),
        start_frame=int(orig.get("start_frame", 0)),
        end_frame=int(orig.get("end_frame", 0)),
        bbox_game_area=bbox_game,
        bbox_controller_overlay=bbox_ctrl,
    )


def _parse_actions_parquet(parquet_path: Path) -> tuple[GamepadAction, ...]:
    """Load per-frame gamepad actions from a NitroGen parquet file."""
    df = pd.read_parquet(parquet_path)
    actions: list[GamepadAction] = []

    for _, row in df.iterrows():
        buttons = tuple(bool(row.get(col, False)) for col in BUTTON_COLUMNS)

        j_left = row.get("j_left", (0.0, 0.0))
        j_right = row.get("j_right", (0.0, 0.0))
        if isinstance(j_left, (list, tuple, np.ndarray)):
            jl_x, jl_y = float(j_left[0]), float(j_left[1])
        else:
            jl_x, jl_y = 0.0, 0.0
        if isinstance(j_right, (list, tuple, np.ndarray)):
            jr_x, jr_y = float(j_right[0]), float(j_right[1])
        else:
            jr_x, jr_y = 0.0, 0.0

        jl_x = max(-1.0, min(1.0, jl_x))
        jl_y = max(-1.0, min(1.0, jl_y))
        jr_x = max(-1.0, min(1.0, jr_x))
        jr_y = max(-1.0, min(1.0, jr_y))

        actions.append(GamepadAction(buttons=buttons, joysticks=(jl_x, jl_y, jr_x, jr_y)))

    return tuple(actions)


def discover_chunks(data_dir: Union[str, Path]) -> list[Path]:
    """Walk the NitroGen data directory and return paths to all chunk directories.

    Expects ``data_dir/SHARD_XXXX/<video_id>/<chunk_id>/`` layout.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")

    chunk_dirs: list[Path] = []
    for shard_dir in sorted(root.iterdir()):
        if not shard_dir.is_dir() or not shard_dir.name.startswith("SHARD_"):
            continue
        for video_dir in sorted(shard_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for chunk_dir in sorted(video_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                if (chunk_dir / "metadata.json").exists():
                    chunk_dirs.append(chunk_dir)

    return chunk_dirs


def load_video_chunk(
    chunk_dir: Path,
    split: SplitName,
    use_processed_actions: bool = True,
) -> Optional[VideoChunk]:
    """Load a single video chunk from its directory. Returns ``None`` if missing data."""
    meta_path = chunk_dir / "metadata.json"
    if not meta_path.exists():
        return None

    metadata = _parse_metadata(meta_path)

    parquet_name = "actions_processed.parquet" if use_processed_actions else "actions_raw.parquet"
    parquet_path = chunk_dir / parquet_name
    if not parquet_path.exists():
        parquet_path = chunk_dir / "actions_raw.parquet"
    if not parquet_path.exists():
        return None

    actions = _parse_actions_parquet(parquet_path)
    if len(actions) == 0:
        return None

    video_path = chunk_dir / "video.mp4"
    if not video_path.is_file():
        return None

    video_path_str = str(video_path)

    shard_name = chunk_dir.parent.parent.name if chunk_dir.parent.parent else "unknown"

    return VideoChunk(
        chunk_id=chunk_dir.name,
        video_id=metadata.video_id or chunk_dir.parent.name,
        shard_id=shard_name,
        game=metadata.game,
        video_path=video_path_str,
        actions=actions,
        metadata=metadata,
        split=split,
    )


class NitroGenDataset(Sequence[VideoChunk]):
    """Lazy-indexed dataset over NitroGen video chunks with deterministic splits."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: Optional[SplitName] = None,
        split_policy: Optional[SplitPolicy] = None,
        seed: int = 42,
        game_filter: Optional[str] = None,
        max_chunks: Optional[int] = None,
        use_processed_actions: bool = True,
        split_granularity: str = "video",
    ) -> None:
        if split_granularity not in ("video", "chunk"):
            raise ValueError("split_granularity must be 'video' or 'chunk'.")

        self._data_dir = Path(data_dir)
        self._use_processed = use_processed_actions

        policy = split_policy or SplitPolicy(train=0.8, val=0.1, test=0.1)
        assigner = VideoSplitAssigner(policy=policy, seed=seed, granularity=split_granularity)

        chunk_dirs = discover_chunks(data_dir)
        chunks: list[VideoChunk] = []

        for chunk_dir in chunk_dirs:
            video_id = chunk_dir.parent.name
            chunk_id = chunk_dir.name
            if split_granularity == "chunk":
                assigned_split = assigner.assign(video_id, chunk_id)
            else:
                assigned_split = assigner.assign(video_id)

            if split is not None and assigned_split is not split:
                continue

            chunk = load_video_chunk(
                chunk_dir, split=assigned_split, use_processed_actions=use_processed_actions
            )
            if chunk is None:
                continue

            if game_filter and chunk.game.lower() != game_filter.lower():
                continue

            chunks.append(chunk)
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        self._chunks = tuple(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> VideoChunk:
        return self._chunks[index]

    def __iter__(self) -> Iterator[VideoChunk]:
        return iter(self._chunks)


def nearest_valid_frame_count(n: int) -> int:
    """Round ``n`` down to the nearest frame count satisfying HunyuanVideo's 4k or 4k+1 rule."""
    if n <= 1:
        return 1
    candidate_4k = (n // 4) * 4
    candidate_4k1 = ((n - 1) // 4) * 4 + 1
    for c in sorted([candidate_4k, candidate_4k1], reverse=True):
        if c <= n and c > 0:
            return c
    return 1


def chunk_to_training_sample(
    chunk: VideoChunk,
    target_resolution: tuple[int, int] = (480, 720),
    max_frames: int = 49,
    prompt_template: str = "Gameplay video of {game}.",
) -> TrainingSample | None:
    """Convert a ``VideoChunk`` to a ``TrainingSample`` for fine-tuning.

    Returns ``None`` if the chunk lacks a video file or has too few frames.
    """
    if not chunk.video_path:
        return None

    num_frames = min(len(chunk.actions), max_frames)
    num_frames = nearest_valid_frame_count(num_frames)
    if num_frames < 4:
        return None

    actions = chunk.actions[:num_frames]
    prompt = prompt_template.format(game=chunk.game)

    return TrainingSample(
        chunk_id=chunk.chunk_id,
        video_path=chunk.video_path,
        actions=actions,
        prompt=prompt,
        num_frames=num_frames,
        resolution=target_resolution,
        split=chunk.split,
    )
