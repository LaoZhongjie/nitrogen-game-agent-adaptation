"""Tests for NitroGen loader when video.mp4 is missing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_video_chunk
from src.data.schema import SplitName


def test_load_video_chunk_requires_mp4(tmp_path: Path) -> None:
    chunk = tmp_path / "SHARD_0000" / "v1" / "c1"
    chunk.mkdir(parents=True)
    meta = {
        "uuid": "u1",
        "chunk_id": "c1",
        "chunk_size": 2,
        "game": "g",
        "start_frame": 0,
        "end_frame": 1,
        "original_video": {
            "video_id": "v1",
            "resolution": [480, 720],
            "url": "",
            "source": "",
            "start_time": 0.0,
            "end_time": 1.0,
            "duration": 1.0,
        },
    }
    (chunk / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    pd.DataFrame({"frame": [0, 1]}).to_parquet(chunk / "actions_raw.parquet")

    assert load_video_chunk(chunk, split=SplitName.TRAIN, use_processed_actions=False) is None
