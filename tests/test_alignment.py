from __future__ import annotations

import json
from pathlib import Path

import pytest


def _touch_frames(frames_dir: Path, frame_names: list[str]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        (frames_dir / name).write_bytes(b"")


def test_builder_raises_on_missing_action(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    ep = raw_root / "ep_missing"
    frame_names = ["0000.png", "0001.png"]
    _touch_frames(ep / "frames", frame_names)

    # Missing annotation for 0001.png
    rows = [{"frame": "0000.png", "action": "jump"}]
    (ep / "actions.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    with pytest.raises(ValueError, match="missing action"):
        build_manifest(
            BuildDatasetConfig(
                input_root=str(raw_root),
                output_manifest_path=str(tmp_path / "out.json"),
                seed=0,
                split_policy=SplitPolicy(train=0.8, val=0.1, test=0.1),
                clip_length=2,
                stride=2,
            )
        )


def test_builder_raises_on_both_csv_and_json(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    ep = raw_root / "ep_both"
    _touch_frames(ep / "frames", ["0000.png"])
    (ep / "actions.json").write_text(json.dumps([{"frame": "0000.png", "action": "a"}]) + "\n", encoding="utf-8")
    (ep / "actions.csv").write_text("frame,action\n0000.png,a\n", encoding="utf-8")

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    with pytest.raises(ValueError, match="both actions.json and actions.csv"):
        build_manifest(
            BuildDatasetConfig(
                input_root=str(raw_root),
                output_manifest_path=str(tmp_path / "out.json"),
                seed=0,
                split_policy=SplitPolicy(train=0.8, val=0.1, test=0.1),
                clip_length=1,
                stride=1,
            )
        )

