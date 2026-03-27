from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.loader import ManifestDataset
from src.data.schema import SplitName


def _write_episode_json_actions(episode_dir: Path, frames: list[str], actions: list[str]) -> None:
    rows = [{"frame": f, "action": a} for f, a in zip(frames, actions, strict=True)]
    with (episode_dir / "actions.json").open("w", encoding="utf-8") as fp:
        json.dump(rows, fp)
        fp.write("\n")


def _touch_frames(frames_dir: Path, frame_names: list[str]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        (frames_dir / name).write_bytes(b"")  # fake image payload


def test_build_and_load_end_to_end(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw"
    ep_a = raw_root / "ep_a"
    ep_b = raw_root / "ep_b"

    # Episode A: 10 frames
    frame_names_a = [f"{i:04d}.png" for i in range(10)]
    actions_a = ["left"] * 10
    _touch_frames(ep_a / "frames", frame_names_a)
    _write_episode_json_actions(ep_a, frame_names_a, actions_a)

    # Episode B: 10 frames
    frame_names_b = [f"{i:04d}.png" for i in range(10)]
    actions_b = ["right"] * 10
    _touch_frames(ep_b / "frames", frame_names_b)
    _write_episode_json_actions(ep_b, frame_names_b, actions_b)

    manifest_path = tmp_path / "data" / "processed" / "manifest.json"

    # Run builder as a module-style import (fast, no subprocess).
    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    cfg = BuildDatasetConfig(
        input_root=str(raw_root),
        output_manifest_path=str(manifest_path),
        seed=123,
        split_policy=SplitPolicy(train=0.5, val=0.25, test=0.25),
        clip_length=4,
        stride=2,
    )
    manifest = build_manifest(cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    ds = ManifestDataset(manifest_path)
    assert len(ds) == 2 * (1 + (10 - 4) // 2)  # per-episode sliding windows

    # Split filtering is clip-level but assigned per-episode.
    train_ds = ManifestDataset(manifest_path, split=SplitName.TRAIN)
    for sample in train_ds:
        assert sample.split is SplitName.TRAIN


def test_loader_rewindowing(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    ep = raw_root / "ep_one"
    frame_names = [f"{i:04d}.png" for i in range(8)]
    _touch_frames(ep / "frames", frame_names)
    _write_episode_json_actions(ep, frame_names, ["a"] * 8)

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    manifest = build_manifest(
        BuildDatasetConfig(
            input_root=str(raw_root),
            output_manifest_path=str(tmp_path / "out.json"),
            seed=0,
            split_policy=SplitPolicy(train=1.0 - 1e-7, val=1e-7, test=1e-7),
            clip_length=8,
            stride=8,
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    ds = ManifestDataset(manifest_path, clip_length=4, stride=2)
    assert len(ds) == 1 + (8 - 4) // 2
    assert len(ds[0].frame_paths) == 4

