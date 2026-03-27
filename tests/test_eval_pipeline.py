from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_dataset import BuildDatasetConfig, build_manifest
from src.data.schema import SplitName, SplitPolicy
from src.eval.pipeline import (
    PredictionRecord,
    build_evaluation_records_from_manifest_predictions,
    load_prediction_records,
)


def _touch_frames(frames_dir: Path, frame_names: list[str]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        (frames_dir / name).write_bytes(b"")


def _write_actions_json(episode_dir: Path, frames: list[str], actions: list[str]) -> None:
    rows = [{"frame": f, "action": a} for f, a in zip(frames, actions, strict=True)]
    (episode_dir / "actions.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")


def _build_manifest(path: Path) -> Path:
    raw_root = path / "raw"
    episode_dir = raw_root / "ep_001"
    frame_names = [f"{i:04d}.png" for i in range(3)]
    _touch_frames(episode_dir / "frames", frame_names)
    _write_actions_json(episode_dir, frame_names, ["left", "jump", "left"])
    manifest = build_manifest(
        BuildDatasetConfig(
            input_root=str(raw_root),
            output_manifest_path=str(path / "manifest.json"),
            seed=0,
            split_policy=SplitPolicy(train=0.9999998, val=1e-7, test=1e-7),
            clip_length=3,
            stride=3,
        )
    )
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_load_prediction_records_reads_valid_payload(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"clip_id": "clip_1", "predicted_action_ids": ["left", "jump"]}], indent=2) + "\n",
        encoding="utf-8",
    )

    records = load_prediction_records(predictions_path)

    assert records == (PredictionRecord(clip_id="clip_1", predicted_action_ids=("left", "jump")),)


def test_build_evaluation_records_joins_manifest_with_predictions(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    predictions = (
        PredictionRecord(
            clip_id="ep_001_clip_000000",
            predicted_action_ids=("left", "slide", "left"),
        ),
    )

    records = build_evaluation_records_from_manifest_predictions(
        manifest_path=manifest_path,
        predictions=predictions,
        split=SplitName.TRAIN,
    )

    assert len(records) == 1
    assert records[0].clip_id == "ep_001_clip_000000"
    assert records[0].target_action_ids == ("left", "jump", "left")
    assert records[0].predicted_action_ids == ("left", "slide", "left")
    assert records[0].split is SplitName.TRAIN


def test_build_evaluation_records_rejects_length_mismatch(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    predictions = (
        PredictionRecord(
            clip_id="ep_001_clip_000000",
            predicted_action_ids=("left", "slide"),
        ),
    )

    with pytest.raises(ValueError, match="length mismatch"):
        build_evaluation_records_from_manifest_predictions(
            manifest_path=manifest_path,
            predictions=predictions,
            split=SplitName.TRAIN,
        )

