from __future__ import annotations

import json
from pathlib import Path

from scripts.finetune import FineTuneConfig, run_finetune
from src.data.schema import SplitName


def _touch_frames(frames_dir: Path, frame_names: list[str]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        (frames_dir / name).write_bytes(b"")


def _write_actions_json(episode_dir: Path, frames: list[str], actions: list[str]) -> None:
    rows = [{"frame": f, "action": a} for f, a in zip(frames, actions, strict=True)]
    (episode_dir / "actions.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")


def test_run_dry_run_reports_counts_and_checkpoint_dir(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    episode_dir = raw_root / "ep_001"
    frame_names = [f"{i:04d}.png" for i in range(4)]
    _touch_frames(episode_dir / "frames", frame_names)
    _write_actions_json(episode_dir, frame_names, ["left", "left", "jump", "slide"])

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    manifest = build_manifest(
        BuildDatasetConfig(
            input_root=str(raw_root),
            output_manifest_path=str(tmp_path / "manifest.json"),
            seed=0,
            split_policy=SplitPolicy(train=0.9999998, val=1e-7, test=1e-7),
            clip_length=4,
            stride=4,
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    cfg = FineTuneConfig(
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "outputs"),
        split=SplitName.TRAIN,
        action_mapping={"left": "move_left", "jump": "jump"},
        action_aliases={},
        unknown_action_id="unknown",
        confidence_floor=0.0,
        dry_run=True,
    )
    report = run_finetune(cfg)

    assert report["split"] == "train"
    assert report["processed_samples"] == 1
    assert report["total_action_labels"] == 4
    assert report["unknown_action_labels"] == 1
    assert report["unknown_ratio"] == 0.25
    assert report["checkpoint_path"] == str(tmp_path / "outputs" / "checkpoints" / "latest.ckpt")


def test_run_finetune_persists_summary_metrics_file(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    episode_dir = raw_root / "ep_001"
    frame_names = [f"{i:04d}.png" for i in range(2)]
    _touch_frames(episode_dir / "frames", frame_names)
    _write_actions_json(episode_dir, frame_names, ["jump", "slide"])

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

    manifest = build_manifest(
        BuildDatasetConfig(
            input_root=str(raw_root),
            output_manifest_path=str(tmp_path / "manifest.json"),
            seed=0,
            split_policy=SplitPolicy(train=0.9999998, val=1e-7, test=1e-7),
            clip_length=2,
            stride=2,
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    cfg = FineTuneConfig(
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "outputs"),
        split=SplitName.TRAIN,
        action_mapping={"jump": "jump"},
        dry_run=True,
        save_summary=True,
    )
    report = run_finetune(cfg)

    metrics_path = Path(report["metrics_path"])
    assert metrics_path.exists()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["processed_samples"] == report["processed_samples"]
    assert metrics_payload["unknown_action_labels"] == report["unknown_action_labels"]

