from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate import main as evaluate_main
from scripts.build_dataset import BuildDatasetConfig, build_manifest
from src.data.schema import SplitPolicy
from src.data.schema import SplitName
from src.eval.report import evaluate_records_file, load_evaluation_records


def _write_records(path: Path) -> None:
    payload = [
        {
            "episode_id": "ep_train",
            "clip_id": "clip_train_000",
            "split": "train",
            "target_action_ids": ["left", "jump", "left"],
            "predicted_action_ids": ["left", "slide", "left"],
        },
        {
            "episode_id": "ep_val",
            "clip_id": "clip_val_000",
            "split": "val",
            "target_action_ids": ["a", "a", "b"],
            "predicted_action_ids": ["a", "a", "b"],
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _touch_frames(frames_dir: Path, frame_names: list[str]) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in frame_names:
        (frames_dir / name).write_bytes(b"")


def _write_actions_json(episode_dir: Path, frames: list[str], actions: list[str]) -> None:
    rows = [{"frame": f, "action": a} for f, a in zip(frames, actions, strict=True)]
    (episode_dir / "actions.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")


def _build_manifest_with_one_clip(path: Path) -> Path:
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

def test_load_evaluation_records_with_split_filter(tmp_path: Path) -> None:
    input_path = tmp_path / "eval" / "records.json"
    _write_records(input_path)

    records = load_evaluation_records(path=input_path, split=SplitName.TRAIN)

    assert len(records) == 1
    assert records[0].split is SplitName.TRAIN
    assert records[0].episode_id == "ep_train"


def test_evaluate_records_file_writes_report_json(tmp_path: Path) -> None:
    input_path = tmp_path / "eval" / "records.json"
    output_path = tmp_path / "reports" / "offline_eval.json"
    _write_records(input_path)

    report = evaluate_records_file(
        input_records_path=str(input_path),
        output_report_path=str(output_path),
        split=None,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1_offline_evaluation_report"
    assert payload["summary"]["total_clip_count"] == 2
    assert payload["summary"]["total_action_count"] == 6
    assert payload["summary"]["total_correct_action_count"] == 5
    assert set(payload["summary"]["split_metrics"].keys()) == {"train", "val"}
    assert report.summary.total_clip_count == 2


def test_evaluate_cli_main_emits_and_writes_report(tmp_path: Path, monkeypatch: object, capsys: object) -> None:
    input_path = tmp_path / "eval" / "records.json"
    output_path = tmp_path / "reports" / "offline_eval_train.json"
    _write_records(input_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--split",
            "train",
        ],
    )

    evaluate_main()
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert printed["evaluated_split"] == "train"
    assert printed["summary"]["total_clip_count"] == 1
    assert printed["summary"]["total_action_count"] == 3
    assert output_path.exists()

def test_evaluate_cli_manifest_predictions_mode(tmp_path: Path, monkeypatch: object, capsys: object) -> None:
    manifest_path = _build_manifest_with_one_clip(tmp_path)
    predictions_path = tmp_path / "eval" / "predictions.json"
    output_path = tmp_path / "reports" / "offline_eval_from_manifest.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "clip_id": "ep_001_clip_000000",
                    "predicted_action_ids": ["left", "slide", "left"],
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--split",
            "train",
        ],
    )

    evaluate_main()
    printed = json.loads(capsys.readouterr().out)

    assert printed["evaluated_split"] == "train"
    assert printed["summary"]["total_clip_count"] == 1
    assert printed["summary"]["total_action_count"] == 3
    assert printed["summary"]["total_correct_action_count"] == 2
    assert printed["summary"]["action_accuracy"] == 2.0 / 3.0
    assert output_path.exists()


def test_evaluate_cli_manifest_predictions_rejects_missing_coverage_by_default(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    raw_root = tmp_path / "raw"
    episode_dir = raw_root / "ep_001"
    frame_names = [f"{i:04d}.png" for i in range(6)]
    _touch_frames(episode_dir / "frames", frame_names)
    _write_actions_json(episode_dir, frame_names, ["left", "jump", "left", "jump", "left", "jump"])
    manifest = build_manifest(
        BuildDatasetConfig(
            input_root=str(raw_root),
            output_manifest_path=str(tmp_path / "manifest.json"),
            seed=0,
            split_policy=SplitPolicy(train=0.9999998, val=1e-7, test=1e-7),
            clip_length=3,
            stride=3,
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    predictions_path = tmp_path / "eval" / "predictions.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "clip_id": "ep_001_clip_000000",
                    "predicted_action_ids": ["left", "slide", "left"],
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(tmp_path / "reports" / "offline_eval_partial.json"),
            "--split",
            "train",
        ],
    )

    import pytest

    with pytest.raises(ValueError, match="missing predictions"):
        evaluate_main()
