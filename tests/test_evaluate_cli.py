from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate import main as evaluate_main
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

