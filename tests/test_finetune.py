from __future__ import annotations

import json
from pathlib import Path

from scripts.finetune import FineTuneConfig, load_config, run_finetune
from src.train.runner import TrainingRunContext, TrainingRunner, TrainingStepResult
from src.data.schema import SplitName
from src.train.state import TRAINING_STATE_SCHEMA, TrainingState


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
    assert "train_backend_metadata" in metrics_payload
    assert metrics_payload["train_backend_metadata"]["runner_backend"] == "train_stub"


def test_run_finetune_training_skeleton_writes_artifacts(tmp_path: Path) -> None:
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
        dry_run=False,
        save_summary=True,
        train_steps=3,
        seed=11,
    )
    report = run_finetune(cfg)

    checkpoint_path = Path(report["checkpoint_path"])
    metrics_path = Path(report["metrics_path"])

    assert report["mode"] == "train_stub"
    assert report["seed"] == 11
    assert checkpoint_path.exists()
    assert metrics_path.exists()

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert checkpoint_payload["mode"] == "train_stub"
    assert checkpoint_payload["seed"] == 11
    assert checkpoint_payload["schema"] == "checkpoint_payload_v1"
    ck_meta = checkpoint_payload["checkpoint_metadata"]
    assert ck_meta["checkpoint_version"] == "v1"
    assert ck_meta["backend"] == "train_stub"
    assert ck_meta["step_count"] == 3
    assert isinstance(ck_meta["state_digest"], str)
    assert len(ck_meta["state_digest"]) == 64
    assert "state" not in checkpoint_payload
    assert metrics_payload["mode"] == "train_stub"
    assert metrics_payload["processed_samples"] == report["processed_samples"]

    training_metadata_path = Path(report["training_metadata_path"])
    assert training_metadata_path.exists()
    training_payload = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    assert training_payload["train_steps"] == 3
    assert len(training_payload["step_metrics"]) == 3
    assert training_payload["step_metrics"][0]["step"] == 1
    assert training_payload["step_metrics"][-1]["step"] == 3
    assert training_payload["step_metrics"][0]["samples_seen"] == report["processed_samples"]


def test_run_finetune_train_noop_backend_writes_empty_steps(tmp_path: Path) -> None:
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
        dry_run=False,
        save_summary=True,
        train_steps=5,
        runner_backend="train_noop",
        seed=7,
    )
    report = run_finetune(cfg)

    assert report["mode"] == "train_noop"
    assert report["runner_backend"] == "train_noop"

    checkpoint_payload = json.loads(Path(report["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint_payload["schema"] == "checkpoint_payload_v1"
    ck_meta = checkpoint_payload["checkpoint_metadata"]
    assert ck_meta["checkpoint_version"] == "v1"
    assert ck_meta["backend"] == "train_noop"
    assert ck_meta["step_count"] == 5
    assert "state" not in checkpoint_payload

    training_metadata_path = Path(report["training_metadata_path"])
    training_payload = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    assert training_payload["mode"] == "train_noop"
    assert training_payload["train_steps"] == 5
    assert len(training_payload["step_metrics"]) == 5
    assert all(step["loss"] == 0.0 for step in training_payload["step_metrics"])


def test_run_finetune_train_mock_backend_writes_nontrivial_steps(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    episode_dir = raw_root / "ep_001"
    frame_names = [f"{i:04d}.png" for i in range(3)]
    _touch_frames(episode_dir / "frames", frame_names)
    _write_actions_json(episode_dir, frame_names, ["jump", "slide", "left"])

    from scripts.build_dataset import BuildDatasetConfig, build_manifest
    from src.data.schema import SplitPolicy

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
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    cfg = FineTuneConfig(
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "outputs"),
        split=SplitName.TRAIN,
        action_mapping={"jump": "jump", "left": "move_left"},
        dry_run=False,
        save_summary=True,
        train_steps=4,
        runner_backend="train_mock",
        mock_learning_rate=0.2,
        seed=3,
    )
    report = run_finetune(cfg)

    assert report["mode"] == "train_mock"
    assert report["runner_backend"] == "train_mock"
    assert report["mock_learning_rate"] == 0.2
    assert report["train_backend_metadata"]["mock_learning_rate"] == 0.2

    checkpoint_payload = json.loads(Path(report["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint_payload["schema"] == "checkpoint_payload_v1"
    ck_meta = checkpoint_payload["checkpoint_metadata"]
    assert ck_meta["checkpoint_version"] == "v1"
    assert ck_meta["backend"] == "train_mock"
    assert ck_meta["step_count"] == 4
    assert isinstance(ck_meta["state_digest"], str)
    assert len(ck_meta["state_digest"]) == 64
    state_payload = checkpoint_payload["state"]
    assert state_payload["schema"] == TRAINING_STATE_SCHEMA
    parsed_state = TrainingState.from_dict(state_payload)
    assert parsed_state.backend == "train_mock"
    assert parsed_state.train_steps == 4
    assert parsed_state.latest_step == 4
    assert parsed_state.mock_learning_rate == 0.2

    training_metadata_path = Path(report["training_metadata_path"])
    training_payload = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    losses = [step["loss"] for step in training_payload["step_metrics"]]
    assert training_payload["runner_backend"] == "train_mock"
    assert len(losses) == 4
    assert losses == sorted(losses, reverse=True)
    assert losses[0] > losses[-1]
    assert report["mock_learning_rate"] == 0.2
    assert training_payload["train_backend_metadata"]["runner_backend"] == "train_mock"


def test_run_finetune_supports_injected_runner_factory(tmp_path: Path) -> None:
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

    class _InjectedRunner:
        def run(self, context: TrainingRunContext) -> list[TrainingStepResult]:
            return [
                TrainingStepResult(
                    step=step_idx + 1,
                    loss=42.0,
                    known_ratio=context.known_ratio,
                    samples_seen=context.processed_samples,
                )
                for step_idx in range(context.train_steps)
            ]

    def _factory(_: str) -> TrainingRunner:
        return _InjectedRunner()

    cfg = FineTuneConfig(
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "outputs"),
        split=SplitName.TRAIN,
        action_mapping={"jump": "jump"},
        dry_run=False,
        save_summary=True,
        train_steps=3,
        runner_backend="train_stub",
    )

    report = run_finetune(cfg, runner_factory=_factory)
    training_payload = json.loads(Path(report["training_metadata_path"]).read_text(encoding="utf-8"))

    assert all(step["loss"] == 42.0 for step in training_payload["step_metrics"])


def test_load_config_supports_inline_mapping_and_aliases(tmp_path: Path) -> None:
    config_path = tmp_path / "finetune_config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest_path": "data/processed/manifest.json",
                "output_dir": "outputs/run_001",
                "split": "train",
                "dry_run": True,
                "save_summary": False,
                "runner_backend": "train_noop",
                "action_mapping": {"LEFT": "move_left", "jump": "jump"},
                "action_aliases": {"move left": "left"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.manifest_path == "data/processed/manifest.json"
    assert cfg.output_dir == "outputs/run_001"
    assert cfg.split is SplitName.TRAIN
    assert cfg.save_summary is False
    assert cfg.runner_backend == "train_noop"
    assert cfg.action_mapping == {"left": "move_left", "jump": "jump"}
    assert cfg.action_aliases == {"move left": "left"}


def test_load_config_supports_file_based_mapping_and_aliases(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    aliases_path = tmp_path / "aliases.json"
    mapping_path.write_text(json.dumps({"JUMP": "jump", "left": "move_left"}) + "\n", encoding="utf-8")
    aliases_path.write_text(json.dumps({"move left": "left"}) + "\n", encoding="utf-8")

    config_path = tmp_path / "finetune_config.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest_path": "data/processed/manifest.json",
                "output_dir": "outputs/run_002",
                "action_mapping_path": str(mapping_path),
                "action_aliases_path": str(aliases_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.action_mapping == {"jump": "jump", "left": "move_left"}
    assert cfg.action_aliases == {"move left": "left"}


def test_finetune_config_rejects_non_positive_mock_learning_rate() -> None:
    try:
        FineTuneConfig(
            manifest_path="data/processed/manifest.json",
            output_dir="outputs/run_bad_lr",
            mock_learning_rate=0.0,
        )
    except ValueError as exc:
        assert "mock_learning_rate must be > 0.0." in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive mock_learning_rate.")

