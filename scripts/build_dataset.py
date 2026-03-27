"""Build a validated dataset manifest from an episodes root directory.

Expected input layout:

- <input_root>/<episode_id>/
  - frames/                       # image files (any extension)
  - actions.json OR actions.csv   # action annotations aligned to frames

The builder emits a clip-level manifest JSON that is easy to index and sample.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.schema import SplitName, SplitPolicy
from src.data.split import EpisodeSplitAssigner


@dataclass(slots=True, frozen=True)
class BuildDatasetConfig:
    """Config for dataset manifest construction from an episodes root."""

    input_root: str
    output_manifest_path: str
    seed: int
    split_policy: SplitPolicy
    clip_length: int
    stride: int


def _is_frame_file(path: Path) -> bool:
    return path.is_file() and path.name.lower() not in {"thumbs.db", ".ds_store"}


def _sorted_frame_paths(frames_dir: Path) -> list[Path]:
    frame_files = [p for p in frames_dir.iterdir() if _is_frame_file(p)]
    # Deterministic order: lexicographic by filename.
    return sorted(frame_files, key=lambda p: p.name)


def _load_actions_json(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("actions.json must be a JSON array.")
    mapping: dict[str, str] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("actions.json rows must be objects.")
        frame_name = str(row["frame"])
        action = str(row["action"])
        mapping[frame_name] = action
    return mapping


def _load_actions_csv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("actions.csv must have a header row.")
        if "frame" not in reader.fieldnames or "action" not in reader.fieldnames:
            raise ValueError("actions.csv must have columns: frame, action.")
        for row in reader:
            mapping[str(row["frame"])] = str(row["action"])
    return mapping


def _load_actions(episode_dir: Path) -> dict[str, str]:
    json_path = episode_dir / "actions.json"
    csv_path = episode_dir / "actions.csv"
    if json_path.exists() and csv_path.exists():
        raise ValueError(f"Episode {episode_dir.name} has both actions.json and actions.csv.")
    if json_path.exists():
        return _load_actions_json(json_path)
    if csv_path.exists():
        return _load_actions_csv(csv_path)
    raise ValueError(f"Episode {episode_dir.name} missing actions.json or actions.csv.")


def _windowed_clips(
    episode_id: str,
    frame_paths: list[str],
    action_labels: list[str],
    clip_length: int,
    stride: int,
) -> list[dict[str, Any]]:
    if clip_length <= 0:
        raise ValueError("clip_length must be > 0.")
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if len(frame_paths) != len(action_labels):
        raise ValueError("frame_paths and action_labels must have equal length.")
    if len(frame_paths) < clip_length:
        return []

    clips: list[dict[str, Any]] = []
    clip_idx = 0
    for start in range(0, len(frame_paths) - clip_length + 1, stride):
        end = start + clip_length
        clips.append(
            {
                "episode_id": episode_id,
                "clip_id": f"{episode_id}_clip_{clip_idx:06d}",
                "frame_paths": frame_paths[start:end],
                "action_labels": action_labels[start:end],
            }
        )
        clip_idx += 1
    return clips


def build_manifest(config: BuildDatasetConfig) -> dict[str, Any]:
    """Construct a clip-level manifest from an input episodes root directory."""
    input_root = Path(config.input_root)
    if not input_root.exists():
        raise ValueError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")

    assigner = EpisodeSplitAssigner(policy=config.split_policy, seed=config.seed)

    episode_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    clips: list[dict[str, Any]] = []
    episode_splits: dict[str, str] = {}

    for episode_dir in episode_dirs:
        episode_id = episode_dir.name
        frames_dir = episode_dir / "frames"
        if not frames_dir.exists() or not frames_dir.is_dir():
            raise ValueError(f"Episode {episode_id} missing frames/ directory.")

        frame_files = _sorted_frame_paths(frames_dir)
        if len(frame_files) == 0:
            raise ValueError(f"Episode {episode_id} has no frames.")

        actions_map = _load_actions(episode_dir)

        rel_frame_paths: list[str] = []
        action_labels: list[str] = []
        for fp in frame_files:
            rel = fp.relative_to(input_root).as_posix()
            rel_frame_paths.append(rel)
            if fp.name not in actions_map:
                raise ValueError(f"Episode {episode_id} missing action for frame {fp.name}.")
            action_labels.append(actions_map[fp.name])

        split = assigner.assign(episode_id)
        episode_splits[episode_id] = split.value

        episode_clips = _windowed_clips(
            episode_id=episode_id,
            frame_paths=rel_frame_paths,
            action_labels=action_labels,
            clip_length=config.clip_length,
            stride=config.stride,
        )
        for c in episode_clips:
            c["split"] = split.value
        clips.extend(episode_clips)

    split_counts = {
        SplitName.TRAIN.value: sum(1 for c in clips if c["split"] == SplitName.TRAIN.value),
        SplitName.VAL.value: sum(1 for c in clips if c["split"] == SplitName.VAL.value),
        SplitName.TEST.value: sum(1 for c in clips if c["split"] == SplitName.TEST.value),
    }

    return {
        "schema_version": "v2_clip_manifest",
        "input_root": str(input_root.as_posix()),
        "split_policy": {
            "train": config.split_policy.train,
            "val": config.split_policy.val,
            "test": config.split_policy.test,
            "seed": config.seed,
        },
        "episode_splits": episode_splits,
        "clip_length": config.clip_length,
        "stride": config.stride,
        "split_counts": split_counts,
        "clips": clips,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build NitroGen adaptation dataset manifest.")
    parser.add_argument("--input", required=True, help="Root directory containing episodes.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic split seed.")
    parser.add_argument("--clip-length", type=int, default=16, help="Fixed clip length.")
    parser.add_argument("--stride", type=int, default=16, help="Clip stride.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Val split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = BuildDatasetConfig(
        input_root=str(args.input),
        output_manifest_path=str(args.output),
        seed=int(args.seed),
        split_policy=SplitPolicy(train=float(args.train), val=float(args.val), test=float(args.test)),
        clip_length=int(args.clip_length),
        stride=int(args.stride),
    )
    manifest = build_manifest(config)

    output_path = Path(config.output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
