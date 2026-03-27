"""Build a validated dataset manifest from raw episode records."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.schema import ActionLabel, ClipRecord, EpisodeRecord, FrameRecord, SplitName, SplitPolicy


@dataclass(slots=True, frozen=True)
class BuildDatasetConfig:
    """Config for dataset manifest construction."""

    dataset_name: str
    source_root: str
    output_manifest_path: str
    seed: int
    split_policy: SplitPolicy
    episodes_file: str


def _load_config(config_path: Path) -> BuildDatasetConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    split_ratios = raw["split_ratios"]
    split_policy = SplitPolicy(
        train=float(split_ratios["train"]),
        val=float(split_ratios["val"]),
        test=float(split_ratios["test"]),
    )
    return BuildDatasetConfig(
        dataset_name=str(raw["dataset_name"]),
        source_root=str(raw["source_root"]),
        output_manifest_path=str(raw["output_manifest_path"]),
        seed=int(raw.get("seed", 0)),
        split_policy=split_policy,
        episodes_file=str(raw["episodes_file"]),
    )


def _assign_split(order_idx: int, total: int, policy: SplitPolicy) -> SplitName:
    train_cutoff = int(total * policy.train)
    val_cutoff = train_cutoff + int(total * policy.val)
    if order_idx < train_cutoff:
        return SplitName.TRAIN
    if order_idx < val_cutoff:
        return SplitName.VAL
    return SplitName.TEST


def _parse_episode(raw_episode: dict[str, Any], split: SplitName) -> EpisodeRecord:
    clips: list[ClipRecord] = []
    for raw_clip in raw_episode["clips"]:
        frames = tuple(
            FrameRecord(
                frame_idx=int(raw_frame["frame_idx"]),
                frame_path=str(raw_frame["frame_path"]),
                timestamp_sec=float(raw_frame["timestamp_sec"]),
            )
            for raw_frame in raw_clip["frames"]
        )
        actions = tuple(
            ActionLabel(
                action_id=str(raw_action["action_id"]),
                action_text=str(raw_action.get("action_text", raw_action["action_id"])),
                confidence=float(raw_action.get("confidence", 1.0)),
            )
            for raw_action in raw_clip["action_labels"]
        )
        clips.append(
            ClipRecord(
                clip_id=str(raw_clip["clip_id"]),
                start_frame_idx=int(raw_clip["start_frame_idx"]),
                end_frame_idx=int(raw_clip["end_frame_idx"]),
                frames=frames,
                action_labels=actions,
                metadata={str(k): str(v) for k, v in raw_clip.get("metadata", {}).items()},
            )
        )

    return EpisodeRecord(
        episode_id=str(raw_episode["episode_id"]),
        game=str(raw_episode["game"]),
        demonstrator_id=str(raw_episode["demonstrator_id"]),
        clips=tuple(clips),
        split=split,
        metadata={str(k): str(v) for k, v in raw_episode.get("metadata", {}).items()},
    )


def _episode_to_dict(episode: EpisodeRecord) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "game": episode.game,
        "demonstrator_id": episode.demonstrator_id,
        "split": episode.split.value,
        "metadata": dict(episode.metadata),
        "clips": [
            {
                "clip_id": clip.clip_id,
                "start_frame_idx": clip.start_frame_idx,
                "end_frame_idx": clip.end_frame_idx,
                "metadata": dict(clip.metadata),
                "frames": [
                    {
                        "frame_idx": frame.frame_idx,
                        "frame_path": frame.frame_path,
                        "timestamp_sec": frame.timestamp_sec,
                    }
                    for frame in clip.frames
                ],
                "action_labels": [
                    {
                        "action_id": action.action_id,
                        "action_text": action.action_text,
                        "confidence": action.confidence,
                    }
                    for action in clip.action_labels
                ],
            }
            for clip in episode.clips
        ],
    }


def build_manifest(config: BuildDatasetConfig) -> dict[str, Any]:
    """Construct and validate a dataset manifest from raw episodes."""
    episodes_path = Path(config.episodes_file)
    with episodes_path.open("r", encoding="utf-8") as f:
        raw_episodes = json.load(f)

    if not isinstance(raw_episodes, list):
        raise ValueError("episodes_file must contain a JSON array.")

    indices = list(range(len(raw_episodes)))
    rng = random.Random(config.seed)
    rng.shuffle(indices)

    parsed: list[EpisodeRecord] = []
    for order_idx, raw_idx in enumerate(indices):
        split = _assign_split(order_idx=order_idx, total=len(indices), policy=config.split_policy)
        parsed.append(_parse_episode(raw_episodes[raw_idx], split=split))

    split_counts = {
        SplitName.TRAIN.value: sum(1 for ep in parsed if ep.split == SplitName.TRAIN),
        SplitName.VAL.value: sum(1 for ep in parsed if ep.split == SplitName.VAL),
        SplitName.TEST.value: sum(1 for ep in parsed if ep.split == SplitName.TEST),
    }

    return {
        "dataset_name": config.dataset_name,
        "source_root": config.source_root,
        "split_policy": {
            "train": config.split_policy.train,
            "val": config.split_policy.val,
            "test": config.split_policy.test,
            "seed": config.seed,
        },
        "split_counts": split_counts,
        "episodes": [_episode_to_dict(ep) for ep in parsed],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build NitroGen adaptation dataset manifest.")
    parser.add_argument("--config", required=True, help="Path to build config JSON.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = _load_config(Path(args.config))
    manifest = build_manifest(config)

    output_path = Path(config.output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
