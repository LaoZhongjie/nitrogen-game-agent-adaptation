"""Build a validated dataset manifest from downloaded NitroGen data.

Scans the NitroGen data directory, parses parquet action files and metadata,
assigns deterministic train/val/test splits, and emits a manifest JSON.

Usage (from repo root)::

    python3.12 -m scripts.build_dataset \\
        --input data/nitrogen \\
        --output data/processed/manifest.json \\
        --seed 42 \\
        --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.loader import discover_chunks, load_video_chunk
from src.data.schema import SplitName, SplitPolicy
from src.data.split import VideoSplitAssigner


@dataclass(frozen=True)
class BuildDatasetConfig:
    """Config for NitroGen dataset manifest construction."""

    input_root: str
    output_manifest_path: str
    seed: int
    split_policy: SplitPolicy
    game_filter: Optional[str] = None
    max_chunks: Optional[int] = None
    use_processed_actions: bool = True
    split_granularity: str = "video"


def build_manifest(config: BuildDatasetConfig) -> dict[str, Any]:
    """Construct a chunk-level manifest from a NitroGen data directory."""
    if config.split_granularity not in ("video", "chunk"):
        raise ValueError("split_granularity must be 'video' or 'chunk'.")

    input_root = Path(config.input_root)
    if not input_root.exists():
        raise ValueError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")

    assigner = VideoSplitAssigner(
        policy=config.split_policy,
        seed=config.seed,
        granularity=config.split_granularity,
    )
    chunk_dirs = discover_chunks(config.input_root)

    chunks: list[dict[str, Any]] = []
    video_splits: dict[str, str] = {}
    games_seen: set[str] = set()

    for chunk_dir in chunk_dirs:
        if config.max_chunks is not None and len(chunks) >= config.max_chunks:
            break

        video_id = chunk_dir.parent.name
        chunk_id = chunk_dir.name
        if config.split_granularity == "chunk":
            split = assigner.assign(video_id, chunk_id)
        else:
            split = assigner.assign(video_id)
            video_splits[video_id] = split.value

        chunk = load_video_chunk(
            chunk_dir,
            split=split,
            use_processed_actions=config.use_processed_actions,
        )
        if chunk is None:
            continue

        if config.game_filter and chunk.game.lower() != config.game_filter.lower():
            continue

        games_seen.add(chunk.game)

        chunks.append({
            "chunk_id": chunk.chunk_id,
            "video_id": chunk.video_id,
            "shard_id": chunk.shard_id,
            "game": chunk.game,
            "video_path": chunk.video_path,
            "num_actions": len(chunk.actions),
            "split": split.value,
            "metadata": {
                "url": chunk.metadata.url,
                "resolution": list(chunk.metadata.resolution),
                "duration": chunk.metadata.duration,
            },
        })

    split_counts = {
        SplitName.TRAIN.value: sum(1 for c in chunks if c["split"] == SplitName.TRAIN.value),
        SplitName.VAL.value: sum(1 for c in chunks if c["split"] == SplitName.VAL.value),
        SplitName.TEST.value: sum(1 for c in chunks if c["split"] == SplitName.TEST.value),
    }

    return {
        "schema_version": "v3_nitrogen_manifest",
        "input_root": str(input_root.as_posix()),
        "split_policy": {
            "train": config.split_policy.train,
            "val": config.split_policy.val,
            "test": config.split_policy.test,
            "seed": config.seed,
        },
        "split_granularity": config.split_granularity,
        "video_splits": video_splits,
        "split_counts": split_counts,
        "games": sorted(games_seen),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NitroGen dataset manifest.")
    parser.add_argument("--input", required=True, help="NitroGen data directory.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Val split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--game-filter", type=str, default=None, help="Only include this game.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit total chunks.")
    parser.add_argument(
        "--split-granularity",
        choices=["video", "chunk"],
        default="video",
        help="Split by video (default) or by chunk (better spread for small downloads).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildDatasetConfig(
        input_root=str(args.input),
        output_manifest_path=str(args.output),
        seed=int(args.seed),
        split_policy=SplitPolicy(train=float(args.train), val=float(args.val), test=float(args.test)),
        game_filter=args.game_filter,
        max_chunks=args.max_chunks,
        split_granularity=str(args.split_granularity),
    )
    manifest = build_manifest(config)

    output_path = Path(config.output_manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps({
        "total_chunks": manifest["total_chunks"],
        "split_counts": manifest["split_counts"],
        "games": manifest["games"],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
