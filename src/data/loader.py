"""Manifest-based dataset loader for NitroGen post-training clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.data.schema import SplitName


@dataclass(slots=True, frozen=True)
class ClipSample:
    """A single clip sample returned by ManifestDataset."""

    clip_id: str
    episode_id: str
    frame_paths: tuple[str, ...]
    action_labels: tuple[str, ...]
    split: SplitName


class ManifestDataset:
    """Loads a clip-level manifest and provides indexed access to clip samples.

    Optionally filters by split and/or re-windows clips to a different
    clip_length/stride than the original manifest.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: SplitName | None = None,
        clip_length: int | None = None,
        stride: int | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        raw_clips: list[dict] = raw["clips"]

        if clip_length is not None:
            if stride is None:
                stride = clip_length
            raw_clips = self._rewindow(raw_clips, clip_length, stride)

        samples: list[ClipSample] = []
        for clip in raw_clips:
            clip_split = SplitName(clip["split"])
            if split is not None and clip_split != split:
                continue
            samples.append(
                ClipSample(
                    clip_id=clip["clip_id"],
                    episode_id=clip["episode_id"],
                    frame_paths=tuple(clip["frame_paths"]),
                    action_labels=tuple(clip["action_labels"]),
                    split=clip_split,
                )
            )

        self._samples = samples

    @staticmethod
    def _rewindow(
        clips: list[dict], clip_length: int, stride: int
    ) -> list[dict]:
        """Re-window existing clips into new clip_length/stride."""
        rewindowed: list[dict] = []
        for clip in clips:
            frames = clip["frame_paths"]
            actions = clip["action_labels"]
            total = len(frames)
            if total < clip_length:
                continue
            idx = 0
            for start in range(0, total - clip_length + 1, stride):
                end = start + clip_length
                rewindowed.append(
                    {
                        "clip_id": f"{clip['clip_id']}_rw_{idx:06d}",
                        "episode_id": clip["episode_id"],
                        "frame_paths": frames[start:end],
                        "action_labels": actions[start:end],
                        "split": clip["split"],
                    }
                )
                idx += 1
        return rewindowed

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ClipSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[ClipSample]:
        return iter(self._samples)
