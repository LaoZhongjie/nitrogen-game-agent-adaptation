"""Dataset loader for clip manifests emitted by ``scripts.build_dataset``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.data.schema import SplitName


@dataclass(slots=True, frozen=True)
class ManifestSample:
    """One clip sample from a built dataset manifest."""

    episode_id: str
    clip_id: str
    frame_paths: tuple[str, ...]
    action_labels: tuple[str, ...]
    split: SplitName


class ManifestDataset(Sequence[ManifestSample]):
    """Load and optionally re-window clip samples from a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: SplitName | None = None,
        clip_length: int | None = None,
        stride: int | None = None,
    ) -> None:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")

        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        if not isinstance(raw, dict):
            raise ValueError("manifest root must be an object.")

        raw_clips = raw.get("clips")
        if not isinstance(raw_clips, list):
            raise ValueError("manifest missing clips array.")

        parsed = [self._parse_clip(item) for item in raw_clips]
        if split is not None:
            parsed = [sample for sample in parsed if sample.split is split]

        if clip_length is None and stride is None:
            self._samples = tuple(parsed)
            return

        # Re-window existing samples if either parameter is provided.
        effective_clip_length = clip_length if clip_length is not None else 0
        effective_stride = stride if stride is not None else 0
        if effective_clip_length <= 0 or effective_stride <= 0:
            raise ValueError("clip_length and stride must both be > 0 when re-windowing.")

        rew_windowed: list[ManifestSample] = []
        for sample in parsed:
            rew_windowed.extend(
                self._rewindow_sample(
                    sample=sample,
                    clip_length=effective_clip_length,
                    stride=effective_stride,
                )
            )
        self._samples = tuple(rew_windowed)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ManifestSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[ManifestSample]:
        return iter(self._samples)

    def _parse_clip(self, raw_clip: Any) -> ManifestSample:
        if not isinstance(raw_clip, dict):
            raise ValueError("each clip entry must be an object.")

        episode_id = str(raw_clip["episode_id"])
        clip_id = str(raw_clip["clip_id"])

        frame_paths_raw = raw_clip["frame_paths"]
        action_labels_raw = raw_clip["action_labels"]
        split_raw = str(raw_clip["split"])

        if not isinstance(frame_paths_raw, list) or not isinstance(action_labels_raw, list):
            raise ValueError("clip frame_paths and action_labels must be arrays.")
        if len(frame_paths_raw) != len(action_labels_raw):
            raise ValueError("clip frame_paths and action_labels must have equal length.")

        try:
            split = SplitName(split_raw)
        except ValueError as exc:
            raise ValueError(f"invalid split value: {split_raw}") from exc

        frame_paths = tuple(str(p) for p in frame_paths_raw)
        action_labels = tuple(str(a) for a in action_labels_raw)

        return ManifestSample(
            episode_id=episode_id,
            clip_id=clip_id,
            frame_paths=frame_paths,
            action_labels=action_labels,
            split=split,
        )

    def _rewindow_sample(self, sample: ManifestSample, clip_length: int, stride: int) -> list[ManifestSample]:
        n = len(sample.frame_paths)
        if n != len(sample.action_labels):
            raise ValueError("sample frame_paths and action_labels must have equal length.")
        if n < clip_length:
            return []

        out: list[ManifestSample] = []
        window_idx = 0
        for start in range(0, n - clip_length + 1, stride):
            end = start + clip_length
            out.append(
                ManifestSample(
                    episode_id=sample.episode_id,
                    clip_id=f"{sample.clip_id}_rw_{window_idx:06d}",
                    frame_paths=sample.frame_paths[start:end],
                    action_labels=sample.action_labels[start:end],
                    split=sample.split,
                )
            )
            window_idx += 1
        return out
