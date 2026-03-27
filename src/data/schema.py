"""Dataset schema contracts for NitroGen post-training adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class SplitName(str, Enum):
    """Allowed dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(slots=True, frozen=True)
class ActionLabel:
    """Action label aligned to a single frame."""

    action_id: str
    action_text: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id must be non-empty.")
        if not self.action_text.strip():
            raise ValueError("action_text must be non-empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")


@dataclass(slots=True, frozen=True)
class FrameRecord:
    """Single frame reference and timing metadata."""

    frame_idx: int
    frame_path: str
    timestamp_sec: float

    def __post_init__(self) -> None:
        if self.frame_idx < 0:
            raise ValueError("frame_idx must be >= 0.")
        if not self.frame_path.strip():
            raise ValueError("frame_path must be non-empty.")
        if self.timestamp_sec < 0.0:
            raise ValueError("timestamp_sec must be >= 0.0.")


@dataclass(slots=True, frozen=True)
class ClipRecord:
    """Contiguous clip containing aligned frames and actions."""

    clip_id: str
    start_frame_idx: int
    end_frame_idx: int
    frames: tuple[FrameRecord, ...]
    action_labels: tuple[ActionLabel, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.clip_id.strip():
            raise ValueError("clip_id must be non-empty.")
        if self.start_frame_idx < 0:
            raise ValueError("start_frame_idx must be >= 0.")
        if self.end_frame_idx < self.start_frame_idx:
            raise ValueError("end_frame_idx must be >= start_frame_idx.")
        if len(self.frames) == 0:
            raise ValueError("frames must be non-empty.")
        if len(self.frames) != len(self.action_labels):
            raise ValueError("frames and action_labels must have equal length.")
        for key, value in self.metadata.items():
            if not key.strip() or not value.strip():
                raise ValueError("metadata keys and values must be non-empty strings.")


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    """Single demonstration episode with clips and split assignment."""

    episode_id: str
    game: str
    demonstrator_id: str
    clips: tuple[ClipRecord, ...]
    split: SplitName
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty.")
        if not self.game.strip():
            raise ValueError("game must be non-empty.")
        if not self.demonstrator_id.strip():
            raise ValueError("demonstrator_id must be non-empty.")
        if len(self.clips) == 0:
            raise ValueError("clips must be non-empty.")
        for key, value in self.metadata.items():
            if not key.strip() or not value.strip():
                raise ValueError("metadata keys and values must be non-empty strings.")


@dataclass(slots=True, frozen=True)
class SplitPolicy:
    """Episode-level split policy ratios."""

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if any(part <= 0.0 for part in (self.train, self.val, self.test)):
            raise ValueError("split parts must be > 0.0.")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0.")
