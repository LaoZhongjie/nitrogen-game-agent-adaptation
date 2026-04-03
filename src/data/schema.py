"""Dataset schema contracts for NitroGen video chunks and gamepad actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class SplitName(str, Enum):
    """Allowed dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(slots=True, frozen=True)
class SplitPolicy:
    """Video-level split policy ratios."""

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if any(part <= 0.0 for part in (self.train, self.val, self.test)):
            raise ValueError("split parts must be > 0.0.")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0.")


BUTTON_COLUMNS: tuple[str, ...] = (
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "west",
    "east",
    "north",
    "back",
    "start",
    "guide",
)

JOYSTICK_COLUMNS: tuple[str, ...] = ("j_left", "j_right")

NUM_BUTTONS: int = len(BUTTON_COLUMNS)
NUM_JOYSTICK_AXES: int = 4  # j_left (x,y) + j_right (x,y)


@dataclass(slots=True, frozen=True)
class GamepadAction:
    """Single-frame gamepad state from the NitroGen dataset.

    ``buttons`` is a length-17 tuple of booleans corresponding to ``BUTTON_COLUMNS``.
    ``joysticks`` is a length-4 tuple of floats: (j_left_x, j_left_y, j_right_x, j_right_y),
    each in the range [-1.0, 1.0].
    """

    buttons: tuple[bool, ...]
    joysticks: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.buttons) != NUM_BUTTONS:
            raise ValueError(f"buttons must have length {NUM_BUTTONS}, got {len(self.buttons)}.")
        if len(self.joysticks) != NUM_JOYSTICK_AXES:
            raise ValueError(
                f"joysticks must have length {NUM_JOYSTICK_AXES}, got {len(self.joysticks)}."
            )
        for val in self.joysticks:
            if not -1.0 <= val <= 1.0:
                raise ValueError(f"joystick value must be in [-1.0, 1.0], got {val}.")


@dataclass(slots=True, frozen=True)
class ChunkMetadata:
    """Metadata for a single NitroGen 20-second video chunk."""

    uuid: str
    chunk_id: str
    chunk_size: int
    video_id: str
    game: str
    controller_type: str
    resolution: tuple[int, int]
    source: str
    url: str
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    bbox_game_area: Mapping[str, float] | None = None
    bbox_controller_overlay: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.uuid.strip():
            raise ValueError("uuid must be non-empty.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")


@dataclass(slots=True, frozen=True)
class VideoChunk:
    """A processed NitroGen video chunk ready for training.

    Pairs a local video path (or frame directory) with its per-frame gamepad
    action sequence and chunk-level metadata.
    """

    chunk_id: str
    video_id: str
    shard_id: str
    game: str
    video_path: str
    actions: tuple[GamepadAction, ...]
    metadata: ChunkMetadata
    split: SplitName

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty.")
        if not self.video_path.strip():
            raise ValueError("video_path must be non-empty.")
        if len(self.actions) == 0:
            raise ValueError("actions must be non-empty.")


@dataclass(slots=True, frozen=True)
class TrainingSample:
    """A single training pair for the world model: action conditioning + video target.

    ``prompt`` is an optional text description used for text-conditioned generation.
    ``num_frames`` stores the target frame count after preprocessing
    (must satisfy HunyuanVideo's 4k or 4k+1 constraint).
    """

    chunk_id: str
    video_path: str
    actions: tuple[GamepadAction, ...]
    prompt: str
    num_frames: int
    resolution: tuple[int, int]
    split: SplitName

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty.")
        if not self.video_path.strip():
            raise ValueError("video_path must be non-empty.")
        if len(self.actions) == 0:
            raise ValueError("actions must be non-empty.")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be > 0.")
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError("resolution dimensions must be > 0.")
