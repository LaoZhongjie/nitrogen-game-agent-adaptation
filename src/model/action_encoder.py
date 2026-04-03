"""Gamepad action encoding for HunyuanVideo conditioning.

Provides two encoding strategies:
- **Text encoding** (MVP): converts gamepad states into natural-language
  descriptions that can be fed directly into the text-conditioning channel.
- **Vector encoding**: packs gamepad states into a dense float tensor suitable
  for injection as an additional conditioning signal (GameCraft-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from src.data.schema import BUTTON_COLUMNS, GamepadAction, NUM_BUTTONS, NUM_JOYSTICK_AXES


class ActionEncoder(Protocol):
    """Interface for encoding gamepad actions into model-compatible signals."""

    def encode_text(self, actions: Sequence[GamepadAction]) -> str:
        """Encode a sequence of gamepad actions as a text prompt fragment."""
        ...

    def encode_vector(self, actions: Sequence[GamepadAction]) -> np.ndarray:
        """Encode a sequence of gamepad actions as a float array.

        Returns shape ``(T, D)`` where T is the number of frames and
        D is the action feature dimension.
        """
        ...


BUTTON_DISPLAY_NAMES: dict[str, str] = {
    "dpad_down": "D-pad Down",
    "dpad_left": "D-pad Left",
    "dpad_right": "D-pad Right",
    "dpad_up": "D-pad Up",
    "left_shoulder": "LB",
    "left_thumb": "L3",
    "left_trigger": "LT",
    "right_shoulder": "RB",
    "right_thumb": "R3",
    "right_trigger": "RT",
    "south": "A",
    "west": "X",
    "east": "B",
    "north": "Y",
    "back": "Back",
    "start": "Start",
    "guide": "Guide",
}


def _joystick_direction(x: float, y: float, deadzone: float = 0.2) -> str:
    """Return a human-readable direction for a joystick position."""
    if abs(x) < deadzone and abs(y) < deadzone:
        return "neutral"
    parts: list[str] = []
    if y < -deadzone:
        parts.append("up")
    elif y > deadzone:
        parts.append("down")
    if x < -deadzone:
        parts.append("left")
    elif x > deadzone:
        parts.append("right")
    return "-".join(parts) if parts else "neutral"


def _summarize_action_text(action: GamepadAction) -> str:
    """Describe a single gamepad frame as a short text string."""
    pressed: list[str] = []
    for col, val in zip(BUTTON_COLUMNS, action.buttons):
        if val:
            pressed.append(BUTTON_DISPLAY_NAMES.get(col, col))

    jl_dir = _joystick_direction(action.joysticks[0], action.joysticks[1])
    jr_dir = _joystick_direction(action.joysticks[2], action.joysticks[3])

    parts: list[str] = []
    if pressed:
        parts.append("buttons: " + ", ".join(pressed))
    if jl_dir != "neutral":
        parts.append(f"left stick: {jl_dir}")
    if jr_dir != "neutral":
        parts.append(f"right stick: {jr_dir}")

    return "; ".join(parts) if parts else "idle"


def _detect_segments(actions: Sequence[GamepadAction]) -> list[tuple[int, int, str]]:
    """Group consecutive frames with the same summarized action into segments."""
    if not actions:
        return []

    segments: list[tuple[int, int, str]] = []
    current_desc = _summarize_action_text(actions[0])
    seg_start = 0

    for i in range(1, len(actions)):
        desc = _summarize_action_text(actions[i])
        if desc != current_desc:
            segments.append((seg_start, i - 1, current_desc))
            current_desc = desc
            seg_start = i
    segments.append((seg_start, len(actions) - 1, current_desc))
    return segments


@dataclass(frozen=True)
class GamepadActionEncoder:
    """Default encoder that supports both text and vector output modes.

    ``joystick_deadzone`` controls the threshold below which joystick
    deflection is treated as neutral in text descriptions.
    """

    joystick_deadzone: float = 0.2
    max_text_segments: int = 10

    def encode_text(self, actions: Sequence[GamepadAction]) -> str:
        """Produce a concise text summary of the action sequence.

        Groups identical consecutive frames into segments:
        ``"frames 0-15: idle; frames 16-30: buttons: A; left stick: up"``
        """
        if not actions:
            return "no actions"

        segments = _detect_segments(actions)

        if len(segments) > self.max_text_segments:
            segments = segments[: self.max_text_segments]

        parts: list[str] = []
        for start, end, desc in segments:
            if start == end:
                parts.append(f"frame {start}: {desc}")
            else:
                parts.append(f"frames {start}-{end}: {desc}")

        return "; ".join(parts)

    def encode_vector(self, actions: Sequence[GamepadAction]) -> np.ndarray:
        """Pack actions into a ``(T, 21)`` float32 array.

        Columns 0-16 are button booleans cast to 0.0/1.0.
        Columns 17-20 are joystick axes (j_left_x, j_left_y, j_right_x, j_right_y).
        """
        dim = NUM_BUTTONS + NUM_JOYSTICK_AXES
        arr = np.zeros((len(actions), dim), dtype=np.float32)
        for i, action in enumerate(actions):
            for j, pressed in enumerate(action.buttons):
                arr[i, j] = 1.0 if pressed else 0.0
            for j, val in enumerate(action.joysticks):
                arr[i, NUM_BUTTONS + j] = val
        return arr

    def encode_conditioning_prompt(
        self,
        actions: Sequence[GamepadAction],
        game_name: str = "",
        base_prompt: str = "",
    ) -> str:
        """Build a full conditioning prompt combining game context and action text.

        This is the primary method for text-based action conditioning in the
        MVP approach. For vector conditioning (Phase 2), use ``encode_vector``.
        """
        action_text = self.encode_text(actions)
        parts: list[str] = []
        if base_prompt:
            parts.append(base_prompt)
        if game_name:
            parts.append(f"Game: {game_name}.")
        parts.append(f"Player actions: {action_text}.")
        return " ".join(parts)
