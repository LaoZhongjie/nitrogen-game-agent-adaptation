"""Action alignment interfaces and baseline mapper implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(slots=True, frozen=True)
class RawActionRecord:
    """Raw action label emitted by a data source before alignment."""

    action_text: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.action_text.strip():
            raise ValueError("action_text must be non-empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")


@dataclass(slots=True, frozen=True)
class AlignedActionRecord:
    """Canonical action label after vocabulary alignment."""

    action_id: str
    action_text: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id must be non-empty.")
        if not self.action_text.strip():
            raise ValueError("action_text must be non-empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in range [0.0, 1.0].")


class ActionAligner(Protocol):
    """Interface for mapping raw actions into canonical vocabulary IDs."""

    def align(self, raw_action: RawActionRecord) -> AlignedActionRecord:
        """Align a raw action into canonical action-space."""


@dataclass(slots=True, frozen=True)
class VocabularyActionAligner:
    """Baseline action aligner using deterministic vocabulary lookup."""

    mapping: Mapping[str, str]
    unknown_action_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.unknown_action_id.strip():
            raise ValueError("unknown_action_id must be non-empty.")

    def align(self, raw_action: RawActionRecord) -> AlignedActionRecord:
        """Map raw action text to canonical action ID, with unknown fallback."""
        normalized = raw_action.action_text.strip().lower()
        action_id = self.mapping.get(normalized, self.unknown_action_id)
        return AlignedActionRecord(
            action_id=action_id,
            action_text=raw_action.action_text,
            confidence=raw_action.confidence,
        )
