"""Action alignment interfaces and baseline mapper implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


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
class AlignmentSummary:
    """Batch-level summary statistics for aligned actions."""

    total_count: int
    known_count: int
    unknown_count: int
    mean_confidence: float

    def __post_init__(self) -> None:
        if self.total_count < 0:
            raise ValueError("total_count must be >= 0.")
        if self.known_count < 0:
            raise ValueError("known_count must be >= 0.")
        if self.unknown_count < 0:
            raise ValueError("unknown_count must be >= 0.")
        if self.known_count + self.unknown_count != self.total_count:
            raise ValueError("known_count + unknown_count must equal total_count.")
        if not 0.0 <= self.mean_confidence <= 1.0:
            raise ValueError("mean_confidence must be in range [0.0, 1.0].")


@dataclass(slots=True, frozen=True)
class AlignmentBatch:
    """Aligned records and summary stats for a batch of raw actions."""

    records: tuple[AlignedActionRecord, ...]
    summary: AlignmentSummary


@dataclass(slots=True, frozen=True)
class VocabularyActionAligner:
    """Baseline action aligner using deterministic vocabulary lookup."""

    mapping: Mapping[str, str]
    aliases: Mapping[str, str] | None = None
    unknown_action_id: str = "unknown"

    def __post_init__(self) -> None:
        if not self.unknown_action_id.strip():
            raise ValueError("unknown_action_id must be non-empty.")

    def align(self, raw_action: RawActionRecord) -> AlignedActionRecord:
        """Map raw action text to canonical action ID, with unknown fallback."""
        normalized = raw_action.action_text.strip().lower()
        action_id = self.mapping.get(normalized)
        if action_id is None and self.aliases is not None:
            alias_normalized = self.aliases.get(normalized)
            if alias_normalized is not None:
                action_id = self.mapping.get(alias_normalized)
        if action_id is None:
            action_id = self.unknown_action_id
        return AlignedActionRecord(
            action_id=action_id,
            action_text=raw_action.action_text,
            confidence=raw_action.confidence,
        )

    def align_many(self, raw_actions: Sequence[RawActionRecord]) -> AlignmentBatch:
        """Align a sequence of raw actions and emit summary statistics."""
        records = tuple(self.align(raw_action) for raw_action in raw_actions)
        total_count = len(records)
        if total_count == 0:
            summary = AlignmentSummary(total_count=0, known_count=0, unknown_count=0, mean_confidence=0.0)
            return AlignmentBatch(records=records, summary=summary)

        unknown_count = sum(1 for record in records if record.action_id == self.unknown_action_id)
        known_count = total_count - unknown_count
        mean_confidence = sum(record.confidence for record in records) / float(total_count)
        summary = AlignmentSummary(
            total_count=total_count,
            known_count=known_count,
            unknown_count=unknown_count,
            mean_confidence=mean_confidence,
        )
        return AlignmentBatch(records=records, summary=summary)
