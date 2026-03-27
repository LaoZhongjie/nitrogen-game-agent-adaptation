"""Unit tests for Stage 2 action alignment module."""

from __future__ import annotations

import pytest

from src.model.alignment import RawActionRecord, VocabularyActionAligner


def test_vocabulary_aligner_maps_known_action() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})

    aligned = aligner.align(RawActionRecord(action_text="Jump", confidence=0.9))

    assert aligned.action_id == "jump"
    assert aligned.action_text == "Jump"
    assert aligned.confidence == 0.9


def test_vocabulary_aligner_falls_back_to_unknown() -> None:
    aligner = VocabularyActionAligner(
        mapping={"move_left": "move_left"},
        unknown_action_id="other",
    )

    aligned = aligner.align(RawActionRecord(action_text="slide", confidence=0.5))

    assert aligned.action_id == "other"


def test_raw_action_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="range"):
        RawActionRecord(action_text="jump", confidence=1.1)


def test_aligner_rejects_empty_unknown_action_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        VocabularyActionAligner(mapping={"jump": "jump"}, unknown_action_id=" ")
