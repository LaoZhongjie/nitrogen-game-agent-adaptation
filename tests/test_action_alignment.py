"""Unit tests for Stage 2 action alignment module."""

from __future__ import annotations

import pytest

from src.model.alignment import AlignedActionRecord, RawActionRecord, VocabularyActionAligner, to_action_labels


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


def test_align_many_returns_empty_summary_for_empty_input() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})

    batch = aligner.align_many([])

    assert batch.records == ()
    assert batch.summary.total_count == 0
    assert batch.summary.known_count == 0
    assert batch.summary.unknown_count == 0
    assert batch.summary.mean_confidence == 0.0


def test_align_many_returns_mixed_known_unknown_summary() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump", "left": "move_left"}, unknown_action_id="other")
    raw_actions = [
        RawActionRecord(action_text="jump", confidence=1.0),
        RawActionRecord(action_text="slide", confidence=0.4),
        RawActionRecord(action_text="LEFT", confidence=0.6),
    ]

    batch = aligner.align_many(raw_actions)

    assert [record.action_id for record in batch.records] == ["jump", "other", "move_left"]
    assert batch.summary.total_count == 3
    assert batch.summary.known_count == 2
    assert batch.summary.unknown_count == 1
    assert batch.summary.mean_confidence == pytest.approx((1.0 + 0.4 + 0.6) / 3.0)


def test_aliases_map_to_canonical_token_before_lookup() -> None:
    aligner = VocabularyActionAligner(
        mapping={"left": "move_left"},
        aliases={"move left": "left"},
        unknown_action_id="other",
    )

    aligned = aligner.align(RawActionRecord(action_text="Move Left", confidence=0.8))

    assert aligned.action_id == "move_left"


def test_aliases_fall_back_when_alias_not_found() -> None:
    aligner = VocabularyActionAligner(
        mapping={"left": "move_left"},
        aliases={"move left": "left"},
        unknown_action_id="other",
    )

    aligned = aligner.align(RawActionRecord(action_text="strafe", confidence=0.3))

    assert aligned.action_id == "other"


def test_mapping_takes_precedence_over_alias_lookup() -> None:
    aligner = VocabularyActionAligner(
        mapping={"move left": "strafe_left", "left": "move_left"},
        aliases={"move left": "left"},
        unknown_action_id="other",
    )

    aligned = aligner.align(RawActionRecord(action_text="Move Left", confidence=1.0))

    assert aligned.action_id == "strafe_left"


def test_to_action_labels_preserves_alignment_fields() -> None:
    aligned_actions = (
        AlignedActionRecord(action_id="jump", action_text="Jump", confidence=0.7),
        AlignedActionRecord(action_id="move_left", action_text="left", confidence=0.8),
    )

    labels = to_action_labels(aligned_actions)

    assert [label.action_id for label in labels] == ["jump", "move_left"]
    assert [label.action_text for label in labels] == ["Jump", "left"]
    assert [label.confidence for label in labels] == [0.7, 0.8]
