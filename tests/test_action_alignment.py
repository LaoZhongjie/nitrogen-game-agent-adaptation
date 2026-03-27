"""Unit tests for Stage 2 action alignment module."""

from __future__ import annotations

import pytest

from src.model.alignment import (
    AlignedActionRecord,
    RawActionRecord,
    VocabularyActionAligner,
    align_action_texts_to_labels,
    to_action_labels,
)


def test_vocabulary_aligner_maps_known_action() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})

    aligned = aligner.align(RawActionRecord(action_text="Jump", confidence=0.9))

    assert aligned.action_id == "jump"
    assert aligned.action_text == "Jump"
    assert aligned.confidence == 0.9
    assert aligned.alignment_source == "direct"


def test_vocabulary_aligner_falls_back_to_unknown() -> None:
    aligner = VocabularyActionAligner(
        mapping={"move_left": "move_left"},
        unknown_action_id="other",
    )

    aligned = aligner.align(RawActionRecord(action_text="slide", confidence=0.5))

    assert aligned.action_id == "other"
    assert aligned.alignment_source == "unknown"


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
    assert aligned.alignment_source == "alias"


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
    assert aligned.alignment_source == "direct"


def test_to_action_labels_preserves_alignment_fields() -> None:
    aligned_actions = (
        AlignedActionRecord(action_id="jump", action_text="Jump", confidence=0.7, alignment_source="direct"),
        AlignedActionRecord(
            action_id="move_left",
            action_text="left",
            confidence=0.8,
            alignment_source="alias",
        ),
    )

    labels = to_action_labels(aligned_actions)

    assert [label.action_id for label in labels] == ["jump", "move_left"]
    assert [label.action_text for label in labels] == ["Jump", "left"]
    assert [label.confidence for label in labels] == [0.7, 0.8]
    assert all(not hasattr(label, "alignment_source") for label in labels)


def test_low_confidence_mapping_falls_back_to_unknown() -> None:
    aligner = VocabularyActionAligner(
        mapping={"jump": "jump"},
        unknown_action_id="other",
        confidence_floor=0.6,
    )

    aligned = aligner.align(RawActionRecord(action_text="jump", confidence=0.59))

    assert aligned.action_id == "other"
    assert aligned.alignment_source == "confidence_floor"


def test_confidence_floor_boundary_is_inclusive() -> None:
    aligner = VocabularyActionAligner(
        mapping={"jump": "jump"},
        unknown_action_id="other",
        confidence_floor=0.6,
    )

    aligned = aligner.align(RawActionRecord(action_text="jump", confidence=0.6))

    assert aligned.action_id == "jump"


def test_align_many_summary_reflects_confidence_floor_unknowns() -> None:
    aligner = VocabularyActionAligner(
        mapping={"jump": "jump", "left": "move_left"},
        unknown_action_id="other",
        confidence_floor=0.5,
    )
    raw_actions = [
        RawActionRecord(action_text="jump", confidence=0.9),
        RawActionRecord(action_text="left", confidence=0.49),
        RawActionRecord(action_text="slide", confidence=0.8),
    ]

    batch = aligner.align_many(raw_actions)

    assert [record.action_id for record in batch.records] == ["jump", "other", "other"]
    assert [record.alignment_source for record in batch.records] == ["direct", "confidence_floor", "unknown"]
    assert batch.summary.known_count == 1
    assert batch.summary.unknown_count == 2


def test_align_action_texts_to_labels_uses_default_confidence() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})

    labels = align_action_texts_to_labels(
        aligner=aligner,
        action_texts=["jump", "slide"],
    )

    assert [label.action_id for label in labels] == ["jump", "unknown"]
    assert [label.confidence for label in labels] == [1.0, 1.0]


def test_align_action_texts_to_labels_rejects_length_mismatch() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})

    with pytest.raises(ValueError, match="must match"):
        align_action_texts_to_labels(
            aligner=aligner,
            action_texts=["jump", "slide"],
            confidences=[0.9],
        )


def test_align_action_texts_to_labels_respects_alias_and_confidence_floor() -> None:
    aligner = VocabularyActionAligner(
        mapping={"left": "move_left"},
        aliases={"move left": "left"},
        unknown_action_id="other",
        confidence_floor=0.6,
    )

    labels = align_action_texts_to_labels(
        aligner=aligner,
        action_texts=["Move Left", "left", "strafe"],
        confidences=[0.7, 0.5, 0.9],
    )

    assert [label.action_id for label in labels] == ["move_left", "other", "other"]
    assert [label.confidence for label in labels] == [0.7, 0.5, 0.9]
