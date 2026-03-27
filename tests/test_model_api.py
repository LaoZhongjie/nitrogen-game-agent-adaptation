"""Import-contract tests for public model API surface."""

from __future__ import annotations

from src.model import (
    AlignedActionRecord,
    AlignedManifestSample,
    AlignmentBatch,
    AlignmentSummary,
    RawActionRecord,
    VocabularyActionAligner,
    align_action_texts_to_labels,
    align_manifest_sample,
    to_action_labels,
)


def test_model_api_exports_alignment_symbols() -> None:
    aligner = VocabularyActionAligner(mapping={"jump": "jump"})
    raw = RawActionRecord(action_text="jump", confidence=1.0)
    aligned = aligner.align(raw)
    labels = align_action_texts_to_labels(aligner=aligner, action_texts=["jump"])
    summary = AlignmentSummary(total_count=1, known_count=1, unknown_count=0, mean_confidence=1.0)
    batch = AlignmentBatch(records=(aligned,), summary=summary)
    labels_from_aligned = to_action_labels((aligned,))

    assert isinstance(aligned, AlignedActionRecord)
    assert isinstance(batch, AlignmentBatch)
    assert isinstance(labels, tuple)
    assert isinstance(labels_from_aligned, tuple)
    assert callable(align_manifest_sample)
    assert AlignedManifestSample is not None
