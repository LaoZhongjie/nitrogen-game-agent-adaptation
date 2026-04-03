"""Tests for video evaluation metrics (no full model required)."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    compute_mean_mae,
    compute_pooled_distribution_metrics,
    compute_temporal_consistency,
    compute_temporal_error_vs_reference,
    evaluate_records,
    evaluate_video_pair,
)


def test_compute_mean_mae_identical() -> None:
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert compute_mean_mae(img, img) == pytest.approx(0.0)


def test_temporal_consistency_single_frame() -> None:
    f = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    assert compute_temporal_consistency(f) == 1.0


def test_evaluate_video_pair_with_reference() -> None:
    g = np.full((3, 16, 16, 3), 100, dtype=np.uint8)
    r = np.full((3, 16, 16, 3), 120, dtype=np.uint8)
    rec = evaluate_video_pair("c1", "g1", g, r)
    assert rec.mean_mae is not None
    assert rec.mean_mae > 0
    assert rec.reference_temporal_consistency is not None
    assert rec.temporal_error_vs_reference is not None
    assert rec.psnr_mean is not None
    assert rec.ssim_mean is not None


def test_evaluate_records_aggregation() -> None:
    g = np.full((2, 8, 8, 3), 50, dtype=np.uint8)
    r = np.full((2, 8, 8, 3), 60, dtype=np.uint8)
    records = [
        evaluate_video_pair("a", "g", g, r),
        evaluate_video_pair("b", "g", g, r),
    ]
    summary = evaluate_records(records)
    assert summary.total_videos == 2
    assert summary.mean_mae is not None
    assert summary.mean_reference_temporal_consistency is not None
    assert summary.total_frames_with_reference == 4


def test_pooled_distribution_metrics_small() -> None:
    gen = np.random.randn(8, 16).astype(np.float64)
    ref = np.random.randn(8, 16).astype(np.float64)
    fid, l2 = compute_pooled_distribution_metrics(gen, ref)
    assert fid is None  # below FID_MIN_FRAMES_PER_POOL
    assert l2 is not None and l2 >= 0.0


def test_pooled_fid_enough_frames() -> None:
    rng = np.random.default_rng(0)
    gen = rng.standard_normal((50, 8)).astype(np.float64)
    ref = rng.standard_normal((50, 8)).astype(np.float64)
    fid, l2 = compute_pooled_distribution_metrics(gen, ref)
    assert l2 is not None
    assert fid is not None and fid >= 0.0


def test_temporal_error_vs_reference_matches_motion() -> None:
    g = np.stack(
        [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 10, dtype=np.uint8)],
        axis=0,
    )
    r = g.copy()
    err = compute_temporal_error_vs_reference(g, r, n=2)
    assert err == pytest.approx(0.0)
