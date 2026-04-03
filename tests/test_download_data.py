"""Tests for NitroGen download helpers."""

from __future__ import annotations

import pytest

from scripts import download_data


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://m.youtube.com/watch?v=abc", True),
        ("https://example.com/video.mp4", False),
        ("not-a-url", False),
    ],
)
def test_needs_ytdlp(url: str, expected: bool) -> None:
    assert download_data._needs_ytdlp(url) is expected


def test_segment_range_seconds() -> None:
    assert download_data._segment_range_seconds(10.0, 30.0) == (10.0, 30.0)
    assert download_data._segment_range_seconds(10.0, 10.0) is None
    assert download_data._segment_range_seconds(None, 5.0) is None
    assert download_data._safe_metadata_float("12.5") == 12.5
    assert download_data._safe_metadata_float("x") is None
