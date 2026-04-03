"""Lightweight checks that a file looks like a playable MP4 (ISO BMFF).

FFmpeg/decord report ``moov atom not found`` when the file is truncated, not an
MP4 at all (e.g. HTML error page), or otherwise missing the ``moov`` box.
"""

from __future__ import annotations

from pathlib import Path


def is_probably_valid_mp4(path: Path, min_size_bytes: int = 1024) -> bool:
    """Return True if ``path`` appears to be a non-empty MP4 with a ``moov`` box.

    Reads at most ~5 MiB from the start and ~4 MiB from the end of large files,
    which covers typical ``moov`` placement (beginning or end).
    """
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < min_size_bytes:
        return False

    try:
        with path.open("rb") as fp:
            head = fp.read(12)
    except OSError:
        return False
    if len(head) < 12 or head[4:8] != b"ftyp":
        return False

    prefix_len = min(size, 5 * 1024 * 1024)
    try:
        with path.open("rb") as fp:
            prefix = fp.read(prefix_len)
    except OSError:
        return False

    if b"moov" in prefix:
        return True

    if size <= prefix_len:
        return False

    tail_len = min(size - prefix_len, 4 * 1024 * 1024)
    try:
        with path.open("rb") as fp:
            fp.seek(size - tail_len)
            suffix = fp.read(tail_len)
    except OSError:
        return False
    if b"moov" in suffix:
        return True

    # Cover bytes between first ``prefix_len`` and last ``tail_len`` (otherwise
    # ``moov`` could sit only in the middle for some container sizes).
    gap_start = prefix_len
    gap_end = size - tail_len
    if gap_end > gap_start:
        try:
            with path.open("rb") as fp:
                fp.seek(gap_start)
                middle = fp.read(gap_end - gap_start)
        except OSError:
            return False
        if b"moov" in middle:
            return True

    return False
