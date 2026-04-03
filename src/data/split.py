"""Deterministic train/val/test assignment for NitroGen chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from src.data.schema import SplitName, SplitPolicy


@dataclass(frozen=True)
class VideoSplitAssigner:
    """Assign a deterministic split via SHA-256 hashing.

    ``granularity``:

    - ``video``: one split per ``video_id`` (avoids same video in train and val).
    - ``chunk``: one split per ``(video_id, chunk_id)`` — better spread when only a
      few videos are downloaded (e.g. smoke tests); same video may appear in multiple splits.
    """

    policy: SplitPolicy
    seed: int = 0
    granularity: str = "video"

    def __post_init__(self) -> None:
        if self.granularity not in ("video", "chunk"):
            raise ValueError("granularity must be 'video' or 'chunk'.")

    def assign(self, video_id: str, chunk_id: Optional[str] = None) -> SplitName:
        """Return a stable split assignment for the given ids."""
        if not video_id.strip():
            raise ValueError("video_id must be non-empty.")

        if self.granularity == "chunk":
            if chunk_id is None or not str(chunk_id).strip():
                raise ValueError("chunk_id is required when granularity is 'chunk'.")
            key = f"{video_id}:{chunk_id}"
        else:
            key = video_id

        score = self._stable_score(key)
        train_cutoff = self.policy.train
        val_cutoff = train_cutoff + self.policy.val

        if score < train_cutoff:
            return SplitName.TRAIN
        if score < val_cutoff:
            return SplitName.VAL
        return SplitName.TEST

    def _stable_score(self, key: str) -> float:
        """Map (seed, key) to a deterministic score in [0.0, 1.0)."""
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return bucket / float(1 << 64)
