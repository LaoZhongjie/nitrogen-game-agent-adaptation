"""Deterministic video-level dataset split assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.data.schema import SplitName, SplitPolicy


@dataclass(frozen=True)
class VideoSplitAssigner:
    """Assign a deterministic split to each video ID via SHA-256 hashing."""

    policy: SplitPolicy
    seed: int = 0

    def assign(self, video_id: str) -> SplitName:
        """Return a stable split assignment for ``video_id``."""
        if not video_id.strip():
            raise ValueError("video_id must be non-empty.")

        score = self._stable_score(video_id)
        train_cutoff = self.policy.train
        val_cutoff = train_cutoff + self.policy.val

        if score < train_cutoff:
            return SplitName.TRAIN
        if score < val_cutoff:
            return SplitName.VAL
        return SplitName.TEST

    def _stable_score(self, video_id: str) -> float:
        """Map (seed, video_id) to a deterministic score in [0.0, 1.0)."""
        digest = hashlib.sha256(f"{self.seed}:{video_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return bucket / float(1 << 64)
