"""Deterministic episode-level dataset split assignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.data.schema import SplitName, SplitPolicy


@dataclass(slots=True, frozen=True)
class EpisodeSplitAssigner:
    """Assign a deterministic split to each episode ID."""

    policy: SplitPolicy
    seed: int = 0

    def assign(self, episode_id: str) -> SplitName:
        """Return a stable split assignment for ``episode_id``."""
        if not episode_id.strip():
            raise ValueError("episode_id must be non-empty.")

        score = self._stable_score(episode_id)
        train_cutoff = self.policy.train
        val_cutoff = train_cutoff + self.policy.val

        if score < train_cutoff:
            return SplitName.TRAIN
        if score < val_cutoff:
            return SplitName.VAL
        return SplitName.TEST

    def _stable_score(self, episode_id: str) -> float:
        """Map (seed, episode_id) to a deterministic score in [0.0, 1.0)."""
        digest = hashlib.sha256(f"{self.seed}:{episode_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return bucket / float(1 << 64)
