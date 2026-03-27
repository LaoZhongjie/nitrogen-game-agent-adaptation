"""Deterministic episode-level split assignment."""

from __future__ import annotations

import hashlib

from src.data.schema import SplitName, SplitPolicy


class EpisodeSplitAssigner:
    """Assigns episodes to train/val/test splits deterministically using a hash-based scheme."""

    def __init__(self, policy: SplitPolicy, seed: int) -> None:
        self._policy = policy
        self._seed = seed

    def assign(self, episode_id: str) -> SplitName:
        """Return the split for the given episode, deterministic from episode_id and seed."""
        key = f"{self._seed}:{episode_id}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        value = int(digest, 16) % 10_000 / 10_000

        train_boundary = self._policy.train
        val_boundary = train_boundary + self._policy.val

        if value < train_boundary:
            return SplitName.TRAIN
        if value < val_boundary:
            return SplitName.VAL
        return SplitName.TEST
