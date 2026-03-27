"""Tests for deterministic episode split assignment."""

from __future__ import annotations

import pytest

from src.data.schema import SplitName, SplitPolicy
from src.data.split import EpisodeSplitAssigner


def test_assigner_is_deterministic_for_same_seed() -> None:
    policy = SplitPolicy(train=0.8, val=0.1, test=0.1)
    assigner_a = EpisodeSplitAssigner(policy=policy, seed=7)
    assigner_b = EpisodeSplitAssigner(policy=policy, seed=7)
    episode_id = "ep_000123"

    assert assigner_a.assign(episode_id) == assigner_b.assign(episode_id)


def test_assigner_returns_valid_split_name() -> None:
    assigner = EpisodeSplitAssigner(policy=SplitPolicy(train=0.8, val=0.1, test=0.1), seed=13)

    split = assigner.assign("ep_42")

    assert split in {SplitName.TRAIN, SplitName.VAL, SplitName.TEST}


def test_assign_rejects_empty_episode_id() -> None:
    assigner = EpisodeSplitAssigner(policy=SplitPolicy(train=0.8, val=0.1, test=0.1), seed=0)

    with pytest.raises(ValueError, match="non-empty"):
        assigner.assign(" ")
