"""Tests for dataset schema contracts."""

from __future__ import annotations

import pytest

from src.data.schema import ActionLabel, ClipRecord, EpisodeRecord, FrameRecord, SplitName, SplitPolicy


def _valid_frame(frame_idx: int) -> FrameRecord:
    return FrameRecord(
        frame_idx=frame_idx,
        frame_path=f"frames/frame_{frame_idx:04d}.png",
        timestamp_sec=float(frame_idx) / 30.0,
    )


def _valid_action(action_id: str) -> ActionLabel:
    return ActionLabel(action_id=action_id, action_text=action_id, confidence=0.9)


def test_clip_record_rejects_mismatched_lengths() -> None:
    frames = (_valid_frame(0), _valid_frame(1))
    actions = (_valid_action("move_left"),)
    with pytest.raises(ValueError, match="equal length"):
        ClipRecord(
            clip_id="clip_001",
            start_frame_idx=0,
            end_frame_idx=1,
            frames=frames,
            action_labels=actions,
        )


def test_action_label_rejects_bad_confidence() -> None:
    with pytest.raises(ValueError, match="range"):
        ActionLabel(action_id="jump", action_text="jump", confidence=1.1)


def test_episode_record_construction() -> None:
    clip = ClipRecord(
        clip_id="clip_001",
        start_frame_idx=0,
        end_frame_idx=1,
        frames=(_valid_frame(0), _valid_frame(1)),
        action_labels=(_valid_action("move_left"), _valid_action("jump")),
    )
    episode = EpisodeRecord(
        episode_id="ep_001",
        game="new_game",
        demonstrator_id="demo_a",
        clips=(clip,),
        split=SplitName.TRAIN,
    )
    assert episode.split is SplitName.TRAIN
    assert episode.clips[0].clip_id == "clip_001"


def test_split_policy_requires_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitPolicy(train=0.7, val=0.2, test=0.2)
