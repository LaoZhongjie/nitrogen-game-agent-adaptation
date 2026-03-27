# Dataset Specification

This document defines the canonical dataset contract for NitroGen post-training adaptation.

## Overview

The dataset is composed of demonstration episodes. Each episode contains one or more clips, and each clip contains an ordered sequence of frames with aligned action labels.

Output artifact from `scripts/build_dataset.py` is a manifest JSON:

- `dataset_name`
- `source_root`
- `episodes` (validated episode records)
- `split_policy` (rules and ratios)
- `split_counts`

## Data Entities

### Episode

An episode is a full demonstration trajectory in a specific game/task context.

Required fields:

- `episode_id` (`str`): globally unique ID, e.g. `ep_000123`.
- `game` (`str`): target game/domain identifier.
- `demonstrator_id` (`str`): source demonstrator identifier.
- `clips` (`list[ClipRecord]`): one or more clips.
- `split` (`SplitName`): one of `train`, `val`, `test`.
- `metadata` (`dict[str, str]`): optional string metadata.

### Clip

A clip is a contiguous chunk within an episode.

Required fields:

- `clip_id` (`str`): unique within an episode.
- `start_frame_idx` (`int`): inclusive start frame index in original episode timeline.
- `end_frame_idx` (`int`): inclusive end frame index in original episode timeline.
- `frames` (`list[FrameRecord]`): ordered frame sequence.
- `action_labels` (`list[ActionLabel]`): action aligned per frame (same length as `frames`).
- `metadata` (`dict[str, str]`): optional string metadata.

Validation constraints:

- `start_frame_idx >= 0`
- `end_frame_idx >= start_frame_idx`
- `len(frames) == len(action_labels)`

### Frame

A frame is a single visual observation reference.

Required fields:

- `frame_idx` (`int`): frame index relative to original episode timeline.
- `frame_path` (`str`): path to frame image or extracted frame file relative to `source_root`.
- `timestamp_sec` (`float`): non-negative timestamp in seconds.

Validation constraints:

- `frame_idx >= 0`
- `timestamp_sec >= 0.0`

### Action Label

Action label aligned to a frame.

Required fields:

- `action_id` (`str`): canonical action vocabulary token, e.g. `move_left`, `jump`, `fire`.
- `action_text` (`str`): optional human-readable label.
- `confidence` (`float`): alignment confidence in `[0.0, 1.0]`.

## Split Policy

`split` assignment is episode-level (never per-frame or per-clip).

Recommended default:

- Train: `0.8`
- Val: `0.1`
- Test: `0.1`

Rules:

- Ratios must sum to `1.0` (within small tolerance).
- Splits are deterministic from a seed.
- All clips/frames of an episode inherit the episode split.
- Test split should contain only held-out episodes.

## Build Config (Input to `scripts/build_dataset.py`)

Example:

```json
{
  "dataset_name": "new_game_small_demos_v1",
  "source_root": "data/raw/new_game_demos",
  "output_manifest_path": "data/processed/new_game_small_demos_v1_manifest.json",
  "seed": 7,
  "split_ratios": {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
  },
  "episodes_file": "data/raw/new_game_demos/episodes.json"
}
```

`episodes_file` is expected to contain a JSON array of raw episode-like dictionaries that conform to the schema fields above (except `split`, which can be omitted and assigned during build).
