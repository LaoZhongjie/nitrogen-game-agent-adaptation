"""Shared JSON config helpers for fine-tuning and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.model.alignment import VocabularyActionAligner


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at: {path}")
    return raw


def normalized_string_mapping(raw: Mapping[str, Any]) -> dict[str, str]:
    """Lowercase/strip keys; strip values (for action lookup tables)."""
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        key_norm = str(key).strip().lower()
        value_norm = str(value).strip()
        if key_norm and value_norm:
            normalized[key_norm] = value_norm
    return normalized


def build_vocab_aligner_from_finetune_config(raw: Mapping[str, Any]) -> VocabularyActionAligner:
    """Build a vocabulary aligner from a finetune-style config dict."""
    mapping_inline = raw.get("action_mapping")
    mapping_path_raw = raw.get("action_mapping_path")
    if mapping_inline is not None and mapping_path_raw is not None:
        raise ValueError("use either action_mapping or action_mapping_path, not both.")
    if mapping_path_raw is not None:
        mapping = normalized_string_mapping(load_json_object(Path(str(mapping_path_raw))))
    else:
        mapping = normalized_string_mapping(mapping_inline if isinstance(mapping_inline, dict) else {})

    aliases_inline = raw.get("action_aliases")
    aliases_path_raw = raw.get("action_aliases_path")
    if aliases_inline is not None and aliases_path_raw is not None:
        raise ValueError("use either action_aliases or action_aliases_path, not both.")
    if aliases_path_raw is not None:
        aliases: dict[str, str] | None = normalized_string_mapping(
            load_json_object(Path(str(aliases_path_raw)))
        )
    elif isinstance(aliases_inline, dict):
        aliases = normalized_string_mapping(aliases_inline)
    else:
        aliases = None

    return VocabularyActionAligner(
        mapping=mapping,
        aliases=aliases,
        unknown_action_id=str(raw.get("unknown_action_id", "unknown")),
        confidence_floor=float(raw.get("confidence_floor", 0.0)),
    )
