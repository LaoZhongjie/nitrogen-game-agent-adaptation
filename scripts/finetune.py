"""Config-driven fine-tuning entry point with dry-run validation.

This Stage 3 starter script intentionally focuses on integration contracts:
- load manifest samples via ``ManifestDataset``
- align raw action labels into schema-level ``ActionLabel`` records
- emit deterministic summary/checkpoint-metadata outputs in dry-run mode
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Support both:
# - python -m scripts.finetune (recommended)
# - python scripts/finetune.py (common)
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.loader import ManifestDataset
from src.data.schema import SplitName
from src.model.alignment import VocabularyActionAligner, align_manifest_sample


def _normalized_mapping(raw: Mapping[str, Any]) -> dict[str, str]:
    """Normalize mapping keys/values for case-insensitive action lookup."""
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        key_norm = str(key).strip().lower()
        value_norm = str(value).strip()
        if key_norm and value_norm:
            normalized[key_norm] = value_norm
    return normalized


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at: {path}")
    return raw


@dataclass(slots=True, frozen=True)
class FineTuneConfig:
    """Config for Stage 3 fine-tuning entry-point execution."""

    manifest_path: str
    output_dir: str
    split: SplitName | None = None
    max_samples: int | None = None
    dry_run: bool = True
    action_mapping: Mapping[str, str] = field(default_factory=dict)
    action_aliases: Mapping[str, str] | None = None
    unknown_action_id: str = "unknown"
    confidence_floor: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.manifest_path.strip():
            raise ValueError("manifest_path must be non-empty.")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided.")
        if not self.unknown_action_id.strip():
            raise ValueError("unknown_action_id must be non-empty.")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in range [0.0, 1.0].")


def load_config(config_path: Path, dry_run_override: bool | None = None) -> FineTuneConfig:
    """Load typed fine-tuning config from a JSON file."""
    raw = _load_json_object(config_path)

    split_raw = raw.get("split")
    split = None if split_raw is None else SplitName(str(split_raw))

    mapping_inline = raw.get("action_mapping")
    mapping_path_raw = raw.get("action_mapping_path")
    if mapping_inline is not None and mapping_path_raw is not None:
        raise ValueError("use either action_mapping or action_mapping_path, not both.")
    if mapping_path_raw is not None:
        mapping = _normalized_mapping(_load_json_object(Path(str(mapping_path_raw))))
    else:
        mapping = _normalized_mapping(mapping_inline if isinstance(mapping_inline, dict) else {})

    aliases_inline = raw.get("action_aliases")
    aliases_path_raw = raw.get("action_aliases_path")
    if aliases_inline is not None and aliases_path_raw is not None:
        raise ValueError("use either action_aliases or action_aliases_path, not both.")
    if aliases_path_raw is not None:
        aliases: dict[str, str] | None = _normalized_mapping(_load_json_object(Path(str(aliases_path_raw))))
    elif isinstance(aliases_inline, dict):
        aliases = _normalized_mapping(aliases_inline)
    else:
        aliases = None

    config = FineTuneConfig(
        manifest_path=str(raw["manifest_path"]),
        output_dir=str(raw["output_dir"]),
        split=split,
        max_samples=int(raw["max_samples"]) if raw.get("max_samples") is not None else None,
        dry_run=bool(raw.get("dry_run", True)),
        action_mapping=mapping,
        action_aliases=aliases,
        unknown_action_id=str(raw.get("unknown_action_id", "unknown")),
        confidence_floor=float(raw.get("confidence_floor", 0.0)),
        seed=int(raw.get("seed", 0)),
    )
    if dry_run_override is not None:
        return FineTuneConfig(
            manifest_path=config.manifest_path,
            output_dir=config.output_dir,
            split=config.split,
            max_samples=config.max_samples,
            dry_run=dry_run_override,
            action_mapping=config.action_mapping,
            action_aliases=config.action_aliases,
            unknown_action_id=config.unknown_action_id,
            confidence_floor=config.confidence_floor,
            seed=config.seed,
        )
    return config


def run_finetune(config: FineTuneConfig) -> dict[str, Any]:
    """Run fine-tuning entry flow; currently dry-run only."""
    if not config.dry_run:
        raise NotImplementedError("Training loop is not implemented yet. Run with dry_run=true.")

    dataset = ManifestDataset(config.manifest_path, split=config.split)
    aligner = VocabularyActionAligner(
        mapping=config.action_mapping,
        aliases=config.action_aliases,
        unknown_action_id=config.unknown_action_id,
        confidence_floor=config.confidence_floor,
    )

    processed_samples = 0
    total_action_labels = 0
    unknown_action_labels = 0

    for sample in dataset:
        if config.max_samples is not None and processed_samples >= config.max_samples:
            break
        aligned = align_manifest_sample(sample=sample, aligner=aligner)
        processed_samples += 1
        total_action_labels += len(aligned.action_labels)
        unknown_action_labels += sum(1 for label in aligned.action_labels if label.action_id == config.unknown_action_id)

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoints" / "latest.ckpt"
    metrics_path = output_dir / "metrics" / "latest_metrics.json"
    unknown_ratio = (unknown_action_labels / float(total_action_labels)) if total_action_labels > 0 else 0.0

    return {
        "mode": "dry_run",
        "manifest_path": config.manifest_path,
        "output_dir": str(output_dir),
        "split": None if config.split is None else config.split.value,
        "processed_samples": processed_samples,
        "total_action_labels": total_action_labels,
        "unknown_action_labels": unknown_action_labels,
        "unknown_ratio": unknown_ratio,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "seed": config.seed,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for config-driven fine-tuning entry."""
    parser = argparse.ArgumentParser(description="NitroGen fine-tuning entry point (dry-run starter).")
    parser.add_argument("--config", required=True, help="Path to finetune JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode override.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(
        config_path=Path(args.config),
        dry_run_override=True if bool(args.dry_run) else None,
    )
    summary = run_finetune(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
