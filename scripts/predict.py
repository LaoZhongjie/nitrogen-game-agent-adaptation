"""Run a fine-tuned HF classifier on a manifest split and write prediction records JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.loader import ManifestDataset
from src.data.schema import SplitName
from src.train.config_io import build_vocab_aligner_from_finetune_config, load_json_object
from src.train.hf_finetune import manifest_input_root, predict_clip_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prediction JSON for offline evaluation (manifest + predictions).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Fine-tune output directory (contains hf_model/ and label2id.json).",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Clip manifest JSON path.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Manifest split to predict on.",
    )
    parser.add_argument(
        "--finetune-config",
        required=True,
        type=Path,
        help="Same finetune JSON used for training (for action alignment).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output predictions JSON path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (frames).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_json_object(args.finetune_config)
    aligner = build_vocab_aligner_from_finetune_config(raw)
    split = SplitName(args.split)
    samples = list(ManifestDataset(args.manifest, split=split))
    if len(samples) == 0:
        raise ValueError(f"no clips for split={split.value} in manifest.")

    input_root = manifest_input_root(args.manifest)
    preds = predict_clip_actions(
        model_dir=args.model_dir,
        samples=samples,
        input_root=input_root,
        aligner=aligner,
        batch_size=int(args.batch_size),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(preds, fp, indent=2)
        fp.write("\n")
    print(json.dumps({"wrote": str(args.output.resolve()), "clip_count": len(preds)}, indent=2))


if __name__ == "__main__":
    main()
