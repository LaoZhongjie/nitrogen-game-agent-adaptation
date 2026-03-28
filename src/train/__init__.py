"""Hugging Face fine-tuning helpers for frame → action classification."""

from src.train.config_io import (
    build_vocab_aligner_from_finetune_config,
    load_json_object,
    normalized_string_mapping,
)
from src.train.hf_finetune import (
    HFFinetuneParams,
    build_label2id,
    collect_aligned_frame_rows,
    manifest_input_root,
    predict_clip_actions,
    resolve_frame_path,
    run_image_classification_finetune,
)

__all__ = [
    "HFFinetuneParams",
    "build_label2id",
    "build_vocab_aligner_from_finetune_config",
    "collect_aligned_frame_rows",
    "load_json_object",
    "manifest_input_root",
    "normalized_string_mapping",
    "predict_clip_actions",
    "resolve_frame_path",
    "run_image_classification_finetune",
]
