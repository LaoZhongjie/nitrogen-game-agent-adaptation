"""Hugging Face image-classification fine-tuning: frames to canonical action IDs."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from src.data.loader import ManifestDataset, ManifestSample
from src.data.schema import SplitName
from src.model.alignment import ActionAligner, align_manifest_sample


def manifest_input_root(manifest_path: Path) -> Path:
    """Return ``input_root`` from a clip manifest (paths are relative to this)."""
    with manifest_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict) or "input_root" not in raw:
        raise ValueError("manifest must be an object with input_root.")
    root = Path(str(raw["input_root"]))
    if not root.is_dir():
        raise ValueError(f"manifest input_root is not a directory: {root}")
    return root


def resolve_frame_path(*, input_root: Path, frame_path: str) -> Path:
    """Resolve a manifest frame path (relative POSIX) to an absolute path."""
    rel = Path(frame_path)
    if rel.is_absolute():
        return rel
    return (input_root / rel).resolve()


@dataclass(slots=True, frozen=True)
class FrameLabelRow:
    """Single supervised frame with canonical action id string."""

    image_path: Path
    action_id: str


def collect_aligned_frame_rows(
    *,
    manifest_path: Path,
    split: SplitName,
    aligner: ActionAligner,
    max_samples: int | None = None,
) -> list[FrameLabelRow]:
    """Expand manifest clips into per-frame rows with aligned action IDs."""
    dataset = ManifestDataset(manifest_path, split=split)
    input_root = manifest_input_root(manifest_path)
    rows: list[FrameLabelRow] = []
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        aligned = align_manifest_sample(sample=sample, aligner=aligner)
        for fp, label in zip(aligned.frame_paths, aligned.action_labels, strict=True):
            rows.append(
                FrameLabelRow(
                    image_path=resolve_frame_path(input_root=input_root, frame_path=fp),
                    action_id=label.action_id,
                )
            )
    return rows


def build_label2id(action_ids: Sequence[str], unknown_action_id: str) -> dict[str, int]:
    """Deterministic label mapping; always includes ``unknown_action_id`` for OOV at eval."""
    unique = sorted(set(action_ids))
    if len(unique) == 0:
        raise ValueError("cannot build label2id from empty action id set.")
    if unknown_action_id not in unique:
        unique = sorted(unique + [unknown_action_id])
    return {aid: i for i, aid in enumerate(unique)}


class FrameClassificationDataset(Dataset[dict[str, Any]]):
    """Loads RGB images and integer class labels for ViT-style classifiers."""

    def __init__(
        self,
        rows: Sequence[FrameLabelRow],
        processor: Any,
        label2id: dict[str, int],
        unknown_action_id: str,
    ) -> None:
        self._rows = list(rows)
        self._processor = processor
        self._label2id = label2id
        self._unknown_action_id = unknown_action_id
        unk_idx = label2id.get(unknown_action_id)
        self._unk_idx: int | None = unk_idx if unk_idx is not None else None

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        if not row.image_path.is_file():
            raise FileNotFoundError(f"missing frame image: {row.image_path}")
        image = Image.open(row.image_path).convert("RGB")
        enc = self._processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)
        label_id = self._label2id.get(row.action_id)
        if label_id is None:
            if self._unk_idx is None:
                raise ValueError(
                    f"action_id {row.action_id!r} not in label2id and no class for unknown_action_id."
                )
            label_id = self._unk_idx
        return {"pixel_values": pixel_values, "labels": torch.tensor(label_id, dtype=torch.long)}


def _default_compute_metrics() -> Callable[[Any], dict[str, float]]:
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean()) if len(labels) > 0 else 0.0
        return {"accuracy": acc}

    return compute_metrics


@dataclass(slots=True, frozen=True)
class HFFinetuneParams:
    """Hyperparameters for HF Trainer."""

    model_id: str
    output_dir: Path
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    seed: int
    logging_steps: int


def run_image_classification_finetune(
    *,
    train_rows: Sequence[FrameLabelRow],
    eval_rows: Sequence[FrameLabelRow] | None,
    label2id: dict[str, int],
    unknown_action_id: str,
    params: HFFinetuneParams,
) -> Trainer:
    """Fine-tune a Hugging Face image classifier; saves model under ``output_dir / hf_model``."""
    processor = AutoImageProcessor.from_pretrained(params.model_id)
    id2label_int = {int(idx): aid for aid, idx in label2id.items()}
    model = AutoModelForImageClassification.from_pretrained(
        params.model_id,
        num_labels=len(label2id),
        id2label=id2label_int,
        label2id=dict(label2id),
        ignore_mismatched_sizes=True,
    )

    train_ds = FrameClassificationDataset(
        rows=train_rows,
        processor=processor,
        label2id=label2id,
        unknown_action_id=unknown_action_id,
    )
    eval_ds: FrameClassificationDataset | None = None
    if eval_rows is not None and len(eval_rows) > 0:
        eval_ds = FrameClassificationDataset(
            rows=eval_rows,
            processor=processor,
            label2id=label2id,
            unknown_action_id=unknown_action_id,
        )

    model_out = params.output_dir / "hf_model"
    model_out.mkdir(parents=True, exist_ok=True)

    has_eval = eval_ds is not None
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    training_args_kw: dict[str, Any] = {
        "output_dir": str(model_out),
        "num_train_epochs": params.num_train_epochs,
        "per_device_train_batch_size": params.per_device_train_batch_size,
        "per_device_eval_batch_size": params.per_device_eval_batch_size,
        "learning_rate": params.learning_rate,
        "seed": params.seed,
        "logging_steps": params.logging_steps,
        eval_key: "epoch" if has_eval else "no",
        "save_strategy": "epoch" if has_eval else "no",
        "save_total_limit": 2 if has_eval else 1,
        "report_to": "none",
    }
    if has_eval:
        training_args_kw["load_best_model_at_end"] = True
        training_args_kw["metric_for_best_model"] = "accuracy"
        training_args_kw["greater_is_better"] = True
    else:
        training_args_kw["load_best_model_at_end"] = False
    training_args = TrainingArguments(**training_args_kw)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_default_compute_metrics() if eval_ds is not None else None,
    )
    trainer.train()
    trainer.save_model(str(model_out))
    processor.save_pretrained(str(model_out))
    with (params.output_dir / "label2id.json").open("w", encoding="utf-8") as fp:
        json.dump(label2id, fp, indent=2)
        fp.write("\n")
    return trainer


def predict_clip_actions(
    *,
    model_dir: Path,
    samples: Sequence[ManifestSample],
    input_root: Path,
    aligner: ActionAligner,
    batch_size: int,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Run inference for clips; returns dicts with clip_id and predicted_action_ids."""
    label2id_path = model_dir / "label2id.json"
    if not label2id_path.is_file():
        raise FileNotFoundError(f"missing label2id.json under {model_dir}")
    with label2id_path.open("r", encoding="utf-8") as fp:
        label2id: dict[str, int] = json.load(fp)
    id2label = {v: k for k, v in label2id.items()}

    hf_dir = model_dir / "hf_model"
    processor = AutoImageProcessor.from_pretrained(str(hf_dir))
    model = AutoModelForImageClassification.from_pretrained(str(hf_dir))
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        aligned = align_manifest_sample(sample=sample, aligner=aligner)
        frame_paths = [resolve_frame_path(input_root=input_root, frame_path=fp) for fp in aligned.frame_paths]
        pred_ids: list[str] = []
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            images = []
            for p in batch_paths:
                if not p.is_file():
                    raise FileNotFoundError(f"missing frame image: {p}")
                images.append(Image.open(p).convert("RGB"))
            enc = processor(images=images, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            pred_idx = logits.argmax(dim=-1).cpu().tolist()
            pred_ids.extend(id2label[int(i)] for i in pred_idx)
        predictions.append({"clip_id": sample.clip_id, "predicted_action_ids": pred_ids})
    return predictions
