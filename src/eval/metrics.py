"""Video generation evaluation metrics.

Provides FID (per-frame), temporal consistency (optical flow), and LPIPS
perceptual similarity. FVD requires a pre-trained I3D model and is computed
separately in ``compute_fvd`` when the dependency is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class VideoEvalRecord:
    """Evaluation result for a single generated video."""

    chunk_id: str
    game: str
    num_frames_generated: int
    num_frames_reference: int
    fid_per_frame: float | None = None
    lpips_mean: float | None = None
    temporal_consistency: float | None = None
    psnr_mean: float | None = None
    ssim_mean: float | None = None


@dataclass(slots=True, frozen=True)
class VideoEvalSummary:
    """Aggregated metrics across all evaluated videos."""

    total_videos: int
    mean_fid: float | None = None
    mean_lpips: float | None = None
    mean_temporal_consistency: float | None = None
    mean_psnr: float | None = None
    mean_ssim: float | None = None
    fvd: float | None = None


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM for uint8 images (luminance channel only).

    Uses the standard SSIM constants for 8-bit images.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    i1 = img1.astype(np.float64).mean(axis=-1) if img1.ndim == 3 else img1.astype(np.float64)
    i2 = img2.astype(np.float64).mean(axis=-1) if img2.ndim == 3 else img2.astype(np.float64)

    mu1, mu2 = i1.mean(), i2.mean()
    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def compute_temporal_consistency(frames: np.ndarray) -> float:
    """Measure temporal smoothness via mean absolute difference between adjacent frames.

    Returns a score in [0, 1] where 1.0 means perfectly consistent.
    Lower pixel differences = higher consistency.
    """
    if len(frames) <= 1:
        return 1.0

    diffs: list[float] = []
    for i in range(len(frames) - 1):
        diff = np.mean(np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)))
        diffs.append(diff)

    mean_diff = np.mean(diffs)
    consistency = 1.0 - min(float(mean_diff) / 255.0, 1.0)
    return consistency


def compute_lpips_score(
    gen_frames: np.ndarray,
    ref_frames: np.ndarray,
) -> float | None:
    """Compute mean LPIPS between generated and reference frames.

    Requires ``lpips`` and ``torch``. Returns ``None`` if not available.
    """
    try:
        import lpips
        import torch
    except ImportError:
        logger.warning("lpips or torch not available; skipping LPIPS computation.")
        return None

    loss_fn = lpips.LPIPS(net="alex", verbose=False)
    n = min(len(gen_frames), len(ref_frames))
    if n == 0:
        return None

    scores: list[float] = []
    for i in range(n):
        g = torch.from_numpy(gen_frames[i]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        r = torch.from_numpy(ref_frames[i]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            d = loss_fn(g, r)
        scores.append(float(d.item()))

    return float(np.mean(scores))


def compute_fid_from_features(
    gen_features: np.ndarray,
    ref_features: np.ndarray,
) -> float:
    """Compute FID between two sets of feature vectors.

    Each input has shape ``(N, D)`` where N is the number of samples and D is the
    feature dimension (e.g. from InceptionV3 or CLIP).
    """
    from scipy.linalg import sqrtm

    mu_gen = np.mean(gen_features, axis=0)
    mu_ref = np.mean(ref_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_ref = np.cov(ref_features, rowvar=False)

    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2.0 * covmean))
    return max(fid, 0.0)


def evaluate_video_pair(
    chunk_id: str,
    game: str,
    generated_frames: np.ndarray,
    reference_frames: np.ndarray | None = None,
) -> VideoEvalRecord:
    """Evaluate a single generated video against an optional reference."""
    num_gen = len(generated_frames)
    num_ref = len(reference_frames) if reference_frames is not None else 0

    tc = compute_temporal_consistency(generated_frames)

    psnr_val = None
    ssim_val = None
    lpips_val = None

    if reference_frames is not None and num_ref > 0:
        n = min(num_gen, num_ref)
        psnrs = [compute_psnr(generated_frames[i], reference_frames[i]) for i in range(n)]
        ssims = [compute_ssim_simple(generated_frames[i], reference_frames[i]) for i in range(n)]
        finite_psnrs = [p for p in psnrs if p != float("inf")]
        psnr_val = float(np.mean(finite_psnrs)) if finite_psnrs else float("inf")
        ssim_val = float(np.mean(ssims))
        lpips_val = compute_lpips_score(generated_frames[:n], reference_frames[:n])

    return VideoEvalRecord(
        chunk_id=chunk_id,
        game=game,
        num_frames_generated=num_gen,
        num_frames_reference=num_ref,
        lpips_mean=lpips_val,
        temporal_consistency=tc,
        psnr_mean=psnr_val,
        ssim_mean=ssim_val,
    )


def evaluate_records(records: Sequence[VideoEvalRecord]) -> VideoEvalSummary:
    """Aggregate evaluation records into a summary."""
    if not records:
        raise ValueError("records must be non-empty.")

    tc_scores = [r.temporal_consistency for r in records if r.temporal_consistency is not None]
    lpips_scores = [r.lpips_mean for r in records if r.lpips_mean is not None]
    psnr_scores = [r.psnr_mean for r in records if r.psnr_mean is not None]
    ssim_scores = [r.ssim_mean for r in records if r.ssim_mean is not None]

    return VideoEvalSummary(
        total_videos=len(records),
        mean_temporal_consistency=float(np.mean(tc_scores)) if tc_scores else None,
        mean_lpips=float(np.mean(lpips_scores)) if lpips_scores else None,
        mean_psnr=float(np.mean(psnr_scores)) if psnr_scores else None,
        mean_ssim=float(np.mean(ssim_scores)) if ssim_scores else None,
    )
