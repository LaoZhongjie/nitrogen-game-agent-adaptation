"""Video generation evaluation metrics.

Provides per-frame and pooled distribution metrics (PSNR, SSIM, LPIPS, MAE),
temporal consistency, motion alignment vs reference, optional Inception-based
Fréchet distance (pooled FID) and feature-space mean L2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Pooled FID needs enough frames that covariance estimates are mildly stable;
# still an approximation when feature_dim >> n (regularization applied).
FID_MIN_FRAMES_PER_POOL: int = 48

_inception_model: Optional[object] = None
_inception_device: Optional[object] = None


@dataclass(frozen=True)
class VideoEvalRecord:
    """Evaluation result for a single generated video."""

    chunk_id: str
    game: str
    num_frames_generated: int
    num_frames_reference: int
    fid_per_frame: Optional[float] = None
    lpips_mean: Optional[float] = None
    temporal_consistency: Optional[float] = None
    psnr_mean: Optional[float] = None
    ssim_mean: Optional[float] = None
    mean_mae: Optional[float] = None
    reference_temporal_consistency: Optional[float] = None
    temporal_error_vs_reference: Optional[float] = None


@dataclass(frozen=True)
class VideoEvalSummary:
    """Aggregated metrics across all evaluated videos."""

    total_videos: int
    total_frames_with_reference: int = 0
    mean_fid: Optional[float] = None
    inception_feature_mean_l2: Optional[float] = None
    mean_lpips: Optional[float] = None
    mean_temporal_consistency: Optional[float] = None
    mean_psnr: Optional[float] = None
    mean_ssim: Optional[float] = None
    mean_mae: Optional[float] = None
    mean_reference_temporal_consistency: Optional[float] = None
    mean_temporal_error_vs_reference: Optional[float] = None
    fvd: Optional[float] = None


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


def compute_mean_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean absolute error per pixel, uint8 RGB."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


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


def compute_temporal_error_vs_reference(
    generated_frames: np.ndarray,
    reference_frames: np.ndarray,
    n: int,
) -> float:
    """Mean L1 difference between consecutive-frame deltas (gen vs ref).

    Low values indicate generated motion (frame-to-frame change) matches
    reference motion more closely in absolute pixel space.
    """
    if n < 2:
        raise ValueError("n must be >= 2 for temporal error vs reference.")

    errs: list[float] = []
    for t in range(n - 1):
        dg = generated_frames[t + 1].astype(np.float64) - generated_frames[t].astype(np.float64)
        dr = reference_frames[t + 1].astype(np.float64) - reference_frames[t].astype(np.float64)
        errs.append(float(np.mean(np.abs(dg - dr))))
    return float(np.mean(errs))


def compute_lpips_score(
    gen_frames: np.ndarray,
    ref_frames: np.ndarray,
) -> Optional[float]:
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
    eps_trace_frac: float = 1e-3,
) -> float:
    """Compute FID between two sets of feature vectors.

    Each input has shape ``(N, D)``. Diagonal covariance shrinkage improves
    stability when N is modest relative to D.
    """
    from scipy.linalg import sqrtm

    n_g, d = gen_features.shape
    n_r, d2 = ref_features.shape
    if d != d2:
        raise ValueError("feature dimensions must match.")

    mu_gen = np.mean(gen_features, axis=0)
    mu_ref = np.mean(ref_features, axis=0)

    sigma_gen = np.cov(gen_features, rowvar=False)
    sigma_ref = np.cov(ref_features, rowvar=False)
    if sigma_gen.ndim == 0:
        sigma_gen = np.array([[float(sigma_gen)]])
    if sigma_ref.ndim == 0:
        sigma_ref = np.array([[float(sigma_ref)]])
    if sigma_gen.shape != (d, d):
        sigma_gen = np.atleast_2d(sigma_gen)
    if sigma_ref.shape != (d, d):
        sigma_ref = np.atleast_2d(sigma_ref)

    eye = np.eye(d, dtype=np.float64)
    shrink = eps_trace_frac * (np.trace(sigma_gen) / max(d, 1) + np.trace(sigma_ref) / max(d, 1)) / 2.0
    shrink = max(shrink, 1e-6)
    sigma_gen = sigma_gen.astype(np.float64) + eye * shrink
    sigma_ref = sigma_ref.astype(np.float64) + eye * shrink

    diff = mu_gen - mu_ref
    covmean = sqrtm(sigma_gen @ sigma_ref)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_gen + sigma_ref - 2.0 * covmean))
    return max(fid, 0.0)


def _get_inception_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_inception_model():
    """Lazy-load Inception v3 (pool3 / pre-logits 2048-d)."""
    global _inception_model, _inception_device

    import torch
    import torch.nn as nn
    from torchvision.models import Inception_V3_Weights, inception_v3

    device = _get_inception_device()
    if _inception_model is not None and _inception_device == device:
        return _inception_model, device

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.aux_logits = False
    model.dropout = nn.Identity()
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    _inception_model = model
    _inception_device = device
    return model, device


def extract_inception_features(frames_uint8: np.ndarray, batch_size: int = 16) -> Optional[np.ndarray]:
    """Extract 2048-d Inception features for each frame. RGB uint8 ``(T, H, W, 3)``.

    Preprocessing matches common pytorch-fid style: resize 299, scale ``(x-128)/128``.
    Returns ``None`` if torchvision/torch unavailable.
    """
    if len(frames_uint8) == 0:
        return None

    try:
        import cv2
        import torch
    except ImportError:
        logger.warning("torch or cv2 missing; skipping Inception features.")
        return None

    try:
        model, device = _get_inception_model()
    except Exception as exc:
        logger.warning("Could not load Inception v3 for FID features: %s", exc)
        return None

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frames_uint8), batch_size):
            batch = frames_uint8[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            for i in range(len(batch)):
                img = cv2.resize(batch[i], (299, 299), interpolation=cv2.INTER_LINEAR)
                x = (img.astype(np.float32) - 128.0) / 128.0
                t = torch.from_numpy(x).permute(2, 0, 1)
                tensors.append(t)
            xb = torch.stack(tensors).to(device)
            out = model(xb)
            feats.append(out.detach().float().cpu().numpy())

    return np.concatenate(feats, axis=0)


def compute_pooled_distribution_metrics(
    gen_features: Optional[np.ndarray],
    ref_features: Optional[np.ndarray],
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(pooled_fid, inception_feature_mean_l2)``.

    ``inception_feature_mean_l2`` is ``|| mean(f_gen) - mean(f_ref) ||_2`` (stable with few clips).
    ``pooled_fid`` is only computed when each pool has at least ``FID_MIN_FRAMES_PER_POOL`` rows.
    """
    if gen_features is None or ref_features is None:
        return None, None
    if len(gen_features) == 0 or len(ref_features) == 0:
        return None, None

    mu_g = np.mean(gen_features, axis=0)
    mu_r = np.mean(ref_features, axis=0)
    mean_l2 = float(np.linalg.norm(mu_g - mu_r))

    fid_val: Optional[float] = None
    if (
        len(gen_features) >= FID_MIN_FRAMES_PER_POOL
        and len(ref_features) >= FID_MIN_FRAMES_PER_POOL
    ):
        try:
            fid_val = compute_fid_from_features(gen_features, ref_features)
        except Exception as exc:
            logger.warning("Pooled FID failed (using mean L2 only): %s", exc)
            fid_val = None

    return fid_val, mean_l2


def evaluate_video_pair(
    chunk_id: str,
    game: str,
    generated_frames: np.ndarray,
    reference_frames: Optional[np.ndarray] = None,
) -> VideoEvalRecord:
    """Evaluate a single generated video against an optional reference."""
    num_gen = len(generated_frames)
    num_ref = len(reference_frames) if reference_frames is not None else 0

    tc = compute_temporal_consistency(generated_frames)

    psnr_val = None
    ssim_val = None
    lpips_val = None
    mae_val = None
    ref_tc: Optional[float] = None
    temporal_err: Optional[float] = None

    if reference_frames is not None and num_ref > 0:
        n = min(num_gen, num_ref)
        psnrs = [compute_psnr(generated_frames[i], reference_frames[i]) for i in range(n)]
        ssims = [compute_ssim_simple(generated_frames[i], reference_frames[i]) for i in range(n)]
        maes = [compute_mean_mae(generated_frames[i], reference_frames[i]) for i in range(n)]
        finite_psnrs = [p for p in psnrs if p != float("inf")]
        psnr_val = float(np.mean(finite_psnrs)) if finite_psnrs else float("inf")
        ssim_val = float(np.mean(ssims))
        lpips_val = compute_lpips_score(generated_frames[:n], reference_frames[:n])
        mae_val = float(np.mean(maes))
        ref_tc = compute_temporal_consistency(reference_frames[:n])
        if n >= 2:
            temporal_err = compute_temporal_error_vs_reference(
                generated_frames[:n], reference_frames[:n], n
            )

    return VideoEvalRecord(
        chunk_id=chunk_id,
        game=game,
        num_frames_generated=num_gen,
        num_frames_reference=num_ref,
        lpips_mean=lpips_val,
        temporal_consistency=tc,
        psnr_mean=psnr_val,
        ssim_mean=ssim_val,
        mean_mae=mae_val,
        reference_temporal_consistency=ref_tc,
        temporal_error_vs_reference=temporal_err,
    )


def evaluate_records(
    records: Sequence[VideoEvalRecord],
    *,
    pooled_gen_features: Optional[np.ndarray] = None,
    pooled_ref_features: Optional[np.ndarray] = None,
) -> VideoEvalSummary:
    """Aggregate per-video records; optional pooled Inception features enable FID / mean L2."""
    if not records:
        raise ValueError("records must be non-empty.")

    tc_scores = [r.temporal_consistency for r in records if r.temporal_consistency is not None]
    lpips_scores = [r.lpips_mean for r in records if r.lpips_mean is not None]
    psnr_scores = [
        r.psnr_mean
        for r in records
        if r.psnr_mean is not None and r.psnr_mean != float("inf")
    ]
    ssim_scores = [r.ssim_mean for r in records if r.ssim_mean is not None]
    mae_scores = [r.mean_mae for r in records if r.mean_mae is not None]
    ref_tc_scores = [
        r.reference_temporal_consistency
        for r in records
        if r.reference_temporal_consistency is not None
    ]
    terr_scores = [
        r.temporal_error_vs_reference for r in records if r.temporal_error_vs_reference is not None
    ]

    total_frames_ref = sum(
        r.num_frames_reference for r in records if r.num_frames_reference > 0
    )

    pooled_fid, mean_l2 = compute_pooled_distribution_metrics(
        pooled_gen_features, pooled_ref_features
    )

    return VideoEvalSummary(
        total_videos=len(records),
        total_frames_with_reference=int(total_frames_ref),
        mean_fid=pooled_fid,
        inception_feature_mean_l2=mean_l2,
        mean_temporal_consistency=float(np.mean(tc_scores)) if tc_scores else None,
        mean_lpips=float(np.mean(lpips_scores)) if lpips_scores else None,
        mean_psnr=float(np.mean(psnr_scores)) if psnr_scores else None,
        mean_ssim=float(np.mean(ssim_scores)) if ssim_scores else None,
        mean_mae=float(np.mean(mae_scores)) if mae_scores else None,
        mean_reference_temporal_consistency=float(np.mean(ref_tc_scores)) if ref_tc_scores else None,
        mean_temporal_error_vs_reference=float(np.mean(terr_scores)) if terr_scores else None,
    )

