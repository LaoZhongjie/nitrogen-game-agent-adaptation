"""Download and prepare NitroGen dataset from HuggingFace.

Downloads action annotation shards (parquet + metadata) from
``nvidia/NitroGen`` on HuggingFace, and optionally downloads source
videos using URLs found in each chunk's ``metadata.json``.

YouTube / youtu.be URLs require ``yt-dlp`` (see ``requirements.txt``). Merging
separate video+audio streams into one MP4 needs ``ffmpeg`` on ``PATH``.

NitroGen ``metadata.json`` provides ``original_video.start_time`` / ``end_time``;
the downloader uses them so yt-dlp fetches only that segment (much faster than
the full upload). Set ``NITROGEN_YTDLP_VERBOSE=1`` to print yt-dlp progress.

Usage (from repo root)::

    python3.12 -m scripts.download_data \\
        --output data/nitrogen \\
        --shards 0 1 2 \\
        --download-videos \\
        --max-chunks-per-shard 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Union

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from urllib.parse import urlparse

from src.data.mp4_validate import is_probably_valid_mp4


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _needs_ytdlp(url: str) -> bool:
    """True for hosts where a plain HTTP GET does not return raw media (e.g. YouTube)."""
    host = _url_host(url)
    if not host:
        return False
    return host == "youtu.be" or host.endswith(".youtube.com") or host == "youtube.com"


def _cleanup_stem_outputs(stem: Path) -> None:
    """Remove files left by yt-dlp for template ``stem.%(ext)s``."""
    parent = stem.parent
    prefix = stem.name
    if not parent.is_dir():
        return
    for p in parent.iterdir():
        if p.is_file() and p.name.startswith(prefix + ".") and p.suffix.lower() != ".part":
            p.unlink(missing_ok=True)


def _ytdlp_verbose() -> bool:
    """If true, show yt-dlp progress (env ``NITROGEN_YTDLP_VERBOSE=1``)."""
    return os.environ.get("NITROGEN_YTDLP_VERBOSE", "").strip() in ("1", "true", "yes")


def _safe_metadata_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_range_seconds(start: Optional[float], end: Optional[float]) -> Optional[tuple[float, float]]:
    """Return ``(start, end)`` for yt-dlp ``download_ranges`` if metadata looks sane."""
    if start is None or end is None:
        return None
    try:
        s = float(start)
        e = float(end)
    except (TypeError, ValueError):
        return None
    if e <= s or (e - s) < 0.05:
        return None
    return (s, e)


def _download_video_ytdlp(
    url: str,
    output_path: Path,
    segment: Optional[tuple[float, float]] = None,
) -> bool:
    """Download via yt-dlp (required for YouTube watch URLs). Writes validated ``output_path``.

    If ``segment`` is ``(start_sec, end_sec)``, only that interval is fetched (NitroGen chunk
    window), which is much faster than downloading the full source video.
    """
    try:
        import yt_dlp  # type: ignore[import-untyped]
    except ImportError:
        print(
            "  [WARN] YouTube (or similar) URL requires yt-dlp. Install: pip install yt-dlp",
        )
        return False

    try:
        from yt_dlp.utils import download_range_func  # type: ignore[import-untyped]
    except ImportError:
        download_range_func = None  # type: ignore[assignment,misc]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    stem = output_path.with_suffix("")
    outtmpl = str(stem) + ".%(ext)s"

    verbose = _ytdlp_verbose()
    base_opts: dict[str, object] = {
        "outtmpl": outtmpl,
        "quiet": not verbose,
        "no_warnings": not verbose,
        "noprogress": not verbose,
        "overwrites": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 120,
    }

    format_with_merge = (
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/"
        "best[ext=mp4]/best"
    )
    format_no_merge = "best[ext=mp4]/best"

    def _run(fmt: str, merge_mp4: bool, use_segment: bool) -> None:
        opts: dict[str, object] = {**base_opts, "format": fmt}
        if merge_mp4:
            opts["merge_output_format"] = "mp4"
        if use_segment and segment is not None and download_range_func is not None:
            opts["download_ranges"] = download_range_func(
                None,
                [(float(segment[0]), float(segment[1]))],
            )
            opts["force_keyframes_at_cuts"] = True
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    def _finalize_downloaded_file() -> bool:
        candidates = sorted(
            stem.parent.glob(stem.name + ".*"),
            key=lambda p: p.stat().st_mtime if p.is_file() else 0,
            reverse=True,
        )
        for cand in candidates:
            if not cand.is_file() or cand.name.endswith(".part"):
                continue
            if is_probably_valid_mp4(cand):
                if cand.resolve() != output_path.resolve():
                    cand.replace(output_path)
                return True
        for cand in candidates:
            if cand.is_file() and not cand.name.endswith(".part"):
                cand.unlink(missing_ok=True)
        return False

    # Prefer time-range download when metadata provides a valid window (typical NitroGen chunk).
    segment_attempts: tuple[bool, ...]
    if segment is not None and download_range_func is not None:
        segment_attempts = (True, False)
    else:
        if segment is not None and download_range_func is None:
            print(
                "  [WARN] yt-dlp is too old (no download_range_func); "
                "downloading full source video — upgrade: pip install -U yt-dlp",
                flush=True,
            )
        segment_attempts = (False,)

    last_exc: Optional[Exception] = None
    for use_segment in segment_attempts:
        for merge_mp4, fmt in ((True, format_with_merge), (False, format_no_merge)):
            _cleanup_stem_outputs(stem)
            try:
                _run(fmt, merge_mp4=merge_mp4, use_segment=use_segment)
            except Exception as exc:
                last_exc = exc
                continue
            if _finalize_downloaded_file():
                return True

        if use_segment and segment is not None:
            print(
                f"  [WARN] yt-dlp segment download failed; retrying full video "
                f"({segment[0]:.1f}–{segment[1]:.1f}s requested): {url[:70]}...",
                flush=True,
            )

    if last_exc is not None:
        print(f"  [WARN] yt-dlp failed for {url[:80]}...: {last_exc}")
    else:
        print(
            f"  [WARN] yt-dlp finished but output was not a valid MP4 "
            f"(install ffmpeg for merge if missing): {url[:80]}...",
        )
    _cleanup_stem_outputs(stem)
    return False


def _download_shard_archive(
    dataset_id: str,
    shard_idx: int,
    cache_dir: Path,
) -> Path:
    """Download a single shard tar.gz from HuggingFace and return local path."""
    from huggingface_hub import hf_hub_download

    shard_name = f"SHARD_{shard_idx:04d}"
    filename = f"actions/{shard_name}.tar.gz"
    local_path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local_path)


def _extract_shard(archive_path: Path, output_dir: Path) -> Path:
    """Extract a shard archive into ``output_dir`` and return shard root."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=str(output_dir))
    shard_dirs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("SHARD_")]
    )
    if not shard_dirs:
        return output_dir
    return shard_dirs[-1]


def _download_video(
    url: str,
    output_path: Path,
    timeout: int = 120,
    segment_start: Optional[float] = None,
    segment_end: Optional[float] = None,
) -> bool:
    """Download a video from ``url`` to ``output_path``. Returns success flag.

    YouTube / youtu.be links use ``yt-dlp`` (merged MP4 when ffmpeg is available).
    Optional ``segment_start`` / ``segment_end`` (seconds in source video) limit the
    download to that window when using yt-dlp (NitroGen ``original_video`` times).
    Other URLs use HTTP GET with a browser-like User-Agent.
    After download, validates MP4 structure; corrupt or non-MP4 files are removed.
    """
    if _needs_ytdlp(url):
        seg = _segment_range_seconds(segment_start, segment_end)
        return _download_video_ytdlp(url, output_path, segment=seg)

    import urllib.error
    import urllib.request

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NitroGenPipeline/1.0; "
                    "+https://github.com/)"
                ),
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        output_path.write_bytes(data)
        if not is_probably_valid_mp4(output_path):
            output_path.unlink(missing_ok=True)
            print(f"  [WARN] Download was not a valid MP4 (removed): {url[:80]}...")
            return False
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  [WARN] Failed to download {url}: {exc}")
        output_path.unlink(missing_ok=True)
        return False


def _process_shard(
    shard_dir: Path,
    output_dir: Path,
    download_videos: bool,
    max_chunks: Optional[int],
    game_filter: Optional[str],
    show_progress: bool = True,
) -> dict[str, int]:
    """Walk a shard directory, copy annotations, optionally download videos."""
    stats = {"chunks_processed": 0, "chunks_skipped": 0, "videos_downloaded": 0, "videos_failed": 0}

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(
                desc=f"Chunks {shard_dir.name}",
                unit="chunk",
                dynamic_ncols=True,
                miniters=1,
            )
        except ImportError:
            pass

    video_dirs = sorted([p for p in shard_dir.iterdir() if p.is_dir()])
    try:
        for video_dir in video_dirs:
            chunk_dirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
            for chunk_dir in chunk_dirs:
                if max_chunks is not None and stats["chunks_processed"] >= max_chunks:
                    return stats

                meta_path = chunk_dir / "metadata.json"
                if not meta_path.exists():
                    stats["chunks_skipped"] += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue

                with meta_path.open("r", encoding="utf-8") as fp:
                    metadata = json.load(fp)

                if game_filter and metadata.get("game", "").lower() != game_filter.lower():
                    stats["chunks_skipped"] += 1
                    if pbar is not None:
                        pbar.update(1)
                    continue

                chunk_id = chunk_dir.name
                video_id = video_dir.name
                dest = output_dir / shard_dir.name / video_id / chunk_id
                dest.mkdir(parents=True, exist_ok=True)

                for parquet_name in ("actions_raw.parquet", "actions_processed.parquet"):
                    src_parquet = chunk_dir / parquet_name
                    if src_parquet.exists():
                        import shutil
                        shutil.copy2(str(src_parquet), str(dest / parquet_name))

                with (dest / "metadata.json").open("w", encoding="utf-8") as fp:
                    json.dump(metadata, fp, indent=2)

                if download_videos:
                    orig = metadata.get("original_video", {}) or {}
                    video_url = str(orig.get("url", "") or "")
                    if video_url:
                        video_out = dest / "video.mp4"
                        need_fetch = not video_out.exists() or not is_probably_valid_mp4(video_out)
                        if need_fetch:
                            if video_out.exists():
                                video_out.unlink(missing_ok=True)
                            t0 = _safe_metadata_float(orig.get("start_time"))
                            t1 = _safe_metadata_float(orig.get("end_time"))
                            span = _segment_range_seconds(t0, t1)
                            if pbar is not None:
                                pbar.set_postfix(
                                    ok=stats["chunks_processed"],
                                    vid_ok=stats["videos_downloaded"],
                                    vid_fail=stats["videos_failed"],
                                    fetching=chunk_id[:16],
                                    sec=(
                                        f"{span[0]:.0f}-{span[1]:.0f}"
                                        if span
                                        else "full"
                                    ),
                                    refresh=True,
                                )
                            if span:
                                print(
                                    f"  [video] {video_id}/{chunk_id}  "
                                    f"segment {span[0]:.2f}s–{span[1]:.2f}s",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"  [video] {video_id}/{chunk_id}  full source (no valid time window)",
                                    flush=True,
                                )
                            ok = _download_video(
                                video_url,
                                video_out,
                                segment_start=t0,
                                segment_end=t1,
                            )
                            if ok:
                                stats["videos_downloaded"] += 1
                            else:
                                stats["videos_failed"] += 1
                        else:
                            stats["videos_downloaded"] += 1

                stats["chunks_processed"] += 1
                if pbar is not None:
                    pbar.set_postfix(
                        ok=stats["chunks_processed"],
                        vid_ok=stats["videos_downloaded"],
                        vid_fail=stats["videos_failed"],
                        refresh=False,
                    )
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return stats


def download_nitrogen(
    output_dir: Union[str, Path],
    shard_indices: Optional[List[int]] = None,
    download_videos: bool = False,
    max_chunks_per_shard: Optional[int] = None,
    game_filter: Optional[str] = None,
    dataset_id: str = "nvidia/NitroGen",
    show_progress: bool = True,
) -> dict[str, int]:
    """Download and prepare NitroGen shards.

    Returns aggregate statistics.
    """
    from tqdm import tqdm

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_dir = output_path / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if shard_indices is None:
        shard_indices = list(range(36))

    totals: dict[str, int] = {
        "shards": 0,
        "chunks_processed": 0,
        "chunks_skipped": 0,
        "videos_downloaded": 0,
        "videos_failed": 0,
    }

    for shard_idx in tqdm(shard_indices, desc="Downloading shards"):
        shard_name = f"SHARD_{shard_idx:04d}"
        print(f"\n--- Processing {shard_name} ---")

        try:
            archive_path = _download_shard_archive(dataset_id, shard_idx, cache_dir)
        except Exception as exc:
            print(f"  [ERROR] Failed to download {shard_name}: {exc}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shard_root = _extract_shard(archive_path, tmp_path)
            stats = _process_shard(
                shard_dir=shard_root,
                output_dir=output_path,
                download_videos=download_videos,
                max_chunks=max_chunks_per_shard,
                game_filter=game_filter,
                show_progress=show_progress,
            )

        totals["shards"] += 1
        for key in ("chunks_processed", "chunks_skipped", "videos_downloaded", "videos_failed"):
            totals[key] += stats[key]

        print(f"  {shard_name}: {stats}")

    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NitroGen dataset from HuggingFace.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory for prepared data.")
    parser.add_argument(
        "--shards",
        nargs="*",
        type=int,
        default=None,
        help="Shard indices to download (e.g. 0 1 2). Downloads all 36 if omitted.",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Also download source videos from URLs in metadata.",
    )
    parser.add_argument(
        "--max-chunks-per-shard",
        type=int,
        default=None,
        help="Limit chunks processed per shard (useful for quick tests).",
    )
    parser.add_argument(
        "--game-filter",
        type=str,
        default=None,
        help="Only process chunks from this game (case-insensitive).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="nvidia/NitroGen",
        help="HuggingFace dataset repository ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals = download_nitrogen(
        output_dir=args.output,
        shard_indices=args.shards,
        download_videos=args.download_videos,
        max_chunks_per_shard=args.max_chunks_per_shard,
        game_filter=args.game_filter,
        dataset_id=args.dataset_id,
    )
    print(f"\n=== Download complete ===\n{json.dumps(totals, indent=2)}")


if __name__ == "__main__":
    main()
