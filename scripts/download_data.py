"""Download and prepare NitroGen dataset from HuggingFace.

Downloads action annotation shards (parquet + metadata) from
``nvidia/NitroGen`` on HuggingFace, and optionally downloads source
videos using URLs found in each chunk's ``metadata.json``.

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
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional, Union

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.data.mp4_validate import is_probably_valid_mp4


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


def _download_video(url: str, output_path: Path, timeout: int = 120) -> bool:
    """Download a video from ``url`` to ``output_path``. Returns success flag.

    Uses a browser-like User-Agent so CDNs are less likely to return HTML.
    After download, validates MP4 structure; corrupt or non-MP4 files are removed.
    """
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
) -> dict[str, int]:
    """Walk a shard directory, copy annotations, optionally download videos."""
    stats = {"chunks_processed": 0, "chunks_skipped": 0, "videos_downloaded": 0, "videos_failed": 0}

    video_dirs = sorted([p for p in shard_dir.iterdir() if p.is_dir()])
    for video_dir in video_dirs:
        chunk_dirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
        for chunk_dir in chunk_dirs:
            if max_chunks is not None and stats["chunks_processed"] >= max_chunks:
                return stats

            meta_path = chunk_dir / "metadata.json"
            if not meta_path.exists():
                stats["chunks_skipped"] += 1
                continue

            with meta_path.open("r", encoding="utf-8") as fp:
                metadata = json.load(fp)

            if game_filter and metadata.get("game", "").lower() != game_filter.lower():
                stats["chunks_skipped"] += 1
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
                video_url = metadata.get("original_video", {}).get("url", "")
                if video_url:
                    video_out = dest / "video.mp4"
                    need_fetch = not video_out.exists() or not is_probably_valid_mp4(video_out)
                    if need_fetch:
                        if video_out.exists():
                            video_out.unlink(missing_ok=True)
                        ok = _download_video(video_url, video_out)
                        if ok:
                            stats["videos_downloaded"] += 1
                        else:
                            stats["videos_failed"] += 1
                    else:
                        stats["videos_downloaded"] += 1

            stats["chunks_processed"] += 1

    return stats


def download_nitrogen(
    output_dir: Union[str, Path],
    shard_indices: Optional[List[int]] = None,
    download_videos: bool = False,
    max_chunks_per_shard: Optional[int] = None,
    game_filter: Optional[str] = None,
    dataset_id: str = "nvidia/NitroGen",
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
