from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from app.domain.asset_execution import AssetMediaType
from app.domain.evaluation import EvaluationMediaUnreadableError
from app.services.asset_probe_service import InvalidAssetFileError, probe_media_file
from app.utils.utils import get_ffmpeg_binary, storage_dir

SAMPLING_POLICY_VERSION: str = "media-sampling-v1"
MAX_VIDEO_FRAMES: int = 6


class PreparedMediaObservation(BaseModel):
    """
    Normalized, bounded media observation prepared for multimodal evaluation.
    Freezes extracted frame paths, timestamps, and sampling policy.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    media_type: AssetMediaType
    frame_paths: tuple[str, ...] = Field(min_length=1)
    timestamps: tuple[float, ...] = Field(min_length=1)
    sampling_policy_version: str = SAMPLING_POLICY_VERSION
    asset_file_hash: str


def calculate_sample_timestamps(
    duration_seconds: float,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> tuple[float, ...]:
    """
    Computes deterministic, bounded sample timestamps for a video asset.

    Rules:
    - If duration <= 0.0s: returns (0.0,)
    - If duration <= 1.0s: returns (0.0,) if <= 0.05s, else (0.0, duration - 0.05s)
    - If duration > 1.0s: computes uniform intermediate timestamps bounded by max_frames (<= 6).
    - Guarantees: strictly monotonic, no duplicate timestamps, bounded count <= max_frames.
    """
    if duration_seconds <= 0.0:
        return (0.0,)

    if duration_seconds <= 1.0:
        if duration_seconds <= 0.05:
            return (0.0,)
        t_end = round(duration_seconds - 0.05, 3)
        return (0.0, max(0.01, t_end))

    # Duration > 1.0s: sample between 3 and max_frames
    n_frames = min(max_frames, max(3, int(duration_seconds) + 1))
    t_end = max(0.0, duration_seconds - 0.05)
    step = t_end / (n_frames - 1)

    raw_timestamps = [round(i * step, 3) for i in range(n_frames)]
    # Deduplicate while preserving order
    deduped = tuple(dict.fromkeys(raw_timestamps).keys())
    return deduped


def prepare_media_observation(
    file_path: str | Path,
    media_type: AssetMediaType | None = None,
    output_dir: str | None = None,
) -> PreparedMediaObservation:
    """
    Prepares and validates media observations for multimodal evaluation.

    - For IMAGE: returns direct reference to file with timestamp (0.0,).
    - For VIDEO: extracts deterministic frame samples via FFmpeg (capped at 6).
    - Raises EvaluationMediaUnreadableError on missing, empty, or corrupt files.
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise EvaluationMediaUnreadableError(f"Asset media file does not exist: {p}")

    try:
        probe = probe_media_file(p, media_type=media_type)
    except (InvalidAssetFileError, OSError, ValueError) as exc:
        raise EvaluationMediaUnreadableError(
            f"Media file cannot be read or probed ({p}): {exc}"
        ) from exc

    if probe.media_type == AssetMediaType.IMAGE:
        # Verify image is readable
        try:
            with Image.open(p) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise EvaluationMediaUnreadableError(
                f"Image file cannot be decoded: {p} ({exc})"
            ) from exc

        return PreparedMediaObservation(
            media_type=AssetMediaType.IMAGE,
            frame_paths=(str(p),),
            timestamps=(0.0,),
            sampling_policy_version=SAMPLING_POLICY_VERSION,
            asset_file_hash=probe.file_hash,
        )

    if probe.media_type == AssetMediaType.VIDEO:
        frames_dir = output_dir or storage_dir("evaluation_cache/frames", create=True)
        os.makedirs(frames_dir, exist_ok=True)

        timestamps = calculate_sample_timestamps(probe.duration_seconds, max_frames=MAX_VIDEO_FRAMES)
        ffmpeg_bin = get_ffmpeg_binary()

        extracted_paths: list[str] = []
        for ts in timestamps:
            frame_filename = f"{probe.file_hash}_{ts:.3f}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)

            if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{ts:.3f}",
                    "-i",
                    str(p),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_path,
                ]
                try:
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                    if res.returncode != 0 or not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
                        raise EvaluationMediaUnreadableError(
                            f"FFmpeg failed to extract frame at {ts:.3f}s from video {p}: "
                            f"{res.stderr.decode('utf-8', errors='ignore')[:300]}"
                        )
                except subprocess.TimeoutExpired as exc:
                    raise EvaluationMediaUnreadableError(
                        f"FFmpeg timed out extracting frame from {p}"
                    ) from exc
                except FileNotFoundError as exc:
                    raise EvaluationMediaUnreadableError(
                        f"FFmpeg binary '{ffmpeg_bin}' not found: {exc}"
                    ) from exc

            # Verify extracted frame is valid image
            try:
                with Image.open(frame_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise EvaluationMediaUnreadableError(
                    f"Extracted frame at {ts:.3f}s is unreadable: {frame_path} ({exc})"
                ) from exc

            extracted_paths.append(frame_path)

        return PreparedMediaObservation(
            media_type=AssetMediaType.VIDEO,
            frame_paths=tuple(extracted_paths),
            timestamps=timestamps,
            sampling_policy_version=SAMPLING_POLICY_VERSION,
            asset_file_hash=probe.file_hash,
        )

    raise EvaluationMediaUnreadableError(
        f"Unsupported media type for evaluation: {probe.media_type}"
    )
