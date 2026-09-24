from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.domain.asset_execution import AssetMediaType
from app.domain.evaluation import EvaluationMediaUnreadableError
from app.services.evaluation.media_preparation import (
    PreparedMediaObservation,
    calculate_sample_timestamps,
    prepare_media_observation,
)


def test_calculate_sample_timestamps_zero_and_negative():
    assert calculate_sample_timestamps(0.0) == (0.0,)
    assert calculate_sample_timestamps(-1.5) == (0.0,)


def test_calculate_sample_timestamps_short_video():
    # <= 0.05s
    assert calculate_sample_timestamps(0.04) == (0.0,)

    # 0.5s -> (0.0, 0.45)
    ts = calculate_sample_timestamps(0.5)
    assert len(ts) == 2
    assert ts[0] == 0.0
    assert ts[1] == 0.45

    # 1.0s -> (0.0, 0.95)
    ts = calculate_sample_timestamps(1.0)
    assert len(ts) == 2
    assert ts[0] == 0.0
    assert ts[1] == 0.95


def test_calculate_sample_timestamps_bounded_to_max_six():
    # For a 10s video, max_frames default 6
    ts = calculate_sample_timestamps(10.0)
    assert len(ts) <= 6
    assert ts[0] == 0.0
    assert ts[-1] == 9.95
    # Strictly monotonic
    for i in range(len(ts) - 1):
        assert ts[i] < ts[i + 1]

    # For a 100s video, still max 6
    ts_long = calculate_sample_timestamps(100.0)
    assert len(ts_long) == 6
    assert ts_long[0] == 0.0
    assert ts_long[-1] == 99.95


def test_prepare_media_observation_image(tmp_path: Path):
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(str(img_path))

    obs = prepare_media_observation(img_path)
    assert isinstance(obs, PreparedMediaObservation)
    assert obs.media_type == AssetMediaType.IMAGE
    assert len(obs.frame_paths) == 1
    assert obs.timestamps == (0.0,)
    assert obs.sampling_policy_version == "media-sampling-v1"
    assert len(obs.asset_file_hash) == 64


def test_prepare_media_observation_missing_file():
    with pytest.raises(EvaluationMediaUnreadableError, match="does not exist"):
        prepare_media_observation("d:/non_existent_path_12345.png")


def test_prepare_media_observation_corrupt_image(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE_DATA")

    with pytest.raises(EvaluationMediaUnreadableError):
        prepare_media_observation(corrupt_file)


def test_prepare_media_observation_video(tmp_path: Path):
    # Mock probe and ffmpeg extraction
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy_video_bytes")

    probe_mock = MagicMock(
        file_path=str(dummy_video),
        file_hash="mock_video_hash_12345",
        file_size_bytes=100,
        media_type=AssetMediaType.VIDEO,
        mime_type="video/mp4",
        width=1920,
        height=1080,
        duration_seconds=5.0,
    )

    def fake_ffmpeg_run(cmd, **kwargs):
        # Create a real small image at target path so Image.open succeeds
        out_path = cmd[-1]
        img = Image.new("RGB", (64, 64), color="green")
        img.save(out_path)
        mock_res = MagicMock()
        mock_res.returncode = 0
        return mock_res

    with (
        patch("app.services.evaluation.media_preparation.probe_media_file", return_value=probe_mock),
        patch("subprocess.run", side_effect=fake_ffmpeg_run),
    ):
        obs = prepare_media_observation(dummy_video, output_dir=str(tmp_path / "frames"))
        assert obs.media_type == AssetMediaType.VIDEO
        assert len(obs.frame_paths) == 6
        assert obs.asset_file_hash == "mock_video_hash_12345"
        assert len(obs.timestamps) == 6
        for f in obs.frame_paths:
            assert os.path.exists(f)
