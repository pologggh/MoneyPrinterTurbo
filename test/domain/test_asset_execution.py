from datetime import timedelta
from pathlib import Path

import pytest
from PIL import Image

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    InvalidAttemptStateTransitionError,
    ShotAssetVersion,
)
from app.domain.asset_router import GenerationMode
from app.services.asset_probe_service import (
    InvalidAssetFileError,
    calculate_sha256,
    probe_media_file,
)


def test_execution_run_creation_and_status_aggregation():
    run = ExecutionRun(
        asset_route_plan_id="plan-1",
        storyboard_snapshot_id="snap-1",
        total_shots=3,
    )
    assert run.status == ExecutionStatus.PENDING
    assert run.calculate_aggregated_status() == ExecutionStatus.PENDING

    # 1 succeeded -> RUNNING
    run.succeeded_shots = 1
    assert run.calculate_aggregated_status() == ExecutionStatus.RUNNING

    # 3 succeeded -> SUCCEEDED
    run.succeeded_shots = 3
    assert run.calculate_aggregated_status() == ExecutionStatus.SUCCEEDED

    # 1 succeeded, 2 failed -> PARTIAL
    run.succeeded_shots = 1
    run.failed_shots = 2
    assert run.calculate_aggregated_status() == ExecutionStatus.PARTIAL

    # 0 succeeded, 3 failed -> FAILED
    run.succeeded_shots = 0
    run.failed_shots = 3
    assert run.calculate_aggregated_status() == ExecutionStatus.FAILED


def test_execution_attempt_lifecycle_success():
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="shot-rev-1",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    assert attempt.status == ExecutionAttemptStatus.PENDING

    running_attempt = attempt.mark_running()
    assert running_attempt.status == ExecutionAttemptStatus.RUNNING

    finished_time = running_attempt.started_at + timedelta(seconds=12)
    succeeded = running_attempt.mark_succeeded(
        finished_at=finished_time,
        raw_response={"task_id": "seedance-123"},
    )
    assert succeeded.status == ExecutionAttemptStatus.SUCCEEDED
    assert succeeded.duration_seconds == pytest.approx(12.0)
    assert succeeded.raw_provider_response == {"task_id": "seedance-123"}


def test_execution_attempt_lifecycle_failure():
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="shot-rev-1",
        attempt_number=1,
        provider="wavespeed",
        model="standard",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    running_attempt = attempt.mark_running()

    finished_time = running_attempt.started_at + timedelta(seconds=5)
    failed = running_attempt.mark_failed(
        error_code="PROVIDER_TIMEOUT",
        error_message="Prediction timed out after 300s",
        finished_at=finished_time,
        raw_response={"status": "failed"},
    )
    assert failed.status == ExecutionAttemptStatus.FAILED
    assert failed.error_code == "PROVIDER_TIMEOUT"
    assert failed.error_message == "Prediction timed out after 300s"
    assert failed.duration_seconds == pytest.approx(5.0)


def test_execution_attempt_invalid_transitions():
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="shot-rev-1",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    running = attempt.mark_running()
    succeeded = running.mark_succeeded()

    # Succeeded cannot transition to running or failed
    with pytest.raises(InvalidAttemptStateTransitionError):
        succeeded.mark_running()

    with pytest.raises(InvalidAttemptStateTransitionError):
        succeeded.mark_failed("ERR", "Fail")

    failed = running.mark_failed("ERR", "Fail")
    with pytest.raises(InvalidAttemptStateTransitionError):
        failed.mark_running()

    with pytest.raises(InvalidAttemptStateTransitionError):
        failed.mark_succeeded()


def test_shot_asset_version_validation():
    valid_hash = "a" * 64
    version = ShotAssetVersion(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path="storage/shot_assets/shot-1/v1.mp4",
        file_hash=valid_hash,
        file_size_bytes=1024,
        media_type=AssetMediaType.VIDEO,
        mime_type="video/mp4",
        width=1920,
        height=1080,
        duration_seconds=5.0,
        fps=30.0,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    assert version.file_hash == valid_hash
    assert version.width == 1920

    # Invalid hash length
    with pytest.raises(ValueError, match="file_hash must be a 64-character"):
        ShotAssetVersion(
            shot_id="shot-1",
            shot_revision_id="rev-1",
            execution_attempt_id="att-1",
            file_path="storage/shot_assets/shot-1/v1.mp4",
            file_hash="short_hash",
            file_size_bytes=1024,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            fps=30.0,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

    # Non-hex characters
    with pytest.raises(ValueError, match="file_hash must be a 64-character"):
        ShotAssetVersion(
            shot_id="shot-1",
            shot_revision_id="rev-1",
            execution_attempt_id="att-1",
            file_path="storage/shot_assets/shot-1/v1.mp4",
            file_hash="z" * 64,
            file_size_bytes=1024,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            fps=30.0,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

    # Empty file path
    with pytest.raises(ValueError, match="file_path must not be empty"):
        ShotAssetVersion(
            shot_id="shot-1",
            shot_revision_id="rev-1",
            execution_attempt_id="att-1",
            file_path="   ",
            file_hash=valid_hash,
            file_size_bytes=1024,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            fps=30.0,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )


def test_asset_probe_service_image(tmp_path: Path):
    img_path = tmp_path / "test_shot.png"
    img = Image.new("RGB", (1280, 720), color="blue")
    img.save(str(img_path))

    probe = probe_media_file(img_path)
    assert probe.media_type == AssetMediaType.IMAGE
    assert probe.width == 1280
    assert probe.height == 720
    assert probe.duration_seconds == 0.0
    assert probe.file_size_bytes == img_path.stat().st_size
    assert len(probe.file_hash) == 64
    assert probe.file_hash == calculate_sha256(img_path)


def test_asset_probe_service_missing_or_empty_file(tmp_path: Path):
    non_existent = tmp_path / "missing.mp4"
    with pytest.raises(InvalidAssetFileError, match="does not exist"):
        probe_media_file(non_existent)

    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(InvalidAssetFileError, match="is empty"):
        probe_media_file(empty_file)


def test_asset_probe_service_video_probe(tmp_path: Path):
    video_path = tmp_path / "test_clip.mp4"
    # Write a simulated MP4 header
    video_path.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    probe = probe_media_file(video_path)
    assert probe.media_type == AssetMediaType.VIDEO
    assert probe.mime_type == "video/mp4"
    assert probe.width > 0
    assert probe.height > 0
    assert probe.duration_seconds >= 0.0
    assert probe.file_size_bytes == video_path.stat().st_size
    assert len(probe.file_hash) == 64


def test_multiple_asset_versions_for_same_shot_revision():
    v1 = ShotAssetVersion(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-1",
        file_path="storage/shot_assets/shot-1/v1.mp4",
        file_hash="1" * 64,
        file_size_bytes=5000,
        media_type=AssetMediaType.VIDEO,
        mime_type="video/mp4",
        width=1920,
        height=1080,
        duration_seconds=4.0,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )
    v2 = ShotAssetVersion(
        shot_id="shot-1",
        shot_revision_id="rev-1",
        execution_attempt_id="att-2",
        file_path="storage/shot_assets/shot-1/v2.mp4",
        file_hash="2" * 64,
        file_size_bytes=6200,
        media_type=AssetMediaType.VIDEO,
        mime_type="video/mp4",
        width=1920,
        height=1080,
        duration_seconds=4.2,
        provider="wavespeed",
        model="standard",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
    )

    assert v1.shot_revision_id == v2.shot_revision_id
    assert v1.shot_asset_version_id != v2.shot_asset_version_id
    assert v1.execution_attempt_id != v2.execution_attempt_id
    assert v1.file_hash != v2.file_hash
