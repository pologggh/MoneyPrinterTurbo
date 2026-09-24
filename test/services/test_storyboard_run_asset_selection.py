from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    GenerationMode,
    ShotAssetVersion,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)
from app.services.storyboard_video_assembly_service import (
    StoryboardAssemblyError,
    StoryboardVideoAssemblyService,
)


def _setup_sqlite_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _setup_storyboard_fixture(tmp_path):
    SessionLocal = _setup_sqlite_db()

    video_run1 = tmp_path / "video_run1.mp4"
    video_run1.write_bytes(b"run 1 mp4 content")

    video_run2 = tmp_path / "video_run2.mp4"
    video_run2.write_bytes(b"run 2 mp4 content")

    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        exec_repo = ExecutionRepository(session)

        beat = ContentBeat(
            beat_id="beat_run_01",
            beat_lineage_id="bl_run_01",
            beat_type=BeatType.HOOK,
            order=1,
            intent="引言",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan_run_01",
            revision_number=1,
            topic="素材选型隔离测试",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        plan_repo.add_revision(plan)

        shot = Shot(
            shot_id="shot_run_01",
            beat_lineage_id="bl_run_01",
            local_order=1,
        )
        shot_repo.add_shot(shot)

        rev = ShotRevision(
            shot_revision_id="srev_run_01",
            shot_id=shot.shot_id,
            revision_number=1,
            beat_lineage_id="bl_run_01",
            created_from_beat_instance_id="beat_run_01",
            narration="测试多Run解说旁白",
            target_duration=5.0,
            visual_goal="测试目标",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="场景",
            generation_prompt="prompt",
            camera_movement="static",
        )
        shot_repo.add_revision(rev)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap_run_test_01",
            content_plan_revision_id=plan.content_plan_revision_id,
            shot_revision_ids=(rev.shot_revision_id,),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        # Create Run 1
        run1 = ExecutionRun(
            execution_run_id="run_001",
            asset_route_plan_id="arp_001",
            storyboard_snapshot_id=snap.storyboard_snapshot_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=1,
            succeeded_shots=1,
        )
        exec_repo.add_execution_run(run1)

        att1 = ExecutionAttempt(
            execution_attempt_id="att_run1_01",
            execution_run_id=run1.execution_run_id,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            candidate_index=0,
            attempt_number=1,
            provider="pexels",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        exec_repo.add_execution_attempt(att1)

        v1 = ShotAssetVersion(
            shot_asset_version_id="asset_run1_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            execution_attempt_id=att1.execution_attempt_id,
            file_path=str(video_run1),
            file_hash="1" * 64,
            file_size_bytes=len(b"run 1 mp4 content"),
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            fps=30.0,
            provider="pexels",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
        )
        exec_repo.add_shot_asset_version(v1)

        # Create Run 2 (newer run)
        run2 = ExecutionRun(
            execution_run_id="run_002",
            asset_route_plan_id="arp_002",
            storyboard_snapshot_id=snap.storyboard_snapshot_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=1,
            succeeded_shots=1,
        )
        exec_repo.add_execution_run(run2)

        att2 = ExecutionAttempt(
            execution_attempt_id="att_run2_01",
            execution_run_id=run2.execution_run_id,
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            candidate_index=0,
            attempt_number=1,
            provider="pexels",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        exec_repo.add_execution_attempt(att2)

        v2 = ShotAssetVersion(
            shot_asset_version_id="asset_run2_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            execution_attempt_id=att2.execution_attempt_id,
            file_path=str(video_run2),
            file_hash="2" * 64,
            file_size_bytes=len(b"run 2 mp4 content"),
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            fps=30.0,
            provider="pexels",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
        )
        exec_repo.add_shot_asset_version(v2)
        session.commit()

    return SessionLocal, video_run1, video_run2


def test_assemble_video_strictly_binds_to_specified_execution_run(tmp_path):
    """Area 3: Verify assemble_video selects material produced in specified execution_run_id."""
    SessionLocal, video_run1, video_run2 = _setup_storyboard_fixture(tmp_path)
    service = StoryboardVideoAssemblyService(session_factory=SessionLocal)

    mock_audio = str(tmp_path / "audio.mp3")
    Path(mock_audio).write_bytes(b"audio")
    mock_out = str(tmp_path / "out.mp4")
    Path(mock_out).write_bytes(b"out")

    captured_materials = []

    def fake_generate_final_videos(**kwargs):
        captured_materials.extend(kwargs.get("downloaded_videos", []))
        return ([mock_out], [], [])

    with (
        patch("app.services.task.generate_audio", return_value=(mock_audio, 5.0, MagicMock())),
        patch("app.services.task.generate_subtitle", return_value=""),
        patch("app.services.task.generate_final_videos", side_effect=fake_generate_final_videos),
    ):
        # Target Run 1
        res1 = service.assemble_video(
            storyboard_snapshot_id="snap_run_test_01",
            execution_run_id="run_001",
        )
        assert res1.execution_run_id == "run_001"
        assert str(video_run1) in captured_materials
        assert str(video_run2) not in captured_materials

        captured_materials.clear()

        # Target Run 2
        res2 = service.assemble_video(
            storyboard_snapshot_id="snap_run_test_01",
            execution_run_id="run_002",
        )
        assert res2.execution_run_id == "run_002"
        assert str(video_run2) in captured_materials
        assert str(video_run1) not in captured_materials


def test_assemble_video_resolves_latest_completed_run_when_unspecified(tmp_path):
    """Area 3: When execution_run_id is omitted, resolve latest terminal run."""
    SessionLocal, _video_run1, _video_run2 = _setup_storyboard_fixture(tmp_path)
    service = StoryboardVideoAssemblyService(session_factory=SessionLocal)

    mock_audio = str(tmp_path / "audio.mp3")
    Path(mock_audio).write_bytes(b"audio")
    mock_out = str(tmp_path / "out.mp4")
    Path(mock_out).write_bytes(b"out")

    with (
        patch("app.services.task.generate_audio", return_value=(mock_audio, 5.0, MagicMock())),
        patch("app.services.task.generate_subtitle", return_value=""),
        patch("app.services.task.generate_final_videos", return_value=([mock_out], [], [])),
    ):
        res = service.assemble_video(storyboard_snapshot_id="snap_run_test_01")
        # Run 2 is the newer run, so it must be selected
        assert res.execution_run_id == "run_002"


def test_assemble_video_rejects_foreign_execution_run(tmp_path):
    """Area 3: Verify error if execution_run_id belongs to a different snapshot."""
    SessionLocal, _, _ = _setup_storyboard_fixture(tmp_path)

    # Insert another run belonging to foreign snapshot
    with SessionLocal() as session:
        exec_repo = ExecutionRepository(session)
        foreign_run = ExecutionRun(
            execution_run_id="run_foreign",
            asset_route_plan_id="arp_foreign",
            storyboard_snapshot_id="foreign_snap_999",
            status=ExecutionStatus.COMPLETED,
            total_shots=1,
        )
        exec_repo.add_execution_run(foreign_run)
        session.commit()

    service = StoryboardVideoAssemblyService(session_factory=SessionLocal)
    with pytest.raises(StoryboardAssemblyError) as exc_info:
        service.assemble_video(
            storyboard_snapshot_id="snap_run_test_01",
            execution_run_id="run_foreign",
        )
    assert "belongs to storyboard_snapshot_id 'foreign_snap_999'" in str(exc_info.value)


def test_assemble_video_reports_missing_shot_in_run_cleanly(tmp_path):
    """Area 3: Verify specific error with shot_id and execution_run_id if asset missing in that run."""
    SessionLocal = _setup_sqlite_db()

    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        exec_repo = ExecutionRepository(session)

        beat = ContentBeat(
            beat_id="beat_missing",
            beat_lineage_id="bl_missing",
            beat_type=BeatType.HOOK,
            order=1,
            intent="引言",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan_missing",
            revision_number=1,
            topic="缺素材细分报错测试",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        plan_repo.add_revision(plan)

        shot = Shot(
            shot_id="shot_missing_01",
            beat_lineage_id="bl_missing",
            local_order=1,
        )
        shot_repo.add_shot(shot)

        rev = ShotRevision(
            shot_revision_id="srev_missing_01",
            shot_id=shot.shot_id,
            revision_number=1,
            beat_lineage_id="bl_missing",
            created_from_beat_instance_id="beat_missing",
            narration="旁白",
            target_duration=5.0,
            visual_goal="目标",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="场景",
            generation_prompt="prompt",
            camera_movement="static",
        )
        shot_repo.add_revision(rev)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap_missing_test",
            content_plan_revision_id=plan.content_plan_revision_id,
            shot_revision_ids=(rev.shot_revision_id,),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        run = ExecutionRun(
            execution_run_id="run_incomplete",
            asset_route_plan_id="arp_inc",
            storyboard_snapshot_id=snap.storyboard_snapshot_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=1,
        )
        exec_repo.add_execution_run(run)
        # Intentionally DO NOT add asset for srev_missing_01 in run_incomplete
        session.commit()

    service = StoryboardVideoAssemblyService(session_factory=SessionLocal)
    with pytest.raises(StoryboardAssemblyError) as exc_info:
        service.assemble_video(
            storyboard_snapshot_id="snap_missing_test",
            execution_run_id="run_incomplete",
        )

    err = str(exc_info.value)
    assert "shot_missing_01" in err
    assert "srev_missing_01" in err
    assert "run_incomplete" in err
