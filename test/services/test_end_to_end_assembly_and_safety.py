from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ExecutionRun,
    ExecutionStatus,
    GenerationMode,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
    create_routing_request_from_shot_revision,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_editing import StoryboardEditingService
from app.models import const
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)
from app.services import state as sm
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.attempt_recovery_service import AttemptRecoveryService
from app.services.shot_execution_service import ShotExecutionService
from app.services.storyboard_generation_pipeline import (
    StoryboardGenerationInput,
    StoryboardGenerationPipeline,
)
from app.services.storyboard_video_assembly_service import (
    StoryboardAssemblyError,
    StoryboardVideoAssemblyService,
)


def _setup_sqlite_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _create_sample_shot(
    shot_id: str = "shot_test_01", beat_lineage_id: str = "bl_01"
) -> tuple[Shot, ShotRevision]:
    shot = Shot(
        shot_id=shot_id,
        beat_lineage_id=beat_lineage_id,
        local_order=1,
    )
    rev = ShotRevision(
        shot_id=shot_id,
        revision_number=1,
        beat_lineage_id=beat_lineage_id,
        created_from_beat_instance_id="beat_inst_01",
        narration="Photosynthesis is the process by which plants use sunlight to produce energy.",
        visual_type=VisualType.STOCK_VIDEO,
        target_duration=5.0,
        visual_goal="Illustrate sunlight hitting leaves",
        scene_description="Sunlight streaming through vibrant green leaves",
        generation_prompt="sunlight rays on forest canopy leaves",
        camera_movement="slow pan",
    )
    return shot, rev


def _create_sample_route_plan(
    snapshot_id: str,
    shot_revs: list[ShotRevision],
    provider: str = "pexels",
) -> AssetRoutePlan:
    shot_routes = []
    for r in shot_revs:
        req = create_routing_request_from_shot_revision(r)
        cand = AssetRouteCandidate(
            capability_id=f"cap_{provider}_01",
            provider=provider,
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            requested_visual_type=r.visual_type,
            is_eligible=True,
        )
        dec = AssetRouteDecision(
            shot_id=r.shot_id,
            shot_revision_id=r.shot_revision_id,
            requested_visual_type=r.visual_type,
            routing_strategy=RoutingStrategy.BALANCED,
            selected_candidate=cand,
            eligible_candidates=(cand,),
            rejected_candidates=(),
        )
        entry = ShotRoutePlanEntry(
            shot_id=r.shot_id,
            shot_revision_id=r.shot_revision_id,
            beat_lineage_id=r.beat_lineage_id,
            requested_visual_type=r.visual_type,
            asset_routing_request=req,
            route_status=ShotRoutePlanStatus.ROUTED,
            route_decision=dec,
        )
        shot_routes.append(entry)

    return AssetRoutePlan(
        asset_route_plan_id=f"arp_{snapshot_id}",
        storyboard_snapshot_id=snapshot_id,
        content_plan_revision_id="plan_test_01",
        routing_strategy=RoutingStrategy.BALANCED,
        routing_policy_version="1.0.0",
        model_selection_mode=ModelSelectionMode.AUTO,
        shot_routes=tuple(shot_routes),
    )


def test_route_plan_execution_exception_safety_and_terminal_status(tmp_path):
    """
    Items 1, 2, 3: Verify that an unexpected exception during execute_route_plan
    never leaves the task permanently stuck in RUNNING, sets finished_at, and marks FAILED in DB.
    """
    SessionLocal = _setup_sqlite_db()
    _shot1, rev1 = _create_sample_shot("shot_fail_1", "bl_01")
    _shot2, rev2 = _create_sample_shot("shot_fail_2", "bl_01")
    snap_id = "snap_fail_test"

    plan = _create_sample_route_plan(snap_id, [rev1, rev2])

    service = AssetRoutePlanExecutionService(session_factory=SessionLocal)

    # Force an unexpected unhandled exception inside the per-shot iteration
    with patch.object(
        service,
        "_execute_single_shot_entry",
        side_effect=RuntimeError("Simulated sudden unhandled kernel/OS crash"),
    ):
        result = service.execute_route_plan(
            plan=plan,
            storage_base_dir=tmp_path,
        )

    # Assert run outcome is strictly FAILED, not stuck in RUNNING
    assert result.state == ExecutionStatus.FAILED
    assert result.finished_at is not None
    assert result.failed_shots >= 1

    # Verify directly in the SQLite database that status is FAILED
    with SessionLocal() as session:
        repo = ExecutionRepository(session)
        db_run = repo.get_execution_run(result.execution_run_id)
        assert db_run is not None
        assert db_run.status == ExecutionStatus.FAILED
        assert db_run.finished_at is not None


def test_adapter_lookup_failure_diagnostic_error_message(tmp_path):
    """
    Item 4: Verify that when an adapter lookup fails for an unknown provider,
    an understandable diagnostic error message is recorded detailing available providers.
    """
    SessionLocal = _setup_sqlite_db()
    shot, rev = _create_sample_shot("shot_bad_prov_1")

    # Create candidate with non-existent provider
    bad_cand = AssetRouteCandidate(
        capability_id="cap_bad_01",
        provider="non_existent_super_gpu",
        model="hyper_model",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=rev.visual_type,
        is_eligible=True,
    )
    req = create_routing_request_from_shot_revision(rev)
    dec = AssetRouteDecision(
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        requested_visual_type=rev.visual_type,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=bad_cand,
        eligible_candidates=(bad_cand,),
        rejected_candidates=(),
    )

    shot_exec = ShotExecution(
        execution_run_id="run_bad_prov",
        shot_id=shot.shot_id,
        shot_revision_id=rev.shot_revision_id,
        route_decision=dec,
        status=ShotExecutionStatus.PENDING,
    )

    exec_service = ShotExecutionService(
        adapter_registry=AdapterRegistry(),
        policy=ExecutionRetryPolicy(max_attempts_per_candidate=1, max_total_attempts_per_shot=1),
        session_factory=SessionLocal,
    )

    final_exec, attempts, _asset_ver = exec_service.execute_shot(
        shot_execution=shot_exec,
        request=req,
        storage_base_dir=tmp_path,
    )

    # Verify attempt failed gracefully
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.status == ExecutionAttemptStatus.FAILED
    assert attempt.error_code == "ADAPTER_NOT_FOUND"
    assert "non_existent_super_gpu" in attempt.error_message
    assert "Available providers:" in attempt.error_message
    # Registry includes seedance, pexels, etc.
    assert "pexels" in attempt.error_message

    # Verify shot marked exhausted / failed without unhandled exception
    assert final_exec.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED)


def test_storyboard_generation_pipeline_topic_to_storyboard(tmp_path):
    """
    Items 5, 6, 7: Formal entrypoint 'User Theme -> ContentPlan -> Storyboard'.
    Verify StoryboardGenerationPipeline coordinates ContentPlanner and StoryboardOrchestrator,
    persisting ContentPlanRevision and StoryboardSnapshot.
    """
    SessionLocal = _setup_sqlite_db()

    mock_plan_output = json.dumps({
        "title": "量子力学入门",
        "beats": [
            {
                "planner_ref": "b1",
                "order": 1,
                "beat_type": "HOOK",
                "intent": "引入量子力学的神秘现象",
                "target_duration": 30.0,
                "importance": 0.8,
                "evidence_refs": [],
            },
            {
                "planner_ref": "b2",
                "order": 2,
                "beat_type": "KNOWLEDGE",
                "intent": "解释波粒二象性",
                "target_duration": 30.0,
                "importance": 0.9,
                "evidence_refs": [],
            },
        ],
    })

    def mock_llm(prompt: str) -> str:
        if "Expert Knowledge Video Content Planner" in prompt:
            return mock_plan_output
        if "Expert Knowledge Video Storyboard Agent" in prompt:
            return json.dumps({
                "beat_ref": "b_any",
                "shots": [
                    {
                        "planner_ref": "s1",
                        "local_order": 1,
                        "narration": "这是一个神奇的微观世界。",
                        "target_duration": 30.0,
                        "visual_goal": "展现微观粒子运动",
                        "visual_type": "STOCK_VIDEO",
                        "scene_description": "发光的微观粒子在暗色背景中振动",
                        "generation_prompt": "quantum particles glowing in dark background",
                        "camera_movement": "zoom in",
                        "evidence_refs": [],
                    }
                ],
            })
        return "{}"

    pipeline = StoryboardGenerationPipeline(
        llm_caller=mock_llm,
        session_factory=SessionLocal,
    )

    inp = StoryboardGenerationInput(
        topic="量子力学入门",
        target_video_duration=60.0,
        target_aspect_ratio="16:9",
        language="zh",
        source_grounded=False,
    )

    result = pipeline.generate(inp)

    assert result.content_plan_revision_id is not None
    assert result.storyboard_snapshot_id is not None
    assert result.beat_count == 2
    assert result.shot_count == 2
    assert result.state == StoryboardSnapshotState.DRAFT

    # Verify records persisted in DB
    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        sb_repo = StoryboardRepository(session)
        db_plan = plan_repo.get_revision(result.content_plan_revision_id)
        assert db_plan is not None
        assert len(db_plan.beats) == 2

        db_snap = sb_repo.get_snapshot(result.storyboard_snapshot_id)
        assert db_snap is not None
        assert len(db_snap.shot_revision_ids) == 2


def test_asset_execution_writeback_to_storyboard_editing_view(tmp_path):
    """
    Item 8: Verify that executed shot asset results are written back and enriched
    into StoryboardEditingView (produced_asset_version_id, file_path, resolution, duration).
    """
    SessionLocal = _setup_sqlite_db()
    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        exec_repo = ExecutionRepository(session)

        beat = ContentBeat(
            beat_id="beat_wb_01",
            beat_lineage_id="bl_wb_01",
            beat_type=BeatType.KNOWLEDGE,
            order=1,
            intent="核心讲解",
            target_duration=10.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan_wb_01",
            revision_number=1,
            topic="素材回写测试",
            overall_target_duration=10.0,
            beats=(beat,),
        )
        plan_repo.add_revision(plan)

        shot, rev = _create_sample_shot("shot_wb_01", "bl_wb_01")
        shot_repo.add_shot(shot)
        shot_repo.add_revision(rev)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap_wb_01",
            content_plan_revision_id=plan.content_plan_revision_id,
            shot_revision_ids=(rev.shot_revision_id,),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        # Add produced asset version
        dummy_file = tmp_path / "rendered_asset.mp4"
        dummy_file.write_bytes(b"fake mp4 video bytes")

        asset_version = ShotAssetVersion(
            shot_asset_version_id="asset_ver_wb_01",
            shot_id=shot.shot_id,
            shot_revision_id=rev.shot_revision_id,
            execution_attempt_id="att_wb_01",
            file_path=str(dummy_file),
            file_hash="a" * 64,
            file_size_bytes=len(b"fake mp4 video bytes"),
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
        exec_repo.add_shot_asset_version(asset_version)
        session.commit()

    # Load editing view with execution repository
    with SessionLocal() as session:
        service = StoryboardEditingService(
            plan_repository=ContentPlanRepository(session),
            shot_repository=ShotRepository(session),
            storyboard_repository=StoryboardRepository(session),
            execution_repository=ExecutionRepository(session),
        )
        view = service.get_storyboard_for_editing("snap_wb_01")

        assert len(view.beats) == 1
        assert len(view.beats[0].shots) == 1
        shot_view = view.beats[0].shots[0]

        # Asset metadata must be cleanly written back into the view
        assert shot_view.produced_asset_version_id == "asset_ver_wb_01"
        assert shot_view.asset_file_path == str(dummy_file)
        assert shot_view.asset_width == 1920
        assert shot_view.asset_height == 1080
        assert shot_view.asset_duration == 5.0
        assert shot_view.execution_status == "SUCCEEDED"


def test_storyboard_video_assembly_service_full_pipeline(tmp_path):
    """
    Items 9–14: Verify StoryboardVideoAssemblyService connects executed shot assets,
    synthesizes TTS audio, generates subtitles, stitches video sequentially,
    mixes BGM, and returns final MP4 paths and task status.
    """
    SessionLocal = _setup_sqlite_db()

    # 1. Create temporary mock media files
    video1_file = tmp_path / "shot_01.mp4"
    video1_file.write_bytes(b"mp4 content 1")
    image2_file = tmp_path / "shot_02.png"
    image2_file.write_bytes(b"png content 2")

    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)
        exec_repo = ExecutionRepository(session)

        beat1 = ContentBeat(
            beat_id="beat_as_01",
            beat_lineage_id="bl_as_01",
            beat_type=BeatType.HOOK,
            order=1,
            intent="引言",
            target_duration=5.0,
            importance=0.8,
        )
        beat2 = ContentBeat(
            beat_id="beat_as_02",
            beat_lineage_id="bl_as_02",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="核心论述",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan_as_01",
            revision_number=1,
            topic="全流程视频合成测试",
            overall_target_duration=10.0,
            beats=(beat1, beat2),
        )
        plan_repo.add_revision(plan)

        shot1, rev1 = _create_sample_shot("shot_as_01", "bl_as_01")
        shot2, rev2 = _create_sample_shot("shot_as_02", "bl_as_02")
        rev2 = rev2.model_copy(update={"narration": "这是第二个镜头的旁白解说。"})
        shot_repo.add_shot(shot1)
        shot_repo.add_revision(rev1)
        shot_repo.add_shot(shot2)
        shot_repo.add_revision(rev2)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap_as_01",
            content_plan_revision_id=plan.content_plan_revision_id,
            shot_revision_ids=(rev1.shot_revision_id, rev2.shot_revision_id),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)

        # Attach asset versions for both shots
        v1 = ShotAssetVersion(
            shot_asset_version_id="asset_as_01",
            shot_id=shot1.shot_id,
            shot_revision_id=rev1.shot_revision_id,
            execution_attempt_id="att_as_01",
            file_path=str(video1_file),
            file_hash="a" * 64,
            file_size_bytes=100,
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
        v2 = ShotAssetVersion(
            shot_asset_version_id="asset_as_02",
            shot_id=shot2.shot_id,
            shot_revision_id=rev2.shot_revision_id,
            execution_attempt_id="att_as_02",
            file_path=str(image2_file),
            file_hash="b" * 64,
            file_size_bytes=100,
            media_type=AssetMediaType.IMAGE,
            mime_type="image/png",
            width=1024,
            height=1024,
            duration_seconds=5.0,
            fps=0.0,
            provider="openai",
            model="dall-e-3",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
        )
        run = ExecutionRun(
            execution_run_id="run_as_01",
            asset_route_plan_id="arp_as_01",
            storyboard_snapshot_id=snap.storyboard_snapshot_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=2,
            succeeded_shots=2,
        )
        exec_repo.add_execution_run(run)
        att1 = ExecutionAttempt(
            execution_attempt_id="att_as_01",
            execution_run_id=run.execution_run_id,
            shot_id=shot1.shot_id,
            shot_revision_id=rev1.shot_revision_id,
            candidate_index=0,
            attempt_number=1,
            provider="pexels",
            model="default",
            generation_mode=GenerationMode.STOCK_SEARCH,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        att2 = ExecutionAttempt(
            execution_attempt_id="att_as_02",
            execution_run_id=run.execution_run_id,
            shot_id=shot2.shot_id,
            shot_revision_id=rev2.shot_revision_id,
            candidate_index=0,
            attempt_number=1,
            provider="openai",
            model="dall-e-3",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        exec_repo.add_execution_attempt(att1)
        exec_repo.add_execution_attempt(att2)
        exec_repo.add_shot_asset_version(v1)
        exec_repo.add_shot_asset_version(v2)
        session.commit()

    # Assemble video with mock media generation methods
    mock_audio_file = str(tmp_path / "audio.mp3")
    Path(mock_audio_file).write_bytes(b"dummy mp3")
    mock_sub_file = str(tmp_path / "subtitle.srt")
    Path(mock_sub_file).write_text("1\n00:00:00,000 --> 00:00:05,000\nSubtitle text")
    mock_final_mp4 = str(tmp_path / "final-1.mp4")
    Path(mock_final_mp4).write_bytes(b"final mp4 content")
    mock_combined_mp4 = str(tmp_path / "combined-1.mp4")
    Path(mock_combined_mp4).write_bytes(b"combined mp4 content")

    assembly_service = StoryboardVideoAssemblyService(session_factory=SessionLocal)

    with (
        patch("app.services.video.render_image_zoom_video", return_value=str(video1_file)),
        patch("app.services.task.generate_audio", return_value=(mock_audio_file, 10.0, MagicMock())),
        patch("app.services.task.generate_subtitle", return_value=mock_sub_file),
        patch(
            "app.services.task.generate_final_videos",
            return_value=([mock_final_mp4], [mock_combined_mp4], []),
        ),
    ):
        result = assembly_service.assemble_video(
            storyboard_snapshot_id="snap_as_01",
            video_params={"video_subject": "全流程视频合成测试", "bgm_type": "random"},
            task_id="test_task_assembly_01",
        )

    assert result.task_id == "test_task_assembly_01"
    assert result.status == "COMPLETED"
    assert result.execution_run_id == "run_as_01"
    assert result.final_video_paths == [mock_final_mp4]
    assert result.combined_video_paths == [mock_combined_mp4]
    assert result.audio_path == mock_audio_file
    assert result.subtitle_path == mock_sub_file
    assert result.audio_duration == 10.0

    # Verify task state in sm.state is completed
    task_state = sm.state.get_task("test_task_assembly_01")
    assert task_state is not None
    assert task_state["state"] == const.TASK_STATE_COMPLETE
    assert task_state["progress"] == 100


def test_missing_asset_fails_assembly_cleanly(tmp_path):
    """Verify that if a shot is missing its executed asset, assembly fails with StoryboardAssemblyError."""
    SessionLocal = _setup_sqlite_db()
    with SessionLocal() as session:
        plan_repo = ContentPlanRepository(session)
        shot_repo = ShotRepository(session)
        sb_repo = StoryboardRepository(session)

        beat = ContentBeat(
            beat_id="b1",
            beat_lineage_id="bl1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="引言",
            target_duration=5.0,
            importance=0.8,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="p1",
            revision_number=1,
            topic="缺素材测试",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        plan_repo.add_revision(plan)

        shot, rev = _create_sample_shot("shot_no_asset", "bl1")
        shot_repo.add_shot(shot)
        shot_repo.add_revision(rev)

        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap_no_asset",
            content_plan_revision_id="p1",
            shot_revision_ids=(rev.shot_revision_id,),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        sb_repo.add_snapshot(snap)
        session.commit()

    service = StoryboardVideoAssemblyService(session_factory=SessionLocal)
    with pytest.raises(StoryboardAssemblyError) as exc_info:
        service.assemble_video(storyboard_snapshot_id="snap_no_asset")

    assert "has no produced asset" in str(exc_info.value)


def test_attempt_recovery_run_reconciles_status(tmp_path):
    """
    Item 15: Verify AttemptRecoveryService.recover_run reconciles shot executions
    needing recovery and updates the overall run status.
    """
    SessionLocal = _setup_sqlite_db()
    cand = AssetRouteCandidate(
        capability_id="cap_rec_01",
        provider="volcengine_seedance",
        model="seedance_v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )
    dec = AssetRouteDecision(
        shot_id="shot_rec_01",
        shot_revision_id="rev_rec_01",
        requested_visual_type=VisualType.AI_VIDEO,
        routing_strategy=RoutingStrategy.BALANCED,
        selected_candidate=cand,
        eligible_candidates=(cand,),
        rejected_candidates=(),
    )

    with SessionLocal() as session:
        exec_repo = ExecutionRepository(session)

        run = ExecutionRun(
            execution_run_id="run_rec_01",
            asset_route_plan_id="arp_rec_01",
            storyboard_snapshot_id="snap_rec_01",
            status=ExecutionStatus.NEEDS_RECOVERY,
            total_shots=1,
            succeeded_shots=0,
            reused_shots=0,
            failed_shots=0,
            recovery_required_shots=1,
            started_at=datetime.now(UTC),
        )
        exec_repo.add_execution_run(run)

        se = ShotExecution(
            shot_execution_id="se_rec_01",
            execution_run_id="run_rec_01",
            shot_id="shot_rec_01",
            shot_revision_id="rev_rec_01",
            route_decision=dec,
            status=ShotExecutionStatus.NEEDS_RECOVERY,
            current_candidate_index=0,
            attempt_ids=["att_rec_01"],
            unconfirmed_remote_task_id="remote_job_123",
        )
        exec_repo.add_shot_execution(se)

        att = ExecutionAttempt(
            execution_attempt_id="att_rec_01",
            execution_run_id="run_rec_01",
            shot_id="shot_rec_01",
            shot_revision_id="rev_rec_01",
            attempt_number=1,
            provider="volcengine_seedance",
            model="seedance_v1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            status=ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
        )
        exec_repo.add_execution_attempt(att)

        from app.domain.asset_execution import AttemptRequest
        req = AttemptRequest.create_sanitized(
            execution_attempt_id="att_rec_01",
            provider="volcengine_seedance",
            model="seedance_v1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            idempotency_key="idemp_rec_01",
            raw_payload={},
        )
        exec_repo.save_attempt_request(req)
        session.commit()

    # Mock adapter returning remote failure
    mock_adapter = MagicMock()
    mock_adapter.get_capabilities.return_value = MagicMock(
        supports_async_status=True,
        supports_safe_resubmission_with_same_key=False,
    )
    from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
    mock_adapter.get_status.return_value = AdapterExecutionResult(
        outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
        error_code="GPU_OOM",
        error_message="GPU out of memory on provider",
    )

    registry = AdapterRegistry(custom_adapters={"volcengine_seedance": mock_adapter})
    recovery_service = AttemptRecoveryService(adapter_registry=registry)

    with SessionLocal() as session:
        rec_res = recovery_service.recover_run(
            run_id="run_rec_01",
            session=session,
            storage_base_dir=tmp_path,
        )
        session.commit()

        assert rec_res["resolved_count"] == 1
        assert rec_res["overall_status"] == ExecutionStatus.FAILED.value

        # Verify in DB
        repo = ExecutionRepository(session)
        updated_run = repo.get_execution_run("run_rec_01")
        assert updated_run.status == ExecutionStatus.FAILED
        assert updated_run.recovery_required_shots == 0
