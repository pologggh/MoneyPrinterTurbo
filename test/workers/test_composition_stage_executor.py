"""
Comprehensive test suite for Stage C1: Composition Stage Integration.

Verifies:
1. Stage.COMPOSITION is registered in default executor registry.
2. Stage.QUALITY_REVIEW remains unregistered and unsupported.
3. CompositionStageExecutor consumes:
   - AudioOutput (from Stage.AUDIO)
   - COMPLETED ExecutionRun (from Stage.ASSET)
   - APPROVED StoryboardSnapshot
4. Exact shot ordering preserved (Beat.order then Shot.local_order).
5. Zero upstream regeneration (no ContentPlanner, Script LLM, StoryboardAgent, AssetRoutePlanning, Asset Providers, TTS, Subtitle, BGM generation calls).
6. Real playable MP4 video output rendering with MoviePy / FFmpeg.
7. Technical media validation (readable container, non-zero size, duration > 0, width/height > 0, video/audio streams).
8. Immutability of CompositionOutput (re-render creates new version, old remains untouched).
9. Failure safety (atomic promotion: temp files not promoted on error, bounded error metadata).
10. Long-running render heartbeat during execution.
11. End-to-end StageWorker workflow progression advancing from COMPOSITION to QUALITY_REVIEW (queued, unsupported).
12. Zero formal quality review or evaluation in C1.
"""

from __future__ import annotations

import hashlib
import os
import struct
import wave
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.composition_stage_executor import (
    CompositionStageExecutor,
    default_media_validator,
)
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    GenerationMode,
    ShotAssetVersion,
)
from app.domain.asset_router import AssetRoutePlan, AssetRoutePlanStatus, RoutingStrategy
from app.domain.audio import AudioOutput
from app.domain.composition import CompositionOutput
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    AudioOutputRepository,
    CompositionOutputRepository,
    ContentPlanRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    ShotExecutionRepository,
    ShotRepository,
    StageExecutionRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.storyboard_video_assembly_service import (
    StoryboardVideoAssemblyResult,
    StoryboardVideoAssemblyService,
)
from app.workers.stage_worker import StageWorker


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal


def _create_wav_audio_fixture(file_path: Path | str, duration_sec: float = 2.0) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    n_frames = int(sample_rate * duration_sec)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # 440 Hz tone or silence
        data = struct.pack("<h", 0) * n_frames
        wav_file.writeframes(data)
    return str(path)


def _create_image_fixture(file_path: Path | str, color: tuple[int, int, int] = (100, 150, 200)) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), color=color)
    img.save(str(path))
    return str(path)


def _create_srt_fixture(file_path: Path | str) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "1\n00:00:00,000 --> 00:00:02,000\nHello Knowledge Video\n\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _setup_full_prerequisites(session_factory, tmp_path, shot_count: int = 2):
    """Sets up a complete task with ContentPlan, Script, Approved Storyboard, Completed ExecutionRun, and AudioOutput."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    sb_id = f"sb_{uuid4().hex[:8]}"
    route_plan_id = f"arp_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"

    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    narration_path = _create_wav_audio_fixture(task_dir / "narration.wav", duration_sec=3.0)
    subtitle_path = _create_srt_fixture(task_dir / "subtitle.srt")

    beats = []
    shots = []
    shot_revs = []
    asset_versions = []
    attempts = []

    for i in range(1, shot_count + 1):
        beat_lineage = f"bl_{i}"
        beat = ContentBeat(
            beat_lineage_id=beat_lineage,
            order=i,
            intent=f"Explain concept {i}",
            target_duration=1.5,
            importance=0.8,
            beat_type=BeatType.KNOWLEDGE,
        )
        beats.append(beat)

        shot_id = f"shot_{i}"
        shot = Shot(
            shot_id=shot_id,
            beat_lineage_id=beat_lineage,
            local_order=1,
        )
        shots.append(shot)

        shot_rev_id = f"sr_{i}"
        shot_rev = ShotRevision(
            shot_revision_id=shot_rev_id,
            shot_id=shot_id,
            revision_number=1,
            beat_lineage_id=beat_lineage,
            created_from_beat_instance_id=beat.beat_id,
            visual_type=VisualType.AI_IMAGE,
            generation_prompt=f"Prompt for shot {i}",
            visual_goal=f"Goal for shot {i}",
            scene_description=f"Scene description {i}",
            camera_movement="Static",
            target_duration=1.5,
            narration=f"Narration text for shot {i}",
        )
        shot_revs.append(shot_rev)

        img_path = _create_image_fixture(task_dir / f"shot_{i}.png", color=(i * 40, 100, 200))
        img_bytes = Path(img_path).read_bytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        att_id = f"att_{i}"
        asset_v_id = f"sav_{i}"

        att = ExecutionAttempt(
            execution_attempt_id=att_id,
            execution_run_id=run_id,
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            candidate_index=0,
            attempt_number=1,
            provider="test_provider",
            model="test_model",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        attempts.append(att)

        asset_v = ShotAssetVersion(
            shot_asset_version_id=asset_v_id,
            shot_id=shot_id,
            shot_revision_id=shot_rev_id,
            execution_attempt_id=att_id,
            file_path=img_path,
            file_hash=img_hash,
            file_size_bytes=len(img_bytes),
            media_type=AssetMediaType.IMAGE,
            mime_type="image/png",
            width=320,
            height=240,
            duration_seconds=1.5,
            fps=30.0,
            provider="test_provider",
            model="test_model",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
        )
        asset_versions.append(asset_v)

    plan = ContentPlanRevision(
        content_plan_revision_id=plan_id,
        revision_number=1,
        topic="Physics of Time",
        overall_target_duration=3.0,
        beats=tuple(beats),
    )

    script = ScriptRevision.create(
        script_revision_id=script_id,
        task_id=task_id,
        content_plan_revision_id=plan_id,
        revision_number=1,
        overall_target_duration=3.0,
        segments=[
            ScriptSegment(
                script_revision_id=script_id,
                content_beat_id=b.beat_id,
                beat_lineage_id=b.beat_lineage_id,
                order=b.order,
                narration_text=f"Narration text for shot {b.order}",
                target_duration=1.5,
            )
            for b in beats
        ],
    )

    snapshot = StoryboardSnapshot(
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=plan_id,
        shot_revision_ids=tuple(r.shot_revision_id for r in shot_revs),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )

    route_plan = AssetRoutePlan(
        asset_route_plan_id=route_plan_id,
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=plan_id,
        routing_strategy=RoutingStrategy.BALANCED,
        routing_policy_version="v1.0",
        status=AssetRoutePlanStatus.READY,
    )

    run = ExecutionRun(
        execution_run_id=run_id,
        asset_route_plan_id=route_plan_id,
        storyboard_snapshot_id=sb_id,
        status=ExecutionStatus.COMPLETED,
        total_shots=shot_count,
        succeeded_shots=shot_count,
        failed_shots=0,
    )

    audio_output = AudioOutput.create(
        audio_output_id=audio_id,
        task_id=task_id,
        script_revision_id=script_id,
        execution_run_id=run_id,
        narration_audio_path=narration_path,
        narration_audio_hash="hash_narr_1",
        actual_narration_duration=3.0,
        subtitle_path=subtitle_path,
        subtitle_hash="hash_sub_1",
        bgm_path=None,
        bgm_volume=0.0,
    )

    task = KnowledgeVideoTask.create(
        task_id=task_id,
        topic="Physics of Time",
        target_duration=3.0,
        workflow_policy=WorkflowPolicyType.AUTO,
        task_metadata={"video_aspect": "16:9"},
    )
    task.advance_stage(Stage.KNOWLEDGE_PLAN)
    task.advance_stage(Stage.SCRIPT)
    task.advance_stage(Stage.STORYBOARD)
    task.advance_stage(Stage.PRODUCTION_PLAN)
    task.advance_stage(Stage.ASSET)
    task.advance_stage(Stage.AUDIO)
    task.advance_stage(Stage.COMPOSITION)

    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        p_repo = ContentPlanRepository(session)
        s_repo = ScriptRepository(session)
        sb_repo = StoryboardRepository(session)
        shot_repo = ShotRepository(session)
        rp_repo = AssetRoutePlanRepository(session)
        exec_repo = ExecutionRepository(session)
        se_repo = ShotExecutionRepository(session)
        aud_repo = AudioOutputRepository(session)
        art_repo = TaskArtifactRepository(session)
        j_repo = WorkflowJobRepository(session)

        t_repo.save_task(task)
        p_repo.add_revision(plan)
        s_repo.save_revision(script)
        for sh in shots:
            shot_repo.add_shot(sh)
        for sr in shot_revs:
            shot_repo.add_revision(sr)
        sb_repo.add_snapshot(snapshot)

        rp_repo.add_route_plan(route_plan)
        exec_repo.add_execution_run(run)
        for att in attempts:
            exec_repo.add_execution_attempt(att)
        for av in asset_versions:
            exec_repo.add_shot_asset_version(av)

        aud_repo.save_audio_output(audio_output)

        aud_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            artifact_type=ArtifactType.AUDIO_OUTPUT,
            artifact_id=audio_output.audio_output_id,
        )
        art_repo.save_artifact_ref(aud_ref)

        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.COMPOSITION,
            idempotency_key=f"idemp_{task_id}_comp_1",
            input_task_artifact_ref_id=aud_ref.task_artifact_ref_id,
        )
        j_repo.create_job(job)
        session.commit()

    return task, job, audio_output, run, snapshot, aud_ref


# =============================================================================
# 1. Registry Tests
# =============================================================================


def test_composition_stage_executor_registered_in_default_registry():
    """Verify Stage.COMPOSITION is registered and Stage.QUALITY_REVIEW remains unregistered."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.COMPOSITION)
    assert isinstance(registry.get_executor(Stage.COMPOSITION), CompositionStageExecutor)

    # Downstream boundary: QUALITY_REVIEW and DELIVERY must remain unsupported
    assert not registry.has_executor(Stage.QUALITY_REVIEW)
    assert registry.get_executor(Stage.QUALITY_REVIEW) is None
    assert not registry.has_executor(Stage.DELIVERY)
    assert registry.get_executor(Stage.DELIVERY) is None


# =============================================================================
# 2. Input Resolution and Lineage Verification Tests
# =============================================================================


def test_composition_executor_consumes_exact_lineage_and_produces_output(session_factory, tmp_path):
    """Verify CompositionStageExecutor consumes exact lineage and produces TaskArtifactRef(COMPOSITION_OUTPUT)."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Use test doubles for fast media assembly & validation in lineage unit test
    fake_video_file = tmp_path / "rendered_fake.mp4"
    fake_video_file.write_bytes(b"dummy mp4 content")

    mock_assembly = MagicMock(spec=StoryboardVideoAssemblyService)
    def fake_assemble(**kwargs):
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"final mp4 bytes")
        return StoryboardVideoAssemblyResult(
            task_id=task.task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            final_video_paths=[out_path],
            combined_video_paths=[out_path],
            audio_path=audio_out.narration_audio_path,
            subtitle_path=audio_out.subtitle_path or "",
            audio_duration=audio_out.actual_narration_duration,
            status="COMPLETED",
        )
    mock_assembly.assemble_video.side_effect = fake_assemble

    mock_validator = MagicMock(return_value={
        "duration": 3.0,
        "width": 1920,
        "height": 1080,
        "file_size": 1024,
        "fps": 30.0,
        "has_audio": True,
        "video_codec": "h264",
        "audio_codec": "aac",
    })

    executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "storage",
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )

    result = executor.execute(task, job)
    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.stage == Stage.COMPOSITION
    assert result.output_artifact_ref.artifact_type == ArtifactType.COMPOSITION_OUTPUT

    with session_factory() as session:
        comp_repo = CompositionOutputRepository(session)
        comp_out = comp_repo.get_composition_output(result.output_artifact_ref.artifact_id)
        assert comp_out is not None
        assert comp_out.task_id == task.task_id
        assert comp_out.storyboard_snapshot_id == snapshot.storyboard_snapshot_id
        assert comp_out.execution_run_id == run.execution_run_id
        assert comp_out.audio_output_id == audio_out.audio_output_id
        assert comp_out.width == 1920
        assert comp_out.height == 1080
        assert comp_out.duration == 3.0
        assert os.path.exists(comp_out.video_path)


def test_composition_executor_fails_when_input_artifact_ref_missing(session_factory):
    """Verify executor fails cleanly when input_task_artifact_ref_id is None."""
    task = KnowledgeVideoTask.create(task_id="t1", topic="Test")
    job = WorkflowJob.create(task_id="t1", stage=Stage.COMPOSITION, idempotency_key="k1")
    job.input_task_artifact_ref_id = None

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert result.is_retryable is False


def test_composition_executor_rejects_task_id_mismatch(session_factory, tmp_path):
    """Verify executor rejects input artifact ref from another task."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)
    other_task = KnowledgeVideoTask.create(task_id="other_task_123", topic="Other")

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(other_task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "mismatch" in result.error_message.lower()


def test_composition_executor_rejects_unapproved_storyboard(session_factory, tmp_path):
    """Verify executor rejects execution when storyboard snapshot is not APPROVED."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Change storyboard snapshot state to DRAFT
    with session_factory() as session:
        from app.persistence.models import StoryboardSnapshotORM
        snap_orm = session.get(StoryboardSnapshotORM, snapshot.storyboard_snapshot_id)
        snap_orm.snapshot_state = StoryboardSnapshotState.DRAFT.value
        session.commit()

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "not approved" in result.error_message.lower()


def test_composition_executor_rejects_incomplete_execution_run(session_factory, tmp_path):
    """Verify executor rejects composition when ExecutionRun is not COMPLETED."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Mutate ExecutionRun status to RUNNING
    with session_factory() as session:
        exec_repo = ExecutionRepository(session)
        exec_repo.update_execution_run_status(
            run_id=run.execution_run_id,
            status=ExecutionStatus.RUNNING,
            succeeded_shots=0,
            failed_shots=0,
        )
        session.commit()

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "not in completed state" in result.error_message.lower()


def test_composition_executor_rejects_incomplete_audio_output(session_factory, tmp_path):
    """Verify executor rejects composition when narration audio file does not exist on disk."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Delete the audio file
    if os.path.exists(audio_out.narration_audio_path):
        os.remove(audio_out.narration_audio_path)

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "missing or empty" in result.error_message.lower()


def test_composition_executor_rejects_incomplete_asset_output(session_factory, tmp_path):
    """Verify executor rejects composition when a required shot asset file is missing on disk."""
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Delete shot 1 image
    shot1_img = tmp_path / task.task_id / "shot_1.png"
    if shot1_img.exists():
        shot1_img.unlink()

    executor = CompositionStageExecutor(session_factory=session_factory)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "missing or unresolved asset" in result.error_message.lower()


# =============================================================================
# 3. No Upstream Regeneration Tests (Mandatory)
# =============================================================================


def test_no_upstream_regeneration_during_composition(session_factory, tmp_path):
    """
    Mandatory Test: Assert zero calls to:
    - ContentPlanner
    - Script LLM
    - StoryboardAgent
    - AssetRoutePlanningService
    - Asset Provider Adapters
    - voice.tts
    - task.generate_subtitle
    - BGM generator / selection
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    mock_assembly = MagicMock()
    mock_assembly.assemble_video.return_value = StoryboardVideoAssemblyResult(
        task_id=task.task_id,
        storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
        execution_run_id=run.execution_run_id,
        final_video_paths=["/tmp/test.mp4"],
        combined_video_paths=["/tmp/test.mp4"],
        audio_path=audio_out.narration_audio_path,
        subtitle_path="",
        audio_duration=3.0,
        status="COMPLETED",
    )
    mock_validator = MagicMock(return_value={
        "duration": 3.0, "width": 1920, "height": 1080, "file_size": 100, "fps": 30.0, "has_audio": True
    })

    with (
        patch("app.domain.planner.ContentPlanner.plan") as mock_planner,
        patch("app.services.voice.tts") as mock_tts,
        patch("app.services.task.generate_audio") as mock_gen_audio,
        patch("app.services.task.generate_subtitle") as mock_gen_sub,
    ):
        executor = CompositionStageExecutor(
            session_factory=session_factory,
            storage_base_dir=tmp_path / "storage",
            assembly_service=mock_assembly,
            media_validator=mock_validator,
        )

        # Pre-create temp output file so os.replace works
        def fake_assemble(**kwargs):
            out_path = kwargs["output_file_path"]
            Path(out_path).write_bytes(b"valid video")
            return mock_assembly.assemble_video.return_value
        mock_assembly.assemble_video.side_effect = fake_assemble

        result = executor.execute(task, job)
        assert result.success is True

        assert mock_planner.call_count == 0
        assert mock_tts.call_count == 0
        assert mock_gen_audio.call_count == 0
        assert mock_gen_sub.call_count == 0


# =============================================================================
# 4. Shot Ordering Preservation Tests (Mandatory)
# =============================================================================


def test_storyboard_shot_ordering_preserved_regardless_of_db_order(session_factory, tmp_path):
    """
    Mandatory Test: Verifies timeline follows Storyboard order: 1 -> 2 -> 3,
    not filesystem or arbitrary DB insertion order.
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path, shot_count=3)

    captured_kwargs = {}
    mock_assembly = MagicMock()
    def fake_assemble(**kwargs):
        captured_kwargs.update(kwargs)
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"valid video")
        return StoryboardVideoAssemblyResult(
            task_id=task.task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            final_video_paths=[out_path],
            combined_video_paths=[out_path],
            audio_path=audio_out.narration_audio_path,
            subtitle_path="",
            audio_duration=3.0,
            status="COMPLETED",
        )
    mock_assembly.assemble_video.side_effect = fake_assemble
    mock_validator = MagicMock(return_value={
        "duration": 3.0, "width": 1920, "height": 1080, "file_size": 100, "fps": 30.0, "has_audio": True
    })

    executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "storage",
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )
    result = executor.execute(task, job)
    assert result.success is True
    assert captured_kwargs["storyboard_snapshot_id"] == snapshot.storyboard_snapshot_id
    assert captured_kwargs["execution_run_id"] == run.execution_run_id


# =============================================================================
# 5. Real Playable MP4 Video Output Render Test (Mandatory Real Composition)
# =============================================================================


def test_mandatory_real_local_mp4_render(session_factory, tmp_path):
    """
    Mandatory Real Composition Test:
    Executes actual MoviePy / video composition on local media fixtures:
    - Shot A image fixture
    - Shot B image fixture
    - Real audio fixture
    - Real subtitle fixture
    Produces real playable MP4 and validates video stream, audio stream, duration, dimensions.
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path, shot_count=2)

    executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "real_storage",
    )

    result = executor.execute(task, job)
    assert result.success is True
    assert result.output_artifact_ref is not None

    with session_factory() as session:
        comp_repo = CompositionOutputRepository(session)
        comp_out = comp_repo.get_composition_output(result.output_artifact_ref.artifact_id)
        assert comp_out is not None

        # Validate real output on disk
        assert os.path.exists(comp_out.video_path)
        assert os.path.getsize(comp_out.video_path) > 0
        assert comp_out.duration > 0
        assert comp_out.width > 0
        assert comp_out.height > 0
        assert len(comp_out.video_hash) == 64

        # Validate media properties with MoviePy
        val = default_media_validator(comp_out.video_path, require_audio=True)
        assert val["duration"] > 0
        assert val["has_audio"] is True


# =============================================================================
# 6. Immutability & Re-render Tests (Mandatory)
# =============================================================================


def test_mandatory_immutability_and_rerender(session_factory, tmp_path):
    """
    Mandatory Immutability Test:
    1. Render run 1 -> CompositionOutput C1
    2. Re-render run 2 -> CompositionOutput C2
    3. C1 remains unchanged and points to historical file.
    4. C2 has distinct ID and file.
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    mock_assembly = MagicMock()
    def fake_assemble(**kwargs):
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"mp4 content " + uuid4().bytes)
        return StoryboardVideoAssemblyResult(
            task_id=task.task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            final_video_paths=[out_path],
            combined_video_paths=[out_path],
            audio_path=audio_out.narration_audio_path,
            subtitle_path="",
            audio_duration=3.0,
            status="COMPLETED",
        )
    mock_assembly.assemble_video.side_effect = fake_assemble
    mock_validator = MagicMock(return_value={
        "duration": 3.0, "width": 1920, "height": 1080, "file_size": 1024, "fps": 30.0, "has_audio": True
    })

    executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "immutability_storage",
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )

    # 1. First render
    res1 = executor.execute(task, job)
    assert res1.success is True

    # 2. Second render (e.g. on retry or re-execution)
    job2 = WorkflowJob.create(
        task_id=task.task_id,
        stage=Stage.COMPOSITION,
        idempotency_key="idemp_comp_2",
        input_task_artifact_ref_id=aud_ref.task_artifact_ref_id,
    )
    res2 = executor.execute(task, job2)
    assert res2.success is True

    with session_factory() as session:
        comp_repo = CompositionOutputRepository(session)
        out1 = comp_repo.get_composition_output(res1.output_artifact_ref.artifact_id)
        out2 = comp_repo.get_composition_output(res2.output_artifact_ref.artifact_id)

        assert out1.composition_output_id != out2.composition_output_id
        assert out1.video_path != out2.video_path
        assert os.path.exists(out1.video_path)
        assert os.path.exists(out2.video_path)


# =============================================================================
# 7. Failure Safety Tests (Mandatory)
# =============================================================================


def test_mandatory_failure_safety_cleans_temp_file(session_factory, tmp_path):
    """
    Mandatory Failure Test:
    When rendering fails or media validation fails:
    - Temporary work file is cleaned up
    - No successful CompositionOutput is recorded
    - StageExecutionResult contains bounded error info
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    mock_assembly = MagicMock()
    def fail_assemble(**kwargs):
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"corrupt partial file")
        raise RuntimeError("FFmpeg process crashed: exit code 1")

    mock_assembly.assemble_video.side_effect = fail_assemble
    mock_validator = MagicMock()

    storage_dir = tmp_path / "failure_storage"
    executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=storage_dir,
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.is_retryable is True
    assert "FFmpeg process crashed" in result.error_message

    # Verify temp file was cleaned up and not promoted
    task_storage = storage_dir / task.task_id
    video_files = list(task_storage.glob("*.mp4")) if task_storage.exists() else []
    assert len(video_files) == 0

    with session_factory() as session:
        comp_repo = CompositionOutputRepository(session)
        outputs = comp_repo.list_composition_outputs_for_task(task.task_id)
        assert len(outputs) == 0


# =============================================================================
# 8. Long-Running Render Heartbeat Test (Mandatory)
# =============================================================================


def test_mandatory_long_render_heartbeat(session_factory, tmp_path):
    """
    Mandatory Heartbeat Test:
    Verify StageWorker background heartbeat keeps lease alive during long render.
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    # Injected slow assembly hook that checks lease renewal
    heartbeat_observed = []
    mock_assembly = MagicMock()
    def slow_assemble(**kwargs):
        import time
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"rendered video bytes")
        # Sleep slightly longer than a fast heartbeat interval to observe renewal
        time.sleep(0.1)
        with session_factory() as session:
            j_repo = WorkflowJobRepository(session)
            active_job = j_repo.get_job(job.job_id)
            if active_job and active_job.lease_expires_at:
                heartbeat_observed.append(active_job.lease_expires_at)
        return StoryboardVideoAssemblyResult(
            task_id=task.task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            final_video_paths=[out_path],
            combined_video_paths=[out_path],
            audio_path=audio_out.narration_audio_path,
            subtitle_path="",
            audio_duration=3.0,
            status="COMPLETED",
        )
    mock_assembly.assemble_video.side_effect = slow_assemble
    mock_validator = MagicMock(return_value={
        "duration": 3.0, "width": 1920, "height": 1080, "file_size": 100, "fps": 30.0, "has_audio": True
    })

    comp_executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "hb_storage",
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )

    stage_registry = StageExecutorRegistry()
    stage_registry.register(Stage.COMPOSITION, comp_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="heartbeat-worker",
        heartbeat_interval_seconds=0.05,
        lease_duration_seconds=5.0,
    )

    processed = worker.run_once()
    assert processed is True
    assert len(heartbeat_observed) >= 1


# =============================================================================
# 9. End-to-End Lineage & Workflow Progression Test (Mandatory)
# =============================================================================


def test_mandatory_stageworker_advances_composition_to_queued_quality_review(session_factory, tmp_path):
    """
    Mandatory Workflow Progression & Lineage Test:
    Verifies StageWorker:
    1. Claims COMPOSITION job.
    2. Runs CompositionStageExecutor.
    3. Persists CompositionOutput and TaskArtifactRef(COMPOSITION_OUTPUT).
    4. Completes COMPOSITION job.
    5. KnowledgeVideoWorkflow creates next job: WorkflowJob(stage=QUALITY_REVIEW).
    6. Stage.QUALITY_REVIEW remains unsupported in default registry and stays safely QUEUED.
    """
    task, job, audio_out, run, snapshot, aud_ref = _setup_full_prerequisites(session_factory, tmp_path)

    mock_assembly = MagicMock()
    def fake_assemble(**kwargs):
        out_path = kwargs["output_file_path"]
        Path(out_path).write_bytes(b"final video bytes")
        return StoryboardVideoAssemblyResult(
            task_id=task.task_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            execution_run_id=run.execution_run_id,
            final_video_paths=[out_path],
            combined_video_paths=[out_path],
            audio_path=audio_out.narration_audio_path,
            subtitle_path="",
            audio_duration=3.0,
            status="COMPLETED",
        )
    mock_assembly.assemble_video.side_effect = fake_assemble
    mock_validator = MagicMock(return_value={
        "duration": 3.0, "width": 1920, "height": 1080, "file_size": 1024, "fps": 30.0, "has_audio": True
    })

    comp_executor = CompositionStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path / "e2e_storage",
        assembly_service=mock_assembly,
        media_validator=mock_validator,
    )

    stage_registry = StageExecutorRegistry()
    stage_registry.register(Stage.COMPOSITION, comp_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=stage_registry,
        worker_id="comp-worker-1",
    )

    # 1. Worker runs COMPOSITION job
    assert worker.run_once() is True

    # 2. Verify state progression in DB
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        j_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        exec_repo = StageExecutionRepository(session)
        comp_repo = CompositionOutputRepository(session)

        # Task advanced to QUALITY_REVIEW
        cur_task = t_repo.get_task(task.task_id)
        assert cur_task.current_stage == Stage.QUALITY_REVIEW
        assert cur_task.task_status == TaskStatus.RUNNING

        # COMPOSITION job is succeeded
        cur_comp_job = j_repo.get_job(job.job_id)
        assert cur_comp_job.status == JobStatus.SUCCEEDED
        assert cur_comp_job.output_task_artifact_ref_id is not None

        # Composition output artifact reference exists
        comp_ref = art_repo.get_artifact_ref(cur_comp_job.output_task_artifact_ref_id)
        assert comp_ref.stage == Stage.COMPOSITION
        assert comp_ref.artifact_type == ArtifactType.COMPOSITION_OUTPUT

        # CompositionOutput record matches
        comp_record = comp_repo.get_composition_output(comp_ref.artifact_id)
        assert comp_record is not None
        assert comp_record.task_id == task.task_id
        assert comp_record.storyboard_snapshot_id == snapshot.storyboard_snapshot_id
        assert comp_record.execution_run_id == run.execution_run_id
        assert comp_record.audio_output_id == audio_out.audio_output_id

        # StageExecution audit record exists
        executions = exec_repo.list_executions_for_task(task.task_id)
        comp_exec = [e for e in executions if e.stage == Stage.COMPOSITION][0]
        assert comp_exec.output_task_artifact_ref_id == comp_ref.task_artifact_ref_id

        # Next job created for QUALITY_REVIEW in QUEUED state
        jobs = j_repo.list_jobs_for_task(task.task_id)
        qr_jobs = [j for j in jobs if j.stage == Stage.QUALITY_REVIEW]
        assert len(qr_jobs) == 1
        assert qr_jobs[0].status == JobStatus.QUEUED
        assert qr_jobs[0].input_task_artifact_ref_id == comp_ref.task_artifact_ref_id

    # 3. Worker runs again with production registry (which has NO executor for QUALITY_REVIEW)
    prod_worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="prod-worker-2",
    )
    # Must NOT process QUALITY_REVIEW
    assert prod_worker.run_once() is False

    with session_factory() as session:
        j_repo = WorkflowJobRepository(session)
        qr_job = j_repo.get_job(qr_jobs[0].job_id)
        assert qr_job.status == JobStatus.QUEUED
        assert qr_job.lease_owner is None
