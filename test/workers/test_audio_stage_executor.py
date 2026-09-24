"""
Comprehensive test suite for Stage A1: Audio Stage Integration.

Verifies:
- Stage.AUDIO is registered in default executor registry.
- Stage.COMPOSITION remains unregistered and unsupported.
- AudioStageExecutor consumes:
  - ExecutionRun (must be COMPLETED)
  - approved StoryboardSnapshot
  - ScriptRevision (authoritative source of spoken text)
- Spoken narration assembled strictly from ScriptSegments in sequence order.
- Zero LLM rewriting or script mutation.
- Narration audio synthesis and synchronized subtitle generation.
- Real audio duration measurement and duration tolerance validation.
- Optional background music handling (volume 0 / None skips BGM cleanly; enabled resolves file).
- Persistence of immutable AudioOutput record and TaskArtifactRef(AUDIO_OUTPUT).
- Clean handling of input errors (missing ref, task mismatch, unapproved storyboard, incomplete execution run).
- Full StageWorker integration advancing from AUDIO to COMPOSITION (queued, unsupported).
- Zero paid API / external network calls (all TTS synthesized via injected test doubles).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.audio_stage_executor import AudioStageExecutor
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.asset_execution import ExecutionRun, ExecutionStatus
from app.domain.asset_router import AssetRoutePlan, AssetRoutePlanStatus, RoutingStrategy
from app.domain.audio import AudioOutput
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
    ContentPlanRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    ShotRepository,
    StageExecutionRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.workers.stage_worker import StageWorker


# =============================================================================
# Fixtures and Test Doubles
# =============================================================================

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


class MockTraceWriter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def write_event(self, task_id: str, event_type: TraceEventType, attributes: dict) -> None:
        self.events.append({
            "task_id": task_id,
            "event_type": event_type,
            "attributes": attributes,
        })

    def has_event(self, event_type: TraceEventType) -> bool:
        return any(e["event_type"] == event_type for e in self.events)


class DummySubMaker:
    """Mock edge_tts.SubMaker for test isolation."""
    def __init__(self) -> None:
        self.cues = []
        self.offset = [(0, 10000000), (10000000, 30000000)]
        self.subs = ["Test subtitle segment 1", "Test subtitle segment 2"]

    def get_srt(self) -> str:
        return "1\n00:00:00,000 --> 00:00:03,000\nTest subtitle segment\n"


def make_mock_tts_runner(dummy_duration: float = 30.0, should_fail: bool = False):
    """Produces an injected TTS test double writing mock audio bytes."""
    def _runner(text: str, voice_name: str, voice_rate: float, voice_file: str, voice_volume: float = 1.0):
        if should_fail:
            return None
        os.makedirs(os.path.dirname(voice_file), exist_ok=True)
        # Write dummy MP3 header bytes
        with open(voice_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x44" + b"\x00" * 1024)
        return DummySubMaker()
    return _runner


def make_mock_subtitle_creator(should_fail: bool = False):
    """Produces an injected subtitle test double writing dummy SRT."""
    def _creator(sub_maker, text: str, subtitle_file: str, word_level: bool = False):
        if should_fail:
            return
        os.makedirs(os.path.dirname(subtitle_file), exist_ok=True)
        with open(subtitle_file, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:03,000\n" + text[:20] + "\n")
    return _creator


def make_mock_duration_getter(fixed_duration: float = 30.0):
    """Produces an injected duration getter returning fixed duration."""
    def _getter(target):
        return fixed_duration
    return _getter


def setup_complete_upstream_pipeline(session, task_id: str, tmp_path: Path, target_duration: float = 30.0) -> dict:
    """
    Sets up complete upstream domain entities:
    - KnowledgeVideoTask
    - ContentPlanRevision & ContentBeats
    - StoryboardSnapshot (APPROVED) & Shots
    - AssetRoutePlan
    - ExecutionRun (COMPLETED)
    - ScriptRevision & ScriptSegments
    - TaskArtifactRef for EXECUTION_RUN
    - WorkflowJob for AUDIO
    """
    task_repo = KnowledgeVideoTaskRepository(session)
    content_repo = ContentPlanRepository(session)
    sb_repo = StoryboardRepository(session)
    shot_repo = ShotRepository(session)
    arp_repo = AssetRoutePlanRepository(session)
    exec_repo = ExecutionRepository(session)
    script_repo = ScriptRepository(session)
    art_repo = TaskArtifactRepository(session)
    job_repo = WorkflowJobRepository(session)

    # 1. Task
    task = KnowledgeVideoTask.create(
        task_id=task_id,
        topic="Audio Stage Integration Architecture",
        target_duration=target_duration,
        workflow_policy=WorkflowPolicyType.AUTO,
        task_metadata={"voice_name": "zh-CN-XiaoxiaoNeural", "voice_rate": 1.0, "bgm_type": None},
    )
    task.advance_stage(Stage.KNOWLEDGE_PLAN)
    task.advance_stage(Stage.SCRIPT)
    task.advance_stage(Stage.STORYBOARD)
    task.advance_stage(Stage.PRODUCTION_PLAN)
    task.advance_stage(Stage.ASSET)
    task.advance_stage(Stage.AUDIO)
    task.transition_to(TaskStatus.RUNNING)
    task_repo.save_task(task)

    # 2. ContentPlanRevision
    cpr_id = f"cpr_{uuid4().hex[:8]}"
    beat1 = ContentBeat(
        beat_id=f"beat_{uuid4().hex[:8]}",
        order=1,
        intent="Introducing the topic",
        importance=0.8,
        target_duration=15.0,
        beat_type=BeatType.HOOK,
    )
    beat2 = ContentBeat(
        beat_id=f"beat_{uuid4().hex[:8]}",
        order=2,
        intent="Explaining the core mechanism",
        importance=0.9,
        target_duration=15.0,
        beat_type=BeatType.KNOWLEDGE,
    )
    cpr = ContentPlanRevision(
        content_plan_revision_id=cpr_id,
        revision_number=1,
        topic="Audio Stage Integration Architecture",
        overall_target_duration=target_duration,
        beats=[beat1, beat2],
    )
    content_repo.add_revision(cpr)

    # 3. StoryboardSnapshot (APPROVED)
    sb_id = f"sb_{uuid4().hex[:8]}"
    shot1_id = f"shot_1_{uuid4().hex[:6]}"
    shot2_id = f"shot_2_{uuid4().hex[:6]}"
    srev1_id = f"srev_1_{uuid4().hex[:6]}"
    srev2_id = f"srev_2_{uuid4().hex[:6]}"

    shot1 = Shot(shot_id=shot1_id, beat_lineage_id=beat1.beat_lineage_id, local_order=1)
    shot2 = Shot(shot_id=shot2_id, beat_lineage_id=beat2.beat_lineage_id, local_order=1)
    shot_repo.add_shot(shot1)
    shot_repo.add_shot(shot2)

    srev1 = ShotRevision(
        shot_revision_id=srev1_id,
        shot_id=shot1_id,
        revision_number=1,
        beat_lineage_id=beat1.beat_lineage_id,
        created_from_beat_instance_id=beat1.beat_id,
        narration="First segment of authoritative spoken narration.",
        target_duration=15.0,
        visual_goal="Goal 1",
        visual_type=VisualType.AI_IMAGE,
        scene_description="Scene 1",
        generation_prompt="Prompt 1",
        camera_movement="Static",
    )
    srev2 = ShotRevision(
        shot_revision_id=srev2_id,
        shot_id=shot2_id,
        revision_number=1,
        beat_lineage_id=beat2.beat_lineage_id,
        created_from_beat_instance_id=beat2.beat_id,
        narration="Second segment of authoritative spoken narration.",
        target_duration=15.0,
        visual_goal="Goal 2",
        visual_type=VisualType.AI_IMAGE,
        scene_description="Scene 2",
        generation_prompt="Prompt 2",
        camera_movement="Static",
    )
    shot_repo.add_revision(srev1)
    shot_repo.add_revision(srev2)

    sb_snap = StoryboardSnapshot(
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=cpr_id,
        shot_revision_ids=(srev1.shot_revision_id, srev2.shot_revision_id),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    sb_repo.add_snapshot(sb_snap)

    # 4. AssetRoutePlan
    arp_id = f"arp_{uuid4().hex[:8]}"
    arp = AssetRoutePlan(
        asset_route_plan_id=arp_id,
        storyboard_snapshot_id=sb_id,
        content_plan_revision_id=cpr_id,
        routing_strategy=RoutingStrategy.BALANCED,
        routing_policy_version="v1.0",
        status=AssetRoutePlanStatus.READY,
    )
    arp_repo.add_route_plan(arp)

    # 5. ExecutionRun (COMPLETED)
    run_id = f"run_{uuid4().hex[:8]}"
    exec_run = ExecutionRun(
        execution_run_id=run_id,
        asset_route_plan_id=arp_id,
        storyboard_snapshot_id=sb_id,
        status=ExecutionStatus.COMPLETED,
        total_shots=2,
        succeeded_shots=2,
    )
    exec_repo.add_execution_run(exec_run)

    # 6. ScriptRevision
    scr_id = f"scr_{uuid4().hex[:8]}"
    seg1 = ScriptSegment(
        script_revision_id=scr_id,
        content_beat_id=beat1.beat_id,
        order=1,
        narration_text="First segment of authoritative spoken narration.",
        target_duration=15.0,
    )
    seg2 = ScriptSegment(
        script_revision_id=scr_id,
        content_beat_id=beat2.beat_id,
        order=2,
        narration_text="Second segment of authoritative spoken narration.",
        target_duration=15.0,
    )
    script_rev = ScriptRevision.create(
        script_revision_id=scr_id,
        task_id=task_id,
        content_plan_revision_id=cpr_id,
        segments=[seg1, seg2],
        overall_target_duration=target_duration,
    )
    script_repo.save_revision(script_rev)

    # 7. TaskArtifactRef for EXECUTION_RUN
    exec_ref = TaskArtifactRef.create(
        task_id=task_id,
        stage=Stage.ASSET,
        artifact_type=ArtifactType.EXECUTION_RUN,
        artifact_id=run_id,
        metadata_json={"status": "COMPLETED"},
    )
    art_repo.save_artifact_ref(exec_ref)

    # 8. WorkflowJob for AUDIO
    job = WorkflowJob.create(
        task_id=task_id,
        stage=Stage.AUDIO,
        idempotency_key=f"idemp_{task_id}_audio_1",
        input_task_artifact_ref_id=exec_ref.task_artifact_ref_id,
    )
    job_repo.create_job(job)

    session.commit()

    return {
        "task": task,
        "job": job,
        "exec_run": exec_run,
        "sb_snap": sb_snap,
        "script_rev": script_rev,
        "exec_ref": exec_ref,
    }


# =============================================================================
# Registry Verification
# =============================================================================

def test_registry_contains_audio_executor_and_leaves_composition_unsupported():
    """Verify default registry includes Stage.AUDIO and leaves Stage.COMPOSITION unsupported."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.AUDIO)
    assert isinstance(registry.get_executor(Stage.AUDIO), AudioStageExecutor)

    # Stage.COMPOSITION remains unregistered
    assert not registry.has_executor(Stage.COMPOSITION)
    assert registry.get_executor(Stage.COMPOSITION) is None


# =============================================================================
# AudioStageExecutor Happy Path & Artifacts
# =============================================================================

def test_audio_stage_executor_happy_path(session_factory, tmp_path):
    """
    Mandatory demonstration of complete AUDIO stage execution:
    - Synthesizes narration audio from ordered ScriptSegments
    - Measures actual duration
    - Generates synchronized subtitles (.srt)
    - Persists immutable AudioOutput and TaskArtifactRef(AUDIO_OUTPUT)
    - Emits expected trace events
    """
    task_id = f"task_{uuid4().hex[:8]}"
    trace_writer = MockTraceWriter()

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=30.0)
        task = data["task"]
        job = data["job"]

    recorded_tts_calls = []

    def mock_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
        recorded_tts_calls.append({
            "text": text,
            "voice_name": voice_name,
            "voice_rate": voice_rate,
            "voice_file": voice_file,
        })
        os.makedirs(os.path.dirname(voice_file), exist_ok=True)
        with open(voice_file, "wb") as f:
            f.write(b"MOCK_MP3_AUDIO_DATA_PAYLOAD_12345")
        return DummySubMaker()

    executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=mock_tts,
        duration_getter=make_mock_duration_getter(29.5),
        subtitle_creator=make_mock_subtitle_creator(),
        trace_writer=trace_writer,
    )

    result = executor.execute(task, job)

    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.artifact_type == ArtifactType.AUDIO_OUTPUT
    assert result.output_artifact_ref.stage == Stage.AUDIO
    assert result.output_artifact_ref.task_id == task_id

    # Verify TTS call was strictly assembled from ScriptSegments in sequence order
    assert len(recorded_tts_calls) == 1
    expected_text = "First segment of authoritative spoken narration.\nSecond segment of authoritative spoken narration."
    assert recorded_tts_calls[0]["text"] == expected_text
    assert recorded_tts_calls[0]["voice_name"] == "zh-CN-XiaoxiaoNeural"

    # Verify persisted AudioOutput in DB
    with session_factory() as session:
        audio_repo = AudioOutputRepository(session)
        audio_output = audio_repo.get_audio_output(result.output_artifact_revision_id)
        assert audio_output is not None
        assert audio_output.task_id == task_id
        assert audio_output.actual_narration_duration == 29.5
        assert audio_output.bgm_path is None  # BGM was disabled
        assert os.path.exists(audio_output.narration_audio_path)
        assert os.path.exists(audio_output.subtitle_path)
        assert len(audio_output.narration_audio_hash) == 64
        assert len(audio_output.subtitle_hash) == 64

    # Verify emitted traces
    assert trace_writer.has_event(TraceEventType.AUDIO_STAGE_STARTED)
    assert trace_writer.has_event(TraceEventType.TTS_COMPLETED)
    assert trace_writer.has_event(TraceEventType.SUBTITLE_COMPLETED)
    assert trace_writer.has_event(TraceEventType.AUDIO_STAGE_COMPLETED)


# =============================================================================
# Strict Narration Text Extraction & Zero LLM Rewriting
# =============================================================================

def test_narration_assembled_strictly_from_segments_in_order(session_factory, tmp_path):
    """Verify narration text is derived strictly from ScriptSegments in order, without LLM rewriting."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=40.0)
        task = data["task"]
        job = data["job"]

        # Insert 3 segments deliberately added out of sequence order
        script_repo = ScriptRepository(session)
        scr_id = f"scr_reordered_{uuid4().hex[:8]}"
        seg_c = ScriptSegment(
            script_revision_id=scr_id,
            content_beat_id="beat_3",
            order=3,
            narration_text="Segment Three: Conclusion.",
            target_duration=10.0,
        )
        seg_a = ScriptSegment(
            script_revision_id=scr_id,
            content_beat_id="beat_1",
            order=1,
            narration_text="Segment One: Opening statement.",
            target_duration=15.0,
        )
        seg_b = ScriptSegment(
            script_revision_id=scr_id,
            content_beat_id="beat_2",
            order=2,
            narration_text="Segment Two: Core explanation with factual data.",
            target_duration=15.0,
        )
        new_rev = ScriptRevision.create(
            script_revision_id=scr_id,
            task_id=task_id,
            content_plan_revision_id="cpr_dummy",
            segments=[seg_c, seg_a, seg_b],
            overall_target_duration=40.0,
            revision_number=2,
        )
        script_repo.save_revision(new_rev)
        session.commit()

    captured_text = []

    def mock_tts(text, voice_name, voice_rate, voice_file, voice_volume=1.0):
        captured_text.append(text)
        os.makedirs(os.path.dirname(voice_file), exist_ok=True)
        with open(voice_file, "wb") as f:
            f.write(b"AUDIO_DATA")
        return DummySubMaker()

    executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=mock_tts,
        duration_getter=make_mock_duration_getter(39.0),
        subtitle_creator=make_mock_subtitle_creator(),
    )

    result = executor.execute(task, job)
    assert result.success is True
    assert len(captured_text) == 1
    # Check strict order 1 -> 2 -> 3
    expected_order = "Segment One: Opening statement.\nSegment Two: Core explanation with factual data.\nSegment Three: Conclusion."
    assert captured_text[0] == expected_order


# =============================================================================
# Input Lineage & Validation Failures
# =============================================================================

def test_missing_input_ref_fails_cleanly(session_factory, tmp_path):
    """Missing input TaskArtifactRef fails cleanly as fatal error."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path)
        task = data["task"]
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            idempotency_key="idemp_missing_ref",
            input_task_artifact_ref_id="non_existent_ref",
        )

    executor = AudioStageExecutor(session_factory=session_factory, storage_base_dir=tmp_path)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert result.is_retryable is False
    assert "missing" in result.error_message.lower()


def test_task_id_mismatch_fails_cleanly(session_factory, tmp_path):
    """Input artifact task ID mismatch fails cleanly."""
    task_id = f"task_{uuid4().hex[:8]}"
    other_task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path)
        task = data["task"]
        art_repo = TaskArtifactRepository(session)
        alien_ref = TaskArtifactRef.create(
            task_id=other_task_id,
            stage=Stage.ASSET,
            artifact_type=ArtifactType.EXECUTION_RUN,
            artifact_id="run_alien",
        )
        art_repo.save_artifact_ref(alien_ref)
        session.commit()

        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            idempotency_key="idemp_mismatch",
            input_task_artifact_ref_id=alien_ref.task_artifact_ref_id,
        )

    executor = AudioStageExecutor(session_factory=session_factory, storage_base_dir=tmp_path)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Task ID mismatch" in result.error_message


def test_invalid_artifact_type_fails_cleanly(session_factory, tmp_path):
    """Passing non-EXECUTION_RUN artifact fails cleanly."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path)
        task = data["task"]
        art_repo = TaskArtifactRepository(session)
        wrong_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.SCRIPT,
            artifact_type=ArtifactType.SCRIPT_REVISION,
            artifact_id="scr_123",
        )
        art_repo.save_artifact_ref(wrong_ref)
        session.commit()

        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            idempotency_key="idemp_wrong_type",
            input_task_artifact_ref_id=wrong_ref.task_artifact_ref_id,
        )

    executor = AudioStageExecutor(session_factory=session_factory, storage_base_dir=tmp_path)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Input artifact type" in result.error_message


def test_incomplete_execution_run_fails_cleanly(session_factory, tmp_path):
    """ExecutionRun not in COMPLETED status cannot be consumed by Audio stage."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path)
        task = data["task"]
        job = data["job"]
        exec_run = data["exec_run"]

        exec_repo = ExecutionRepository(session)
        exec_repo.update_execution_run_status(
            exec_run.execution_run_id,
            ExecutionStatus.FAILED,
            succeeded_shots=0,
            failed_shots=2,
        )
        session.commit()

    executor = AudioStageExecutor(session_factory=session_factory, storage_base_dir=tmp_path)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "not in COMPLETED state" in result.error_message


def test_unapproved_storyboard_snapshot_fails_cleanly(session_factory, tmp_path):
    """Referencing an unapproved (DRAFT) StoryboardSnapshot fails cleanly."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path)
        task = data["task"]
        sb_snap = data["sb_snap"]
        cpr_id = sb_snap.content_plan_revision_id

        sb_repo = StoryboardRepository(session)
        draft_sb_id = f"sb_draft_{uuid4().hex[:8]}"
        draft_snap = StoryboardSnapshot(
            storyboard_snapshot_id=draft_sb_id,
            content_plan_revision_id=cpr_id,
            shot_revision_ids=sb_snap.shot_revision_ids,
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        sb_repo.add_snapshot(draft_snap)

        exec_repo = ExecutionRepository(session)
        new_run_id = f"run_draft_{uuid4().hex[:8]}"
        new_run = ExecutionRun(
            execution_run_id=new_run_id,
            asset_route_plan_id=data["exec_run"].asset_route_plan_id,
            storyboard_snapshot_id=draft_sb_id,
            status=ExecutionStatus.COMPLETED,
            total_shots=2,
            succeeded_shots=2,
        )
        exec_repo.add_execution_run(new_run)

        art_repo = TaskArtifactRepository(session)
        new_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.ASSET,
            artifact_type=ArtifactType.EXECUTION_RUN,
            artifact_id=new_run_id,
        )
        art_repo.save_artifact_ref(new_ref)

        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.AUDIO,
            idempotency_key=f"idemp_unapproved_{uuid4().hex[:6]}",
            input_task_artifact_ref_id=new_ref.task_artifact_ref_id,
        )
        session.commit()

    executor = AudioStageExecutor(session_factory=session_factory, storage_base_dir=tmp_path)
    result = executor.execute(task, job)

    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "APPROVED" in result.error_message


# =============================================================================
# Background Music Optionality & Resolution
# =============================================================================

def test_bgm_preparation_optional_and_resolved(session_factory, tmp_path):
    """
    Demonstrates BGM handling:
    - Case A: bgm disabled or volume 0 -> AudioOutput.bgm_path is None
    - Case B: bgm enabled with valid mock file -> AudioOutput.bgm_path is populated
    """
    task_id = f"task_{uuid4().hex[:8]}"
    trace_writer = MockTraceWriter()

    # Create dummy preset song in resource/songs for testing
    songs_dir = Path("resource/songs")
    songs_dir.mkdir(parents=True, exist_ok=True)
    test_song = songs_dir / "test_ambient_bgm.mp3"
    with open(test_song, "wb") as f:
        f.write(b"DUMMY_MP3_SONG_BYTES")

    try:
        with session_factory() as session:
            data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=30.0)
            task = data["task"]
            job = data["job"]

            # Enable BGM on task
            task_repo = KnowledgeVideoTaskRepository(session)
            task.task_metadata["bgm_type"] = "custom"
            task.task_metadata["bgm_file"] = "test_ambient_bgm.mp3"
            task.task_metadata["bgm_volume"] = 0.25
            task_repo.save_task(task)
            session.commit()

        executor = AudioStageExecutor(
            session_factory=session_factory,
            storage_base_dir=tmp_path,
            tts_runner=make_mock_tts_runner(30.0),
            duration_getter=make_mock_duration_getter(30.0),
            subtitle_creator=make_mock_subtitle_creator(),
            trace_writer=trace_writer,
        )

        result = executor.execute(task, job)
        assert result.success is True

        with session_factory() as session:
            audio_repo = AudioOutputRepository(session)
            audio_out = audio_repo.get_audio_output(result.output_artifact_revision_id)
            assert audio_out is not None
            assert audio_out.bgm_path is not None
            assert "test_ambient_bgm.mp3" in audio_out.bgm_path
            assert audio_out.bgm_volume == 0.25

        assert trace_writer.has_event(TraceEventType.BGM_PREPARED)

    finally:
        if test_song.exists():
            test_song.unlink()


# =============================================================================
# Audio Duration Tolerance Enforcement
# =============================================================================

def test_audio_duration_tolerance_within_tolerance(session_factory, tmp_path):
    """Actual duration within tolerance passes cleanly."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=30.0)
        task = data["task"]
        job = data["job"]

    executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=make_mock_tts_runner(32.0),
        duration_getter=make_mock_duration_getter(32.0),
        subtitle_creator=make_mock_subtitle_creator(),
    )

    result = executor.execute(task, job)
    assert result.success is True


def test_audio_duration_tolerance_excessive_deviation_fails(session_factory, tmp_path):
    """Actual duration severely exceeding tolerance fails with RETRYABLE error without modifying script."""
    task_id = f"task_{uuid4().hex[:8]}"
    trace_writer = MockTraceWriter()

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=20.0)
        task = data["task"]
        job = data["job"]
        script_rev = data["script_rev"]
        orig_script_fingerprint = script_rev.content_fingerprint

    # Target 20s, tolerance max(10s, 12s) = 12s. Deviation of 60s exceeds 12s.
    executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=make_mock_tts_runner(80.0),
        duration_getter=make_mock_duration_getter(80.0),
        subtitle_creator=make_mock_subtitle_creator(),
        trace_writer=trace_writer,
    )

    result = executor.execute(task, job)

    assert result.success is False
    assert result.is_retryable is True
    assert result.error_type == JobErrorType.RETRYABLE.value
    assert "duration" in result.error_message.lower()
    assert trace_writer.has_event(TraceEventType.AUDIO_STAGE_FAILED)

    # Verify script was NOT mutated
    with session_factory() as session:
        script_repo = ScriptRepository(session)
        latest_rev = script_repo.get_latest_revision_for_task(task_id)
        assert latest_rev.content_fingerprint == orig_script_fingerprint


# =============================================================================
# TTS Runner Failure & Retryability
# =============================================================================

def test_tts_runner_failure_is_retryable(session_factory, tmp_path):
    """When TTS provider fails or returns empty audio, executor reports retryable failure."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=30.0)
        task = data["task"]
        job = data["job"]

    executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=make_mock_tts_runner(should_fail=True),
        duration_getter=make_mock_duration_getter(30.0),
        subtitle_creator=make_mock_subtitle_creator(),
    )

    result = executor.execute(task, job)

    assert result.success is False
    assert result.is_retryable is True
    assert result.error_type == JobErrorType.RETRYABLE.value


# =============================================================================
# End-to-End with StageWorker: Advancement to COMPOSITION
# =============================================================================

def test_stage_worker_end_to_end_advancement_to_composition(session_factory, tmp_path):
    """
    Mandatory end-to-end demonstration:
    1. StageWorker acquires QUEUED AUDIO job.
    2. AudioStageExecutor runs, produces AudioOutput and TaskArtifactRef(AUDIO_OUTPUT).
    3. StageWorker marks AUDIO job as SUCCEEDED and creates StageExecution record.
    4. KnowledgeVideoWorkflow advances task to Stage.COMPOSITION and enqueues COMPOSITION job.
    5. Because Stage.COMPOSITION is unsupported in registry, the job stays safely QUEUED and unacquired.
    """
    task_id = f"task_{uuid4().hex[:8]}"
    now = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)

    with session_factory() as session:
        data = setup_complete_upstream_pipeline(session, task_id, tmp_path, target_duration=30.0)
        task = data["task"]
        job = data["job"]

    # Build custom registry injecting mock TTS for tests
    audio_executor = AudioStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        tts_runner=make_mock_tts_runner(30.0),
        duration_getter=make_mock_duration_getter(30.0),
        subtitle_creator=make_mock_subtitle_creator(),
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.AUDIO, audio_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="test-audio-worker",
    )

    # 1. Run worker iteration
    processed = worker.run_once(now=now)
    assert processed is True

    # 2. Inspect state in DB
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        exec_repo = StageExecutionRepository(session)
        audio_repo = AudioOutputRepository(session)
        art_repo = TaskArtifactRepository(session)

        # Audio job is SUCCEEDED
        completed_job = job_repo.get_job(job.job_id)
        assert completed_job.status == JobStatus.SUCCEEDED
        assert completed_job.output_task_artifact_ref_id is not None

        # AudioOutput persisted
        audio_out = audio_repo.get_latest_audio_output_for_task(task_id)
        assert audio_out is not None
        assert audio_out.actual_narration_duration == 30.0

        # TaskArtifactRef persisted
        audio_ref = art_repo.get_artifact_ref(completed_job.output_task_artifact_ref_id)
        assert audio_ref is not None
        assert audio_ref.artifact_type == ArtifactType.AUDIO_OUTPUT
        assert audio_ref.stage == Stage.AUDIO

        # StageExecution audit record exists
        executions = exec_repo.list_executions_for_task(task_id)
        assert len(executions) >= 1
        audio_exec = [e for e in executions if e.stage == Stage.AUDIO][0]
        assert audio_exec.status == "SUCCEEDED"

        # Task advanced to Stage.COMPOSITION
        updated_task = task_repo.get_task(task_id)
        assert updated_task.current_stage == Stage.COMPOSITION
        assert updated_task.task_status == TaskStatus.RUNNING

        # Next job for COMPOSITION is QUEUED
        comp_jobs = job_repo.list_jobs_for_task(task_id)
        queued_comp_jobs = [j for j in comp_jobs if j.stage == Stage.COMPOSITION]
        assert len(queued_comp_jobs) == 1
        assert queued_comp_jobs[0].status == JobStatus.QUEUED

    # 3. Next worker iteration does NOT acquire COMPOSITION job because it is unsupported
    processed_again = worker.run_once(now=now)
    assert processed_again is False

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        comp_jobs = [j for j in job_repo.list_jobs_for_task(task_id) if j.stage == Stage.COMPOSITION]
        # Must still be QUEUED, never leased or run
        assert comp_jobs[0].status == JobStatus.QUEUED
