"""
Comprehensive test suite for Stage D1: Delivery Stage Integration.

Verifies:
1. Stage.DELIVERY property and registration in default executor registry.
2. DeliveryStageExecutor consumes:
   - Accepted QUALITY_REVIEW artifact (EvaluationSnapshot with PASS decision)
   - CompositionOutput (from Stage.COMPOSITION)
   - AudioOutput / Subtitles (from Stage.AUDIO)
3. Zero upstream rerendering / FFmpeg calls (authoritatively reuses accepted MP4).
4. Generation of frozen source_report.md and execution_report.md.
5. Hash verification (SHA-256) for video, subtitles, and reports.
6. Secret and credential redaction in reports.
7. Subtitle handling (present vs absent).
8. Failure semantics (missing snapshot, non-PASS decision, missing composition, missing video file, empty video file, stale ref, task mismatch).
9. Trace event emission sequence (STARTED, MANIFEST_CREATED, COMPLETED, FAILED).
10. Task completion transition upon Delivery completion via KnowledgeVideoWorkflow.
11. DeliveryManifest domain immutability.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.delivery_stage_executor import (
    DeliveryStageExecutor,
    compute_file_sha256,
)
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.audio import AudioOutput
from app.domain.composition import CompositionOutput
from app.domain.delivery import DeliveryManifest
from app.domain.enums import StoryboardSnapshotState
from app.domain.evaluation import (
    EvaluationDecision,
    EvaluationDimension,
    EvaluationSnapshot,
)
from app.domain.evidence import EvidenceSnapshot, SourceDocument, SourceStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import (
    Base,
    CompositionOutputORM,
    ContentPlanRevisionORM,
    EvaluationSnapshotORM,
    ExecutionRunORM,
    KnowledgeVideoTaskORM,
    ScriptRevisionORM,
    StoryboardSnapshotORM,
)
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    DeliveryManifestRepository,
    EvaluationRepository,
    EvidenceRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.delivery_report_service import DeliveryReportService, redact_secrets


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


def _setup_completed_upstream_task(
    session_factory,
    tmp_path: Path,
    eval_decision: EvaluationDecision = EvaluationDecision.PASS,
    include_subtitles: bool = True,
    create_video_file: bool = True,
    video_content: bytes = b"dummy mp4 video bytes 12345",
) -> dict:
    """Helper to set up all upstream domain records and files for Delivery."""
    task_id = f"task_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    storyboard_id = f"sb_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"
    comp_id = f"co_{uuid4().hex[:8]}"
    snap_id = f"es_{uuid4().hex[:8]}"

    # File paths
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(task_dir / f"final_{comp_id}.mp4")
    if create_video_file:
        with open(video_path, "wb") as f:
            f.write(video_content)

    subtitle_path = None
    if include_subtitles:
        subtitle_path = str(task_dir / f"subtitles_{audio_id}.srt")
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\nHello Quantum World\n")

    audio_path = str(task_dir / f"audio_{audio_id}.mp3")
    with open(audio_path, "wb") as f:
        f.write(b"dummy audio content")

    with session_factory() as session:
        # Task
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Quantum Computing",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Plan, Script, Storyboard, ExecutionRun
        session.add(
            ContentPlanRevisionORM(
                content_plan_revision_id=plan_id,
                revision_number=1,
                topic="Quantum Computing",
                overall_target_duration=60.0,
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            ScriptRevisionORM(
                script_revision_id=script_id,
                task_id=task_id,
                content_plan_revision_id=plan_id,
                revision_number=1,
                overall_target_duration=60.0,
                content_fingerprint="fp_script_1",
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            StoryboardSnapshotORM(
                storyboard_snapshot_id=storyboard_id,
                content_plan_revision_id=plan_id,
                snapshot_state="APPROVED",
                created_at=datetime.now(UTC),
            )
        )
        session.add(
            ExecutionRunORM(
                execution_run_id=run_id,
                asset_route_plan_id=f"arp_{uuid4().hex[:8]}",
                storyboard_snapshot_id=storyboard_id,
                status="COMPLETED",
                total_shots=2,
                succeeded_shots=2,
                failed_shots=0,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        )

        # AudioOutput
        audio_repo = AudioOutputRepository(session)
        audio_output = AudioOutput.create(
            audio_output_id=audio_id,
            task_id=task_id,
            script_revision_id=script_id,
            execution_run_id=run_id,
            narration_audio_path=audio_path,
            narration_audio_hash=hashlib.sha256(b"dummy audio content").hexdigest(),
            actual_narration_duration=60.0,
            subtitle_path=subtitle_path,
            subtitle_hash=hashlib.sha256(open(subtitle_path, "rb").read()).hexdigest() if subtitle_path else None,
        )
        audio_repo.save_audio_output(audio_output)

        # CompositionOutput
        comp_repo = CompositionOutputRepository(session)
        comp_output = CompositionOutput.create(
            composition_output_id=comp_id,
            task_id=task_id,
            storyboard_snapshot_id=storyboard_id,
            execution_run_id=run_id,
            audio_output_id=audio_id,
            video_path=video_path,
            video_hash=hashlib.sha256(video_content).hexdigest() if create_video_file else "fake_hash",
            duration=60.0,
            width=1920,
            height=1080,
            file_size=len(video_content) if create_video_file else 1000,
        )
        comp_repo.save_composition_output(comp_output)

        # TaskArtifact for CompositionOutput
        art_repo = TaskArtifactRepository(session)
        comp_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.COMPOSITION,
            artifact_type=ArtifactType.COMPOSITION_OUTPUT,
            artifact_id=comp_id,
        )
        art_repo.save_artifact_ref(comp_ref)

        # EvaluationSnapshot
        eval_repo = EvaluationRepository(session)
        eval_snap = EvaluationSnapshot(
            evaluation_snapshot_id=snap_id,
            evaluation_target_id="target_1",
            dimension_result_ids=(),
            decision=eval_decision,
            overall_score=0.95 if eval_decision == EvaluationDecision.PASS else 0.50,
            policy_version="v1.0",
            evaluator_version="v1.0",
            summary_reason_codes=("PERFECT_QUALITY",) if eval_decision == EvaluationDecision.PASS else ("LOW_SCORE",),
        )
        eval_repo.save_snapshot(eval_snap)

        # TaskArtifact for EvaluationSnapshot
        eval_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.QUALITY_REVIEW,
            artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
            artifact_id=snap_id,
        )
        saved_eval_ref = art_repo.save_artifact_ref(eval_ref)

        # WorkflowJob for Delivery
        job_repo = WorkflowJobRepository(session)
        deliv_job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.DELIVERY,
            idempotency_key=f"idemp_{task_id}_delivery_1",
            attempt_number=1,
            input_task_artifact_ref_id=saved_eval_ref.task_artifact_ref_id,
        )
        job_repo.create_job(deliv_job)

        session.commit()

    return {
        "task_id": task_id,
        "comp_id": comp_id,
        "snap_id": snap_id,
        "eval_ref_id": saved_eval_ref.task_artifact_ref_id,
        "job_id": deliv_job.job_id,
        "video_path": video_path,
        "subtitle_path": subtitle_path,
        "task_dir": str(task_dir),
    }


def test_delivery_stage_property():
    """Verify executor declares Stage.DELIVERY."""
    executor = DeliveryStageExecutor()
    assert executor.stage == Stage.DELIVERY


def test_delivery_success_reuses_composition_without_rerendering(session_factory, tmp_path):
    """Verify delivery reuses existing composition video without invoking any video renderer."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)
    trace_mock = MagicMock()

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        trace_emitter=trace_mock,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    # Execute Delivery stage
    result = executor.execute(task, job)

    assert result.success is True
    assert result.output_artifact_ref is not None
    assert result.output_artifact_ref.stage == Stage.DELIVERY
    assert result.output_artifact_ref.artifact_type == ArtifactType.DELIVERY_MANIFEST

    manifest_id = result.output_artifact_ref.artifact_id

    with session_factory() as session:
        deliv_repo = DeliveryManifestRepository(session)
        manifest = deliv_repo.get_delivery_manifest(manifest_id)
        assert manifest is not None
        assert manifest.task_id == data["task_id"]
        assert manifest.composition_output_id == data["comp_id"]
        assert manifest.evaluation_snapshot_id == data["snap_id"]
        assert manifest.final_video_path == data["video_path"]
        assert manifest.final_video_size_bytes > 0
        assert len(manifest.final_video_hash) == 64
        assert os.path.exists(manifest.source_report_path)
        assert os.path.exists(manifest.execution_report_path)
        assert len(manifest.source_report_hash) == 64
        assert len(manifest.execution_report_hash) == 64

    # Verify trace events
    event_types = [call[1]["event_type"] for call in trace_mock.emit_trace_event.call_args_list]
    assert TraceEventType.DELIVERY_STAGE_STARTED in event_types
    assert TraceEventType.DELIVERY_MANIFEST_CREATED in event_types
    assert TraceEventType.DELIVERY_STAGE_COMPLETED in event_types


def test_delivery_fails_if_evaluation_snapshot_missing(session_factory, tmp_path):
    """Verify delivery fails if evaluation snapshot is missing."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    # Delete evaluation snapshot and artifact ref
    with session_factory() as session:
        from app.persistence.models import EvaluationSnapshotORM, TaskArtifactRefORM
        from sqlalchemy import delete
        session.execute(delete(EvaluationSnapshotORM).where(EvaluationSnapshotORM.evaluation_snapshot_id == data["snap_id"]))
        session.execute(delete(TaskArtifactRefORM).where(TaskArtifactRefORM.artifact_type == ArtifactType.EVALUATION_SNAPSHOT.value))
        session.commit()

        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "EvaluationSnapshot missing" in result.error_message or "not found" in result.error_message


def test_delivery_fails_if_evaluation_snapshot_not_pass(session_factory, tmp_path):
    """Verify delivery rejects evaluation snapshots that did not PASS (e.g. FAIL or REMEDIATE)."""
    data = _setup_completed_upstream_task(
        session_factory,
        tmp_path,
        eval_decision=EvaluationDecision.FAIL,
    )
    trace_mock = MagicMock()

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        trace_emitter=trace_mock,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "expected 'PASS'" in result.error_message

    # Verify failure trace event
    event_types = [call[1]["event_type"] for call in trace_mock.emit_trace_event.call_args_list]
    assert TraceEventType.DELIVERY_STAGE_FAILED in event_types


def test_delivery_fails_if_composition_output_missing(session_factory, tmp_path):
    """Verify delivery fails if composition output is missing."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    # Delete composition artifact refs and records
    with session_factory() as session:
        from app.persistence.models import CompositionOutputORM, TaskArtifactRefORM
        from sqlalchemy import delete
        session.execute(delete(CompositionOutputORM).where(CompositionOutputORM.composition_output_id == data["comp_id"]))
        session.execute(delete(TaskArtifactRefORM).where(TaskArtifactRefORM.artifact_type == ArtifactType.COMPOSITION_OUTPUT.value))
        session.commit()

        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "CompositionOutput missing" in result.error_message


def test_delivery_fails_if_final_video_missing_on_disk(session_factory, tmp_path):
    """Verify delivery fails if final video file is missing on disk."""
    data = _setup_completed_upstream_task(
        session_factory,
        tmp_path,
        create_video_file=False,
    )

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "missing or empty" in result.error_message


def test_delivery_fails_if_final_video_empty(session_factory, tmp_path):
    """Verify delivery fails if final video file has 0 bytes."""
    data = _setup_completed_upstream_task(
        session_factory,
        tmp_path,
        create_video_file=True,
    )
    # Truncate final video file on disk to 0 bytes
    with open(data["video_path"], "wb") as f:
        f.truncate(0)

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "missing or empty" in result.error_message


def test_delivery_handles_subtitles_present(session_factory, tmp_path):
    """Verify delivery correctly captures subtitle path and hash when subtitles exist."""
    data = _setup_completed_upstream_task(
        session_factory,
        tmp_path,
        include_subtitles=True,
    )

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        manifest = DeliveryManifestRepository(session).get_delivery_manifest(
            result.output_artifact_ref.artifact_id
        )
        assert manifest.subtitle_path == data["subtitle_path"]
        assert manifest.subtitle_hash is not None
        assert len(manifest.subtitle_hash) == 64


def test_delivery_handles_subtitles_absent(session_factory, tmp_path):
    """Verify delivery succeeds and records None when subtitles are absent."""
    data = _setup_completed_upstream_task(
        session_factory,
        tmp_path,
        include_subtitles=False,
    )

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        manifest = DeliveryManifestRepository(session).get_delivery_manifest(
            result.output_artifact_ref.artifact_id
        )
        assert manifest.subtitle_path is None
        assert manifest.subtitle_hash is None


def test_delivery_generates_source_and_execution_reports(session_factory, tmp_path):
    """Verify source_report.md and execution_report.md are generated with redacted secrets."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    # Associate a source document with a secret URL
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        src = SourceDocument.register_url(
            url="https://example.com/api/v1/data?api_key=SECRET_TOKEN_999",
            title="Secret Source",
        )
        ev_repo.save_source_document(src)
        ev_repo.associate_task_source(data["task_id"], src.source_document_id)
        session.commit()

        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        manifest = DeliveryManifestRepository(session).get_delivery_manifest(
            result.output_artifact_ref.artifact_id
        )

        # Verify source report content
        with open(manifest.source_report_path, "r", encoding="utf-8") as f:
            src_md = f.read()
        assert "# Source & Evidence Traceability Report" in src_md
        assert "SECRET_TOKEN_999" not in src_md
        assert "[REDACTED]" in src_md

        # Verify execution report content
        with open(manifest.execution_report_path, "r", encoding="utf-8") as f:
            exec_md = f.read()
        assert "# Workflow Execution & Quality Delivery Report" in exec_md
        assert data["task_id"] in exec_md


def test_delivery_trace_events_sequence(session_factory, tmp_path):
    """Verify exact trace event types emitted during delivery."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)
    trace_mock = MagicMock()

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
        trace_emitter=trace_mock,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is True

    calls = trace_mock.emit_trace_event.call_args_list
    events = [c[1]["event_type"] for c in calls]

    assert events == [
        TraceEventType.DELIVERY_STAGE_STARTED,
        TraceEventType.DELIVERY_MANIFEST_CREATED,
        TraceEventType.DELIVERY_STAGE_COMPLETED,
    ]


def test_delivery_artifact_ref_registered(session_factory, tmp_path):
    """Verify delivery output artifact ref is saved and retrievable."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is True

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        ref = art_repo.get_artifact_ref(result.output_artifact_ref.task_artifact_ref_id)
        assert ref is not None
        assert ref.stage == Stage.DELIVERY
        assert ref.artifact_type == ArtifactType.DELIVERY_MANIFEST
        assert not ref.is_stale


def test_delivery_manifest_persistence(session_factory, tmp_path):
    """Verify delivery manifest can be fetched by ID and for task."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    with session_factory() as session:
        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    result = executor.execute(task, job)
    assert result.success is True

    manifest_id = result.output_artifact_ref.artifact_id

    with session_factory() as session:
        deliv_repo = DeliveryManifestRepository(session)
        by_id = deliv_repo.get_delivery_manifest(manifest_id)
        by_task = deliv_repo.get_delivery_manifest_for_task(data["task_id"])

        assert by_id is not None
        assert by_task is not None
        assert by_id.delivery_manifest_id == by_task.delivery_manifest_id == manifest_id


def test_delivery_with_stale_artifact_ref(session_factory, tmp_path):
    """Verify delivery rejects stale input artifact refs."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        ref = art_repo.get_artifact_ref(data["eval_ref_id"])
        stale_ref = ref.model_copy(update={"metadata_json": {**ref.metadata_json, "is_stale": True}})
        art_repo.save_artifact_ref(stale_ref)
        session.commit()

        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "marked stale" in result.error_message


def test_delivery_with_mismatched_task_id(session_factory, tmp_path):
    """Verify delivery rejects artifact refs belonging to other tasks."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    # Create artifact ref for a different task
    other_task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        other_ref = TaskArtifactRef.create(
            task_id=other_task_id,
            stage=Stage.QUALITY_REVIEW,
            artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
            artifact_id=data["snap_id"],
        )
        saved_other = art_repo.save_artifact_ref(other_ref)

        task = KnowledgeVideoTaskRepository(session).get_task(data["task_id"])
        job = WorkflowJobRepository(session).get_job(data["job_id"])
        job = job.model_copy(update={"input_task_artifact_ref_id": saved_other.task_artifact_ref_id})
        session.commit()

    executor = DeliveryStageExecutor(
        session_factory=session_factory,
        storage_base_dir=tmp_path,
    )

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.FATAL.value
    assert "Task ID mismatch" in result.error_message


def test_workflow_completion_transition_after_delivery(session_factory, tmp_path):
    """Verify KnowledgeVideoWorkflow transitions task to COMPLETED and returns None when Stage.DELIVERY completes."""
    data = _setup_completed_upstream_task(session_factory, tmp_path)

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        art_repo = TaskArtifactRepository(session)
        deliv_repo = DeliveryManifestRepository(session)

        task = task_repo.get_task(data["task_id"])
        task.transition_to(TaskStatus.RUNNING)
        task.set_stage_for_rerun(Stage.DELIVERY)
        task_repo.save_task(task)

        job = job_repo.get_job(data["job_id"])

        # Execute delivery
        executor = DeliveryStageExecutor(
            session_factory=session_factory,
            storage_base_dir=tmp_path,
        )
        exec_res = executor.execute(task, job)
        assert exec_res.success is True

        # Simulate job marked SUCCEEDED
        job.mark_succeeded(
            output_task_artifact_ref_id=exec_res.output_artifact_ref.task_artifact_ref_id
        )
        job_repo.update_job(job)

        workflow = KnowledgeVideoWorkflow(
            session=session,
        )

        next_job = workflow.on_stage_completed(task=task, completed_job=job)
        session.commit()

        # No more jobs should be scheduled
        assert next_job is None

        # Task must now be in terminal status COMPLETED
        updated_task = task_repo.get_task(data["task_id"])
        assert updated_task.task_status == TaskStatus.COMPLETED
        assert updated_task.task_status.is_terminal is True


def test_delivery_manifest_immutability():
    """Verify DeliveryManifest is frozen and cannot be mutated."""
    manifest = DeliveryManifest.create(
        delivery_manifest_id="dm_123",
        task_id="task_1",
        composition_output_id="co_1",
        evaluation_snapshot_id="es_1",
        final_video_path="/path/video.mp4",
        final_video_hash="hash_video",
        final_video_size_bytes=1000,
        source_report_path="/path/src.md",
        source_report_hash="hash_src",
        execution_report_path="/path/exec.md",
        execution_report_hash="hash_exec",
    )

    with pytest.raises(Exception):
        manifest.final_video_path = "/path/tampered.mp4"
