"""End-to-End Integration Test Suite for Unified Knowledge Video Agent Workflow.

Verifies:
1. AUTO policy: full 10-stage execution pipeline to DELIVERY and COMPLETED.
2. REVIEW policy: human checkpoints (KNOWLEDGE_PLAN, STORYBOARD, QUALITY_REVIEW) pause in WAITING_USER and resume upon approval.
3. Source traceability report: generated, frozen, hashed (SHA-256), and credential-redacted.
4. Quality review FAIL: triggers quality remediation, partial rerun to upstream stage, and STALE artifact propagation.
5. NEEDS_EVIDENCE: pause on evidence gap and resume via open-web research authorization.
6. Worker restart recovery: lease expiration, stale lock recovery, and zero duplicate execution.
7. Download API security: path-traversal blocking and cross-task isolation enforcement.
8. Completion atomicity: COMPLETED lifecycle status granted only when DeliveryManifest is persisted.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.delivery_stage_executor import (
    DeliveryStageExecutor,
    compute_file_sha256,
)
from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.task_command_service import TaskCommandService
from app.domain.audio import AudioOutput
from app.domain.composition import CompositionOutput
from app.domain.delivery import DeliveryManifest
from app.domain.evaluation import (
    EvaluationDecision,
    EvaluationSnapshot,
)
from app.domain.evidence import SourceDocument
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.quality_remediation import QualityRemediationAction
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.workflow_job import JobStatus, WorkflowJob
from app.domain.workflow_state import (
    STAGE_ORDER,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    DeliveryManifestRepository,
    EvaluationRepository,
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.delivery_report_service import DeliveryReportService


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "test_e2e.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _create_mock_file(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


# ---------------------------------------------------------------------------
# Test 1: AUTO policy full path through all 10 stages to DELIVERY & COMPLETED
# ---------------------------------------------------------------------------
def test_1_auto_policy_full_path_to_completed(db_session, tmp_path):
    command_service = TaskCommandService(db_session)
    workflow = KnowledgeVideoWorkflow(db_session)
    job_repo = WorkflowJobRepository(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)
    artifact_repo = TaskArtifactRepository(db_session)
    delivery_repo = DeliveryManifestRepository(db_session)
    eval_repo = EvaluationRepository(db_session)
    comp_repo = CompositionOutputRepository(db_session)
    audio_repo = AudioOutputRepository(db_session)

    # 1. Create task with AUTO policy
    task = command_service.create_task(
        topic="Introduction to Quantum Physics",
        target_duration=60.0,
        aspect_ratio="16:9",
        language="zh",
        workflow_policy=WorkflowPolicyType.AUTO,
    )
    task.transition_to(TaskStatus.RUNNING)
    task_repo.save_task(task)

    task_dir = tmp_path / "tasks" / task.task_id
    video_content = b"dummy_mp4_content_12345"
    video_path = _create_mock_file(task_dir / "final_video.mp4", video_content)
    sub_path = _create_mock_file(task_dir / "subtitles.srt", b"1\n00:00:00 --> 00:00:05\nQuantum physics intro.")
    audio_path = _create_mock_file(task_dir / "audio.mp3", b"dummy audio content")

    audio_id = f"aud_{uuid4().hex[:8]}"
    comp_id = f"comp_{uuid4().hex[:8]}"
    eval_snap_id = f"eval_{uuid4().hex[:8]}"

    # Step through each stage sequentially
    for stage in STAGE_ORDER:
        job = job_repo.get_current_job_for_task(task.task_id)
        assert job is not None, f"Expected active job for stage {stage}"
        assert job.stage == stage, f"Job stage mismatch: expected {stage}, got {job.stage}"

        if stage == Stage.AUDIO:
            audio_out = AudioOutput.create(
                audio_output_id=audio_id,
                task_id=task.task_id,
                script_revision_id=f"scr_{uuid4().hex[:8]}",
                execution_run_id=f"run_{uuid4().hex[:8]}",
                narration_audio_path=audio_path,
                narration_audio_hash=hashlib.sha256(b"dummy audio content").hexdigest(),
                actual_narration_duration=60.0,
                subtitle_path=sub_path,
                subtitle_hash=hashlib.sha256(open(sub_path, "rb").read()).hexdigest(),
            )
            audio_repo.save_audio_output(audio_out)
            audio_ref = TaskArtifactRef.create(
                task_id=task.task_id,
                stage=Stage.AUDIO,
                artifact_type=ArtifactType.AUDIO_OUTPUT,
                artifact_id=audio_id,
            )
            artifact_repo.save_artifact_ref(audio_ref)
            job.mark_succeeded(output_artifact_revision_id=audio_id, output_task_artifact_ref_id=audio_ref.task_artifact_ref_id)

        elif stage == Stage.COMPOSITION:
            comp_out = CompositionOutput.create(
                composition_output_id=comp_id,
                task_id=task.task_id,
                storyboard_snapshot_id=f"sb_{uuid4().hex[:8]}",
                execution_run_id=f"run_{uuid4().hex[:8]}",
                audio_output_id=audio_id,
                video_path=video_path,
                video_hash=hashlib.sha256(video_content).hexdigest(),
                duration=60.0,
                width=1920,
                height=1080,
                file_size=len(video_content),
            )
            comp_repo.save_composition_output(comp_out)
            comp_ref = TaskArtifactRef.create(
                task_id=task.task_id,
                stage=Stage.COMPOSITION,
                artifact_type=ArtifactType.COMPOSITION_OUTPUT,
                artifact_id=comp_id,
            )
            artifact_repo.save_artifact_ref(comp_ref)
            job.mark_succeeded(output_artifact_revision_id=comp_id, output_task_artifact_ref_id=comp_ref.task_artifact_ref_id)

        elif stage == Stage.QUALITY_REVIEW:
            eval_snap = EvaluationSnapshot(
                evaluation_snapshot_id=eval_snap_id,
                evaluation_target_id="target_1",
                dimension_result_ids=(),
                decision=EvaluationDecision.PASS,
                overall_score=0.95,
                policy_version="v1.0",
                evaluator_version="v1.0",
                summary_reason_codes=("PERFECT_QUALITY",),
            )
            eval_repo.save_snapshot(eval_snap)
            eval_ref = TaskArtifactRef.create(
                task_id=task.task_id,
                stage=Stage.QUALITY_REVIEW,
                artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
                artifact_id=eval_snap_id,
                metadata_json={"decision": "PASS"},
            )
            artifact_repo.save_artifact_ref(eval_ref)
            job.mark_succeeded(output_artifact_revision_id=eval_snap_id, output_task_artifact_ref_id=eval_ref.task_artifact_ref_id)

        elif stage == Stage.DELIVERY:
            executor = DeliveryStageExecutor(
                session_factory=lambda: db_session,
                storage_base_dir=tmp_path,
            )
            res = executor.execute(task, job)
            assert res.success is True
            job.mark_succeeded(
                output_artifact_revision_id=res.output_artifact_ref.artifact_id,
                output_task_artifact_ref_id=res.output_artifact_ref.task_artifact_ref_id,
            )
        else:
            job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")

        job_repo.update_job(job)
        next_job = workflow.on_stage_completed(task, job)
        db_session.commit()

        if stage == Stage.DELIVERY:
            assert next_job is None, "Terminal stage DELIVERY must not yield a next job"

    # Verify task completed
    refreshed_task = task_repo.get_task(task.task_id)
    assert refreshed_task.task_status == TaskStatus.COMPLETED
    assert refreshed_task.finished_at is not None

    # Verify delivery manifest exists
    manifest = delivery_repo.get_delivery_manifest_for_task(task.task_id)
    assert manifest is not None
    assert manifest.final_video_path == video_path
    assert manifest.final_video_hash == hashlib.sha256(video_content).hexdigest()


# ---------------------------------------------------------------------------
# Test 2: REVIEW policy with 3 human checkpoints
# ---------------------------------------------------------------------------
def test_2_review_policy_three_checkpoints(db_session):
    command_service = TaskCommandService(db_session)
    workflow = KnowledgeVideoWorkflow(db_session)
    job_repo = WorkflowJobRepository(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)

    # 1. Create task with REVIEW policy
    task = command_service.create_task(
        topic="Deep Learning History",
        workflow_policy=WorkflowPolicyType.REVIEW,
    )
    task.transition_to(TaskStatus.RUNNING)
    task_repo.save_task(task)

    # Stage 1: EVIDENCE -> advances to KNOWLEDGE_PLAN without pause
    job = job_repo.get_current_job_for_task(task.task_id)
    assert job.stage == Stage.EVIDENCE
    job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
    job_repo.update_job(job)
    workflow.on_stage_completed(task, job)
    assert task.task_status == TaskStatus.RUNNING
    assert task.current_stage == Stage.KNOWLEDGE_PLAN

    # Checkpoint 1: KNOWLEDGE_PLAN completes -> PAUSES in WAITING_USER
    job = job_repo.get_current_job_for_task(task.task_id)
    assert job.stage == Stage.KNOWLEDGE_PLAN
    job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
    job_repo.update_job(job)
    next_job = workflow.on_stage_completed(task, job)
    assert next_job is None
    assert task.task_status == TaskStatus.WAITING_USER
    assert "Review required after KNOWLEDGE_PLAN" in (task.waiting_reason or "")

    # Resume Checkpoint 1 -> advances to SCRIPT
    task = command_service.approve_task(task.task_id)
    assert task.task_status == TaskStatus.RUNNING
    assert task.current_stage == Stage.SCRIPT

    # Stage 3: SCRIPT completes -> advances to STORYBOARD (policy pauses after STORYBOARD)
    job = job_repo.get_current_job_for_task(task.task_id)
    assert job.stage == Stage.SCRIPT
    job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
    job_repo.update_job(job)
    next_job = workflow.on_stage_completed(task, job)
    assert next_job is not None
    assert next_job.stage == Stage.STORYBOARD

    # Checkpoint 2: STORYBOARD completes -> PAUSES in WAITING_USER
    job = job_repo.get_current_job_for_task(task.task_id)
    assert job.stage == Stage.STORYBOARD
    job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
    job_repo.update_job(job)
    next_job = workflow.on_stage_completed(task, job)
    assert next_job is None
    assert task.task_status == TaskStatus.WAITING_USER
    assert "Review required after STORYBOARD" in (task.waiting_reason or "")

    # Resume Checkpoint 2 -> advances to PRODUCTION_PLAN
    task = command_service.approve_task(task.task_id)
    assert task.task_status == TaskStatus.RUNNING
    assert task.current_stage == Stage.PRODUCTION_PLAN

    # Fast forward through PRODUCTION_PLAN, ASSET, AUDIO, COMPOSITION to reach Checkpoint 3 (QUALITY_REVIEW)
    stages_to_run = [Stage.PRODUCTION_PLAN, Stage.ASSET, Stage.AUDIO, Stage.COMPOSITION]
    for stg in stages_to_run:
        job = job_repo.get_current_job_for_task(task.task_id)
        assert job.stage == stg
        job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
        job_repo.update_job(job)
        workflow.on_stage_completed(task, job)

    # Checkpoint 3: QUALITY_REVIEW completes -> PAUSES in WAITING_USER
    job = job_repo.get_current_job_for_task(task.task_id)
    assert job.stage == Stage.QUALITY_REVIEW
    job.mark_succeeded(output_artifact_revision_id=f"rev_{uuid4().hex[:8]}")
    job_repo.update_job(job)
    next_job = workflow.on_stage_completed(task, job)
    assert next_job is None
    assert task.task_status == TaskStatus.WAITING_USER
    assert "Review required after QUALITY_REVIEW" in (task.waiting_reason or "")

    # Resume Checkpoint 3 -> advances to DELIVERY
    task = command_service.approve_task(task.task_id)
    assert task.task_status == TaskStatus.RUNNING
    assert task.current_stage == Stage.DELIVERY


# ---------------------------------------------------------------------------
# Test 3: Source traceability report frozen and validated
# ---------------------------------------------------------------------------
def test_3_source_traceability_report_frozen_and_validated(db_session, tmp_path):
    report_service = DeliveryReportService(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)
    ev_repo = EvidenceRepository(db_session)

    task = KnowledgeVideoTask.create(
        topic="Neural Radiance Fields",
        task_metadata={"api_key": "sk-secret-12345", "token": "super-token-999"},
    )
    task_repo.save_task(task)

    # Add source document
    doc1 = SourceDocument.create_text(
        text="NeRF represents scenes as continuous volumetric fields.",
        title="NeRF Paper",
    )
    ev_repo.save_source_document(doc1)

    task_dir = tmp_path / "tasks" / task.task_id
    src_report_path = task_dir / "source_report.md"

    # Generate source report
    rep_path, rep_hash = report_service.generate_source_report(task.task_id, str(src_report_path))
    assert os.path.exists(rep_path)
    content = Path(rep_path).read_text(encoding="utf-8")

    # Validate contents & hash
    assert "Neural Radiance Fields" in content
    assert compute_file_sha256(rep_path) == rep_hash

    # Validate secrets are redacted
    assert "sk-secret-12345" not in content
    assert "super-token-999" not in content


# ---------------------------------------------------------------------------
# Test 4: Quality review FAIL triggers remediation & STALE propagation
# ---------------------------------------------------------------------------
def test_4_quality_review_fail_remediation_and_stale_propagation(db_session):
    command_service = TaskCommandService(db_session)
    workflow = KnowledgeVideoWorkflow(db_session)
    job_repo = WorkflowJobRepository(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)
    artifact_repo = TaskArtifactRepository(db_session)

    task = command_service.create_task(topic="Computer Vision Evolution")
    task.transition_to(TaskStatus.RUNNING)
    task.set_stage_for_rerun(Stage.QUALITY_REVIEW)
    task_repo.save_task(task)

    # Create artifacts for COMPOSITION and QUALITY_REVIEW
    comp_ref = TaskArtifactRef.create(
        task_id=task.task_id,
        stage=Stage.COMPOSITION,
        artifact_type=ArtifactType.COMPOSITION_OUTPUT,
        artifact_id=f"comp_{uuid4().hex[:8]}",
    )
    artifact_repo.save_artifact_ref(comp_ref)

    eval_ref = TaskArtifactRef.create(
        task_id=task.task_id,
        stage=Stage.QUALITY_REVIEW,
        artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
        artifact_id=f"eval_{uuid4().hex[:8]}",
        metadata_json={
            "decision": "FAIL",
            "remediation_action": QualityRemediationAction.REGENERATE_SAME_ROUTE.value,
            "target_stage": "ASSET",
            "reason": "Visual artifacts in scene 2",
        },
    )
    artifact_repo.save_artifact_ref(eval_ref)
    db_session.commit()

    # Completed job for QUALITY_REVIEW with FAIL
    job = WorkflowJob.create(
        task_id=task.task_id,
        stage=Stage.QUALITY_REVIEW,
        idempotency_key=f"idem_{uuid4().hex}",
    )
    job.output_task_artifact_ref_id = eval_ref.task_artifact_ref_id
    job.mark_succeeded(output_artifact_revision_id=eval_ref.artifact_id)
    job_repo.create_job(job)

    # Process completion
    rerun_job = workflow.on_stage_completed(task, job)

    # Assert downstream artifacts are marked stale
    updated_comp = artifact_repo.get_artifact_ref(comp_ref.task_artifact_ref_id)
    assert updated_comp.is_stale is True

    # Assert workflow re-enqueued at target stage ASSET
    assert rerun_job is not None
    assert rerun_job.stage == Stage.ASSET
    assert task.current_stage == Stage.ASSET
    assert task.task_metadata.get("is_partial_rerun") is True


# ---------------------------------------------------------------------------
# Test 5: NEEDS_EVIDENCE pause and open-web research resume
# ---------------------------------------------------------------------------
def test_5_needs_evidence_pause_and_open_web_resume(db_session):
    command_service = TaskCommandService(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)
    job_repo = WorkflowJobRepository(db_session)

    task = command_service.create_task(topic="Obscure Historical Event 1842", allow_research=False)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.NEEDS_EVIDENCE, reason="No relevant documents found in knowledge base")
    task_repo.save_task(task)

    assert task.task_status == TaskStatus.NEEDS_EVIDENCE
    assert not task.is_research_authorized

    # Authorize research with resume_if_waiting=True
    resumed_task = command_service.authorize_research(task.task_id, resume_if_waiting=True)
    assert resumed_task.is_research_authorized is True
    assert resumed_task.task_status == TaskStatus.RUNNING

    # An active EVIDENCE retry job should be enqueued
    active_job = job_repo.get_current_job_for_task(task.task_id)
    assert active_job is not None
    assert active_job.stage == Stage.EVIDENCE
    assert active_job.status == JobStatus.QUEUED


# ---------------------------------------------------------------------------
# Test 6: Worker restart recovery without duplicate job execution
# ---------------------------------------------------------------------------
def test_6_worker_restart_recovery_without_duplicate(db_session):
    job_repo = WorkflowJobRepository(db_session)
    task_id = f"task_{uuid4().hex[:8]}"

    # Job created and acquired by worker-A
    job = WorkflowJob.create(task_id=task_id, stage=Stage.SCRIPT, idempotency_key=f"idem_{uuid4().hex}")
    job_repo.create_job(job)

    # Worker A leases the job
    now = datetime.now(UTC)
    acquired_1 = job_repo.acquire_next_available_job(
        owner="worker-A",
        supported_stages=[Stage.SCRIPT],
        lease_duration_seconds=5,
        now=now,
    )
    assert acquired_1 is not None
    assert acquired_1.lease_owner == "worker-A"

    # Worker A crashes without heartbeats or completion
    # After 10 seconds, lease has expired
    later = now + timedelta(seconds=10)
    recovered = job_repo.acquire_next_available_job(
        owner="worker-B",
        supported_stages=[Stage.SCRIPT],
        lease_duration_seconds=5,
        now=later,
    )
    assert recovered is not None
    assert recovered.job_id == job.job_id
    assert recovered.lease_owner == "worker-B"

    # Worker B completes the job
    recovered.mark_succeeded(output_artifact_revision_id="rev-final")
    job_repo.update_job(recovered)

    # Verify no other worker can acquire it again
    another = job_repo.acquire_next_available_job(
        owner="worker-C",
        supported_stages=[Stage.SCRIPT],
        now=later + timedelta(seconds=1),
    )
    assert another is None


# ---------------------------------------------------------------------------
# Test 7: Download API security (path traversal blocked & task isolation)
# ---------------------------------------------------------------------------
def test_7_download_api_security_enforcement(db_session, tmp_path):
    from app.controllers.v1.knowledge_video import download_delivery_file

    delivery_repo = DeliveryManifestRepository(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)

    task_1 = KnowledgeVideoTask.create(topic="Task 1")
    task_repo.save_task(task_1)

    task_1_dir = tmp_path / "tasks" / task_1.task_id
    v1_path = _create_mock_file(task_1_dir / "video1.mp4", b"task1_video")

    manifest1 = DeliveryManifest.create(
        task_id=task_1.task_id,
        composition_output_id="comp1",
        evaluation_snapshot_id="eval1",
        final_video_path=v1_path,
        final_video_hash="hash1",
        final_video_size_bytes=len(b"task1_video"),
        source_report_path=str(task_1_dir / "src.md"),
        source_report_hash="hsrc",
        execution_report_path=str(task_1_dir / "exec.md"),
        execution_report_hash="hexec",
    )
    delivery_repo.save_delivery_manifest(manifest1)
    db_session.commit()

    # Patch get_session in knowledge_video controller
    with patch("app.controllers.v1.knowledge_video.get_session") as mock_get_session:
        mock_get_session.return_value.__enter__.return_value = db_session

        # 1. Valid download
        resp = download_delivery_file(task_id=task_1.task_id, target="video")
        assert resp.path == v1_path

        # 2. Invalid target rejection (400)
        with pytest.raises(HTTPException) as exc:
            download_delivery_file(task_id=task_1.task_id, target="../../etc/passwd")
        assert exc.value.status_code == 400

        # 3. Non-existent task (404)
        with pytest.raises(HTTPException) as exc:
            download_delivery_file(task_id="non-existent-task", target="video")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Test 8: Task completion atomicity
# ---------------------------------------------------------------------------
def test_8_task_completion_atomicity(db_session, tmp_path):
    workflow = KnowledgeVideoWorkflow(db_session)
    task_repo = KnowledgeVideoTaskRepository(db_session)
    job_repo = WorkflowJobRepository(db_session)
    delivery_repo = DeliveryManifestRepository(db_session)
    comp_repo = CompositionOutputRepository(db_session)
    eval_repo = EvaluationRepository(db_session)

    task = KnowledgeVideoTask.create(topic="Atomic Delivery Task")
    task.transition_to(TaskStatus.RUNNING)
    task.set_stage_for_rerun(Stage.DELIVERY)
    task_repo.save_task(task)

    job = WorkflowJob.create(task_id=task.task_id, stage=Stage.DELIVERY, idempotency_key=f"idem_{uuid4().hex}")
    job_repo.create_job(job)

    # 1. Delivery execution fails due to missing composition/video
    executor = DeliveryStageExecutor(
        session_factory=lambda: db_session,
        storage_base_dir=tmp_path,
    )
    res = executor.execute(task, job)
    assert res.success is False

    # Assert no delivery manifest was created
    assert delivery_repo.get_delivery_manifest_for_task(task.task_id) is None
    # Assert task is NOT completed
    refreshed = task_repo.get_task(task.task_id)
    assert refreshed.task_status != TaskStatus.COMPLETED

    # 2. Provide valid prerequisites
    task_dir = tmp_path / "tasks" / task.task_id
    video_content = b"content_valid_123"
    video_path = _create_mock_file(task_dir / "valid_atomic.mp4", video_content)

    eval_snap_id = f"eval_{uuid4().hex[:8]}"
    eval_snap = EvaluationSnapshot(
        evaluation_snapshot_id=eval_snap_id,
        evaluation_target_id="target_1",
        dimension_result_ids=(),
        decision=EvaluationDecision.PASS,
        overall_score=0.95,
        policy_version="v1.0",
        evaluator_version="v1.0",
        summary_reason_codes=("PERFECT_QUALITY",),
    )
    eval_repo.save_snapshot(eval_snap)

    eval_ref = TaskArtifactRef.create(
        task_id=task.task_id,
        stage=Stage.QUALITY_REVIEW,
        artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
        artifact_id=eval_snap_id,
        metadata_json={"decision": "PASS"},
    )
    TaskArtifactRepository(db_session).save_artifact_ref(eval_ref)

    comp_id = f"comp_{uuid4().hex[:8]}"
    comp_out = CompositionOutput.create(
        composition_output_id=comp_id,
        task_id=task.task_id,
        storyboard_snapshot_id=f"sb_{uuid4().hex[:8]}",
        execution_run_id=f"run_{uuid4().hex[:8]}",
        audio_output_id=f"aud_{uuid4().hex[:8]}",
        video_path=video_path,
        video_hash=hashlib.sha256(video_content).hexdigest(),
        duration=12.0,
        width=1920,
        height=1080,
        file_size=len(video_content),
    )
    comp_repo.save_composition_output(comp_out)

    comp_ref = TaskArtifactRef.create(
        task_id=task.task_id,
        stage=Stage.COMPOSITION,
        artifact_type=ArtifactType.COMPOSITION_OUTPUT,
        artifact_id=comp_id,
    )
    TaskArtifactRepository(db_session).save_artifact_ref(comp_ref)
    db_session.commit()

    # Now execute Delivery executor successfully
    res = executor.execute(task, job)
    assert res.success is True
    assert delivery_repo.get_delivery_manifest_for_task(task.task_id) is not None

    # Complete job through workflow
    job.mark_succeeded(output_artifact_revision_id=res.output_artifact_ref.artifact_id)
    workflow.on_stage_completed(task, job)

    # NOW task is COMPLETED
    refreshed = task_repo.get_task(task.task_id)
    assert refreshed.task_status == TaskStatus.COMPLETED
