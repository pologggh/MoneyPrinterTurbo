from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.stage_executor_registry import get_default_executor_registry
from app.domain.audio import AudioOutput
from app.domain.composition import CompositionOutput
from app.domain.enums import BeatType
from app.domain.evaluation import EvaluationDecision, EvaluationSnapshot
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.workflow_job import JobStatus, WorkflowJob
from app.domain.workflow_state import Stage, TaskStatus, WorkflowPolicyType
from app.persistence.models import (
    Base,
    ContentPlanRevisionORM,
    ExecutionRunORM,
    StoryboardSnapshotORM,
)
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    DeliveryManifestRepository,
    EvaluationRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
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
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def test_worker_executes_delivery_from_seeded_upstream_artifacts(session_factory, tmp_path):
    """
    P0-1 & P0-2 delivery-stage integration proof.

    This test deliberately seeds upstream artifacts; it does not exercise the
    complete ten-stage workflow or any external provider.

    - 1 KnowledgeVideoTask with AUTO policy
    - 1 persistent WorkflowJob for DELIVERY
    - StageWorker instance using get_default_executor_registry(session_factory)
    - Non-empty ScriptRevision and ScriptSegments with narration_text and evidence_refs
    - Pre-existing valid CompositionOutput and passing EvaluationSnapshot
    - worker.run_once() executes DeliveryStageExecutor, creates delivery manifest,
      persists task completion, and updates the job.
    - Assert source_report.md contains real narration excerpt and evidence references
    - Assert delivery manifest is persisted and retrievable
    - Assert task.status == TaskStatus.COMPLETED
    - NO test shortcut: the test does NOT manually invoke job.mark_succeeded().
    """
    task_id = f"task_p0_real_{uuid4().hex[:8]}"
    plan_id = f"cpr_{uuid4().hex[:8]}"
    script_id = f"scr_{uuid4().hex[:8]}"
    storyboard_id = f"sb_{uuid4().hex[:8]}"
    run_id = f"run_{uuid4().hex[:8]}"
    audio_id = f"ao_{uuid4().hex[:8]}"
    comp_id = f"co_{uuid4().hex[:8]}"
    snap_id = f"es_{uuid4().hex[:8]}"
    job_id = f"job_del_{uuid4().hex[:8]}"

    # Setup directories and files
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    video_content = b"SEEDED_VIDEO_MP4_BYTES_STREAM_12345"
    video_path = str(task_dir / f"final_{comp_id}.mp4")
    with open(video_path, "wb") as f:
        f.write(video_content)
    video_hash = hashlib.sha256(video_content).hexdigest()

    subtitle_content = "1\n00:00:00,000 --> 00:00:05,000\nQuantum computing principles.\n"
    subtitle_path = str(task_dir / f"subtitles_{audio_id}.srt")
    with open(subtitle_path, "wb") as f:
        f.write(subtitle_content.encode("utf-8"))
    subtitle_hash = hashlib.sha256(Path(subtitle_path).read_bytes()).hexdigest()

    audio_content = b"SEEDED_AUDIO_MP3_STREAM"
    audio_path = str(task_dir / f"audio_{audio_id}.mp3")
    with open(audio_path, "wb") as f:
        f.write(audio_content)
    audio_hash = hashlib.sha256(audio_content).hexdigest()

    # Seed upstream database records
    with session_factory() as session:
        # 1. Task (AUTO policy, RUNNING status)
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="量子计算前沿导论",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # 2. ContentPlan
        session.add(
            ContentPlanRevisionORM(
                content_plan_revision_id=plan_id,
                revision_number=1,
                topic="量子计算前沿导论",
                overall_target_duration=60.0,
                created_at=datetime.now(UTC),
            )
        )

        # 3. Real non-empty ScriptRevision and ScriptSegments (verifying P0-1 fix!)
        script_repo = ScriptRepository(session)
        seg1 = ScriptSegment(
            script_revision_id=script_id,
            content_beat_id="beat_01",
            order=1,
            narration_text="Quantum computing utilizes superposition and entanglement principles.",
            target_duration=30.0,
            beat_type=BeatType.KNOWLEDGE,
            evidence_refs=("ev_nature_2023_quantum", "ev_arxiv_2301_001"),
        )
        seg2 = ScriptSegment(
            script_revision_id=script_id,
            content_beat_id="beat_02",
            order=2,
            narration_text="Superconducting qubits and photonic quantum circuits are making rapid progress.",
            target_duration=30.0,
            beat_type=BeatType.SUMMARY,
            evidence_refs=("ev_science_2024_qubits",),
        )
        script_rev = ScriptRevision.create(
            task_id=task_id,
            content_plan_revision_id=plan_id,
            segments=[seg1, seg2],
            overall_target_duration=60.0,
            script_revision_id=script_id,
        )
        script_repo.save_revision(script_rev)

        # 4. Storyboard & ExecutionRun
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

        # 5. AudioOutput
        audio_repo = AudioOutputRepository(session)
        audio_out = AudioOutput.create(
            audio_output_id=audio_id,
            task_id=task_id,
            script_revision_id=script_id,
            execution_run_id=run_id,
            narration_audio_path=audio_path,
            narration_audio_hash=audio_hash,
            actual_narration_duration=60.0,
            subtitle_path=subtitle_path,
            subtitle_hash=subtitle_hash,
        )
        audio_repo.save_audio_output(audio_out)

        # 6. CompositionOutput & Artifact Ref
        comp_repo = CompositionOutputRepository(session)
        comp_out = CompositionOutput.create(
            composition_output_id=comp_id,
            task_id=task_id,
            storyboard_snapshot_id=storyboard_id,
            execution_run_id=run_id,
            audio_output_id=audio_id,
            video_path=video_path,
            video_hash=video_hash,
            duration=60.0,
            width=1920,
            height=1080,
            file_size=len(video_content),
        )
        comp_repo.save_composition_output(comp_out)

        art_repo = TaskArtifactRepository(session)
        comp_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.COMPOSITION,
            artifact_type=ArtifactType.COMPOSITION_OUTPUT,
            artifact_id=comp_id,
        )
        art_repo.save_artifact_ref(comp_ref)

        # 7. Passing EvaluationSnapshot & Artifact Ref
        eval_repo = EvaluationRepository(session)
        eval_snap = EvaluationSnapshot(
            evaluation_snapshot_id=snap_id,
            evaluation_target_id=f"target_{comp_id}",
            dimension_result_ids=(),
            decision=EvaluationDecision.PASS,
            overall_score=0.98,
            policy_version="v1.0",
            evaluator_version="v1.0",
            summary_reason_codes=("PERFECT_EVALUATION",),
        )
        eval_repo.save_snapshot(eval_snap)

        eval_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.QUALITY_REVIEW,
            artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
            artifact_id=snap_id,
        )
        art_repo.save_artifact_ref(eval_ref)

        # 8. Real persistent WorkflowJob for DELIVERY
        job_repo = WorkflowJobRepository(session)
        delivery_job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.DELIVERY,
            idempotency_key=f"idemp_{job_id}",
            job_id=job_id,
        )
        job_repo.create_job(delivery_job)
        session.commit()

    # Instantiate StageWorker using get_default_executor_registry(session_factory)
    registry = get_default_executor_registry(session_factory=session_factory)
    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="p0-delivery-real-worker-01",
        poll_interval_seconds=0.1,
    )

    # EXECUTE STAGEWORKER
    handled = worker.run_once()
    assert handled is True, "Worker should have acquired and handled the queued DELIVERY job"

    # VERIFY RESULTS
    with session_factory() as session:
        # 1. Job completed successfully
        job_repo = WorkflowJobRepository(session)
        db_job = job_repo.get_job(job_id)
        assert db_job is not None
        assert db_job.status == JobStatus.SUCCEEDED
        assert db_job.finished_at is not None

        # 2. KnowledgeVideoTask reached COMPLETED
        task_repo = KnowledgeVideoTaskRepository(session)
        db_task = task_repo.get_task(task_id)
        assert db_task is not None
        assert db_task.task_status == TaskStatus.COMPLETED

        # 3. DeliveryManifest persisted and retrievable
        manifest_repo = DeliveryManifestRepository(session)
        manifest = manifest_repo.get_delivery_manifest_for_task(task_id)
        assert manifest is not None
        assert manifest.task_id == task_id
        assert manifest.final_video_path == video_path
        assert manifest.final_video_hash == video_hash
        assert manifest.subtitle_path == subtitle_path
        assert manifest.subtitle_hash == subtitle_hash

        # 4. Reports generated and contain real narration excerpts and evidence refs
        source_report_path = Path(manifest.source_report_path)
        execution_report_path = Path(manifest.execution_report_path)
        assert source_report_path.exists(), f"Source report {source_report_path} must exist"
        assert execution_report_path.exists(), f"Execution report {execution_report_path} must exist"

        source_report_text = source_report_path.read_text(encoding="utf-8")
        assert "Quantum computing utilizes superposition and entanglement" in source_report_text, "Source report must contain narration excerpt"
        assert "ev_nature_2023_quantum" in source_report_text, "Source report must cite evidence_refs"
        assert "ev_arxiv_2301_001" in source_report_text, "Source report must cite evidence_refs"
        assert "ev_science_2024_qubits" in source_report_text, "Source report must cite evidence_refs"

        # 5. Hashes match
        calculated_source_hash = hashlib.sha256(source_report_path.read_bytes()).hexdigest()
        assert manifest.source_report_hash == calculated_source_hash
