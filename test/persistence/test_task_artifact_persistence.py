from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.stage_execution import StageExecution
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import ArtifactType, JobStatus, Stage, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)


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


def test_task_artifact_repository_crud_and_queries(session_factory):
    """Verify TaskArtifactRepository save, retrieval, task listing, and latest queries."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Quantum Computing",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        artifact_repo = TaskArtifactRepository(session)

        # Create two artifact refs across stages
        ref1 = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=f"ev_{uuid4().hex[:8]}",
            metadata_json={"source_count": 3},
        )
        ref2 = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=f"cpr_{uuid4().hex[:8]}",
            artifact_version="rev_1",
            metadata_json={"beat_count": 4},
        )
        ref3 = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=f"cpr_{uuid4().hex[:8]}",
            artifact_version="rev_2",
            metadata_json={"beat_count": 5},
        )

        artifact_repo.save_artifact_ref(ref1)
        artifact_repo.save_artifact_ref(ref2)
        artifact_repo.save_artifact_ref(ref3)
        session.commit()

    with session_factory() as session:
        artifact_repo = TaskArtifactRepository(session)

        # 1. Get by ID
        loaded_ref1 = artifact_repo.get_artifact_ref(ref1.task_artifact_ref_id)
        assert loaded_ref1 is not None
        assert loaded_ref1.artifact_id == ref1.artifact_id
        assert loaded_ref1.stage == Stage.EVIDENCE
        assert loaded_ref1.artifact_type == ArtifactType.EVIDENCE_SNAPSHOT
        assert loaded_ref1.metadata_json == {"source_count": 3}

        # 2. List all for task
        all_refs = artifact_repo.list_artifact_refs_for_task(task_id)
        assert len(all_refs) == 3
        assert [r.task_artifact_ref_id for r in all_refs] == [
            ref1.task_artifact_ref_id,
            ref2.task_artifact_ref_id,
            ref3.task_artifact_ref_id,
        ]

        # 3. Get latest by stage
        latest_plan = artifact_repo.get_latest_artifact_ref(task_id, stage=Stage.KNOWLEDGE_PLAN)
        assert latest_plan is not None
        assert latest_plan.task_artifact_ref_id == ref3.task_artifact_ref_id
        assert latest_plan.artifact_version == "rev_2"

        # 4. Non-existent returns None
        assert artifact_repo.get_artifact_ref("non_existent_id") is None
        assert (
            artifact_repo.get_latest_artifact_ref(task_id, stage=Stage.DELIVERY)
            is None
        )


def test_acquire_next_available_job_supported_stages_filter(session_factory):
    """Verify acquire_next_available_job filters strictly by supported_stages."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Astrophysics",
            target_duration=120.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Job 1: EVIDENCE stage
        job_evidence = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task_id}_evidence",
        )
        job_repo.create_job(job_evidence)
        session.commit()

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)

        # Case 1: supported_stages is empty set -> returns None immediately
        job = job_repo.acquire_next_available_job(
            owner="worker-1",
            supported_stages=set(),
        )
        assert job is None

        # Case 2: supported_stages does NOT include EVIDENCE -> returns None
        job = job_repo.acquire_next_available_job(
            owner="worker-1",
            supported_stages={Stage.KNOWLEDGE_PLAN, Stage.SCRIPT},
        )
        assert job is None

        # Case 3: supported_stages includes EVIDENCE -> acquires the job
        job = job_repo.acquire_next_available_job(
            owner="worker-1",
            supported_stages={Stage.EVIDENCE},
        )
        assert job is not None
        assert job.stage == Stage.EVIDENCE
        assert job.status == JobStatus.LEASED
        assert job.lease_owner == "worker-1"


def test_workflow_job_and_stage_execution_artifact_ref_persistence(session_factory):
    """Verify input_task_artifact_ref_id and output_task_artifact_ref_id are stored and reloaded."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Cell Biology",
            target_duration=90.0,
        )
        task_repo.save_task(task)

        artifact_repo = TaskArtifactRepository(session)
        in_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=f"ev_{uuid4().hex[:8]}",
        )
        out_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            artifact_id=f"cpr_{uuid4().hex[:8]}",
        )
        artifact_repo.save_artifact_ref(in_ref)
        artifact_repo.save_artifact_ref(out_ref)

        job_repo = WorkflowJobRepository(session)
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            idempotency_key=f"idemp_{task_id}_kp",
            input_task_artifact_ref_id=in_ref.task_artifact_ref_id,
        )
        job_repo.create_job(job)

        # Mark job completed with output_task_artifact_ref_id
        job.mark_succeeded(output_task_artifact_ref_id=out_ref.task_artifact_ref_id)
        job_repo.update_job(job)

        # Record StageExecution
        stage_exec_repo = StageExecutionRepository(session)
        now = datetime.now(UTC)
        stage_exec = StageExecution.record(
            task_id=task_id,
            job_id=job.job_id,
            stage=Stage.KNOWLEDGE_PLAN,
            attempt_number=1,
            status="SUCCEEDED",
            started_at=now,
            finished_at=now + timedelta(seconds=2),
            input_task_artifact_ref_id=in_ref.task_artifact_ref_id,
            output_task_artifact_ref_id=out_ref.task_artifact_ref_id,
        )
        stage_exec_repo.record_execution(stage_exec)
        session.commit()

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        loaded_job = job_repo.get_job(job.job_id)
        assert loaded_job is not None
        assert loaded_job.input_task_artifact_ref_id == in_ref.task_artifact_ref_id
        assert loaded_job.output_task_artifact_ref_id == out_ref.task_artifact_ref_id

        stage_exec_repo = StageExecutionRepository(session)
        loaded_exec = stage_exec_repo.get_latest_execution_for_stage(task_id, Stage.KNOWLEDGE_PLAN)
        assert loaded_exec is not None
        assert loaded_exec.input_task_artifact_ref_id == in_ref.task_artifact_ref_id
        assert loaded_exec.output_task_artifact_ref_id == out_ref.task_artifact_ref_id


def test_alembic_migration_0015_upgrade_downgrade(tmp_path, monkeypatch):
    """Verify Alembic migration 0015 upgrades from 0014, downgrades, and re-upgrades cleanly."""
    from pathlib import Path
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    project_root = Path(__file__).resolve().parent.parent.parent

    alembic_ini_path = project_root / "alembic.ini"
    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(project_root / "migrations"))

    # Upgrade to 0014 first
    command.upgrade(cfg, "0014_knowledge_video_workflow")

    # Upgrade to 0015
    command.upgrade(cfg, "0015_stage_worker_and_task_artifacts")

    # Downgrade back to 0014
    command.downgrade(cfg, "0014_knowledge_video_workflow")

    # Re-upgrade to 0015
    command.upgrade(cfg, "0015_stage_worker_and_task_artifacts")

