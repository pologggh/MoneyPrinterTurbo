from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.stage_execution import StageExecution
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobStatus, Stage, TaskStatus, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
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


def test_task_and_initial_job_transactional_consistency(session_factory):
    """Verify task and initial job are persisted transactionally."""
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            topic="General Relativity",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.REVIEW,
        )
        job = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task.task_id}_evidence_1",
        )

        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        loaded_task = task_repo.get_task(task.task_id)
        loaded_job = job_repo.get_job(job.job_id)

        assert loaded_task is not None
        assert loaded_task.topic == "General Relativity"
        assert loaded_task.task_status == TaskStatus.CREATED
        assert loaded_task.current_stage == Stage.EVIDENCE

        assert loaded_job is not None
        assert loaded_job.task_id == task.task_id
        assert loaded_job.stage == Stage.EVIDENCE
        assert loaded_job.status == JobStatus.QUEUED


def test_idempotency_key_uniqueness_enforced(session_factory):
    """Verify unique constraint on WorkflowJob idempotency_key."""
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(topic="Quantum Mechanics")
        task_repo.save_task(task)

        job1 = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key="unique_key_100",
        )
        job_repo.create_job(job1)
        session.commit()

        job2 = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key="unique_key_100",  # Duplicate key
        )
        with pytest.raises(IntegrityError):
            job_repo.create_job(job2)
            session.commit()


def test_job_lease_acquisition_and_concurrency(session_factory):
    """Verify job lease acquisition assigns owner and blocks concurrent workers."""
    now = datetime.now(UTC)
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(topic="Thermodynamics")
        task_repo.save_task(task)

        job = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task.task_id}_1",
            now=now,
        )
        job_repo.create_job(job)
        session.commit()

    # Worker 1 acquires lease
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        acquired = job_repo.acquire_next_available_job(owner="worker_1", lease_duration_seconds=300, now=now)
        assert acquired is not None
        assert acquired.job_id == job.job_id
        assert acquired.status == JobStatus.LEASED
        assert acquired.lease_owner == "worker_1"
        session.commit()

    # Worker 2 attempts to acquire at t+10s (should find no available job)
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        acquired_2 = job_repo.acquire_next_available_job(owner="worker_2", lease_duration_seconds=300, now=now + timedelta(seconds=10))
        assert acquired_2 is None


def test_recover_expired_leases(session_factory):
    """Verify expired leases are reclaimed and can be picked up by new workers."""
    now = datetime.now(UTC)
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(topic="Astrophysics")
        task_repo.save_task(task)

        job = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task.task_id}_1",
            now=now,
        )
        job_repo.create_job(job)
        session.commit()

        # Leased with 60s expiration
        acquired = job_repo.acquire_next_available_job(owner="worker_dead", lease_duration_seconds=60, now=now)
        assert acquired is not None
        session.commit()

    # At t=120s, reclaim expired leases
    t_later = now + timedelta(seconds=120)
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        reclaimed_count = job_repo.recover_expired_leases(now=t_later)
        assert reclaimed_count == 1
        session.commit()

    # New worker can now acquire the job
    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        new_acquired = job_repo.acquire_next_available_job(owner="worker_alive", lease_duration_seconds=300, now=t_later)
        assert new_acquired is not None
        assert new_acquired.job_id == job.job_id
        assert new_acquired.lease_owner == "worker_alive"


def test_stage_executions_history_preserved_across_attempts(session_factory):
    """Verify multiple attempts for a stage append records without overwriting past attempts."""
    now = datetime.now(UTC)
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        exec_repo = StageExecutionRepository(session)

        task = KnowledgeVideoTask.create(topic="Optics")
        task_repo.save_task(task)

        job = WorkflowJob.create(
            task_id=task.task_id,
            stage=Stage.ASSET,
            idempotency_key=f"idemp_{task.task_id}_asset_1",
            now=now,
        )
        job_repo.create_job(job)

        # Attempt 1: Failed
        exec1 = StageExecution.record(
            task_id=task.task_id,
            job_id=job.job_id,
            stage=Stage.ASSET,
            attempt_number=1,
            status="FAILED",
            error_type="RETRYABLE",
            error_message="Network timeout",
            started_at=now,
            finished_at=now + timedelta(seconds=5),
        )
        exec_repo.record_execution(exec1)

        # Attempt 2: Succeeded
        exec2 = StageExecution.record(
            task_id=task.task_id,
            job_id=job.job_id,
            stage=Stage.ASSET,
            attempt_number=2,
            status="SUCCEEDED",
            output_artifact_revision_id="art_asset_v2",
            started_at=now + timedelta(seconds=10),
            finished_at=now + timedelta(seconds=15),
        )
        exec_repo.record_execution(exec2)
        session.commit()

    with session_factory() as session:
        exec_repo = StageExecutionRepository(session)
        history = exec_repo.list_executions_for_task(task.task_id)
        assert len(history) == 2
        assert history[0].attempt_number == 1
        assert history[0].status == "FAILED"
        assert history[1].attempt_number == 2
        assert history[1].status == "SUCCEEDED"
        assert history[1].output_artifact_revision_id == "art_asset_v2"
