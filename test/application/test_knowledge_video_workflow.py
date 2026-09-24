from __future__ import annotations

from datetime import UTC, datetime, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.application.task_command_service import TaskCommandService
from app.application.task_query_service import TaskQueryService
from app.domain.workflow_state import (
    JobStatus,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowConflictError,
    WorkflowPolicyType,
)
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


def test_create_task_transactionally_creates_task_and_evidence_job(session_factory):
    """Verifies that create_task generates a task and its initial EVIDENCE job in a single transaction."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(
            topic="Quantum Computing",
            target_duration=120.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        session.commit()

        task_id = task.task_id

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        persisted_task = task_repo.get_task(task_id)
        assert persisted_task is not None
        assert persisted_task.topic == "Quantum Computing"
        assert persisted_task.task_status == TaskStatus.CREATED
        assert persisted_task.current_stage == Stage.EVIDENCE
        assert persisted_task.target_duration == 120.0

        current_job = job_repo.get_current_job_for_task(task_id)
        assert current_job is not None
        assert current_job.task_id == task_id
        assert current_job.stage == Stage.EVIDENCE
        assert current_job.status == JobStatus.QUEUED
        assert current_job.attempt_number == 1
        assert current_job.idempotency_key == f"idemp_{task_id}_evidence_1"


def test_auto_policy_advances_stages_without_pausing(session_factory):
    """Under AUTO policy, non-terminal stages automatically advance to the next stage."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(
            topic="Black Holes",
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        session.commit()
        task_id = task.task_id

    # Simulate worker completing EVIDENCE stage
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = task_repo.get_task(task_id)
        job = job_repo.get_current_job_for_task(task_id)

        # Mark job completed
        job.mark_succeeded(output_artifact_revision_id="art_evidence_v1")
        job_repo.update_job(job)

        next_job = workflow.on_stage_completed(task=task, completed_job=job)
        session.commit()

        assert next_job is not None
        assert next_job.stage == Stage.KNOWLEDGE_PLAN
        assert next_job.status == JobStatus.QUEUED

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        updated_task = task_repo.get_task(task_id)
        assert updated_task.current_stage == Stage.KNOWLEDGE_PLAN
        assert updated_task.task_status == TaskStatus.RUNNING


def test_review_policy_pauses_at_review_checkpoints(session_factory):
    """Under REVIEW policy, completing KNOWLEDGE_PLAN transitions the task to WAITING_USER."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(
            topic="Artificial Intelligence",
            workflow_policy=WorkflowPolicyType.REVIEW,
        )
        session.commit()
        task_id = task.task_id

    # 1. EVIDENCE finishes -> advances to KNOWLEDGE_PLAN (not a pause checkpoint)
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = task_repo.get_task(task_id)
        job = job_repo.get_current_job_for_task(task_id)
        job.mark_succeeded()
        job_repo.update_job(job)

        next_job = workflow.on_stage_completed(task=task, completed_job=job)
        session.commit()
        assert next_job is not None
        assert next_job.stage == Stage.KNOWLEDGE_PLAN

    # 2. KNOWLEDGE_PLAN finishes -> IS a pause checkpoint for REVIEW policy
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = task_repo.get_task(task_id)
        job = job_repo.get_current_job_for_task(task_id)
        job.mark_succeeded(output_artifact_revision_id="art_plan_v1")
        job_repo.update_job(job)

        next_job = workflow.on_stage_completed(task=task, completed_job=job)
        session.commit()

        # No immediate job created; task enters WAITING_USER
        assert next_job is None

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        paused_task = task_repo.get_task(task_id)
        assert paused_task.task_status == TaskStatus.WAITING_USER
        assert paused_task.current_stage == Stage.KNOWLEDGE_PLAN
        assert "KNOWLEDGE_PLAN" in (paused_task.waiting_reason or "")


def test_approve_task_resumes_workflow_and_creates_next_job(session_factory):
    """Approving a task paused in WAITING_USER advances it to the next stage and enqueues a job."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(
            topic="Special Relativity",
            workflow_policy=WorkflowPolicyType.REVIEW,
        )
        session.commit()
        task_id = task.task_id

    # Advance to WAITING_USER at KNOWLEDGE_PLAN
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        # Complete EVIDENCE
        task = task_repo.get_task(task_id)
        job1 = job_repo.get_current_job_for_task(task_id)
        job1.mark_succeeded()
        job_repo.update_job(job1)
        workflow.on_stage_completed(task, job1)

        # Complete KNOWLEDGE_PLAN -> pauses
        task = task_repo.get_task(task_id)
        job2 = job_repo.get_current_job_for_task(task_id)
        job2.mark_succeeded()
        job_repo.update_job(job2)
        workflow.on_stage_completed(task, job2)
        session.commit()

    # User calls approve
    with session_factory() as session:
        command_service = TaskCommandService(session)
        approved_task = command_service.approve_task(task_id)
        session.commit()

        assert approved_task.task_status == TaskStatus.RUNNING
        assert approved_task.current_stage == Stage.SCRIPT

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        new_job = job_repo.get_current_job_for_task(task_id)
        assert new_job is not None
        assert new_job.stage == Stage.SCRIPT
        assert new_job.status == JobStatus.QUEUED


def test_approve_task_raises_conflict_if_not_waiting_user(session_factory):
    """Calling approve on a task not in WAITING_USER raises WorkflowConflictError."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(topic="Thermodynamics")
        session.commit()
        task_id = task.task_id

    with session_factory() as session:
        command_service = TaskCommandService(session)
        with pytest.raises(WorkflowConflictError):
            command_service.approve_task(task_id)


def test_cancel_task_aborts_task_and_active_job(session_factory):
    """Cancelling a task marks both the task and any active job as CANCELLED."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(topic="Nanotechnology")
        session.commit()
        task_id = task.task_id

    with session_factory() as session:
        command_service = TaskCommandService(session)
        cancelled_task = command_service.cancel_task(task_id, reason="User cancelled from UI")
        session.commit()

        assert cancelled_task.task_status == TaskStatus.CANCELLED
        assert cancelled_task.finished_at is not None

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        current_job = job_repo.get_current_job_for_task(task_id)
        assert current_job is not None
        assert current_job.status == JobStatus.CANCELLED

    # Subsequent approve or retry on cancelled task must fail
    with session_factory() as session:
        command_service = TaskCommandService(session)
        with pytest.raises(TerminalStateImmutableError):
            command_service.cancel_task(task_id)


def test_retry_task_requeues_job_for_failed_stage(session_factory):
    """Retrying a FAILED task transitions task to RUNNING and creates a new attempt job."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(topic="Fluid Dynamics")
        session.commit()
        task_id = task.task_id

    # Simulate job failure
    with session_factory() as session:
        workflow = KnowledgeVideoWorkflow(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = task_repo.get_task(task_id)
        job = job_repo.get_current_job_for_task(task_id)
        job.mark_failed(error_type="RETRYABLE", error_message="API connection reset")
        job_repo.update_job(job)

        workflow.on_stage_failed(task, job, error_type="RETRYABLE", error_message="API connection reset")
        session.commit()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        failed_task = task_repo.get_task(task_id)
        assert failed_task.task_status in (TaskStatus.FAILED, TaskStatus.NEEDS_RECOVERY)

    # User retries
    with session_factory() as session:
        command_service = TaskCommandService(session)
        retried_task = command_service.retry_task(task_id)
        session.commit()

        assert retried_task.task_status == TaskStatus.RUNNING

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        new_job = job_repo.get_current_job_for_task(task_id)
        assert new_job is not None
        assert new_job.status == JobStatus.QUEUED
        assert new_job.attempt_number == 2


def test_query_service_returns_detail_and_summary(session_factory):
    """Verifies TaskQueryService provides comprehensive task inspection."""
    with session_factory() as session:
        command_service = TaskCommandService(session)
        task = command_service.create_task(
            topic="Cosmology",
            target_duration=90.0,
        )
        session.commit()
        task_id = task.task_id

    with session_factory() as session:
        query_service = TaskQueryService(session)
        detail = query_service.get_task_detail(task_id)

        assert detail is not None
        assert detail["task_id"] == task_id
        assert detail["topic"] == "Cosmology"
        assert detail["task_status"] == TaskStatus.CREATED.value
        assert detail["current_stage"] == Stage.EVIDENCE.value
        assert detail["current_job"] is not None
        assert detail["current_job"]["stage"] == Stage.EVIDENCE.value
        assert isinstance(detail["stage_executions"], list)

        tasks = query_service.list_tasks(limit=10, offset=0)
        assert len(tasks) >= 1
        assert any(t["task_id"] == task_id for t in tasks)
