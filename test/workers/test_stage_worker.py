from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.stage_execution import StageExecution
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.workers.stage_worker import StageWorker


class FakeStageExecutor:
    """Test-only stage executor that returns configurable results."""

    def __init__(
        self,
        success: bool = True,
        artifact_type: ArtifactType = ArtifactType.EVIDENCE_SNAPSHOT,
        error_type: str | None = None,
        error_message: str | None = None,
        is_retryable: bool = False,
        raise_exception: Exception | None = None,
    ) -> None:
        self.success = success
        self.artifact_type = artifact_type
        self.error_type = error_type
        self.error_message = error_message
        self.is_retryable = is_retryable
        self.raise_exception = raise_exception
        self.executed_count = 0

    def execute(self, task: KnowledgeVideoTask, job: WorkflowJob) -> StageExecutionResult:
        self.executed_count += 1
        if self.raise_exception is not None:
            raise self.raise_exception

        output_ref = None
        if self.success:
            output_ref = TaskArtifactRef.create(
                task_id=task.task_id,
                stage=job.stage,
                artifact_type=self.artifact_type,
                artifact_id=f"art_{uuid4().hex[:8]}",
                metadata_json={"executed_by": "FakeStageExecutor"},
            )

        return StageExecutionResult(
            success=self.success,
            output_artifact_ref=output_ref,
            error_type=self.error_type,
            error_message=self.error_message,
            is_retryable=self.is_retryable,
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


def test_unsupported_stage_remains_queued_in_production_registry(session_factory):
    """
    CRITICAL INVARIANT:
    In Stage P1, AUDIO and subsequent stages are not yet implemented.
    Since default production registry does not register them, an unsupported job MUST
    remain QUEUED, safe, unacquired, and unfailed.
    """
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="General Relativity",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        task.advance_stage(Stage.SCRIPT)
        task.advance_stage(Stage.STORYBOARD)
        task.advance_stage(Stage.PRODUCTION_PLAN)
        task.advance_stage(Stage.ASSET)
        task.advance_stage(Stage.AUDIO)
        task.advance_stage(Stage.COMPOSITION)
        task.advance_stage(Stage.QUALITY_REVIEW)
        task.advance_stage(Stage.DELIVERY)
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.DELIVERY,
            idempotency_key=f"idemp_{task_id}_delivery_1",
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    # Create worker with default production registry
    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="prod-worker-1",
    )

    # Worker runs one iteration
    processed = worker.run_once()
    assert processed is False

    # Verify task and job status are pristine
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        exec_repo = StageExecutionRepository(session)

        current_task = task_repo.get_task(task_id)
        current_job = job_repo.get_job(job.job_id)
        executions = exec_repo.list_executions_for_task(task_id)

        assert current_task.task_status == TaskStatus.CREATED
        assert current_task.current_stage == Stage.DELIVERY
        assert current_job.status == JobStatus.QUEUED
        assert current_job.lease_owner is None
        assert current_job.lease_expires_at is None
        assert len(executions) == 0


def test_supported_stage_execution_and_identity_continuity(session_factory):
    """
    Verify that when a test executor is registered:
    1. Worker acquires and runs the job.
    2. StageExecution and TaskArtifactRef are created.
    3. Workflow advances to next stage under AUTO policy.
    4. Exact task_id continuity holds across KnowledgeVideoTask, WorkflowJob, StageExecution, and TaskArtifactRef.
    """
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Quantum Mechanics",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task_id}_evidence_1",
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    fake_executor = FakeStageExecutor(success=True, artifact_type=ArtifactType.EVIDENCE_SNAPSHOT)
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, fake_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="test-worker-1",
    )

    processed = worker.run_once()
    assert processed is True
    assert fake_executor.executed_count == 1

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        exec_repo = StageExecutionRepository(session)
        artifact_repo = TaskArtifactRepository(session)

        current_task = task_repo.get_task(task_id)
        completed_job = job_repo.get_job(job.job_id)
        executions = exec_repo.list_executions_for_task(task_id)
        artifacts = artifact_repo.list_artifact_refs_for_task(task_id)

        # 1. Job and Task progression
        assert completed_job.status == JobStatus.SUCCEEDED
        assert completed_job.output_task_artifact_ref_id is not None

        # Auto policy advanced to KNOWLEDGE_PLAN
        assert current_task.task_status == TaskStatus.RUNNING
        assert current_task.current_stage == Stage.KNOWLEDGE_PLAN

        # Next job created
        next_job = job_repo.get_current_job_for_task(task_id)
        assert next_job is not None
        assert next_job.job_id != completed_job.job_id
        assert next_job.stage == Stage.KNOWLEDGE_PLAN
        assert next_job.status == JobStatus.QUEUED
        assert next_job.input_task_artifact_ref_id == completed_job.output_task_artifact_ref_id

        # 2. Execution audit trail
        assert len(executions) == 1
        exec_record = executions[0]
        assert exec_record.status == "SUCCEEDED"
        assert exec_record.stage == Stage.EVIDENCE
        assert exec_record.output_task_artifact_ref_id == completed_job.output_task_artifact_ref_id

        # 3. Artifact reference
        assert len(artifacts) == 1
        artifact_ref = artifacts[0]
        assert artifact_ref.task_artifact_ref_id == completed_job.output_task_artifact_ref_id
        assert artifact_ref.artifact_type == ArtifactType.EVIDENCE_SNAPSHOT

        # 4. Identity continuity invariant
        assert current_task.task_id == task_id
        assert completed_job.task_id == task_id
        assert next_job.task_id == task_id
        assert exec_record.task_id == task_id
        assert artifact_ref.task_id == task_id


def test_review_policy_pauses_at_review_checkpoint(session_factory):
    """Verify that under REVIEW policy, completion of a checkpoint stage pauses task in WAITING_USER."""
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        # Create task under REVIEW policy and advance to KNOWLEDGE_PLAN checkpoint stage
        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Biochemistry",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.REVIEW,
        )
        task.advance_stage(Stage.KNOWLEDGE_PLAN)
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            idempotency_key=f"idemp_{task_id}_kp_1",
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    fake_executor = FakeStageExecutor(success=True)
    registry = StageExecutorRegistry()
    registry.register(Stage.KNOWLEDGE_PLAN, fake_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="review-worker-1",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        current_task = task_repo.get_task(task_id)
        assert current_task.task_status == TaskStatus.WAITING_USER
        assert current_task.current_stage == Stage.KNOWLEDGE_PLAN
        assert "Review required after KNOWLEDGE_PLAN" in (current_task.waiting_reason or "")

        # Completed job is marked succeeded, and no next job is created
        current_job = job_repo.get_current_job_for_task(task_id)
        assert current_job is not None
        assert current_job.status == JobStatus.SUCCEEDED
        assert current_job.stage == Stage.KNOWLEDGE_PLAN


def test_lease_recovery_and_heartbeat(session_factory):
    """Verify expired leases are reclaimed to QUEUED while valid leases are preserved."""
    task_id = f"task_{uuid4().hex[:8]}"
    now = datetime.now(UTC)

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Black Holes",
            target_duration=60.0,
        )
        task_repo.save_task(task)

        # Job 1: Active lease in future
        job_active = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.KNOWLEDGE_PLAN,
            idempotency_key=f"idemp_{task_id}_active",
            now=now - timedelta(seconds=60),
        )
        job_repo.create_job(job_active)
        acq_active = job_repo.acquire_next_available_job(
            owner="worker-active",
            supported_stages={Stage.KNOWLEDGE_PLAN},
            lease_duration_seconds=300,
            now=now,
        )
        assert acq_active is not None

        # Job 2: Expired lease in past
        job_expired = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.SCRIPT,
            idempotency_key=f"idemp_{task_id}_expired",
            now=now - timedelta(seconds=120),
        )
        job_repo.create_job(job_expired)
        acq_expired = job_repo.acquire_next_available_job(
            owner="worker-crashed",
            supported_stages={Stage.SCRIPT},
            lease_duration_seconds=10,
            now=now - timedelta(seconds=60),
        )
        assert acq_expired is not None

        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=StageExecutorRegistry(),
        worker_id="recovery-worker",
    )

    # Trigger recovery at current time `now`
    recovered_count = worker.recover_leases_if_needed(now=now, force=True)
    assert recovered_count == 1

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        active_reloaded = job_repo.get_job(job_active.job_id)
        expired_reloaded = job_repo.get_job(job_expired.job_id)

        # Active job remains leased
        assert active_reloaded.status == JobStatus.LEASED
        assert active_reloaded.lease_owner == "worker-active"

        # Expired job recovered to QUEUED
        assert expired_reloaded.status == JobStatus.QUEUED
        assert expired_reloaded.lease_owner is None
        assert expired_reloaded.lease_expires_at is None


def test_retryable_error_handling(session_factory):
    """Verify retryable error consumes attempt budget and updates task to NEEDS_RECOVERY."""
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(task_id=task_id, topic="Genetics")
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task_id}_evidence_1",
            attempt_number=1,
            max_attempts=3,
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    failing_executor = FakeStageExecutor(
        success=False,
        error_type="API_RATE_LIMIT",
        error_message="Upstream API rate limited",
        is_retryable=True,
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, failing_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="failing-worker",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)

        current_job = job_repo.get_job(job.job_id)
        current_task = task_repo.get_task(task_id)

        assert current_job.status == JobStatus.RETRYABLE
        assert current_job.error_type == "API_RATE_LIMIT"
        assert current_task.task_status == TaskStatus.NEEDS_RECOVERY


def test_fatal_error_and_unhandled_exception(session_factory):
    """Verify unhandled exceptions in executor are caught, audit logged, and task marked FAILED."""
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(task_id=task_id, topic="Thermodynamics")
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=f"idemp_{task_id}_evidence_1",
        )
        task_repo.save_task(task)
        job_repo.create_job(job)
        session.commit()

    crashing_executor = FakeStageExecutor(
        raise_exception=RuntimeError("GPU connection severed unexpectedly")
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, crashing_executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="crash-worker",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        exec_repo = StageExecutionRepository(session)

        current_job = job_repo.get_job(job.job_id)
        current_task = task_repo.get_task(task_id)
        executions = exec_repo.list_executions_for_task(task_id)

        assert current_job.status == JobStatus.FAILED
        assert current_job.error_type == "FATAL"
        assert "GPU connection severed unexpectedly" in (current_job.error_message or "")
        assert current_task.task_status == TaskStatus.FAILED

        assert len(executions) == 1
        assert executions[0].status == "FAILED"
        assert executions[0].error_type == "FATAL"
