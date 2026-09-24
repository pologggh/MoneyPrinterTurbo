from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.application.workflow_policy import WorkflowPolicy
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobErrorType,
    Stage,
    TaskStatus,
    WorkflowConflictError,
    get_next_stage,
)
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)


class KnowledgeVideoWorkflow:
    """
    Authoritative workflow orchestration engine for KnowledgeVideoTask.
    Controls stage transitions, review pause points, job generation, and error handling.
    Single engine shared by both AUTO and REVIEW execution policies.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.job_repo = WorkflowJobRepository(session)
        self.exec_repo = StageExecutionRepository(session)
        self.artifact_repo = TaskArtifactRepository(session)

    def on_stage_completed(
        self,
        task: KnowledgeVideoTask,
        completed_job: WorkflowJob,
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        """
        Invoked when a stage worker successfully finishes a job.
        Advances stage and creates next job, or pauses at review checkpoint, or marks task COMPLETED.
        """
        ts = now or datetime.now(UTC)

        # 1. If completed stage was the terminal stage (DELIVERY)
        if completed_job.stage == Stage.DELIVERY:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        # 2. Check workflow policy for review checkpoints
        policy = WorkflowPolicy(policy_type=task.workflow_policy)
        if policy.should_pause_for_review(completed_job.stage):
            task.transition_to(
                TaskStatus.WAITING_USER,
                reason=f"Review required after {completed_job.stage.value}",
                now=ts,
            )
            self.task_repo.save_task(task)
            return None

        # 3. Advance to the next stage
        next_stage = get_next_stage(completed_job.stage)
        if next_stage is None:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        task.advance_stage(next_stage)
        if task.task_status != TaskStatus.RUNNING:
            task.transition_to(TaskStatus.RUNNING, now=ts)
        self.task_repo.save_task(task)

        # 4. Generate next persistent WorkflowJob
        seq = 1
        idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"
        while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
            seq += 1
            idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"

        next_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=next_stage,
            idempotency_key=idempotency_key,
            attempt_number=1,
            input_artifact_revision_id=completed_job.output_artifact_revision_id,
            input_task_artifact_ref_id=completed_job.output_task_artifact_ref_id,
            now=ts,
        )
        return self.job_repo.create_job(next_job)

    def on_stage_failed(
        self,
        task: KnowledgeVideoTask,
        failed_job: WorkflowJob,
        error_type: str = "RETRYABLE",
        error_message: str = "",
        now: datetime | None = None,
    ) -> None:
        """
        Invoked when a stage job fails. Sets appropriate failure / pause state.
        """
        ts = now or datetime.now(UTC)
        err_enum = None
        try:
            err_enum = JobErrorType(error_type)
        except ValueError:
            err_enum = JobErrorType.RETRYABLE

        if task.task_status == TaskStatus.CREATED:
            task.transition_to(TaskStatus.RUNNING, now=ts)

        if err_enum == JobErrorType.NEEDS_EVIDENCE:
            task.transition_to(
                TaskStatus.NEEDS_EVIDENCE,
                reason="Core claims lack verified evidence",
                error_type=err_enum,
                error_message=error_message,
                now=ts,
            )
        elif err_enum in (JobErrorType.NEEDS_RECOVERY, JobErrorType.RETRYABLE) or failed_job.attempt_number >= failed_job.max_attempts:
            task.transition_to(
                TaskStatus.NEEDS_RECOVERY,
                reason=error_message or "Stage execution failed and requires recovery",
                error_type=JobErrorType.NEEDS_RECOVERY,
                error_message=error_message,
                now=ts,
            )
        else:
            task.transition_to(
                TaskStatus.FAILED,
                reason=f"Stage execution failed: {error_type}",
                error_type=err_enum,
                error_message=error_message,
                now=ts,
            )
        self.task_repo.save_task(task)

    def resume_after_approval(
        self,
        task: KnowledgeVideoTask,
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        """
        Resumes execution of a task currently paused in WAITING_USER.
        Advances stage and creates the next stage job.
        """
        ts = now or datetime.now(UTC)
        if task.task_status != TaskStatus.WAITING_USER:
            raise WorkflowConflictError(
                f"Task '{task.task_id}' cannot be approved: status is '{task.task_status.value}', expected WAITING_USER."
            )

        prior_stage = task.current_stage
        next_stage = get_next_stage(task.current_stage)
        if next_stage is None:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        task.advance_stage(next_stage)
        task.transition_to(TaskStatus.RUNNING, now=ts)
        self.task_repo.save_task(task)

        latest_ref = self.artifact_repo.get_latest_artifact_ref(task.task_id, stage=prior_stage)
        input_ref_id = latest_ref.task_artifact_ref_id if latest_ref else None

        seq = 1
        idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"
        while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
            seq += 1
            idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"

        next_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=next_stage,
            idempotency_key=idempotency_key,
            attempt_number=1,
            input_task_artifact_ref_id=input_ref_id,
            now=ts,
        )
        return self.job_repo.create_job(next_job)
