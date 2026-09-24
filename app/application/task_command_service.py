from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.application.knowledge_video_workflow import KnowledgeVideoWorkflow
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobStatus,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowConflictError,
    WorkflowPolicyType,
)
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    WorkflowJobRepository,
)


class TaskCommandService:
    """
    Application command service handling state-changing lifecycle operations for KnowledgeVideoTask.
    Ensures transactional consistency across tasks, jobs, and state transitions.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.job_repo = WorkflowJobRepository(session)
        self.exec_repo = StageExecutionRepository(session)
        self.workflow = KnowledgeVideoWorkflow(session)

    def create_task(
        self,
        topic: str,
        target_duration: float = 60.0,
        aspect_ratio: str = "16:9",
        language: str = "zh",
        workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
        task_metadata: dict[str, Any] | None = None,
        allow_research: bool = False,
        task_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Atomically creates a new KnowledgeVideoTask and enqueues its initial EVIDENCE job.
        Returns the created domain task.
        """
        ts = now or datetime.now(UTC)

        # 1. Create and persist the Task aggregate root
        task = KnowledgeVideoTask.create(
            topic=topic,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            language=language,
            workflow_policy=workflow_policy,
            task_id=task_id,
            task_metadata=task_metadata,
            allow_research=allow_research,
            now=ts,
        )
        persisted_task = self.task_repo.save_task(task)

        # 2. Create and persist the initial EVIDENCE job
        idempotency_key = f"idemp_{persisted_task.task_id}_evidence_1"
        initial_job = WorkflowJob.create(
            task_id=persisted_task.task_id,
            stage=Stage.EVIDENCE,
            idempotency_key=idempotency_key,
            attempt_number=1,
            now=ts,
        )
        self.job_repo.create_job(initial_job)

        return persisted_task

    def approve_task(
        self,
        task_id: str,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Resumes a task paused in WAITING_USER, advancing to next stage and scheduling its job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        self.workflow.resume_after_approval(task, now=ts)
        return self.task_repo.get_task(task_id) or task

    def authorize_research(
        self,
        task_id: str,
        resume_if_waiting: bool = True,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Authorizes web research for a task. If the task is currently waiting
        in NEEDS_EVIDENCE or WAITING_USER, automatically re-enqueues an EVIDENCE job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot authorize research: task '{task_id}' is in terminal state '{task.task_status.value}'."
            )

        task.authorize_research(now=ts)
        self.task_repo.save_task(task)

        if resume_if_waiting and task.task_status in (
            TaskStatus.NEEDS_EVIDENCE,
            TaskStatus.WAITING_USER,
        ):
            return self.retry_task(task_id, reason="Web research authorized", now=ts)

        return task

    def retry_task(
        self,
        task_id: str,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Retries a failed or paused task by re-enqueuing a job for its current stage.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot retry task '{task_id}' in terminal state '{task.task_status.value}'."
            )

        if task.task_status not in (
            TaskStatus.NEEDS_RECOVERY,
            TaskStatus.NEEDS_EVIDENCE,
        ):
            raise WorkflowConflictError(
                f"Task '{task_id}' is in status '{task.task_status.value}' and cannot be retried."
            )

        cur_job = self.job_repo.get_current_job_for_task(task_id)
        next_attempt = (cur_job.attempt_number + 1) if cur_job else 1

        task.transition_to(TaskStatus.RUNNING, reason=reason, now=ts)
        self.task_repo.save_task(task)

        idempotency_key = f"idemp_{task.task_id}_{task.current_stage.value.lower()}_{next_attempt}"
        new_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=task.current_stage,
            idempotency_key=idempotency_key,
            attempt_number=next_attempt,
            now=ts,
        )
        self.job_repo.create_job(new_job)

        return task

    def cancel_task(
        self,
        task_id: str,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        """
        Authoritatively cancels a task and aborts any active or queued job.
        """
        ts = now or datetime.now(UTC)
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        task.transition_to(TaskStatus.CANCELLED, reason=reason, now=ts)
        self.task_repo.save_task(task)

        cur_job = self.job_repo.get_current_job_for_task(task_id)
        if cur_job and cur_job.status in (
            JobStatus.QUEUED,
            JobStatus.LEASED,
            JobStatus.RUNNING,
        ):
            cur_job.status = JobStatus.CANCELLED
            cur_job.finished_at = ts
            self.job_repo.update_job(cur_job)

        return task
