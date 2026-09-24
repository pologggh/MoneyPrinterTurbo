from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.workflow_state import (
    InvalidStageTransitionError,
    InvalidStateTransitionError,
    JobErrorType,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowPolicyType,
    get_next_stage,
)


class KnowledgeVideoTask(BaseModel):
    """
    Authoritative aggregate root for an end-to-end Knowledge Video production lifecycle.
    Single task_id traverses research, planning, generation, composition, and delivery.
    """
    model_config = ConfigDict(frozen=False)

    task_id: str
    topic: str
    task_status: TaskStatus = TaskStatus.CREATED
    current_stage: Stage = Stage.EVIDENCE
    workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO
    target_duration: float = 60.0
    aspect_ratio: str = "16:9"
    language: str = "zh"
    waiting_reason: str | None = None
    error_type: JobErrorType | None = None
    error_message: str | None = None
    task_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @property
    def is_research_authorized(self) -> bool:
        """Indicates whether open-web research has been explicitly authorized for this task."""
        return bool(self.task_metadata.get("research_authorized", False))

    def authorize_research(self, now: datetime | None = None) -> None:
        """Authoritatively authorizes web research for this task."""
        ts = now or datetime.now(UTC)
        if self.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot authorize research: task '{self.task_id}' is in terminal state '{self.task_status}'."
            )
        self.task_metadata["research_authorized"] = True
        self.updated_at = ts

    @classmethod
    def create(
        cls,
        topic: str,
        target_duration: float = 60.0,
        aspect_ratio: str = "16:9",
        language: str = "zh",
        workflow_policy: WorkflowPolicyType = WorkflowPolicyType.AUTO,
        task_id: str | None = None,
        task_metadata: dict[str, Any] | None = None,
        allow_research: bool = False,
        now: datetime | None = None,
    ) -> KnowledgeVideoTask:
        ts = now or datetime.now(UTC)
        meta = dict(task_metadata or {})
        if allow_research:
            meta["research_authorized"] = True
        return cls(
            task_id=task_id or uuid4().hex,
            topic=topic,
            task_status=TaskStatus.CREATED,
            current_stage=Stage.EVIDENCE,
            workflow_policy=workflow_policy,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            language=language,
            task_metadata=meta,
            created_at=ts,
            updated_at=ts,
        )

    def transition_to(
        self,
        new_status: TaskStatus,
        reason: str | None = None,
        error_type: JobErrorType | None = None,
        error_message: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """
        Executes an authoritative state transition under strict lifecycle invariants.
        """
        ts = now or datetime.now(UTC)

        # 1. Terminal states are strictly immutable
        if self.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Task '{self.task_id}' is in terminal state '{self.task_status}' "
                f"and cannot transition to '{new_status}'."
            )

        # 2. Universal exits to FAILED and CANCELLED from any non-terminal state
        if new_status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            self.task_status = new_status
            self.waiting_reason = reason
            self.error_type = error_type
            self.error_message = error_message
            self.updated_at = ts
            self.finished_at = ts
            return

        # 3. Explicit allowed transitions
        valid_transitions: dict[TaskStatus, set[TaskStatus]] = {
            TaskStatus.CREATED: {TaskStatus.RUNNING},
            TaskStatus.RUNNING: {
                TaskStatus.WAITING_USER,
                TaskStatus.NEEDS_EVIDENCE,
                TaskStatus.NEEDS_RECOVERY,
                TaskStatus.COMPLETED,
            },
            TaskStatus.WAITING_USER: {TaskStatus.RUNNING},
            TaskStatus.NEEDS_EVIDENCE: {TaskStatus.RUNNING},
            TaskStatus.NEEDS_RECOVERY: {TaskStatus.RUNNING},
        }

        allowed = valid_transitions.get(self.task_status, set())
        if new_status not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal task status transition for task '{self.task_id}': "
                f"'{self.task_status}' -> '{new_status}'."
            )

        self.task_status = new_status
        self.waiting_reason = reason if new_status in (TaskStatus.WAITING_USER, TaskStatus.NEEDS_EVIDENCE, TaskStatus.NEEDS_RECOVERY) else None
        self.error_type = error_type
        self.error_message = error_message
        self.updated_at = ts
        if new_status == TaskStatus.COMPLETED:
            self.finished_at = ts

    def advance_stage(self, new_stage: Stage, now: datetime | None = None) -> None:
        """
        Advances the current linear execution stage. Disallows skipping stages.
        """
        ts = now or datetime.now(UTC)
        if self.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot advance stage: task '{self.task_id}' is in terminal state '{self.task_status}'."
            )

        expected_next = get_next_stage(self.current_stage)
        if new_stage != expected_next:
            raise InvalidStageTransitionError(
                f"Cannot advance stage from '{self.current_stage}' to '{new_stage}'. "
                f"Expected sequential next stage: '{expected_next}'."
            )

        self.current_stage = new_stage
        self.updated_at = ts

    def set_stage_for_rerun(self, new_stage: Stage, now: datetime | None = None) -> None:
        """
        Sets stage for partial rerun or skipped stage transition during remediation.
        """
        ts = now or datetime.now(UTC)
        if self.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot change stage: task '{self.task_id}' is in terminal state '{self.task_status}'."
            )
        self.current_stage = new_stage
        self.updated_at = ts
