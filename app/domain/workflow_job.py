from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.workflow_state import JobStatus, Stage, WorkflowConflictError


class WorkflowJob(BaseModel):
    """
    Persistent execution unit for a single stage of a KnowledgeVideoTask.
    Maintains leases, heartbeats, attempts, idempotency keys, and artifact links.
    """
    model_config = ConfigDict(frozen=False)

    job_id: str
    task_id: str
    stage: Stage
    status: JobStatus = JobStatus.QUEUED
    idempotency_key: str
    attempt_number: int = 1
    max_attempts: int = 3
    available_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    input_artifact_revision_id: str | None = None
    output_artifact_revision_id: str | None = None
    input_task_artifact_ref_id: str | None = None
    output_task_artifact_ref_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def create(
        cls,
        task_id: str,
        stage: Stage,
        idempotency_key: str,
        job_id: str | None = None,
        attempt_number: int = 1,
        max_attempts: int = 3,
        input_artifact_revision_id: str | None = None,
        input_task_artifact_ref_id: str | None = None,
        now: datetime | None = None,
    ) -> WorkflowJob:
        ts = now or datetime.now(UTC)
        ref_id = input_task_artifact_ref_id or input_artifact_revision_id
        return cls(
            job_id=job_id or uuid4().hex,
            task_id=task_id,
            stage=stage,
            status=JobStatus.QUEUED,
            idempotency_key=idempotency_key,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            available_at=ts,
            input_artifact_revision_id=ref_id,
            input_task_artifact_ref_id=ref_id,
            created_at=ts,
        )

    def is_lease_expired(self, now: datetime | None = None) -> bool:
        """Determines if any existing lease on this job has expired."""
        if not self.lease_expires_at:
            return True
        ts = now or datetime.now(UTC)
        return ts >= self.lease_expires_at

    def acquire_lease(
        self,
        owner: str,
        lease_duration_seconds: int = 300,
        now: datetime | None = None,
    ) -> None:
        """Acquires or re-acquires an expired lease for a specific worker owner."""
        ts = now or datetime.now(UTC)
        if (
            self.status in (JobStatus.LEASED, JobStatus.RUNNING)
            and self.lease_owner != owner
            and not self.is_lease_expired(now=ts)
        ):
            raise WorkflowConflictError(
                f"Job '{self.job_id}' is actively leased to '{self.lease_owner}' until '{self.lease_expires_at}'."
            )

        self.status = JobStatus.LEASED
        self.lease_owner = owner
        self.lease_expires_at = ts + timedelta(seconds=lease_duration_seconds)
        self.heartbeat_at = ts

    def renew_lease(
        self,
        owner: str,
        extend_seconds: int = 300,
        now: datetime | None = None,
    ) -> None:
        """Renews an active lease held by the current owner."""
        ts = now or datetime.now(UTC)
        if self.lease_owner != owner:
            raise WorkflowConflictError(
                f"Cannot renew lease on job '{self.job_id}': held by '{self.lease_owner}', not '{owner}'."
            )
        self.lease_expires_at = ts + timedelta(seconds=extend_seconds)
        self.heartbeat_at = ts

    def record_heartbeat(self, now: datetime | None = None) -> None:
        """Records an ongoing heartbeat from the worker."""
        self.heartbeat_at = now or datetime.now(UTC)

    def start(self, now: datetime | None = None) -> None:
        """Marks the leased job as actively running."""
        ts = now or datetime.now(UTC)
        self.status = JobStatus.RUNNING
        self.started_at = ts
        self.record_heartbeat(now=ts)

    def mark_succeeded(
        self,
        output_artifact_revision_id: str | None = None,
        output_task_artifact_ref_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Marks the job as successfully completed."""
        ts = now or datetime.now(UTC)
        self.status = JobStatus.SUCCEEDED
        ref_id = output_task_artifact_ref_id or output_artifact_revision_id
        self.output_artifact_revision_id = ref_id
        self.output_task_artifact_ref_id = ref_id
        self.finished_at = ts
        self.lease_expires_at = None

    def mark_failed(
        self,
        error_type: str,
        error_message: str,
        is_retryable: bool = False,
        now: datetime | None = None,
    ) -> None:
        """Marks the job as failed or retryable."""
        ts = now or datetime.now(UTC)
        self.error_type = error_type
        self.error_message = error_message
        self.finished_at = ts
        self.lease_expires_at = None

        if is_retryable and self.attempt_number < self.max_attempts:
            self.status = JobStatus.RETRYABLE
        else:
            self.status = JobStatus.FAILED
