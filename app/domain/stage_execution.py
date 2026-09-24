from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.workflow_state import Stage


class StageExecution(BaseModel):
    """
    Historical record of an individual stage execution attempt.
    Preserves audit trail across retries and restarts.
    """
    model_config = ConfigDict(frozen=True)

    stage_execution_id: str
    task_id: str
    job_id: str
    stage: Stage
    attempt_number: int
    status: str
    input_artifact_revision_id: str | None = None
    output_artifact_revision_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    @classmethod
    def record(
        cls,
        task_id: str,
        job_id: str,
        stage: Stage,
        attempt_number: int,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        input_artifact_revision_id: str | None = None,
        output_artifact_revision_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        stage_execution_id: str | None = None,
    ) -> StageExecution:
        duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
        return cls(
            stage_execution_id=stage_execution_id or uuid4().hex,
            task_id=task_id,
            job_id=job_id,
            stage=stage,
            attempt_number=attempt_number,
            status=status,
            input_artifact_revision_id=input_artifact_revision_id,
            output_artifact_revision_id=output_artifact_revision_id,
            error_type=error_type,
            error_message=error_message,
            duration_ms=duration_ms,
            started_at=started_at,
            finished_at=finished_at,
        )
