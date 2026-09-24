from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_job import WorkflowJob


class StageExecutionResult(BaseModel):
    success: bool
    output_artifact_revision_id: str | None = None
    output_artifact_ref: TaskArtifactRef | None = None
    output_task_artifact_ref_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    is_retryable: bool = False
    metadata_json: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class StageExecutorProtocol(Protocol):
    """
    Formal contract for executing a single stage of a KnowledgeVideoTask.
    Stage executors receive the task context and the active job, producing a StageExecutionResult.
    """
    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        ...
