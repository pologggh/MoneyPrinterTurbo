from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob


class StageExecutionResult(BaseModel):
    success: bool
    output_artifact_revision_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    is_retryable: bool = False


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
