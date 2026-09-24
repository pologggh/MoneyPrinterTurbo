from __future__ import annotations

import pytest

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import Stage


class DummyStageExecutor:
    def execute(self, task: KnowledgeVideoTask, job: WorkflowJob) -> StageExecutionResult:
        return StageExecutionResult(success=True)


def test_default_executor_registry_has_zero_executors():
    """Verify that default production registry has exactly 0 business executors registered."""
    registry = get_default_executor_registry()
    assert len(registry.list_supported_stages()) == 0
    for stage in Stage:
        assert not registry.has_executor(stage)
        assert registry.get_executor(stage) is None


def test_stage_executor_registry_registration_and_lookup():
    """Verify explicit registration and lookup of stage executors."""
    registry = StageExecutorRegistry()
    executor = DummyStageExecutor()

    assert not registry.has_executor(Stage.KNOWLEDGE_PLAN)
    registry.register(Stage.KNOWLEDGE_PLAN, executor)

    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    assert registry.get_executor(Stage.KNOWLEDGE_PLAN) is executor
    assert registry.list_supported_stages() == {Stage.KNOWLEDGE_PLAN}
    assert registry.get_executor(Stage.EVIDENCE) is None
