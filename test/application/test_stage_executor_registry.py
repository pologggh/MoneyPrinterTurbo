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


def test_default_executor_registry_has_evidence_and_knowledge_plan_executors():
    """Verify that default production registry has EvidenceStageExecutor for EVIDENCE and KnowledgePlanStageExecutor for KNOWLEDGE_PLAN."""
    from app.application.evidence_stage_executor import EvidenceStageExecutor
    from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor

    registry = get_default_executor_registry()
    assert len(registry.list_supported_stages()) == 2
    assert registry.has_executor(Stage.EVIDENCE)
    assert isinstance(registry.get_executor(Stage.EVIDENCE), EvidenceStageExecutor)
    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    assert isinstance(registry.get_executor(Stage.KNOWLEDGE_PLAN), KnowledgePlanStageExecutor)

    for stage in Stage:
        if stage not in (Stage.EVIDENCE, Stage.KNOWLEDGE_PLAN):
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
