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


def test_default_executor_registry_has_production_plan_and_asset_executors():
    """Verify that default production registry has executors for EVIDENCE, KNOWLEDGE_PLAN, SCRIPT, STORYBOARD, PRODUCTION_PLAN, and ASSET."""
    from app.application.asset_stage_executor import AssetStageExecutor
    from app.application.evidence_stage_executor import EvidenceStageExecutor
    from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
    from app.application.production_plan_stage_executor import ProductionPlanStageExecutor
    from app.application.script_stage_executor import ScriptStageExecutor
    from app.application.storyboard_stage_executor import StoryboardStageExecutor

    registry = get_default_executor_registry()
    assert len(registry.list_supported_stages()) == 6
    assert registry.has_executor(Stage.EVIDENCE)
    assert isinstance(registry.get_executor(Stage.EVIDENCE), EvidenceStageExecutor)
    assert registry.has_executor(Stage.KNOWLEDGE_PLAN)
    assert isinstance(registry.get_executor(Stage.KNOWLEDGE_PLAN), KnowledgePlanStageExecutor)
    assert registry.has_executor(Stage.SCRIPT)
    assert isinstance(registry.get_executor(Stage.SCRIPT), ScriptStageExecutor)
    assert registry.has_executor(Stage.STORYBOARD)
    assert isinstance(registry.get_executor(Stage.STORYBOARD), StoryboardStageExecutor)
    assert registry.has_executor(Stage.PRODUCTION_PLAN)
    assert isinstance(registry.get_executor(Stage.PRODUCTION_PLAN), ProductionPlanStageExecutor)
    assert registry.has_executor(Stage.ASSET)
    assert isinstance(registry.get_executor(Stage.ASSET), AssetStageExecutor)

    for stage in Stage:
        if stage not in (
            Stage.EVIDENCE,
            Stage.KNOWLEDGE_PLAN,
            Stage.SCRIPT,
            Stage.STORYBOARD,
            Stage.PRODUCTION_PLAN,
            Stage.ASSET,
        ):
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
