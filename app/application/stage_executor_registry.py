from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.domain.workflow_state import Stage

if TYPE_CHECKING:
    from app.application.stage_executor_protocol import StageExecutorProtocol


class StageExecutorRegistry:
    """
    Authoritative registry mapping workflow Stages to StageExecutorProtocol implementations.

    In Stage E3, this registry holds the production EvidenceStageExecutor for Stage.EVIDENCE.
    Other business executors (KNOWLEDGE_PLAN, SCRIPT, etc.) remain unregistered and unsupported.
    """

    def __init__(self) -> None:
        self._executors: dict[Stage, StageExecutorProtocol] = {}

    def register(self, stage: Stage, executor: StageExecutorProtocol) -> None:
        """Register an executor for a specific stage."""
        self._executors[stage] = executor

    def get_executor(self, stage: Stage) -> StageExecutorProtocol | None:
        """Retrieve executor for a given stage, or None if unsupported."""
        return self._executors.get(stage)

    def has_executor(self, stage: Stage) -> bool:
        """Check if an executor is registered for a stage."""
        return stage in self._executors

    def list_supported_stages(self) -> set[Stage]:
        """Return the set of stages currently supported by registered executors."""
        return set(self._executors.keys())


def get_default_executor_registry(
    session_factory: Callable[[], Session] | None = None,
) -> StageExecutorRegistry:
    """
    Returns default production executor registry.

    In Stage S1:
    - Stage.EVIDENCE is registered with production EvidenceStageExecutor.
    - Stage.KNOWLEDGE_PLAN is registered with production KnowledgePlanStageExecutor.
    - Stage.SCRIPT is registered with production ScriptStageExecutor.
    - Subsequent stages (Stage.STORYBOARD, etc.) remain unregistered and unsupported.
    """
    from app.application.evidence_stage_executor import EvidenceStageExecutor
    from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
    from app.application.script_stage_executor import ScriptStageExecutor

    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, EvidenceStageExecutor(session_factory=session_factory))
    registry.register(Stage.KNOWLEDGE_PLAN, KnowledgePlanStageExecutor(session_factory=session_factory))
    registry.register(Stage.SCRIPT, ScriptStageExecutor(session_factory=session_factory))
    return registry
