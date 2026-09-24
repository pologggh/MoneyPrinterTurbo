from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.workflow_state import Stage

if TYPE_CHECKING:
    from app.application.stage_executor_protocol import StageExecutorProtocol


class StageExecutorRegistry:
    """
    Authoritative registry mapping workflow Stages to StageExecutorProtocol implementations.

    In stage U1-C, this registry holds 0 business executors in production.
    Executors are registered explicitly by application configuration or tests.
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


def get_default_executor_registry() -> StageExecutorRegistry:
    """
    Returns default production executor registry.

    In U1-C foundation, NO business executors are registered (empty registry).
    Unsupported stages (including the initial EVIDENCE stage) remain safely QUEUED.
    """
    return StageExecutorRegistry()
