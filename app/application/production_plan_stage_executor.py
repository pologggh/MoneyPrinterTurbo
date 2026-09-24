"""
Production Plan Stage Executor.

Implements StageExecutorProtocol for Stage.PRODUCTION_PLAN.
Consumes an APPROVED StoryboardSnapshot and invokes AssetRoutePlanningService
to produce an immutable AssetRoutePlan without executing any external media providers.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.asset_router import (
    AssetRoutePlan,
    AssetRoutePlanStatus,
    ModelSelectionMode,
    RoutingStrategy,
    StoryboardNotApprovedError,
)
from app.domain.enums import StoryboardSnapshotState
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.storyboard import StoryboardSnapshot
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    Stage,
)
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ShotRepository,
    StoryboardRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
from app.services.asset_capability_registry import (
    AssetCapabilityRegistry,
    get_default_capability_registry,
)
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)
from app.services.hybrid_asset_router import HybridAssetRouter


@contextmanager
def _managed_session(
    session_factory: Callable[[], Session] | None,
) -> Generator[Session, None, None]:
    if session_factory is None:
        with get_session() as session:
            yield session
    else:
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class ProductionPlanStageExecutor(StageExecutorProtocol):
    """
    Executes the PRODUCTION_PLAN workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (STORYBOARD_SNAPSHOT).
    2. Validates that the StoryboardSnapshot is in APPROVED state.
       (DRAFT or unapproved snapshots are rejected).
    3. Invokes existing AssetRoutePlanningService without calling external providers.
    4. Persists the immutable AssetRoutePlan in AssetRoutePlanRepository.
    5. Returns StageExecutionResult with TaskArtifactRef(ASSET_ROUTE_PLAN).
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        route_planning_service: AssetRoutePlanningService | None = None,
        capability_registry: AssetCapabilityRegistry | None = None,
        trace_writer: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.route_planning_service = route_planning_service
        self.capability_registry = capability_registry
        self.trace_writer = trace_writer

    def _emit_trace(
        self,
        task_id: str,
        event_type: TraceEventType,
        attributes: dict[str, Any],
    ) -> None:
        if self.trace_writer is None:
            return
        try:
            self.trace_writer.write_event(
                task_id=task_id,
                event_type=event_type,
                attributes=attributes,
            )
        except Exception as exc:
            logger.warning(
                f"[ProductionPlanStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[ProductionPlanStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.PRODUCTION_PLAN_STARTED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "attempt_number": job.attempt_number,
            },
        )

        # ---------------------------------------------------------------------
        # 1. Input Resolution & Lineage Verification
        # ---------------------------------------------------------------------
        input_ref_id = job.input_task_artifact_ref_id

        with _managed_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            sb_repo = StoryboardRepository(session)
            shot_repo = ShotRepository(session)
            plan_repo = ContentPlanRepository(session)
            route_plan_repo = AssetRoutePlanRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.STORYBOARD,
                    artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
                )

            if input_ref is None:
                err = f"Input artifact ref missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[ProductionPlanStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if input_ref.task_id != task_id:
                err = (
                    f"Task ID mismatch: Job task '{task_id}' != "
                    f"input artifact task '{input_ref.task_id}'."
                )
                logger.error(f"[ProductionPlanStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.STORYBOARD_SNAPSHOT:
                err = (
                    f"Input artifact type '{input_ref.artifact_type}' is invalid, "
                    f"expected '{ArtifactType.STORYBOARD_SNAPSHOT}'."
                )
                logger.error(f"[ProductionPlanStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            snapshot = sb_repo.get_snapshot(input_ref.artifact_id)
            if snapshot is None:
                err = f"StoryboardSnapshot '{input_ref.artifact_id}' not found in repository."
                logger.error(f"[ProductionPlanStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 2. Approved Storyboard Invariant Verification
            # -----------------------------------------------------------------
            if snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
                err = (
                    f"Cannot route storyboard snapshot '{snapshot.storyboard_snapshot_id}' "
                    f"because state is '{snapshot.snapshot_state.value}', expected APPROVED."
                )
                logger.error(f"[ProductionPlanStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.PRODUCTION_PLAN_FAILED,
                    attributes={"task_id": task_id, "error": err},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 3. Route Planning Invocation
            # -----------------------------------------------------------------
            planning_service = self.route_planning_service
            if planning_service is None:
                registry = self.capability_registry or get_default_capability_registry()
                router = HybridAssetRouter(registry=registry)
                planning_service = AssetRoutePlanningService(
                    plan_repository=plan_repo,
                    shot_repository=shot_repo,
                    storyboard_repository=sb_repo,
                    route_plan_repository=route_plan_repo,
                    capability_registry=registry,
                    router=router,
                )

            # Extract strategy and mode from task metadata if provided
            strat_raw = task.task_metadata.get("routing_strategy", RoutingStrategy.BALANCED)
            try:
                strategy = RoutingStrategy(strat_raw)
            except ValueError:
                strategy = RoutingStrategy.BALANCED

            mode_raw = task.task_metadata.get("model_selection_mode", ModelSelectionMode.AUTO)
            try:
                selection_mode = ModelSelectionMode(mode_raw)
            except ValueError:
                selection_mode = ModelSelectionMode.AUTO

            input_data = CreateAssetRoutePlanInput(
                approved_storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
                routing_strategy=strategy,
                model_selection_mode=selection_mode,
                selected_provider=task.task_metadata.get("selected_provider"),
                selected_model=task.task_metadata.get("selected_model"),
                target_aspect_ratio=task.aspect_ratio or "16:9",
            )

            try:
                route_plan = planning_service.create_route_plan(input_data)
            except Exception as exc:
                err = f"Route planning failed: {exc}"
                logger.exception(f"[ProductionPlanStageExecutor] {err}")
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.PRODUCTION_PLAN_FAILED,
                    attributes={"task_id": task_id, "error": str(exc)},
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=err,
                    is_retryable=True,
                )

            # Ensure route plan is persisted in database
            existing_plan = route_plan_repo.get_route_plan(route_plan.asset_route_plan_id)
            if existing_plan is None:
                route_plan_repo.add_route_plan(route_plan)

            # -----------------------------------------------------------------
            # 4. Result Evaluation & TaskArtifactRef Output
            # -----------------------------------------------------------------
            if route_plan.is_blocked:
                err = (
                    f"AssetRoutePlan '{route_plan.asset_route_plan_id}' is BLOCKED "
                    f"({route_plan.blocked_shots} of {route_plan.total_shots} shots blocked)."
                )
                logger.warning(f"[ProductionPlanStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            output_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.PRODUCTION_PLAN,
                artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
                artifact_id=route_plan.asset_route_plan_id,
                metadata_json={
                    "producer_job_id": job.job_id,
                    "storyboard_snapshot_id": snapshot.storyboard_snapshot_id,
                    "content_plan_revision_id": snapshot.content_plan_revision_id,
                    "total_shots": route_plan.total_shots,
                    "routed_shots": route_plan.routed_shots,
                    "blocked_shots": route_plan.blocked_shots,
                    "status": route_plan.status.value,
                    "routing_strategy": route_plan.routing_strategy.value,
                },
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.ASSET_ROUTE_PLAN_CREATED,
                attributes={
                    "task_id": task_id,
                    "asset_route_plan_id": route_plan.asset_route_plan_id,
                    "storyboard_snapshot_id": snapshot.storyboard_snapshot_id,
                    "total_shots": route_plan.total_shots,
                    "routed_shots": route_plan.routed_shots,
                    "blocked_shots": route_plan.blocked_shots,
                },
            )

            logger.info(
                f"[ProductionPlanStageExecutor] Successfully generated AssetRoutePlan "
                f"'{route_plan.asset_route_plan_id}' with {route_plan.routed_shots} routed shots "
                f"for task '{task_id}'."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_ref=output_artifact_ref,
                output_task_artifact_ref_id=output_artifact_ref.task_artifact_ref_id,
                output_artifact_revision_id=route_plan.asset_route_plan_id,
                metadata_json={
                    "storyboard_snapshot_id": snapshot.storyboard_snapshot_id,
                    "content_plan_revision_id": snapshot.content_plan_revision_id,
                    "asset_route_plan_id": route_plan.asset_route_plan_id,
                    "total_shots": route_plan.total_shots,
                    "routed_shots": route_plan.routed_shots,
                    "blocked_shots": route_plan.blocked_shots,
                    "status": route_plan.status.value,
                },
            )
