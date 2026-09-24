"""
Asset Stage Executor.

Implements StageExecutorProtocol for Stage.ASSET.
Consumes an immutable AssetRoutePlan produced by Stage.PRODUCTION_PLAN
and invokes AssetRoutePlanExecutionService to coordinate provider attempts,
durable pre-submission AttemptRequests, ProviderReceipts, and ShotAssetVersions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.asset_execution import (
    AssetReuseMode,
    ExecutionRunResult,
    ExecutionStatus,
)
from app.domain.asset_router import AssetRoutePlan
from app.domain.knowledge_video_task import KnowledgeVideoTask
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
    ExecutionRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.shot_execution_service import ShotExecutionService


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


class AssetStageExecutor(StageExecutorProtocol):
    """
    Executes the ASSET workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (ASSET_ROUTE_PLAN).
    2. Loads the exact frozen AssetRoutePlan (zero re-routing).
    3. Invokes existing AssetRoutePlanExecutionService and ShotExecutionService.
    4. Persists durable AttemptRequests before provider network submission.
    5. Stores ProviderReceipts upon provider acceptance and produces ShotAssetVersions on success.
    6. Halts cleanly into NEEDS_RECOVERY on ambiguous submission outcomes.
    7. Creates TaskArtifactRef(EXECUTION_RUN) on complete success.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        execution_service: AssetRoutePlanExecutionService | None = None,
        adapter_registry: AdapterRegistry | None = None,
        storage_base_dir: Path | str | None = None,
        reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE,
        trace_writer: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.execution_service = execution_service
        self.adapter_registry = adapter_registry
        self.storage_base_dir = storage_base_dir
        self.reuse_mode = reuse_mode
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
                f"[AssetStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[AssetStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.ASSET_STAGE_STARTED,
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
            route_plan_repo = AssetRoutePlanRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.PRODUCTION_PLAN,
                    artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
                )

            if input_ref is None:
                err = f"Input artifact ref missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[AssetStageExecutor] {err}")
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
                logger.error(f"[AssetStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.ASSET_ROUTE_PLAN:
                err = (
                    f"Input artifact type '{input_ref.artifact_type}' is invalid, "
                    f"expected '{ArtifactType.ASSET_ROUTE_PLAN}'."
                )
                logger.error(f"[AssetStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            plan = route_plan_repo.get_route_plan(input_ref.artifact_id)
            if plan is None:
                err = f"AssetRoutePlan '{input_ref.artifact_id}' not found in repository."
                logger.error(f"[AssetStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if not plan.is_ready:
                err = (
                    f"AssetRoutePlan '{plan.asset_route_plan_id}' has status '{plan.status.value}', "
                    f"expected READY."
                )
                logger.error(f"[AssetStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

        # ---------------------------------------------------------------------
        # 2. Execution Setup & Invocation
        # ---------------------------------------------------------------------
        storage_dir = (
            Path(self.storage_base_dir)
            if self.storage_base_dir is not None
            else Path("storage") / "tasks" / task_id / "assets"
        ).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)

        exec_service = self.execution_service
        if exec_service is None:
            shot_exec_svc = ShotExecutionService(
                adapter_registry=self.adapter_registry,
                session_factory=self._session_factory,
            )
            exec_service = AssetRoutePlanExecutionService(
                session_factory=self._session_factory,
                shot_execution_service=shot_exec_svc,
            )

        try:
            run_result: ExecutionRunResult = exec_service.execute_route_plan(
                plan=plan,
                storage_base_dir=storage_dir,
                reuse_mode=self.reuse_mode,
                trace_writer=self.trace_writer,
            )
        except Exception as exc:
            err = f"Asset execution crashed: {exc}"
            logger.exception(f"[AssetStageExecutor] {err}")
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.ASSET_STAGE_FAILED,
                attributes={"task_id": task_id, "error": str(exc)},
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.RETRYABLE.value,
                error_message=err,
                is_retryable=True,
            )

        # ---------------------------------------------------------------------
        # 3. Outcome Evaluation
        # ---------------------------------------------------------------------
        if run_result.state == ExecutionStatus.COMPLETED:
            output_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.ASSET,
                artifact_type=ArtifactType.EXECUTION_RUN,
                artifact_id=run_result.execution_run_id,
                metadata_json={
                    "producer_job_id": job.job_id,
                    "asset_route_plan_id": plan.asset_route_plan_id,
                    "storyboard_snapshot_id": plan.storyboard_snapshot_id,
                    "total_shots": run_result.total_shots,
                    "generated_shots": run_result.generated_shots,
                    "reused_shots": run_result.reused_shots,
                    "failed_shots": run_result.failed_shots,
                    "recovery_required_shots": run_result.recovery_required_shots,
                },
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.ASSET_STAGE_COMPLETED,
                attributes={
                    "task_id": task_id,
                    "execution_run_id": run_result.execution_run_id,
                    "total_shots": run_result.total_shots,
                    "generated_shots": run_result.generated_shots,
                    "reused_shots": run_result.reused_shots,
                },
            )

            logger.info(
                f"[AssetStageExecutor] Successfully executed AssetRoutePlan '{plan.asset_route_plan_id}' "
                f"via ExecutionRun '{run_result.execution_run_id}' for task '{task_id}' "
                f"(generated={run_result.generated_shots}, reused={run_result.reused_shots})."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_ref=output_artifact_ref,
                output_task_artifact_ref_id=output_artifact_ref.task_artifact_ref_id,
                output_artifact_revision_id=run_result.execution_run_id,
                metadata_json={
                    "execution_run_id": run_result.execution_run_id,
                    "asset_route_plan_id": plan.asset_route_plan_id,
                    "storyboard_snapshot_id": plan.storyboard_snapshot_id,
                    "total_shots": run_result.total_shots,
                    "generated_shots": run_result.generated_shots,
                    "reused_shots": run_result.reused_shots,
                    "failed_shots": run_result.failed_shots,
                    "recovery_required_shots": run_result.recovery_required_shots,
                },
            )

        if run_result.state == ExecutionStatus.NEEDS_RECOVERY:
            err = (
                f"ExecutionRun '{run_result.execution_run_id}' has unconfirmed submission outcome "
                f"for {run_result.recovery_required_shots} shots, requiring recovery."
            )
            logger.warning(f"[AssetStageExecutor] {err}")
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.ASSET_STAGE_NEEDS_RECOVERY,
                attributes={
                    "task_id": task_id,
                    "execution_run_id": run_result.execution_run_id,
                    "recovery_required_shots": run_result.recovery_required_shots,
                },
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.NEEDS_RECOVERY.value,
                error_message=err,
                is_retryable=False,
            )

        # FAILED or PARTIAL failure
        err = (
            f"ExecutionRun '{run_result.execution_run_id}' failed: "
            f"{run_result.failed_shots} of {run_result.total_shots} shots failed."
        )
        logger.warning(f"[AssetStageExecutor] {err}")
        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.ASSET_STAGE_FAILED,
            attributes={
                "task_id": task_id,
                "execution_run_id": run_result.execution_run_id,
                "failed_shots": run_result.failed_shots,
            },
        )
        return StageExecutionResult(
            success=False,
            error_type=JobErrorType.RETRYABLE.value,
            error_message=err,
            is_retryable=True,
        )
