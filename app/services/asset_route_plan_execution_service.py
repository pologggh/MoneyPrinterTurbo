from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from app.domain.asset_execution import (
    AssetReuseDecisionType,
    AssetReuseMode,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionRunResult,
    ExecutionStatus,
    ExecutionTransition,
    RoutePlanNotReadyError,
    ShotExecution,
    ShotExecutionStatus,
    ShotExecutionSummary,
)
from app.domain.asset_router import AssetRoutePlan
from app.persistence.repositories import ExecutionRepository, ShotExecutionRepository
from app.persistence.session import get_session
from app.services.asset_reuse_policy import AssetReusePolicy
from app.services.shot_execution_service import ShotExecutionService


class AssetRoutePlanExecutionService:
    """
    Orchestrates execution of an entire AssetRoutePlan and supports incremental rerun.

    Key Invariants:
    1. Only AssetRoutePlan in READY status can be executed (otherwise RoutePlanNotReadyError).
    2. Zero router re-computation: uses frozen plan route candidates directly.
    3. Technical reuse: compatible existing ShotAssetVersions are reused without calling providers.
    4. Exact immutable provenance: reused shots reference original asset versions.
    5. Continuation across technical failures: a failed shot does not abort subsequent independent shots.
    6. Halt without blind retry: ambiguous submissions (NEEDS_RECOVERY) halt that shot cleanly.
    7. Clean transaction boundaries: provider network calls are NEVER made inside open DB transactions.
    8. Deterministic aggregation: run outcome is strictly COMPLETED, FAILED, or NEEDS_RECOVERY.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        shot_execution_service: ShotExecutionService | None = None,
        reuse_policy: AssetReusePolicy | None = None,
    ):
        if isinstance(session_factory, Session):
            raise TypeError(
                "session_factory must be a sqlalchemy.orm.sessionmaker, not a live Session"
            )
        self._session_factory = session_factory
        self._shot_execution_service = shot_execution_service or ShotExecutionService()
        self._reuse_policy = reuse_policy or AssetReusePolicy()

    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        """Provides a safe DB session with automatic commit/rollback and closure."""
        if self._session_factory is None:
            with get_session() as session:
                yield session
        else:
            with get_session(self._session_factory) as session:
                yield session

    def execute_route_plan(
        self,
        plan: AssetRoutePlan,
        storage_base_dir: Path | str,
        reuse_mode: AssetReuseMode = AssetReuseMode.REUSE_COMPATIBLE,
        run_id: str | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> ExecutionRunResult:
        """
        Executes all shots in an AssetRoutePlan sequentially or evaluates for reuse.
        Returns a typed ExecutionRunResult detailing the outcome of each shot and the overall run.
        """
        # Invariant 1: Only READY plans may be executed
        if not plan.is_ready:
            raise RoutePlanNotReadyError(
                f"AssetRoutePlan {plan.asset_route_plan_id} has status {plan.status.value}, expected READY"
            )

        execution_run_id = run_id or str(uuid4())
        started_at = datetime.now(UTC)
        total_shots = len(plan.shot_routes)

        run = ExecutionRun(
            execution_run_id=execution_run_id,
            asset_route_plan_id=plan.asset_route_plan_id,
            storyboard_snapshot_id=plan.storyboard_snapshot_id,
            status=ExecutionStatus.RUNNING,
            total_shots=total_shots,
            succeeded_shots=0,
            reused_shots=0,
            failed_shots=0,
            recovery_required_shots=0,
            started_at=started_at,
        )

        # Persist initial ExecutionRun in short DB transaction
        with self._session_scope() as session:
            exec_repo = ExecutionRepository(session)
            exec_repo.add_execution_run(run)

        run_ev = None
        run_ctx = None
        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import TraceEventStatus, TraceEventType

            run_ctx = trace_context.child_context(
                parent_event_id=trace_context.parent_event_id,
                execution_run_id=execution_run_id,
                asset_route_plan_id=plan.asset_route_plan_id,
                storyboard_snapshot_id=plan.storyboard_snapshot_id,
            )
            run_ev = trace_writer.start_event(
                context=run_ctx,
                event_type=TraceEventType.EXECUTION_RUN_STARTED,
                attributes={
                    "execution_run_id": execution_run_id,
                    "asset_route_plan_id": plan.asset_route_plan_id,
                    "storyboard_snapshot_id": plan.storyboard_snapshot_id,
                    "total_shots": total_shots,
                },
            )

        shot_summaries: list[ShotExecutionSummary] = []
        succeeded_shots = 0
        reused_shots = 0
        failed_shots = 0
        recovery_required_shots = 0

        try:
            for entry in plan.shot_routes:
                base_c = run_ctx if run_ctx is not None else trace_context
                summary, status = self._execute_single_shot_entry(
                    entry=entry,
                    execution_run_id=execution_run_id,
                    storage_base_dir=storage_base_dir,
                    reuse_mode=reuse_mode,
                    base_ctx=base_c,
                    trace_writer=trace_writer,
                    run_ev=run_ev,
                )
                shot_summaries.append(summary)
                if status == ShotExecutionStatus.SUCCEEDED:
                    succeeded_shots += 1
                elif status == ShotExecutionStatus.REUSED:
                    reused_shots += 1
                elif status in (
                    ShotExecutionStatus.NEEDS_RECOVERY,
                    ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
                ):
                    recovery_required_shots += 1
                else:
                    failed_shots += 1
        except Exception as unhandled_run_exc:  # noqa: BLE001
            finished_at = datetime.now(UTC)
            final_status = ExecutionStatus.FAILED
            logger.exception(
                f"Unhandled execution exception in plan {plan.asset_route_plan_id}, "
                f"run {execution_run_id}: {unhandled_run_exc}"
            )
            failed_shots = max(1, total_shots - succeeded_shots - reused_shots)
            try:
                with self._session_scope() as session:
                    exec_repo = ExecutionRepository(session)
                    exec_repo.update_execution_run_status(
                        run_id=execution_run_id,
                        status=final_status,
                        succeeded_shots=succeeded_shots,
                        reused_shots=reused_shots,
                        failed_shots=failed_shots,
                        recovery_required_shots=recovery_required_shots,
                        finished_at=finished_at,
                    )
            except Exception as db_exc:  # noqa: BLE001
                logger.error(f"Failed to update ExecutionRun status to FAILED in DB: {db_exc}")

            if run_ev is not None and trace_writer is not None and run_ctx is not None:
                from app.domain.trace import TraceEventStatus
                try:
                    trace_writer.complete_event(
                        event=run_ev,
                        status=TraceEventStatus.FAILED,
                        error_code="UNHANDLED_EXECUTION_EXCEPTION",
                        attributes_update={"error": str(unhandled_run_exc)},
                    )
                except Exception as tr_exc:  # noqa: BLE001
                    logger.debug(f"Trace completion error: {tr_exc}")

            return ExecutionRunResult(
                execution_run_id=execution_run_id,
                asset_route_plan_id=plan.asset_route_plan_id,
                storyboard_snapshot_id=plan.storyboard_snapshot_id,
                state=final_status,
                total_shots=total_shots,
                generated_shots=succeeded_shots,
                reused_shots=reused_shots,
                failed_shots=failed_shots,
                recovery_required_shots=recovery_required_shots,
                shot_results=tuple(shot_summaries),
                started_at=started_at,
                finished_at=finished_at,
            )

        # 3. Final aggregation of run status
        finished_at = datetime.now(UTC)
        if recovery_required_shots > 0:
            final_status = ExecutionStatus.NEEDS_RECOVERY
        elif succeeded_shots + reused_shots == total_shots:
            final_status = ExecutionStatus.COMPLETED
        else:
            final_status = ExecutionStatus.FAILED

        # Update ExecutionRun record in short DB transaction
        with self._session_scope() as session:
            exec_repo = ExecutionRepository(session)
            exec_repo.update_execution_run_status(
                run_id=execution_run_id,
                status=final_status,
                succeeded_shots=succeeded_shots,
                reused_shots=reused_shots,
                failed_shots=failed_shots,
                recovery_required_shots=recovery_required_shots,
                finished_at=finished_at,
            )

        if run_ev is not None and trace_writer is not None and run_ctx is not None:
            run_trace_status = (
                TraceEventStatus.SUCCEEDED
                if final_status == ExecutionStatus.COMPLETED
                else (
                    TraceEventStatus.INDETERMINATE
                    if final_status == ExecutionStatus.NEEDS_RECOVERY
                    else TraceEventStatus.FAILED
                )
            )
            trace_writer.complete_event(
                event=run_ev,
                status=run_trace_status,
                attributes_update={
                    "final_status": final_status.value,
                    "succeeded_shots": succeeded_shots,
                    "reused_shots": reused_shots,
                    "failed_shots": failed_shots,
                    "recovery_required_shots": recovery_required_shots,
                },
            )
            trace_writer.record_event(
                context=run_ctx,
                event_type=TraceEventType.EXECUTION_RUN_COMPLETED,
                status=run_trace_status,
                attributes={
                    "execution_run_id": execution_run_id,
                    "final_status": final_status.value,
                    "total_shots": total_shots,
                    "succeeded_shots": succeeded_shots,
                    "reused_shots": reused_shots,
                    "failed_shots": failed_shots,
                    "recovery_required_shots": recovery_required_shots,
                },
                parent_event_id=run_ev.trace_event_id,
            )

        return ExecutionRunResult(
            execution_run_id=execution_run_id,
            asset_route_plan_id=plan.asset_route_plan_id,
            storyboard_snapshot_id=plan.storyboard_snapshot_id,
            state=final_status,
            total_shots=total_shots,
            generated_shots=succeeded_shots,
            reused_shots=reused_shots,
            failed_shots=failed_shots,
            recovery_required_shots=recovery_required_shots,
            shot_results=tuple(shot_summaries),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _execute_single_shot_entry(
        self,
        entry: Any,
        execution_run_id: str,
        storage_base_dir: Path | str,
        reuse_mode: AssetReuseMode,
        base_ctx: Any | None,
        trace_writer: Any | None,
        run_ev: Any | None,
    ) -> tuple[ShotExecutionSummary, ShotExecutionStatus]:
        from app.domain.trace import TraceEventStatus, TraceEventType

        shot_ev = None
        shot_ctx = None
        if trace_writer is not None and base_ctx is not None:
            shot_ctx = base_ctx.child_context(
                parent_event_id=run_ev.trace_event_id if run_ev else base_ctx.parent_event_id,
                shot_id=entry.shot_id,
                shot_revision_id=entry.shot_revision_id,
                execution_run_id=execution_run_id,
            )
            shot_ev = trace_writer.start_event(
                context=shot_ctx,
                event_type=TraceEventType.SHOT_EXECUTION_STARTED,
                attributes={
                    "shot_id": entry.shot_id,
                    "shot_revision_id": entry.shot_revision_id,
                    "execution_run_id": execution_run_id,
                    "requested_visual_type": entry.asset_routing_request.requested_visual_type.value
                    if hasattr(entry.asset_routing_request.requested_visual_type, "value")
                    else str(entry.asset_routing_request.requested_visual_type),
                },
            )

        # 1. Fetch historical candidate asset versions in short DB transaction
        with self._session_scope() as session:
            exec_repo = ExecutionRepository(session)
            candidate_versions = exec_repo.list_asset_versions_for_shot_revision(
                entry.shot_revision_id
            )

        # 2. Evaluate technical compatibility for reuse
        expected_gen_mode = (
            entry.route_decision.selected_candidate.generation_mode
            if entry.route_decision.selected_candidate
            else None
        )
        decision = self._reuse_policy.evaluate(
            request=entry.asset_routing_request,
            candidate_versions=candidate_versions,
            reuse_mode=reuse_mode,
            expected_generation_mode=expected_gen_mode,
        )

        # Branch A: Reuse existing compatible asset
        if decision.decision == AssetReuseDecisionType.REUSE:
            shot_exec = ShotExecution(
                execution_run_id=execution_run_id,
                shot_id=entry.shot_id,
                shot_revision_id=entry.shot_revision_id,
                route_decision=entry.route_decision,
                status=ShotExecutionStatus.REUSED,
                produced_asset_version_id=decision.candidate_asset_version_id,
                attempt_ids=[],
                finished_at=datetime.now(UTC),
            )
            with self._session_scope() as session:
                shot_repo = ShotExecutionRepository(session)
                shot_repo.add_shot_execution(shot_exec)
                shot_repo.record_transition(
                    ExecutionTransition(
                        entity_type="SHOT_EXECUTION",
                        entity_id=shot_exec.shot_execution_id,
                        from_state=ShotExecutionStatus.PENDING.value,
                        to_state=ShotExecutionStatus.REUSED.value,
                        reason_code="REUSE_COMPATIBLE_ASSET",
                    )
                )

            if shot_ev is not None and trace_writer is not None:
                trace_writer.complete_event(
                    event=shot_ev,
                    status=TraceEventStatus.SUCCEEDED,
                    attributes_update={
                        "status": "REUSED",
                        "candidate_asset_version_id": decision.candidate_asset_version_id,
                    },
                )

            summary = ShotExecutionSummary(
                shot_id=entry.shot_id,
                shot_revision_id=entry.shot_revision_id,
                shot_execution_id=shot_exec.shot_execution_id,
                asset_version_id=decision.candidate_asset_version_id,
                result_type="REUSED",
                status=ShotExecutionStatus.REUSED,
                failure_reason=None,
                attempts_count=0,
            )
            return summary, ShotExecutionStatus.REUSED

        # Branch B: Execute shot via provider adapters
        shot_exec = ShotExecution(
            execution_run_id=execution_run_id,
            shot_id=entry.shot_id,
            shot_revision_id=entry.shot_revision_id,
            route_decision=entry.route_decision,
            status=ShotExecutionStatus.PENDING,
        )
        with self._session_scope() as session:
            shot_repo = ShotExecutionRepository(session)
            shot_repo.add_shot_execution(shot_exec)
            shot_repo.record_transition(
                ExecutionTransition(
                    entity_type="SHOT_EXECUTION",
                    entity_id=shot_exec.shot_execution_id,
                    from_state="NONE",
                    to_state=ShotExecutionStatus.PENDING.value,
                    reason_code="INITIALIZED",
                )
            )

        # Execution happens OUTSIDE DB transaction (no open tx across network call!)
        try:
            updated_shot_exec, attempts, asset_version = (
                self._shot_execution_service.execute_shot(
                    shot_execution=shot_exec,
                    request=entry.asset_routing_request,
                    storage_base_dir=storage_base_dir,
                    trace_context=shot_ctx,
                    trace_writer=trace_writer,
                    session_factory=self._session_factory,
                )
            )
        except Exception as shot_exc:  # noqa: BLE001
            logger.exception(
                f"Unhandled exception during shot execution for shot {entry.shot_id}: {shot_exc}"
            )
            updated_shot_exec = shot_exec.record_failure(
                error_code="UNHANDLED_SHOT_EXCEPTION",
                error_message=f"Shot execution crashed: {shot_exc}",
            )
            attempts = []
            asset_version = None

        # Persist execution results in short DB transaction
        with self._session_scope() as session:
            shot_repo = ShotExecutionRepository(session)
            exec_repo = ExecutionRepository(session)

            for attempt in attempts:
                exec_repo.save_or_update_execution_attempt(attempt)
                if attempt.provider_receipt:
                    shot_repo.save_provider_receipt(attempt.provider_receipt)
                if attempt.attempt_result:
                    shot_repo.save_attempt_result(attempt.attempt_result)
                shot_repo.record_transition(
                    ExecutionTransition(
                        entity_type="EXECUTION_ATTEMPT",
                        entity_id=attempt.execution_attempt_id,
                        from_state=ExecutionAttemptStatus.SUBMITTING.value,
                        to_state=attempt.status.value,
                        reason_code=attempt.error_code or "ATTEMPT_FINISHED",
                    )
                )

            if asset_version:
                exec_repo.add_shot_asset_version(asset_version)

            shot_repo.update_shot_execution(updated_shot_exec)
            shot_repo.record_transition(
                ExecutionTransition(
                    entity_type="SHOT_EXECUTION",
                    entity_id=updated_shot_exec.shot_execution_id,
                    from_state=ShotExecutionStatus.RUNNING.value,
                    to_state=updated_shot_exec.status.value,
                    reason_code=updated_shot_exec.error_code or "COMPLETED",
                )
            )

        # Aggregate shot results
        if updated_shot_exec.status == ShotExecutionStatus.SUCCEEDED:
            res_type = "GENERATED"
        elif updated_shot_exec.status in (
            ShotExecutionStatus.NEEDS_RECOVERY,
            ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
        ):
            res_type = "NEEDS_RECOVERY"
        else:
            res_type = "FAILED"

        if shot_ev is not None and trace_writer is not None:
            st = (
                TraceEventStatus.SUCCEEDED
                if updated_shot_exec.status == ShotExecutionStatus.SUCCEEDED
                else (
                    TraceEventStatus.INDETERMINATE
                    if updated_shot_exec.status in (
                        ShotExecutionStatus.NEEDS_RECOVERY,
                        ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
                    )
                    else TraceEventStatus.FAILED
                )
            )
            trace_writer.complete_event(
                event=shot_ev,
                status=st,
                attributes_update={
                    "status": updated_shot_exec.status.value,
                    "produced_asset_version_id": updated_shot_exec.produced_asset_version_id,
                    "error_code": updated_shot_exec.error_code,
                },
                error_code=updated_shot_exec.error_code,
            )

        summary = ShotExecutionSummary(
            shot_id=entry.shot_id,
            shot_revision_id=entry.shot_revision_id,
            shot_execution_id=updated_shot_exec.shot_execution_id,
            asset_version_id=updated_shot_exec.produced_asset_version_id,
            result_type=res_type,
            status=updated_shot_exec.status,
            failure_reason=updated_shot_exec.error_message,
            attempts_count=len(attempts),
        )
        return summary, updated_shot_exec.status

    def get_execution_run(self, run_id: str) -> ExecutionRun | None:
        """Loads an ExecutionRun by ID."""
        with self._session_scope() as session:
            exec_repo = ExecutionRepository(session)
            return exec_repo.get_execution_run(run_id)

    def list_runs_for_plan(self, plan_id: str) -> list[ExecutionRun]:
        """Lists all execution runs for a route plan."""
        with self._session_scope() as session:
            exec_repo = ExecutionRepository(session)
            return exec_repo.list_runs_for_plan(plan_id)

    def list_shot_executions(self, run_id: str) -> list[ShotExecution]:
        """Lists all shot executions for an execution run."""
        with self._session_scope() as session:
            shot_repo = ShotExecutionRepository(session)
            return shot_repo.list_shot_executions_for_run(run_id)
