from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType
from app.domain.evidence import EvidenceItem, EvidenceSnapshot
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.planner import (
    ContentPlanner,
    InsufficientEvidenceError,
    InvalidBeatLineageInheritanceError,
    InvalidDurationPlanError,
    PlannerInput,
    PlannerOutputInvalidError,
    PlannerOutputSchemaError,
    PlannerProviderError,
)
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobErrorType, Stage
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
from app.services.trace_service import TraceWriter


class KnowledgePlanStageExecutor:
    """Production StageExecutor for the KNOWLEDGE_PLAN stage in the unified workflow.

    Connects the durable KnowledgeVideoWorkflow to the existing ContentPlanner:
    1. Resolves and validates the exact EvidenceSnapshot input artifact produced by EVIDENCE stage.
    2. Validates task_id lineage and resolves all referenced EvidenceItem entities.
    3. Maps EvidenceSnapshot -> PlannerInput with strict evidence-first grounding (source_grounded=True).
    4. Executes the existing ContentPlanner with bounded retries and schema/duration validation.
    5. Deterministically post-validates that factual KNOWLEDGE beats cite real EvidenceItem IDs from the snapshot.
    6. Persists the immutable ContentPlanRevision and records TaskArtifactRef(CONTENT_PLAN_REVISION).
    7. Emits structured Trace events without leaking prompt or source payloads.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        llm_caller: Callable[[str], str] | None = None,
        max_retries: int = 2,
        duration_tolerance_ratio: float = 0.15,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.llm_caller = llm_caller
        self.max_retries = max_retries
        self.duration_tolerance_ratio = duration_tolerance_ratio
        self.trace_writer = trace_writer

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[KnowledgePlanStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        # ---------------------------------------------------------------------
        # 1. Input Resolution & Invariant Verification
        # ---------------------------------------------------------------------
        input_ref_id = job.input_task_artifact_ref_id

        with get_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            ev_repo = EvidenceRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.EVIDENCE,
                    artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
                )

            if input_ref is None:
                logger.error(
                    f"[KnowledgePlanStageExecutor] Input artifact ref missing for task '{task_id}', "
                    f"job '{job.job_id}'."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Input artifact reference not found for task '{task_id}'.",
                    is_retryable=False,
                )

            # Validate input artifact lineage and stage type
            if input_ref.task_id != task_id:
                logger.error(
                    f"[KnowledgePlanStageExecutor] Input artifact ref task_id mismatch: "
                    f"ref.task_id='{input_ref.task_id}' != task_id='{task_id}'."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=(
                        f"TaskArtifactRef task_id mismatch: '{input_ref.task_id}' != '{task_id}'."
                    ),
                    is_retryable=False,
                )

            if input_ref.stage != Stage.EVIDENCE:
                logger.error(
                    f"[KnowledgePlanStageExecutor] Input artifact ref stage mismatch: "
                    f"expected EVIDENCE, got {input_ref.stage}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Input artifact stage mismatch: expected EVIDENCE, got {input_ref.stage}.",
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.EVIDENCE_SNAPSHOT:
                logger.error(
                    f"[KnowledgePlanStageExecutor] Input artifact ref type mismatch: "
                    f"expected EVIDENCE_SNAPSHOT, got {input_ref.artifact_type}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=(
                        f"Input artifact type mismatch: expected EVIDENCE_SNAPSHOT, "
                        f"got {input_ref.artifact_type}."
                    ),
                    is_retryable=False,
                )

            # Load EvidenceSnapshot
            snapshot: EvidenceSnapshot | None = ev_repo.get_evidence_snapshot(input_ref.artifact_id)
            if snapshot is None:
                logger.error(
                    f"[KnowledgePlanStageExecutor] EvidenceSnapshot '{input_ref.artifact_id}' "
                    f"not found for task '{task_id}'."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"EvidenceSnapshot '{input_ref.artifact_id}' not found.",
                    is_retryable=False,
                )

            if snapshot.task_id != task_id:
                logger.error(
                    f"[KnowledgePlanStageExecutor] EvidenceSnapshot task_id mismatch: "
                    f"snapshot.task_id='{snapshot.task_id}' != task_id='{task_id}'."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"EvidenceSnapshot task_id mismatch: '{snapshot.task_id}' != '{task_id}'.",
                    is_retryable=False,
                )

            if not snapshot.evidence_ids:
                logger.warning(
                    f"[KnowledgePlanStageExecutor] EvidenceSnapshot '{snapshot.evidence_snapshot_id}' "
                    f"has 0 evidence_ids -> NEEDS_EVIDENCE."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message="EvidenceSnapshot contains 0 evidence items.",
                    is_retryable=False,
                    metadata_json={"reason": "EMPTY_EVIDENCE_SNAPSHOT"},
                )

            # Resolve all referenced EvidenceItems
            evidence_items: list[EvidenceItem] = []
            for eid in snapshot.evidence_ids:
                item = ev_repo.get_evidence_item(eid)
                if item is None:
                    logger.error(
                        f"[KnowledgePlanStageExecutor] Referenced EvidenceItem '{eid}' "
                        f"missing in repository for snapshot '{snapshot.evidence_snapshot_id}'."
                    )
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.FATAL.value,
                        error_message=f"Referenced EvidenceItem '{eid}' not found in repository.",
                        is_retryable=False,
                    )
                evidence_items.append(item)

            # Resolve global_retrieval_snapshot_id
            global_retrieval_snapshot_id = input_ref.metadata_json.get("retrieval_snapshot_id")
            if not global_retrieval_snapshot_id:
                ret_snaps = ev_repo.list_retrieval_snapshots_for_task(task_id)
                if ret_snaps:
                    global_retrieval_snapshot_id = ret_snaps[0].retrieval_snapshot_id

            # Resolve previous plan revision if replanning
            prev_plan_ref = art_repo.get_latest_artifact_ref(
                task_id=task_id,
                stage=Stage.KNOWLEDGE_PLAN,
                artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
            )
            previous_content_plan_revision_id = (
                prev_plan_ref.artifact_id if prev_plan_ref else None
            )

        # ---------------------------------------------------------------------
        # 2. Map EvidenceSnapshot -> PlannerInput
        # ---------------------------------------------------------------------
        available_evidence_ids = tuple(item.evidence_id for item in evidence_items)
        knowledge_context = tuple(
            f"[{item.evidence_id}] {item.normalized_fact or item.original_excerpt}"
            for item in evidence_items
        )

        title = task.task_metadata.get("title") or task.topic
        user_instruction = (
            task.task_metadata.get("user_instruction")
            or task.task_metadata.get("instruction")
        )
        target_video_duration = float(task.target_duration or 60.0)

        planner_input = PlannerInput(
            topic=task.topic,
            target_video_duration=target_video_duration,
            title=title,
            user_instruction=user_instruction,
            knowledge_context=knowledge_context,
            available_evidence_ids=available_evidence_ids,
            previous_content_plan_revision_id=previous_content_plan_revision_id,
            source_grounded=True,
            global_retrieval_snapshot_id=global_retrieval_snapshot_id,
        )

        # ---------------------------------------------------------------------
        # 3. Execute Existing ContentPlanner
        # ---------------------------------------------------------------------
        logger.info(
            f"[KnowledgePlanStageExecutor] Executing ContentPlanner for task '{task_id}', "
            f"topic='{task.topic}', evidence_count={len(available_evidence_ids)}."
        )

        plan_revision: ContentPlanRevision
        with get_session(self._session_factory) as session:
            plan_repo = ContentPlanRepository(session)
            planner = ContentPlanner(
                repository=plan_repo,
                llm_caller=self.llm_caller,
                max_retries=self.max_retries,
                duration_tolerance_ratio=self.duration_tolerance_ratio,
            )

            try:
                plan_revision = planner.plan(
                    input_data=planner_input,
                    repository=plan_repo,
                )
            except InsufficientEvidenceError as exc:
                logger.warning(
                    f"[KnowledgePlanStageExecutor] ContentPlanner raised InsufficientEvidenceError: {exc}"
                )
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.KNOWLEDGE_PLAN_NEEDS_EVIDENCE,
                    attributes={
                        "task_id": task_id,
                        "job_id": job.job_id,
                        "reason": str(exc),
                        "available_evidence_count": len(available_evidence_ids),
                    },
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message=str(exc),
                    is_retryable=False,
                    metadata_json={"reason": "INSUFFICIENT_EVIDENCE"},
                )
            except (
                PlannerOutputSchemaError,
                PlannerProviderError,
                InvalidDurationPlanError,
                PlannerOutputInvalidError,
                InvalidBeatLineageInheritanceError,
            ) as exc:
                logger.warning(
                    f"[KnowledgePlanStageExecutor] ContentPlanner output validation error: {exc}"
                )
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.KNOWLEDGE_PLAN_VALIDATION_FAILED,
                    attributes={
                        "task_id": task_id,
                        "job_id": job.job_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=str(exc),
                    is_retryable=True,
                    metadata_json={"reason": type(exc).__name__},
                )
            except Exception as exc:
                logger.exception(
                    f"[KnowledgePlanStageExecutor] Unexpected error during planning: {exc}"
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"ContentPlanner execution failed: {exc}",
                    is_retryable=False,
                )

        # ---------------------------------------------------------------------
        # 4. Deterministic Post-Validation
        # ---------------------------------------------------------------------
        if not plan_revision.beats:
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.RETRYABLE.value,
                error_message="Plan contains 0 beats.",
                is_retryable=True,
            )

        allowed_evidence_set = set(available_evidence_ids)

        # Validate each beat against evidence rules
        for beat in plan_revision.beats:
            # 1. Factual KNOWLEDGE beats must have evidence
            if beat.beat_type == BeatType.KNOWLEDGE:
                if not beat.evidence_refs:
                    logger.warning(
                        f"[KnowledgePlanStageExecutor] Beat '{beat.beat_id}' (KNOWLEDGE) "
                        "lacks evidence references -> NEEDS_EVIDENCE."
                    )
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.NEEDS_EVIDENCE.value,
                        error_message=(
                            f"KNOWLEDGE beat (order {beat.order}, id '{beat.beat_id}') "
                            "requires valid evidence references."
                        ),
                        is_retryable=False,
                        metadata_json={"unsupported_beat_id": beat.beat_id},
                    )

            # 2. All cited evidence refs must resolve to the EvidenceSnapshot
            for ref in beat.evidence_refs:
                if ref not in allowed_evidence_set:
                    logger.warning(
                        f"[KnowledgePlanStageExecutor] Beat '{beat.beat_id}' cites ungrounded "
                        f"evidence ref '{ref}' -> NEEDS_EVIDENCE."
                    )
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.NEEDS_EVIDENCE.value,
                        error_message=(
                            f"Evidence ref '{ref}' in beat {beat.order} is not grounded in "
                            f"the current EvidenceSnapshot."
                        ),
                        is_retryable=False,
                        metadata_json={"invalid_evidence_ref": ref, "beat_id": beat.beat_id},
                    )

        # ---------------------------------------------------------------------
        # 5. Persist Output Plan and TaskArtifactRef
        # ---------------------------------------------------------------------
        with get_session(self._session_factory) as session:
            plan_repo = ContentPlanRepository(session)
            art_repo = TaskArtifactRepository(session)

            # Ensure plan revision is persisted in DB
            existing_plan = plan_repo.get_revision(plan_revision.content_plan_revision_id)
            if existing_plan is None:
                plan_repo.add_revision(plan_revision)

            output_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.KNOWLEDGE_PLAN,
                artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
                artifact_id=plan_revision.content_plan_revision_id,
                artifact_version=str(plan_revision.revision_number),
                metadata_json={
                    "beat_count": len(plan_revision.beats),
                    "overall_target_duration": plan_revision.overall_target_duration,
                    "global_retrieval_snapshot_id": plan_revision.global_retrieval_snapshot_id,
                    "evidence_snapshot_id": snapshot.evidence_snapshot_id,
                    "evidence_ref_count": sum(len(b.evidence_refs) for b in plan_revision.beats),
                },
            )
            session.commit()

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.KNOWLEDGE_PLAN_COMPLETED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "content_plan_revision_id": plan_revision.content_plan_revision_id,
                "revision_number": plan_revision.revision_number,
                "beat_count": len(plan_revision.beats),
                "evidence_ref_count": sum(len(b.evidence_refs) for b in plan_revision.beats),
                "global_retrieval_snapshot_id": plan_revision.global_retrieval_snapshot_id,
            },
        )

        logger.info(
            f"[KnowledgePlanStageExecutor] KNOWLEDGE_PLAN succeeded for task '{task_id}'. "
            f"PlanRevision: '{plan_revision.content_plan_revision_id}' (rev {plan_revision.revision_number}), "
            f"ArtifactRef: '{output_artifact_ref.task_artifact_ref_id}'."
        )

        return StageExecutionResult(
            success=True,
            output_artifact_ref=output_artifact_ref,
            output_task_artifact_ref_id=output_artifact_ref.task_artifact_ref_id,
            output_artifact_revision_id=plan_revision.content_plan_revision_id,
            metadata_json={
                "content_plan_revision_id": plan_revision.content_plan_revision_id,
                "revision_number": plan_revision.revision_number,
                "beat_count": len(plan_revision.beats),
                "evidence_snapshot_id": snapshot.evidence_snapshot_id,
                "global_retrieval_snapshot_id": plan_revision.global_retrieval_snapshot_id,
            },
        )

    def _emit_trace(
        self,
        task_id: str,
        event_type: TraceEventType,
        attributes: dict[str, Any],
    ) -> None:
        """Emits structured domain trace event if trace_writer is configured."""
        if self.trace_writer is None:
            return
        try:
            self.trace_writer.write_event(
                task_id=task_id,
                event_type=event_type,
                attributes=attributes,
            )
        except Exception as exc:
            logger.warning(f"[KnowledgePlanStageExecutor] Failed to emit trace event {event_type}: {exc}")
