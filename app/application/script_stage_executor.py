from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.content_plan import ContentPlanRevision
from app.domain.evidence import EvidenceItem, EvidenceSnapshot
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobErrorType, Stage
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    ScriptRepository,
    TaskArtifactRepository,
)
from app.services.script.script_generator import (
    ScriptGenerator,
    ScriptNeedsEvidenceError,
    ScriptOutputInvalidError,
    ScriptOutputSchemaError,
)
from app.services.trace_service import TraceWriter


@contextlib.contextmanager
def get_session(
    session_factory: Callable[[], Session] | None,
) -> Generator[Session, None, None]:
    """Provides a managed database session context."""
    if session_factory is None:
        from app.database import get_db

        db_gen = get_db()
        session = next(db_gen)
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            with contextlib.suppress(Exception):
                next(db_gen)
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


class ScriptStageExecutor(StageExecutorProtocol):
    """
    Executes the SCRIPT workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (CONTENT_PLAN_REVISION).
    2. Loads ContentPlanRevision and the task's grounding EvidenceSnapshot.
    3. Prompts the project LLM abstraction via ScriptGenerator.
    4. Enforces 1:1 beat-to-segment mapping, beat order preservation, and strict
       evidence subset containment (Segment.evidence_refs ⊆ Beat.evidence_refs).
    5. Rejects ungrounded factual additions and scope expansion -> NEEDS_EVIDENCE.
    6. Persists immutable ScriptRevision and ScriptSegments in ScriptRepository.
    7. Returns StageExecutionResult with output TaskArtifactRef(SCRIPT_REVISION).
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        llm_caller: Callable[[str], str] | None = None,
        duration_tolerance_ratio: float = 0.25,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.llm_caller = llm_caller
        self.duration_tolerance_ratio = duration_tolerance_ratio
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
            logger.warning(f"[ScriptStageExecutor] Failed to emit trace event {event_type}: {exc}")

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[ScriptStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.SCRIPT_STAGE_STARTED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "attempt_number": job.attempt_number,
            },
        )

        # ---------------------------------------------------------------------
        # 1. Input Resolution & Lineage Invariant Verification
        # ---------------------------------------------------------------------
        input_ref_id = job.input_task_artifact_ref_id

        with get_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            plan_repo = ContentPlanRepository(session)
            ev_repo = EvidenceRepository(session)
            script_repo = ScriptRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.KNOWLEDGE_PLAN,
                    artifact_type=ArtifactType.CONTENT_PLAN_REVISION,
                )

            if input_ref is None:
                logger.error(
                    f"[ScriptStageExecutor] Input artifact ref missing for task '{task_id}', "
                    f"job '{job.job_id}'."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Input artifact reference not found for task '{task_id}'.",
                    is_retryable=False,
                )

            if input_ref.task_id != task_id:
                logger.error(
                    f"[ScriptStageExecutor] Input artifact ref task_id mismatch: "
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

            if input_ref.stage != Stage.KNOWLEDGE_PLAN:
                logger.error(
                    f"[ScriptStageExecutor] Input artifact ref stage mismatch: "
                    f"expected KNOWLEDGE_PLAN, got {input_ref.stage}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Input artifact stage mismatch: expected KNOWLEDGE_PLAN, got {input_ref.stage}.",
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.CONTENT_PLAN_REVISION:
                logger.error(
                    f"[ScriptStageExecutor] Input artifact ref type mismatch: "
                    f"expected CONTENT_PLAN_REVISION, got {input_ref.artifact_type}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=(
                        f"Input artifact type mismatch: expected CONTENT_PLAN_REVISION, "
                        f"got {input_ref.artifact_type}."
                    ),
                    is_retryable=False,
                )

            # Load ContentPlanRevision
            plan_revision: ContentPlanRevision | None = plan_repo.get_revision(input_ref.artifact_id)
            if plan_revision is None:
                logger.error(
                    f"[ScriptStageExecutor] ContentPlanRevision '{input_ref.artifact_id}' not found."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"ContentPlanRevision '{input_ref.artifact_id}' not found.",
                    is_retryable=False,
                )

            # Load latest EvidenceSnapshot for grounding
            ev_ref = art_repo.get_latest_artifact_ref(
                task_id=task_id,
                stage=Stage.EVIDENCE,
                artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            )
            evidence_by_id: dict[str, EvidenceItem] = {}
            if ev_ref:
                snapshot = ev_repo.get_evidence_snapshot(ev_ref.artifact_id)
                if snapshot and snapshot.evidence_ids:
                    for eid in snapshot.evidence_ids:
                        item = ev_repo.get_evidence_item(eid)
                        if item:
                            evidence_by_id[item.evidence_id] = item

            # Determine revision number for script
            latest_script = script_repo.get_latest_revision_for_task(task_id)
            script_revision_number = (
                (latest_script.revision_number + 1) if latest_script else 1
            )

        # ---------------------------------------------------------------------
        # 2. Execute Script Generation
        # ---------------------------------------------------------------------
        generator = ScriptGenerator(
            llm_caller=self.llm_caller,
            duration_tolerance_ratio=self.duration_tolerance_ratio,
        )

        user_instruction = (
            task.task_metadata.get("user_instruction")
            or task.task_metadata.get("instruction")
        )
        target_duration = float(task.target_duration or plan_revision.overall_target_duration)

        try:
            script_revision = generator.generate_script(
                task_id=task_id,
                plan=plan_revision,
                evidence_by_id=evidence_by_id,
                topic=task.topic,
                target_duration=target_duration,
                language="zh",
                user_instruction=user_instruction,
                revision_number=script_revision_number,
            )
        except ScriptNeedsEvidenceError as exc:
            logger.warning(
                f"[ScriptStageExecutor] ScriptGenerator raised ScriptNeedsEvidenceError: {exc}"
            )
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.SCRIPT_STAGE_NEEDS_EVIDENCE,
                attributes={
                    "task_id": task_id,
                    "job_id": job.job_id,
                    "reason": str(exc),
                },
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.NEEDS_EVIDENCE.value,
                error_message=str(exc),
                is_retryable=False,
                metadata_json={"reason": "NEEDS_EVIDENCE"},
            )
        except (ScriptOutputSchemaError, ScriptOutputInvalidError) as exc:
            logger.warning(
                f"[ScriptStageExecutor] ScriptGenerator validation error: {exc}"
            )
            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.SCRIPT_VALIDATION_FAILED,
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
                f"[ScriptStageExecutor] Unexpected error during script generation: {exc}"
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.FATAL.value,
                error_message=f"Script generation failed: {exc}",
                is_retryable=False,
            )

        # ---------------------------------------------------------------------
        # 3. Persist ScriptRevision & Create TaskArtifactRef
        # ---------------------------------------------------------------------
        with get_session(self._session_factory) as session:
            script_repo = ScriptRepository(session)
            saved_revision = script_repo.save_revision(script_revision)

        output_artifact_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.SCRIPT,
            artifact_type=ArtifactType.SCRIPT_REVISION,
            artifact_id=saved_revision.script_revision_id,
            artifact_version=str(saved_revision.revision_number),
            metadata_json={
                "content_plan_revision_id": saved_revision.content_plan_revision_id,
                "content_fingerprint": saved_revision.content_fingerprint,
                "segment_count": len(saved_revision.segments),
                "overall_target_duration": saved_revision.overall_target_duration,
            },
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.SCRIPT_REVISION_CREATED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "script_revision_id": saved_revision.script_revision_id,
                "content_fingerprint": saved_revision.content_fingerprint,
                "segment_count": len(saved_revision.segments),
                "target_duration": saved_revision.overall_target_duration,
            },
        )

        logger.info(
            f"[ScriptStageExecutor] Successfully generated ScriptRevision '{saved_revision.script_revision_id}' "
            f"with {len(saved_revision.segments)} segments for task '{task_id}'."
        )

        return StageExecutionResult(
            success=True,
            output_artifact_ref=output_artifact_ref,
            output_task_artifact_ref_id=output_artifact_ref.task_artifact_ref_id,
            metadata_json={
                "script_revision_id": saved_revision.script_revision_id,
                "segment_count": len(saved_revision.segments),
                "content_fingerprint": saved_revision.content_fingerprint,
            },
        )
