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
from app.domain.enums import StoryboardSnapshotState
from app.domain.evidence import EvidenceItem, EvidenceSnapshot
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision
from app.domain.storyboard_agent import StoryboardAgent
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobErrorType, Stage
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    ScriptRepository,
    ShotRepository,
    StoryboardRepository,
    TaskArtifactRepository,
)
from app.services.storyboard.storyboard_adapter import (
    StoryboardAdapter,
    StoryboardAdapterError,
    StoryboardDurationMismatchError,
    StoryboardEvidenceScopeError,
    StoryboardScriptMappingError,
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


class StoryboardStageExecutor(StageExecutorProtocol):
    """
    Executes the STORYBOARD workflow stage for a KnowledgeVideoTask.

    Workflow contract:
    1. Resolves and validates the input TaskArtifactRef (SCRIPT_REVISION).
    2. Loads ScriptRevision, ContentPlanRevision, and grounding EvidenceSnapshot.
    3. Adapts the structured ScriptRevision to the existing StoryboardAgent without
       regenerating ContentPlan or rewriting authoritative narration.
    4. Deterministically validates shot existence, ordering, non-empty narration,
       evidence subset containment (Shot.evidence_refs ⊆ Segment.evidence_refs),
       and aggregate duration matching.
    5. Persists Shot entities, initial immutable ShotRevisions, and DRAFT StoryboardSnapshot
       using existing repositories (ShotRepository, StoryboardRepository).
    6. Returns StageExecutionResult with TaskArtifactRef(STORYBOARD_SNAPSHOT).
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
            logger.warning(
                f"[StoryboardStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[StoryboardStageExecutor] Starting execution for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.STORYBOARD_STAGE_STARTED,
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
            script_repo = ScriptRepository(session)
            ev_repo = EvidenceRepository(session)

            input_ref: TaskArtifactRef | None = None
            if input_ref_id:
                input_ref = art_repo.get_artifact_ref(input_ref_id)
            else:
                input_ref = art_repo.get_latest_artifact_ref(
                    task_id=task_id,
                    stage=Stage.SCRIPT,
                    artifact_type=ArtifactType.SCRIPT_REVISION,
                )

            if input_ref is None:
                logger.error(
                    f"[StoryboardStageExecutor] Input artifact ref missing for task '{task_id}', "
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
                    f"[StoryboardStageExecutor] Input artifact ref task_id mismatch: "
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

            if input_ref.stage != Stage.SCRIPT:
                logger.error(
                    f"[StoryboardStageExecutor] Input artifact ref stage mismatch: "
                    f"expected SCRIPT, got {input_ref.stage}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Input artifact stage mismatch: expected SCRIPT, got {input_ref.stage}.",
                    is_retryable=False,
                )

            if input_ref.artifact_type != ArtifactType.SCRIPT_REVISION:
                logger.error(
                    f"[StoryboardStageExecutor] Input artifact ref type mismatch: "
                    f"expected SCRIPT_REVISION, got {input_ref.artifact_type}."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=(
                        f"Input artifact type mismatch: expected SCRIPT_REVISION, "
                        f"got {input_ref.artifact_type}."
                    ),
                    is_retryable=False,
                )

            # Load ScriptRevision
            script_revision: ScriptRevision | None = script_repo.get_revision(
                input_ref.artifact_id
            )
            if script_revision is None:
                logger.error(
                    f"[StoryboardStageExecutor] ScriptRevision '{input_ref.artifact_id}' not found."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"ScriptRevision '{input_ref.artifact_id}' not found.",
                    is_retryable=False,
                )

            # Load ContentPlanRevision
            plan_revision: ContentPlanRevision | None = plan_repo.get_revision(
                script_revision.content_plan_revision_id
            )
            if plan_revision is None:
                logger.error(
                    f"[StoryboardStageExecutor] ContentPlanRevision '{script_revision.content_plan_revision_id}' not found."
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"ContentPlanRevision '{script_revision.content_plan_revision_id}' not found.",
                    is_retryable=False,
                )

            # Load latest EvidenceSnapshot
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

        # ---------------------------------------------------------------------
        # 2. Execute Storyboard Generation via StoryboardAdapter & StoryboardAgent
        # ---------------------------------------------------------------------
        with get_session(self._session_factory) as session:
            shot_repo = ShotRepository(session)
            storyboard_repo = StoryboardRepository(session)

            agent = StoryboardAgent(
                shot_repository=shot_repo,
                storyboard_repository=storyboard_repo,
                llm_caller=self.llm_caller,
                duration_tolerance_ratio=self.duration_tolerance_ratio,
            )
            adapter = StoryboardAdapter(
                storyboard_agent=agent,
                duration_tolerance_ratio=self.duration_tolerance_ratio,
            )

            user_instruction = (
                task.task_metadata.get("user_instruction")
                or task.task_metadata.get("instruction")
            )

            try:
                adapter_result = adapter.build_storyboard_from_script(
                    plan=plan_revision,
                    script=script_revision,
                    video_title=task.topic or plan_revision.topic,
                    target_aspect_ratio=task.aspect_ratio,
                    user_instruction=user_instruction,
                    language=task.language or script_revision.language,
                    shot_repository=shot_repo,
                    storyboard_repository=storyboard_repo,
                )
            except StoryboardEvidenceScopeError as exc:
                logger.warning(
                    f"[StoryboardStageExecutor] Evidence scope error during storyboard planning: {exc}"
                )
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.STORYBOARD_VALIDATION_FAILED,
                    attributes={
                        "task_id": task_id,
                        "job_id": job.job_id,
                        "error_type": "StoryboardEvidenceScopeError",
                        "error_message": str(exc),
                    },
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message=str(exc),
                    is_retryable=False,
                    metadata_json={"reason": "NEEDS_EVIDENCE"},
                )
            except (
                StoryboardDurationMismatchError,
                StoryboardScriptMappingError,
                StoryboardAdapterError,
            ) as exc:
                logger.warning(
                    f"[StoryboardStageExecutor] Storyboard adapter validation error: {exc}"
                )
                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.STORYBOARD_VALIDATION_FAILED,
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
                    f"[StoryboardStageExecutor] Unexpected error during storyboard generation: {exc}"
                )
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=f"Storyboard generation failed: {exc}",
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 3. Deterministic Validation of Generated Storyboard
            # -----------------------------------------------------------------
            draft_snapshot = adapter_result.snapshot
            if draft_snapshot.snapshot_state != StoryboardSnapshotState.DRAFT:
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=(
                        f"Expected DRAFT storyboard snapshot, got {draft_snapshot.snapshot_state.value}."
                    ),
                    is_retryable=False,
                )

            if not adapter_result.shots or not adapter_result.shot_revisions:
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message="Storyboard contains zero shots.",
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 4. Create TaskArtifactRef
            # -----------------------------------------------------------------
            total_duration = sum(r.target_duration for r in adapter_result.shot_revisions)
            output_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.STORYBOARD,
                artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
                artifact_id=draft_snapshot.storyboard_snapshot_id,
                artifact_version="1",
                metadata_json={
                    "content_plan_revision_id": plan_revision.content_plan_revision_id,
                    "script_revision_id": script_revision.script_revision_id,
                    "shot_count": len(adapter_result.shots),
                    "snapshot_state": draft_snapshot.snapshot_state.value,
                    "total_duration": total_duration,
                },
            )

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.STORYBOARD_CREATED,
                attributes={
                    "task_id": task_id,
                    "job_id": job.job_id,
                    "storyboard_snapshot_id": draft_snapshot.storyboard_snapshot_id,
                    "shot_count": len(adapter_result.shots),
                    "total_duration": total_duration,
                    "state": draft_snapshot.snapshot_state.value,
                },
            )

            logger.info(
                f"[StoryboardStageExecutor] Successfully generated StoryboardSnapshot '{draft_snapshot.storyboard_snapshot_id}' "
                f"with {len(adapter_result.shots)} shots for task '{task_id}'."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_ref=output_artifact_ref,
                output_task_artifact_ref_id=output_artifact_ref.task_artifact_ref_id,
                metadata_json={
                    "storyboard_snapshot_id": draft_snapshot.storyboard_snapshot_id,
                    "script_revision_id": script_revision.script_revision_id,
                    "content_plan_revision_id": plan_revision.content_plan_revision_id,
                    "shot_count": len(adapter_result.shots),
                    "total_duration": total_duration,
                    "state": draft_snapshot.snapshot_state.value,
                },
            )
