"""
Quality Review Stage Executor.

Implements StageExecutorProtocol for Stage.QUALITY_REVIEW.
Orchestrates existing Phase 1–7 evaluation and deterministic quality remediation:
1. Resolves and validates input CompositionOutput, StoryboardSnapshot, ExecutionRun,
   EvidenceSnapshot, and AudioOutput from task lineage.
2. Evaluates storyboard shots using existing ShotAssetEvaluationService and
   dimension evaluators with frozen task evidence (zero web-search calls).
3. Applies deterministic EvaluationPolicyEngine (critical gates: semantic alignment,
   knowledge accuracy cannot be averaged away).
4. For PASS: produces immutable EvaluationSnapshot TaskArtifactRef.
5. For FAIL: evaluates deterministic QualityRemediationPolicyEngine to determine bounded
   action (RETRY_EVALUATION, REGENERATE_SAME_ROUTE, FALLBACK_NEXT_ROUTE_CANDIDATE,
   CONTROLLED_VISUAL_REPLAN, NEEDS_USER_ACTION).
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.enums import StoryboardSnapshotState
from app.domain.evaluation import (
    DimensionEvaluationResult,
    EvaluationContextRef,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationSnapshot,
    EvaluationTarget,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy, EvaluationPolicyEngine
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationDecision,
    QualityRemediationPolicy,
)
from app.domain.quality_remediation_policy import QualityRemediationPolicyEngine
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.asset_execution import ExecutionStatus
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    Stage,
)
from app.persistence.models import EvaluationTargetORM
from app.persistence.repositories import (
    AudioOutputRepository,
    CompositionOutputRepository,
    EvaluationRepository,
    ExecutionRepository,
    KnowledgeVideoTaskRepository,
    ShotRepository,
    StoryboardRepository,
    TaskArtifactRepository,
)
from app.persistence.session import get_session
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluatorAdapter,
)
from app.services.evaluation.shot_asset_evaluation_service import (
    ShotAssetEvaluationService,
)


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


class QualityReviewStageExecutor(StageExecutorProtocol):
    """
    Executes the QUALITY_REVIEW stage for a KnowledgeVideoTask.
    Evaluates media artifacts against frozen evidence using deterministic policy.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        evaluation_policy: EvaluationPolicy | None = None,
        remediation_policy: QualityRemediationPolicy | None = None,
        evaluator_adapter: MultimodalEvaluatorAdapter | None = None,
        trace_writer: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._evaluation_policy = evaluation_policy or EvaluationPolicy()
        self._remediation_policy = remediation_policy or QualityRemediationPolicy()
        self._evaluator_adapter = evaluator_adapter
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
                f"[QualityReviewStageExecutor] Failed to emit trace event {event_type}: {exc}"
            )

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[QualityReviewStageExecutor] Starting evaluation for task '{task_id}', "
            f"job '{job.job_id}', attempt {job.attempt_number}."
        )

        self._emit_trace(
            task_id=task_id,
            event_type=TraceEventType.QUALITY_REVIEW_STARTED,
            attributes={
                "task_id": task_id,
                "job_id": job.job_id,
                "attempt_number": job.attempt_number,
            },
        )

        with _managed_session(self._session_factory) as session:
            art_repo = TaskArtifactRepository(session)
            comp_repo = CompositionOutputRepository(session)
            sb_repo = StoryboardRepository(session)
            exec_repo = ExecutionRepository(session)
            shot_repo = ShotRepository(session)
            eval_repo = EvaluationRepository(session)
            audio_repo = AudioOutputRepository(session)

            # -----------------------------------------------------------------
            # 1. Resolve & Validate CompositionOutput Input
            # -----------------------------------------------------------------
            comp_output_id = None
            if job.input_task_artifact_ref_id:
                ref = art_repo.get_artifact_ref(job.input_task_artifact_ref_id)
                if ref is not None:
                    if ref.task_id != task_id:
                        err = f"Task ID mismatch: job task '{task_id}' != input artifact task '{ref.task_id}'."
                        logger.error(f"[QualityReviewStageExecutor] {err}")
                        return StageExecutionResult(
                            success=False,
                            error_type=JobErrorType.FATAL.value,
                            error_message=err,
                            is_retryable=False,
                        )
                    if ref.is_stale:
                        err = f"Input composition artifact ref '{ref.task_artifact_ref_id}' is marked stale."
                        logger.error(f"[QualityReviewStageExecutor] {err}")
                        return StageExecutionResult(
                            success=False,
                            error_type=JobErrorType.FATAL.value,
                            error_message=err,
                            is_retryable=False,
                        )
                    comp_output_id = ref.artifact_id

            if comp_output_id is None:
                latest_comp_ref = art_repo.get_current_artifact_ref(
                    task_id=task_id,
                    stage=Stage.COMPOSITION,
                    artifact_type=ArtifactType.COMPOSITION_OUTPUT,
                )
                if latest_comp_ref is not None:
                    comp_output_id = latest_comp_ref.artifact_id

            if comp_output_id is None:
                latest_comp = comp_repo.get_latest_composition_output_for_task(task_id)
                if latest_comp is not None:
                    comp_output_id = latest_comp.composition_output_id

            if comp_output_id is None:
                err = f"CompositionOutput missing for task '{task_id}', job '{job.job_id}'."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            composition_output = comp_repo.get_composition_output(comp_output_id)
            if composition_output is None:
                err = f"CompositionOutput '{comp_output_id}' not found in repository."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if composition_output.task_id != task_id:
                err = (
                    f"Lineage mismatch: CompositionOutput '{comp_output_id}' belongs to "
                    f"task '{composition_output.task_id}', expected '{task_id}'."
                )
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 2. Resolve Lineage: Storyboard, ExecutionRun, Evidence, Audio
            # -----------------------------------------------------------------
            sb_snapshot = sb_repo.get_snapshot(composition_output.storyboard_snapshot_id)
            if sb_snapshot is None:
                err = f"StoryboardSnapshot '{composition_output.storyboard_snapshot_id}' not found."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if sb_snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
                err = f"StoryboardSnapshot '{sb_snapshot.storyboard_snapshot_id}' is not approved (state={sb_snapshot.snapshot_state})."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            exec_run = exec_repo.get_execution_run(composition_output.execution_run_id)
            if exec_run is None:
                err = f"ExecutionRun '{composition_output.execution_run_id}' not found."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            if exec_run.status != ExecutionStatus.COMPLETED:
                err = f"ExecutionRun '{exec_run.execution_run_id}' status is '{exec_run.status}', expected COMPLETED."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            audio_output = audio_repo.get_audio_output(composition_output.audio_output_id)
            if audio_output is None:
                err = f"AudioOutput '{composition_output.audio_output_id}' not found."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            evidence_ref = art_repo.get_current_artifact_ref(
                task_id=task_id,
                stage=Stage.EVIDENCE,
                artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            ) or art_repo.get_latest_artifact_ref(
                task_id=task_id,
                stage=Stage.EVIDENCE,
                artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            )
            if evidence_ref is None or evidence_ref.task_id != task_id:
                err = f"EvidenceSnapshot artifact missing or mismatched for task '{task_id}'."
                logger.error(f"[QualityReviewStageExecutor] {err}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=err,
                    is_retryable=False,
                )

            # -----------------------------------------------------------------
            # 3. Evaluate Storyboard Shots via ShotAssetEvaluationService
            # -----------------------------------------------------------------
            eval_service = ShotAssetEvaluationService(session)
            shot_executions = exec_repo.list_shot_executions_for_run(exec_run.execution_run_id)
            shot_exec_by_shot_id = {se.shot_id: se for se in shot_executions}

            evaluated_shots_info = []
            failing_shots_info = []

            for shot_rev_id in sb_snapshot.shot_revision_ids:
                shot_rev = shot_repo.get_revision(shot_rev_id)
                if shot_rev is None:
                    err = f"ShotRevision '{shot_rev_id}' not found in repository."
                    logger.error(f"[QualityReviewStageExecutor] {err}")
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.FATAL.value,
                        error_message=err,
                        is_retryable=False,
                    )

                se = shot_exec_by_shot_id.get(shot_rev.shot_id)
                if se is None or not se.produced_asset_version_id:
                    err = f"No produced asset version for shot '{shot_rev.shot_id}' in run '{exec_run.execution_run_id}'."
                    logger.error(f"[QualityReviewStageExecutor] {err}")
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.FATAL.value,
                        error_message=err,
                        is_retryable=False,
                    )

                asset_ver = exec_repo.get_shot_asset_version(se.produced_asset_version_id)
                if asset_ver is None:
                    err = f"ShotAssetVersion '{se.produced_asset_version_id}' not found in repository."
                    logger.error(f"[QualityReviewStageExecutor] {err}")
                    return StageExecutionResult(
                        success=False,
                        error_type=JobErrorType.FATAL.value,
                        error_message=err,
                        is_retryable=False,
                    )

                # Ensure EvaluationTarget exists
                stmt = select(EvaluationTargetORM).where(
                    EvaluationTargetORM.shot_id == shot_rev.shot_id,
                    EvaluationTargetORM.shot_revision_id == shot_rev.shot_revision_id,
                    EvaluationTargetORM.shot_asset_version_id == asset_ver.shot_asset_version_id,
                )
                existing_target_orm = session.scalars(stmt).first()
                target_id = None
                if existing_target_orm is not None:
                    target_id = existing_target_orm.evaluation_target_id
                else:
                    target = create_evaluation_target_from_shot(
                        shot_revision=shot_rev,
                        shot_asset_version=asset_ver,
                        context_refs=(
                            EvaluationContextRef(
                                type="COMPOSITION_OUTPUT",
                                reference_id=composition_output.composition_output_id,
                            ),
                        ),
                    )
                    eval_repo.save_target(target)
                    session.commit()
                    target_id = target.evaluation_target_id

                # Perform evaluation
                shot_snapshot = eval_service.evaluate_target(
                    evaluation_target_id=target_id,
                    policy=self._evaluation_policy,
                    evaluator_adapter=self._evaluator_adapter,
                )
                dim_results = eval_repo.get_snapshot_dimension_results(shot_snapshot.evaluation_snapshot_id)

                shot_info = {
                    "shot_id": shot_rev.shot_id,
                    "shot_revision": shot_rev,
                    "asset_version": asset_ver,
                    "target_id": target_id,
                    "snapshot": shot_snapshot,
                    "dimension_results": dim_results,
                    "shot_execution": se,
                }
                evaluated_shots_info.append(shot_info)

                if shot_snapshot.decision != EvaluationDecision.PASS:
                    failing_shots_info.append(shot_info)

            # -----------------------------------------------------------------
            # 4. Aggregation & Decision Policy
            # -----------------------------------------------------------------
            if not failing_shots_info:
                # All evaluated shots passed!
                primary_snap = evaluated_shots_info[0]["snapshot"] if evaluated_shots_info else None
                snap_id = primary_snap.evaluation_snapshot_id if primary_snap else "none"

                artifact_ref = TaskArtifactRef.create(
                    task_id=task_id,
                    stage=Stage.QUALITY_REVIEW,
                    artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
                    artifact_id=snap_id,
                    metadata_json={
                        "decision": "PASS",
                        "composition_output_id": composition_output.composition_output_id,
                        "overall_score": primary_snap.overall_score if primary_snap else 1.0,
                        "evaluated_shot_count": len(evaluated_shots_info),
                        "is_current": True,
                    },
                )
                art_repo.save_artifact_ref(artifact_ref)
                session.commit()

                self._emit_trace(
                    task_id=task_id,
                    event_type=TraceEventType.QUALITY_ACCEPTED,
                    attributes={
                        "task_id": task_id,
                        "evaluation_snapshot_id": snap_id,
                        "composition_output_id": composition_output.composition_output_id,
                        "decision": "PASS",
                    },
                )

                logger.info(
                    f"[QualityReviewStageExecutor] Quality review PASSED for task '{task_id}', "
                    f"snapshot '{snap_id}' accepted."
                )

                return StageExecutionResult(
                    success=True,
                    output_artifact_ref=artifact_ref,
                    output_task_artifact_ref_id=artifact_ref.task_artifact_ref_id,
                    output_artifact_revision_id=snap_id,
                    metadata_json={
                        "decision": "PASS",
                        "evaluation_snapshot_id": snap_id,
                        "composition_output_id": composition_output.composition_output_id,
                        "evaluated_shot_count": len(evaluated_shots_info),
                    },
                )

            # -----------------------------------------------------------------
            # 5. Quality Failure & Remediation Decision
            # -----------------------------------------------------------------
            failing_shot = failing_shots_info[0]
            failing_snap = failing_shot["snapshot"]
            failing_se = failing_shot["shot_execution"]

            past_remediations = eval_repo.list_remediation_decisions_for_shot(failing_shot["shot_id"])

            remed_decision = QualityRemediationPolicyEngine.evaluate(
                snapshot=failing_snap,
                dimension_results=failing_shot["dimension_results"],
                route_decision=failing_se.route_decision,
                history=past_remediations,
                policy=self._remediation_policy,
                shot_id=failing_shot["shot_id"],
                shot_revision_id=failing_shot["shot_revision"].shot_revision_id,
                shot_asset_version_id=failing_shot["asset_version"].shot_asset_version_id,
            )
            eval_repo.save_remediation_decision(remed_decision)
            session.commit()

            # Record historical failed TaskArtifactRef
            failed_artifact_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.QUALITY_REVIEW,
                artifact_type=ArtifactType.EVALUATION_SNAPSHOT,
                artifact_id=failing_snap.evaluation_snapshot_id,
                metadata_json={
                    "decision": "FAIL",
                    "composition_output_id": composition_output.composition_output_id,
                    "remediation_action": remed_decision.action.value,
                    "remediation_decision_id": remed_decision.remediation_decision_id,
                    "affected_shot_ids": [f["shot_id"] for f in failing_shots_info],
                    "selected_route_candidate": remed_decision.selected_route_candidate.model_dump()
                    if remed_decision.selected_route_candidate
                    else None,
                    "reason_codes": list(remed_decision.reason_codes),
                    "is_current": False,
                    "is_stale": True,
                },
            )
            art_repo.save_artifact_ref(failed_artifact_ref)
            session.commit()

            self._emit_trace(
                task_id=task_id,
                event_type=TraceEventType.QUALITY_REMEDIATION_REQUESTED,
                attributes={
                    "task_id": task_id,
                    "evaluation_snapshot_id": failing_snap.evaluation_snapshot_id,
                    "remediation_action": remed_decision.action.value,
                    "affected_shot_ids": [f["shot_id"] for f in failing_shots_info],
                    "remediation_decision_id": remed_decision.remediation_decision_id,
                },
            )

            logger.warning(
                f"[QualityReviewStageExecutor] Quality review FAILED for task '{task_id}', "
                f"remediation action decided: '{remed_decision.action.value}' for shots "
                f"{[f['shot_id'] for f in failing_shots_info]}."
            )

            return StageExecutionResult(
                success=True,
                output_artifact_ref=failed_artifact_ref,
                output_task_artifact_ref_id=failed_artifact_ref.task_artifact_ref_id,
                output_artifact_revision_id=failing_snap.evaluation_snapshot_id,
                metadata_json={
                    "decision": "FAIL",
                    "evaluation_snapshot_id": failing_snap.evaluation_snapshot_id,
                    "composition_output_id": composition_output.composition_output_id,
                    "remediation_action": remed_decision.action.value,
                    "remediation_decision_id": remed_decision.remediation_decision_id,
                    "affected_shot_ids": [f["shot_id"] for f in failing_shots_info],
                    "selected_route_candidate": remed_decision.selected_route_candidate.model_dump()
                    if remed_decision.selected_route_candidate
                    else None,
                },
            )
