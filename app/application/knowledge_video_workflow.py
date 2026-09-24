from __future__ import annotations

from datetime import UTC, datetime

from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session

from app.application.workflow_policy import WorkflowPolicy
from app.domain.enums import StoryboardSnapshotState
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.quality_remediation import QualityRemediationAction
from app.domain.storyboard_approval import StoryboardApprovalService
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.trace import TraceEventType
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobErrorType,
    Stage,
    TaskStatus,
    WorkflowConflictError,
    get_next_stage,
)
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    KnowledgeVideoTaskRepository,
    ShotRepository,
    StageExecutionRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)


class KnowledgeVideoWorkflow:
    """
    Authoritative workflow orchestration engine for KnowledgeVideoTask.
    Controls stage transitions, review pause points, job generation, and error handling.
    Single engine shared by both AUTO and REVIEW execution policies.
    """

    def __init__(self, session: Session, trace_writer: Any | None = None) -> None:
        self._session = session
        self.trace_writer = trace_writer
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.job_repo = WorkflowJobRepository(session)
        self.exec_repo = StageExecutionRepository(session)
        self.artifact_repo = TaskArtifactRepository(session)

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
                f"[KnowledgeVideoWorkflow] Failed to emit trace event {event_type}: {exc}"
            )

    def on_stage_completed(
        self,
        task: KnowledgeVideoTask,
        completed_job: WorkflowJob,
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        """
        Invoked when a stage worker successfully finishes a job.
        Advances stage and creates next job, or pauses at review checkpoint, or marks task COMPLETED.
        """
        ts = now or datetime.now(UTC)

        # 1. If completed stage was the terminal stage (DELIVERY)
        if completed_job.stage == Stage.DELIVERY:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        # Special handling for QUALITY_REVIEW
        if completed_job.stage == Stage.QUALITY_REVIEW:
            output_ref = None
            if completed_job.output_task_artifact_ref_id:
                output_ref = self.artifact_repo.get_artifact_ref(completed_job.output_task_artifact_ref_id)
            if output_ref is None:
                output_ref = self.artifact_repo.get_latest_artifact_ref(task.task_id, stage=Stage.QUALITY_REVIEW)

            meta = output_ref.metadata_json if output_ref and output_ref.metadata_json else {}
            decision = meta.get("decision", "PASS")
            if decision == "FAIL":
                return self._handle_quality_remediation(task, completed_job, meta, now=ts)
            else:
                if task.task_metadata.get("is_partial_rerun"):
                    self._emit_trace(
                        task_id=task.task_id,
                        event_type=TraceEventType.PARTIAL_RERUN_COMPLETED,
                        attributes={
                            "task_id": task.task_id,
                            "completed_job_id": completed_job.job_id,
                        },
                    )
                    task.task_metadata["is_partial_rerun"] = False
                    self.task_repo.save_task(task)

        # 2. Check workflow policy for review checkpoints
        policy = WorkflowPolicy(policy_type=task.workflow_policy)
        if policy.should_pause_for_review(completed_job.stage):
            task.transition_to(
                TaskStatus.WAITING_USER,
                reason=f"Review required after {completed_job.stage.value}",
                now=ts,
            )
            self.task_repo.save_task(task)
            return None

        # 3. Advance to the next stage
        next_stage = get_next_stage(completed_job.stage)
        output_ref_id = completed_job.output_task_artifact_ref_id

        # If completed stage was ASSET during partial rerun (valid AudioOutput exists), skip AUDIO and proceed directly to COMPOSITION
        if completed_job.stage == Stage.ASSET:
            current_audio_ref = self.artifact_repo.get_current_artifact_ref(
                task_id=task.task_id,
                stage=Stage.AUDIO,
                artifact_type=ArtifactType.AUDIO_OUTPUT,
            )
            if current_audio_ref is not None and not current_audio_ref.is_stale:
                next_stage = Stage.COMPOSITION
                output_ref_id = current_audio_ref.task_artifact_ref_id

        if next_stage is None:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        # If completed stage was STORYBOARD under AUTO, invoke domain Storyboard approval automatically
        if completed_job.stage == Stage.STORYBOARD:
            output_ref_id = self._approve_storyboard_for_task(
                task_id=task.task_id,
                draft_output_ref_id=completed_job.output_task_artifact_ref_id,
                approved_by="workflow_auto",
                now=ts,
            )

        if next_stage == get_next_stage(task.current_stage):
            task.advance_stage(next_stage, now=ts)
        else:
            task.set_stage_for_rerun(next_stage, now=ts)

        if task.task_status != TaskStatus.RUNNING:
            task.transition_to(TaskStatus.RUNNING, now=ts)
        self.task_repo.save_task(task)

        # 4. Generate next persistent WorkflowJob
        seq = 1
        idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"
        while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
            seq += 1
            idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"

        next_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=next_stage,
            idempotency_key=idempotency_key,
            attempt_number=1,
            input_artifact_revision_id=completed_job.output_artifact_revision_id,
            input_task_artifact_ref_id=output_ref_id,
            now=ts,
        )
        return self.job_repo.create_job(next_job)

    def _handle_quality_remediation(
        self,
        task: KnowledgeVideoTask,
        completed_job: WorkflowJob,
        meta: dict[str, Any],
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        """
        Handles deterministic quality remediation based on evaluation results and remediation policy.
        """
        ts = now or datetime.now(UTC)
        action_str = meta.get("remediation_action")
        affected_shot_ids = meta.get("affected_shot_ids", [])
        remed_id = meta.get("remediation_decision_id")
        eval_snap_id = meta.get("evaluation_snapshot_id")
        reason_codes = meta.get("reason_codes", [])

        # Check for exhaustion or needs user action / replan
        if action_str in (
            QualityRemediationAction.NEEDS_USER_ACTION.value,
            QualityRemediationAction.CONTROLLED_VISUAL_REPLAN.value,
        ) or any("EXHAUSTED" in str(r) for r in reason_codes):
            if any("EXHAUSTED" in str(r) for r in reason_codes):
                self._emit_trace(
                    task_id=task.task_id,
                    event_type=TraceEventType.QUALITY_REVIEW_EXHAUSTED,
                    attributes={
                        "task_id": task.task_id,
                        "remediation_decision_id": remed_id,
                        "reason_codes": reason_codes,
                    },
                )
            reason = (
                "Controlled visual replan required"
                if action_str == QualityRemediationAction.CONTROLLED_VISUAL_REPLAN.value
                else f"Quality remediation requires user action: {reason_codes or action_str}"
            )
            task.transition_to(
                TaskStatus.WAITING_USER,
                reason=reason,
                now=ts,
            )
            self.task_repo.save_task(task)
            return None

        if action_str == QualityRemediationAction.RETRY_EVALUATION.value:
            task.task_metadata["is_partial_rerun"] = True
            self._emit_trace(
                task_id=task.task_id,
                event_type=TraceEventType.PARTIAL_RERUN_STARTED,
                attributes={
                    "task_id": task.task_id,
                    "action": action_str,
                    "evaluation_snapshot_id": eval_snap_id,
                },
            )
            self.task_repo.save_task(task)

            seq = 1
            idempotency_key = f"idemp_{task.task_id}_quality_review_{seq}"
            while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
                seq += 1
                idempotency_key = f"idemp_{task.task_id}_quality_review_{seq}"

            next_job = WorkflowJob.create(
                task_id=task.task_id,
                stage=Stage.QUALITY_REVIEW,
                idempotency_key=idempotency_key,
                attempt_number=1,
                input_task_artifact_ref_id=completed_job.input_task_artifact_ref_id,
                now=ts,
            )
            return self.job_repo.create_job(next_job)

        if action_str in (
            QualityRemediationAction.REGENERATE_SAME_ROUTE.value,
            QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE.value,
        ):
            # Invalidate downstream artifacts: COMPOSITION, QUALITY_REVIEW, DELIVERY
            self.artifact_repo.invalidate_downstream_artifacts(
                task_id=task.task_id,
                stages=[Stage.COMPOSITION, Stage.QUALITY_REVIEW, Stage.DELIVERY],
                reason=f"Quality remediation: {action_str}",
                now=ts,
            )

            task.task_metadata["is_partial_rerun"] = True
            task.set_stage_for_rerun(Stage.ASSET, now=ts)
            self.task_repo.save_task(task)

            self._emit_trace(
                task_id=task.task_id,
                event_type=TraceEventType.PARTIAL_RERUN_STARTED,
                attributes={
                    "task_id": task.task_id,
                    "action": action_str,
                    "affected_shot_ids": affected_shot_ids,
                    "remediation_decision_id": remed_id,
                },
            )

            input_ref_id = None
            if action_str == QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE.value:
                candidate_dict = meta.get("selected_route_candidate")
                route_plan_repo = AssetRoutePlanRepository(self._session)
                current_plan_ref = self.artifact_repo.get_latest_artifact_ref(
                    task_id=task.task_id,
                    stage=Stage.PRODUCTION_PLAN,
                    artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
                )
                if current_plan_ref is not None:
                    curr_plan = route_plan_repo.get_route_plan(current_plan_ref.artifact_id)
                    if curr_plan is not None and candidate_dict:
                        from app.domain.asset_router import AssetRouteCandidate
                        candidate_b = AssetRouteCandidate.model_validate(candidate_dict)
                        new_entries = []
                        for entry in curr_plan.shot_routes:
                            if entry.shot_id in affected_shot_ids:
                                updated_decision = entry.route_decision.model_copy(
                                    update={"selected_candidate": candidate_b}
                                )
                                updated_entry = entry.model_copy(
                                    update={"route_decision": updated_decision}
                                )
                                new_entries.append(updated_entry)
                            else:
                                new_entries.append(entry)
                        new_plan = curr_plan.model_copy(
                            update={
                                "asset_route_plan_id": str(uuid4()),
                                "shot_routes": tuple(new_entries),
                                "created_at": ts,
                            }
                        )
                        route_plan_repo.add_route_plan(new_plan)
                        new_plan_ref = TaskArtifactRef.create(
                            task_id=task.task_id,
                            stage=Stage.PRODUCTION_PLAN,
                            artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
                            artifact_id=new_plan.asset_route_plan_id,
                            metadata_json={
                                "fallback_from_route_plan_id": curr_plan.asset_route_plan_id,
                                "remediation_decision_id": remed_id,
                                "affected_shot_ids": affected_shot_ids,
                            },
                            now=ts,
                        )
                        saved_ref = self.artifact_repo.save_artifact_ref(new_plan_ref)
                        input_ref_id = saved_ref.task_artifact_ref_id

            if input_ref_id is None:
                plan_ref = self.artifact_repo.get_latest_artifact_ref(
                    task_id=task.task_id,
                    stage=Stage.PRODUCTION_PLAN,
                    artifact_type=ArtifactType.ASSET_ROUTE_PLAN,
                )
                input_ref_id = plan_ref.task_artifact_ref_id if plan_ref else None

            seq = 1
            idempotency_key = f"idemp_{task.task_id}_asset_{seq}"
            while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
                seq += 1
                idempotency_key = f"idemp_{task.task_id}_asset_{seq}"

            next_job = WorkflowJob.create(
                task_id=task.task_id,
                stage=Stage.ASSET,
                idempotency_key=idempotency_key,
                attempt_number=1,
                input_task_artifact_ref_id=input_ref_id,
                now=ts,
            )
            return self.job_repo.create_job(next_job)

        return None

    def on_stage_failed(
        self,
        task: KnowledgeVideoTask,
        failed_job: WorkflowJob,
        error_type: str = "RETRYABLE",
        error_message: str = "",
        now: datetime | None = None,
    ) -> None:
        """
        Invoked when a stage job fails. Sets appropriate failure / pause state.
        """
        ts = now or datetime.now(UTC)
        err_enum = None
        try:
            err_enum = JobErrorType(error_type)
        except ValueError:
            err_enum = JobErrorType.RETRYABLE

        if task.task_status == TaskStatus.CREATED:
            task.transition_to(TaskStatus.RUNNING, now=ts)

        if err_enum == JobErrorType.NEEDS_EVIDENCE:
            task.transition_to(
                TaskStatus.NEEDS_EVIDENCE,
                reason="Core claims lack verified evidence",
                error_type=err_enum,
                error_message=error_message,
                now=ts,
            )
        elif err_enum in (JobErrorType.NEEDS_RECOVERY, JobErrorType.RETRYABLE) or failed_job.attempt_number >= failed_job.max_attempts:
            task.transition_to(
                TaskStatus.NEEDS_RECOVERY,
                reason=error_message or "Stage execution failed and requires recovery",
                error_type=JobErrorType.NEEDS_RECOVERY,
                error_message=error_message,
                now=ts,
            )
        else:
            task.transition_to(
                TaskStatus.FAILED,
                reason=f"Stage execution failed: {error_type}",
                error_type=err_enum,
                error_message=error_message,
                now=ts,
            )
        self.task_repo.save_task(task)

    def resume_after_approval(
        self,
        task: KnowledgeVideoTask,
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        """
        Resumes execution of a task currently paused in WAITING_USER.
        Advances stage and creates the next stage job.
        """
        ts = now or datetime.now(UTC)
        if task.task_status != TaskStatus.WAITING_USER:
            raise WorkflowConflictError(
                f"Task '{task.task_id}' cannot be approved: status is '{task.task_status.value}', expected WAITING_USER."
            )

        prior_stage = task.current_stage
        next_stage = get_next_stage(task.current_stage)
        if next_stage is None:
            task.transition_to(TaskStatus.COMPLETED, now=ts)
            self.task_repo.save_task(task)
            return None

        latest_ref = self.artifact_repo.get_latest_artifact_ref(task.task_id, stage=prior_stage)
        input_ref_id = latest_ref.task_artifact_ref_id if latest_ref else None

        # If resuming from STORYBOARD checkpoint, perform domain approval
        if prior_stage == Stage.STORYBOARD:
            input_ref_id = self._approve_storyboard_for_task(
                task_id=task.task_id,
                draft_output_ref_id=input_ref_id,
                approved_by="user_approval",
                now=ts,
            )

        task.advance_stage(next_stage)
        task.transition_to(TaskStatus.RUNNING, now=ts)
        self.task_repo.save_task(task)

        seq = 1
        idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"
        while self.job_repo.get_job_by_idempotency_key(idempotency_key) is not None:
            seq += 1
            idempotency_key = f"idemp_{task.task_id}_{next_stage.value.lower()}_{seq}"

        next_job = WorkflowJob.create(
            task_id=task.task_id,
            stage=next_stage,
            idempotency_key=idempotency_key,
            attempt_number=1,
            input_task_artifact_ref_id=input_ref_id,
            now=ts,
        )
        return self.job_repo.create_job(next_job)

    def _approve_storyboard_for_task(
        self,
        task_id: str,
        draft_output_ref_id: str | None,
        approved_by: str = "workflow",
        user_note: str | None = None,
        now: datetime | None = None,
    ) -> str | None:
        """
        Executes domain approval via StoryboardApprovalService for a task's DRAFT storyboard.
        Persists a new TaskArtifactRef for the APPROVED snapshot and returns its task_artifact_ref_id.
        """
        ts = now or datetime.now(UTC)
        draft_ref = None
        if draft_output_ref_id:
            draft_ref = self.artifact_repo.get_artifact_ref(draft_output_ref_id)
        if draft_ref is None:
            draft_ref = self.artifact_repo.get_latest_artifact_ref(
                task_id=task_id,
                stage=Stage.STORYBOARD,
                artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            )
        if draft_ref is None:
            return draft_output_ref_id

        storyboard_repo = StoryboardRepository(self._session)
        snap = storyboard_repo.get_snapshot(draft_ref.artifact_id)
        if snap is not None and snap.snapshot_state == StoryboardSnapshotState.APPROVED:
            return draft_ref.task_artifact_ref_id

        plan_repo = ContentPlanRepository(self._session)
        shot_repo = ShotRepository(self._session)
        approval_repo = StoryboardApprovalRepository(self._session)

        approval_service = StoryboardApprovalService(
            plan_repository=plan_repo,
            shot_repository=shot_repo,
            storyboard_repository=storyboard_repo,
            approval_repository=approval_repo,
        )

        outcome = approval_service.approve_storyboard(
            storyboard_snapshot_id=draft_ref.artifact_id,
            approved_by=approved_by,
            user_note=user_note,
        )

        approved_snapshot = outcome.approved_snapshot
        approved_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.STORYBOARD,
            artifact_type=ArtifactType.STORYBOARD_SNAPSHOT,
            artifact_id=approved_snapshot.storyboard_snapshot_id,
            artifact_version="approved",
            metadata_json={
                "source_draft_snapshot_id": draft_ref.artifact_id,
                "approved_snapshot_id": approved_snapshot.storyboard_snapshot_id,
                "approved_by": approved_by,
                "shot_count": len(approved_snapshot.shot_revision_ids),
                "state": approved_snapshot.snapshot_state.value,
            },
            now=ts,
        )
        saved_ref = self.artifact_repo.save_artifact_ref(approved_ref)
        return saved_ref.task_artifact_ref_id
