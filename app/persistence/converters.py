from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

from app.domain.asset_router import AssetRoutePlan
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.shot import Shot, ShotRevision
from app.domain.stage_execution import StageExecution
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import (
    AssetRoutePlanORM,
    AttemptRequestORM,
    AttemptResultORM,
    BenchmarkCaseResultORM,
    BenchmarkComparisonReportORM,
    BenchmarkReportORM,
    BenchmarkRunORM,
    CompositionPreviewORM,
    ContentBeatORM,
    ContentPlanRevisionORM,
    DimensionEvaluationResultORM,
    EvaluationSnapshotDimensionResultORM,
    EvaluationSnapshotORM,
    EvaluationTargetORM,
    ExecutionAttemptORM,
    ExecutionRunORM,
    ExecutionTransitionORM,
    KnowledgeVideoTaskORM,
    ProviderReceiptORM,
    QualityRemediationDecisionORM,
    ShotAssetVersionORM,
    ShotExecutionORM,
    ShotORM,
    ShotQualitySelectionORM,
    ShotRevisionORM,
    ScriptRevisionORM,
    ScriptSegmentORM,
    StageExecutionORM,
    StoryboardApprovalRecordORM,
    StoryboardSnapshotORM,
    StoryboardSnapshotShotRevisionORM,
    TaskArtifactRefORM,
    TraceEventORM,
    TraceRootORM,
    WorkflowJobORM,
)

if TYPE_CHECKING:
    from app.domain.asset_execution import (
        AttemptRequest,
        AttemptResult,
        ExecutionAttempt,
        ExecutionRun,
        ExecutionTransition,
        ProviderReceipt,
        ShotAssetVersion,
        ShotExecution,
    )
    from app.domain.benchmark import (
        BenchmarkCaseResult,
        BenchmarkRun,
    )
    from app.domain.benchmark_comparison import BenchmarkComparisonReport
    from app.domain.benchmark_report import BenchmarkReport
    from app.domain.evaluation import (
        DimensionEvaluationResult,
        EvaluationSnapshot,
        EvaluationTarget,
    )
    from app.domain.quality_remediation import (
        QualityRemediationDecision,
        ShotQualitySelection,
    )
    from app.domain.trace import TraceEvent, TraceRoot
    from app.services.evaluation.composition_preview import CompositionPreview



def content_beat_to_orm(
    beat: ContentBeat, content_plan_revision_id: str
) -> ContentBeatORM:
    """Converts a domain ContentBeat to a ContentBeatORM record."""
    return ContentBeatORM(
        beat_id=beat.beat_id,
        content_plan_revision_id=content_plan_revision_id,
        beat_lineage_id=beat.beat_lineage_id,
        beat_type=beat.beat_type.value
        if hasattr(beat.beat_type, "value")
        else str(beat.beat_type),
        order=beat.order,
        intent=beat.intent,
        target_duration=beat.target_duration,
        importance=beat.importance,
        evidence_refs=list(beat.evidence_refs),
    )


def content_beat_from_orm(orm: ContentBeatORM) -> ContentBeat:
    """Reconstructs a domain ContentBeat from a ContentBeatORM record."""
    return ContentBeat(
        beat_id=orm.beat_id,
        beat_lineage_id=orm.beat_lineage_id,
        beat_type=BeatType(orm.beat_type),
        order=orm.order,
        intent=orm.intent,
        target_duration=orm.target_duration,
        importance=orm.importance,
        evidence_refs=tuple(orm.evidence_refs or []),
    )


def content_plan_revision_to_orm(plan: ContentPlanRevision) -> ContentPlanRevisionORM:
    """Converts a domain ContentPlanRevision to a ContentPlanRevisionORM entity with associated beats."""
    orm_plan = ContentPlanRevisionORM(
        content_plan_revision_id=plan.content_plan_revision_id,
        revision_number=plan.revision_number,
        topic=plan.topic,
        overall_target_duration=plan.overall_target_duration,
        global_retrieval_snapshot_id=plan.global_retrieval_snapshot_id,
        created_at=plan.created_at,
    )
    orm_plan.beats = [
        content_beat_to_orm(b, plan.content_plan_revision_id) for b in plan.beats
    ]
    return orm_plan


def content_plan_revision_from_orm(orm: ContentPlanRevisionORM) -> ContentPlanRevision:
    """Reconstructs an immutable domain ContentPlanRevision from a ContentPlanRevisionORM entity."""
    beats = tuple(
        content_beat_from_orm(b) for b in sorted(orm.beats, key=lambda x: x.order)
    )
    return ContentPlanRevision(
        content_plan_revision_id=orm.content_plan_revision_id,
        revision_number=orm.revision_number,
        topic=orm.topic,
        overall_target_duration=orm.overall_target_duration,
        beats=beats,
        global_retrieval_snapshot_id=orm.global_retrieval_snapshot_id,
        created_at=orm.created_at,
    )


def shot_to_orm(shot: Shot) -> ShotORM:
    """Converts a domain Shot entity to a ShotORM record."""
    return ShotORM(
        shot_id=shot.shot_id,
        beat_lineage_id=shot.beat_lineage_id,
        local_order=shot.local_order,
        created_at=shot.created_at,
    )


def shot_from_orm(orm: ShotORM) -> Shot:
    """Reconstructs a domain Shot entity from a ShotORM record."""
    return Shot(
        shot_id=orm.shot_id,
        beat_lineage_id=orm.beat_lineage_id,
        local_order=orm.local_order,
        created_at=orm.created_at,
    )


def shot_revision_to_orm(rev: ShotRevision) -> ShotRevisionORM:
    """Converts a domain ShotRevision to a ShotRevisionORM record."""
    return ShotRevisionORM(
        shot_revision_id=rev.shot_revision_id,
        shot_id=rev.shot_id,
        revision_number=rev.revision_number,
        beat_lineage_id=rev.beat_lineage_id,
        created_from_beat_instance_id=rev.created_from_beat_instance_id,
        script_segment_id=rev.script_segment_id,
        narration=rev.narration,
        target_duration=rev.target_duration,
        visual_goal=rev.visual_goal,
        visual_type=rev.visual_type.value
        if hasattr(rev.visual_type, "value")
        else str(rev.visual_type),
        scene_description=rev.scene_description,
        generation_prompt=rev.generation_prompt,
        camera_movement=rev.camera_movement,
        evidence_refs=list(rev.evidence_refs),
        created_at=rev.created_at,
    )


def shot_revision_from_orm(orm: ShotRevisionORM) -> ShotRevision:
    """Reconstructs an immutable domain ShotRevision from a ShotRevisionORM record."""
    return ShotRevision(
        shot_revision_id=orm.shot_revision_id,
        shot_id=orm.shot_id,
        revision_number=orm.revision_number,
        beat_lineage_id=orm.beat_lineage_id,
        created_from_beat_instance_id=orm.created_from_beat_instance_id,
        script_segment_id=orm.script_segment_id,
        narration=orm.narration,
        target_duration=orm.target_duration,
        visual_goal=orm.visual_goal,
        visual_type=VisualType(orm.visual_type),
        scene_description=orm.scene_description,
        generation_prompt=orm.generation_prompt,
        camera_movement=orm.camera_movement,
        evidence_refs=tuple(orm.evidence_refs or []),
        created_at=orm.created_at,
    )


def storyboard_snapshot_to_orm(snap: StoryboardSnapshot) -> StoryboardSnapshotORM:
    """Converts a domain StoryboardSnapshot to a StoryboardSnapshotORM entity with associations."""
    orm_snap = StoryboardSnapshotORM(
        storyboard_snapshot_id=snap.storyboard_snapshot_id,
        content_plan_revision_id=snap.content_plan_revision_id,
        snapshot_state=(
            snap.snapshot_state.value
            if hasattr(snap.snapshot_state, "value")
            else str(snap.snapshot_state)
        ),
        created_at=snap.created_at,
    )
    orm_snap.snapshot_revisions = [
        StoryboardSnapshotShotRevisionORM(
            storyboard_snapshot_id=snap.storyboard_snapshot_id,
            shot_revision_id=rev_id,
            sequence=idx,
        )
        for idx, rev_id in enumerate(snap.shot_revision_ids)
    ]
    return orm_snap


def storyboard_snapshot_from_orm(orm: StoryboardSnapshotORM) -> StoryboardSnapshot:
    """Reconstructs an immutable domain StoryboardSnapshot from a StoryboardSnapshotORM entity."""
    sorted_associations = sorted(orm.snapshot_revisions, key=lambda a: a.sequence)
    shot_rev_ids = tuple(a.shot_revision_id for a in sorted_associations)
    return StoryboardSnapshot(
        storyboard_snapshot_id=orm.storyboard_snapshot_id,
        content_plan_revision_id=orm.content_plan_revision_id,
        shot_revision_ids=shot_rev_ids,
        snapshot_state=StoryboardSnapshotState(orm.snapshot_state),
        created_at=orm.created_at,
    )


def storyboard_approval_record_to_orm(
    record: StoryboardApprovalRecord,
) -> StoryboardApprovalRecordORM:
    """Converts a domain StoryboardApprovalRecord to a StoryboardApprovalRecordORM record."""
    return StoryboardApprovalRecordORM(
        storyboard_approval_id=record.storyboard_approval_id,
        source_draft_snapshot_id=record.source_draft_snapshot_id,
        approved_storyboard_snapshot_id=record.approved_storyboard_snapshot_id,
        content_plan_revision_id=record.content_plan_revision_id,
        exact_shot_revision_ids=list(record.exact_shot_revision_ids),
        approved_by=record.approved_by,
        approved_at=record.approved_at,
        user_note=record.user_note,
    )


def storyboard_approval_record_from_orm(
    orm: StoryboardApprovalRecordORM,
) -> StoryboardApprovalRecord:
    """Reconstructs an immutable domain StoryboardApprovalRecord from a StoryboardApprovalRecordORM record."""
    return StoryboardApprovalRecord(
        storyboard_approval_id=orm.storyboard_approval_id,
        source_draft_snapshot_id=orm.source_draft_snapshot_id,
        approved_storyboard_snapshot_id=orm.approved_storyboard_snapshot_id,
        content_plan_revision_id=orm.content_plan_revision_id,
        exact_shot_revision_ids=tuple(orm.exact_shot_revision_ids or []),
        approved_by=orm.approved_by,
        approved_at=orm.approved_at,
        user_note=orm.user_note,
    )


def asset_route_plan_to_orm(plan: AssetRoutePlan) -> AssetRoutePlanORM:
    """Converts a domain AssetRoutePlan to an AssetRoutePlanORM and child ShotRoutePlanEntryORM records."""
    from uuid import uuid4

    from app.persistence.models import AssetRoutePlanORM, ShotRoutePlanEntryORM

    orm_plan = AssetRoutePlanORM(
        asset_route_plan_id=plan.asset_route_plan_id,
        storyboard_snapshot_id=plan.storyboard_snapshot_id,
        content_plan_revision_id=plan.content_plan_revision_id,
        routing_strategy=plan.routing_strategy.value,
        routing_policy_version=plan.routing_policy_version,
        model_selection_mode=plan.model_selection_mode.value,
        selected_provider=plan.selected_provider,
        selected_model=plan.selected_model,
        status=plan.status.value,
        total_shots=plan.total_shots,
        routed_shots=plan.routed_shots,
        blocked_shots=plan.blocked_shots,
        created_at=plan.created_at,
    )

    orm_entries = []
    for seq, entry in enumerate(plan.shot_routes):
        orm_entries.append(
            ShotRoutePlanEntryORM(
                entry_id=str(uuid4()),
                asset_route_plan_id=plan.asset_route_plan_id,
                shot_id=entry.shot_id,
                shot_revision_id=entry.shot_revision_id,
                beat_lineage_id=entry.beat_lineage_id,
                sequence=seq,
                requested_visual_type=entry.requested_visual_type.value,
                route_status=entry.route_status.value,
                asset_routing_request_json=entry.asset_routing_request.model_dump_json(),
                route_decision_json=entry.route_decision.model_dump_json(),
            )
        )
    orm_plan.entries = orm_entries
    return orm_plan


def asset_route_plan_from_orm(orm: AssetRoutePlanORM) -> AssetRoutePlan:
    """Reconstructs an immutable domain AssetRoutePlan from an AssetRoutePlanORM record."""
    from app.domain.asset_router import (
        AssetRouteDecision,
        AssetRoutePlan,
        AssetRoutePlanStatus,
        AssetRoutingRequest,
        ModelSelectionMode,
        RoutingStrategy,
        ShotRoutePlanEntry,
        ShotRoutePlanStatus,
    )
    from app.domain.enums import VisualType

    entries = []
    sorted_entries = sorted(orm.entries or [], key=lambda e: e.sequence)
    for e in sorted_entries:
        req = AssetRoutingRequest.model_validate_json(e.asset_routing_request_json)
        dec = AssetRouteDecision.model_validate_json(e.route_decision_json)
        entries.append(
            ShotRoutePlanEntry(
                shot_id=e.shot_id,
                shot_revision_id=e.shot_revision_id,
                beat_lineage_id=e.beat_lineage_id,
                requested_visual_type=VisualType(e.requested_visual_type),
                asset_routing_request=req,
                route_decision=dec,
                route_status=ShotRoutePlanStatus(e.route_status),
            )
        )

    return AssetRoutePlan(
        asset_route_plan_id=orm.asset_route_plan_id,
        storyboard_snapshot_id=orm.storyboard_snapshot_id,
        content_plan_revision_id=orm.content_plan_revision_id,
        routing_strategy=RoutingStrategy(orm.routing_strategy),
        routing_policy_version=orm.routing_policy_version,
        model_selection_mode=ModelSelectionMode(orm.model_selection_mode),
        selected_provider=orm.selected_provider,
        selected_model=orm.selected_model,
        shot_routes=tuple(entries),
        status=AssetRoutePlanStatus(orm.status),
        total_shots=orm.total_shots,
        routed_shots=orm.routed_shots,
        blocked_shots=orm.blocked_shots,
        created_at=orm.created_at,
    )


def execution_run_to_orm(run: ExecutionRun) -> ExecutionRunORM:
    """Converts a domain ExecutionRun to an ExecutionRunORM record."""
    from app.persistence.models import ExecutionRunORM

    return ExecutionRunORM(
        execution_run_id=run.execution_run_id,
        asset_route_plan_id=run.asset_route_plan_id,
        storyboard_snapshot_id=run.storyboard_snapshot_id,
        status=run.status.value,
        total_shots=run.total_shots,
        succeeded_shots=run.succeeded_shots,
        reused_shots=run.reused_shots,
        failed_shots=run.failed_shots,
        recovery_required_shots=run.recovery_required_shots,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def execution_run_from_orm(orm: ExecutionRunORM) -> ExecutionRun:
    """Reconstructs a domain ExecutionRun from an ExecutionRunORM record."""
    from app.domain.asset_execution import ExecutionRun, ExecutionStatus

    return ExecutionRun(
        execution_run_id=orm.execution_run_id,
        asset_route_plan_id=orm.asset_route_plan_id,
        storyboard_snapshot_id=orm.storyboard_snapshot_id,
        status=ExecutionStatus(orm.status),
        total_shots=orm.total_shots,
        succeeded_shots=orm.succeeded_shots,
        reused_shots=getattr(orm, "reused_shots", 0) or 0,
        failed_shots=orm.failed_shots,
        recovery_required_shots=getattr(orm, "recovery_required_shots", 0) or 0,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def execution_attempt_to_orm(attempt: ExecutionAttempt) -> ExecutionAttemptORM:
    """Converts a domain ExecutionAttempt to an ExecutionAttemptORM record."""
    from app.persistence.models import ExecutionAttemptORM

    raw_json = (
        json.dumps(attempt.raw_provider_response, ensure_ascii=False)
        if attempt.raw_provider_response is not None
        else None
    )

    return ExecutionAttemptORM(
        execution_attempt_id=attempt.execution_attempt_id,
        execution_run_id=attempt.execution_run_id,
        shot_id=attempt.shot_id,
        shot_revision_id=attempt.shot_revision_id,
        attempt_number=attempt.attempt_number,
        provider=attempt.provider,
        model=attempt.model,
        generation_mode=attempt.generation_mode.value,
        status=attempt.status.value,
        started_at=attempt.started_at,
        finished_at=attempt.finished_at,
        duration_seconds=attempt.duration_seconds,
        error_code=attempt.error_code,
        error_message=attempt.error_message,
        raw_provider_response_json=raw_json,
    )


def execution_attempt_from_orm(orm: ExecutionAttemptORM) -> ExecutionAttempt:
    """Reconstructs a domain ExecutionAttempt from an ExecutionAttemptORM record."""
    from app.domain.asset_execution import ExecutionAttempt, ExecutionAttemptStatus
    from app.domain.asset_router import GenerationMode

    raw = (
        json.loads(orm.raw_provider_response_json)
        if orm.raw_provider_response_json
        else None
    )

    return ExecutionAttempt(
        execution_attempt_id=orm.execution_attempt_id,
        execution_run_id=orm.execution_run_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        attempt_number=orm.attempt_number,
        provider=orm.provider,
        model=orm.model,
        generation_mode=GenerationMode(orm.generation_mode),
        status=ExecutionAttemptStatus(orm.status),
        started_at=orm.started_at,
        finished_at=orm.finished_at,
        duration_seconds=orm.duration_seconds,
        error_code=orm.error_code,
        error_message=orm.error_message,
        raw_provider_response=raw,
    )


def shot_asset_version_to_orm(version: ShotAssetVersion) -> ShotAssetVersionORM:
    """Converts a domain ShotAssetVersion to a ShotAssetVersionORM record."""
    from app.persistence.models import ShotAssetVersionORM

    return ShotAssetVersionORM(
        shot_asset_version_id=version.shot_asset_version_id,
        shot_id=version.shot_id,
        shot_revision_id=version.shot_revision_id,
        execution_attempt_id=version.execution_attempt_id,
        file_path=version.file_path,
        file_hash=version.file_hash,
        file_size_bytes=version.file_size_bytes,
        media_type=version.media_type.value,
        mime_type=version.mime_type,
        width=version.width,
        height=version.height,
        duration_seconds=version.duration_seconds,
        fps=version.fps,
        provider=version.provider,
        model=version.model,
        generation_mode=version.generation_mode.value,
        created_at=version.created_at,
    )


def shot_asset_version_from_orm(orm: ShotAssetVersionORM) -> ShotAssetVersion:
    """Reconstructs a domain ShotAssetVersion from a ShotAssetVersionORM record."""
    from app.domain.asset_execution import AssetMediaType, ShotAssetVersion
    from app.domain.asset_router import GenerationMode

    return ShotAssetVersion(
        shot_asset_version_id=orm.shot_asset_version_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        execution_attempt_id=orm.execution_attempt_id,
        file_path=orm.file_path,
        file_hash=orm.file_hash,
        file_size_bytes=orm.file_size_bytes,
        media_type=AssetMediaType(orm.media_type),
        mime_type=orm.mime_type,
        width=orm.width,
        height=orm.height,
        duration_seconds=orm.duration_seconds,
        fps=orm.fps,
        provider=orm.provider,
        model=orm.model,
        generation_mode=GenerationMode(orm.generation_mode),
        created_at=orm.created_at,
    )


def shot_execution_to_orm(execution: ShotExecution) -> ShotExecutionORM:
    """Converts a domain ShotExecution to a ShotExecutionORM record."""
    return ShotExecutionORM(
        shot_execution_id=execution.shot_execution_id,
        execution_run_id=execution.execution_run_id,
        shot_id=execution.shot_id,
        shot_revision_id=execution.shot_revision_id,
        status=execution.status.value,
        current_candidate_index=execution.current_candidate_index,
        attempt_ids_json=json.dumps(execution.attempt_ids, ensure_ascii=False),
        produced_asset_version_id=execution.produced_asset_version_id,
        error_code=execution.error_code,
        error_message=execution.error_message,
        unconfirmed_remote_task_id=execution.unconfirmed_remote_task_id,
        route_decision_json=execution.route_decision.model_dump_json(),
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


def shot_execution_from_orm(orm: ShotExecutionORM) -> ShotExecution:
    """Reconstructs a domain ShotExecution from a ShotExecutionORM record."""
    from app.domain.asset_execution import ShotExecution, ShotExecutionStatus
    from app.domain.asset_router import AssetRouteDecision

    decision = AssetRouteDecision.model_validate_json(orm.route_decision_json)
    attempt_ids = json.loads(orm.attempt_ids_json) if orm.attempt_ids_json else []

    return ShotExecution(
        shot_execution_id=orm.shot_execution_id,
        execution_run_id=orm.execution_run_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        route_decision=decision,
        status=ShotExecutionStatus(orm.status),
        current_candidate_index=orm.current_candidate_index,
        attempt_ids=attempt_ids,
        produced_asset_version_id=orm.produced_asset_version_id,
        error_code=orm.error_code,
        error_message=orm.error_message,
        unconfirmed_remote_task_id=orm.unconfirmed_remote_task_id,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def attempt_request_to_orm(req: AttemptRequest) -> AttemptRequestORM:
    """Converts a domain AttemptRequest to an AttemptRequestORM record."""
    from app.persistence.models import AttemptRequestORM

    return AttemptRequestORM(
        attempt_request_id=str(uuid4()),
        execution_attempt_id=req.execution_attempt_id,
        provider=req.provider,
        model=req.model,
        generation_mode=req.generation_mode.value,
        idempotency_key=req.idempotency_key,
        sanitized_payload_json=json.dumps(req.sanitized_payload, ensure_ascii=False),
        request_hash=req.request_hash,
        created_at=req.created_at,
    )


def attempt_request_from_orm(orm: AttemptRequestORM) -> AttemptRequest:
    """Reconstructs a domain AttemptRequest from an AttemptRequestORM record."""
    from app.domain.asset_execution import AttemptRequest
    from app.domain.asset_router import GenerationMode

    payload = json.loads(orm.sanitized_payload_json) if orm.sanitized_payload_json else {}
    return AttemptRequest(
        execution_attempt_id=orm.execution_attempt_id,
        provider=orm.provider,
        model=orm.model,
        generation_mode=GenerationMode(orm.generation_mode),
        idempotency_key=orm.idempotency_key,
        sanitized_payload=payload,
        request_hash=orm.request_hash,
        created_at=orm.created_at,
    )


def provider_receipt_to_orm(receipt: ProviderReceipt) -> ProviderReceiptORM:
    """Converts a domain ProviderReceipt to a ProviderReceiptORM record."""
    from app.persistence.models import ProviderReceiptORM

    return ProviderReceiptORM(
        receipt_id=str(uuid4()),
        execution_attempt_id=receipt.execution_attempt_id,
        provider=receipt.provider,
        provider_job_id=receipt.provider_job_id,
        accepted_at=receipt.accepted_at,
        provider_status=receipt.provider_status,
        sanitized_metadata_json=json.dumps(receipt.sanitized_metadata, ensure_ascii=False),
    )


def provider_receipt_from_orm(orm: ProviderReceiptORM) -> ProviderReceipt:
    """Reconstructs a domain ProviderReceipt from a ProviderReceiptORM record."""
    from app.domain.asset_execution import ProviderReceipt

    metadata = json.loads(orm.sanitized_metadata_json) if orm.sanitized_metadata_json else {}
    return ProviderReceipt(
        execution_attempt_id=orm.execution_attempt_id,
        provider=orm.provider,
        provider_job_id=orm.provider_job_id,
        accepted_at=orm.accepted_at,
        provider_status=orm.provider_status,
        sanitized_metadata=metadata,
    )


def attempt_result_to_orm(result: AttemptResult) -> AttemptResultORM:
    """Converts a domain AttemptResult to an AttemptResultORM record."""
    from app.persistence.models import AttemptResultORM

    return AttemptResultORM(
        result_id=str(uuid4()),
        execution_attempt_id=result.execution_attempt_id,
        outcome=result.outcome.value,
        provider_status=result.provider_status,
        asset_reference=result.asset_reference,
        error_category=result.error_category.value if result.error_category else None,
        error_code=result.error_code,
        diagnostic_summary=result.diagnostic_summary,
        completed_at=result.completed_at,
    )


def attempt_result_from_orm(orm: AttemptResultORM) -> AttemptResult:
    """Reconstructs a domain AttemptResult from an AttemptResultORM record."""
    from app.domain.asset_execution import (
        AttemptResult,
        FailureCategory,
        ProviderOutcomeType,
    )

    cat = FailureCategory(orm.error_category) if orm.error_category else None
    return AttemptResult(
        execution_attempt_id=orm.execution_attempt_id,
        outcome=ProviderOutcomeType(orm.outcome),
        provider_status=orm.provider_status,
        asset_reference=orm.asset_reference,
        error_category=cat,
        error_code=orm.error_code,
        diagnostic_summary=orm.diagnostic_summary,
        completed_at=orm.completed_at,
    )


def execution_transition_to_orm(transition: ExecutionTransition) -> ExecutionTransitionORM:
    """Converts a domain ExecutionTransition to an ExecutionTransitionORM record."""
    from app.persistence.models import ExecutionTransitionORM

    return ExecutionTransitionORM(
        transition_id=transition.transition_id,
        entity_type=transition.entity_type,
        entity_id=transition.entity_id,
        from_state=transition.from_state,
        to_state=transition.to_state,
        reason_code=transition.reason_code,
        created_at=transition.created_at,
    )


def execution_transition_from_orm(orm: ExecutionTransitionORM) -> ExecutionTransition:
    """Reconstructs a domain ExecutionTransition from an ExecutionTransitionORM record."""
    from app.domain.asset_execution import ExecutionTransition

    return ExecutionTransition(
        transition_id=orm.transition_id,
        entity_type=orm.entity_type,
        entity_id=orm.entity_id,
        from_state=orm.from_state,
        to_state=orm.to_state,
        reason_code=orm.reason_code,
        created_at=orm.created_at,
    )


def evaluation_target_to_orm(target: EvaluationTarget) -> EvaluationTargetORM:
    """Converts a domain EvaluationTarget to an EvaluationTargetORM record."""
    return EvaluationTargetORM(
        evaluation_target_id=target.evaluation_target_id,
        shot_id=target.shot_id,
        shot_revision_id=target.shot_revision_id,
        shot_asset_version_id=target.shot_asset_version_id,
        evidence_refs_snapshot=list(target.evidence_refs_snapshot),
        context_refs=[ref.model_dump() for ref in target.context_refs],
        target_version=target.target_version,
        created_at=target.created_at,
    )


def evaluation_target_from_orm(orm: EvaluationTargetORM) -> EvaluationTarget:
    """Reconstructs an immutable domain EvaluationTarget from an EvaluationTargetORM record."""
    from app.domain.evaluation import EvaluationContextRef, EvaluationTarget

    context_refs = tuple(
        EvaluationContextRef(**item) if isinstance(item, dict) else item
        for item in (orm.context_refs or [])
    )
    return EvaluationTarget(
        evaluation_target_id=orm.evaluation_target_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        shot_asset_version_id=orm.shot_asset_version_id,
        evidence_refs_snapshot=tuple(orm.evidence_refs_snapshot or []),
        context_refs=context_refs,
        target_version=orm.target_version,
        created_at=orm.created_at,
    )


def dimension_evaluation_result_to_orm(
    result: DimensionEvaluationResult,
) -> DimensionEvaluationResultORM:
    """Converts a domain DimensionEvaluationResult to a DimensionEvaluationResultORM record."""
    return DimensionEvaluationResultORM(
        dimension_result_id=result.dimension_result_id,
        evaluation_target_id=result.evaluation_target_id,
        dimension=result.dimension.value if hasattr(result.dimension, "value") else str(result.dimension),
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        score=result.score,
        reason_codes=list(result.reason_codes),
        concise_summary=result.concise_summary,
        evaluator_version=result.evaluator_version,
        dimension_semantics_version=result.dimension_semantics_version,
        dependency_fingerprint=result.dependency_fingerprint,
        error_code=result.error_code,
        created_at=result.created_at,
    )


def dimension_evaluation_result_from_orm(
    orm: DimensionEvaluationResultORM,
) -> DimensionEvaluationResult:
    """Reconstructs an immutable domain DimensionEvaluationResult from a DimensionEvaluationResultORM record."""
    from app.domain.evaluation import (
        DimensionEvaluationResult,
        DimensionEvaluationStatus,
        EvaluationDimension,
    )

    return DimensionEvaluationResult(
        dimension_result_id=orm.dimension_result_id,
        evaluation_target_id=orm.evaluation_target_id,
        dimension=EvaluationDimension(orm.dimension),
        status=DimensionEvaluationStatus(orm.status),
        score=orm.score,
        reason_codes=tuple(orm.reason_codes or []),
        concise_summary=orm.concise_summary,
        evaluator_version=orm.evaluator_version,
        dimension_semantics_version=orm.dimension_semantics_version,
        dependency_fingerprint=orm.dependency_fingerprint,
        error_code=orm.error_code,
        created_at=orm.created_at,
    )


def evaluation_snapshot_to_orm(snapshot: EvaluationSnapshot) -> EvaluationSnapshotORM:
    """Converts a domain EvaluationSnapshot to an EvaluationSnapshotORM entity and associations."""
    orm_snap = EvaluationSnapshotORM(
        evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
        evaluation_target_id=snapshot.evaluation_target_id,
        decision=snapshot.decision.value if hasattr(snapshot.decision, "value") else str(snapshot.decision),
        policy_version=snapshot.policy_version,
        evaluator_version=snapshot.evaluator_version,
        overall_score=snapshot.overall_score,
        summary_reason_codes=list(snapshot.summary_reason_codes),
        created_at=snapshot.created_at,
    )
    orm_snap.dimension_associations = [
        EvaluationSnapshotDimensionResultORM(
            evaluation_snapshot_id=snapshot.evaluation_snapshot_id,
            dimension_result_id=dim_res_id,
            sequence=idx,
        )
        for idx, dim_res_id in enumerate(snapshot.dimension_result_ids)
    ]
    return orm_snap


def evaluation_snapshot_from_orm(orm: EvaluationSnapshotORM) -> EvaluationSnapshot:
    """Reconstructs an immutable domain EvaluationSnapshot from an EvaluationSnapshotORM entity."""
    from app.domain.evaluation import EvaluationDecision, EvaluationSnapshot

    sorted_associations = sorted(orm.dimension_associations, key=lambda a: a.sequence)
    dimension_result_ids = tuple(a.dimension_result_id for a in sorted_associations)
    return EvaluationSnapshot(
        evaluation_snapshot_id=orm.evaluation_snapshot_id,
        evaluation_target_id=orm.evaluation_target_id,
        dimension_result_ids=dimension_result_ids,
        decision=EvaluationDecision(orm.decision),
        policy_version=orm.policy_version,
        evaluator_version=orm.evaluator_version,
        overall_score=orm.overall_score,
        summary_reason_codes=tuple(orm.summary_reason_codes or []),
        created_at=orm.created_at,
    )


def composition_preview_to_orm(preview: CompositionPreview) -> CompositionPreviewORM:
    """Converts a domain CompositionPreview to a CompositionPreviewORM entity."""
    return CompositionPreviewORM(
        composition_preview_id=preview.composition_preview_id,
        shot_asset_version_id=preview.shot_asset_version_id,
        render_context_fingerprint=preview.render_context_fingerprint,
        preview_path=preview.preview_path,
        file_hash=preview.file_hash,
        width=preview.width,
        height=preview.height,
        preview_policy_version=preview.preview_policy_version,
        created_at=preview.created_at,
    )


def composition_preview_from_orm(orm: CompositionPreviewORM) -> CompositionPreview:
    """Reconstructs a domain CompositionPreview from a CompositionPreviewORM entity."""
    from app.services.evaluation.composition_preview import CompositionPreview

    return CompositionPreview(
        composition_preview_id=orm.composition_preview_id,
        shot_asset_version_id=orm.shot_asset_version_id,
        render_context_fingerprint=orm.render_context_fingerprint,
        preview_path=orm.preview_path,
        file_hash=orm.file_hash,
        width=orm.width,
        height=orm.height,
        preview_policy_version=orm.preview_policy_version,
        created_at=orm.created_at,
    )


def quality_remediation_decision_to_orm(
    decision: QualityRemediationDecision,
) -> QualityRemediationDecisionORM:
    """Converts a domain QualityRemediationDecision to a QualityRemediationDecisionORM entity."""
    cand_json = (
        decision.selected_route_candidate.model_dump(mode="json")
        if decision.selected_route_candidate
        else None
    )
    return QualityRemediationDecisionORM(
        remediation_decision_id=decision.remediation_decision_id,
        quality_chain_id=decision.quality_chain_id,
        evaluation_snapshot_id=decision.evaluation_snapshot_id,
        shot_id=decision.shot_id,
        shot_revision_id=decision.shot_revision_id,
        shot_asset_version_id=decision.shot_asset_version_id,
        action=decision.action.value if hasattr(decision.action, "value") else str(decision.action),
        reason_codes=list(decision.reason_codes),
        remediation_policy_version=decision.remediation_policy_version,
        quality_attempt_index=decision.quality_attempt_index,
        selected_route_candidate_json=cand_json,
        explanation=decision.explanation,
        created_at=decision.created_at,
    )


def quality_remediation_decision_from_orm(
    orm: QualityRemediationDecisionORM,
) -> QualityRemediationDecision:
    """Reconstructs a domain QualityRemediationDecision from a QualityRemediationDecisionORM entity."""
    from app.domain.asset_router import AssetRouteCandidate
    from app.domain.quality_remediation import (
        QualityRemediationAction,
        QualityRemediationDecision,
    )

    cand = (
        AssetRouteCandidate.model_validate(orm.selected_route_candidate_json)
        if orm.selected_route_candidate_json
        else None
    )
    return QualityRemediationDecision(
        remediation_decision_id=orm.remediation_decision_id,
        quality_chain_id=orm.quality_chain_id,
        evaluation_snapshot_id=orm.evaluation_snapshot_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        shot_asset_version_id=orm.shot_asset_version_id,
        action=QualityRemediationAction(orm.action),
        reason_codes=tuple(orm.reason_codes or []),
        remediation_policy_version=orm.remediation_policy_version,
        quality_attempt_index=orm.quality_attempt_index,
        selected_route_candidate=cand,
        explanation=orm.explanation,
        created_at=orm.created_at,
    )


def shot_quality_selection_to_orm(
    selection: ShotQualitySelection,
) -> ShotQualitySelectionORM:
    """Converts a domain ShotQualitySelection to a ShotQualitySelectionORM entity."""
    return ShotQualitySelectionORM(
        quality_selection_id=selection.quality_selection_id,
        quality_chain_id=selection.quality_chain_id,
        shot_id=selection.shot_id,
        shot_revision_id=selection.shot_revision_id,
        shot_asset_version_id=selection.shot_asset_version_id,
        evaluation_snapshot_id=selection.evaluation_snapshot_id,
        selection_source=selection.selection_source,
        created_at=selection.created_at,
    )


def shot_quality_selection_from_orm(
    orm: ShotQualitySelectionORM,
) -> ShotQualitySelection:
    """Reconstructs a domain ShotQualitySelection from a ShotQualitySelectionORM entity."""
    from app.domain.quality_remediation import ShotQualitySelection

    return ShotQualitySelection(
        quality_selection_id=orm.quality_selection_id,
        quality_chain_id=orm.quality_chain_id,
        shot_id=orm.shot_id,
        shot_revision_id=orm.shot_revision_id,
        shot_asset_version_id=orm.shot_asset_version_id,
        evaluation_snapshot_id=orm.evaluation_snapshot_id,
        selection_source=orm.selection_source,
        created_at=orm.created_at,
    )


def trace_root_to_orm(root: TraceRoot) -> TraceRootORM:
    """Converts a domain TraceRoot to a TraceRootORM entity."""
    return TraceRootORM(
        trace_id=root.trace_id,
        root_type=root.root_type,
        root_reference_id=root.root_reference_id,
        metadata_version=root.metadata_version,
        created_at=root.created_at,
    )


def trace_root_from_orm(orm: TraceRootORM) -> TraceRoot:
    """Reconstructs a domain TraceRoot from a TraceRootORM entity."""
    from app.domain.trace import TraceRoot

    return TraceRoot(
        trace_id=orm.trace_id,
        root_type=orm.root_type,
        root_reference_id=orm.root_reference_id,
        metadata_version=orm.metadata_version,
        created_at=orm.created_at,
    )


def trace_event_to_orm(event: TraceEvent) -> TraceEventORM:
    """Converts a domain TraceEvent to a TraceEventORM entity."""
    ctx_dict = event.context.model_dump(mode="json")
    return TraceEventORM(
        trace_event_id=event.trace_event_id,
        trace_id=event.trace_id,
        parent_event_id=event.parent_event_id,
        event_type=event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
        event_version=event.event_version,
        status=event.status.value if hasattr(event.status, "value") else str(event.status),
        started_at=event.started_at,
        completed_at=event.completed_at,
        duration_ms=event.duration_ms,
        context_json=ctx_dict,
        attributes_json=event.attributes,
        error_code=event.error_code,
        execution_run_id=event.context.execution_run_id,
        shot_id=event.context.shot_id,
        created_at=event.started_at,
    )


def trace_event_from_orm(orm: TraceEventORM) -> TraceEvent:
    """Reconstructs a domain TraceEvent from a TraceEventORM entity."""
    from app.domain.trace import (
        TraceContext,
        TraceEvent,
        TraceEventStatus,
        TraceEventType,
    )

    ctx = TraceContext.model_validate(orm.context_json)
    return TraceEvent(
        trace_event_id=orm.trace_event_id,
        trace_id=orm.trace_id,
        parent_event_id=orm.parent_event_id,
        event_type=TraceEventType(orm.event_type),
        event_version=orm.event_version,
        status=TraceEventStatus(orm.status),
        started_at=orm.started_at,
        completed_at=orm.completed_at,
        duration_ms=orm.duration_ms,
        context=ctx,
        attributes=orm.attributes_json or {},
        error_code=orm.error_code,
    )


def benchmark_run_to_orm(run: BenchmarkRun) -> BenchmarkRunORM:
    """Converts a domain BenchmarkRun to a BenchmarkRunORM entity."""
    from app.persistence.models import BenchmarkRunORM

    return BenchmarkRunORM(
        benchmark_run_id=run.benchmark_run_id,
        benchmark_suite_id=run.benchmark_suite_id,
        suite_key=run.suite_key,
        suite_version=run.suite_version,
        suite_fingerprint=run.suite_fingerprint,
        variant_json=run.variant.to_dict(),
        execution_mode=run.execution_mode.value,
        status=run.status.value,
        started_at=run.started_at,
        finished_at=run.finished_at,
        application_version=run.application_version,
        git_commit=run.git_commit,
        total_cases=run.total_cases,
        completed_cases=run.completed_cases,
        failed_cases=run.failed_cases,
        created_at=run.started_at,
    )


def benchmark_run_from_orm(orm: BenchmarkRunORM) -> BenchmarkRun:
    """Reconstructs a domain BenchmarkRun from a BenchmarkRunORM entity."""
    from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
    from app.domain.benchmark import (
        BenchmarkExecutionMode,
        BenchmarkRun,
        BenchmarkRunStatus,
        BenchmarkVariant,
    )

    v_data = orm.variant_json or {}
    variant = BenchmarkVariant(
        variant_key=v_data.get("variant_key", "default"),
        routing_strategy=RoutingStrategy(v_data.get("routing_strategy", RoutingStrategy.BALANCED.value)),
        model_selection_mode=ModelSelectionMode(v_data.get("model_selection_mode", ModelSelectionMode.AUTO.value)),
        configured_provider=v_data.get("configured_provider"),
        configured_model=v_data.get("configured_model"),
        evaluation_policy_version=v_data.get("evaluation_policy_version", "v1.0"),
        reuse_mode=v_data.get("reuse_mode", "DEFAULT"),
        application_metadata=v_data.get("application_metadata", {}),
    )

    return BenchmarkRun(
        benchmark_run_id=orm.benchmark_run_id,
        benchmark_suite_id=orm.benchmark_suite_id,
        suite_key=orm.suite_key,
        suite_version=orm.suite_version,
        suite_fingerprint=orm.suite_fingerprint,
        variant=variant,
        execution_mode=BenchmarkExecutionMode(orm.execution_mode),
        status=BenchmarkRunStatus(orm.status),
        started_at=orm.started_at,
        finished_at=orm.finished_at,
        application_version=orm.application_version,
        git_commit=orm.git_commit,
        total_cases=orm.total_cases,
        completed_cases=orm.completed_cases,
        failed_cases=orm.failed_cases,
    )


def benchmark_case_result_to_orm(result: BenchmarkCaseResult) -> BenchmarkCaseResultORM:
    """Converts a domain BenchmarkCaseResult to a BenchmarkCaseResultORM entity."""
    from app.persistence.models import BenchmarkCaseResultORM

    return BenchmarkCaseResultORM(
        benchmark_case_result_id=result.benchmark_case_result_id,
        benchmark_run_id=result.benchmark_run_id,
        benchmark_case_id=result.benchmark_case_id,
        case_key=result.case_key,
        case_version=result.case_version,
        case_fingerprint=result.case_fingerprint,
        trace_id=result.trace_id,
        status=result.status.value,
        started_at=result.started_at,
        finished_at=result.finished_at,
        duration_ms=result.duration_ms,
        failure_stage=result.failure_stage.value if result.failure_stage else None,
        failure_code=result.failure_code,
        failure_message=result.failure_message,
        production_result_refs_json=result.production_result_refs.to_dict(),
        attributes_json=result.attributes,
        created_at=result.started_at,
    )


def benchmark_case_result_from_orm(orm: BenchmarkCaseResultORM) -> BenchmarkCaseResult:
    """Reconstructs a domain BenchmarkCaseResult from a BenchmarkCaseResultORM entity."""
    from app.domain.benchmark import (
        BenchmarkCaseResult,
        BenchmarkCaseStatus,
        BenchmarkFailureStage,
        BenchmarkProductionResultRefs,
    )

    refs_data = orm.production_result_refs_json or {}
    refs = BenchmarkProductionResultRefs(
        content_plan_revision_id=refs_data.get("content_plan_revision_id"),
        approved_storyboard_snapshot_id=refs_data.get("approved_storyboard_snapshot_id"),
        asset_route_plan_id=refs_data.get("asset_route_plan_id"),
        execution_run_id=refs_data.get("execution_run_id"),
        accepted_shot_asset_version_ids=tuple(refs_data.get("accepted_shot_asset_version_ids", ())),
        final_evaluation_snapshot_ids=tuple(refs_data.get("final_evaluation_snapshot_ids", ())),
        remediation_decision_ids=tuple(refs_data.get("remediation_decision_ids", ())),
    )

    f_stage = BenchmarkFailureStage(orm.failure_stage) if orm.failure_stage else None

    return BenchmarkCaseResult(
        benchmark_case_result_id=orm.benchmark_case_result_id,
        benchmark_run_id=orm.benchmark_run_id,
        benchmark_case_id=orm.benchmark_case_id,
        case_key=orm.case_key,
        case_version=orm.case_version,
        case_fingerprint=orm.case_fingerprint,
        trace_id=orm.trace_id,
        status=BenchmarkCaseStatus(orm.status),
        started_at=orm.started_at,
        finished_at=orm.finished_at,
        duration_ms=orm.duration_ms,
        failure_stage=f_stage,
        failure_code=orm.failure_code,
        failure_message=orm.failure_message,
        production_result_refs=refs,
        attributes=orm.attributes_json or {},
    )


def benchmark_report_to_orm(report: BenchmarkReport) -> BenchmarkReportORM:
    """Converts a domain BenchmarkReport to a BenchmarkReportORM record."""
    from app.persistence.models import BenchmarkReportORM

    return BenchmarkReportORM(
        report_id=report.report_id,
        benchmark_run_id=report.benchmark_run_id,
        suite_key=report.suite_key,
        suite_version=report.suite_version,
        suite_fingerprint=report.suite_fingerprint,
        variant_key=report.variant.variant_key,
        variant_json=report.variant.to_dict(),
        execution_mode=report.execution_mode.value,
        metric_definition_set_version=report.metric_definition_set_version,
        source_fingerprint=report.source_fingerprint,
        metrics_json={k: v.to_dict() for k, v in report.metrics.items()},
        stage_latencies_json={k: v.to_dict() for k, v in report.stage_latencies.items()},
        cost_summary_json=report.cost_summary.to_dict(),
        provider_model_usage_json=report.provider_model_usage,
        case_count=report.case_count,
        shot_count=report.shot_count,
        generated_at=report.generated_at,
    )


def benchmark_report_from_orm(orm: BenchmarkReportORM) -> BenchmarkReport:
    """Reconstructs a domain BenchmarkReport from a BenchmarkReportORM record."""
    from app.domain.asset_router import ModelSelectionMode, RoutingStrategy
    from app.domain.benchmark import BenchmarkExecutionMode, BenchmarkVariant
    from app.domain.benchmark_metrics import (
        CostSummary,
        MetricValue,
        MetricValueStatus,
        StageLatencySummary,
    )
    from app.domain.benchmark_report import BenchmarkReport

    v_json = orm.variant_json or {}
    variant = BenchmarkVariant(
        variant_key=v_json.get("variant_key", orm.variant_key),
        routing_strategy=RoutingStrategy(v_json.get("routing_strategy", RoutingStrategy.BALANCED.value)),
        model_selection_mode=ModelSelectionMode(v_json.get("model_selection_mode", ModelSelectionMode.AUTO.value)),
        configured_provider=v_json.get("configured_provider"),
        configured_model=v_json.get("configured_model"),
        evaluation_policy_version=v_json.get("evaluation_policy_version", "v1.0"),
        reuse_mode=v_json.get("reuse_mode", "DEFAULT"),
        application_metadata=v_json.get("application_metadata", {}),
    )

    metrics: dict[str, MetricValue] = {}
    for k, m_dict in (orm.metrics_json or {}).items():
        metrics[k] = MetricValue(
            metric_key=m_dict.get("metric_key", k),
            metric_version=m_dict.get("metric_version", orm.metric_definition_set_version),
            status=MetricValueStatus(m_dict.get("status", "AVAILABLE")),
            value=m_dict.get("value"),
            sample_count=m_dict.get("sample_count", 0),
            eligible_count=m_dict.get("eligible_count", 0),
            coverage_ratio=m_dict.get("coverage_ratio"),
            notes=m_dict.get("notes"),
        )

    latencies: dict[str, StageLatencySummary] = {}
    for k, l_dict in (orm.stage_latencies_json or {}).items():
        latencies[k] = StageLatencySummary(
            stage=l_dict.get("stage", k),
            mean_ms=l_dict.get("mean_ms"),
            p50_ms=l_dict.get("p50_ms"),
            min_ms=l_dict.get("min_ms"),
            max_ms=l_dict.get("max_ms"),
            sample_count=l_dict.get("sample_count", 0),
        )

    c_dict = orm.cost_summary_json or {}
    cost_summary = CostSummary(
        observed_by_currency=c_dict.get("observed_by_currency", {}),
        estimated_by_currency=c_dict.get("estimated_by_currency", {}),
        observed_ops=c_dict.get("observed_ops", 0),
        estimated_ops=c_dict.get("estimated_ops", 0),
        unknown_ops=c_dict.get("unknown_ops", 0),
        completeness=c_dict.get("completeness", "UNKNOWN"),
        is_synthetic=c_dict.get("is_synthetic", False),
    )

    return BenchmarkReport(
        report_id=orm.report_id,
        benchmark_run_id=orm.benchmark_run_id,
        suite_key=orm.suite_key,
        suite_version=orm.suite_version,
        suite_fingerprint=orm.suite_fingerprint,
        variant=variant,
        execution_mode=BenchmarkExecutionMode(orm.execution_mode),
        metrics=metrics,
        stage_latencies=latencies,
        cost_summary=cost_summary,
        provider_model_usage=orm.provider_model_usage_json or {},
        case_count=orm.case_count,
        shot_count=orm.shot_count,
        metric_definition_set_version=orm.metric_definition_set_version,
        generated_at=orm.generated_at,
        source_fingerprint=orm.source_fingerprint,
    )


def benchmark_comparison_report_to_orm(
    comp: BenchmarkComparisonReport,
) -> BenchmarkComparisonReportORM:
    """Converts a domain BenchmarkComparisonReport to an ORM record."""
    from app.persistence.models import BenchmarkComparisonReportORM

    return BenchmarkComparisonReportORM(
        comparison_id=comp.comparison_id,
        baseline_report_id=comp.baseline_report_id,
        candidate_report_id=comp.candidate_report_id,
        suite_key=comp.suite_key,
        suite_version=comp.suite_version,
        suite_fingerprint=comp.suite_fingerprint,
        baseline_variant_key=comp.baseline_variant_key,
        candidate_variant_key=comp.candidate_variant_key,
        execution_mode=comp.execution_mode.value,
        metric_definition_set_version=comp.metric_definition_set_version,
        source_fingerprint=comp.source_fingerprint,
        metric_deltas_json={k: v.to_dict() for k, v in comp.metric_deltas.items()},
        latency_deltas_json={k: v.to_dict() for k, v in comp.latency_deltas.items()},
        cost_deltas_json=comp.cost_deltas,
        tradeoff_summary_json=comp.tradeoff_summary,
        case_drilldowns_json=[c.to_dict() for c in comp.case_drilldowns],
        failure_breakdown_json=comp.failure_breakdown,
        created_at=comp.created_at,
    )


def benchmark_comparison_report_from_orm(
    orm: BenchmarkComparisonReportORM,
) -> BenchmarkComparisonReport:
    """Reconstructs a domain BenchmarkComparisonReport from an ORM record."""
    from app.domain.benchmark import BenchmarkExecutionMode
    from app.domain.benchmark_comparison import (
        BenchmarkComparisonReport,
        CaseDrilldownDelta,
        LatencyDelta,
        MetricDelta,
    )
    from app.domain.benchmark_metrics import MetricDirection

    deltas: dict[str, MetricDelta] = {}
    for k, d_dict in (orm.metric_deltas_json or {}).items():
        deltas[k] = MetricDelta(
            metric_key=d_dict.get("metric_key", k),
            metric_version=d_dict.get("metric_version", orm.metric_definition_set_version),
            baseline_value=d_dict.get("baseline_value"),
            candidate_value=d_dict.get("candidate_value"),
            absolute_delta=d_dict.get("absolute_delta"),
            relative_delta=d_dict.get("relative_delta"),
            direction=MetricDirection(d_dict.get("direction", MetricDirection.NEUTRAL.value)),
            is_improvement=d_dict.get("is_improvement"),
            status=d_dict.get("status", "COMPARED"),
            coverage_warning=d_dict.get("coverage_warning"),
        )

    lat_deltas: dict[str, LatencyDelta] = {}
    for k, l_dict in (orm.latency_deltas_json or {}).items():
        lat_deltas[k] = LatencyDelta(
            stage=l_dict.get("stage", k),
            baseline_mean_ms=l_dict.get("baseline_mean_ms"),
            candidate_mean_ms=l_dict.get("candidate_mean_ms"),
            absolute_delta_ms=l_dict.get("absolute_delta_ms"),
            relative_delta=l_dict.get("relative_delta"),
            is_improvement=l_dict.get("is_improvement"),
        )

    drilldowns: list[CaseDrilldownDelta] = []
    for cd in (orm.case_drilldowns_json or []):
        drilldowns.append(
            CaseDrilldownDelta(
                case_key=cd.get("case_key", ""),
                baseline_trace_id=cd.get("baseline_trace_id", ""),
                candidate_trace_id=cd.get("candidate_trace_id", ""),
                baseline_status=cd.get("baseline_status", "UNKNOWN"),
                candidate_status=cd.get("candidate_status", "UNKNOWN"),
                baseline_duration_ms=cd.get("baseline_duration_ms"),
                candidate_duration_ms=cd.get("candidate_duration_ms"),
                duration_delta_ms=cd.get("duration_delta_ms"),
                baseline_failure_stage=cd.get("baseline_failure_stage"),
                candidate_failure_stage=cd.get("candidate_failure_stage"),
                baseline_failure_code=cd.get("baseline_failure_code"),
                candidate_failure_code=cd.get("candidate_failure_code"),
            )
        )

    return BenchmarkComparisonReport(
        comparison_id=orm.comparison_id,
        baseline_report_id=orm.baseline_report_id,
        candidate_report_id=orm.candidate_report_id,
        baseline_variant_key=orm.baseline_variant_key,
        candidate_variant_key=orm.candidate_variant_key,
        suite_key=orm.suite_key,
        suite_version=orm.suite_version,
        suite_fingerprint=orm.suite_fingerprint,
        execution_mode=BenchmarkExecutionMode(orm.execution_mode),
        metric_deltas=deltas,
        latency_deltas=lat_deltas,
        cost_deltas=orm.cost_deltas_json or {},
        tradeoff_summary=orm.tradeoff_summary_json or {},
        case_drilldowns=drilldowns,
        failure_breakdown=orm.failure_breakdown_json or {},
        metric_definition_set_version=orm.metric_definition_set_version,
        created_at=orm.created_at,
        source_fingerprint=orm.source_fingerprint,
    )


def knowledge_video_task_to_orm(task: KnowledgeVideoTask) -> KnowledgeVideoTaskORM:
    """Converts a domain KnowledgeVideoTask to a KnowledgeVideoTaskORM record."""
    return KnowledgeVideoTaskORM(
        task_id=task.task_id,
        topic=task.topic,
        task_status=task.task_status.value if hasattr(task.task_status, "value") else str(task.task_status),
        current_stage=task.current_stage.value if hasattr(task.current_stage, "value") else str(task.current_stage),
        workflow_policy=task.workflow_policy.value if hasattr(task.workflow_policy, "value") else str(task.workflow_policy),
        target_duration=task.target_duration,
        aspect_ratio=task.aspect_ratio,
        language=task.language,
        waiting_reason=task.waiting_reason,
        error_type=task.error_type.value if (task.error_type and hasattr(task.error_type, "value")) else (str(task.error_type) if task.error_type else None),
        error_message=task.error_message,
        metadata_json=task.task_metadata,
        created_at=task.created_at,
        updated_at=task.updated_at,
        finished_at=task.finished_at,
    )


def knowledge_video_task_from_orm(orm: KnowledgeVideoTaskORM) -> KnowledgeVideoTask:
    """Reconstructs a domain KnowledgeVideoTask from a KnowledgeVideoTaskORM record."""
    error_type = JobErrorType(orm.error_type) if orm.error_type else None
    return KnowledgeVideoTask(
        task_id=orm.task_id,
        topic=orm.topic,
        task_status=TaskStatus(orm.task_status),
        current_stage=Stage(orm.current_stage),
        workflow_policy=WorkflowPolicyType(orm.workflow_policy),
        target_duration=orm.target_duration,
        aspect_ratio=orm.aspect_ratio,
        language=orm.language,
        waiting_reason=orm.waiting_reason,
        error_type=error_type,
        error_message=orm.error_message,
        task_metadata=orm.metadata_json or {},
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        finished_at=orm.finished_at,
    )


def workflow_job_to_orm(job: WorkflowJob) -> WorkflowJobORM:
    """Converts a domain WorkflowJob to a WorkflowJobORM record."""
    return WorkflowJobORM(
        job_id=job.job_id,
        task_id=job.task_id,
        stage=job.stage.value if hasattr(job.stage, "value") else str(job.stage),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        idempotency_key=job.idempotency_key,
        attempt_number=job.attempt_number,
        max_attempts=job.max_attempts,
        available_at=job.available_at,
        lease_owner=job.lease_owner,
        lease_expires_at=job.lease_expires_at,
        heartbeat_at=job.heartbeat_at,
        input_artifact_revision_id=job.input_artifact_revision_id,
        output_artifact_revision_id=job.output_artifact_revision_id,
        input_task_artifact_ref_id=job.input_task_artifact_ref_id or job.input_artifact_revision_id,
        output_task_artifact_ref_id=job.output_task_artifact_ref_id or job.output_artifact_revision_id,
        error_type=job.error_type,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def workflow_job_from_orm(orm: WorkflowJobORM) -> WorkflowJob:
    """Reconstructs a domain WorkflowJob from a WorkflowJobORM record."""
    ref_in = orm.input_task_artifact_ref_id or orm.input_artifact_revision_id
    ref_out = orm.output_task_artifact_ref_id or orm.output_artifact_revision_id
    return WorkflowJob(
        job_id=orm.job_id,
        task_id=orm.task_id,
        stage=Stage(orm.stage),
        status=JobStatus(orm.status),
        idempotency_key=orm.idempotency_key,
        attempt_number=orm.attempt_number,
        max_attempts=orm.max_attempts,
        available_at=orm.available_at,
        lease_owner=orm.lease_owner,
        lease_expires_at=orm.lease_expires_at,
        heartbeat_at=orm.heartbeat_at,
        input_artifact_revision_id=ref_in,
        output_artifact_revision_id=ref_out,
        input_task_artifact_ref_id=ref_in,
        output_task_artifact_ref_id=ref_out,
        error_type=orm.error_type,
        error_message=orm.error_message,
        created_at=orm.created_at,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def stage_execution_to_orm(execution: StageExecution) -> StageExecutionORM:
    """Converts a domain StageExecution to a StageExecutionORM record."""
    return StageExecutionORM(
        stage_execution_id=execution.stage_execution_id,
        task_id=execution.task_id,
        job_id=execution.job_id,
        stage=execution.stage.value if hasattr(execution.stage, "value") else str(execution.stage),
        attempt_number=execution.attempt_number,
        status=execution.status,
        input_artifact_revision_id=execution.input_artifact_revision_id,
        output_artifact_revision_id=execution.output_artifact_revision_id,
        input_task_artifact_ref_id=execution.input_task_artifact_ref_id or execution.input_artifact_revision_id,
        output_task_artifact_ref_id=execution.output_task_artifact_ref_id or execution.output_artifact_revision_id,
        error_type=execution.error_type,
        error_message=execution.error_message,
        duration_ms=execution.duration_ms,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


def stage_execution_from_orm(orm: StageExecutionORM) -> StageExecution:
    """Reconstructs a domain StageExecution from a StageExecutionORM record."""
    ref_in = orm.input_task_artifact_ref_id or orm.input_artifact_revision_id
    ref_out = orm.output_task_artifact_ref_id or orm.output_artifact_revision_id
    return StageExecution(
        stage_execution_id=orm.stage_execution_id,
        task_id=orm.task_id,
        job_id=orm.job_id,
        stage=Stage(orm.stage),
        attempt_number=orm.attempt_number,
        status=orm.status,
        input_artifact_revision_id=ref_in,
        output_artifact_revision_id=ref_out,
        input_task_artifact_ref_id=ref_in,
        output_task_artifact_ref_id=ref_out,
        error_type=orm.error_type,
        error_message=orm.error_message,
        duration_ms=orm.duration_ms,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def task_artifact_ref_to_orm(ref: TaskArtifactRef) -> TaskArtifactRefORM:
    """Converts a domain TaskArtifactRef to a TaskArtifactRefORM record."""
    return TaskArtifactRefORM(
        task_artifact_ref_id=ref.task_artifact_ref_id,
        task_id=ref.task_id,
        stage=ref.stage.value if hasattr(ref.stage, "value") else str(ref.stage),
        artifact_type=ref.artifact_type.value if hasattr(ref.artifact_type, "value") else str(ref.artifact_type),
        artifact_id=ref.artifact_id,
        artifact_version=ref.artifact_version,
        metadata_json=ref.metadata_json,
        created_at=ref.created_at,
    )


def task_artifact_ref_from_orm(orm: TaskArtifactRefORM) -> TaskArtifactRef:
    """Reconstructs a domain TaskArtifactRef from a TaskArtifactRefORM record."""
    return TaskArtifactRef(
        task_artifact_ref_id=orm.task_artifact_ref_id,
        task_id=orm.task_id,
        stage=Stage(orm.stage),
        artifact_type=ArtifactType(orm.artifact_type),
        artifact_id=orm.artifact_id,
        artifact_version=orm.artifact_version,
        created_at=orm.created_at,
        metadata_json=orm.metadata_json or {},
    )


def source_document_to_orm(doc: Any) -> Any:
    """Converts a domain SourceDocument to a SourceDocumentORM record."""
    from app.persistence.models import SourceDocumentORM

    return SourceDocumentORM(
        source_document_id=doc.source_document_id,
        source_type=doc.source_type.value if hasattr(doc.source_type, "value") else str(doc.source_type),
        title=doc.title,
        source_locator=doc.source_locator,
        content_snapshot=doc.content_snapshot,
        content_hash=doc.content_hash,
        source_fingerprint=doc.source_fingerprint,
        author=doc.author,
        published_at=doc.published_at,
        captured_at=doc.captured_at,
        media_type=doc.media_type,
        status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
        metadata_json=doc.metadata_json,
        created_at=doc.created_at,
    )


def source_document_from_orm(orm: Any) -> Any:
    """Reconstructs a domain SourceDocument from a SourceDocumentORM record."""
    from app.domain.evidence import SourceDocument, SourceStatus, SourceType

    return SourceDocument(
        source_document_id=orm.source_document_id,
        source_type=SourceType(orm.source_type),
        title=orm.title,
        source_locator=orm.source_locator,
        content_snapshot=orm.content_snapshot,
        content_hash=orm.content_hash,
        source_fingerprint=orm.source_fingerprint,
        author=orm.author,
        published_at=orm.published_at,
        captured_at=orm.captured_at,
        media_type=orm.media_type,
        status=SourceStatus(orm.status),
        metadata_json=orm.metadata_json or {},
        created_at=orm.created_at,
    )


def evidence_item_to_orm(item: Any) -> Any:
    """Converts a domain EvidenceItem to an EvidenceItemORM record."""
    from app.persistence.models import EvidenceItemORM

    return EvidenceItemORM(
        evidence_id=item.evidence_id,
        source_document_id=item.source_document_id,
        locator_json=item.locator,
        original_excerpt=item.original_excerpt,
        normalized_fact=item.normalized_fact,
        evidence_role=item.evidence_role.value if hasattr(item.evidence_role, "value") else str(item.evidence_role),
        confidence=item.confidence,
        extraction_method=item.extraction_method,
        content_hash=item.content_hash,
        created_at=item.created_at,
    )


def evidence_item_from_orm(orm: Any) -> Any:
    """Reconstructs a domain EvidenceItem from an EvidenceItemORM record."""
    from app.domain.evidence import EvidenceItem, EvidenceRole

    return EvidenceItem(
        evidence_id=orm.evidence_id,
        source_document_id=orm.source_document_id,
        locator=orm.locator_json or {},
        original_excerpt=orm.original_excerpt,
        normalized_fact=orm.normalized_fact,
        evidence_role=EvidenceRole(orm.evidence_role),
        confidence=orm.confidence,
        extraction_method=orm.extraction_method,
        content_hash=orm.content_hash,
        created_at=orm.created_at,
    )


def knowledge_claim_to_orm(claim: Any) -> Any:
    """Converts a domain KnowledgeClaim to a KnowledgeClaimORM record."""
    from app.persistence.models import KnowledgeClaimORM

    return KnowledgeClaimORM(
        knowledge_claim_id=claim.knowledge_claim_id,
        claim_type=claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type),
        claim_text=claim.claim_text,
        evidence_refs_json=list(claim.evidence_refs),
        verification_status=(
            claim.verification_status.value
            if hasattr(claim.verification_status, "value")
            else str(claim.verification_status)
        ),
        conflict_evidence_refs_json=list(claim.conflict_evidence_refs),
        created_at=claim.created_at,
    )


def knowledge_claim_from_orm(orm: Any) -> Any:
    """Reconstructs a domain KnowledgeClaim from a KnowledgeClaimORM record."""
    from app.domain.evidence import ClaimType, KnowledgeClaim, VerificationStatus

    return KnowledgeClaim(
        knowledge_claim_id=orm.knowledge_claim_id,
        claim_type=ClaimType(orm.claim_type),
        claim_text=orm.claim_text,
        evidence_refs=tuple(orm.evidence_refs_json or []),
        verification_status=VerificationStatus(orm.verification_status),
        conflict_evidence_refs=tuple(orm.conflict_evidence_refs_json or []),
        created_at=orm.created_at,
    )


def evidence_snapshot_to_orm(snapshot: Any) -> Any:
    """Converts a domain EvidenceSnapshot to an EvidenceSnapshotORM record."""
    from app.persistence.models import EvidenceSnapshotORM

    return EvidenceSnapshotORM(
        evidence_snapshot_id=snapshot.evidence_snapshot_id,
        task_id=snapshot.task_id,
        snapshot_version=snapshot.snapshot_version,
        source_document_ids_json=list(snapshot.source_document_ids),
        evidence_ids_json=list(snapshot.evidence_ids),
        knowledge_claim_ids_json=list(snapshot.knowledge_claim_ids),
        content_fingerprint=snapshot.content_fingerprint,
        created_at=snapshot.created_at,
    )


def evidence_snapshot_from_orm(orm: Any) -> Any:
    """Reconstructs a domain EvidenceSnapshot from an EvidenceSnapshotORM record."""
    from app.domain.evidence import EvidenceSnapshot

    return EvidenceSnapshot(
        evidence_snapshot_id=orm.evidence_snapshot_id,
        task_id=orm.task_id,
        snapshot_version=orm.snapshot_version,
        source_document_ids=tuple(orm.source_document_ids_json or []),
        evidence_ids=tuple(orm.evidence_ids_json or []),
        knowledge_claim_ids=tuple(orm.knowledge_claim_ids_json or []),
        content_fingerprint=orm.content_fingerprint,
        created_at=orm.created_at,
    )


def knowledge_chunk_to_orm(chunk: Any) -> Any:
    """Converts a domain KnowledgeChunk to a KnowledgeChunkORM record."""
    from app.persistence.models import KnowledgeChunkORM

    return KnowledgeChunkORM(
        chunk_id=chunk.chunk_id,
        source_document_id=chunk.source_document_id,
        processing_version=chunk.processing_version,
        chunk_index=chunk.chunk_index,
        normalized_text=chunk.normalized_text,
        text_hash=chunk.text_hash,
        locator_json=chunk.locator,
        content_fingerprint=chunk.content_fingerprint,
        created_at=chunk.created_at,
    )


def knowledge_chunk_from_orm(orm: Any) -> Any:
    """Reconstructs a domain KnowledgeChunk from a KnowledgeChunkORM record."""
    from app.domain.evidence import KnowledgeChunk

    return KnowledgeChunk(
        chunk_id=orm.chunk_id,
        source_document_id=orm.source_document_id,
        processing_version=orm.processing_version,
        chunk_index=orm.chunk_index,
        normalized_text=orm.normalized_text,
        text_hash=orm.text_hash,
        locator=orm.locator_json or {},
        content_fingerprint=orm.content_fingerprint,
        created_at=orm.created_at,
    )


def retrieval_snapshot_to_orm(snapshot: Any) -> Any:
    """Converts a domain RetrievalSnapshot to a RetrievalSnapshotORM record."""
    from app.persistence.models import RetrievalSnapshotORM

    candidates_list = [c.model_dump() if hasattr(c, "model_dump") else c for c in snapshot.candidates]
    return RetrievalSnapshotORM(
        retrieval_snapshot_id=snapshot.retrieval_snapshot_id,
        task_id=snapshot.task_id,
        query=snapshot.query,
        source_scope_ids_json=list(snapshot.source_scope_ids),
        retrieval_policy_version=snapshot.retrieval_policy_version,
        processing_version=snapshot.processing_version,
        candidates_json=candidates_list,
        selected_evidence_ids_json=list(snapshot.selected_evidence_ids),
        content_fingerprint=snapshot.content_fingerprint,
        created_at=snapshot.created_at,
    )


def retrieval_snapshot_from_orm(orm: Any) -> Any:
    """Reconstructs a domain RetrievalSnapshot from a RetrievalSnapshotORM record."""
    from app.domain.evidence import RetrievalCandidate, RetrievalSnapshot

    candidates = [
        RetrievalCandidate.model_validate(c) if isinstance(c, dict) else c
        for c in (orm.candidates_json or [])
    ]
    return RetrievalSnapshot(
        retrieval_snapshot_id=orm.retrieval_snapshot_id,
        task_id=orm.task_id,
        query=orm.query,
        source_scope_ids=tuple(orm.source_scope_ids_json or []),
        retrieval_policy_version=orm.retrieval_policy_version,
        processing_version=orm.processing_version,
        candidates=tuple(candidates),
        selected_evidence_ids=tuple(orm.selected_evidence_ids_json or []),
        content_fingerprint=orm.content_fingerprint,
        created_at=orm.created_at,
    )


def web_research_snapshot_to_orm(snapshot: Any) -> Any:
    """Converts a domain WebResearchSnapshot to a WebResearchSnapshotORM record."""
    from app.persistence.models import WebResearchSnapshotORM

    results_list = [
        r.model_dump() if hasattr(r, "model_dump") else r
        for r in snapshot.search_results
    ]
    return WebResearchSnapshotORM(
        web_research_snapshot_id=snapshot.web_research_snapshot_id,
        task_id=snapshot.task_id,
        stage_attempt=snapshot.stage_attempt,
        query=snapshot.query,
        provider=snapshot.provider,
        search_results_json=results_list,
        selected_urls_json=list(snapshot.selected_urls),
        fetch_outcomes_json=list(snapshot.fetch_outcomes),
        created_source_document_ids_json=list(snapshot.created_source_document_ids),
        content_fingerprint=snapshot.content_fingerprint,
        created_at=snapshot.created_at,
    )


def web_research_snapshot_from_orm(orm: Any) -> Any:
    """Reconstructs a domain WebResearchSnapshot from a WebResearchSnapshotORM record."""
    from app.domain.evidence import SearchResult, WebResearchSnapshot

    search_results = [
        SearchResult.model_validate(r) if isinstance(r, dict) else r
        for r in (orm.search_results_json or [])
    ]
    return WebResearchSnapshot(
        web_research_snapshot_id=orm.web_research_snapshot_id,
        task_id=orm.task_id,
        stage_attempt=orm.stage_attempt,
        query=orm.query,
        provider=orm.provider,
        search_results=tuple(search_results),
        selected_urls=tuple(orm.selected_urls_json or []),
        fetch_outcomes=tuple(orm.fetch_outcomes_json or []),
        created_source_document_ids=tuple(orm.created_source_document_ids_json or []),
        content_fingerprint=orm.content_fingerprint,
        created_at=orm.created_at,
    )


def script_segment_to_orm(segment: Any) -> ScriptSegmentORM:
    """Converts a domain ScriptSegment into a ScriptSegmentORM record."""
    from app.domain.script import ScriptSegment

    return ScriptSegmentORM(
        script_segment_id=segment.script_segment_id,
        script_revision_id=segment.script_revision_id,
        content_beat_id=segment.content_beat_id,
        beat_lineage_id=segment.beat_lineage_id,
        order=segment.order,
        narration_text=segment.narration_text,
        target_duration=segment.target_duration,
        beat_type=segment.beat_type.value if segment.beat_type else None,
        evidence_refs=list(segment.evidence_refs),
        created_at=segment.created_at,
    )


def script_segment_from_orm(orm: ScriptSegmentORM) -> Any:
    """Reconstructs a domain ScriptSegment from a ScriptSegmentORM record."""
    from app.domain.script import ScriptSegment

    return ScriptSegment(
        script_segment_id=orm.script_segment_id,
        script_revision_id=orm.script_revision_id,
        content_beat_id=orm.content_beat_id,
        beat_lineage_id=orm.beat_lineage_id,
        order=orm.order,
        narration_text=orm.narration_text,
        target_duration=orm.target_duration,
        beat_type=BeatType(orm.beat_type) if orm.beat_type else None,
        evidence_refs=tuple(orm.evidence_refs or []),
        created_at=orm.created_at,
    )


def script_revision_to_orm(revision: Any) -> ScriptRevisionORM:
    """Converts a domain ScriptRevision into a ScriptRevisionORM record."""
    from app.domain.script import ScriptRevision

    segments_orm = [script_segment_to_orm(seg) for seg in revision.segments]
    return ScriptRevisionORM(
        script_revision_id=revision.script_revision_id,
        task_id=revision.task_id,
        content_plan_revision_id=revision.content_plan_revision_id,
        revision_number=revision.revision_number,
        overall_target_duration=revision.overall_target_duration,
        language=revision.language,
        content_fingerprint=revision.content_fingerprint,
        created_at=revision.created_at,
        segments=segments_orm,
    )


def script_revision_from_orm(orm: ScriptRevisionORM) -> Any:
    """Reconstructs a domain ScriptRevision from a ScriptRevisionORM record."""
    from app.domain.script import ScriptRevision

    segments = (
        tuple(
            script_segment_from_orm(seg_orm)
            for seg_orm in sorted(orm.segments, key=lambda s: s.order)
        )
        if orm.segments
        else ()
    )
    return ScriptRevision(
        script_revision_id=orm.script_revision_id,
        task_id=orm.task_id,
        content_plan_revision_id=orm.content_plan_revision_id,
        revision_number=orm.revision_number,
        overall_target_duration=orm.overall_target_duration,
        language=orm.language,
        content_fingerprint=orm.content_fingerprint,
        created_at=orm.created_at,
        segments=segments,
    )
