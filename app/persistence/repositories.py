from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.domain.asset_execution import (
    AttemptRequest,
    AttemptResult,
    ExecutionAttempt,
    ExecutionRun,
    ExecutionStatus,
    ExecutionTransition,
    ProviderReceipt,
    ShotAssetVersion,
    ShotExecution,
)
from app.domain.asset_router import AssetRoutePlan
from app.domain.benchmark import BenchmarkCaseResult, BenchmarkRun
from app.domain.content_plan import ContentPlanRevision
from app.domain.evaluation import (
    DimensionEvaluationResult,
    EvaluationSnapshot,
    EvaluationTarget,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.plan_diff import validate_content_plan_revision_lineages
from app.domain.evidence import (
    EvidenceItem,
    EvidenceSnapshot,
    KnowledgeClaim,
    SourceDocument,
    WebResearchSnapshot,
)
from app.domain.quality_remediation import (
    QualityRemediationDecision,
    ShotQualitySelection,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.stage_execution import StageExecution
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.domain.task_artifact import TaskArtifactRef
from app.domain.trace import TraceEvent, TraceEventStatus, TraceRoot
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobStatus,
    Stage,
    WorkflowConflictError,
)
from app.persistence.converters import (
    asset_route_plan_from_orm,
    asset_route_plan_to_orm,
    attempt_request_from_orm,
    attempt_request_to_orm,
    attempt_result_from_orm,
    attempt_result_to_orm,
    composition_preview_from_orm,
    composition_preview_to_orm,
    content_plan_revision_from_orm,
    content_plan_revision_to_orm,
    dimension_evaluation_result_from_orm,
    dimension_evaluation_result_to_orm,
    evaluation_snapshot_from_orm,
    evaluation_snapshot_to_orm,
    evaluation_target_from_orm,
    evaluation_target_to_orm,
    execution_attempt_from_orm,
    execution_attempt_to_orm,
    execution_run_from_orm,
    execution_run_to_orm,
    execution_transition_from_orm,
    execution_transition_to_orm,
    knowledge_video_task_from_orm,
    knowledge_video_task_to_orm,
    provider_receipt_from_orm,
    provider_receipt_to_orm,
    quality_remediation_decision_from_orm,
    quality_remediation_decision_to_orm,
    shot_asset_version_from_orm,
    shot_asset_version_to_orm,
    shot_execution_from_orm,
    shot_execution_to_orm,
    shot_from_orm,
    shot_quality_selection_from_orm,
    shot_quality_selection_to_orm,
    shot_revision_from_orm,
    shot_revision_to_orm,
    shot_to_orm,
    stage_execution_from_orm,
    stage_execution_to_orm,
    storyboard_approval_record_from_orm,
    storyboard_approval_record_to_orm,
    storyboard_snapshot_from_orm,
    storyboard_snapshot_to_orm,
    task_artifact_ref_from_orm,
    task_artifact_ref_to_orm,
    trace_event_from_orm,
    trace_event_to_orm,
    trace_root_from_orm,
    trace_root_to_orm,
    workflow_job_from_orm,
    workflow_job_to_orm,
)
from app.persistence.models import (
    AssetRoutePlanORM,
    AttemptRequestORM,
    AttemptResultORM,
    CompositionPreviewORM,
    ContentPlanRevisionORM,
    DimensionEvaluationResultORM,
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
    StageExecutionORM,
    StoryboardApprovalRecordORM,
    StoryboardSnapshotORM,
    TaskArtifactRefORM,
    TraceEventORM,
    TraceRootORM,
    WorkflowJobORM,
)
from app.services.evaluation.composition_preview import CompositionPreview


class ContentPlanRepository:
    """
    Append-oriented repository for ContentPlanRevisions.

    Historical plan revisions and their content beats are append-only.
    No update methods are exposed to prevent mutating historical revisions.
    """

    def __init__(self, session: Session):
        self._session = session

    def add_revision(self, plan: ContentPlanRevision) -> ContentPlanRevision:
        """
        Atomically persists a new ContentPlanRevision with all its ContentBeats.

        Validates beat_lineage_id uniqueness within the revision prior to insertion.
        """
        validate_content_plan_revision_lineages(plan)
        orm_plan = content_plan_revision_to_orm(plan)
        self._session.add(orm_plan)
        self._session.flush()
        return plan

    def get_revision(self, content_plan_revision_id: str) -> ContentPlanRevision | None:
        """Loads a ContentPlanRevision by its unique revision ID."""
        orm_plan = self._session.get(ContentPlanRevisionORM, content_plan_revision_id)
        if orm_plan is None:
            return None
        return content_plan_revision_from_orm(orm_plan)

    def get_latest_revision(
        self, topic: str | None = None
    ) -> ContentPlanRevision | None:
        """
        Loads the latest revision by revision_number, optionally filtered by topic.
        """
        stmt = select(ContentPlanRevisionORM)
        if topic is not None:
            stmt = stmt.where(ContentPlanRevisionORM.topic == topic)
        stmt = stmt.order_by(ContentPlanRevisionORM.revision_number.desc())
        orm_plan = self._session.scalars(stmt).first()
        if orm_plan is None:
            return None
        return content_plan_revision_from_orm(orm_plan)


class ShotRepository:
    """
    Repository for Shot entities and append-oriented ShotRevisions.

    Shot entity identity is stable; revision contents are immutable and versioned.
    No update methods are exposed to prevent mutating historical revisions.
    """

    def __init__(self, session: Session):
        self._session = session

    def add_shot(self, shot: Shot) -> Shot:
        """Persists a new stable Shot identity."""
        orm_shot = shot_to_orm(shot)
        self._session.add(orm_shot)
        self._session.flush()
        return shot

    def add_revision(self, revision: ShotRevision) -> ShotRevision:
        """
        Persists a new immutable ShotRevision for an existing Shot.

        Enforces uniqueness of (shot_id, revision_number) at the database level.
        """
        orm_rev = shot_revision_to_orm(revision)
        self._session.add(orm_rev)
        self._session.flush()
        return revision

    def get_shot(self, shot_id: str) -> Shot | None:
        """Loads a Shot entity by its ID."""
        orm_shot = self._session.get(ShotORM, shot_id)
        if orm_shot is None:
            return None
        return shot_from_orm(orm_shot)

    def get_revision(self, shot_revision_id: str) -> ShotRevision | None:
        """Loads a specific ShotRevision by its unique revision ID."""
        orm_rev = self._session.get(ShotRevisionORM, shot_revision_id)
        if orm_rev is None:
            return None
        return shot_revision_from_orm(orm_rev)

    def update_shot_local_order(self, shot_id: str, local_order: int) -> Shot | None:
        """Updates the local order of an existing Shot entity within its beat."""
        orm_shot = self._session.get(ShotORM, shot_id)
        if orm_shot is None:
            return None
        orm_shot.local_order = local_order
        self._session.flush()
        return shot_from_orm(orm_shot)

    def list_revisions(self, shot_id: str) -> tuple[ShotRevision, ...]:
        """Lists all revisions for a given Shot, ordered by revision_number ascending."""
        stmt = (
            select(ShotRevisionORM)
            .where(ShotRevisionORM.shot_id == shot_id)
            .order_by(ShotRevisionORM.revision_number.asc())
        )
        results = self._session.scalars(stmt).all()
        return tuple(shot_revision_from_orm(r) for r in results)


class StoryboardRepository:
    """
    Repository for StoryboardSnapshots.

    Snapshots freeze exact ShotRevision references. Reconstruction always resolves
    exact frozen revision IDs, never dynamically through shot.latest_revision.
    """

    def __init__(self, session: Session):
        self._session = session

    def add_snapshot(self, snapshot: StoryboardSnapshot) -> StoryboardSnapshot:
        """
        Atomically persists a StoryboardSnapshot with exact ShotRevision associations.
        """
        orm_snap = storyboard_snapshot_to_orm(snapshot)
        self._session.add(orm_snap)
        self._session.flush()
        return snapshot

    def get_snapshot(self, storyboard_snapshot_id: str) -> StoryboardSnapshot | None:
        """Loads a StoryboardSnapshot by its unique ID."""
        orm_snap = self._session.get(StoryboardSnapshotORM, storyboard_snapshot_id)
        if orm_snap is None:
            return None
        return storyboard_snapshot_from_orm(orm_snap)

    def get_snapshot_shot_revisions(
        self, storyboard_snapshot_id: str
    ) -> tuple[ShotRevision, ...]:
        """
        Reconstructs the exact sequence of ShotRevision domain objects frozen in this snapshot.

        Strictly retrieves by frozen shot_revision_id; does NOT resolve to mutable current revisions.
        """
        snap = self.get_snapshot(storyboard_snapshot_id)
        if snap is None or not snap.shot_revision_ids:
            return ()

        stmt = select(ShotRevisionORM).where(
            ShotRevisionORM.shot_revision_id.in_(snap.shot_revision_ids)
        )
        loaded = {
            r.shot_revision_id: shot_revision_from_orm(r)
            for r in self._session.scalars(stmt).all()
        }

        # Return in the exact frozen sequence from the snapshot
        return tuple(
            loaded[rev_id] for rev_id in snap.shot_revision_ids if rev_id in loaded
        )


class StoryboardApprovalRepository:
    """
    Append-only repository for StoryboardApprovalRecords.

    Guarantees that human approval audit logs are immutable and traceable to both
    the source draft snapshot and the approved snapshot.
    """

    def __init__(self, session: Session):
        self._session = session

    def add_approval_record(
        self, record: StoryboardApprovalRecord
    ) -> StoryboardApprovalRecord:
        """
        Atomically persists a StoryboardApprovalRecord.
        """
        orm_record = storyboard_approval_record_to_orm(record)
        self._session.add(orm_record)
        self._session.flush()
        return record

    def get_approval_record(
        self, approval_id: str
    ) -> StoryboardApprovalRecord | None:
        """Loads an approval record by its unique approval ID."""
        orm_record = self._session.get(StoryboardApprovalRecordORM, approval_id)
        if orm_record is None:
            return None
        return storyboard_approval_record_from_orm(orm_record)

    def get_approval_by_approved_snapshot_id(
        self, approved_snapshot_id: str
    ) -> StoryboardApprovalRecord | None:
        """Finds the approval record for an approved storyboard snapshot."""
        stmt = (
            select(StoryboardApprovalRecordORM)
            .where(
                StoryboardApprovalRecordORM.approved_storyboard_snapshot_id
                == approved_snapshot_id
            )
            .limit(1)
        )
        orm_record = self._session.scalars(stmt).first()
        if orm_record is None:
            return None
        return storyboard_approval_record_from_orm(orm_record)

    def get_approval_by_source_draft_id(
        self, source_draft_snapshot_id: str
    ) -> StoryboardApprovalRecord | None:
        """Finds an approval record originating from a given draft snapshot."""
        stmt = (
            select(StoryboardApprovalRecordORM)
            .where(
                StoryboardApprovalRecordORM.source_draft_snapshot_id
                == source_draft_snapshot_id
            )
            .limit(1)
        )
        orm_record = self._session.scalars(stmt).first()
        if orm_record is None:
            return None
        return storyboard_approval_record_from_orm(orm_record)


class AssetRoutePlanRepository:
    """Repository managing persistence for AssetRoutePlan and child ShotRoutePlanEntry records."""

    def __init__(self, session: Session):
        self._session = session

    def add_route_plan(self, plan: AssetRoutePlan) -> AssetRoutePlan:
        """Atomically persists an AssetRoutePlan with its ordered ShotRoutePlanEntry records."""
        orm_plan = asset_route_plan_to_orm(plan)
        self._session.add(orm_plan)
        self._session.flush()
        return plan

    def get_route_plan(self, plan_id: str) -> AssetRoutePlan | None:
        """Loads an AssetRoutePlan by its unique plan ID, including all child route entries."""
        orm_plan = self._session.get(AssetRoutePlanORM, plan_id)
        if orm_plan is None:
            return None
        return asset_route_plan_from_orm(orm_plan)

    def get_latest_route_plan_for_snapshot(
        self, snapshot_id: str
    ) -> AssetRoutePlan | None:
        """Finds the most recently created AssetRoutePlan for an approved storyboard snapshot."""
        stmt = (
            select(AssetRoutePlanORM)
            .where(AssetRoutePlanORM.storyboard_snapshot_id == snapshot_id)
            .order_by(AssetRoutePlanORM.created_at.desc())
            .limit(1)
        )
        orm_plan = self._session.scalars(stmt).first()
        if orm_plan is None:
            return None
        return asset_route_plan_from_orm(orm_plan)

    def list_route_plans_for_snapshot(
        self, snapshot_id: str
    ) -> list[AssetRoutePlan]:
        """Lists all historical route plans for a storyboard snapshot, newest first."""
        stmt = (
            select(AssetRoutePlanORM)
            .where(AssetRoutePlanORM.storyboard_snapshot_id == snapshot_id)
            .order_by(AssetRoutePlanORM.created_at.desc())
        )
        orm_plans = self._session.scalars(stmt).all()
        return [asset_route_plan_from_orm(p) for p in orm_plans]


class ExecutionRepository:
    """Repository managing persistence for ExecutionRun, ExecutionAttempt, and ShotAssetVersion records."""

    def __init__(self, session: Session):
        self._session = session

    def add_execution_run(self, run: ExecutionRun) -> ExecutionRun:
        """Persists an ExecutionRun."""
        orm_run = execution_run_to_orm(run)
        self._session.add(orm_run)
        self._session.flush()
        return run

    def get_execution_run(self, run_id: str) -> ExecutionRun | None:
        """Loads an ExecutionRun by ID."""
        orm_run = self._session.get(ExecutionRunORM, run_id)
        if orm_run is None:
            return None
        return execution_run_from_orm(orm_run)

    def update_execution_run_status(
        self,
        run_id: str,
        status: ExecutionStatus,
        succeeded_shots: int,
        failed_shots: int,
        finished_at: datetime | None = None,
        reused_shots: int = 0,
        recovery_required_shots: int = 0,
    ) -> ExecutionRun | None:
        """Updates the progress counts and status of an ExecutionRun."""
        orm_run = self._session.get(ExecutionRunORM, run_id)
        if orm_run is None:
            return None
        orm_run.status = status.value
        orm_run.succeeded_shots = succeeded_shots
        orm_run.reused_shots = reused_shots
        orm_run.failed_shots = failed_shots
        orm_run.recovery_required_shots = recovery_required_shots
        if finished_at is not None:
            orm_run.finished_at = finished_at
        self._session.flush()
        return execution_run_from_orm(orm_run)

    def list_runs_for_plan(self, plan_id: str) -> list[ExecutionRun]:
        """Lists all execution runs for a given asset route plan, ordered newest first."""
        stmt = (
            select(ExecutionRunORM)
            .where(ExecutionRunORM.asset_route_plan_id == plan_id)
            .order_by(ExecutionRunORM.started_at.desc())
        )
        orm_runs = self._session.scalars(stmt).all()
        return [execution_run_from_orm(r) for r in orm_runs]

    def add_execution_attempt(self, attempt: ExecutionAttempt) -> ExecutionAttempt:
        """Persists an ExecutionAttempt."""
        orm_attempt = execution_attempt_to_orm(attempt)
        self._session.add(orm_attempt)
        self._session.flush()
        return attempt

    def update_execution_attempt(
        self, attempt: ExecutionAttempt
    ) -> ExecutionAttempt | None:
        """Updates status, timestamps, and error/response details of an ExecutionAttempt."""
        import json

        orm_attempt = self._session.get(
            ExecutionAttemptORM, attempt.execution_attempt_id
        )
        if orm_attempt is None:
            return None
        orm_attempt.status = attempt.status.value
        orm_attempt.finished_at = attempt.finished_at
        orm_attempt.duration_seconds = attempt.duration_seconds
        orm_attempt.error_code = attempt.error_code
        orm_attempt.error_message = attempt.error_message
        if attempt.raw_provider_response is not None:
            orm_attempt.raw_provider_response_json = json.dumps(
                attempt.raw_provider_response, ensure_ascii=False
            )
        self._session.flush()
        return execution_attempt_from_orm(orm_attempt)

    def get_execution_attempt(self, attempt_id: str) -> ExecutionAttempt | None:
        """Loads an ExecutionAttempt by ID."""
        orm_attempt = self._session.get(ExecutionAttemptORM, attempt_id)
        if orm_attempt is None:
            return None
        return execution_attempt_from_orm(orm_attempt)

    def save_or_update_execution_attempt(
        self, attempt: ExecutionAttempt
    ) -> ExecutionAttempt:
        """Persists a new ExecutionAttempt or updates an existing one without duplicate key violations."""
        orm_attempt = self._session.get(
            ExecutionAttemptORM, attempt.execution_attempt_id
        )
        if orm_attempt is not None:
            updated = self.update_execution_attempt(attempt)
            return updated or attempt
        return self.add_execution_attempt(attempt)

    def list_attempts_for_run(self, run_id: str) -> list[ExecutionAttempt]:
        """Lists all execution attempts for a given execution run, ordered by started_at."""
        stmt = (
            select(ExecutionAttemptORM)
            .where(ExecutionAttemptORM.execution_run_id == run_id)
            .order_by(ExecutionAttemptORM.started_at.asc())
        )
        orm_attempts = self._session.scalars(stmt).all()
        return [execution_attempt_from_orm(a) for a in orm_attempts]

    def add_shot_asset_version(self, version: ShotAssetVersion) -> ShotAssetVersion:
        """Persists a new immutable ShotAssetVersion."""
        orm_version = shot_asset_version_to_orm(version)
        self._session.add(orm_version)
        self._session.flush()
        return version

    def get_shot_asset_version(self, version_id: str) -> ShotAssetVersion | None:
        """Loads a ShotAssetVersion by ID."""
        orm_version = self._session.get(ShotAssetVersionORM, version_id)
        if orm_version is None:
            return None
        return shot_asset_version_from_orm(orm_version)

    def list_asset_versions_for_shot_revision(
        self, shot_revision_id: str
    ) -> list[ShotAssetVersion]:
        """Lists all asset versions for a given shot revision, newest first."""
        stmt = (
            select(ShotAssetVersionORM)
            .where(ShotAssetVersionORM.shot_revision_id == shot_revision_id)
            .order_by(ShotAssetVersionORM.created_at.desc())
        )
        orm_versions = self._session.scalars(stmt).all()
        return [shot_asset_version_from_orm(v) for v in orm_versions]

    def get_asset_version_for_run_and_shot(
        self, execution_run_id: str, shot_revision_id: str
    ) -> ShotAssetVersion | None:
        """
        Loads the specific ShotAssetVersion produced or bound to this execution run
        for the specified shot revision, ensuring no cross-run contamination.
        """
        # Strategy 1: Through ShotExecution.produced_asset_version_id
        stmt1 = (
            select(ShotAssetVersionORM)
            .join(
                ShotExecutionORM,
                ShotExecutionORM.produced_asset_version_id == ShotAssetVersionORM.shot_asset_version_id,
            )
            .where(
                ShotExecutionORM.execution_run_id == execution_run_id,
                ShotExecutionORM.shot_revision_id == shot_revision_id,
            )
        )
        orm_asset = self._session.scalars(stmt1).first()
        if orm_asset is not None:
            return shot_asset_version_from_orm(orm_asset)

        # Strategy 2: Direct link through ExecutionAttemptORM.execution_run_id
        stmt2 = (
            select(ShotAssetVersionORM)
            .join(
                ExecutionAttemptORM,
                ExecutionAttemptORM.execution_attempt_id == ShotAssetVersionORM.execution_attempt_id,
            )
            .where(
                ExecutionAttemptORM.execution_run_id == execution_run_id,
                ShotAssetVersionORM.shot_revision_id == shot_revision_id,
            )
            .order_by(ShotAssetVersionORM.created_at.desc())
        )
        orm_asset2 = self._session.scalars(stmt2).first()
        if orm_asset2 is not None:
            return shot_asset_version_from_orm(orm_asset2)

        return None

    def get_latest_terminal_execution_run(
        self, storyboard_snapshot_id: str
    ) -> ExecutionRun | None:
        """Loads the most recent completed or terminal ExecutionRun for the storyboard."""
        stmt = (
            select(ExecutionRunORM)
            .where(
                ExecutionRunORM.storyboard_snapshot_id == storyboard_snapshot_id,
                ExecutionRunORM.status.in_(["COMPLETED", "SUCCEEDED", "PARTIALLY_COMPLETED"]),
            )
            .order_by(ExecutionRunORM.started_at.desc())
        )
        orm_run = self._session.scalars(stmt).first()
        if orm_run is not None:
            return execution_run_from_orm(orm_run)
        return None

    def get_latest_execution_run_for_storyboard(
        self, storyboard_snapshot_id: str
    ) -> ExecutionRun | None:
        """Loads the most recent ExecutionRun for the storyboard, regardless of status."""
        stmt = (
            select(ExecutionRunORM)
            .where(ExecutionRunORM.storyboard_snapshot_id == storyboard_snapshot_id)
            .order_by(ExecutionRunORM.started_at.desc())
        )
        orm_run = self._session.scalars(stmt).first()
        if orm_run is not None:
            return execution_run_from_orm(orm_run)
        return None

    def add_shot_execution(self, execution: ShotExecution) -> ShotExecution:
        """Persists a new ShotExecution."""
        orm_exec = shot_execution_to_orm(execution)
        self._session.add(orm_exec)
        self._session.flush()
        return execution

    def get_shot_execution(self, shot_execution_id: str) -> ShotExecution | None:
        """Loads a ShotExecution by ID."""
        orm_exec = self._session.get(ShotExecutionORM, shot_execution_id)
        if orm_exec is None:
            return None
        return shot_execution_from_orm(orm_exec)

    def update_shot_execution(
        self, execution: ShotExecution
    ) -> ShotExecution | None:
        """Updates status, candidates, attempts, and error details of a ShotExecution."""
        import json

        orm_exec = self._session.get(
            ShotExecutionORM, execution.shot_execution_id
        )
        if orm_exec is None:
            return None
        orm_exec.status = execution.status.value
        orm_exec.current_candidate_index = execution.current_candidate_index
        orm_exec.attempt_ids_json = json.dumps(
            execution.attempt_ids, ensure_ascii=False
        )
        orm_exec.produced_asset_version_id = execution.produced_asset_version_id
        orm_exec.error_code = execution.error_code
        orm_exec.error_message = execution.error_message
        orm_exec.unconfirmed_remote_task_id = execution.unconfirmed_remote_task_id
        orm_exec.finished_at = execution.finished_at
        self._session.flush()
        return shot_execution_from_orm(orm_exec)

    def list_shot_executions_for_run(
        self, run_id: str
    ) -> list[ShotExecution]:
        """Lists all shot executions for an execution run."""
        stmt = (
            select(ShotExecutionORM)
            .where(ShotExecutionORM.execution_run_id == run_id)
            .order_by(ShotExecutionORM.started_at.asc())
        )
        orm_execs = self._session.scalars(stmt).all()
        return [shot_execution_from_orm(e) for e in orm_execs]

    def save_attempt_request(self, req: AttemptRequest) -> AttemptRequest:
        """Persists a new AttemptRequest, returning existing record if already saved."""
        stmt = select(AttemptRequestORM).where(
            AttemptRequestORM.execution_attempt_id == req.execution_attempt_id
        )
        existing = self._session.scalars(stmt).first()
        if existing is not None:
            return attempt_request_from_orm(existing)
        orm_req = attempt_request_to_orm(req)
        self._session.add(orm_req)
        self._session.flush()
        return req

    def get_attempt_request(self, attempt_id: str) -> AttemptRequest | None:
        """Loads an AttemptRequest by execution attempt ID."""
        stmt = select(AttemptRequestORM).where(AttemptRequestORM.execution_attempt_id == attempt_id)
        orm_req = self._session.scalars(stmt).first()
        if orm_req is None:
            return None
        return attempt_request_from_orm(orm_req)

    def save_provider_receipt(self, receipt: ProviderReceipt) -> ProviderReceipt:
        """Persists a new ProviderReceipt."""
        orm_receipt = provider_receipt_to_orm(receipt)
        self._session.add(orm_receipt)
        self._session.flush()
        return receipt

    def get_provider_receipt(self, attempt_id: str) -> ProviderReceipt | None:
        """Loads a ProviderReceipt by execution attempt ID."""
        stmt = select(ProviderReceiptORM).where(ProviderReceiptORM.execution_attempt_id == attempt_id)
        orm_receipt = self._session.scalars(stmt).first()
        if orm_receipt is None:
            return None
        return provider_receipt_from_orm(orm_receipt)

    def save_attempt_result(self, result: AttemptResult) -> AttemptResult:
        """Persists an AttemptResult."""
        orm_res = attempt_result_to_orm(result)
        self._session.add(orm_res)
        self._session.flush()
        return result

    def get_attempt_result(self, attempt_id: str) -> AttemptResult | None:
        """Loads an AttemptResult by execution attempt ID."""
        stmt = select(AttemptResultORM).where(AttemptResultORM.execution_attempt_id == attempt_id)
        orm_res = self._session.scalars(stmt).first()
        if orm_res is None:
            return None
        return attempt_result_from_orm(orm_res)

    def record_transition(self, transition: ExecutionTransition) -> ExecutionTransition:
        """Appends an immutable transition audit event."""
        orm_trans = execution_transition_to_orm(transition)
        self._session.add(orm_trans)
        self._session.flush()
        return transition

    def list_transitions(self, entity_id: str) -> list[ExecutionTransition]:
        """Lists all transitions for a given entity, ordered chronologically."""
        stmt = (
            select(ExecutionTransitionORM)
            .where(ExecutionTransitionORM.entity_id == entity_id)
            .order_by(ExecutionTransitionORM.created_at.asc())
        )
        orm_trans = self._session.scalars(stmt).all()
        return [execution_transition_from_orm(t) for t in orm_trans]


# Alias for explicit domain semantics
ShotExecutionRepository = ExecutionRepository


class EvaluationRepository:
    """
    Append-only repository for Evaluation targets, dimension results, and snapshots.

    Guarantees strict auditability and historical immutability.
    Contains NO update methods to ensure that past evaluation judgments cannot be mutated.
    """

    def __init__(self, session: Session):
        self._session = session

    def save_target(self, target: EvaluationTarget) -> EvaluationTarget:
        """Atomically persists an immutable EvaluationTarget."""
        orm_target = evaluation_target_to_orm(target)
        self._session.add(orm_target)
        self._session.flush()
        return target

    add_target = save_target

    def get_target(self, evaluation_target_id: str) -> EvaluationTarget | None:
        """Loads an EvaluationTarget by ID."""
        stmt = select(EvaluationTargetORM).where(
            EvaluationTargetORM.evaluation_target_id == evaluation_target_id
        )
        orm_target = self._session.scalars(stmt).first()
        if orm_target is None:
            return None
        return evaluation_target_from_orm(orm_target)

    def save_dimension_result(
        self, result: DimensionEvaluationResult
    ) -> DimensionEvaluationResult:
        """Atomically persists an immutable DimensionEvaluationResult."""
        orm_res = dimension_evaluation_result_to_orm(result)
        self._session.add(orm_res)
        self._session.flush()
        return result

    add_dimension_result = save_dimension_result

    def get_dimension_result(
        self, dimension_result_id: str
    ) -> DimensionEvaluationResult | None:
        """Loads a DimensionEvaluationResult by ID."""
        stmt = select(DimensionEvaluationResultORM).where(
            DimensionEvaluationResultORM.dimension_result_id == dimension_result_id
        )
        orm_res = self._session.scalars(stmt).first()
        if orm_res is None:
            return None
        return dimension_evaluation_result_from_orm(orm_res)

    def list_dimension_results_for_target(
        self, evaluation_target_id: str
    ) -> list[DimensionEvaluationResult]:
        """Lists all dimension evaluation results for a target, chronologically ordered."""
        stmt = (
            select(DimensionEvaluationResultORM)
            .where(
                DimensionEvaluationResultORM.evaluation_target_id
                == evaluation_target_id
            )
            .order_by(DimensionEvaluationResultORM.created_at.asc())
        )
        orm_results = self._session.scalars(stmt).all()
        return [dimension_evaluation_result_from_orm(r) for r in orm_results]

    def save_snapshot(self, snapshot: EvaluationSnapshot) -> EvaluationSnapshot:
        """
        Atomically persists an immutable EvaluationSnapshot and its dimension result associations.
        """
        orm_snap = evaluation_snapshot_to_orm(snapshot)
        self._session.add(orm_snap)
        self._session.flush()
        return snapshot

    add_snapshot = save_snapshot

    def get_snapshot(self, evaluation_snapshot_id: str) -> EvaluationSnapshot | None:
        """Loads an EvaluationSnapshot by ID."""
        stmt = (
            select(EvaluationSnapshotORM)
            .options(
                selectinload(
                    EvaluationSnapshotORM.dimension_associations
                )
            )
            .where(
                EvaluationSnapshotORM.evaluation_snapshot_id == evaluation_snapshot_id
            )
        )
        orm_snap = self._session.scalars(stmt).first()
        if orm_snap is None:
            return None
        return evaluation_snapshot_from_orm(orm_snap)

    def list_snapshots_for_target(
        self, evaluation_target_id: str
    ) -> list[EvaluationSnapshot]:
        """Lists all evaluation snapshots for a target, chronologically ordered."""
        stmt = (
            select(EvaluationSnapshotORM)
            .options(
                selectinload(
                    EvaluationSnapshotORM.dimension_associations
                )
            )
            .where(
                EvaluationSnapshotORM.evaluation_target_id == evaluation_target_id
            )
            .order_by(EvaluationSnapshotORM.created_at.asc())
        )
        orm_snaps = self._session.scalars(stmt).all()
        return [evaluation_snapshot_from_orm(s) for s in orm_snaps]

    def get_snapshot_dimension_results(
        self, evaluation_snapshot_id: str
    ) -> tuple[DimensionEvaluationResult, ...]:
        """
        Reconstructs the exact sequence of DimensionEvaluationResult domain objects frozen in this snapshot.

        Strictly retrieves by frozen dimension_result_ids; does NOT resolve to mutable current results.
        """
        snap = self.get_snapshot(evaluation_snapshot_id)
        if snap is None or not snap.dimension_result_ids:
            return ()

        stmt = select(DimensionEvaluationResultORM).where(
            DimensionEvaluationResultORM.dimension_result_id.in_(snap.dimension_result_ids)
        )
        loaded = {
            r.dimension_result_id: dimension_evaluation_result_from_orm(r)
            for r in self._session.scalars(stmt).all()
        }
        return tuple(
            loaded[res_id] for res_id in snap.dimension_result_ids if res_id in loaded
        )

    def save_composition_preview(
        self, preview: CompositionPreview
    ) -> CompositionPreview:
        """Atomically persists an immutable derived CompositionPreview."""
        orm_preview = composition_preview_to_orm(preview)
        self._session.add(orm_preview)
        self._session.flush()
        return preview

    add_composition_preview = save_composition_preview

    def get_composition_preview(
        self, composition_preview_id: str
    ) -> CompositionPreview | None:
        """Loads a CompositionPreview by ID."""
        stmt = select(CompositionPreviewORM).where(
            CompositionPreviewORM.composition_preview_id == composition_preview_id
        )
        orm_prev = self._session.scalars(stmt).first()
        if orm_prev is None:
            return None
        return composition_preview_from_orm(orm_prev)

    def get_composition_preview_by_fingerprint(
        self,
        shot_asset_version_id: str,
        render_context_fingerprint: str,
        preview_policy_version: str = "composition-preview-v1",
    ) -> CompositionPreview | None:
        """Loads an existing CompositionPreview matching asset, render context, and policy version."""
        stmt = (
            select(CompositionPreviewORM)
            .where(
                CompositionPreviewORM.shot_asset_version_id == shot_asset_version_id,
                CompositionPreviewORM.render_context_fingerprint == render_context_fingerprint,
                CompositionPreviewORM.preview_policy_version == preview_policy_version,
            )
            .order_by(CompositionPreviewORM.created_at.desc())
        )
        orm_prev = self._session.scalars(stmt).first()
        if orm_prev is None:
            return None
        return composition_preview_from_orm(orm_prev)

    def save_remediation_decision(
        self, decision: QualityRemediationDecision
    ) -> QualityRemediationDecision:
        """Atomically persists an immutable QualityRemediationDecision."""
        orm_dec = quality_remediation_decision_to_orm(decision)
        self._session.add(orm_dec)
        self._session.flush()
        return decision

    add_remediation_decision = save_remediation_decision

    def get_remediation_decision(
        self, remediation_decision_id: str
    ) -> QualityRemediationDecision | None:
        """Loads a QualityRemediationDecision by ID."""
        stmt = select(QualityRemediationDecisionORM).where(
            QualityRemediationDecisionORM.remediation_decision_id == remediation_decision_id
        )
        orm_dec = self._session.scalars(stmt).first()
        if orm_dec is None:
            return None
        return quality_remediation_decision_from_orm(orm_dec)

    def list_remediation_decisions_for_shot(
        self, shot_id: str
    ) -> list[QualityRemediationDecision]:
        """Lists all quality remediation decisions for a shot, chronologically ordered."""
        stmt = (
            select(QualityRemediationDecisionORM)
            .where(QualityRemediationDecisionORM.shot_id == shot_id)
            .order_by(QualityRemediationDecisionORM.created_at.asc())
        )
        return [quality_remediation_decision_from_orm(r) for r in self._session.scalars(stmt).all()]

    def list_remediation_decisions_for_chain(
        self, quality_chain_id: str
    ) -> list[QualityRemediationDecision]:
        """Lists all quality remediation decisions for a quality chain, chronologically ordered."""
        stmt = (
            select(QualityRemediationDecisionORM)
            .where(QualityRemediationDecisionORM.quality_chain_id == quality_chain_id)
            .order_by(QualityRemediationDecisionORM.created_at.asc())
        )
        return [quality_remediation_decision_from_orm(r) for r in self._session.scalars(stmt).all()]

    def save_quality_selection(
        self, selection: ShotQualitySelection
    ) -> ShotQualitySelection:
        """Atomically persists an immutable ShotQualitySelection."""
        orm_sel = shot_quality_selection_to_orm(selection)
        self._session.add(orm_sel)
        self._session.flush()
        return selection

    add_quality_selection = save_quality_selection

    def get_quality_selection(
        self, quality_selection_id: str
    ) -> ShotQualitySelection | None:
        """Loads a ShotQualitySelection by ID."""
        stmt = select(ShotQualitySelectionORM).where(
            ShotQualitySelectionORM.quality_selection_id == quality_selection_id
        )
        orm_sel = self._session.scalars(stmt).first()
        if orm_sel is None:
            return None
        return shot_quality_selection_from_orm(orm_sel)

    def get_quality_selection_for_shot(
        self, shot_id: str
    ) -> ShotQualitySelection | None:
        """Loads the most recent quality selection for a shot."""
        stmt = (
            select(ShotQualitySelectionORM)
            .where(ShotQualitySelectionORM.shot_id == shot_id)
            .order_by(ShotQualitySelectionORM.created_at.desc())
        )
        orm_sel = self._session.scalars(stmt).first()
        if orm_sel is None:
            return None
        return shot_quality_selection_from_orm(orm_sel)


class TraceRepository:
    """Repository for managing TraceRoot and immutable/lifecycle-managed TraceEvents."""

    def __init__(self, session: Session):
        self._session = session

    def add_root(self, root: TraceRoot) -> TraceRoot:
        """Persists a new TraceRoot entity."""
        orm_root = trace_root_to_orm(root)
        self._session.add(orm_root)
        self._session.flush()
        return root

    def get_root(self, trace_id: str) -> TraceRoot | None:
        """Loads a TraceRoot by its trace_id."""
        stmt = select(TraceRootORM).where(TraceRootORM.trace_id == trace_id)
        orm_root = self._session.scalars(stmt).first()
        if orm_root is None:
            return None
        return trace_root_from_orm(orm_root)

    def add_event(self, event: TraceEvent) -> TraceEvent:
        """Persists a new TraceEvent entity."""
        orm_event = trace_event_to_orm(event)
        self._session.add(orm_event)
        self._session.flush()
        return event

    def get_event(self, trace_event_id: str) -> TraceEvent | None:
        """Loads a TraceEvent by its trace_event_id."""
        stmt = select(TraceEventORM).where(TraceEventORM.trace_event_id == trace_event_id)
        orm_event = self._session.scalars(stmt).first()
        if orm_event is None:
            return None
        return trace_event_from_orm(orm_event)

    def complete_event(
        self,
        trace_event_id: str,
        status: TraceEventStatus,
        completed_at: datetime,
        duration_ms: float,
        attributes_update: dict | None = None,
        error_code: str | None = None,
    ) -> TraceEvent:
        """
        Completes a previously STARTED TraceEvent (lifecycle update).
        Updates status, completed_at, duration_ms, error_code, and merges attributes_update.
        """
        stmt = select(TraceEventORM).where(TraceEventORM.trace_event_id == trace_event_id)
        orm_event = self._session.scalars(stmt).first()
        if orm_event is None:
            raise ValueError(f"TraceEvent '{trace_event_id}' not found.")

        if orm_event.status != TraceEventStatus.STARTED.value:
            raise ValueError(
                f"Cannot complete event '{trace_event_id}' already in status '{orm_event.status}'."
            )

        orm_event.status = status.value if hasattr(status, "value") else str(status)
        orm_event.completed_at = completed_at
        orm_event.duration_ms = duration_ms
        if error_code:
            orm_event.error_code = error_code
        if attributes_update:
            attrs = dict(orm_event.attributes_json or {})
            attrs.update(attributes_update)
            orm_event.attributes_json = attrs

        self._session.flush()
        return trace_event_from_orm(orm_event)

    def list_events_by_trace(self, trace_id: str) -> list[TraceEvent]:
        """Lists all TraceEvents belonging to a trace_id ordered chronologically."""
        stmt = (
            select(TraceEventORM)
            .where(TraceEventORM.trace_id == trace_id)
            .order_by(TraceEventORM.started_at.asc(), TraceEventORM.created_at.asc())
        )
        return [trace_event_from_orm(e) for e in self._session.scalars(stmt).all()]

    def list_events_by_shot(self, shot_id: str, trace_id: str | None = None) -> list[TraceEvent]:
        """Lists all TraceEvents associated with a shot_id ordered chronologically."""
        stmt = select(TraceEventORM).where(TraceEventORM.shot_id == shot_id)
        if trace_id:
            stmt = stmt.where(TraceEventORM.trace_id == trace_id)
        stmt = stmt.order_by(TraceEventORM.started_at.asc(), TraceEventORM.created_at.asc())
        return [trace_event_from_orm(e) for e in self._session.scalars(stmt).all()]

    def list_events_by_attempt(
        self,
        execution_attempt_id: str,
        trace_id: str | None = None,
    ) -> list[TraceEvent]:
        """Lists all TraceEvents associated with an execution_attempt_id ordered chronologically."""
        stmt = select(TraceEventORM)
        if trace_id:
            stmt = stmt.where(TraceEventORM.trace_id == trace_id)
        stmt = stmt.order_by(TraceEventORM.started_at.asc(), TraceEventORM.created_at.asc())
        events = [trace_event_from_orm(e) for e in self._session.scalars(stmt).all()]
        return [e for e in events if e.context.execution_attempt_id == execution_attempt_id]

    def list_events_by_evaluation(
        self,
        evaluation_snapshot_id: str,
        trace_id: str | None = None,
    ) -> list[TraceEvent]:
        """Lists all TraceEvents associated with an evaluation_snapshot_id ordered chronologically."""
        stmt = select(TraceEventORM)
        if trace_id:
            stmt = stmt.where(TraceEventORM.trace_id == trace_id)
        stmt = stmt.order_by(TraceEventORM.started_at.asc(), TraceEventORM.created_at.asc())
        events = [trace_event_from_orm(e) for e in self._session.scalars(stmt).all()]
        return [e for e in events if e.context.evaluation_snapshot_id == evaluation_snapshot_id]


class BenchmarkRepository:
    """
    Repository for persisting and querying BenchmarkRun and BenchmarkCaseResult entities.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_run(self, run: BenchmarkRun) -> BenchmarkRun:
        """Persists a new BenchmarkRun."""
        from app.persistence.converters import (
            benchmark_run_from_orm,
            benchmark_run_to_orm,
        )
        from app.persistence.models import BenchmarkRunORM

        existing = self._session.get(BenchmarkRunORM, run.benchmark_run_id)
        if existing:
            raise ValueError(f"BenchmarkRun {run.benchmark_run_id} already exists")

        orm = benchmark_run_to_orm(run)
        self._session.add(orm)
        self._session.flush()
        return benchmark_run_from_orm(orm)

    def update_run(self, run: BenchmarkRun) -> BenchmarkRun:
        """Updates an existing BenchmarkRun."""
        from app.persistence.converters import benchmark_run_from_orm
        from app.persistence.models import BenchmarkRunORM

        orm = self._session.get(BenchmarkRunORM, run.benchmark_run_id)
        if not orm:
            raise ValueError(f"BenchmarkRun {run.benchmark_run_id} does not exist")

        orm.status = run.status.value
        orm.finished_at = run.finished_at
        orm.completed_cases = run.completed_cases
        orm.failed_cases = run.failed_cases
        orm.total_cases = run.total_cases
        self._session.flush()
        return benchmark_run_from_orm(orm)

    def get_run(self, benchmark_run_id: str) -> BenchmarkRun | None:
        """Retrieves a BenchmarkRun by ID."""
        from app.persistence.converters import benchmark_run_from_orm
        from app.persistence.models import BenchmarkRunORM

        orm = self._session.get(BenchmarkRunORM, benchmark_run_id)
        return benchmark_run_from_orm(orm) if orm else None

    def add_case_result(self, result: BenchmarkCaseResult) -> BenchmarkCaseResult:
        """Persists a new BenchmarkCaseResult."""
        from app.persistence.converters import (
            benchmark_case_result_from_orm,
            benchmark_case_result_to_orm,
        )
        from app.persistence.models import BenchmarkCaseResultORM

        existing = self._session.get(BenchmarkCaseResultORM, result.benchmark_case_result_id)
        if existing:
            raise ValueError(f"BenchmarkCaseResult {result.benchmark_case_result_id} already exists")

        orm = benchmark_case_result_to_orm(result)
        self._session.add(orm)
        self._session.flush()
        return benchmark_case_result_from_orm(orm)

    def get_case_result(self, benchmark_case_result_id: str) -> BenchmarkCaseResult | None:
        """Retrieves a BenchmarkCaseResult by ID."""
        from app.persistence.converters import benchmark_case_result_from_orm
        from app.persistence.models import BenchmarkCaseResultORM

        orm = self._session.get(BenchmarkCaseResultORM, benchmark_case_result_id)
        return benchmark_case_result_from_orm(orm) if orm else None

    def list_case_results_for_run(self, benchmark_run_id: str) -> list[BenchmarkCaseResult]:
        """Lists all BenchmarkCaseResults for a BenchmarkRun ordered chronologically."""
        from app.persistence.converters import benchmark_case_result_from_orm
        from app.persistence.models import BenchmarkCaseResultORM

        stmt = (
            select(BenchmarkCaseResultORM)
            .where(BenchmarkCaseResultORM.benchmark_run_id == benchmark_run_id)
            .order_by(BenchmarkCaseResultORM.started_at.asc())
        )
        return [benchmark_case_result_from_orm(r) for r in self._session.scalars(stmt).all()]

    def add_report(self, report: Any) -> Any:
        """Persists a new BenchmarkReport."""
        from app.persistence.converters import (
            benchmark_report_from_orm,
            benchmark_report_to_orm,
        )
        from app.persistence.models import BenchmarkReportORM

        existing = self._session.get(BenchmarkReportORM, report.report_id)
        if existing:
            raise ValueError(f"BenchmarkReport {report.report_id} already exists")

        orm = benchmark_report_to_orm(report)
        self._session.add(orm)
        self._session.flush()
        return benchmark_report_from_orm(orm)

    def get_report(self, report_id: str) -> Any | None:
        """Retrieves a BenchmarkReport by its report_id."""
        from app.persistence.converters import benchmark_report_from_orm
        from app.persistence.models import BenchmarkReportORM

        orm = self._session.get(BenchmarkReportORM, report_id)
        return benchmark_report_from_orm(orm) if orm else None

    def get_report_by_run(
        self, benchmark_run_id: str, metric_version: str = "v1"
    ) -> Any | None:
        """Retrieves a BenchmarkReport by benchmark_run_id and metric_definition_set_version."""
        from app.persistence.converters import benchmark_report_from_orm
        from app.persistence.models import BenchmarkReportORM

        stmt = select(BenchmarkReportORM).where(
            BenchmarkReportORM.benchmark_run_id == benchmark_run_id,
            BenchmarkReportORM.metric_definition_set_version == metric_version,
        )
        orm = self._session.scalars(stmt).first()
        return benchmark_report_from_orm(orm) if orm else None

    def list_reports_for_suite(self, suite_key: str) -> list[Any]:
        """Lists all BenchmarkReports for a suite ordered by generation time descending."""
        from app.persistence.converters import benchmark_report_from_orm
        from app.persistence.models import BenchmarkReportORM

        stmt = (
            select(BenchmarkReportORM)
            .where(BenchmarkReportORM.suite_key == suite_key)
            .order_by(BenchmarkReportORM.generated_at.desc())
        )
        return [benchmark_report_from_orm(r) for r in self._session.scalars(stmt).all()]

    def add_comparison_report(self, comp: Any) -> Any:
        """Persists a new BenchmarkComparisonReport."""
        from app.persistence.converters import (
            benchmark_comparison_report_from_orm,
            benchmark_comparison_report_to_orm,
        )
        from app.persistence.models import BenchmarkComparisonReportORM

        existing = self._session.get(BenchmarkComparisonReportORM, comp.comparison_id)
        if existing:
            raise ValueError(f"BenchmarkComparisonReport {comp.comparison_id} already exists")

        orm = benchmark_comparison_report_to_orm(comp)
        self._session.add(orm)
        self._session.flush()
        return benchmark_comparison_report_from_orm(orm)

    def get_comparison_report(self, comparison_id: str) -> Any | None:
        """Retrieves a BenchmarkComparisonReport by comparison_id."""
        from app.persistence.converters import benchmark_comparison_report_from_orm
        from app.persistence.models import BenchmarkComparisonReportORM

        orm = self._session.get(BenchmarkComparisonReportORM, comparison_id)
        return benchmark_comparison_report_from_orm(orm) if orm else None


class KnowledgeVideoTaskRepository:
    """Repository for managing KnowledgeVideoTask persistence and state."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_task(self, task: KnowledgeVideoTask) -> KnowledgeVideoTask:
        existing = self._session.get(KnowledgeVideoTaskORM, task.task_id)
        if existing is None:
            orm = knowledge_video_task_to_orm(task)
            self._session.add(orm)
        else:
            existing.topic = task.topic
            existing.task_status = task.task_status.value if hasattr(task.task_status, "value") else str(task.task_status)
            existing.current_stage = task.current_stage.value if hasattr(task.current_stage, "value") else str(task.current_stage)
            existing.workflow_policy = task.workflow_policy.value if hasattr(task.workflow_policy, "value") else str(task.workflow_policy)
            existing.target_duration = task.target_duration
            existing.aspect_ratio = task.aspect_ratio
            existing.language = task.language
            existing.waiting_reason = task.waiting_reason
            existing.error_type = task.error_type.value if (task.error_type and hasattr(task.error_type, "value")) else (str(task.error_type) if task.error_type else None)
            existing.error_message = task.error_message
            existing.metadata_json = task.task_metadata
            existing.updated_at = task.updated_at
            existing.finished_at = task.finished_at
            orm = existing
        self._session.flush()
        return knowledge_video_task_from_orm(orm)

    def get_task(self, task_id: str) -> KnowledgeVideoTask | None:
        orm = self._session.get(KnowledgeVideoTaskORM, task_id)
        return knowledge_video_task_from_orm(orm) if orm is not None else None

    def list_tasks(self, limit: int = 50, offset: int = 0) -> list[KnowledgeVideoTask]:
        stmt = (
            select(KnowledgeVideoTaskORM)
            .order_by(KnowledgeVideoTaskORM.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return [knowledge_video_task_from_orm(r) for r in self._session.scalars(stmt).all()]


class WorkflowJobRepository:
    """Repository for managing durable WorkflowJob execution units, leases, and concurrency."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_job(self, job: WorkflowJob) -> WorkflowJob:
        orm = workflow_job_to_orm(job)
        self._session.add(orm)
        self._session.flush()
        return workflow_job_from_orm(orm)

    def get_job(self, job_id: str) -> WorkflowJob | None:
        orm = self._session.get(WorkflowJobORM, job_id)
        return workflow_job_from_orm(orm) if orm is not None else None

    def get_job_by_idempotency_key(self, idempotency_key: str) -> WorkflowJob | None:
        stmt = select(WorkflowJobORM).where(WorkflowJobORM.idempotency_key == idempotency_key)
        orm = self._session.scalars(stmt).first()
        return workflow_job_from_orm(orm) if orm is not None else None

    def get_current_job_for_task(self, task_id: str) -> WorkflowJob | None:
        stmt = (
            select(WorkflowJobORM)
            .where(WorkflowJobORM.task_id == task_id)
            .order_by(WorkflowJobORM.attempt_number.desc(), WorkflowJobORM.created_at.desc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return workflow_job_from_orm(orm) if orm is not None else None

    def update_job(self, job: WorkflowJob) -> WorkflowJob:
        orm = self._session.get(WorkflowJobORM, job.job_id)
        if orm is None:
            raise ValueError(f"Job {job.job_id} not found")
        orm.status = job.status.value if hasattr(job.status, "value") else str(job.status)
        orm.stage = job.stage.value if hasattr(job.stage, "value") else str(job.stage)
        orm.attempt_number = job.attempt_number
        orm.max_attempts = job.max_attempts
        orm.available_at = job.available_at
        orm.lease_owner = job.lease_owner
        orm.lease_expires_at = job.lease_expires_at
        orm.heartbeat_at = job.heartbeat_at
        orm.input_task_artifact_ref_id = job.input_task_artifact_ref_id
        orm.output_task_artifact_ref_id = job.output_task_artifact_ref_id
        orm.input_artifact_revision_id = job.input_artifact_revision_id
        orm.output_artifact_revision_id = job.output_artifact_revision_id
        orm.error_type = job.error_type
        orm.error_message = job.error_message
        orm.started_at = job.started_at
        orm.finished_at = job.finished_at
        self._session.flush()
        return workflow_job_from_orm(orm)

    def acquire_next_available_job(
        self,
        owner: str,
        supported_stages: set[Stage] | Sequence[Stage] | None = None,
        lease_duration_seconds: int = 300,
        now: datetime | None = None,
    ) -> WorkflowJob | None:
        from datetime import UTC, timedelta
        ts = now or datetime.now(UTC)

        if supported_stages is not None:
            if len(supported_stages) == 0:
                return None
            stage_vals = [s.value if hasattr(s, "value") else str(s) for s in supported_stages]
        else:
            stage_vals = None

        conditions = [
            (
                (WorkflowJobORM.status == JobStatus.QUEUED.value)
                & (WorkflowJobORM.available_at <= ts)
            )
            | (
                WorkflowJobORM.status.in_([JobStatus.LEASED.value, JobStatus.RUNNING.value])
                & (WorkflowJobORM.lease_expires_at != None)
                & (WorkflowJobORM.lease_expires_at <= ts)
            )
        ]
        if stage_vals is not None:
            conditions.append(WorkflowJobORM.stage.in_(stage_vals))

        stmt = (
            select(WorkflowJobORM)
            .where(*conditions)
            .order_by(WorkflowJobORM.available_at.asc(), WorkflowJobORM.created_at.asc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        if orm is None:
            return None

        orm.status = JobStatus.LEASED.value
        orm.lease_owner = owner
        orm.lease_expires_at = ts + timedelta(seconds=lease_duration_seconds)
        orm.heartbeat_at = ts
        self._session.flush()
        return workflow_job_from_orm(orm)

    def renew_lease(
        self,
        job_id: str,
        owner: str,
        extend_seconds: int = 300,
        now: datetime | None = None,
    ) -> WorkflowJob:
        from datetime import UTC, timedelta
        ts = now or datetime.now(UTC)
        orm = self._session.get(WorkflowJobORM, job_id)
        if orm is None:
            raise ValueError(f"Job '{job_id}' not found")
        if orm.lease_owner != owner:
            raise WorkflowConflictError(
                f"Cannot renew lease on job '{job_id}': held by '{orm.lease_owner}', not '{owner}'."
            )
        orm.lease_expires_at = ts + timedelta(seconds=extend_seconds)
        orm.heartbeat_at = ts
        self._session.flush()
        return workflow_job_from_orm(orm)

    def record_heartbeat(
        self,
        job_id: str,
        owner: str,
        now: datetime | None = None,
    ) -> None:
        from datetime import UTC
        ts = now or datetime.now(UTC)
        orm = self._session.get(WorkflowJobORM, job_id)
        if orm is None:
            raise ValueError(f"Job '{job_id}' not found")
        if orm.lease_owner != owner:
            raise WorkflowConflictError(
                f"Cannot heartbeat job '{job_id}': held by '{orm.lease_owner}', not '{owner}'."
            )
        orm.heartbeat_at = ts
        self._session.flush()

    def recover_expired_leases(self, now: datetime | None = None) -> int:
        from datetime import UTC
        ts = now or datetime.now(UTC)
        stmt = (
            select(WorkflowJobORM)
            .where(
                WorkflowJobORM.status.in_([JobStatus.LEASED.value, JobStatus.RUNNING.value]),
                WorkflowJobORM.lease_expires_at != None,
                WorkflowJobORM.lease_expires_at <= ts,
            )
        )
        expired_jobs = self._session.scalars(stmt).all()
        count = len(expired_jobs)
        for job_orm in expired_jobs:
            job_orm.status = JobStatus.QUEUED.value
            job_orm.lease_owner = None
            job_orm.lease_expires_at = None
        if count > 0:
            self._session.flush()
        return count

    def list_jobs_for_task(self, task_id: str) -> list[WorkflowJob]:
        """Lists all workflow jobs associated with a task ordered by creation time."""
        stmt = (
            select(WorkflowJobORM)
            .where(WorkflowJobORM.task_id == task_id)
            .order_by(WorkflowJobORM.attempt_number.asc(), WorkflowJobORM.created_at.asc())
        )
        return [workflow_job_from_orm(r) for r in self._session.scalars(stmt).all()]


class StageExecutionRepository:
    """Repository for managing audit history of stage execution attempts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record_execution(self, execution: StageExecution) -> StageExecution:
        orm = stage_execution_to_orm(execution)
        self._session.add(orm)
        self._session.flush()
        return stage_execution_from_orm(orm)

    def list_executions_for_task(self, task_id: str) -> list[StageExecution]:
        stmt = (
            select(StageExecutionORM)
            .where(StageExecutionORM.task_id == task_id)
            .order_by(StageExecutionORM.attempt_number.asc(), StageExecutionORM.started_at.asc())
        )
        return [stage_execution_from_orm(r) for r in self._session.scalars(stmt).all()]

    def get_latest_execution_for_stage(self, task_id: str, stage: Stage) -> StageExecution | None:
        stage_val = stage.value if hasattr(stage, "value") else str(stage)
        stmt = (
            select(StageExecutionORM)
            .where(
                StageExecutionORM.task_id == task_id,
                StageExecutionORM.stage == stage_val,
            )
            .order_by(StageExecutionORM.attempt_number.desc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return stage_execution_from_orm(orm) if orm is not None else None


class TaskArtifactRepository:
    """Repository for managing immutable task artifact references."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_artifact_ref(self, ref: TaskArtifactRef) -> TaskArtifactRef:
        orm = task_artifact_ref_to_orm(ref)
        orm = self._session.merge(orm)
        self._session.flush()
        return task_artifact_ref_from_orm(orm)

    def get_artifact_ref(self, ref_id: str) -> TaskArtifactRef | None:
        orm = self._session.get(TaskArtifactRefORM, ref_id)
        return task_artifact_ref_from_orm(orm) if orm is not None else None

    def list_artifact_refs_for_task(
        self,
        task_id: str,
        stage: Stage | None = None,
    ) -> list[TaskArtifactRef]:
        conditions = [TaskArtifactRefORM.task_id == task_id]
        if stage is not None:
            stage_val = stage.value if hasattr(stage, "value") else str(stage)
            conditions.append(TaskArtifactRefORM.stage == stage_val)
        stmt = (
            select(TaskArtifactRefORM)
            .where(*conditions)
            .order_by(TaskArtifactRefORM.created_at.asc())
        )
        return [task_artifact_ref_from_orm(r) for r in self._session.scalars(stmt).all()]

    def get_latest_artifact_ref(
        self,
        task_id: str,
        stage: Stage | None = None,
        artifact_type: ArtifactType | None = None,
    ) -> TaskArtifactRef | None:
        stmt = select(TaskArtifactRefORM).where(TaskArtifactRefORM.task_id == task_id)
        if stage is not None:
            stage_val = stage.value if hasattr(stage, "value") else str(stage)
            stmt = stmt.where(TaskArtifactRefORM.stage == stage_val)
        if artifact_type is not None:
            type_val = artifact_type.value if hasattr(artifact_type, "value") else str(artifact_type)
            stmt = stmt.where(TaskArtifactRefORM.artifact_type == type_val)
        stmt = stmt.order_by(TaskArtifactRefORM.created_at.desc()).limit(1)
        orm = self._session.scalars(stmt).first()
        return task_artifact_ref_from_orm(orm) if orm is not None else None

    def mark_artifact_stale(
        self,
        task_artifact_ref_id: str,
        reason: str = "",
        now: datetime | None = None,
    ) -> TaskArtifactRef | None:
        """
        Marks an existing task artifact reference as stale/superseded.
        Does not mutate the immutable artifact payload itself.
        """
        orm = self._session.get(TaskArtifactRefORM, task_artifact_ref_id)
        if orm is None:
            return None
        meta = dict(orm.metadata_json or {})
        meta["is_current"] = False
        meta["is_stale"] = True
        meta["stale_reason"] = reason
        meta["stale_at"] = (now or datetime.now(UTC)).isoformat()
        orm.metadata_json = meta
        self._session.flush()
        return task_artifact_ref_from_orm(orm)

    def get_current_artifact_ref(
        self,
        task_id: str,
        stage: Stage | None = None,
        artifact_type: ArtifactType | None = None,
    ) -> TaskArtifactRef | None:
        """
        Returns the latest CURRENT (non-stale) artifact reference for the task.
        """
        stmt = select(TaskArtifactRefORM).where(TaskArtifactRefORM.task_id == task_id)
        if stage is not None:
            stage_val = stage.value if hasattr(stage, "value") else str(stage)
            stmt = stmt.where(TaskArtifactRefORM.stage == stage_val)
        if artifact_type is not None:
            type_val = artifact_type.value if hasattr(artifact_type, "value") else str(artifact_type)
            stmt = stmt.where(TaskArtifactRefORM.artifact_type == type_val)
        stmt = stmt.order_by(TaskArtifactRefORM.created_at.desc())
        all_orms = self._session.scalars(stmt).all()
        for orm in all_orms:
            ref = task_artifact_ref_from_orm(orm)
            if ref.is_current and not ref.is_stale:
                return ref
        return None

    def invalidate_downstream_artifacts(
        self,
        task_id: str,
        stages: Sequence[Stage],
        reason: str = "",
        now: datetime | None = None,
    ) -> list[TaskArtifactRef]:
        """
        Marks all current artifacts for the specified downstream stages as stale/superseded.
        Historical artifacts remain queryable.
        """
        invalidated: list[TaskArtifactRef] = []
        for stg in stages:
            stg_val = stg.value if hasattr(stg, "value") else str(stg)
            stmt = (
                select(TaskArtifactRefORM)
                .where(
                    TaskArtifactRefORM.task_id == task_id,
                    TaskArtifactRefORM.stage == stg_val,
                )
            )
            for orm in self._session.scalars(stmt).all():
                ref = task_artifact_ref_from_orm(orm)
                if ref.is_current and not ref.is_stale:
                    updated = self.mark_artifact_stale(ref.task_artifact_ref_id, reason=reason, now=now)
                    if updated:
                        invalidated.append(updated)
        return invalidated


class EvidenceRepository:
    """Repository for managing evidence sources, items, claims, and snapshots."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_source_document(self, doc: SourceDocument) -> SourceDocument:
        from app.persistence.converters import source_document_from_orm, source_document_to_orm
        from app.persistence.models import SourceDocumentORM

        orm = self._session.get(SourceDocumentORM, doc.source_document_id)
        if orm is None:
            orm = source_document_to_orm(doc)
            self._session.add(orm)
        else:
            orm.title = doc.title
            orm.source_locator = doc.source_locator
            orm.content_snapshot = doc.content_snapshot
            orm.content_hash = doc.content_hash
            orm.source_fingerprint = doc.source_fingerprint
            orm.author = doc.author
            orm.published_at = doc.published_at
            orm.captured_at = doc.captured_at
            orm.media_type = doc.media_type
            orm.status = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
            orm.metadata_json = doc.metadata_json
        self._session.flush()
        return source_document_from_orm(orm)

    def get_source_document(self, doc_id: str) -> SourceDocument | None:
        from app.persistence.converters import source_document_from_orm
        from app.persistence.models import SourceDocumentORM

        orm = self._session.get(SourceDocumentORM, doc_id)
        return source_document_from_orm(orm) if orm is not None else None

    def associate_task_source(
        self,
        task_id: str,
        source_document_id: str,
        role: str = "PRIMARY",
        now: datetime | None = None,
    ) -> None:
        from datetime import UTC
        from uuid import uuid4
        from app.persistence.models import TaskSourceORM

        ts = now or datetime.now(UTC)
        stmt = select(TaskSourceORM).where(
            TaskSourceORM.task_id == task_id,
            TaskSourceORM.source_document_id == source_document_id,
        )
        existing = self._session.scalars(stmt).first()
        if existing is not None:
            existing.role = role
        else:
            orm = TaskSourceORM(
                task_source_id=uuid4().hex,
                task_id=task_id,
                source_document_id=source_document_id,
                role=role,
                associated_at=ts,
            )
            self._session.add(orm)
        self._session.flush()

    def list_sources_for_task(self, task_id: str) -> list[SourceDocument]:
        from app.persistence.converters import source_document_from_orm
        from app.persistence.models import SourceDocumentORM, TaskSourceORM

        stmt = (
            select(SourceDocumentORM)
            .join(
                TaskSourceORM,
                TaskSourceORM.source_document_id == SourceDocumentORM.source_document_id,
            )
            .where(TaskSourceORM.task_id == task_id)
            .order_by(TaskSourceORM.associated_at.asc())
        )
        return [source_document_from_orm(r) for r in self._session.scalars(stmt).all()]

    def save_evidence_item(self, item: EvidenceItem) -> EvidenceItem:
        from app.persistence.converters import evidence_item_from_orm, evidence_item_to_orm
        from app.persistence.models import EvidenceItemORM

        orm = self._session.get(EvidenceItemORM, item.evidence_id)
        if orm is None:
            orm = evidence_item_to_orm(item)
            self._session.add(orm)
        else:
            orm.source_document_id = item.source_document_id
            orm.locator_json = item.locator
            orm.original_excerpt = item.original_excerpt
            orm.normalized_fact = item.normalized_fact
            orm.evidence_role = (
                item.evidence_role.value
                if hasattr(item.evidence_role, "value")
                else str(item.evidence_role)
            )
            orm.confidence = item.confidence
            orm.extraction_method = item.extraction_method
            orm.content_hash = item.content_hash
        self._session.flush()
        return evidence_item_from_orm(orm)

    def get_evidence_item(self, evidence_id: str) -> EvidenceItem | None:
        from app.persistence.converters import evidence_item_from_orm
        from app.persistence.models import EvidenceItemORM

        orm = self._session.get(EvidenceItemORM, evidence_id)
        return evidence_item_from_orm(orm) if orm is not None else None

    def list_evidence_items_for_source(self, source_id: str) -> list[EvidenceItem]:
        from app.persistence.converters import evidence_item_from_orm
        from app.persistence.models import EvidenceItemORM

        stmt = (
            select(EvidenceItemORM)
            .where(EvidenceItemORM.source_document_id == source_id)
            .order_by(EvidenceItemORM.created_at.asc())
        )
        return [evidence_item_from_orm(r) for r in self._session.scalars(stmt).all()]

    def save_knowledge_claim(self, claim: KnowledgeClaim) -> KnowledgeClaim:
        from app.persistence.converters import knowledge_claim_from_orm, knowledge_claim_to_orm
        from app.persistence.models import KnowledgeClaimORM

        orm = self._session.get(KnowledgeClaimORM, claim.knowledge_claim_id)
        if orm is None:
            orm = knowledge_claim_to_orm(claim)
            self._session.add(orm)
        else:
            orm.claim_type = claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type)
            orm.claim_text = claim.claim_text
            orm.evidence_refs_json = list(claim.evidence_refs)
            orm.verification_status = (
                claim.verification_status.value
                if hasattr(claim.verification_status, "value")
                else str(claim.verification_status)
            )
            orm.conflict_evidence_refs_json = list(claim.conflict_evidence_refs)
        self._session.flush()
        return knowledge_claim_from_orm(orm)

    def get_knowledge_claim(self, claim_id: str) -> KnowledgeClaim | None:
        from app.persistence.converters import knowledge_claim_from_orm
        from app.persistence.models import KnowledgeClaimORM

        orm = self._session.get(KnowledgeClaimORM, claim_id)
        return knowledge_claim_from_orm(orm) if orm is not None else None

    def save_evidence_snapshot(self, snapshot: EvidenceSnapshot) -> EvidenceSnapshot:
        from app.persistence.converters import evidence_snapshot_from_orm, evidence_snapshot_to_orm
        from app.persistence.models import EvidenceSnapshotORM

        orm = self._session.get(EvidenceSnapshotORM, snapshot.evidence_snapshot_id)
        if orm is None:
            orm = evidence_snapshot_to_orm(snapshot)
            self._session.add(orm)
        else:
            orm.task_id = snapshot.task_id
            orm.snapshot_version = snapshot.snapshot_version
            orm.source_document_ids_json = list(snapshot.source_document_ids)
            orm.evidence_ids_json = list(snapshot.evidence_ids)
            orm.knowledge_claim_ids_json = list(snapshot.knowledge_claim_ids)
            orm.content_fingerprint = snapshot.content_fingerprint
        self._session.flush()
        return evidence_snapshot_from_orm(orm)

    def get_evidence_snapshot(self, snapshot_id: str) -> EvidenceSnapshot | None:
        from app.persistence.converters import evidence_snapshot_from_orm
        from app.persistence.models import EvidenceSnapshotORM

        orm = self._session.get(EvidenceSnapshotORM, snapshot_id)
        return evidence_snapshot_from_orm(orm) if orm is not None else None

    def get_latest_snapshot_for_task(self, task_id: str) -> EvidenceSnapshot | None:
        from app.persistence.converters import evidence_snapshot_from_orm
        from app.persistence.models import EvidenceSnapshotORM

        stmt = (
            select(EvidenceSnapshotORM)
            .where(EvidenceSnapshotORM.task_id == task_id)
            .order_by(
                EvidenceSnapshotORM.snapshot_version.desc(),
                EvidenceSnapshotORM.created_at.desc(),
            )
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return evidence_snapshot_from_orm(orm) if orm is not None else None

    def save_knowledge_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """Persists a sequence of KnowledgeChunk records idempotently."""
        from app.persistence.converters import knowledge_chunk_from_orm, knowledge_chunk_to_orm
        from app.persistence.models import KnowledgeChunkORM

        saved: list[KnowledgeChunk] = []
        for chunk in chunks:
            orm = self._session.get(KnowledgeChunkORM, chunk.chunk_id)
            if orm is None:
                orm = knowledge_chunk_to_orm(chunk)
                self._session.add(orm)
            else:
                orm.source_document_id = chunk.source_document_id
                orm.processing_version = chunk.processing_version
                orm.chunk_index = chunk.chunk_index
                orm.normalized_text = chunk.normalized_text
                orm.text_hash = chunk.text_hash
                orm.locator_json = chunk.locator
                orm.content_fingerprint = chunk.content_fingerprint
            self._session.flush()
            saved.append(knowledge_chunk_from_orm(orm))
        return saved

    def list_chunks_for_source(self, source_document_id: str) -> list[KnowledgeChunk]:
        """Lists all chunks for a source document ordered by chunk_index."""
        from app.persistence.converters import knowledge_chunk_from_orm
        from app.persistence.models import KnowledgeChunkORM

        stmt = (
            select(KnowledgeChunkORM)
            .where(KnowledgeChunkORM.source_document_id == source_document_id)
            .order_by(KnowledgeChunkORM.chunk_index.asc())
        )
        return [knowledge_chunk_from_orm(r) for r in self._session.scalars(stmt).all()]

    def list_chunks_for_sources(self, source_document_ids: Sequence[str]) -> list[KnowledgeChunk]:
        """Lists all chunks for multiple source documents ordered deterministically."""
        if not source_document_ids:
            return []
        from app.persistence.converters import knowledge_chunk_from_orm
        from app.persistence.models import KnowledgeChunkORM

        stmt = (
            select(KnowledgeChunkORM)
            .where(KnowledgeChunkORM.source_document_id.in_(source_document_ids))
            .order_by(KnowledgeChunkORM.source_document_id.asc(), KnowledgeChunkORM.chunk_index.asc())
        )
        return [knowledge_chunk_from_orm(r) for r in self._session.scalars(stmt).all()]

    def list_chunks_for_task(self, task_id: str) -> list[KnowledgeChunk]:
        """Lists all chunks for all sources associated with a task."""
        from app.persistence.converters import knowledge_chunk_from_orm
        from app.persistence.models import KnowledgeChunkORM, TaskSourceORM

        stmt = (
            select(KnowledgeChunkORM)
            .join(
                TaskSourceORM,
                TaskSourceORM.source_document_id == KnowledgeChunkORM.source_document_id,
            )
            .where(TaskSourceORM.task_id == task_id)
            .order_by(KnowledgeChunkORM.source_document_id.asc(), KnowledgeChunkORM.chunk_index.asc())
        )
        return [knowledge_chunk_from_orm(r) for r in self._session.scalars(stmt).all()]

    def save_retrieval_snapshot(self, snapshot: RetrievalSnapshot) -> RetrievalSnapshot:
        """Persists a frozen RetrievalSnapshot record."""
        from app.persistence.converters import retrieval_snapshot_from_orm, retrieval_snapshot_to_orm
        from app.persistence.models import RetrievalSnapshotORM

        orm = self._session.get(RetrievalSnapshotORM, snapshot.retrieval_snapshot_id)
        if orm is None:
            orm = retrieval_snapshot_to_orm(snapshot)
            self._session.add(orm)
        else:
            orm.task_id = snapshot.task_id
            orm.query = snapshot.query
            orm.source_scope_ids_json = list(snapshot.source_scope_ids)
            orm.retrieval_policy_version = snapshot.retrieval_policy_version
            orm.processing_version = snapshot.processing_version
            orm.candidates_json = [c.model_dump() if hasattr(c, "model_dump") else c for c in snapshot.candidates]
            orm.selected_evidence_ids_json = list(snapshot.selected_evidence_ids)
            orm.content_fingerprint = snapshot.content_fingerprint
        self._session.flush()
        return retrieval_snapshot_from_orm(orm)

    def get_retrieval_snapshot(self, snapshot_id: str) -> RetrievalSnapshot | None:
        """Loads a frozen RetrievalSnapshot by ID."""
        from app.persistence.converters import retrieval_snapshot_from_orm
        from app.persistence.models import RetrievalSnapshotORM

        orm = self._session.get(RetrievalSnapshotORM, snapshot_id)
        return retrieval_snapshot_from_orm(orm) if orm is not None else None

    def list_retrieval_snapshots_for_task(self, task_id: str) -> list[RetrievalSnapshot]:
        """Lists all retrieval snapshots executed for a task."""
        from app.persistence.converters import retrieval_snapshot_from_orm
        from app.persistence.models import RetrievalSnapshotORM

        stmt = (
            select(RetrievalSnapshotORM)
            .where(RetrievalSnapshotORM.task_id == task_id)
            .order_by(RetrievalSnapshotORM.created_at.desc())
        )
        return [retrieval_snapshot_from_orm(r) for r in self._session.scalars(stmt).all()]

    def save_web_research_snapshot(self, snapshot: WebResearchSnapshot) -> WebResearchSnapshot:
        """Persists an immutable WebResearchSnapshot record."""
        from app.persistence.converters import web_research_snapshot_from_orm, web_research_snapshot_to_orm
        from app.persistence.models import WebResearchSnapshotORM

        orm = self._session.get(WebResearchSnapshotORM, snapshot.web_research_snapshot_id)
        if orm is None:
            orm = web_research_snapshot_to_orm(snapshot)
            self._session.add(orm)
        else:
            orm.task_id = snapshot.task_id
            orm.stage_attempt = snapshot.stage_attempt
            orm.query = snapshot.query
            orm.provider = snapshot.provider
            orm.search_results_json = [r.model_dump() if hasattr(r, "model_dump") else r for r in snapshot.search_results]
            orm.selected_urls_json = list(snapshot.selected_urls)
            orm.fetch_outcomes_json = list(snapshot.fetch_outcomes)
            orm.created_source_document_ids_json = list(snapshot.created_source_document_ids)
            orm.content_fingerprint = snapshot.content_fingerprint
        self._session.flush()
        return web_research_snapshot_from_orm(orm)

    def get_web_research_snapshot(self, snapshot_id: str) -> WebResearchSnapshot | None:
        """Loads a WebResearchSnapshot by ID."""
        from app.persistence.converters import web_research_snapshot_from_orm
        from app.persistence.models import WebResearchSnapshotORM

        orm = self._session.get(WebResearchSnapshotORM, snapshot_id)
        return web_research_snapshot_from_orm(orm) if orm is not None else None

    def list_web_research_snapshots_for_task(self, task_id: str) -> list[WebResearchSnapshot]:
        """Lists all web research snapshots executed for a task."""
        from app.persistence.converters import web_research_snapshot_from_orm
        from app.persistence.models import WebResearchSnapshotORM

        stmt = (
            select(WebResearchSnapshotORM)
            .where(WebResearchSnapshotORM.task_id == task_id)
            .order_by(WebResearchSnapshotORM.created_at.desc())
        )
        return [web_research_snapshot_from_orm(r) for r in self._session.scalars(stmt).all()]


class ScriptRepository:
    """Repository for persisting and querying immutable ScriptRevision and ScriptSegments."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_revision(self, revision: Any) -> Any:
        """Persists a new ScriptRevision and its child ScriptSegments."""
        from app.persistence.converters import (
            script_revision_from_orm,
            script_revision_to_orm,
        )
        from app.persistence.models import ScriptRevisionORM

        orm = self._session.get(ScriptRevisionORM, revision.script_revision_id)
        if orm is None:
            orm = script_revision_to_orm(revision)
            self._session.add(orm)
        self._session.flush()
        return script_revision_from_orm(orm)

    def get_revision(self, script_revision_id: str) -> Any | None:
        """Retrieves a ScriptRevision by primary key ID."""
        from app.persistence.converters import script_revision_from_orm
        from app.persistence.models import ScriptRevisionORM

        orm = self._session.get(ScriptRevisionORM, script_revision_id)
        return script_revision_from_orm(orm) if orm is not None else None

    def get_latest_revision_for_task(self, task_id: str) -> Any | None:
        """Returns the latest ScriptRevision for a task based on revision_number / created_at."""
        from app.persistence.converters import script_revision_from_orm
        from app.persistence.models import ScriptRevisionORM

        stmt = (
            select(ScriptRevisionORM)
            .where(ScriptRevisionORM.task_id == task_id)
            .order_by(
                ScriptRevisionORM.revision_number.desc(),
                ScriptRevisionORM.created_at.desc(),
            )
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return script_revision_from_orm(orm) if orm is not None else None

    def list_revisions_for_task(self, task_id: str) -> list[Any]:
        """Lists all ScriptRevisions created for a task in chronological order."""
        from app.persistence.converters import script_revision_from_orm
        from app.persistence.models import ScriptRevisionORM

        stmt = (
            select(ScriptRevisionORM)
            .where(ScriptRevisionORM.task_id == task_id)
            .order_by(
                ScriptRevisionORM.revision_number.asc(),
                ScriptRevisionORM.created_at.asc(),
            )
        )
        return [script_revision_from_orm(r) for r in self._session.scalars(stmt).all()]


class AudioOutputRepository:
    """Repository for persisting and querying immutable AudioOutput records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_audio_output(self, audio_output: Any) -> Any:
        """Persists a new AudioOutput."""
        from app.persistence.converters import (
            audio_output_from_orm,
            audio_output_to_orm,
        )
        from app.persistence.models import AudioOutputORM

        orm = self._session.get(AudioOutputORM, audio_output.audio_output_id)
        if orm is None:
            orm = audio_output_to_orm(audio_output)
            self._session.add(orm)
        self._session.flush()
        return audio_output_from_orm(orm)

    def get_audio_output(self, audio_output_id: str) -> Any | None:
        """Retrieves an AudioOutput by primary key ID."""
        from app.persistence.converters import audio_output_from_orm
        from app.persistence.models import AudioOutputORM

        orm = self._session.get(AudioOutputORM, audio_output_id)
        return audio_output_from_orm(orm) if orm is not None else None

    def get_latest_audio_output_for_task(self, task_id: str) -> Any | None:
        """Returns the latest AudioOutput for a task based on created_at."""
        from app.persistence.converters import audio_output_from_orm
        from app.persistence.models import AudioOutputORM

        stmt = (
            select(AudioOutputORM)
            .where(AudioOutputORM.task_id == task_id)
            .order_by(AudioOutputORM.created_at.desc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return audio_output_from_orm(orm) if orm is not None else None

    def list_audio_outputs_for_task(self, task_id: str) -> list[Any]:
        """Lists all AudioOutputs created for a task in chronological order."""
        from app.persistence.converters import audio_output_from_orm
        from app.persistence.models import AudioOutputORM

        stmt = (
            select(AudioOutputORM)
            .where(AudioOutputORM.task_id == task_id)
            .order_by(AudioOutputORM.created_at.asc())
        )
        return [audio_output_from_orm(r) for r in self._session.scalars(stmt).all()]


class CompositionOutputRepository:
    """Repository for persisting and querying immutable CompositionOutput records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_composition_output(self, composition_output: Any) -> Any:
        """Persists a new CompositionOutput."""
        from app.persistence.converters import (
            composition_output_from_orm,
            composition_output_to_orm,
        )
        from app.persistence.models import CompositionOutputORM

        orm = self._session.get(CompositionOutputORM, composition_output.composition_output_id)
        if orm is None:
            orm = composition_output_to_orm(composition_output)
            self._session.add(orm)
        self._session.flush()
        return composition_output_from_orm(orm)

    def get_composition_output(self, composition_output_id: str) -> Any | None:
        """Retrieves a CompositionOutput by primary key ID."""
        from app.persistence.converters import composition_output_from_orm
        from app.persistence.models import CompositionOutputORM

        orm = self._session.get(CompositionOutputORM, composition_output_id)
        return composition_output_from_orm(orm) if orm is not None else None

    def get_latest_composition_output_for_task(self, task_id: str) -> Any | None:
        """Returns the latest CompositionOutput for a task based on created_at."""
        from app.persistence.converters import composition_output_from_orm
        from app.persistence.models import CompositionOutputORM

        stmt = (
            select(CompositionOutputORM)
            .where(CompositionOutputORM.task_id == task_id)
            .order_by(CompositionOutputORM.created_at.desc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return composition_output_from_orm(orm) if orm is not None else None

    def list_composition_outputs_for_task(self, task_id: str) -> list[Any]:
        """Lists all CompositionOutputs created for a task in chronological order."""
        from app.persistence.converters import composition_output_from_orm
        from app.persistence.models import CompositionOutputORM

        stmt = (
            select(CompositionOutputORM)
            .where(CompositionOutputORM.task_id == task_id)
            .order_by(CompositionOutputORM.created_at.asc())
        )
        return [composition_output_from_orm(r) for r in self._session.scalars(stmt).all()]


class DeliveryManifestRepository:
    """Repository for managing immutable DeliveryManifest artifacts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_delivery_manifest(self, manifest: Any) -> Any:
        """Persists a domain DeliveryManifest."""
        from app.persistence.converters import (
            delivery_manifest_from_orm,
            delivery_manifest_to_orm,
        )
        from app.persistence.models import DeliveryManifestORM

        orm = self._session.get(DeliveryManifestORM, manifest.delivery_manifest_id)
        if orm is None:
            orm = delivery_manifest_to_orm(manifest)
            self._session.add(orm)
        self._session.flush()
        return delivery_manifest_from_orm(orm)

    def get_delivery_manifest(self, delivery_manifest_id: str) -> Any | None:
        """Retrieves a DeliveryManifest by primary key ID."""
        from app.persistence.converters import delivery_manifest_from_orm
        from app.persistence.models import DeliveryManifestORM

        orm = self._session.get(DeliveryManifestORM, delivery_manifest_id)
        return delivery_manifest_from_orm(orm) if orm is not None else None

    def get_delivery_manifest_for_task(self, task_id: str) -> Any | None:
        """Returns the latest DeliveryManifest for a task based on created_at."""
        from app.persistence.converters import delivery_manifest_from_orm
        from app.persistence.models import DeliveryManifestORM

        stmt = (
            select(DeliveryManifestORM)
            .where(DeliveryManifestORM.task_id == task_id)
            .order_by(DeliveryManifestORM.created_at.desc())
            .limit(1)
        )
        orm = self._session.scalars(stmt).first()
        return delivery_manifest_from_orm(orm) if orm is not None else None



