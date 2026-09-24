from __future__ import annotations

from datetime import datetime
from typing import Any

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
from app.domain.plan_diff import validate_content_plan_revision_lineages
from app.domain.quality_remediation import (
    QualityRemediationDecision,
    ShotQualitySelection,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.domain.trace import TraceEvent, TraceEventStatus, TraceRoot
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
    storyboard_approval_record_from_orm,
    storyboard_approval_record_to_orm,
    storyboard_snapshot_from_orm,
    storyboard_snapshot_to_orm,
    trace_event_from_orm,
    trace_event_to_orm,
    trace_root_from_orm,
    trace_root_to_orm,
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
    ProviderReceiptORM,
    QualityRemediationDecisionORM,
    ShotAssetVersionORM,
    ShotExecutionORM,
    ShotORM,
    ShotQualitySelectionORM,
    ShotRevisionORM,
    StoryboardApprovalRecordORM,
    StoryboardSnapshotORM,
    TraceEventORM,
    TraceRootORM,
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


