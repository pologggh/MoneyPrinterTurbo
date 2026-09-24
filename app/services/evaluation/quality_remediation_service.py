from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.domain.asset_execution import (
    ShotAssetVersion,
    ShotExecution,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutingRequest,
)
from app.domain.content_plan import ContentBeat
from app.domain.evaluation import (
    DimensionEvaluationResult,
    EvaluationSnapshot,
)
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationDecision,
    QualityRemediationPolicy,
    QualityRemediationReasonCode,
    ShotQualitySelection,
)
from app.domain.quality_remediation_policy import QualityRemediationPolicyEngine
from app.domain.shot import ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.services.evaluation.controlled_visual_replan import (
    ControlledVisualReplanRequest,
    ControlledVisualReplanService,
)
from app.services.hybrid_asset_router import HybridAssetRouter
from app.services.shot_execution_service import ShotExecutionService


class QualityRemediationResult(BaseModel):
    """
    Final outcome of executing a quality remediation chain for a Shot.
    Preserves all historical assets, snapshots, and remediation decisions.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_chain_id: str
    shot_id: str
    final_action: QualityRemediationAction
    status: str  # "ACCEPTED" or "NEEDS_USER_ACTION"
    accepted_asset_version: ShotAssetVersion | None = None
    accepted_evaluation_snapshot: EvaluationSnapshot | None = None
    quality_selection: ShotQualitySelection | None = None
    remediation_decisions: tuple[QualityRemediationDecision, ...] = ()
    all_generated_asset_versions: tuple[ShotAssetVersion, ...] = ()
    all_evaluation_snapshots: tuple[EvaluationSnapshot, ...] = ()
    candidate_draft_snapshot: StoryboardSnapshot | None = None


# Evaluation function protocol: (ShotRevision, ShotAssetVersion) -> (EvaluationSnapshot, list[DimensionEvaluationResult])
EvaluationCallable = Callable[
    [ShotRevision, ShotAssetVersion],
    tuple[EvaluationSnapshot, list[DimensionEvaluationResult]],
]


class QualityRemediationService:
    """
    Orchestrates the deterministic quality remediation cycle for a Shot:
    1. Evaluates snapshot with QualityRemediationPolicyEngine.
    2. Executes allowed remediation action:
       - ACCEPT_ASSET: Creates immutable ShotQualitySelection.
       - RETRY_EVALUATION: Re-evaluates exact target (bounded to max_evaluation_retries).
       - REGENERATE_SAME_ROUTE: Executes same route candidate without re-routing.
       - FALLBACK_NEXT_ROUTE_CANDIDATE: Advances to next frozen candidate without re-routing.
       - CONTROLLED_VISUAL_REPLAN: Generates new ShotRevision, re-routes, executes 1 cycle.
       - NEEDS_USER_ACTION: Halts remediation.
    3. Loop is guaranteed to terminate within configured budgets.
    """

    def __init__(
        self,
        execution_service: ShotExecutionService | None = None,
        router: HybridAssetRouter | None = None,
        policy: QualityRemediationPolicy | None = None,
        storage_base_dir: Path | str = "data/storage",
    ):
        self._execution_service = execution_service or ShotExecutionService()
        self._router = router or HybridAssetRouter()
        self._policy = policy or QualityRemediationPolicy()
        self._storage_base_dir = Path(storage_base_dir)

    def run_remediation_chain(
        self,
        shot_revision: ShotRevision,
        initial_asset_version: ShotAssetVersion,
        initial_snapshot: EvaluationSnapshot,
        initial_dimension_results: Sequence[DimensionEvaluationResult],
        route_decision: AssetRouteDecision,
        beat: ContentBeat,
        evaluation_fn: EvaluationCallable,
        approved_storyboard_snapshot: StoryboardSnapshot | None = None,
        quality_chain_id: str | None = None,
        max_loop_iterations: int = 6,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> QualityRemediationResult:
        chain_id = quality_chain_id or str(uuid4())
        history: list[QualityRemediationDecision] = []
        all_assets: list[ShotAssetVersion] = [initial_asset_version]
        all_snapshots: list[EvaluationSnapshot] = [initial_snapshot]

        current_revision = shot_revision
        current_asset = initial_asset_version
        current_snapshot = initial_snapshot
        current_dims = list(initial_dimension_results)
        current_route_decision = route_decision
        candidate_draft_snapshot: StoryboardSnapshot | None = None

        iteration = 0

        while iteration < max_loop_iterations:
            iteration += 1

            # Step 1: Deterministic policy engine decision
            decision = QualityRemediationPolicyEngine.evaluate(
                snapshot=current_snapshot,
                dimension_results=current_dims,
                route_decision=current_route_decision,
                history=history,
                policy=self._policy,
                quality_chain_id=chain_id,
                shot_id=current_revision.shot_id,
                shot_revision_id=current_revision.shot_revision_id,
                shot_asset_version_id=current_asset.shot_asset_version_id,
            )
            history.append(decision)

            remed_ctx = None
            if trace_writer is not None and trace_context is not None:
                from app.domain.trace import (
                    QualityAssetAcceptedTraceData,
                    RemediationTraceData,
                    TraceEventStatus,
                    TraceEventType,
                )

                remed_ctx = trace_context.child_context(
                    parent_event_id=trace_context.parent_event_id,
                    shot_id=current_revision.shot_id,
                    shot_revision_id=current_revision.shot_revision_id,
                    shot_asset_version_id=current_asset.shot_asset_version_id,
                    evaluation_snapshot_id=current_snapshot.evaluation_snapshot_id,
                )
                remed_data = RemediationTraceData(
                    remediation_decision_id=decision.remediation_decision_id,
                    quality_chain_id=chain_id,
                    source_evaluation_snapshot_id=current_snapshot.evaluation_snapshot_id,
                    action=decision.action.value if hasattr(decision.action, "value") else str(decision.action),
                    reason_codes=tuple(
                        rc.value if hasattr(rc, "value") else str(rc) for rc in decision.reason_codes
                    ),
                    quality_attempt_index=iteration,
                    old_shot_asset_version_id=current_asset.shot_asset_version_id,
                    old_route_candidate=current_route_decision.selected_candidate.capability_id
                    if current_route_decision.selected_candidate
                    else None,
                    new_route_candidate=decision.selected_route_candidate.capability_id
                    if decision.selected_route_candidate
                    else None,
                )
                trace_writer.record_event(
                    context=remed_ctx,
                    event_type=TraceEventType.QUALITY_REMEDIATION_DECIDED,
                    status=TraceEventStatus.SUCCEEDED,
                    attributes=remed_data,
                )

            # Step 2: Handle Terminal Actions
            if decision.action == QualityRemediationAction.ACCEPT_ASSET:
                selection = ShotQualitySelection(
                    quality_chain_id=chain_id,
                    shot_id=current_revision.shot_id,
                    shot_revision_id=current_revision.shot_revision_id,
                    shot_asset_version_id=current_asset.shot_asset_version_id,
                    evaluation_snapshot_id=current_snapshot.evaluation_snapshot_id,
                    selection_source="EVALUATION_PASS",
                )

                if trace_writer is not None and remed_ctx is not None:
                    accept_data = QualityAssetAcceptedTraceData(
                        quality_selection_id=selection.quality_selection_id,
                        shot_id=current_revision.shot_id,
                        shot_revision_id=current_revision.shot_revision_id,
                        accepted_shot_asset_version_id=current_asset.shot_asset_version_id,
                        evaluation_snapshot_id=current_snapshot.evaluation_snapshot_id,
                    )
                    trace_writer.record_event(
                        context=remed_ctx,
                        event_type=TraceEventType.QUALITY_ASSET_ACCEPTED,
                        status=TraceEventStatus.SUCCEEDED,
                        attributes=accept_data,
                    )
                return QualityRemediationResult(
                    quality_chain_id=chain_id,
                    shot_id=current_revision.shot_id,
                    final_action=QualityRemediationAction.ACCEPT_ASSET,
                    status="ACCEPTED",
                    accepted_asset_version=current_asset,
                    accepted_evaluation_snapshot=current_snapshot,
                    quality_selection=selection,
                    remediation_decisions=tuple(history),
                    all_generated_asset_versions=tuple(all_assets),
                    all_evaluation_snapshots=tuple(all_snapshots),
                    candidate_draft_snapshot=candidate_draft_snapshot,
                )

            if decision.action == QualityRemediationAction.NEEDS_USER_ACTION:
                return QualityRemediationResult(
                    quality_chain_id=chain_id,
                    shot_id=current_revision.shot_id,
                    final_action=QualityRemediationAction.NEEDS_USER_ACTION,
                    status="NEEDS_USER_ACTION",
                    accepted_asset_version=None,
                    accepted_evaluation_snapshot=None,
                    quality_selection=None,
                    remediation_decisions=tuple(history),
                    all_generated_asset_versions=tuple(all_assets),
                    all_evaluation_snapshots=tuple(all_snapshots),
                    candidate_draft_snapshot=candidate_draft_snapshot,
                )

            # Step 3: Handle RETRY_EVALUATION
            if decision.action == QualityRemediationAction.RETRY_EVALUATION:
                new_snap, new_dims = evaluation_fn(current_revision, current_asset)
                all_snapshots.append(new_snap)
                current_snapshot = new_snap
                current_dims = list(new_dims)
                continue

            # Step 4: Handle REGENERATE_SAME_ROUTE
            if decision.action == QualityRemediationAction.REGENERATE_SAME_ROUTE:
                # Same route candidate, no router call
                new_asset = self._execute_candidate(
                    shot_revision=current_revision,
                    candidate=current_route_decision.selected_candidate,
                    route_decision=current_route_decision,
                )
                if new_asset is None:
                    # Technical generation failure: handled as technical issue, terminates to user action
                    break
                all_assets.append(new_asset)
                current_asset = new_asset
                new_snap, new_dims = evaluation_fn(current_revision, current_asset)
                all_snapshots.append(new_snap)
                current_snapshot = new_snap
                current_dims = list(new_dims)
                continue

            # Step 5: Handle FALLBACK_NEXT_ROUTE_CANDIDATE
            if decision.action == QualityRemediationAction.FALLBACK_NEXT_ROUTE_CANDIDATE:
                fallback_candidate = decision.selected_route_candidate
                if fallback_candidate is None:
                    break
                # Do NOT rerun router. Update current route decision selected candidate to the fallback
                current_route_decision = current_route_decision.model_copy(
                    update={"selected_candidate": fallback_candidate}
                )
                new_asset = self._execute_candidate(
                    shot_revision=current_revision,
                    candidate=fallback_candidate,
                    route_decision=current_route_decision,
                )
                if new_asset is None:
                    break
                all_assets.append(new_asset)
                current_asset = new_asset
                new_snap, new_dims = evaluation_fn(current_revision, current_asset)
                all_snapshots.append(new_snap)
                current_snapshot = new_snap
                current_dims = list(new_dims)
                continue

            # Step 6: Handle CONTROLLED_VISUAL_REPLAN
            if decision.action == QualityRemediationAction.CONTROLLED_VISUAL_REPLAN:
                replan_req = ControlledVisualReplanRequest(
                    shot_revision=current_revision,
                    beat=beat,
                    evaluation_snapshot=current_snapshot,
                )
                replan_res = ControlledVisualReplanService.create_candidate_revision(
                    replan_req,
                    existing_approved_snapshot=approved_storyboard_snapshot,
                )
                current_revision = replan_res.candidate_shot_revision
                candidate_draft_snapshot = replan_res.candidate_draft_snapshot

                if trace_writer is not None and trace_context is not None and remed_ctx is not None:
                    trace_writer.record_event(
                        context=remed_ctx.child_context(
                            parent_event_id=remed_ctx.parent_event_id,
                            shot_revision_id=current_revision.shot_revision_id,
                            storyboard_snapshot_id=candidate_draft_snapshot.storyboard_snapshot_id
                            if candidate_draft_snapshot
                            else None,
                        ),
                        event_type=TraceEventType.CONTROLLED_VISUAL_REPLAN_CREATED,
                        status=TraceEventStatus.SUCCEEDED,
                        attributes={
                            "quality_chain_id": chain_id,
                            "shot_id": current_revision.shot_id,
                            "original_shot_revision_id": shot_revision.shot_revision_id,
                            "candidate_shot_revision_id": current_revision.shot_revision_id,
                            "candidate_draft_snapshot_id": candidate_draft_snapshot.storyboard_snapshot_id
                            if candidate_draft_snapshot
                            else None,
                            "trigger_evaluation_snapshot_id": current_snapshot.evaluation_snapshot_id,
                        },
                    )

                # Legitimate Router Call: Visual requirements changed
                routing_req = AssetRoutingRequest(
                    shot_id=current_revision.shot_id,
                    shot_revision_id=current_revision.shot_revision_id,
                    requested_visual_type=current_revision.visual_type,
                    target_duration=current_revision.target_duration,
                    aspect_ratio="16:9",
                    visual_goal=current_revision.visual_goal,
                    scene_description=current_revision.scene_description,
                    generation_prompt=current_revision.generation_prompt,
                    camera_movement=current_revision.camera_movement,
                )
                current_route_decision = self._router.route(routing_req)

                if (
                    not current_route_decision.selected_candidate
                    or not current_route_decision.selected_candidate.is_eligible
                ):
                    break

                new_asset = self._execute_candidate(
                    shot_revision=current_revision,
                    candidate=current_route_decision.selected_candidate,
                    route_decision=current_route_decision,
                )
                if new_asset is None:
                    break
                all_assets.append(new_asset)
                current_asset = new_asset
                new_snap, new_dims = evaluation_fn(current_revision, current_asset)
                all_snapshots.append(new_snap)
                current_snapshot = new_snap
                current_dims = list(new_dims)
                continue

        # If loop exited without explicit return (e.g. max iterations reached or execution failed)
        final_dec = QualityRemediationDecision(
            quality_chain_id=chain_id,
            evaluation_snapshot_id=current_snapshot.evaluation_snapshot_id,
            shot_id=current_revision.shot_id,
            shot_revision_id=current_revision.shot_revision_id,
            shot_asset_version_id=current_asset.shot_asset_version_id,
            action=QualityRemediationAction.NEEDS_USER_ACTION,
            reason_codes=(QualityRemediationReasonCode.TOTAL_BUDGET_EXHAUSTED.value,),
            remediation_policy_version=self._policy.policy_version,
            quality_attempt_index=len(history),
            selected_route_candidate=current_route_decision.selected_candidate,
            explanation="Remediation loop terminated: max loop iterations reached or execution halted.",
        )
        history.append(final_dec)
        return QualityRemediationResult(
            quality_chain_id=chain_id,
            shot_id=current_revision.shot_id,
            final_action=QualityRemediationAction.NEEDS_USER_ACTION,
            status="NEEDS_USER_ACTION",
            accepted_asset_version=None,
            accepted_evaluation_snapshot=None,
            quality_selection=None,
            remediation_decisions=tuple(history),
            all_generated_asset_versions=tuple(all_assets),
            all_evaluation_snapshots=tuple(all_snapshots),
            candidate_draft_snapshot=candidate_draft_snapshot,
        )

    def _execute_candidate(
        self,
        shot_revision: ShotRevision,
        candidate: AssetRouteCandidate,
        route_decision: AssetRouteDecision,
    ) -> ShotAssetVersion | None:
        """
        Executes an attempt for a specific candidate route via Phase 5 ShotExecutionService.
        """
        shot_execution = ShotExecution(
            execution_run_id=f"run_{uuid4().hex[:8]}",
            shot_id=shot_revision.shot_id,
            shot_revision_id=shot_revision.shot_revision_id,
            route_decision=route_decision,
        )
        routing_request = AssetRoutingRequest(
            shot_id=shot_revision.shot_id,
            shot_revision_id=shot_revision.shot_revision_id,
            requested_visual_type=shot_revision.visual_type,
            target_duration=shot_revision.target_duration,
            aspect_ratio="16:9",
            visual_goal=shot_revision.visual_goal,
            scene_description=shot_revision.scene_description,
            generation_prompt=shot_revision.generation_prompt,
            camera_movement=shot_revision.camera_movement,
        )
        _, _, asset_version = self._execution_service.execute_shot(
            shot_execution=shot_execution,
            request=routing_request,
            storage_base_dir=self._storage_base_dir,
        )
        return asset_version
