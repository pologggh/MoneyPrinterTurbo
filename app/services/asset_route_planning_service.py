"""
Asset Route Planning Service (Phase 4.3).

Orchestrates the conversion of an APPROVED StoryboardSnapshot into an immutable
AssetRoutePlan, evaluating deterministic routing for all shots in storyboard order
without executing media generation providers.
"""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.asset_router import (
    AssetCapability,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    ConfiguredModelNotFoundError,
    ModelSelectionMode,
    RoutingPolicy,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
    StoryboardNotApprovedError,
    create_routing_request_from_shot_revision,
    get_default_policy,
)
from app.domain.enums import StoryboardSnapshotState
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_editing import StoryboardNotFoundError
from app.services.asset_capability_registry import (
    AssetCapabilityRegistry,
    get_default_capability_registry,
)
from app.services.hybrid_asset_router import HybridAssetRouter, filter_hard_constraints


class CreateAssetRoutePlanInput(BaseModel):
    """
    Input data for planning asset routes for an approved storyboard snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    approved_storyboard_snapshot_id: str
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    model_selection_mode: ModelSelectionMode = ModelSelectionMode.AUTO
    selected_provider: str | None = None
    selected_model: str | None = None
    target_aspect_ratio: str | None = None
    source_asset_refs_map: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Optional mapping of shot_id -> source_asset_refs overrides",
    )
    user_asset_refs_map: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Optional mapping of shot_id -> user_asset_refs",
    )


class AssetRoutePlanningService:
    """
    Application service orchestrating whole-storyboard asset route planning.
    """

    def __init__(
        self,
        plan_repository: Any = None,
        shot_repository: Any = None,
        storyboard_repository: Any = None,
        route_plan_repository: Any = None,
        capability_registry: AssetCapabilityRegistry | None = None,
        router: HybridAssetRouter | None = None,
    ):
        self._plan_repo = plan_repository
        self._shot_repo = shot_repository
        self._storyboard_repo = storyboard_repository
        self._route_plan_repo = route_plan_repository
        self._registry = capability_registry or get_default_capability_registry()
        self._router = router or HybridAssetRouter(registry=self._registry)

    def create_route_plan(
        self,
        input_data: CreateAssetRoutePlanInput,
        capabilities: Sequence[AssetCapability] | None = None,
        policy: RoutingPolicy | None = None,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> AssetRoutePlan:
        """
        Generates an immutable AssetRoutePlan for an APPROVED StoryboardSnapshot.
        """
        # 1. Validate snapshot existence
        snapshot: StoryboardSnapshot | None = self._storyboard_repo.get_snapshot(
            input_data.approved_storyboard_snapshot_id
        )
        if snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{input_data.approved_storyboard_snapshot_id}' not found."
            )

        # 2. Enforce APPROVED Storyboard requirement
        if snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
            raise StoryboardNotApprovedError(
                f"Cannot route storyboard snapshot '{snapshot.storyboard_snapshot_id}' "
                f"because state is '{snapshot.snapshot_state.value}', expected APPROVED."
            )

        # 3. Resolve ContentPlanRevision for beat ordering
        plan_rev = self._plan_repo.get_revision(snapshot.content_plan_revision_id)
        if plan_rev is None:
            raise StoryboardNotFoundError(
                f"ContentPlanRevision '{snapshot.content_plan_revision_id}' not found."
            )
        beat_order_map = {b.beat_lineage_id: b.order for b in plan_rev.beats}

        # 4. Resolve exact frozen ShotRevisions (NEVER query latest shot revisions)
        frozen_revisions: Sequence[ShotRevision] = (
            self._storyboard_repo.get_snapshot_shot_revisions(
                snapshot.storyboard_snapshot_id
            )
        )
        if len(frozen_revisions) != len(snapshot.shot_revision_ids):
            raise StoryboardNotFoundError(
                "Mismatch between snapshot revision count and resolved shot revisions."
            )

        # 5. Order shots deterministically by ContentBeat.order, then Shot.local_order
        shot_order_pairs: list[tuple[int, int, ShotRevision]] = []
        for rev in frozen_revisions:
            shot_entity: Shot | None = self._shot_repo.get_shot(rev.shot_id)
            local_order = shot_entity.local_order if shot_entity is not None else 1
            b_order = beat_order_map.get(rev.beat_lineage_id, 999999)
            shot_order_pairs.append((b_order, local_order, rev))

        shot_order_pairs.sort(key=lambda t: (t[0], t[1], t[2].shot_id))
        sorted_revisions = [t[2] for t in shot_order_pairs]

        # 6. Freeze in-memory capability set and routing policy for entire plan operation
        frozen_capabilities = (
            tuple(capabilities)
            if capabilities is not None
            else tuple(self._registry.list_capabilities())
        )
        frozen_policy = policy or get_default_policy(input_data.routing_strategy)

        # 7. Validate PINNED and PREFERRED model existence
        if input_data.model_selection_mode in (
            ModelSelectionMode.PINNED,
            ModelSelectionMode.PREFERRED,
        ):
            if not input_data.selected_provider or not input_data.selected_model:
                raise ConfiguredModelNotFoundError(
                    f"ModelSelectionMode '{input_data.model_selection_mode.value}' requires "
                    f"'selected_provider' and 'selected_model' to be specified."
                )
            # Must exist in the active capability set
            found_model = any(
                c.provider == input_data.selected_provider
                and c.model == input_data.selected_model
                for c in frozen_capabilities
            )
            if not found_model:
                raise ConfiguredModelNotFoundError(
                    f"Configured model '{input_data.selected_model}' for provider "
                    f"'{input_data.selected_provider}' not found in available capabilities."
                )

        # 8. Route every ShotRevision in deterministic order
        shot_route_entries: list[ShotRoutePlanEntry] = []
        plan_id = str(uuid4())

        for rev in sorted_revisions:
            source_refs = input_data.source_asset_refs_map.get(
                rev.shot_id, rev.evidence_refs
            )
            user_refs = input_data.user_asset_refs_map.get(rev.shot_id, ())

            routing_req: AssetRoutingRequest = (
                create_routing_request_from_shot_revision(
                    rev,
                    aspect_ratio=input_data.target_aspect_ratio,
                    routing_strategy=input_data.routing_strategy,
                    source_asset_refs=tuple(source_refs),
                    user_asset_refs=tuple(user_refs),
                )
            )

            # Resolve capability scope for this shot based on selection mode
            candidate_caps = frozen_capabilities
            if input_data.model_selection_mode == ModelSelectionMode.PINNED:
                pinned_caps = tuple(
                    c
                    for c in frozen_capabilities
                    if c.provider == input_data.selected_provider
                    and c.model == input_data.selected_model
                )
                candidate_caps = pinned_caps
            elif input_data.model_selection_mode == ModelSelectionMode.PREFERRED:
                # PREFERRED: If the preferred model supports this visual type and passes hard constraints,
                # prioritize it. If not, allow other eligible capabilities to participate.
                preferred_caps = tuple(
                    c
                    for c in frozen_capabilities
                    if c.provider == input_data.selected_provider
                    and c.model == input_data.selected_model
                )
                if preferred_caps:
                    pref_eligible, _ = filter_hard_constraints(
                        preferred_caps[0], routing_req, frozen_policy
                    )
                    if pref_eligible:
                        candidate_caps = preferred_caps

            route_decision: AssetRouteDecision = self._router.route(
                routing_req,
                capabilities=candidate_caps,
                policy=frozen_policy,
                raise_if_unavailable=False,
                trace_context=trace_context,
                trace_writer=trace_writer,
                asset_route_plan_id=plan_id,
            )

            route_status = (
                ShotRoutePlanStatus.ROUTED
                if route_decision.selected_candidate is not None
                else ShotRoutePlanStatus.BLOCKED
            )

            shot_route_entries.append(
                ShotRoutePlanEntry(
                    shot_id=rev.shot_id,
                    shot_revision_id=rev.shot_revision_id,
                    beat_lineage_id=rev.beat_lineage_id,
                    requested_visual_type=rev.visual_type,
                    asset_routing_request=routing_req,
                    route_decision=route_decision,
                    route_status=route_status,
                )
            )

        # 9. Aggregate plan statistics
        total_shots = len(shot_route_entries)
        routed_shots = sum(
            1 for e in shot_route_entries if e.route_status == ShotRoutePlanStatus.ROUTED
        )
        blocked_shots = sum(
            1 for e in shot_route_entries if e.route_status == ShotRoutePlanStatus.BLOCKED
        )

        plan_status = (
            AssetRoutePlanStatus.READY
            if blocked_shots == 0 and total_shots > 0
            else AssetRoutePlanStatus.BLOCKED
        )

        plan = AssetRoutePlan(
            asset_route_plan_id=plan_id,
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            content_plan_revision_id=snapshot.content_plan_revision_id,
            routing_strategy=input_data.routing_strategy,
            routing_policy_version=frozen_policy.policy_version,
            model_selection_mode=input_data.model_selection_mode,
            selected_provider=input_data.selected_provider,
            selected_model=input_data.selected_model,
            shot_routes=tuple(shot_route_entries),
            status=plan_status,
            total_shots=total_shots,
            routed_shots=routed_shots,
            blocked_shots=blocked_shots,
        )

        # 10. Persist atomically if repository is configured
        if self._route_plan_repo is not None:
            self._route_plan_repo.add_route_plan(plan)

        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import TraceEventStatus, TraceEventType

            ctx = trace_context.with_ids(
                storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
                content_plan_revision_id=snapshot.content_plan_revision_id,
                asset_route_plan_id=plan.asset_route_plan_id,
            )
            trace_writer.record_event(
                context=ctx,
                event_type=TraceEventType.ASSET_ROUTE_PLAN_CREATED,
                status=TraceEventStatus.SUCCEEDED,
                attributes={
                    "asset_route_plan_id": plan.asset_route_plan_id,
                    "total_shots": total_shots,
                    "routed_shots": routed_shots,
                    "blocked_shots": blocked_shots,
                    "status": plan.status.value if hasattr(plan.status, "value") else str(plan.status),
                    "routing_strategy": plan.routing_strategy.value if hasattr(plan.routing_strategy, "value") else str(plan.routing_strategy),
                },
            )

        return plan
