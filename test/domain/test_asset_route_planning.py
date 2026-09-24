from unittest.mock import patch

import pytest

from app.domain.asset_router import (
    AssetCapability,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    ConfiguredModelNotFoundError,
    GenerationMode,
    ModelSelectionMode,
    RouteUnavailableReason,
    RoutingStrategy,
    ShotRoutePlanStatus,
    StaticCapabilityMetadata,
    StoryboardNotApprovedError,
    TierLevel,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)


class FakePlanRepository:
    def __init__(self):
        self._plans: dict[str, ContentPlanRevision] = {}

    def add_revision(self, plan: ContentPlanRevision) -> ContentPlanRevision:
        self._plans[plan.content_plan_revision_id] = plan
        return plan

    def get_revision(self, plan_id: str) -> ContentPlanRevision | None:
        return self._plans.get(plan_id)


class FakeShotRepository:
    def __init__(self):
        self._shots: dict[str, Shot] = {}
        self._revisions: dict[str, ShotRevision] = {}

    def add_shot(self, shot: Shot) -> Shot:
        self._shots[shot.shot_id] = shot
        return shot

    def get_shot(self, shot_id: str) -> Shot | None:
        return self._shots.get(shot_id)

    def add_revision(self, revision: ShotRevision) -> ShotRevision:
        self._revisions[revision.shot_revision_id] = revision
        return revision

    def get_revision(self, rev_id: str) -> ShotRevision | None:
        return self._revisions.get(rev_id)


class FakeStoryboardRepository:
    def __init__(self, shot_repo: FakeShotRepository):
        self._snapshots: dict[str, StoryboardSnapshot] = {}
        self._shot_repo = shot_repo

    def add_snapshot(self, snapshot: StoryboardSnapshot) -> StoryboardSnapshot:
        self._snapshots[snapshot.storyboard_snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> StoryboardSnapshot | None:
        return self._snapshots.get(snapshot_id)

    def get_snapshot_shot_revisions(
        self, snapshot_id: str
    ) -> list[ShotRevision]:
        snap = self._snapshots.get(snapshot_id)
        if not snap:
            return []
        return [
            self._shot_repo.get_revision(rid)
            for rid in snap.shot_revision_ids
            if self._shot_repo.get_revision(rid) is not None
        ]


class FakeRoutePlanRepository:
    def __init__(self):
        self._plans: dict[str, AssetRoutePlan] = {}

    def add_route_plan(self, plan: AssetRoutePlan) -> AssetRoutePlan:
        self._plans[plan.asset_route_plan_id] = plan
        return plan

    def get_route_plan(self, plan_id: str) -> AssetRoutePlan | None:
        return self._plans.get(plan_id)

    def get_latest_route_plan_for_snapshot(
        self, snapshot_id: str
    ) -> AssetRoutePlan | None:
        matching = [
            p
            for p in self._plans.values()
            if p.storyboard_snapshot_id == snapshot_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda p: p.created_at)


def _build_test_capabilities() -> list[AssetCapability]:
    return [
        AssetCapability(
            capability_id="seedance:pro:TEXT_TO_VIDEO",
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
            min_duration=2.0,
            max_duration=12.0,
            enabled=True,
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.HIGH,
                latency_tier=TierLevel.HIGH,
            ),
        ),
        AssetCapability(
            capability_id="wavespeed:fast:TEXT_TO_VIDEO",
            provider="wavespeed",
            model="fast",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
            min_duration=4.0,
            max_duration=15.0,
            enabled=True,
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.MEDIUM,
                cost_tier=TierLevel.LOW,
                latency_tier=TierLevel.LOW,
            ),
        ),
        AssetCapability(
            capability_id="pexels:api:STOCK_SEARCH",
            provider="pexels",
            model="video_search",
            generation_mode=GenerationMode.STOCK_SEARCH,
            supported_visual_types=(VisualType.STOCK_VIDEO,),
            enabled=True,
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.MEDIUM,
                cost_tier=TierLevel.LOW,
                latency_tier=TierLevel.LOW,
            ),
        ),
        AssetCapability(
            capability_id="source:extractor:SOURCE_ASSET_USE",
            provider="source_extractor",
            model="extractor",
            generation_mode=GenerationMode.SOURCE_ASSET_USE,
            supported_visual_types=(VisualType.SOURCE_ASSET,),
            supports_source_asset=True,
            enabled=True,
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.LOW,
                latency_tier=TierLevel.LOW,
            ),
        ),
    ]


def _setup_fixture():
    plan_repo = FakePlanRepository()
    shot_repo = FakeShotRepository()
    sb_repo = FakeStoryboardRepository(shot_repo)
    route_repo = FakeRoutePlanRepository()

    # 1. Content Plan with 2 beats
    b1 = ContentBeat(
        beat_id="b1",
        beat_lineage_id="lin-b1",
        beat_type=BeatType.HOOK,
        order=1,
        intent="Hook viewers",
        target_duration=4.0,
        importance=0.9,
    )
    b2 = ContentBeat(
        beat_id="b2",
        beat_lineage_id="lin-b2",
        beat_type=BeatType.KNOWLEDGE,
        order=2,
        intent="Explain architecture",
        target_duration=5.0,
        importance=0.8,
    )
    plan = ContentPlanRevision(
        content_plan_revision_id="plan-01",
        revision_number=1,
        topic="Microservices",
        overall_target_duration=9.0,
        beats=(b1, b2),
    )
    plan_repo.add_revision(plan)

    # 2. Shots and Revisions
    # Shot 1 in Beat 1 (local_order=1)
    s1 = Shot(shot_id="shot-1", beat_lineage_id="lin-b1", local_order=1)
    shot_repo.add_shot(s1)
    s1_rev1 = ShotRevision(
        shot_revision_id="rev-s1-v1",
        shot_id="shot-1",
        revision_number=1,
        beat_lineage_id="lin-b1",
        created_from_beat_instance_id="b1",
        narration="Look at this architecture.",
        target_duration=4.0,
        visual_goal="Grab attention",
        visual_type=VisualType.AI_VIDEO,
        scene_description="Futuristic network grid",
        generation_prompt="cinematic network grid",
        camera_movement="zoom in",
    )
    shot_repo.add_revision(s1_rev1)

    # Shot 2 in Beat 1 (local_order=2)
    s2 = Shot(shot_id="shot-2", beat_lineage_id="lin-b1", local_order=2)
    shot_repo.add_shot(s2)
    s2_rev1 = ShotRevision(
        shot_revision_id="rev-s2-v1",
        shot_id="shot-2",
        revision_number=1,
        beat_lineage_id="lin-b1",
        created_from_beat_instance_id="b1",
        narration="Here is real server footage.",
        target_duration=4.0,
        visual_goal="Show realism",
        visual_type=VisualType.STOCK_VIDEO,
        scene_description="Data center server racks",
        generation_prompt="server racks blinking",
        camera_movement="pan",
    )
    shot_repo.add_revision(s2_rev1)

    # Shot 3 in Beat 2 (local_order=1)
    s3 = Shot(shot_id="shot-3", beat_lineage_id="lin-b2", local_order=1)
    shot_repo.add_shot(s3)
    s3_rev1 = ShotRevision(
        shot_revision_id="rev-s3-v1",
        shot_id="shot-3",
        revision_number=1,
        beat_lineage_id="lin-b2",
        created_from_beat_instance_id="b2",
        narration="Now let us zoom into services.",
        target_duration=5.0,
        visual_goal="Explain microservice interaction",
        visual_type=VisualType.AI_VIDEO,
        scene_description="Abstract glowing nodes communicating",
        generation_prompt="glowing microservice nodes",
        camera_movement="orbit",
    )
    shot_repo.add_revision(s3_rev1)

    # Approved Storyboard Snapshot
    approved_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-approved-01",
        content_plan_revision_id="plan-01",
        shot_revision_ids=("rev-s1-v1", "rev-s2-v1", "rev-s3-v1"),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    sb_repo.add_snapshot(approved_snap)

    # Draft Storyboard Snapshot
    draft_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-draft-01",
        content_plan_revision_id="plan-01",
        shot_revision_ids=("rev-s1-v1", "rev-s2-v1", "rev-s3-v1"),
        snapshot_state=StoryboardSnapshotState.DRAFT,
    )
    sb_repo.add_snapshot(draft_snap)

    service = AssetRoutePlanningService(
        plan_repository=plan_repo,
        shot_repository=shot_repo,
        storyboard_repository=sb_repo,
        route_plan_repository=route_repo,
    )

    return service, plan_repo, shot_repo, sb_repo, route_repo


# 1. DRAFT Storyboard cannot produce formal AssetRoutePlan.
def test_draft_storyboard_cannot_produce_formal_route_plan():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-draft-01"
    )
    with pytest.raises(StoryboardNotApprovedError) as exc_info:
        service.create_route_plan(inp, capabilities=_build_test_capabilities())
    assert "expected APPROVED" in str(exc_info.value)


# 2. APPROVED Storyboard can produce AssetRoutePlan.
def test_approved_storyboard_can_produce_route_plan():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())
    assert isinstance(plan, AssetRoutePlan)
    assert plan.storyboard_snapshot_id == "snap-approved-01"
    assert plan.status == AssetRoutePlanStatus.READY
    assert plan.is_ready is True


# 3. Route planning uses exact ShotRevision IDs frozen by approved Snapshot.
def test_route_planning_uses_exact_shot_revisions_frozen_in_snapshot():
    service, _, _, sb_repo, _ = _setup_fixture()
    snap = sb_repo.get_snapshot("snap-approved-01")
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    planned_rev_ids = tuple(e.shot_revision_id for e in plan.shot_routes)
    assert planned_rev_ids == snap.shot_revision_ids


# 4. A later ShotRevision is NOT silently selected.
def test_later_shot_revision_not_silently_selected():
    service, _, shot_repo, _, _ = _setup_fixture()
    # Create a newer revision v2 on shot-1 in the DB after snapshot was approved
    newer_rev = ShotRevision(
        shot_revision_id="rev-s1-v2-newer",
        shot_id="shot-1",
        revision_number=2,
        beat_lineage_id="lin-b1",
        created_from_beat_instance_id="b1",
        narration="Unapproved newer narration",
        target_duration=4.0,
        visual_goal="Newer goal",
        visual_type=VisualType.AI_VIDEO,
        scene_description="Newer scene",
        generation_prompt="Newer prompt",
        camera_movement="static",
    )
    shot_repo.add_revision(newer_rev)

    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    s1_entry = next(e for e in plan.shot_routes if e.shot_id == "shot-1")
    assert s1_entry.shot_revision_id == "rev-s1-v1"
    assert s1_entry.shot_revision_id != "rev-s1-v2-newer"


# 5. All Shots are routed in: Beat.order + Shot.local_order.
def test_shots_routed_in_beat_order_then_shot_local_order():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    # b1 (order 1): shot-1 (local 1), shot-2 (local 2)
    # b2 (order 2): shot-3 (local 1)
    planned_shot_ids = [e.shot_id for e in plan.shot_routes]
    assert planned_shot_ids == ["shot-1", "shot-2", "shot-3"]


# 6. Every Shot creates one ShotRoutePlanEntry.
def test_every_shot_creates_one_shot_route_plan_entry():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    assert len(plan.shot_routes) == 3
    assert plan.total_shots == 3


# 7. READY requires every Shot to have a valid route.
def test_ready_requires_every_shot_to_have_valid_route():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    assert plan.status == AssetRoutePlanStatus.READY
    assert plan.is_ready is True
    assert plan.blocked_shots == 0
    assert all(e.route_status == ShotRoutePlanStatus.ROUTED for e in plan.shot_routes)


# 8. One ROUTE_UNAVAILABLE produces BLOCKED plan.
def test_one_route_unavailable_produces_blocked_plan():
    service, _, shot_repo, sb_repo, _ = _setup_fixture()
    # Add a shot requesting SOURCE_ASSET without source asset references
    s4 = Shot(shot_id="shot-4", beat_lineage_id="lin-b2", local_order=2)
    shot_repo.add_shot(s4)
    s4_rev = ShotRevision(
        shot_revision_id="rev-s4-v1",
        shot_id="shot-4",
        revision_number=1,
        beat_lineage_id="lin-b2",
        created_from_beat_instance_id="b2",
        narration="Show original source diagram.",
        target_duration=5.0,
        visual_goal="Show primary source",
        visual_type=VisualType.SOURCE_ASSET,
        scene_description="Document figure",
        generation_prompt="",
        camera_movement="static",
        evidence_refs=(),  # Missing!
    )
    shot_repo.add_revision(s4_rev)

    blocked_snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-with-blocked-shot",
        content_plan_revision_id="plan-01",
        shot_revision_ids=("rev-s1-v1", "rev-s2-v1", "rev-s4-v1"),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    sb_repo.add_snapshot(blocked_snap)

    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-with-blocked-shot"
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    assert plan.status == AssetRoutePlanStatus.BLOCKED
    assert plan.is_blocked is True
    assert plan.blocked_shots == 1
    assert plan.routed_shots == 2


# 9. BLOCKED plan preserves blocked Shot/reason information.
def test_blocked_plan_preserves_blocked_shot_and_reason():
    service, _, shot_repo, sb_repo, _ = _setup_fixture()
    s4 = Shot(shot_id="shot-blocked", beat_lineage_id="lin-b2", local_order=2)
    shot_repo.add_shot(s4)
    s4_rev = ShotRevision(
        shot_revision_id="rev-blocked-v1",
        shot_id="shot-blocked",
        revision_number=1,
        beat_lineage_id="lin-b2",
        created_from_beat_instance_id="b2",
        narration="Show original source diagram.",
        target_duration=5.0,
        visual_goal="Show primary source",
        visual_type=VisualType.SOURCE_ASSET,
        scene_description="Document figure",
        generation_prompt="",
        camera_movement="static",
        evidence_refs=(),
    )
    shot_repo.add_revision(s4_rev)

    snap = StoryboardSnapshot(
        storyboard_snapshot_id="snap-blocked-audit",
        content_plan_revision_id="plan-01",
        shot_revision_ids=("rev-blocked-v1",),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    sb_repo.add_snapshot(snap)

    inp = CreateAssetRoutePlanInput(approved_storyboard_snapshot_id="snap-blocked-audit")
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    blocked_entry = plan.shot_routes[0]
    assert blocked_entry.route_status == ShotRoutePlanStatus.BLOCKED
    assert blocked_entry.shot_id == "shot-blocked"
    assert blocked_entry.shot_revision_id == "rev-blocked-v1"
    assert RouteUnavailableReason.MISSING_SOURCE_ASSET in blocked_entry.route_decision.decision_reason_codes


# 10. HybridAssetRouter receives requests built from exact approved ShotRevision fields.
def test_router_receives_exact_approved_shot_fields():
    service, _, shot_repo, _, _ = _setup_fixture()
    s1_rev = shot_repo.get_revision("rev-s1-v1")
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        target_aspect_ratio="16:9",
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    entry = next(e for e in plan.shot_routes if e.shot_id == "shot-1")
    req = entry.asset_routing_request
    assert req.requested_visual_type == s1_rev.visual_type
    assert req.target_duration == s1_rev.target_duration
    assert req.visual_goal == s1_rev.visual_goal
    assert req.scene_description == s1_rev.scene_description
    assert req.generation_prompt == s1_rev.generation_prompt
    assert req.camera_movement == s1_rev.camera_movement
    assert req.aspect_ratio == "16:9"


# 11. Same frozen RoutingPolicy version is used for every Shot.
def test_same_frozen_policy_version_used_for_every_shot():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        routing_strategy=RoutingStrategy.QUALITY_FIRST,
    )
    plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())

    assert plan.routing_policy_version == "v1.0"
    for e in plan.shot_routes:
        assert e.route_decision.routing_policy_version == "v1.0"
        assert e.route_decision.routing_strategy == RoutingStrategy.QUALITY_FIRST


# 12. Same capability set is used throughout one plan operation.
def test_same_capability_set_used_throughout_one_plan():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    for e in plan.shot_routes:
        total_eval = len(e.route_decision.eligible_candidates) + len(e.route_decision.rejected_candidates)
        assert total_eval == len(caps)


# 13. AUTO can consider multiple models exposed by existing MPT configuration.
def test_auto_mode_considers_multiple_configured_models():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.AUTO,
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    # Shot 1 is AI_VIDEO, both Seedance and WaveSpeed are eligible and considered
    s1_entry = plan.shot_routes[0]
    evaluated_ids = {c.capability_id for c in s1_entry.route_decision.eligible_candidates}
    assert "seedance:pro:TEXT_TO_VIDEO" in evaluated_ids
    assert "wavespeed:fast:TEXT_TO_VIDEO" in evaluated_ids


# 14. PINNED considers only the selected configured Provider + Model.
def test_pinned_mode_considers_only_selected_configured_model():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.PINNED,
        selected_provider="wavespeed",
        selected_model="fast",
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    s1_entry = plan.shot_routes[0]
    assert s1_entry.route_decision.selected_candidate is not None
    assert s1_entry.route_decision.selected_candidate.provider == "wavespeed"
    assert s1_entry.route_decision.selected_candidate.model == "fast"
    # Seedance was excluded from consideration
    evaluated_ids = {c.capability_id for c in s1_entry.route_decision.eligible_candidates}
    assert "seedance:pro:TEXT_TO_VIDEO" not in evaluated_ids


# 15. PINNED does not silently fallback.
def test_pinned_mode_does_not_silently_fallback():
    service, _, _, _, _ = _setup_fixture()
    # PINNED to wavespeed (AI_VIDEO provider), but shot 2 is STOCK_VIDEO
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.PINNED,
        selected_provider="wavespeed",
        selected_model="fast",
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    # Shot 2 cannot use wavespeed -> MUST be BLOCKED, cannot silently fallback to pexels!
    s2_entry = plan.shot_routes[1]
    assert s2_entry.route_status == ShotRoutePlanStatus.BLOCKED
    assert s2_entry.route_decision.selected_candidate is None
    assert RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE in s2_entry.route_decision.decision_reason_codes
    assert plan.status == AssetRoutePlanStatus.BLOCKED


# 16. PREFERRED uses preferred model when eligible, while allowing deterministic fallback when unavailable.
def test_preferred_mode_uses_preferred_when_eligible_and_falls_back():
    service, _, _, _, _ = _setup_fixture()
    # Preferred is seedance (pro)
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.PREFERRED,
        selected_provider="seedance",
        selected_model="pro",
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    # Shot 1 (AI_VIDEO): seedance is eligible and preferred -> selected
    s1_entry = plan.shot_routes[0]
    assert s1_entry.route_decision.selected_candidate.provider == "seedance"

    # Shot 2 (STOCK_VIDEO): seedance cannot satisfy stock video -> falls back to pexels!
    s2_entry = plan.shot_routes[1]
    assert s2_entry.route_status == ShotRoutePlanStatus.ROUTED
    assert s2_entry.route_decision.selected_candidate.provider == "pexels"
    assert plan.status == AssetRoutePlanStatus.READY


# 17. Unknown configured model reference is rejected.
def test_unknown_configured_model_reference_rejected():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.PINNED,
        selected_provider="unknown_provider",
        selected_model="unknown_model",
    )
    caps = _build_test_capabilities()
    with pytest.raises(ConfiguredModelNotFoundError) as exc_info:
        service.create_route_plan(inp, capabilities=caps)
    assert "not found in available capabilities" in str(exc_info.value)


# 18. Existing MPT provider/model configuration remains source of truth.
def test_mpt_provider_configuration_remains_source_of_truth():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        model_selection_mode=ModelSelectionMode.AUTO,
    )
    caps = _build_test_capabilities()
    plan = service.create_route_plan(inp, capabilities=caps)

    # All selected providers come strictly from caps
    valid_providers = {c.provider for c in caps}
    for e in plan.shot_routes:
        if e.route_decision.selected_candidate:
            assert e.route_decision.selected_candidate.provider in valid_providers


# 19. No provider execution method is called.
def test_zero_provider_execution_called_during_route_planning():
    service, _, _, _, _ = _setup_fixture()
    inp = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01"
    )

    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        plan = service.create_route_plan(inp, capabilities=_build_test_capabilities())
        assert plan.status == AssetRoutePlanStatus.READY
        assert mock_post.call_count == 0
        assert mock_get.call_count == 0


# 20. Creating a new strategy results in a NEW AssetRoutePlan, not mutation of old plan.
def test_new_strategy_creates_new_route_plan_not_mutation():
    service, _, _, _, route_repo = _setup_fixture()
    inp_balanced = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        routing_strategy=RoutingStrategy.BALANCED,
    )
    plan_1 = service.create_route_plan(inp_balanced, capabilities=_build_test_capabilities())

    inp_quality = CreateAssetRoutePlanInput(
        approved_storyboard_snapshot_id="snap-approved-01",
        routing_strategy=RoutingStrategy.QUALITY_FIRST,
    )
    plan_2 = service.create_route_plan(inp_quality, capabilities=_build_test_capabilities())

    assert plan_1.asset_route_plan_id != plan_2.asset_route_plan_id
    assert plan_1.routing_strategy == RoutingStrategy.BALANCED
    assert plan_2.routing_strategy == RoutingStrategy.QUALITY_FIRST
    assert route_repo.get_route_plan(plan_1.asset_route_plan_id) is not None
    assert route_repo.get_route_plan(plan_2.asset_route_plan_id) is not None


# 21. A different APPROVED StoryboardSnapshot requires a new RoutePlan.
def test_different_approved_snapshot_requires_new_route_plan():
    service, _, _, sb_repo, _ = _setup_fixture()
    snap_2 = StoryboardSnapshot(
        storyboard_snapshot_id="snap-approved-02",
        content_plan_revision_id="plan-01",
        shot_revision_ids=("rev-s1-v1",),
        snapshot_state=StoryboardSnapshotState.APPROVED,
    )
    sb_repo.add_snapshot(snap_2)

    plan_snap_1 = service.create_route_plan(
        CreateAssetRoutePlanInput(approved_storyboard_snapshot_id="snap-approved-01"),
        capabilities=_build_test_capabilities(),
    )
    plan_snap_2 = service.create_route_plan(
        CreateAssetRoutePlanInput(approved_storyboard_snapshot_id="snap-approved-02"),
        capabilities=_build_test_capabilities(),
    )

    assert plan_snap_1.asset_route_plan_id != plan_snap_2.asset_route_plan_id
    assert plan_snap_1.storyboard_snapshot_id == "snap-approved-01"
    assert plan_snap_2.storyboard_snapshot_id == "snap-approved-02"
    assert plan_snap_1.total_shots == 3
    assert plan_snap_2.total_shots == 1
