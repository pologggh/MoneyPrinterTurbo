from unittest.mock import patch

import pytest

from app.domain.asset_router import (
    AssetCapability,
    AssetRoutingRequest,
    GenerationMode,
    RouteUnavailableError,
    RouteUnavailableReason,
    RoutingPolicy,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
    get_default_policy,
)
from app.domain.enums import VisualType
from app.services.hybrid_asset_router import HybridAssetRouter


def _make_dummy_request(
    requested_visual_type: VisualType = VisualType.AI_VIDEO,
    target_duration: float = 5.0,
    aspect_ratio: str | None = "16:9",
    source_asset_refs: tuple[str, ...] = (),
    user_asset_refs: tuple[str, ...] = (),
    routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
) -> AssetRoutingRequest:
    return AssetRoutingRequest(
        shot_id="shot-test-01",
        shot_revision_id="shot-rev-test-01",
        requested_visual_type=requested_visual_type,
        target_duration=target_duration,
        visual_goal="Explain architecture layers clearly",
        scene_description="Animated diagram of modular pipeline",
        generation_prompt="cinematic architectural visualization",
        camera_movement="pan right",
        aspect_ratio=aspect_ratio,
        source_asset_refs=source_asset_refs,
        user_asset_refs=user_asset_refs,
        routing_strategy=routing_strategy,
    )


# 1. Disabled capability is rejected before scoring.
def test_disabled_capability_rejected_before_scoring():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="provider_disabled:v1:TEXT_TO_VIDEO",
        provider="provider_disabled",
        model="v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=False,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    req = _make_dummy_request()
    decision = router.route(req, capabilities=[cap])

    assert decision.selected_candidate is None
    assert len(decision.eligible_candidates) == 0
    assert len(decision.rejected_candidates) == 1
    rejected = decision.rejected_candidates[0]
    assert rejected.is_eligible is False
    assert rejected.score is None
    assert (
        RouteUnavailableReason.DISABLED_CAPABILITY in rejected.rejection_reasons
        or RouteUnavailableReason.NO_ENABLED_PROVIDER in rejected.rejection_reasons
    )


# 2. Unsupported VisualType is rejected.
def test_unsupported_visual_type_rejected():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="stock_provider:v1:STOCK_SEARCH",
        provider="stock_provider",
        model="v1",
        generation_mode=GenerationMode.STOCK_SEARCH,
        supported_visual_types=(VisualType.STOCK_VIDEO,),
        enabled=True,
    )
    req = _make_dummy_request(requested_visual_type=VisualType.AI_VIDEO)
    decision = router.route(req, capabilities=[cap])

    assert decision.selected_candidate is None
    assert len(decision.rejected_candidates) == 1
    rejected = decision.rejected_candidates[0]
    assert RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE in rejected.rejection_reasons
    assert rejected.score is None


# 3. Unsupported duration is rejected.
def test_unsupported_duration_rejected():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="provider_bounded:v1:TEXT_TO_VIDEO",
        provider="provider_bounded",
        model="v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        min_duration=3.0,
        max_duration=10.0,
        enabled=True,
    )
    # Target duration 15.0s exceeds max_duration 10.0s
    req = _make_dummy_request(target_duration=15.0)
    decision = router.route(req, capabilities=[cap])

    assert decision.selected_candidate is None
    assert len(decision.rejected_candidates) == 1
    rejected = decision.rejected_candidates[0]
    assert RouteUnavailableReason.UNSUPPORTED_DURATION in rejected.rejection_reasons
    assert rejected.score is None


# 4. Missing SOURCE_ASSET reference rejects source route.
def test_missing_source_asset_reference_rejected():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="system:source_extractor:SOURCE_ASSET_USE",
        provider="system",
        model="extractor",
        generation_mode=GenerationMode.SOURCE_ASSET_USE,
        supported_visual_types=(VisualType.SOURCE_ASSET,),
        enabled=True,
    )
    # Empty source_asset_refs
    req = _make_dummy_request(
        requested_visual_type=VisualType.SOURCE_ASSET,
        source_asset_refs=(),
    )
    decision = router.route(req, capabilities=[cap])

    assert decision.selected_candidate is None
    assert len(decision.rejected_candidates) == 1
    assert (
        RouteUnavailableReason.MISSING_SOURCE_ASSET
        in decision.rejected_candidates[0].rejection_reasons
    )


# 5. Missing USER_ASSET reference rejects user route.
def test_missing_user_asset_reference_rejected():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="system:user_loader:USER_ASSET_USE",
        provider="system",
        model="loader",
        generation_mode=GenerationMode.USER_ASSET_USE,
        supported_visual_types=(VisualType.USER_ASSET,),
        enabled=True,
    )
    # Empty user_asset_refs
    req = _make_dummy_request(
        requested_visual_type=VisualType.USER_ASSET,
        user_asset_refs=(),
    )
    decision = router.route(req, capabilities=[cap])

    assert decision.selected_candidate is None
    assert len(decision.rejected_candidates) == 1
    assert (
        RouteUnavailableReason.MISSING_USER_ASSET
        in decision.rejected_candidates[0].rejection_reasons
    )


# 6. QUALITY_FIRST prefers higher-quality eligible candidate in a representative case.
def test_quality_first_prefers_higher_quality_candidate():
    router = HybridAssetRouter()
    # Candidate A: high quality (0.9), high cost (0.3), medium latency (0.6)
    cand_a = AssetCapability(
        capability_id="provider_a:quality_king:TEXT_TO_VIDEO",
        provider="provider_a",
        model="quality_king",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.HIGH,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    # Candidate B: medium quality (0.6), low cost (0.9), low latency (0.9)
    cand_b = AssetCapability(
        capability_id="provider_b:budget_fast:TEXT_TO_VIDEO",
        provider="provider_b",
        model="budget_fast",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    req = _make_dummy_request(routing_strategy=RoutingStrategy.QUALITY_FIRST)
    policy = get_default_policy(RoutingStrategy.QUALITY_FIRST)
    decision = router.route(req, capabilities=[cand_a, cand_b], policy=policy)

    assert decision.selected_candidate is not None
    assert decision.selected_candidate.capability_id == cand_a.capability_id
    assert decision.selected_candidate.score.ranking_position == 1
    assert decision.eligible_candidates[1].capability_id == cand_b.capability_id
    assert decision.eligible_candidates[1].score.ranking_position == 2


# 7. COST_FIRST prefers lower-cost acceptable candidate in a representative case.
def test_cost_first_prefers_lower_cost_candidate():
    router = HybridAssetRouter()
    cand_a = AssetCapability(
        capability_id="provider_a:quality_king:TEXT_TO_VIDEO",
        provider="provider_a",
        model="quality_king",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.HIGH,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    cand_b = AssetCapability(
        capability_id="provider_b:budget_fast:TEXT_TO_VIDEO",
        provider="provider_b",
        model="budget_fast",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    req = _make_dummy_request(routing_strategy=RoutingStrategy.COST_FIRST)
    policy = get_default_policy(RoutingStrategy.COST_FIRST)
    decision = router.route(req, capabilities=[cand_a, cand_b], policy=policy)

    assert decision.selected_candidate is not None
    assert decision.selected_candidate.capability_id == cand_b.capability_id
    assert decision.selected_candidate.score.ranking_position == 1


# 8. BALANCED produces deterministic ranking.
def test_balanced_produces_deterministic_ranking():
    router = HybridAssetRouter()
    cand_a = AssetCapability(
        capability_id="provider_a:quality_king:TEXT_TO_VIDEO",
        provider="provider_a",
        model="quality_king",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.HIGH,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    cand_b = AssetCapability(
        capability_id="provider_b:budget_fast:TEXT_TO_VIDEO",
        provider="provider_b",
        model="budget_fast",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    req = _make_dummy_request(routing_strategy=RoutingStrategy.BALANCED)
    policy = get_default_policy(RoutingStrategy.BALANCED)
    decision = router.route(req, capabilities=[cand_a, cand_b], policy=policy)

    assert decision.selected_candidate is not None
    # For balanced: B has score 0.45*0.6 + 0.35*0.9 + 0.20*0.9 = 0.765
    # A has score 0.45*0.9 + 0.35*0.3 + 0.20*0.6 = 0.630
    assert decision.selected_candidate.capability_id == cand_b.capability_id
    assert decision.eligible_candidates[0].score.final_score > decision.eligible_candidates[1].score.final_score


# 9. Hard constraint failure cannot be compensated by scoring.
def test_hard_constraint_failure_cannot_be_compensated_by_scoring():
    router = HybridAssetRouter()
    # Candidate C: Perfect scores (HIGH quality, LOW cost, LOW latency), but target duration incompatible
    cand_c = AssetCapability(
        capability_id="provider_c:super_model:TEXT_TO_VIDEO",
        provider="provider_c",
        model="super_model",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        min_duration=1.0,
        max_duration=3.0,  # Max 3s
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    # Candidate D: Modest scores, but satisfies duration
    cand_d = AssetCapability(
        capability_id="provider_d:modest_model:TEXT_TO_VIDEO",
        provider="provider_d",
        model="modest_model",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        min_duration=1.0,
        max_duration=10.0,
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    req = _make_dummy_request(target_duration=5.0)
    decision = router.route(req, capabilities=[cand_c, cand_d])

    # Candidate C is rejected despite superior static tiers
    assert decision.selected_candidate is not None
    assert decision.selected_candidate.capability_id == cand_d.capability_id
    assert any(c.capability_id == cand_c.capability_id for c in decision.rejected_candidates)


# 10. Minimum quality floor prevents an unacceptable cheap candidate.
def test_minimum_quality_floor_prevents_unacceptable_cheap_candidate():
    router = HybridAssetRouter()
    # Candidate LowQuality: cheap, but LOW quality
    cand_low = AssetCapability(
        capability_id="provider_low:dirt_cheap:TEXT_TO_VIDEO",
        provider="provider_low",
        model="dirt_cheap",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.LOW,
            cost_tier=TierLevel.LOW,
            latency_tier=TierLevel.LOW,
        ),
    )
    # Candidate MedQuality: acceptable quality, medium cost
    cand_med = AssetCapability(
        capability_id="provider_med:balanced_model:TEXT_TO_VIDEO",
        provider="provider_med",
        model="balanced_model",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.MEDIUM,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    req = _make_dummy_request(routing_strategy=RoutingStrategy.COST_FIRST)
    # Policy with minimum_quality_tier = TierLevel.MEDIUM
    policy = RoutingPolicy(
        strategy=RoutingStrategy.COST_FIRST,
        quality_weight=0.20,
        cost_weight=0.70,
        latency_weight=0.10,
        minimum_quality_tier=TierLevel.MEDIUM,
    )
    decision = router.route(req, capabilities=[cand_low, cand_med], policy=policy)

    # cand_low rejected because of minimum quality floor
    assert decision.selected_candidate is not None
    assert decision.selected_candidate.capability_id == cand_med.capability_id
    rejected_low = next(c for c in decision.rejected_candidates if c.capability_id == cand_low.capability_id)
    assert RouteUnavailableReason.BELOW_MINIMUM_QUALITY in rejected_low.rejection_reasons


# 11. Tie-breaking is deterministic.
def test_tie_breaking_is_deterministic():
    router = HybridAssetRouter()
    # Two candidates with identical tiers and scores
    cand_z = AssetCapability(
        capability_id="provider_z:model_1:TEXT_TO_VIDEO",
        provider="provider_z",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    cand_a = AssetCapability(
        capability_id="provider_a:model_1:TEXT_TO_VIDEO",
        provider="provider_a",
        model="model_1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.MEDIUM,
        ),
    )
    req = _make_dummy_request()

    # Pass in reverse alphabetical order
    decision = router.route(req, capabilities=[cand_z, cand_a])
    # Tie-breaking rules specify capability_id ASC as final tie-breaker
    assert decision.selected_candidate is not None
    assert decision.selected_candidate.capability_id == cand_a.capability_id
    assert decision.eligible_candidates[0].capability_id == cand_a.capability_id
    assert decision.eligible_candidates[1].capability_id == cand_z.capability_id


# 12. Identical: request + capabilities + policy produces identical selected capability.
def test_identical_inputs_produce_identical_selection():
    router = HybridAssetRouter()
    caps = [
        AssetCapability(
            capability_id=f"provider_{i}:model:TEXT_TO_VIDEO",
            provider=f"provider_{i}",
            model="model",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
            enabled=True,
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.MEDIUM,
                cost_tier=TierLevel.MEDIUM,
                latency_tier=TierLevel.MEDIUM,
            ),
        )
        for i in range(5)
    ]
    req = _make_dummy_request()
    policy = get_default_policy(RoutingStrategy.BALANCED)

    decision_1 = router.route(req, capabilities=caps, policy=policy)
    decision_2 = router.route(req, capabilities=caps, policy=policy)

    assert decision_1.selected_candidate.capability_id == decision_2.selected_candidate.capability_id
    assert len(decision_1.eligible_candidates) == len(decision_2.eligible_candidates)
    for c1, c2 in zip(decision_1.eligible_candidates, decision_2.eligible_candidates, strict=True):
        assert c1.capability_id == c2.capability_id
        assert c1.score.final_score == c2.score.final_score
        assert c1.score.ranking_position == c2.score.ranking_position


# 13. RoutingDecision records: policy version, component scores, rejected reasons.
def test_routing_decision_records_provenance_and_explainability():
    router = HybridAssetRouter()
    cap_eligible = AssetCapability(
        capability_id="provider_ok:m1:TEXT_TO_VIDEO",
        provider="provider_ok",
        model="m1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=True,
        metadata=StaticCapabilityMetadata(
            quality_tier=TierLevel.HIGH,
            cost_tier=TierLevel.MEDIUM,
            latency_tier=TierLevel.LOW,
        ),
    )
    cap_rejected = AssetCapability(
        capability_id="provider_bad:m1:TEXT_TO_VIDEO",
        provider="provider_bad",
        model="m1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=False,  # Rejected
    )
    req = _make_dummy_request()
    policy = RoutingPolicy(
        policy_version="v2.5-special",
        strategy=RoutingStrategy.BALANCED,
        quality_weight=0.5,
        cost_weight=0.3,
        latency_weight=0.2,
    )
    decision = router.route(req, capabilities=[cap_eligible, cap_rejected], policy=policy)

    assert decision.routing_policy_version == "v2.5-special"
    assert decision.routing_strategy == RoutingStrategy.BALANCED
    assert decision.shot_id == req.shot_id
    assert decision.shot_revision_id == req.shot_revision_id
    assert bool(decision.routing_decision_id)
    assert decision.created_at is not None

    # Selected candidate explainability
    score = decision.selected_score
    assert score is not None
    assert score.quality_score == 0.9
    assert score.cost_score == 0.6
    assert score.latency_score == 0.9
    assert score.final_score == 0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.9  # 0.45 + 0.18 + 0.18 = 0.81
    assert score.ranking_position == 1

    # Rejected candidate reasons
    assert len(decision.rejected_candidates) == 1
    assert RouteUnavailableReason.DISABLED_CAPABILITY in decision.rejected_candidates[0].rejection_reasons


# 14. No real provider execution method is called.
def test_zero_provider_execution_during_routing():
    router = HybridAssetRouter()
    req = _make_dummy_request()

    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        decision = router.route(req)
        assert decision is not None
        assert mock_post.call_count == 0
        assert mock_get.call_count == 0


# 15. No eligible candidates produces ROUTE_UNAVAILABLE.
def test_no_eligible_candidates_produces_route_unavailable():
    router = HybridAssetRouter()
    cap = AssetCapability(
        capability_id="only_provider:m1:TEXT_TO_VIDEO",
        provider="only_provider",
        model="m1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=False,
    )
    req = _make_dummy_request()

    decision = router.route(req, capabilities=[cap])
    assert decision.selected_candidate is None
    assert decision.selected_score is None
    assert len(decision.eligible_candidates) == 0
    assert RouteUnavailableReason.DISABLED_CAPABILITY in decision.decision_reason_codes

    # With raise_if_unavailable=True
    with pytest.raises(RouteUnavailableError) as exc_info:
        router.route(req, capabilities=[cap], raise_if_unavailable=True)

    err = exc_info.value
    assert RouteUnavailableReason.DISABLED_CAPABILITY in err.reasons
    assert "ROUTE_UNAVAILABLE" in str(err)
