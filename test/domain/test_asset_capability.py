from unittest.mock import patch

import pytest

from app.config import config
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteDecision,
    AssetRoutingRequest,
    GenerationMode,
    RouteUnavailableError,
    RouteUnavailableReason,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
    create_routing_request_from_shot_revision,
)
from app.domain.enums import VisualType
from app.domain.shot import Shot, ShotRevision
from app.services.asset_capability_registry import (
    AssetCapabilityProvider,
    AssetCapabilityRegistry,
    SourceAssetAdapter,
    UserAssetAdapter,
    evaluate_candidate,
    get_default_capability_registry,
)


def _make_dummy_shot_revision(
    visual_type: VisualType = VisualType.AI_VIDEO,
    target_duration: float = 5.0,
    evidence_refs: tuple[str, ...] = (),
) -> ShotRevision:
    return ShotRevision(
        shot_id="shot-001",
        revision_number=1,
        beat_lineage_id="beat-lineage-001",
        created_from_beat_instance_id="beat-inst-001",
        narration="Test narration for knowledge shot",
        target_duration=target_duration,
        visual_goal="Explain knowledge concept clearly",
        visual_type=visual_type,
        scene_description="A clear motion graphic explaining the mechanism",
        generation_prompt="cinematic animation explaining concept",
        camera_movement="pan right",
        evidence_refs=evidence_refs,
    )


# 1. Existing VisualTypes can be represented in routing requests.
def test_visual_types_represented_in_routing_requests():
    for vt in VisualType:
        shot_rev = _make_dummy_shot_revision(visual_type=vt)
        req = create_routing_request_from_shot_revision(
            shot_rev,
            aspect_ratio="16:9",
            routing_strategy=RoutingStrategy.BALANCED,
        )
        assert req.requested_visual_type == vt
        assert req.target_duration == 5.0
        assert req.shot_id == "shot-001"
        assert req.shot_revision_id == shot_rev.shot_revision_id
        assert req.routing_strategy == RoutingStrategy.BALANCED
        assert req.aspect_ratio == "16:9"


# 2. AI_VIDEO request returns only capabilities that claim AI_VIDEO support.
def test_ai_video_request_returns_only_ai_video_capabilities():
    registry = get_default_capability_registry()
    req = AssetRoutingRequest(
        shot_id="shot-100",
        shot_revision_id="shot-rev-100",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=6.0,
        visual_goal="Show dynamic AI visualization",
        scene_description="futuristic visualization",
        generation_prompt="neural network pulses with light",
        camera_movement="zoom in",
        aspect_ratio="16:9",
    )
    candidates = registry.find_candidates(req)
    assert len(candidates) > 0

    # Stock search or image capabilities must be rejected due to UNSUPPORTED_VISUAL_TYPE
    for cand in candidates:
        if cand.is_eligible:
            assert cand.requested_visual_type == VisualType.AI_VIDEO
            assert cand.generation_mode in (
                GenerationMode.TEXT_TO_VIDEO,
                GenerationMode.IMAGE_TO_VIDEO,
            )
        else:
            if cand.provider in ("pexels", "pixabay", "coverr", "openai"):
                assert (
                    RouteUnavailableReason.UNSUPPORTED_VISUAL_TYPE
                    in cand.rejection_reasons
                )


# 3. STOCK_VIDEO request can discover existing stock capability.
def test_stock_video_request_discovers_stock_capability():
    registry = get_default_capability_registry()
    req = AssetRoutingRequest(
        shot_id="shot-101",
        shot_revision_id="shot-rev-101",
        requested_visual_type=VisualType.STOCK_VIDEO,
        target_duration=4.0,
        visual_goal="Show real-world footage",
        scene_description="city street at daytime",
        generation_prompt="busy city street",
        camera_movement="static",
        aspect_ratio="16:9",
    )
    candidates = registry.find_candidates(req)
    stock_candidates = [
        c
        for c in candidates
        if c.generation_mode == GenerationMode.STOCK_SEARCH
    ]
    assert len(stock_candidates) >= 3  # Pexels, Pixabay, Coverr
    # Coverr is always enabled without an API key in MPT
    coverr_candidates = [c for c in stock_candidates if c.provider == "coverr"]
    assert len(coverr_candidates) == 1
    assert coverr_candidates[0].is_eligible is True


# 4. Disabled capability is identifiable as ineligible.
def test_disabled_capability_identified_as_ineligible():
    cap = AssetCapability(
        capability_id="fake_provider:model_v1:TEXT_TO_VIDEO",
        provider="fake_provider",
        model="model_v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        enabled=False,  # Disabled
    )
    req = AssetRoutingRequest(
        shot_id="shot-102",
        shot_revision_id="shot-rev-102",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="goal",
        scene_description="desc",
        generation_prompt="prompt",
        camera_movement="pan",
    )
    cand = evaluate_candidate(cap, req)
    assert cand.is_eligible is False
    assert RouteUnavailableReason.NO_ENABLED_PROVIDER in cand.rejection_reasons


# 5. Duration constraints can be represented and validated.
def test_duration_constraints_validated():
    cap = AssetCapability(
        capability_id="bounded_provider:model_v1:TEXT_TO_VIDEO",
        provider="bounded_provider",
        model="model_v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        min_duration=3.0,
        max_duration=10.0,
        enabled=True,
    )

    # Valid duration
    req_valid = AssetRoutingRequest(
        shot_id="shot-103",
        shot_revision_id="shot-rev-103",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="goal",
        scene_description="desc",
        generation_prompt="prompt",
        camera_movement="pan",
    )
    cand_valid = evaluate_candidate(cap, req_valid)
    assert cand_valid.is_eligible is True

    # Too short duration
    req_short = req_valid.model_copy(update={"target_duration": 2.0})
    cand_short = evaluate_candidate(cap, req_short)
    assert cand_short.is_eligible is False
    assert RouteUnavailableReason.UNSUPPORTED_DURATION in cand_short.rejection_reasons

    # Too long duration
    req_long = req_valid.model_copy(update={"target_duration": 15.0})
    cand_long = evaluate_candidate(cap, req_long)
    assert cand_long.is_eligible is False
    assert RouteUnavailableReason.UNSUPPORTED_DURATION in cand_long.rejection_reasons


# 6. Aspect-ratio constraints can be represented and validated if supported.
def test_aspect_ratio_constraints_validated():
    cap = AssetCapability(
        capability_id="aspect_bounded:model_v1:TEXT_TO_VIDEO",
        provider="aspect_bounded",
        model="model_v1",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        supported_visual_types=(VisualType.AI_VIDEO,),
        supported_aspect_ratios=("16:9", "9:16"),
        enabled=True,
    )

    req_match = AssetRoutingRequest(
        shot_id="shot-104",
        shot_revision_id="shot-rev-104",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="goal",
        scene_description="desc",
        generation_prompt="prompt",
        camera_movement="pan",
        aspect_ratio="16:9",
    )
    assert evaluate_candidate(cap, req_match).is_eligible is True

    req_mismatch = req_match.model_copy(update={"aspect_ratio": "1:1"})
    cand_mismatch = evaluate_candidate(cap, req_mismatch)
    assert cand_mismatch.is_eligible is False
    assert (
        RouteUnavailableReason.UNSUPPORTED_ASPECT_RATIO
        in cand_mismatch.rejection_reasons
    )


# 7. SOURCE_ASSET route requires source asset reference.
def test_source_asset_route_requires_source_asset_reference():
    adapter = SourceAssetAdapter()
    cap = adapter.get_capabilities()[0]

    # Without source asset ref -> Ineligible
    req_no_ref = AssetRoutingRequest(
        shot_id="shot-105",
        shot_revision_id="shot-rev-105",
        requested_visual_type=VisualType.SOURCE_ASSET,
        target_duration=4.0,
        visual_goal="Show textbook figure",
        scene_description="diagram of cell structure",
        generation_prompt="",
        camera_movement="static",
        source_asset_refs=(),
    )
    cand_no_ref = evaluate_candidate(cap, req_no_ref)
    assert cand_no_ref.is_eligible is False
    assert (
        RouteUnavailableReason.MISSING_SOURCE_ASSET in cand_no_ref.rejection_reasons
    )

    # With source asset ref -> Eligible
    req_with_ref = req_no_ref.model_copy(
        update={"source_asset_refs": ("source_doc_fig_3",)}
    )
    cand_with_ref = evaluate_candidate(cap, req_with_ref)
    assert cand_with_ref.is_eligible is True


# 8. USER_ASSET route requires user asset reference.
def test_user_asset_route_requires_user_asset_reference():
    adapter = UserAssetAdapter()
    cap = adapter.get_capabilities()[0]

    # Without user asset ref -> Ineligible
    req_no_ref = AssetRoutingRequest(
        shot_id="shot-106",
        shot_revision_id="shot-rev-106",
        requested_visual_type=VisualType.USER_ASSET,
        target_duration=4.0,
        visual_goal="Show user demonstration clip",
        scene_description="hands-on demonstration",
        generation_prompt="",
        camera_movement="static",
        user_asset_refs=(),
    )
    cand_no_ref = evaluate_candidate(cap, req_no_ref)
    assert cand_no_ref.is_eligible is False
    assert RouteUnavailableReason.MISSING_USER_ASSET in cand_no_ref.rejection_reasons

    # With user asset ref -> Eligible
    req_with_ref = req_no_ref.model_copy(
        update={"user_asset_refs": ("user_upload_001.mp4",)}
    )
    cand_with_ref = evaluate_candidate(cap, req_with_ref)
    assert cand_with_ref.is_eligible is True


# 9. AI_VIDEO route does not hard-code Seedance in Shot/Storyboard domain.
def test_ai_video_does_not_hardcode_seedance_in_shot_domain():
    shot = Shot(beat_lineage_id="beat-1", local_order=1)
    shot_rev = _make_dummy_shot_revision(visual_type=VisualType.AI_VIDEO)

    # Domain models have no provider fields or Seedance hardcoding
    assert not hasattr(shot, "provider")
    assert not hasattr(shot_rev, "provider")
    assert not hasattr(shot_rev, "seedance_model")
    assert "provider" not in Shot.model_fields
    assert "provider" not in ShotRevision.model_fields

    # ShotRevision only knows semantic visual_type
    assert shot_rev.visual_type == VisualType.AI_VIDEO


# 10. Capability catalog can represent multiple provider/model/mode candidates.
def test_capability_catalog_represents_multiple_candidates():
    # Construct a registry with multiple distinct AI_VIDEO providers
    class DummyProviderA(AssetCapabilityProvider):
        def get_capabilities(self) -> tuple[AssetCapability, ...]:
            return (
                AssetCapability(
                    capability_id="provider_a:model_x:TEXT_TO_VIDEO",
                    provider="provider_a",
                    model="model_x",
                    generation_mode=GenerationMode.TEXT_TO_VIDEO,
                    supported_visual_types=(VisualType.AI_VIDEO,),
                    enabled=True,
                    metadata=StaticCapabilityMetadata(
                        quality_tier=TierLevel.HIGH,
                        cost_tier=TierLevel.HIGH,
                        latency_tier=TierLevel.MEDIUM,
                    ),
                ),
            )

    class DummyProviderB(AssetCapabilityProvider):
        def get_capabilities(self) -> tuple[AssetCapability, ...]:
            return (
                AssetCapability(
                    capability_id="provider_b:model_y:IMAGE_TO_VIDEO",
                    provider="provider_b",
                    model="model_y",
                    generation_mode=GenerationMode.IMAGE_TO_VIDEO,
                    supported_visual_types=(VisualType.AI_VIDEO,),
                    enabled=True,
                    metadata=StaticCapabilityMetadata(
                        quality_tier=TierLevel.MEDIUM,
                        cost_tier=TierLevel.LOW,
                        latency_tier=TierLevel.LOW,
                    ),
                ),
            )

    registry = AssetCapabilityRegistry(providers=[DummyProviderA(), DummyProviderB()])
    req = AssetRoutingRequest(
        shot_id="shot-107",
        shot_revision_id="shot-rev-107",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="goal",
        scene_description="desc",
        generation_prompt="prompt",
        camera_movement="pan",
    )
    candidates = registry.find_candidates(req)
    assert len(candidates) == 2
    assert {c.provider for c in candidates} == {"provider_a", "provider_b"}
    assert {c.generation_mode for c in candidates} == {
        GenerationMode.TEXT_TO_VIDEO,
        GenerationMode.IMAGE_TO_VIDEO,
    }
    assert all(c.is_eligible for c in candidates)


# 11. No provider execution method is invoked during capability discovery.
def test_zero_provider_execution_during_discovery():
    registry = get_default_capability_registry()
    req = AssetRoutingRequest(
        shot_id="shot-108",
        shot_revision_id="shot-rev-108",
        requested_visual_type=VisualType.AI_VIDEO,
        target_duration=5.0,
        visual_goal="goal",
        scene_description="desc",
        generation_prompt="prompt",
        camera_movement="pan",
        aspect_ratio="16:9",
    )

    # Patch requests.post and requests.get to guarantee zero network activity
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        caps = registry.list_capabilities()
        candidates = registry.find_candidates(req)
        decision = registry.create_route_decision(req)

        assert len(caps) > 0
        assert len(candidates) > 0
        assert isinstance(decision, AssetRouteDecision)
        assert mock_post.call_count == 0
        assert mock_get.call_count == 0


# 12. ROUTE_UNAVAILABLE can carry structured reason codes.
def test_route_unavailable_carries_structured_reason_codes():
    class DisabledDiagramProvider(AssetCapabilityProvider):
        def get_capabilities(self) -> tuple[AssetCapability, ...]:
            return (
                AssetCapability(
                    capability_id="disabled:diagram:DIAGRAM_RENDER",
                    provider="disabled_diagram",
                    model="disabled",
                    generation_mode=GenerationMode.DIAGRAM_RENDER,
                    supported_visual_types=(VisualType.DIAGRAM,),
                    enabled=False,
                ),
            )

    registry = AssetCapabilityRegistry(
        providers=[
            DisabledDiagramProvider(),
        ]
    )

    # Request diagram when no enabled diagram provider exists
    req = AssetRoutingRequest(
        shot_id="shot-109",
        shot_revision_id="shot-rev-109",
        requested_visual_type=VisualType.DIAGRAM,
        target_duration=5.0,
        visual_goal="Show architectural block diagram",
        scene_description="diagram of layers",
        generation_prompt="graph LR",
        camera_movement="static",
    )

    decision = registry.create_route_decision(req)
    assert len(decision.eligible_candidates) == 0
    assert len(decision.rejected_candidates) == 1
    assert RouteUnavailableReason.NO_ENABLED_PROVIDER in decision.reason_codes

    # Verify raise_if_unavailable raises RouteUnavailableError with structured reasons
    with pytest.raises(RouteUnavailableError) as exc_info:
        registry.create_route_decision(req, raise_if_unavailable=True)

    err = exc_info.value
    assert RouteUnavailableReason.NO_ENABLED_PROVIDER in err.reasons
    assert "ROUTE_UNAVAILABLE" in str(err)


def test_default_diagram_capability_is_enabled_and_routable():
    registry = get_default_capability_registry()
    request = AssetRoutingRequest(
        shot_id="shot-diagram",
        shot_revision_id="shot-rev-diagram",
        requested_visual_type=VisualType.DIAGRAM,
        target_duration=4.0,
        visual_goal="Explain how Ollama connects to Dify",
        scene_description="Ollama connects to Dify, then builds a local knowledge base",
        generation_prompt="Ollama -> Dify -> local knowledge base",
        camera_movement="static",
        aspect_ratio="16:9",
    )

    decision = registry.create_route_decision(request)

    assert len(decision.eligible_candidates) == 1
    candidate = decision.eligible_candidates[0]
    assert candidate.provider == "system_diagram"
    assert candidate.generation_mode == GenerationMode.DIAGRAM_RENDER


# 14. AI_IMAGE fallback: when no real image provider is enabled, system_knowledge_card is eligible
def test_ai_image_knowledge_card_fallback_when_no_real_provider():
    with patch.dict(
        config.app,
        {
            "openai_image_api_key": None,
            "openai_api_key": "",
            "openai_image_base_url": "",
        },
    ):
        registry = get_default_capability_registry()
        request = AssetRoutingRequest(
            shot_id="shot-ai-image-fallback",
            shot_revision_id="rev-ai-image-fallback",
            requested_visual_type=VisualType.AI_IMAGE,
            target_duration=4.0,
            visual_goal="Explain Transformer self-attention mechanism",
            scene_description="Transformer self-attention calculation flow",
            generation_prompt="Attention(Q, K, V) softmax formula breakdown",
            camera_movement="static",
            aspect_ratio="16:9",
        )

        candidates = registry.find_candidates(request)
        kc_candidates = [
            c for c in candidates if c.provider == "system_knowledge_card"
        ]
        assert len(kc_candidates) == 1
        kc = kc_candidates[0]
        assert kc.is_eligible is True
        assert kc.model == "knowledge_card_v1"
        assert kc.generation_mode == GenerationMode.TEXT_TO_IMAGE
        assert kc.requested_visual_type == VisualType.AI_IMAGE

        # Real provider (openai) must be ineligible due to NO_ENABLED_PROVIDER
        openai_candidates = [c for c in candidates if c.provider == "openai"]
        assert len(openai_candidates) == 1
        assert openai_candidates[0].is_eligible is False
        assert (
            RouteUnavailableReason.NO_ENABLED_PROVIDER
            in openai_candidates[0].rejection_reasons
        )

        decision = registry.create_route_decision(request)
        assert len(decision.eligible_candidates) >= 1
        assert any(
            c.provider == "system_knowledge_card" for c in decision.eligible_candidates
        )


# 15. Real image provider priority: when real provider is configured, knowledge card fallback is disabled
def test_ai_image_real_provider_prioritized_over_knowledge_card_fallback():
    with patch.dict(
        config.app,
        {
            "openai_image_base_url": "https://api.openai.com/v1",
            "openai_image_model": "dall-e-3",
            "openai_image_api_keys": ["sk-mock-real-image-key"],
            "openai_api_key": "",
        },
    ):
        registry = get_default_capability_registry()
        request = AssetRoutingRequest(
            shot_id="shot-ai-image-real",
            shot_revision_id="rev-ai-image-real",
            requested_visual_type=VisualType.AI_IMAGE,
            target_duration=4.0,
            visual_goal="Photorealistic neural network core",
            scene_description="Glowing optical computing chip",
            generation_prompt="hyperrealistic optical neural core",
            camera_movement="static",
            aspect_ratio="16:9",
        )

        candidates = registry.find_candidates(request)
        openai_cands = [c for c in candidates if c.provider == "openai"]
        assert len(openai_cands) == 1
        assert openai_cands[0].is_eligible is True

        kc_cands = [
            c for c in candidates if c.provider == "system_knowledge_card"
        ]
        assert len(kc_cands) == 1
        # Fallback must be disabled or ineligible when real provider is enabled
        assert kc_cands[0].is_eligible is False
        assert (
            RouteUnavailableReason.NO_ENABLED_PROVIDER in kc_cands[0].rejection_reasons
        )

        decision = registry.create_route_decision(request)
        assert len(decision.eligible_candidates) == 1
        assert decision.eligible_candidates[0].provider == "openai"


# 16. Knowledge card fallback respects aspect ratios (16:9, 9:16, 1:1) and rejects unsupported ones
def test_knowledge_card_fallback_aspect_ratio_constraints():
    with patch.dict(
        config.app,
        {
            "openai_image_api_key": None,
            "openai_api_key": "",
            "openai_image_base_url": "",
        },
    ):
        registry = get_default_capability_registry()
        for ar in ("16:9", "9:16", "1:1"):
            req = AssetRoutingRequest(
                shot_id=f"shot-{ar}",
                shot_revision_id=f"rev-{ar}",
                requested_visual_type=VisualType.AI_IMAGE,
                target_duration=4.0,
                visual_goal="Knowledge concept",
                scene_description="Concept details",
                generation_prompt="Concept diagram",
                camera_movement="static",
                aspect_ratio=ar,
            )
            cands = [
                c
                for c in registry.find_candidates(req)
                if c.provider == "system_knowledge_card"
            ]
            assert len(cands) == 1
            assert cands[0].is_eligible is True

        # Unsupported aspect ratio like 4:3
        req_unsupported = AssetRoutingRequest(
            shot_id="shot-unsupported",
            shot_revision_id="rev-unsupported",
            requested_visual_type=VisualType.AI_IMAGE,
            target_duration=4.0,
            visual_goal="Knowledge concept",
            scene_description="Concept details",
            generation_prompt="Concept diagram",
            camera_movement="static",
            aspect_ratio="4:3",
        )
        cands_unsupported = [
            c
            for c in registry.find_candidates(req_unsupported)
            if c.provider == "system_knowledge_card"
        ]
        assert len(cands_unsupported) == 1
        assert cands_unsupported[0].is_eligible is False
        assert (
            RouteUnavailableReason.UNSUPPORTED_ASPECT_RATIO
            in cands_unsupported[0].rejection_reasons
        )


# 17. OpenAI text LLM key (openai_api_key) must NOT enable OpenAIImageAdapter or disable knowledge card fallback
def test_openai_text_llm_key_does_not_enable_image_provider_and_knowledge_card_falls_back():
    with patch.dict(
        config.app,
        {
            "openai_api_key": "sk-mock-text-llm-key",
            "openai_image_api_key": "",
            "openai_image_api_keys": [],
            "openai_image_base_url": "",
            "openai_image_model": "",
        },
    ):
        registry = get_default_capability_registry()
        request = AssetRoutingRequest(
            shot_id="shot-ai-image-text-key",
            shot_revision_id="rev-ai-image-text-key",
            requested_visual_type=VisualType.AI_IMAGE,
            target_duration=4.0,
            visual_goal="Explain LLM attention map",
            scene_description="LLM attention heatmap visualization",
            generation_prompt="attention map heatmap breakdown",
            camera_movement="static",
            aspect_ratio="16:9",
        )

        candidates = registry.find_candidates(request)
        openai_cands = [c for c in candidates if c.provider == "openai"]
        assert len(openai_cands) == 1
        # OpenAI image provider must NOT be enabled just because text LLM key is set!
        assert openai_cands[0].is_eligible is False
        assert (
            RouteUnavailableReason.NO_ENABLED_PROVIDER in openai_cands[0].rejection_reasons
        )

        kc_cands = [
            c for c in candidates if c.provider == "system_knowledge_card"
        ]
        assert len(kc_cands) == 1
        # Knowledge card fallback MUST remain eligible!
        assert kc_cands[0].is_eligible is True

        decision = registry.create_route_decision(request)
        assert len(decision.eligible_candidates) == 1
        assert decision.eligible_candidates[0].provider == "system_knowledge_card"
