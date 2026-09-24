import unittest
from unittest.mock import MagicMock

from app.domain.asset_router import (
    AssetCapability,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.enums import VisualType
from app.domain.trace import (
    TraceContext,
    TraceEventType,
)
from app.services.asset_capability_registry import AssetCapabilityRegistry
from app.services.hybrid_asset_router import HybridAssetRouter
from app.services.trace_service import TraceWriter


class TestTraceRouter(unittest.TestCase):

    def test_router_trace_event(self):
        """Invariant 9 & 10: Router trace preserves Provider, Model, GenerationMode, strategy."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        cap = AssetCapability(
            capability_id="cap-seedance",
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.HIGH,
                latency_tier=TierLevel.MEDIUM,
            ),
        )
        registry = AssetCapabilityRegistry()
        router = HybridAssetRouter(registry=registry)

        req = AssetRoutingRequest(
            shot_id="shot-r-1",
            shot_revision_id="srev-r-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="Space nebula",
            scene_description="Nebula swirling",
            generation_prompt="nebula, space, colorful",
            camera_movement="pan",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )

        ctx = TraceContext(trace_id="tr-router-1", asset_route_plan_id="arp-1")
        decision = router.route(
            req,
            capabilities=(cap,),
            trace_context=ctx,
            trace_writer=writer,
            asset_route_plan_id="arp-1",
        )
        self.assertIsNotNone(decision.selected_candidate)
        self.assertEqual(decision.selected_candidate.provider, "seedance")

        # Verify event was recorded
        mock_repo.add_event.assert_called_once()
        recorded_event = mock_repo.add_event.call_args[0][0]
        self.assertEqual(recorded_event.event_type, TraceEventType.ASSET_ROUTE_DECISION_CREATED)
        self.assertEqual(recorded_event.trace_id, "tr-router-1")
        self.assertEqual(recorded_event.attributes["selected_provider"], "seedance")
        self.assertEqual(recorded_event.attributes["selected_model"], "pro")
        self.assertEqual(recorded_event.attributes["strategy"], "QUALITY_FIRST")
