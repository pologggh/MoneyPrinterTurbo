import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.domain.asset_execution import (
    ExecutionStatus,
    ProviderOutcomeType,
    ShotExecution,
    ShotExecutionStatus,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
)
from app.domain.enums import VisualType
from app.domain.trace import (
    TraceContext,
    TraceEventType,
)
from app.services.asset_adapters.base import AdapterExecutionResult
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.shot_execution_service import ShotExecutionService
from app.services.trace_service import TraceWriter


class TestTraceExecution(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_shot_execution_trace_success_and_fallback(self):
        """Verify attempt start, failure, fallback, retry success, and asset created trace events."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        # Create dummy file to probe
        media_file = self.storage_dir / "sample.mp4"
        media_file.write_bytes(b"dummy mp4 file content for probing")

        candidate_1 = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_p1",
            model="m1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )
        candidate_2 = AssetRouteCandidate(
            capability_id="cap-2",
            provider="mock_p2",
            model="m2",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )

        # Mock adapter 1 to fail, mock adapter 2 to succeed
        adapter_1 = MagicMock()
        adapter_1.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            status="FAILED",
            error_code="RATE_LIMIT_EXCEEDED",
            error_message="Too many requests",
        )

        adapter_2 = MagicMock()
        adapter_2.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(media_file),
            raw_response={"task_id": "t-123"},
        )

        registry = AdapterRegistry()
        registry.register_adapter("mock_p1", adapter_1)
        registry.register_adapter("mock_p2", adapter_2)

        decision = AssetRouteDecision(
            shot_id="shot-e-1",
            shot_revision_id="srev-e-1",
            requested_visual_type=VisualType.AI_VIDEO,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            model_selection_mode="AUTO",
            selected_candidate=candidate_1,
            eligible_candidates=(candidate_1, candidate_2),
        )

        shot_exec = ShotExecution(
            execution_run_id="run-1",
            shot_id="shot-e-1",
            shot_revision_id="srev-e-1",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )

        routing_req = AssetRoutingRequest(
            shot_id="shot-e-1",
            shot_revision_id="srev-e-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="A sunrise",
            scene_description="Sunrise glowing over hills",
            generation_prompt="Sunrise over mountains",
            camera_movement="pan",
        )

        # Service with max_retries_per_candidate = 1 so failure on candidate 1 triggers fallback to candidate 2
        svc = ShotExecutionService(
            adapter_registry=registry,
            max_retries_per_candidate=1,
        )

        ctx = TraceContext(trace_id="tr-exec-1", execution_run_id="run-1")
        final_exec, attempts, asset_ver = svc.execute_shot(
            shot_execution=shot_exec,
            request=routing_req,
            storage_base_dir=self.storage_dir,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(final_exec.status, ShotExecutionStatus.SUCCEEDED)
        self.assertIsNotNone(asset_ver)
        self.assertEqual(len(attempts), 2)

        # Verify emitted events
        event_types = [call[0][0].event_type for call in mock_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.EXECUTION_ATTEMPT_STARTED, event_types)
        self.assertIn(TraceEventType.EXECUTION_ATTEMPT_COMPLETED, event_types)
        self.assertIn(TraceEventType.EXECUTION_FALLBACK_SELECTED, event_types)
        self.assertIn(TraceEventType.SHOT_ASSET_CREATED, event_types)

        # Check fallback event attributes
        fallback_events = [
            call[0][0]
            for call in mock_repo.add_event.call_args_list
            if call[0][0].event_type == TraceEventType.EXECUTION_FALLBACK_SELECTED
        ]
        self.assertEqual(len(fallback_events), 1)
        fe = fallback_events[0]
        self.assertEqual(fe.attributes["from_provider"], "mock_p1")
        self.assertEqual(fe.attributes["to_provider"], "mock_p2")

    def test_shot_execution_trace_unknown_recovery(self):
        """Verify SUBMISSION_OUTCOME_UNKNOWN records EXECUTION_RECOVERY_REQUIRED."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        candidate_1 = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_unknown",
            model="m1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )
        adapter = MagicMock()
        adapter.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            status="UNKNOWN",
            remote_task_id="remote-unconfirmed-42",
            error_code="SUBMISSION_OUTCOME_UNKNOWN",
            error_message="Network dropped during handshake",
        )

        registry = AdapterRegistry()
        registry.register_adapter("mock_unknown", adapter)

        decision = AssetRouteDecision(
            shot_id="shot-e-2",
            shot_revision_id="srev-e-2",
            requested_visual_type=VisualType.AI_VIDEO,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            model_selection_mode="AUTO",
            selected_candidate=candidate_1,
            eligible_candidates=(candidate_1,),
        )
        shot_exec = ShotExecution(
            execution_run_id="run-2",
            shot_id="shot-e-2",
            shot_revision_id="srev-e-2",
            route_decision=decision,
            status=ShotExecutionStatus.PENDING,
        )
        routing_req = AssetRoutingRequest(
            shot_id="shot-e-2",
            shot_revision_id="srev-e-2",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="Sunset",
            scene_description="Sunset glowing",
            generation_prompt="Sunset",
            camera_movement="static",
        )

        svc = ShotExecutionService(adapter_registry=registry)
        ctx = TraceContext(trace_id="tr-exec-2", execution_run_id="run-2")

        final_exec, _attempts, _asset_ver = svc.execute_shot(
            shot_execution=shot_exec,
            request=routing_req,
            storage_base_dir=self.storage_dir,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(final_exec.status, ShotExecutionStatus.NEEDS_RECOVERY)
        event_types = [call[0][0].event_type for call in mock_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.EXECUTION_RECOVERY_REQUIRED, event_types)

    def test_route_plan_execution_trace_run_lifecycle(self):
        """Verify EXECUTION_RUN_STARTED, SHOT_EXECUTION_STARTED, and EXECUTION_RUN_COMPLETED."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        media_file = self.storage_dir / "sample.mp4"
        media_file.write_bytes(b"sample video bytes")

        candidate = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_p1",
            model="m1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )
        decision = AssetRouteDecision(
            shot_id="shot-plan-1",
            shot_revision_id="srev-plan-1",
            requested_visual_type=VisualType.AI_VIDEO,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            model_selection_mode="AUTO",
            selected_candidate=candidate,
            eligible_candidates=(candidate,),
        )
        routing_req = AssetRoutingRequest(
            shot_id="shot-plan-1",
            shot_revision_id="srev-plan-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="Goal",
            scene_description="Desc",
            generation_prompt="Prompt",
            camera_movement="zoom",
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-plan-1",
            shot_revision_id="srev-plan-1",
            beat_lineage_id="lin-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        plan = AssetRoutePlan(
            asset_route_plan_id="arp-trace-1",
            storyboard_snapshot_id="sbs-trace-1",
            content_plan_revision_id="cpr-1",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            routing_policy_version="v1.0",
            status=AssetRoutePlanStatus.READY,
            shot_routes=(entry,),
            total_shots=1,
            routed_shots=1,
        )

        adapter = MagicMock()
        adapter.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(media_file),
        )
        registry = AdapterRegistry()
        registry.register_adapter("mock_p1", adapter)

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.persistence.models import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)

        shot_svc = ShotExecutionService(adapter_registry=registry)
        plan_svc = AssetRoutePlanExecutionService(
            session_factory=SessionLocal,
            shot_execution_service=shot_svc,
        )

        ctx = TraceContext(trace_id="tr-run-1", asset_route_plan_id="arp-trace-1")
        result = plan_svc.execute_route_plan(
            plan=plan,
            storage_base_dir=self.storage_dir,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(result.state, ExecutionStatus.COMPLETED)
        event_types = [call[0][0].event_type for call in mock_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.EXECUTION_RUN_STARTED, event_types)
        self.assertIn(TraceEventType.SHOT_EXECUTION_STARTED, event_types)
        self.assertIn(TraceEventType.EXECUTION_RUN_COMPLETED, event_types)
