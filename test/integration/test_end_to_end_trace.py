import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    ExecutionStatus,
    ProviderOutcomeType,
)
from app.domain.asset_router import (
    AssetCapability,
    AssetRoutePlanStatus,
    GenerationMode,
    RoutingStrategy,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.enums import StoryboardSnapshotState, VisualType
from app.domain.evaluation import (
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationSnapshot,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy
from app.domain.planner import ContentPlanner, PlannerInput
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationPolicy,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard_agent import StoryboardExecutionResult
from app.domain.storyboard_approval import StoryboardApprovalService
from app.domain.storyboard_orchestrator import (
    StoryboardBuildInput,
    StoryboardOrchestrator,
)
from app.domain.trace import (
    TraceContext,
    TraceEventType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
    TraceRepository,
)
from app.services.asset_adapters.base import AdapterExecutionResult
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_capability_registry import AssetCapabilityRegistry
from app.services.asset_route_plan_execution_service import (
    AssetRoutePlanExecutionService,
)
from app.services.asset_route_planning_service import (
    AssetRoutePlanningService,
    CreateAssetRoutePlanInput,
)
from app.services.evaluation.mpt_multimodal_adapter import (
    MultimodalEvaluationObserverResult,
    MultimodalEvaluatorAdapter,
)
from app.services.evaluation.quality_remediation_service import (
    QualityRemediationService,
)
from app.services.evaluation.shot_asset_evaluation_service import (
    ShotAssetEvaluationService,
)
from app.services.shot_execution_service import ShotExecutionService
from app.services.trace_service import TraceService


class StaticProvider:
    def __init__(self, caps):
        self._caps = caps

    def get_capabilities(self):
        return self._caps


class TestEndToEndTrace(unittest.TestCase):
    """
    Comprehensive integration test validating that exactly ONE trace_id spans the
    entire video production lifecycle (Planner -> Storyboard -> Router -> Execution
    -> Evaluation -> Remediation) with full auditability, monotonic durations,
    and capability to compute all benchmark metrics.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)

        # Setup SQLite in-memory database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()

        # Initialize repositories
        self.plan_repo = ContentPlanRepository(self.session)
        self.shot_repo = ShotRepository(self.session)
        self.sb_repo = StoryboardRepository(self.session)
        self.appr_repo = StoryboardApprovalRepository(self.session)
        self.route_plan_repo = AssetRoutePlanRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)
        self.eval_repo = EvaluationRepository(self.session)
        self.trace_repo = TraceRepository(self.session)

        self.trace_service = TraceService(repository=self.trace_repo)

    def tearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    def test_full_lifecycle_trace_scenario(self):
        # 0. Initialize production trace
        root = self.trace_service.create_root(root_type="PRODUCTION_WORKFLOW")
        trace_id = root.trace_id
        ctx = TraceContext(trace_id=trace_id)

        # ---------------------------------------------------------------------
        # a. Content Planner -> produces ContentPlanRevision (2 beats)
        # ---------------------------------------------------------------------
        planner_llm_json = """
        {
            "title": "Introduction to Quantum Computing",
            "beats": [
                {
                    "planner_ref": "b1",
                    "beat_type": "HOOK",
                    "intent": "Hook viewer with qubits",
                    "order": 1,
                    "target_duration": 4.0,
                    "importance": 0.9,
                    "evidence_refs": ["ev_qubits"]
                },
                {
                    "planner_ref": "b2",
                    "beat_type": "KNOWLEDGE",
                    "intent": "Explain superposition",
                    "order": 2,
                    "target_duration": 4.0,
                    "importance": 0.8,
                    "evidence_refs": ["ev_superposition"]
                }
            ]
        }
        """
        planner = ContentPlanner(
            repository=self.plan_repo,
            llm_caller=lambda _: planner_llm_json,
        )
        plan_rev = planner.plan(
            PlannerInput(
                topic="Quantum Computing",
                target_video_duration=8.0,
                available_evidence_ids=["ev_qubits", "ev_superposition"],
            ),
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(len(plan_rev.beats), 2)
        ctx = ctx.child_context(
            parent_event_id=ctx.parent_event_id,
            content_plan_revision_id=plan_rev.content_plan_revision_id,
        )

        # ---------------------------------------------------------------------
        # b. Storyboard Orchestrator -> produces APPROVED StoryboardSnapshot
        # ---------------------------------------------------------------------
        shot_1 = Shot(shot_id="shot-1", beat_lineage_id=plan_rev.beats[0].beat_lineage_id, local_order=1)
        rev_1 = ShotRevision(
            shot_revision_id="srev-1",
            shot_id="shot-1",
            revision_number=1,
            beat_lineage_id=plan_rev.beats[0].beat_lineage_id,
            created_from_beat_instance_id=plan_rev.beats[0].beat_id,
            narration="Qubits can exist in quantum states.",
            target_duration=4.0,
            visual_goal="Glowing qubit sphere",
            scene_description="Spinning glowing sphere",
            generation_prompt="glowing qubit sphere in quantum computer",
            visual_type=VisualType.AI_IMAGE,
            camera_movement="pan",
            evidence_refs=("ev_qubits",),
        )

        shot_2 = Shot(shot_id="shot-2", beat_lineage_id=plan_rev.beats[1].beat_lineage_id, local_order=1)
        rev_2 = ShotRevision(
            shot_revision_id="srev-2",
            shot_id="shot-2",
            revision_number=1,
            beat_lineage_id=plan_rev.beats[1].beat_lineage_id,
            created_from_beat_instance_id=plan_rev.beats[1].beat_id,
            narration="Superposition allows simultaneous states.",
            target_duration=4.0,
            visual_goal="Two overlapping wave forms",
            scene_description="Overlapping coherent waves",
            generation_prompt="quantum superposition overlapping waves, scientific",
            visual_type=VisualType.AI_IMAGE,
            camera_movement="static",
            evidence_refs=("ev_superposition",),
        )

        mock_sb_agent = MagicMock()
        mock_sb_agent.generate_shots_for_beat.side_effect = [
            StoryboardExecutionResult(shots=(shot_1,), shot_revisions=(rev_1,)),
            StoryboardExecutionResult(shots=(shot_2,), shot_revisions=(rev_2,)),
        ]

        sb_orchestrator = StoryboardOrchestrator(
            plan_repository=self.plan_repo,
            shot_repository=self.shot_repo,
            storyboard_repository=self.sb_repo,
            storyboard_agent=mock_sb_agent,
        )
        build_res = sb_orchestrator.build_storyboard(
            StoryboardBuildInput(new_content_plan_revision_id=plan_rev.content_plan_revision_id),
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        draft_snapshot = build_res.storyboard_snapshot
        self.assertEqual(draft_snapshot.snapshot_state, StoryboardSnapshotState.DRAFT)

        approval_svc = StoryboardApprovalService(
            plan_repository=self.plan_repo,
            shot_repository=self.shot_repo,
            storyboard_repository=self.sb_repo,
            approval_repository=self.appr_repo,
        )
        approval_res = approval_svc.approve_storyboard(
            storyboard_snapshot_id=draft_snapshot.storyboard_snapshot_id,
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        approved_snapshot = approval_res.approved_snapshot
        self.assertEqual(approved_snapshot.snapshot_state, StoryboardSnapshotState.APPROVED)
        ctx = ctx.child_context(
            parent_event_id=ctx.parent_event_id,
            storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
        )

        # ---------------------------------------------------------------------
        # c. Asset Router -> produces AssetRoutePlan
        # ---------------------------------------------------------------------
        cap_p1 = AssetCapability(
            capability_id="cap-provider-1",
            provider="provider_alpha",
            model="model_alpha",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            supported_visual_types=(VisualType.AI_IMAGE,),
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.MEDIUM,
                latency_tier=TierLevel.LOW,
            ),
        )
        cap_p2 = AssetCapability(
            capability_id="cap-provider-2",
            provider="provider_beta",
            model="model_beta",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            supported_visual_types=(VisualType.AI_IMAGE,),
            metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.HIGH,
                latency_tier=TierLevel.MEDIUM,
            ),
        )
        cap_registry = AssetCapabilityRegistry(providers=[StaticProvider((cap_p1, cap_p2))])
        router_svc = AssetRoutePlanningService(
            plan_repository=self.plan_repo,
            shot_repository=self.shot_repo,
            storyboard_repository=self.sb_repo,
            route_plan_repository=self.route_plan_repo,
            capability_registry=cap_registry,
        )
        route_plan = router_svc.create_route_plan(
            CreateAssetRoutePlanInput(
                approved_storyboard_snapshot_id=approved_snapshot.storyboard_snapshot_id,
                routing_strategy=RoutingStrategy.QUALITY_FIRST,
            ),
            capabilities=(cap_p1, cap_p2),
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(route_plan.status, AssetRoutePlanStatus.READY)
        self.assertEqual(len(route_plan.shot_routes), 2)
        ctx = ctx.child_context(
            parent_event_id=ctx.parent_event_id,
            asset_route_plan_id=route_plan.asset_route_plan_id,
        )

        # ---------------------------------------------------------------------
        # d. Asset Execution:
        #    - Shot 1: succeeds first attempt (provider_alpha)
        #    - Shot 2: provider_alpha fails -> fallback to provider_beta -> succeeds
        # ---------------------------------------------------------------------
        file_shot_1 = self.storage_dir / "shot_1_media.png"
        img1 = Image.new("RGB", (640, 360), color=(50, 100, 150))
        img1.save(file_shot_1)

        file_shot_2 = self.storage_dir / "shot_2_media.png"
        img2 = Image.new("RGB", (640, 360), color=(150, 50, 100))
        img2.save(file_shot_2)

        adapter_alpha = MagicMock()
        def alpha_submit(req, cand, path, idempotency_key=None):
            if req.shot_id == "shot-1":
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUCCESS,
                    status="SUCCEEDED",
                    file_path=str(file_shot_1),
                    raw_response={"task": "shot_1_task"},
                )
            else:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    status="FAILED",
                    error_code="RATE_LIMIT",
                    error_message="Provider alpha rate limit",
                )
        adapter_alpha.submit.side_effect = alpha_submit

        adapter_beta = MagicMock()
        adapter_beta.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(file_shot_2),
            raw_response={"task": "shot_2_fallback_task"},
        )

        adapter_registry = AdapterRegistry()
        adapter_registry.register_adapter("provider_alpha", adapter_alpha)
        adapter_registry.register_adapter("provider_beta", adapter_beta)

        shot_exec_service = ShotExecutionService(
            adapter_registry=adapter_registry,
            max_retries_per_candidate=1,
        )
        plan_exec_service = AssetRoutePlanExecutionService(
            session_factory=self.session_factory,
            shot_execution_service=shot_exec_service,
        )
        run_res = plan_exec_service.execute_route_plan(
            plan=route_plan,
            storage_base_dir=self.storage_dir,
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(run_res.state, ExecutionStatus.COMPLETED)
        self.assertEqual(run_res.generated_shots, 2)
        ctx = ctx.child_context(
            parent_event_id=ctx.parent_event_id,
            execution_run_id=run_res.execution_run_id,
        )

        # Retrieve produced assets
        asset_v1 = self.exec_repo.get_shot_asset_version(run_res.shot_results[0].asset_version_id)
        asset_v2 = self.exec_repo.get_shot_asset_version(run_res.shot_results[1].asset_version_id)
        self.assertIsNotNone(asset_v1)
        self.assertIsNotNone(asset_v2)

        # ---------------------------------------------------------------------
        # e. Evaluation:
        #    - Shot 1: passes evaluation
        #    - Shot 2: fails evaluation on Visual Quality
        # ---------------------------------------------------------------------
        target_1 = create_evaluation_target_from_shot(
            shot_revision=rev_1,
            shot_asset_version=asset_v1,
        )
        self.eval_repo.save_target(target_1)

        target_2 = create_evaluation_target_from_shot(
            shot_revision=rev_2,
            shot_asset_version=asset_v2,
        )
        self.eval_repo.save_target(target_2)
        self.session.commit()

        mock_eval_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
        mock_eval_adapter.evaluator_version = "multimodal-eval-v1"

        def eval_observer(prompt, image_paths):
            # If prompt asks for Visual Quality on shot 2 -> return low score
            if "VISUAL QUALITY" in prompt and any("shot_2_media" in str(p) for p in image_paths):
                return MultimodalEvaluationObserverResult(
                    status=DimensionEvaluationStatus.SCORED,
                    score=0.45,
                    reason_codes=("BLURRY_ARTIFACTS",),
                    concise_summary="Artifacts detected in visual rendering",
                    evaluator_version="multimodal-eval-v1",
                )
            return MultimodalEvaluationObserverResult(
                status=DimensionEvaluationStatus.SCORED,
                score=0.92,
                reason_codes=("HIGH_QUALITY",),
                concise_summary="Accurate and sharp presentation",
                evaluator_version="multimodal-eval-v1",
            )

        mock_eval_adapter.evaluate_observation.side_effect = eval_observer

        eval_service = ShotAssetEvaluationService(self.session)
        snap_1 = eval_service.evaluate_target(
            evaluation_target_id=target_1.evaluation_target_id,
            policy=EvaluationPolicy(),
            evaluator_adapter=mock_eval_adapter,
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(snap_1.decision, EvaluationDecision.PASS)

        snap_2 = eval_service.evaluate_target(
            evaluation_target_id=target_2.evaluation_target_id,
            policy=EvaluationPolicy(),
            evaluator_adapter=mock_eval_adapter,
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(snap_2.decision, EvaluationDecision.FAIL)

        # ---------------------------------------------------------------------
        # f. Quality Remediation:
        #    - Shot 2: policy decides REGENERATE_SAME_ROUTE
        #    - Execution produces second asset version
        #    - Second evaluation passes -> ACCEPT_ASSET
        # ---------------------------------------------------------------------
        file_shot_2_regen = self.storage_dir / "shot_2_regen.png"
        img_regen = Image.new("RGB", (640, 360), color=(100, 150, 50))
        img_regen.save(file_shot_2_regen)

        adapter_beta.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(file_shot_2_regen),
        )

        dim_res_2 = self.eval_repo.list_dimension_results_for_target(target_2.evaluation_target_id)
        route_dec_2 = route_plan.shot_routes[1].route_decision

        pass_snap_2 = EvaluationSnapshot(
            evaluation_snapshot_id="snap-2-pass",
            evaluation_target_id=target_2.evaluation_target_id,
            dimension_result_ids=tuple(d.dimension_result_id for d in dim_res_2),
            decision=EvaluationDecision.PASS,
            overall_score=0.92,
            policy_version="v1.0",
            evaluator_version="multimodal-eval-v1",
        )

        def remediation_eval_fn(srev, aver):
            return pass_snap_2, list(dim_res_2)

        remed_service = QualityRemediationService(
            execution_service=shot_exec_service,
            policy=QualityRemediationPolicy(),
            storage_base_dir=self.storage_dir,
        )
        remed_res = remed_service.run_remediation_chain(
            shot_revision=rev_2,
            initial_asset_version=asset_v2,
            initial_snapshot=snap_2,
            initial_dimension_results=dim_res_2,
            route_decision=route_dec_2,
            beat=plan_rev.beats[1],
            evaluation_fn=remediation_eval_fn,
            trace_context=ctx,
            trace_writer=self.trace_service.writer,
        )
        self.assertEqual(remed_res.status, "ACCEPTED")
        self.assertEqual(remed_res.final_action, QualityRemediationAction.ACCEPT_ASSET)

        # ---------------------------------------------------------------------
        # Trace Assertions & Read-Model / Benchmark Verification
        # ---------------------------------------------------------------------
        events = self.trace_service.reader.list_events_by_trace(trace_id)
        self.assertGreater(len(events), 15)

        # Invariant: EXACTLY ONE trace_id spans the entire production run
        trace_ids = {e.trace_id for e in events}
        self.assertEqual(trace_ids, {trace_id})

        event_types = [e.event_type for e in events]

        # Verify presence of all expected key lifecycle decision events
        expected_types = [
            TraceEventType.CONTENT_PLANNER_STARTED,
            TraceEventType.STORYBOARD_GENERATION_STARTED,
            TraceEventType.STORYBOARD_APPROVED,
            TraceEventType.ASSET_ROUTE_PLAN_CREATED,
            TraceEventType.EXECUTION_RUN_STARTED,
            TraceEventType.SHOT_EXECUTION_STARTED,
            TraceEventType.EXECUTION_ATTEMPT_STARTED,
            TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
            TraceEventType.EXECUTION_FALLBACK_SELECTED,
            TraceEventType.SHOT_ASSET_CREATED,
            TraceEventType.EXECUTION_RUN_COMPLETED,
            TraceEventType.EVALUATION_STARTED,
            TraceEventType.DIMENSION_EVALUATION_COMPLETED,
            TraceEventType.EVALUATION_COMPLETED,
            TraceEventType.QUALITY_REMEDIATION_DECIDED,
            TraceEventType.QUALITY_ASSET_ACCEPTED,
        ]
        for exp_type in expected_types:
            self.assertIn(exp_type, event_types, f"Missing expected event: {exp_type.value}")

        # Check detail view
        detail_view = self.trace_service.get_trace_detail_view(trace_id)
        self.assertIsNotNone(detail_view)
        self.assertEqual(detail_view.trace_id, trace_id)
        self.assertGreaterEqual(detail_view.total_duration_ms, 0.0)
        self.assertIn("shot-1", detail_view.shot_ids)
        self.assertIn("shot-2", detail_view.shot_ids)

        # Check that durations on completed events are non-negative
        for ev in events:
            if ev.duration_ms is not None:
                self.assertGreaterEqual(ev.duration_ms, 0.0)

        # Phase 7.2 Benchmark Metric Computability Verification:
        # 1. Total shots planned and executed
        shot_started_events = [e for e in events if e.event_type == TraceEventType.SHOT_EXECUTION_STARTED]
        self.assertEqual(len(shot_started_events), 2)

        # 2. Provider fallback rate
        fallback_events = [e for e in events if e.event_type == TraceEventType.EXECUTION_FALLBACK_SELECTED]
        self.assertEqual(len(fallback_events), 1)
        self.assertEqual(fallback_events[0].attributes["from_provider"], "provider_alpha")
        self.assertEqual(fallback_events[0].attributes["to_provider"], "provider_beta")

        # 3. First-pass success rate (evaluation)
        eval_comp_events = [e for e in events if e.event_type == TraceEventType.EVALUATION_COMPLETED]
        first_pass_decisions = [e.attributes["decision"] for e in eval_comp_events[:2]]
        self.assertEqual(first_pass_decisions, ["PASS", "FAIL"])
        first_pass_pass_count = sum(1 for d in first_pass_decisions if d == "PASS")
        first_pass_rate = first_pass_pass_count / len(first_pass_decisions)
        self.assertEqual(first_pass_rate, 0.5)

        # 4. Remediation loop count
        remed_decisions = [e for e in events if e.event_type == TraceEventType.QUALITY_REMEDIATION_DECIDED]
        self.assertGreaterEqual(len(remed_decisions), 1)

        # 5. Asset acceptance
        accepted_events = [e for e in events if e.event_type == TraceEventType.QUALITY_ASSET_ACCEPTED]
        self.assertEqual(len(accepted_events), 1)
