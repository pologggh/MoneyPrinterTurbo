import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ProviderOutcomeType,
    ShotAssetVersion,
)
from app.domain.asset_router import (
    AssetCapability,
    AssetRouteCandidate,
    AssetRouteDecision,
    GenerationMode,
    RoutingStrategy,
)
from app.domain.content_plan import ContentBeat
from app.domain.enums import BeatType, VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    EvaluationSnapshot,
    create_evaluation_target_from_shot,
)
from app.domain.evaluation_policy import EvaluationPolicy
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationPolicy,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.trace import (
    TraceContext,
    TraceEventType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
)
from app.services.asset_adapters.base import AdapterExecutionResult
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_capability_registry import AssetCapabilityRegistry
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
from app.services.hybrid_asset_router import HybridAssetRouter
from app.services.shot_execution_service import ShotExecutionService
from app.services.trace_service import TraceWriter


class TestTraceEvaluationRemediation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)

        # Setup SQLite in-memory DB
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()

        self.shot_repo = ShotRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)
        self.eval_repo = EvaluationRepository(self.session)

    def tearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    def test_shot_asset_evaluation_trace(self):
        """Verify EVALUATION_STARTED, DIMENSION_EVALUATION_COMPLETED, and EVALUATION_COMPLETED."""
        mock_trace_repo = MagicMock()
        writer = TraceWriter(repository=mock_trace_repo)

        # Create dummy image
        img_path = self.storage_dir / "test_shot.png"
        img = Image.new("RGB", (640, 360), color=(100, 150, 200))
        img.save(img_path)

        shot = Shot(shot_id="shot_eval_1", beat_lineage_id="lin_eval_1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev_eval_1",
            shot_id="shot_eval_1",
            revision_number=1,
            beat_lineage_id="lin_eval_1",
            created_from_beat_instance_id="beat_inst_1",
            narration="Explaining quantum tunneling phenomenon",
            visual_goal="Particles tunneling through barrier",
            scene_description="Animated diagram of wave function",
            generation_prompt="quantum tunneling wave function, scientific",
            visual_type=VisualType.AI_IMAGE,
            target_duration=4.0,
            camera_movement="static",
            evidence_refs=("quantum tunneling evidence",),
        )
        self.shot_repo.add_revision(shot_rev)

        attempt = ExecutionAttempt(
            execution_attempt_id="att_eval_1",
            execution_run_id="run_eval_1",
            shot_id="shot_eval_1",
            shot_revision_id="rev_eval_1",
            attempt_number=1,
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        self.exec_repo.add_execution_attempt(attempt)

        asset_ver = ShotAssetVersion(
            shot_asset_version_id="asset_eval_1",
            shot_id="shot_eval_1",
            shot_revision_id="rev_eval_1",
            execution_attempt_id="att_eval_1",
            file_path=str(img_path),
            file_hash="a" * 64,
            file_size_bytes=1024,
            media_type=AssetMediaType.IMAGE,
            mime_type="image/png",
            width=640,
            height=360,
            duration_seconds=4.0,
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_IMAGE,
        )
        self.exec_repo.add_shot_asset_version(asset_ver)

        target = create_evaluation_target_from_shot(
            shot_revision=shot_rev,
            shot_asset_version=asset_ver,
        )
        self.eval_repo.save_target(target)
        self.session.commit()

        # Mock adapter
        mock_adapter = MagicMock(spec=MultimodalEvaluatorAdapter)
        mock_adapter.evaluator_version = "mock-evaluator-v1"
        mock_adapter.evaluate_observation.return_value = MultimodalEvaluationObserverResult(
            status=DimensionEvaluationStatus.SCORED,
            score=0.88,
            reason_codes=("HIGH_QUALITY",),
            concise_summary="Clear scientific accuracy",
            evaluator_version="mock-evaluator-v1",
        )

        eval_svc = ShotAssetEvaluationService(self.session)
        ctx = TraceContext(
            trace_id="tr-eval-1",
            shot_id="shot_eval_1",
            shot_revision_id="rev_eval_1",
            shot_asset_version_id="asset_eval_1",
        )

        snapshot = eval_svc.evaluate_target(
            evaluation_target_id=target.evaluation_target_id,
            policy=EvaluationPolicy(),
            evaluator_adapter=mock_adapter,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(snapshot.decision, EvaluationDecision.PASS)

        # Verify emitted events
        event_types = [call[0][0].event_type for call in mock_trace_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.EVALUATION_STARTED, event_types)
        self.assertIn(TraceEventType.DIMENSION_EVALUATION_COMPLETED, event_types)
        self.assertIn(TraceEventType.EVALUATION_COMPLETED, event_types)

        dim_events = [
            call[0][0]
            for call in mock_trace_repo.add_event.call_args_list
            if call[0][0].event_type == TraceEventType.DIMENSION_EVALUATION_COMPLETED
        ]
        self.assertEqual(len(dim_events), 4)

        comp_events = [
            call[0][0]
            for call in mock_trace_repo.add_event.call_args_list
            if call[0][0].event_type == TraceEventType.EVALUATION_COMPLETED
        ]
        self.assertEqual(len(comp_events), 1)
        self.assertEqual(comp_events[0].attributes["decision"], "PASS")

    def test_quality_remediation_trace_decide_and_accept(self):
        """Verify QUALITY_REMEDIATION_DECIDED and QUALITY_ASSET_ACCEPTED."""
        mock_trace_repo = MagicMock()
        writer = TraceWriter(repository=mock_trace_repo)

        shot_rev = ShotRevision(
            shot_revision_id="srev-rem-1",
            shot_id="shot-rem-1",
            revision_number=1,
            beat_lineage_id="lin-rem-1",
            created_from_beat_instance_id="b-1",
            narration="Intro narration",
            target_duration=4.0,
            visual_goal="Goal",
            scene_description="Desc",
            generation_prompt="Prompt",
            visual_type=VisualType.AI_VIDEO,
            camera_movement="pan",
        )
        asset_ver = ShotAssetVersion(
            shot_asset_version_id="asset-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="srev-rem-1",
            execution_attempt_id="att-rem-1",
            file_path="/tmp/fake.mp4",
            file_hash="b" * 64,
            file_size_bytes=100,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=4.0,
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

        dim_res = DimensionEvaluationResult(
            evaluation_target_id="target-rem-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.95,
            dependency_fingerprint="fp1",
            evaluator_version="v1",
        )
        from app.domain.evaluation import EvaluationSnapshot
        snapshot = EvaluationSnapshot(
            evaluation_snapshot_id="snap-rem-1",
            evaluation_target_id="target-rem-1",
            dimension_result_ids=(dim_res.dimension_result_id,),
            decision=EvaluationDecision.PASS,
            overall_score=0.95,
            policy_version="v1",
            evaluator_version="v1",
        )

        candidate = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-rem-1",
            shot_revision_id="srev-rem-1",
            requested_visual_type=VisualType.AI_VIDEO,
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            model_selection_mode="AUTO",
            selected_candidate=candidate,
            eligible_candidates=(candidate,),
        )
        beat = ContentBeat(
            beat_id="b-1",
            beat_lineage_id="lin-rem-1",
            beat_type=BeatType.KNOWLEDGE,
            order=1,
            intent="Explain",
            target_duration=4.0,
            importance=0.8,
        )

        def dummy_eval(srev, aver):
            return snapshot, [dim_res]

        remed_svc = QualityRemediationService(
            policy=QualityRemediationPolicy(),
        )
        ctx = TraceContext(trace_id="tr-rem-1", shot_id="shot-rem-1")

        result = remed_svc.run_remediation_chain(
            shot_revision=shot_rev,
            initial_asset_version=asset_ver,
            initial_snapshot=snapshot,
            initial_dimension_results=[dim_res],
            route_decision=route_decision,
            beat=beat,
            evaluation_fn=dummy_eval,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(result.final_action, QualityRemediationAction.ACCEPT_ASSET)
        self.assertEqual(result.status, "ACCEPTED")

        event_types = [call[0][0].event_type for call in mock_trace_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.QUALITY_REMEDIATION_DECIDED, event_types)
        self.assertIn(TraceEventType.QUALITY_ASSET_ACCEPTED, event_types)

    def test_quality_remediation_trace_controlled_replan(self):
        """Verify CONTROLLED_VISUAL_REPLAN_CREATED is emitted when controlled replan triggers."""
        mock_trace_repo = MagicMock()
        writer = TraceWriter(repository=mock_trace_repo)

        cap1 = AssetCapability(
            capability_id="cap-1",
            provider="mock_p1",
            model="model-1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
        )
        cap2 = AssetCapability(
            capability_id="cap-2",
            provider="mock_p2",
            model="model-2",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            supported_visual_types=(VisualType.AI_VIDEO,),
        )

        class StaticProvider:
            def get_capabilities(self):
                return (cap1, cap2)

        cap_reg = AssetCapabilityRegistry(providers=[StaticProvider()])
        router = HybridAssetRouter(registry=cap_reg)

        adapter = MagicMock()
        adapter.submit.return_value = AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUCCESS,
            status="SUCCEEDED",
            file_path=str(self.storage_dir / "replan.mp4"),
        )
        # Create physical dummy file
        (self.storage_dir / "replan.mp4").write_bytes(b"dummy replan video")

        reg = AdapterRegistry()
        reg.register_adapter("mock_p1", adapter)
        reg.register_adapter("mock_p2", adapter)
        exec_service = ShotExecutionService(adapter_registry=reg)

        shot_rev = ShotRevision(
            shot_revision_id="srev-cr-1",
            shot_id="shot-cr-1",
            revision_number=1,
            beat_lineage_id="lineage-1",
            created_from_beat_instance_id="beat-1",
            narration="Narrative",
            target_duration=5.0,
            visual_goal="Goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Desc",
            generation_prompt="Prompt",
            camera_movement="pan",
            evidence_refs=("ev-1",),
        )

        beat = ContentBeat(
            beat_id="beat-1",
            beat_lineage_id="lineage-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=5.0,
            importance=0.9,
            evidence_refs=("ev-1",),
        )

        initial_asset = ShotAssetVersion(
            shot_asset_version_id="asset-cr-0",
            shot_id="shot-cr-1",
            shot_revision_id="srev-cr-1",
            execution_attempt_id="att-cr-0",
            file_path=str(self.storage_dir / "replan.mp4"),
            file_hash="c" * 64,
            file_size_bytes=100,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=5.0,
            provider="mock_p1",
            model="model-1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )

        cand1 = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_p1",
            model="model-1",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )
        cand2 = AssetRouteCandidate(
            capability_id="cap-2",
            provider="mock_p2",
            model="model-2",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
        )

        route_decision = AssetRouteDecision(
            shot_id="shot-cr-1",
            shot_revision_id="srev-cr-1",
            requested_visual_type=VisualType.AI_VIDEO,
            routing_strategy=RoutingStrategy.BALANCED,
            selected_candidate=cand1,
            eligible_candidates=(cand1, cand2),
            model_selection_mode="AUTO",
        )

        # Dimension result failing semantic alignment to trigger controlled replan
        dim_fail = DimensionEvaluationResult(
            evaluation_target_id="target-cr-1",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.40,
            dependency_fingerprint="fp-cr-1",
            evaluator_version="v1",
        )
        dim_pass = DimensionEvaluationResult(
            evaluation_target_id="target-cr-2",
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.95,
            dependency_fingerprint="fp-cr-2",
            evaluator_version="v1",
        )

        fail_snap = EvaluationSnapshot(
            evaluation_snapshot_id="snap-cr-fail",
            evaluation_target_id="target-cr-1",
            dimension_result_ids=(dim_fail.dimension_result_id,),
            decision=EvaluationDecision.FAIL,
            overall_score=0.40,
            policy_version="v1",
            evaluator_version="v1",
        )
        pass_snap = EvaluationSnapshot(
            evaluation_snapshot_id="snap-cr-pass",
            evaluation_target_id="target-cr-2",
            dimension_result_ids=(dim_pass.dimension_result_id,),
            decision=EvaluationDecision.PASS,
            overall_score=0.95,
            policy_version="v1",
            evaluator_version="v1",
        )

        eval_count = 0

        def stepwise_eval(rev, asset):
            nonlocal eval_count
            eval_count += 1
            if eval_count == 3:  # After replan
                return pass_snap, [dim_pass]
            return fail_snap, [dim_fail]

        policy = QualityRemediationPolicy(
            max_same_route_quality_regenerations=1,
            max_route_fallbacks=1,
            max_controlled_auto_replans=1,
        )
        remed_svc = QualityRemediationService(
            execution_service=exec_service,
            router=router,
            policy=policy,
            storage_base_dir=self.storage_dir,
        )

        ctx = TraceContext(trace_id="tr-cr-1", shot_id="shot-cr-1")
        result = remed_svc.run_remediation_chain(
            shot_revision=shot_rev,
            initial_asset_version=initial_asset,
            initial_snapshot=fail_snap,
            initial_dimension_results=[dim_fail],
            route_decision=route_decision,
            beat=beat,
            evaluation_fn=stepwise_eval,
            trace_context=ctx,
            trace_writer=writer,
        )

        self.assertEqual(result.status, "ACCEPTED")
        event_types = [call[0][0].event_type for call in mock_trace_repo.add_event.call_args_list]
        self.assertIn(TraceEventType.CONTROLLED_VISUAL_REPLAN_CREATED, event_types)

