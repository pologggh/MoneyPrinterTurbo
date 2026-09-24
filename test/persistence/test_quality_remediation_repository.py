import os
import unittest

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ShotAssetVersion,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
    StaticCapabilityMetadata,
    TierLevel,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationDecision,
    EvaluationDimension,
    create_evaluation_snapshot,
    create_evaluation_target_from_shot,
)
from app.domain.quality_remediation import (
    QualityRemediationAction,
    QualityRemediationDecision,
    QualityRemediationReasonCode,
    ShotQualitySelection,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot, StoryboardSnapshotState
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)


class TestQualityRemediationRepository(unittest.TestCase):

    def setUp(self):
        db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not db_url or "sqlite" in db_url:
            self.engine = create_engine("sqlite:///:memory:")

            @event.listens_for(self.engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            self.engine = create_engine(db_url)

        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self.session: Session = self.session_factory()
        self.plan_repo = ContentPlanRepository(self.session)
        self.shot_repo = ShotRepository(self.session)
        self.storyboard_repo = StoryboardRepository(self.session)
        self.route_plan_repo = AssetRoutePlanRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)
        self.eval_repo = EvaluationRepository(self.session)

        self._setup_prerequisites()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _setup_prerequisites(self):
        beat = ContentBeat(
            beat_id="b-rem-1",
            beat_lineage_id="lin-rem-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Remediation hook",
            target_duration=3.0,
            importance=0.9,
            evidence_refs=("ev-1",),
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-rem-1",
            revision_number=1,
            topic="Remediation Test Topic",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-rem-1", beat_lineage_id="lin-rem-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-rem-1",
            shot_id="shot-rem-1",
            revision_number=1,
            beat_lineage_id="lin-rem-1",
            created_from_beat_instance_id="b-rem-1",
            narration="Remediation narration",
            target_duration=3.0,
            visual_goal="Visual goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Visual desc",
            generation_prompt="Generation prompt",
            camera_movement="pan",
            evidence_refs=("ev-1",),
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="sb-rem-1",
            content_plan_revision_id="plan-rem-1",
            shot_revision_ids=("rev-rem-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        routing_req = AssetRoutingRequest(
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=3.0,
            visual_goal="Visual goal",
            scene_description="Visual desc",
            generation_prompt="Generation prompt",
            camera_movement="pan",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            beat_lineage_id="lin-rem-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="arp-rem-1",
            storyboard_snapshot_id="sb-rem-1",
            content_plan_revision_id="plan-rem-1",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            routing_policy_version="v1.0",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=(entry,),
            status=AssetRoutePlanStatus.READY,
            total_shots=1,
            routed_shots=1,
            blocked_shots=0,
        )
        self.route_plan_repo.add_route_plan(route_plan)

        run = ExecutionRun(
            execution_run_id="run-rem-1",
            asset_route_plan_id="arp-rem-1",
            content_plan_revision_id="plan-rem-1",
            storyboard_snapshot_id="sb-rem-1",
            total_shots=1,
        )
        self.exec_repo.add_execution_run(run)

        attempt = ExecutionAttempt(
            execution_attempt_id="att-rem-1",
            execution_run_id="run-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            attempt_number=1,
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        self.exec_repo.add_execution_attempt(attempt)

        asset_version = ShotAssetVersion(
            shot_asset_version_id="asset-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            execution_attempt_id="att-rem-1",
            file_path="/tmp/asset.mp4",
            file_hash="e" * 64,
            file_size_bytes=2048,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            duration_seconds=3.0,
            width=1920,
            height=1080,
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )
        self.exec_repo.add_shot_asset_version(asset_version)

        target = create_evaluation_target_from_shot(shot_rev, asset_version)
        self.eval_repo.save_target(target)

        # Create dimension results and snapshot
        dim_results = [
            DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=dim,
                status=DimensionEvaluationStatus.SCORED,
                score=0.90,
            )
            for dim in EvaluationDimension
        ]
        for r in dim_results:
            self.eval_repo.save_dimension_result(r)

        snapshot = create_evaluation_snapshot(
            target=target,
            dimension_results=dim_results,
            decision=EvaluationDecision.PASS,
            snapshot_id="snap-rem-1",
        )
        self.eval_repo.save_snapshot(snapshot)

    def test_save_and_get_remediation_decision(self):
        cand = AssetRouteCandidate(
            capability_id="cap-1",
            provider="mock_p",
            model="mock_m",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            requested_visual_type=VisualType.AI_VIDEO,
            is_eligible=True,
            static_metadata=StaticCapabilityMetadata(
                quality_tier=TierLevel.HIGH,
                cost_tier=TierLevel.MEDIUM,
                latency_tier=TierLevel.LOW,
            ),
        )
        decision = QualityRemediationDecision(
            remediation_decision_id="rem-dec-1",
            quality_chain_id="chain-rem-1",
            evaluation_snapshot_id="snap-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            shot_asset_version_id="asset-rem-1",
            action=QualityRemediationAction.ACCEPT_ASSET,
            reason_codes=(QualityRemediationReasonCode.QUALITY_PASS_ACCEPTED.value,),
            selected_route_candidate=cand,
            explanation="Asset accepted.",
        )

        self.eval_repo.save_remediation_decision(decision)
        loaded = self.eval_repo.get_remediation_decision("rem-dec-1")

        assert loaded is not None
        assert loaded.remediation_decision_id == "rem-dec-1"
        assert loaded.quality_chain_id == "chain-rem-1"
        assert loaded.action == QualityRemediationAction.ACCEPT_ASSET
        assert loaded.selected_route_candidate is not None
        assert loaded.selected_route_candidate.capability_id == "cap-1"

    def test_list_remediation_decisions_for_shot_and_chain(self):
        d1 = QualityRemediationDecision(
            remediation_decision_id="dec-1",
            quality_chain_id="chain-alpha",
            evaluation_snapshot_id="snap-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            shot_asset_version_id="asset-rem-1",
            action=QualityRemediationAction.REGENERATE_SAME_ROUTE,
            quality_attempt_index=0,
        )
        d2 = QualityRemediationDecision(
            remediation_decision_id="dec-2",
            quality_chain_id="chain-alpha",
            evaluation_snapshot_id="snap-rem-1",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            shot_asset_version_id="asset-rem-1",
            action=QualityRemediationAction.ACCEPT_ASSET,
            quality_attempt_index=1,
        )

        self.eval_repo.save_remediation_decision(d1)
        self.eval_repo.save_remediation_decision(d2)

        by_shot = self.eval_repo.list_remediation_decisions_for_shot("shot-rem-1")
        assert len(by_shot) == 2
        assert by_shot[0].remediation_decision_id == "dec-1"
        assert by_shot[1].remediation_decision_id == "dec-2"

        by_chain = self.eval_repo.list_remediation_decisions_for_chain("chain-alpha")
        assert len(by_chain) == 2

    def test_save_and_get_quality_selection(self):
        selection = ShotQualitySelection(
            quality_selection_id="sel-1",
            quality_chain_id="chain-alpha",
            shot_id="shot-rem-1",
            shot_revision_id="rev-rem-1",
            shot_asset_version_id="asset-rem-1",
            evaluation_snapshot_id="snap-rem-1",
            selection_source="EVALUATION_PASS",
        )

        self.eval_repo.save_quality_selection(selection)
        loaded = self.eval_repo.get_quality_selection("sel-1")

        assert loaded is not None
        assert loaded.quality_selection_id == "sel-1"
        assert loaded.shot_asset_version_id == "asset-rem-1"
        assert loaded.evaluation_snapshot_id == "snap-rem-1"

        latest = self.eval_repo.get_quality_selection_for_shot("shot-rem-1")
        assert latest is not None
        assert latest.quality_selection_id == "sel-1"
