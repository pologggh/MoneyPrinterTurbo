import inspect
import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ShotAssetVersion,
)
from app.domain.asset_router import (
    AssetRouteDecision,
    AssetRoutePlan,
    AssetRoutePlanStatus,
    AssetRoutingRequest,
    GenerationMode,
    ModelSelectionMode,
    RoutingStrategy,
    ShotRoutePlanEntry,
    ShotRoutePlanStatus,
)
from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.evaluation import (
    DimensionEvaluationResult,
    DimensionEvaluationStatus,
    EvaluationContextRef,
    EvaluationDecision,
    EvaluationDimension,
    create_evaluation_snapshot,
    create_evaluation_target_from_shot,
)
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.persistence.models import Base, EvaluationTargetORM
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    EvaluationRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardRepository,
)


class TestEvaluationRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not db_url or "sqlite" in db_url:
            cls.engine = create_engine("sqlite:///:memory:")

            @event.listens_for(cls.engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            cls.engine = create_engine(db_url)

        cls.session_factory = sessionmaker(bind=cls.engine, expire_on_commit=False)

    def setUp(self):
        Base.metadata.create_all(self.engine)
        self.session: Session = self.session_factory()
        self.plan_repo = ContentPlanRepository(self.session)
        self.shot_repo = ShotRepository(self.session)
        self.storyboard_repo = StoryboardRepository(self.session)
        self.route_plan_repo = AssetRoutePlanRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)
        self.eval_repo = EvaluationRepository(self.session)

        # Setup standard prerequisite records (Shot, Revision, Attempt, AssetVersion)
        self._setup_prerequisites()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def _setup_prerequisites(self):
        beat = ContentBeat(
            beat_id="b-eval-1",
            beat_lineage_id="lin-eval-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro evaluation hook",
            target_duration=3.0,
            importance=0.9,
            evidence_refs=("ev-eval-1", "ev-eval-2"),
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-eval-rev-1",
            revision_number=1,
            topic="Evaluation Test Topic",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-eval-1", beat_lineage_id="lin-eval-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-eval-1",
            shot_id="shot-eval-1",
            revision_number=1,
            beat_lineage_id="lin-eval-1",
            created_from_beat_instance_id="b-eval-1",
            narration="Narration of evaluated shot",
            target_duration=3.0,
            visual_goal="Visual goal of evaluated shot",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Laboratory setup",
            generation_prompt="laboratory setup, high quality",
            camera_movement="Static",
            evidence_refs=("ev-eval-1", "ev-eval-2"),
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-eval-1",
            content_plan_revision_id="plan-eval-rev-1",
            shot_revision_ids=("rev-eval-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        routing_req = AssetRoutingRequest(
            shot_id="shot-eval-1",
            shot_revision_id="rev-eval-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=3.0,
            visual_goal="Visual goal",
            scene_description="Laboratory setup",
            generation_prompt="laboratory setup, high quality",
            camera_movement="Static",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-eval-1",
            shot_revision_id="rev-eval-1",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-eval-1",
            shot_revision_id="rev-eval-1",
            beat_lineage_id="lin-eval-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-eval-1",
            storyboard_snapshot_id="snap-eval-1",
            content_plan_revision_id="plan-eval-rev-1",
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
            execution_run_id="run-eval-001",
            asset_route_plan_id="plan-route-eval-1",
            storyboard_snapshot_id="snap-eval-1",
            total_shots=1,
        )
        self.exec_repo.add_execution_run(run)

        attempt = ExecutionAttempt(
            execution_attempt_id="att-eval-001",
            execution_run_id="run-eval-001",
            shot_id="shot-eval-1",
            shot_revision_id="rev-eval-1",
            attempt_number=1,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            status=ExecutionAttemptStatus.SUCCEEDED,
        )
        self.exec_repo.add_execution_attempt(attempt)

        asset_version = ShotAssetVersion(
            shot_asset_version_id="asset-eval-v1",
            shot_id="shot-eval-1",
            shot_revision_id="rev-eval-1",
            execution_attempt_id="att-eval-001",
            file_path="storage/shot_assets/shot-eval-1/asset-eval-v1.mp4",
            file_hash="a" * 64,
            file_size_bytes=1048576,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=3.0,
            fps=30.0,
            provider="seedance",
            model="pro",
            generation_mode="TEXT_TO_VIDEO",
        )
        self.exec_repo.add_shot_asset_version(asset_version)
        self.session.commit()

        self.shot_rev = shot_rev
        self.asset_version = asset_version

    def test_target_persistence_and_retrieval(self):
        """Tests persisting and retrieving an EvaluationTarget with exact frozen identities."""
        target = create_evaluation_target_from_shot(
            shot_revision=self.shot_rev,
            shot_asset_version=self.asset_version,
            context_refs=(
                EvaluationContextRef(type="beat", reference_id="b-eval-1"),
            ),
        )

        saved = self.eval_repo.save_target(target)
        self.session.commit()

        self.assertEqual(saved.evaluation_target_id, target.evaluation_target_id)

        loaded = self.eval_repo.get_target(target.evaluation_target_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.evaluation_target_id, target.evaluation_target_id)
        self.assertEqual(loaded.shot_id, "shot-eval-1")
        self.assertEqual(loaded.shot_revision_id, "rev-eval-1")
        self.assertEqual(loaded.shot_asset_version_id, "asset-eval-v1")
        self.assertEqual(loaded.evidence_refs_snapshot, ("ev-eval-1", "ev-eval-2"))
        self.assertEqual(len(loaded.context_refs), 1)
        self.assertEqual(loaded.context_refs[0].type, "beat")
        self.assertEqual(loaded.context_refs[0].reference_id, "b-eval-1")
        self.assertEqual(loaded.target_version, "v1.0")

    def test_dimension_results_persistence_and_listing(self):
        """Tests persisting various DimensionEvaluationResult statuses and querying them."""
        target = create_evaluation_target_from_shot(self.shot_rev, self.asset_version)
        self.eval_repo.save_target(target)
        self.session.commit()

        res_scored = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.92,
            reason_codes=("SEMANTIC_MATCH_HIGH",),
            concise_summary="Visual corresponds well to narration",
            evaluator_version="mock-eval-1.0",
            dimension_semantics_version="sem-1.0",
        )
        res_indet = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.INDETERMINATE,
            score=None,
            reason_codes=("INSUFFICIENT_EVIDENCE",),
            concise_summary="Evidence insufficient to verify scientific claim",
        )
        res_error = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=DimensionEvaluationStatus.ERROR,
            score=None,
            error_code="TIMEOUT_OR_PARSE_ERROR",
            concise_summary="Evaluation evaluator call failed",
        )

        self.eval_repo.save_dimension_result(res_scored)
        self.eval_repo.save_dimension_result(res_indet)
        self.eval_repo.save_dimension_result(res_error)
        self.session.commit()

        # Check individual retrieval
        loaded_scored = self.eval_repo.get_dimension_result(res_scored.dimension_result_id)
        self.assertIsNotNone(loaded_scored)
        self.assertEqual(loaded_scored.status, DimensionEvaluationStatus.SCORED)
        self.assertEqual(loaded_scored.score, 0.92)
        self.assertEqual(loaded_scored.reason_codes, ("SEMANTIC_MATCH_HIGH",))
        self.assertEqual(loaded_scored.evaluator_version, "mock-eval-1.0")

        loaded_indet = self.eval_repo.get_dimension_result(res_indet.dimension_result_id)
        self.assertIsNotNone(loaded_indet)
        self.assertEqual(loaded_indet.status, DimensionEvaluationStatus.INDETERMINATE)
        self.assertIsNone(loaded_indet.score)

        # List all for target
        all_results = self.eval_repo.list_dimension_results_for_target(target.evaluation_target_id)
        self.assertEqual(len(all_results), 3)
        dims = {r.dimension for r in all_results}
        self.assertEqual(
            dims,
            {
                EvaluationDimension.SEMANTIC_ALIGNMENT,
                EvaluationDimension.KNOWLEDGE_ACCURACY,
                EvaluationDimension.VISUAL_QUALITY,
            },
        )

    def test_snapshot_atomic_persistence_and_eager_loading(self):
        """Tests persisting an EvaluationSnapshot and eagerly reconstructing all dimension results."""
        target = create_evaluation_target_from_shot(self.shot_rev, self.asset_version)
        self.eval_repo.save_target(target)

        d1 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.90,
        )
        d2 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.85,
        )
        d3 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.95,
        )
        d4 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.88,
        )

        for d in (d1, d2, d3, d4):
            self.eval_repo.save_dimension_result(d)

        snapshot = create_evaluation_snapshot(
            target=target,
            dimension_results=(d1, d2, d3, d4),
            decision=EvaluationDecision.PASS,
            evaluator_version="v1.0",
            policy_version="v1.0",
            summary_reason_codes=("ALL_DIMENSIONS_SCORED",),
        )

        saved = self.eval_repo.save_snapshot(snapshot)
        self.session.commit()

        self.assertEqual(saved.evaluation_snapshot_id, snapshot.evaluation_snapshot_id)

        loaded = self.eval_repo.get_snapshot(snapshot.evaluation_snapshot_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.evaluation_snapshot_id, snapshot.evaluation_snapshot_id)
        self.assertEqual(loaded.evaluation_target_id, target.evaluation_target_id)
        self.assertEqual(loaded.decision, EvaluationDecision.PASS)
        self.assertEqual(loaded.summary_reason_codes, ("ALL_DIMENSIONS_SCORED",))
        self.assertEqual(len(loaded.dimension_result_ids), 4)

        # Validate loaded dimension results via get_snapshot_dimension_results
        dim_results = self.eval_repo.get_snapshot_dimension_results(snapshot.evaluation_snapshot_id)
        self.assertEqual(len(dim_results), 4)
        dim_map = {r.dimension: r for r in dim_results}
        self.assertEqual(dim_map[EvaluationDimension.SEMANTIC_ALIGNMENT].score, 0.90)
        self.assertEqual(dim_map[EvaluationDimension.VISUAL_QUALITY].score, 0.85)
        self.assertEqual(dim_map[EvaluationDimension.KNOWLEDGE_ACCURACY].score, 0.95)
        self.assertEqual(dim_map[EvaluationDimension.COMPOSITION_SUITABILITY].score, 0.88)

    def test_multiple_snapshots_for_target_immutable_history(self):
        """Tests that re-evaluation appends new snapshots while strictly preserving older snapshots."""
        target = create_evaluation_target_from_shot(self.shot_rev, self.asset_version)
        self.eval_repo.save_target(target)

        # Snapshot 1: Failed on knowledge accuracy
        d1_v1 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
            status=DimensionEvaluationStatus.SCORED,
            score=0.90,
        )
        d2_v1 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.VISUAL_QUALITY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.85,
        )
        d3_v1 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.30,
        )
        d4_v1 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.88,
        )
        for d in (d1_v1, d2_v1, d3_v1, d4_v1):
            self.eval_repo.save_dimension_result(d)

        snap_v1 = create_evaluation_snapshot(
            target=target,
            dimension_results=(d1_v1, d2_v1, d3_v1, d4_v1),
            decision=EvaluationDecision.FAIL,
            summary_reason_codes=("KNOWLEDGE_ACCURACY_LOW",),
        )
        self.eval_repo.save_snapshot(snap_v1)
        self.session.commit()

        # Snapshot 2: Re-evaluated with updated rubric/evaluator
        d3_v2 = DimensionEvaluationResult(
            evaluation_target_id=target.evaluation_target_id,
            dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
            status=DimensionEvaluationStatus.SCORED,
            score=0.92,
            evaluator_version="v2.0",
        )
        self.eval_repo.save_dimension_result(d3_v2)

        snap_v2 = create_evaluation_snapshot(
            target=target,
            dimension_results=(d1_v1, d2_v1, d3_v2, d4_v1),
            decision=EvaluationDecision.PASS,
            summary_reason_codes=("KNOWLEDGE_ACCURACY_PASSED",),
        )
        self.eval_repo.save_snapshot(snap_v2)
        self.session.commit()

        # Verify history
        all_snapshots = self.eval_repo.list_snapshots_for_target(target.evaluation_target_id)
        self.assertEqual(len(all_snapshots), 2)
        self.assertEqual(all_snapshots[0].evaluation_snapshot_id, snap_v1.evaluation_snapshot_id)
        self.assertEqual(all_snapshots[0].decision, EvaluationDecision.FAIL)
        self.assertEqual(all_snapshots[1].evaluation_snapshot_id, snap_v2.evaluation_snapshot_id)
        self.assertEqual(all_snapshots[1].decision, EvaluationDecision.PASS)

        # Confirm snap_v1 still points to d3_v1 with score 0.30
        dim_results_v1 = self.eval_repo.get_snapshot_dimension_results(snap_v1.evaluation_snapshot_id)
        v1_knowledge = next(
            r for r in dim_results_v1 if r.dimension == EvaluationDimension.KNOWLEDGE_ACCURACY
        )
        self.assertEqual(v1_knowledge.score, 0.30)
        self.assertEqual(v1_knowledge.dimension_result_id, d3_v1.dimension_result_id)

    def test_repository_strictly_has_no_update_or_delete_methods(self):
        """Invariant: EvaluationRepository must be strictly append-only."""
        methods = [
            name
            for name, func in inspect.getmembers(EvaluationRepository, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]

        # No update or delete methods allowed
        forbidden_prefixes = ("update", "delete", "remove", "drop", "modify")
        for m in methods:
            for prefix in forbidden_prefixes:
                self.assertFalse(
                    m.startswith(prefix),
                    f"EvaluationRepository must be append-only, but found forbidden method '{m}'",
                )

    def test_foreign_key_constraints_enforced(self):
        """Invariant: Foreign key constraints ensure target cannot reference non-existent shot revisions."""
        invalid_target_orm = EvaluationTargetORM(
            evaluation_target_id="target-invalid-fk",
            shot_id="shot-eval-1",
            shot_revision_id="non-existent-revision",
            shot_asset_version_id="asset-eval-v1",
            evidence_refs_snapshot=[],
            context_refs=[],
            target_version="v1.0",
            created_at=self.shot_rev.created_at,
        )
        self.session.add(invalid_target_orm)
        with self.assertRaises(IntegrityError):
            self.session.flush()

    def test_snapshot_with_overall_score(self):
        """Invariant: EvaluationSnapshot properly persists and recovers overall_score."""
        target = self.eval_repo.save_target(
            create_evaluation_target_from_shot(self.shot_rev, self.asset_version)
        )

        d1 = self.eval_repo.save_dimension_result(
            DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=EvaluationDimension.SEMANTIC_ALIGNMENT,
                status=DimensionEvaluationStatus.SCORED,
                score=0.90,
            )
        )
        d2 = self.eval_repo.save_dimension_result(
            DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=EvaluationDimension.VISUAL_QUALITY,
                status=DimensionEvaluationStatus.SCORED,
                score=0.85,
            )
        )
        d3 = self.eval_repo.save_dimension_result(
            DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=EvaluationDimension.KNOWLEDGE_ACCURACY,
                status=DimensionEvaluationStatus.SCORED,
                score=0.92,
            )
        )
        d4 = self.eval_repo.save_dimension_result(
            DimensionEvaluationResult(
                evaluation_target_id=target.evaluation_target_id,
                dimension=EvaluationDimension.COMPOSITION_SUITABILITY,
                status=DimensionEvaluationStatus.SCORED,
                score=0.80,
            )
        )

        snap = create_evaluation_snapshot(
            target=target,
            dimension_results=[d1, d2, d3, d4],
            decision=EvaluationDecision.PASS,
            overall_score=0.875,
            summary_reason_codes=["ALL_DIMENSIONS_PASSED"],
        )

        saved = self.eval_repo.save_snapshot(snap)
        self.assertEqual(saved.overall_score, 0.875)

        # Clear session cache and reload
        self.session.expunge_all()
        reloaded = self.eval_repo.get_snapshot(snap.evaluation_snapshot_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.overall_score, 0.875)
        self.assertEqual(reloaded.decision, EvaluationDecision.PASS)

    def test_composition_preview_persistence(self):
        """Invariant: CompositionPreview persists, queries by ID and fingerprint."""
        from app.services.evaluation.composition_preview import CompositionPreview

        preview = CompositionPreview(
            shot_asset_version_id=self.asset_version.shot_asset_version_id,
            render_context_fingerprint="ctx_fp_test_123",
            preview_path="/fake/path/preview.png",
            file_hash="hash_preview_12345",
            width=1080,
            height=1920,
            preview_policy_version="composition-preview-v1",
        )

        saved = self.eval_repo.save_composition_preview(preview)
        self.assertEqual(saved.composition_preview_id, preview.composition_preview_id)

        # Reload by ID
        self.session.expunge_all()
        by_id = self.eval_repo.get_composition_preview(preview.composition_preview_id)
        self.assertIsNotNone(by_id)
        self.assertEqual(by_id.file_hash, "hash_preview_12345")
        self.assertEqual(by_id.width, 1080)
        self.assertEqual(by_id.height, 1920)

        # Reload by fingerprint
        by_fp = self.eval_repo.get_composition_preview_by_fingerprint(
            shot_asset_version_id=self.asset_version.shot_asset_version_id,
            render_context_fingerprint="ctx_fp_test_123",
        )
        self.assertIsNotNone(by_fp)
        self.assertEqual(by_fp.composition_preview_id, preview.composition_preview_id)


if __name__ == "__main__":
    unittest.main()

