import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.domain.asset_execution import (
    AssetMediaType,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRun,
    ExecutionStatus,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
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
from app.domain.plan_diff import DuplicateBeatLineageError
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.domain.storyboard_approval import StoryboardApprovalRecord
from app.persistence.models import Base
from app.persistence.repositories import (
    AssetRoutePlanRepository,
    ContentPlanRepository,
    ExecutionRepository,
    ShotRepository,
    StoryboardApprovalRepository,
    StoryboardRepository,
)


class TestRepositories(unittest.TestCase):
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
        self.approval_repo = StoryboardApprovalRepository(self.session)
        self.route_plan_repo = AssetRoutePlanRepository(self.session)
        self.exec_repo = ExecutionRepository(self.session)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_content_plan_repository_add_get_and_latest(self):
        """Tests ContentPlanRepository adding revisions, retrieving by ID, and finding latest by topic."""
        beat = ContentBeat(
            beat_id="b-inst-1",
            beat_lineage_id="lin-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=3.0,
            importance=0.8,
        )
        plan_v1 = ContentPlanRevision(
            content_plan_revision_id="plan-v1",
            revision_number=1,
            topic="Space",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        plan_v2 = ContentPlanRevision(
            content_plan_revision_id="plan-v2",
            revision_number=2,
            topic="Space",
            overall_target_duration=3.0,
            beats=(beat.model_copy(update={"beat_id": "b-inst-2"}),),
        )

        self.plan_repo.add_revision(plan_v1)
        self.plan_repo.add_revision(plan_v2)
        self.session.commit()

        loaded_v1 = self.plan_repo.get_revision("plan-v1")
        self.assertIsNotNone(loaded_v1)
        self.assertEqual(loaded_v1.revision_number, 1)

        latest = self.plan_repo.get_latest_revision(topic="Space")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.content_plan_revision_id, "plan-v2")
        self.assertEqual(latest.revision_number, 2)

    def test_content_plan_repository_rejects_duplicate_lineage(self):
        """Tests ContentPlanRepository validates beat_lineage_id uniqueness within a revision."""
        dup_beat_1 = ContentBeat(
            beat_id="b1",
            beat_lineage_id="dup-lineage",
            beat_type=BeatType.HOOK,
            order=1,
            intent="First",
            target_duration=2.0,
            importance=0.5,
        )
        dup_beat_2 = ContentBeat(
            beat_id="b2",
            beat_lineage_id="dup-lineage",
            beat_type=BeatType.EXAMPLE,
            order=2,
            intent="Second",
            target_duration=2.0,
            importance=0.5,
        )
        invalid_plan = ContentPlanRevision(
            content_plan_revision_id="plan-invalid",
            revision_number=1,
            topic="Invalid",
            overall_target_duration=4.0,
            beats=(dup_beat_1, dup_beat_2),
        )

        with self.assertRaises(DuplicateBeatLineageError):
            self.plan_repo.add_revision(invalid_plan)

    def test_shot_repository_crud_and_revisions_listing(self):
        """Tests ShotRepository adding shot and multiple revisions, then listing revisions in order."""
        # Create beat for provenance
        beat = ContentBeat(
            beat_id="b-prov",
            beat_lineage_id="lin-prov",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Hook",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="p-prov",
            revision_number=1,
            topic="T",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-10", beat_lineage_id="lin-prov", local_order=1)
        self.shot_repo.add_shot(shot)

        rev_1 = ShotRevision(
            shot_revision_id="rev-10-1",
            shot_id="shot-10",
            revision_number=1,
            beat_lineage_id="lin-prov",
            created_from_beat_instance_id="b-prov",
            narration="V1",
            target_duration=5.0,
            visual_goal="Goal 1",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="Desc 1",
            generation_prompt="Prompt 1",
            camera_movement="Pan",
        )
        rev_2 = ShotRevision(
            shot_revision_id="rev-10-2",
            shot_id="shot-10",
            revision_number=2,
            beat_lineage_id="lin-prov",
            created_from_beat_instance_id="b-prov",
            narration="V2",
            target_duration=5.0,
            visual_goal="Goal 2",
            visual_type=VisualType.DIAGRAM,
            scene_description="Desc 2",
            generation_prompt="Prompt 2",
            camera_movement="Zoom",
        )

        self.shot_repo.add_revision(rev_1)
        self.shot_repo.add_revision(rev_2)
        self.session.commit()

        loaded_shot = self.shot_repo.get_shot("shot-10")
        self.assertIsNotNone(loaded_shot)
        self.assertEqual(loaded_shot.shot_id, "shot-10")

        revisions = self.shot_repo.list_revisions("shot-10")
        self.assertEqual(len(revisions), 2)
        self.assertEqual(revisions[0].revision_number, 1)
        self.assertEqual(revisions[1].revision_number, 2)
        self.assertEqual(revisions[0].narration, "V1")
        self.assertEqual(revisions[1].narration, "V2")

    def test_storyboard_repository_reconstructs_exact_shot_revisions_not_latest(self):
        """Case 11: StoryboardRepository reconstructs exact frozen ShotRevisions rather than latest ShotRevision."""
        beat = ContentBeat(
            beat_id="b-prov-sb",
            beat_lineage_id="lin-sb",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Hook",
            target_duration=5.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="p-sb",
            revision_number=1,
            topic="Storyboard Test",
            overall_target_duration=5.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-sb", beat_lineage_id="lin-sb", local_order=1)
        self.shot_repo.add_shot(shot)

        # Revision 1 (historical, frozen into snapshot)
        rev_v1 = ShotRevision(
            shot_revision_id="rev-sb-v1",
            shot_id="shot-sb",
            revision_number=1,
            beat_lineage_id="lin-sb",
            created_from_beat_instance_id="b-prov-sb",
            narration="Historical frozen narration",
            target_duration=5.0,
            visual_goal="Historical visual",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="Historical scene",
            generation_prompt="Historical prompt",
            camera_movement="Static",
        )
        self.shot_repo.add_revision(rev_v1)

        # Snapshot freezes rev-sb-v1
        snapshot = StoryboardSnapshot(
            storyboard_snapshot_id="snapshot-frozen-1",
            content_plan_revision_id="p-sb",
            shot_revision_ids=("rev-sb-v1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(snapshot)

        # Later in time, revision 2 is created on the same Shot
        rev_v2 = ShotRevision(
            shot_revision_id="rev-sb-v2",
            shot_id="shot-sb",
            revision_number=2,
            beat_lineage_id="lin-sb",
            created_from_beat_instance_id="b-prov-sb",
            narration="Updated newer narration",
            target_duration=5.0,
            visual_goal="Updated visual",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Updated scene",
            generation_prompt="Updated prompt",
            camera_movement="Dynamic",
        )
        self.shot_repo.add_revision(rev_v2)
        self.session.commit()

        # Reconstruct storyboard from repository
        reconstructed_revisions = self.storyboard_repo.get_snapshot_shot_revisions(
            "snapshot-frozen-1"
        )
        self.assertEqual(len(reconstructed_revisions), 1)
        reconstructed_rev = reconstructed_revisions[0]

        # Must be exact rev 1, NEVER rev 2
        self.assertEqual(reconstructed_rev.shot_revision_id, "rev-sb-v1")
        self.assertEqual(reconstructed_rev.revision_number, 1)
        self.assertEqual(reconstructed_rev.narration, "Historical frozen narration")
        self.assertEqual(reconstructed_rev.visual_type, VisualType.STOCK_VIDEO)

    def test_storyboard_approval_repository_lifecycle(self):
        """Tests StoryboardApprovalRepository: add record, get by ID, by approved ID, and by source ID."""
        # 1. Setup prerequisite plan and storyboard snapshots
        beat = ContentBeat(
            beat_id="b-appr-1",
            beat_lineage_id="lin-appr-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=4.0,
            importance=0.9,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-appr-1",
            revision_number=1,
            topic="Approval Lifecycle",
            overall_target_duration=4.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        draft_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-draft-1",
            content_plan_revision_id="plan-appr-1",
            shot_revision_ids=(),
            snapshot_state=StoryboardSnapshotState.DRAFT,
        )
        self.storyboard_repo.add_snapshot(draft_snap)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-approved-1",
            content_plan_revision_id="plan-appr-1",
            shot_revision_ids=(),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        # 2. Create and persist StoryboardApprovalRecord
        record = StoryboardApprovalRecord(
            storyboard_approval_id="appr-rec-1",
            source_draft_snapshot_id="snap-draft-1",
            approved_storyboard_snapshot_id="snap-approved-1",
            content_plan_revision_id="plan-appr-1",
            exact_shot_revision_ids=("rev-1", "rev-2"),
            approved_by="tester_alice",
            user_note="Approved for production",
        )
        self.approval_repo.add_approval_record(record)
        self.session.commit()

        # 3. Retrieve by approval_id
        loaded_by_id = self.approval_repo.get_approval_record("appr-rec-1")
        self.assertIsNotNone(loaded_by_id)
        self.assertEqual(loaded_by_id.storyboard_approval_id, "appr-rec-1")
        self.assertEqual(loaded_by_id.source_draft_snapshot_id, "snap-draft-1")
        self.assertEqual(loaded_by_id.approved_storyboard_snapshot_id, "snap-approved-1")
        self.assertEqual(loaded_by_id.content_plan_revision_id, "plan-appr-1")
        self.assertEqual(loaded_by_id.exact_shot_revision_ids, ("rev-1", "rev-2"))
        self.assertEqual(loaded_by_id.approved_by, "tester_alice")
        self.assertEqual(loaded_by_id.user_note, "Approved for production")

        # 4. Retrieve by approved_snapshot_id
        loaded_by_approved = self.approval_repo.get_approval_by_approved_snapshot_id(
            "snap-approved-1"
        )
        self.assertIsNotNone(loaded_by_approved)
        self.assertEqual(loaded_by_approved.storyboard_approval_id, "appr-rec-1")

        # 5. Retrieve by source_draft_id
        loaded_by_source = self.approval_repo.get_approval_by_source_draft_id(
            "snap-draft-1"
        )
        self.assertIsNotNone(loaded_by_source)
        self.assertEqual(loaded_by_source.storyboard_approval_id, "appr-rec-1")

        # 6. Non-existent returns None
        self.assertIsNone(self.approval_repo.get_approval_record("non-existent"))
        self.assertIsNone(
            self.approval_repo.get_approval_by_approved_snapshot_id("non-existent")
        )

    def test_asset_route_plan_repository_lifecycle(self):
        """Tests AssetRoutePlanRepository adding plans with shot entries and querying by ID and snapshot ID."""
        # 1. Prerequisite entities
        beat = ContentBeat(
            beat_id="b-route-1",
            beat_lineage_id="lin-route-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=4.0,
            importance=0.8,
        )
        plan_rev = ContentPlanRevision(
            content_plan_revision_id="plan-route-rev-1",
            revision_number=1,
            topic="Routing Test",
            overall_target_duration=4.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan_rev)

        shot = Shot(shot_id="shot-r-1", beat_lineage_id="lin-route-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-r-1",
            shot_id="shot-r-1",
            revision_number=1,
            beat_lineage_id="lin-route-1",
            created_from_beat_instance_id="b-route-1",
            narration="Test routing narration",
            target_duration=4.0,
            visual_goal="Test visual",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-route-appr-1",
            content_plan_revision_id="plan-route-rev-1",
            shot_revision_ids=("rev-r-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        # 2. Build ShotRoutePlanEntry and AssetRoutePlan
        routing_req = AssetRoutingRequest(
            shot_id="shot-r-1",
            shot_revision_id="rev-r-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=4.0,
            visual_goal="Test visual",
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-r-1",
            shot_revision_id="rev-r-1",
            routing_strategy=RoutingStrategy.BALANCED,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-r-1",
            shot_revision_id="rev-r-1",
            beat_lineage_id="lin-route-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-001",
            storyboard_snapshot_id="snap-route-appr-1",
            content_plan_revision_id="plan-route-rev-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=(entry,),
            status=AssetRoutePlanStatus.READY,
            total_shots=1,
            routed_shots=1,
            blocked_shots=0,
        )

        # 3. Add and commit
        self.route_plan_repo.add_route_plan(route_plan)
        self.session.commit()

        # 4. Query by ID
        loaded = self.route_plan_repo.get_route_plan("plan-route-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.asset_route_plan_id, "plan-route-001")
        self.assertEqual(loaded.storyboard_snapshot_id, "snap-route-appr-1")
        self.assertEqual(loaded.status, AssetRoutePlanStatus.READY)
        self.assertTrue(loaded.is_ready)
        self.assertEqual(len(loaded.shot_routes), 1)
        self.assertEqual(loaded.shot_routes[0].shot_id, "shot-r-1")
        self.assertEqual(loaded.shot_routes[0].shot_revision_id, "rev-r-1")

        # 5. Query latest by snapshot ID
        latest = self.route_plan_repo.get_latest_route_plan_for_snapshot("snap-route-appr-1")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.asset_route_plan_id, "plan-route-001")

        # 6. List all for snapshot
        all_plans = self.route_plan_repo.list_route_plans_for_snapshot("snap-route-appr-1")
        self.assertEqual(len(all_plans), 1)

    def test_execution_repository_lifecycle(self):
        """Tests ExecutionRepository adding/updating ExecutionRun, ExecutionAttempt, and ShotAssetVersion."""
        # 1. Setup prerequisite entities
        beat = ContentBeat(
            beat_id="b-ex-1",
            beat_lineage_id="lin-ex-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=3.0,
            importance=0.8,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-ex-rev-1",
            revision_number=1,
            topic="Execution Test",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-ex-1", beat_lineage_id="lin-ex-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-ex-1",
            shot_id="shot-ex-1",
            revision_number=1,
            beat_lineage_id="lin-ex-1",
            created_from_beat_instance_id="b-ex-1",
            narration="Narration",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Futuristic city",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-ex-1",
            content_plan_revision_id="plan-ex-rev-1",
            shot_revision_ids=("rev-ex-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        routing_req = AssetRoutingRequest(
            shot_id="shot-ex-1",
            shot_revision_id="rev-ex-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=3.0,
            visual_goal="Goal",
            scene_description="Futuristic city",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-ex-1",
            shot_revision_id="rev-ex-1",
            routing_strategy=RoutingStrategy.BALANCED,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-ex-1",
            shot_revision_id="rev-ex-1",
            beat_lineage_id="lin-ex-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-ex-1",
            storyboard_snapshot_id="snap-ex-1",
            content_plan_revision_id="plan-ex-rev-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=(entry,),
            status=AssetRoutePlanStatus.READY,
            total_shots=1,
            routed_shots=1,
            blocked_shots=0,
        )
        self.route_plan_repo.add_route_plan(route_plan)
        self.session.commit()

        # 2. Add ExecutionRun
        run = ExecutionRun(
            execution_run_id="run-001",
            asset_route_plan_id="plan-route-ex-1",
            storyboard_snapshot_id="snap-ex-1",
            total_shots=1,
        )
        self.exec_repo.add_execution_run(run)
        self.session.commit()

        loaded_run = self.exec_repo.get_execution_run("run-001")
        self.assertIsNotNone(loaded_run)
        self.assertEqual(loaded_run.status, ExecutionStatus.PENDING)
        self.assertEqual(loaded_run.total_shots, 1)

        # 3. Add ExecutionAttempt
        attempt = ExecutionAttempt(
            execution_attempt_id="att-001",
            execution_run_id="run-001",
            shot_id="shot-ex-1",
            shot_revision_id="rev-ex-1",
            attempt_number=1,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )
        self.exec_repo.add_execution_attempt(attempt)
        self.session.commit()

        loaded_attempt = self.exec_repo.get_execution_attempt("att-001")
        self.assertIsNotNone(loaded_attempt)
        self.assertEqual(loaded_attempt.status, ExecutionAttemptStatus.PENDING)

        # 4. Update ExecutionAttempt
        succeeded_attempt = loaded_attempt.mark_running().mark_succeeded(
            duration_seconds=8.5,
            raw_response={"task_id": "seedance-xyz"},
        )
        self.exec_repo.update_execution_attempt(succeeded_attempt)
        self.session.commit()

        updated_attempt = self.exec_repo.get_execution_attempt("att-001")
        self.assertEqual(updated_attempt.status, ExecutionAttemptStatus.SUCCEEDED)
        self.assertEqual(updated_attempt.duration_seconds, 8.5)
        self.assertEqual(
            updated_attempt.raw_provider_response, {"task_id": "seedance-xyz"}
        )

        attempts = self.exec_repo.list_attempts_for_run("run-001")
        self.assertEqual(len(attempts), 1)

        # 5. Add ShotAssetVersion
        asset_version = ShotAssetVersion(
            shot_asset_version_id="asset-v1",
            shot_id="shot-ex-1",
            shot_revision_id="rev-ex-1",
            execution_attempt_id="att-001",
            file_path="storage/shot_assets/shot-ex-1/asset-v1.mp4",
            file_hash="e" * 64,
            file_size_bytes=204800,
            media_type=AssetMediaType.VIDEO,
            mime_type="video/mp4",
            width=1920,
            height=1080,
            duration_seconds=3.0,
            fps=30.0,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
        )
        self.exec_repo.add_shot_asset_version(asset_version)
        self.session.commit()

        loaded_version = self.exec_repo.get_shot_asset_version("asset-v1")
        self.assertIsNotNone(loaded_version)
        self.assertEqual(loaded_version.file_hash, "e" * 64)
        self.assertEqual(loaded_version.width, 1920)

        versions = self.exec_repo.list_asset_versions_for_shot_revision("rev-ex-1")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].shot_asset_version_id, "asset-v1")

        # 6. Update ExecutionRun progress
        updated_run = self.exec_repo.update_execution_run_status(
            run_id="run-001",
            status=ExecutionStatus.SUCCEEDED,
            succeeded_shots=1,
            failed_shots=0,
        )
        self.assertIsNotNone(updated_run)
        self.assertEqual(updated_run.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(updated_run.succeeded_shots, 1)

    def test_shot_execution_repository_lifecycle(self):
        """Tests ShotExecution CRUD lifecycle, status updates, and listing in ExecutionRepository."""
        # 1. Prerequisite entities
        beat = ContentBeat(
            beat_id="b-se-1",
            beat_lineage_id="lin-se-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=3.0,
            importance=0.8,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-se-rev-1",
            revision_number=1,
            topic="ShotExecution Test",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-se-1", beat_lineage_id="lin-se-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-se-1",
            shot_id="shot-se-1",
            revision_number=1,
            beat_lineage_id="lin-se-1",
            created_from_beat_instance_id="b-se-1",
            narration="Narration",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-se-1",
            content_plan_revision_id="plan-se-rev-1",
            shot_revision_ids=("rev-se-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        routing_req = AssetRoutingRequest(
            shot_id="shot-se-1",
            shot_revision_id="rev-se-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=3.0,
            visual_goal="Goal",
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-se-1",
            shot_revision_id="rev-se-1",
            routing_strategy=RoutingStrategy.BALANCED,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-se-1",
            shot_revision_id="rev-se-1",
            beat_lineage_id="lin-se-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-se-1",
            storyboard_snapshot_id="snap-se-1",
            content_plan_revision_id="plan-se-rev-1",
            routing_strategy=RoutingStrategy.BALANCED,
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
            execution_run_id="run-se-001",
            asset_route_plan_id="plan-route-se-1",
            storyboard_snapshot_id="snap-se-1",
            total_shots=1,
        )
        self.exec_repo.add_execution_run(run)
        self.session.commit()

        # 2. Add ShotExecution
        shot_exec = ShotExecution(
            shot_execution_id="se-001",
            execution_run_id="run-se-001",
            shot_id="shot-se-1",
            shot_revision_id="rev-se-1",
            route_decision=route_decision,
        )
        self.exec_repo.add_shot_execution(shot_exec)
        self.session.commit()

        # 3. Retrieve
        loaded = self.exec_repo.get_shot_execution("se-001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, ShotExecutionStatus.PENDING)
        self.assertEqual(loaded.shot_id, "shot-se-1")

        # 4. Update with SUBMISSION_OUTCOME_UNKNOWN
        updated_exec = loaded.mark_submission_unknown(
            error_code="SUBMISSION_OUTCOME_UNKNOWN",
            error_message="Task pending confirmation",
            remote_task_id="remote-task-888",
            attempt_id="att-test-1",
        )
        self.exec_repo.update_shot_execution(updated_exec)
        self.session.commit()

        reloaded = self.exec_repo.get_shot_execution("se-001")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.status, ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN)
        self.assertEqual(reloaded.unconfirmed_remote_task_id, "remote-task-888")
        self.assertIn("att-test-1", reloaded.attempt_ids)

        # 5. List executions for run
        run_execs = self.exec_repo.list_shot_executions_for_run("run-se-001")
        self.assertEqual(len(run_execs), 1)
        self.assertEqual(run_execs[0].shot_execution_id, "se-001")

    def test_provider_lifecycle_records_persistence(self):
        """Tests persisting and retrieving AttemptRequest, ProviderReceipt, AttemptResult, and ExecutionTransition."""
        from app.domain.asset_execution import (
            AttemptRequest,
            AttemptResult,
            ExecutionAttempt,
            ExecutionAttemptStatus,
            ExecutionRun,
            ExecutionStatus,
            ExecutionTransition,
            ProviderOutcomeType,
            ProviderReceipt,
        )
        from app.domain.asset_router import GenerationMode

        # Prerequisite plan, shot, snapshot, route_plan
        beat = ContentBeat(
            beat_id="b-lc-1",
            beat_lineage_id="lin-lc-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Intro",
            target_duration=3.0,
            importance=0.8,
        )
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-lc-rev-1",
            revision_number=1,
            topic="Lifecycle Test",
            overall_target_duration=3.0,
            beats=(beat,),
        )
        self.plan_repo.add_revision(plan)

        shot = Shot(shot_id="shot-lc-1", beat_lineage_id="lin-lc-1", local_order=1)
        self.shot_repo.add_shot(shot)

        shot_rev = ShotRevision(
            shot_revision_id="rev-lc-1",
            shot_id="shot-lc-1",
            revision_number=1,
            beat_lineage_id="lin-lc-1",
            created_from_beat_instance_id="b-lc-1",
            narration="Narration",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type=VisualType.AI_VIDEO,
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        self.shot_repo.add_revision(shot_rev)

        approved_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-lc-1",
            content_plan_revision_id="plan-lc-rev-1",
            shot_revision_ids=("rev-lc-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.storyboard_repo.add_snapshot(approved_snap)

        routing_req = AssetRoutingRequest(
            shot_id="shot-lc-1",
            shot_revision_id="rev-lc-1",
            requested_visual_type=VisualType.AI_VIDEO,
            target_duration=3.0,
            visual_goal="Goal",
            scene_description="Futuristic scene",
            generation_prompt="cinematic visual",
            camera_movement="Pan",
        )
        route_decision = AssetRouteDecision(
            shot_id="shot-lc-1",
            shot_revision_id="rev-lc-1",
            routing_strategy=RoutingStrategy.BALANCED,
            requested_visual_type=VisualType.AI_VIDEO,
        )
        entry = ShotRoutePlanEntry(
            shot_id="shot-lc-1",
            shot_revision_id="rev-lc-1",
            beat_lineage_id="lin-lc-1",
            requested_visual_type=VisualType.AI_VIDEO,
            asset_routing_request=routing_req,
            route_decision=route_decision,
            route_status=ShotRoutePlanStatus.ROUTED,
        )
        route_plan = AssetRoutePlan(
            asset_route_plan_id="plan-route-lc-1",
            storyboard_snapshot_id="snap-lc-1",
            content_plan_revision_id="plan-lc-rev-1",
            routing_strategy=RoutingStrategy.BALANCED,
            routing_policy_version="v1.0",
            model_selection_mode=ModelSelectionMode.AUTO,
            shot_routes=(entry,),
            status=AssetRoutePlanStatus.READY,
            total_shots=1,
            routed_shots=1,
            blocked_shots=0,
        )
        self.route_plan_repo.add_route_plan(route_plan)
        self.session.commit()

        run = ExecutionRun(
            execution_run_id="run-lc-1",
            asset_route_plan_id="plan-route-lc-1",
            storyboard_snapshot_id="snap-lc-1",
            status=ExecutionStatus.RUNNING,
            total_shots=1,
        )
        self.exec_repo.add_execution_run(run)
        attempt = ExecutionAttempt(
            execution_attempt_id="att-lc-1",
            execution_run_id="run-lc-1",
            shot_id="shot-lc-1",
            shot_revision_id="rev-lc-1",
            attempt_number=1,
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            status=ExecutionAttemptStatus.CREATED,
        )
        self.exec_repo.add_execution_attempt(attempt)
        self.session.commit()

        # 1. AttemptRequest
        req = AttemptRequest.create_sanitized(
            execution_attempt_id="att-lc-1",
            provider="seedance",
            model="pro",
            generation_mode=GenerationMode.TEXT_TO_VIDEO,
            idempotency_key="idemp-lc-1",
            raw_payload={"prompt": "test prompt", "api_key": "secret123"},
        )
        self.exec_repo.save_attempt_request(req)
        self.session.commit()

        loaded_req = self.exec_repo.get_attempt_request("att-lc-1")
        self.assertIsNotNone(loaded_req)
        self.assertEqual(loaded_req.idempotency_key, "idemp-lc-1")
        self.assertEqual(loaded_req.provider, "seedance")
        self.assertNotIn("secret123", str(loaded_req.sanitized_payload))

        # 2. ProviderReceipt
        receipt = ProviderReceipt(
            execution_attempt_id="att-lc-1",
            provider="seedance",
            provider_job_id="job-seedance-888",
            provider_status="queued",
            sanitized_metadata={"task_type": "async"},
        )
        self.exec_repo.save_provider_receipt(receipt)
        self.session.commit()

        loaded_rcpt = self.exec_repo.get_provider_receipt("att-lc-1")
        self.assertIsNotNone(loaded_rcpt)
        self.assertEqual(loaded_rcpt.provider_job_id, "job-seedance-888")

        # 3. AttemptResult
        result = AttemptResult(
            execution_attempt_id="att-lc-1",
            outcome=ProviderOutcomeType.SUCCESS,
            provider_status="succeeded",
            asset_reference="storage/shot_assets/shot-se-1/out.mp4",
            error_category=None,
            error_code=None,
        )
        self.exec_repo.save_attempt_result(result)
        self.session.commit()

        loaded_res = self.exec_repo.get_attempt_result("att-lc-1")
        self.assertIsNotNone(loaded_res)
        self.assertEqual(loaded_res.outcome, ProviderOutcomeType.SUCCESS)

        # 4. ExecutionTransition
        trans1 = ExecutionTransition(
            entity_type="EXECUTION_ATTEMPT",
            entity_id="att-lc-1",
            from_state="CREATED",
            to_state="READY_TO_SUBMIT",
            reason_code="READY",
        )
        trans2 = ExecutionTransition(
            entity_type="EXECUTION_ATTEMPT",
            entity_id="att-lc-1",
            from_state="READY_TO_SUBMIT",
            to_state="SUBMITTING",
            reason_code="SUBMITTING",
        )
        self.exec_repo.record_transition(trans1)
        self.exec_repo.record_transition(trans2)
        self.session.commit()

        transitions = self.exec_repo.list_transitions("att-lc-1")
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0].from_state, "CREATED")
        self.assertEqual(transitions[1].to_state, "SUBMITTING")


if __name__ == "__main__":
    unittest.main()

