import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.domain.content_plan import ContentBeat, ContentPlanRevision
from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot
from app.persistence.converters import (
    content_plan_revision_from_orm,
    content_plan_revision_to_orm,
    shot_from_orm,
    shot_revision_from_orm,
    shot_revision_to_orm,
    shot_to_orm,
    storyboard_snapshot_from_orm,
    storyboard_snapshot_to_orm,
)
from app.persistence.models import (
    Base,
    ContentBeatORM,
    ContentPlanRevisionORM,
    ShotORM,
    ShotRevisionORM,
    StoryboardSnapshotORM,
)


class TestPersistenceLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Allow running against live PostgreSQL if DATABASE_URL or TEST_DATABASE_URL is provided,
        # otherwise use isolated in-memory SQLite with strict foreign keys enabled.
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

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_persist_and_reconstruct_content_plan_revision_with_beats(self):
        """Invariant 1: ContentPlanRevision + Beats can be persisted and reconstructed into domain models."""
        beat_1 = ContentBeat(
            beat_id="beat-inst-1",
            beat_lineage_id="lineage-1",
            beat_type=BeatType.HOOK,
            order=1,
            intent="Capture user attention",
            target_duration=4.0,
            importance=0.95,
            evidence_refs=("doc-ref-1",),
        )
        beat_2 = ContentBeat(
            beat_id="beat-inst-2",
            beat_lineage_id="lineage-2",
            beat_type=BeatType.KNOWLEDGE,
            order=2,
            intent="Explain theorem",
            target_duration=12.0,
            importance=1.0,
            evidence_refs=("doc-ref-2", "doc-ref-3"),
        )
        domain_plan = ContentPlanRevision(
            content_plan_revision_id="plan-rev-1",
            revision_number=1,
            topic="Deep Learning Architecture",
            overall_target_duration=16.0,
            beats=(beat_1, beat_2),
            global_retrieval_snapshot_id="rag-snap-001",
        )

        orm_plan = content_plan_revision_to_orm(domain_plan)
        self.session.add(orm_plan)
        self.session.commit()

        # Query back and reconstruct
        loaded_orm = self.session.get(ContentPlanRevisionORM, "plan-rev-1")
        self.assertIsNotNone(loaded_orm)
        reconstructed_plan = content_plan_revision_from_orm(loaded_orm)

        self.assertEqual(reconstructed_plan.content_plan_revision_id, "plan-rev-1")
        self.assertEqual(reconstructed_plan.topic, "Deep Learning Architecture")
        self.assertEqual(reconstructed_plan.revision_number, 1)
        self.assertEqual(reconstructed_plan.global_retrieval_snapshot_id, "rag-snap-001")
        self.assertEqual(len(reconstructed_plan.beats), 2)
        self.assertEqual(reconstructed_plan.beats[0].beat_id, "beat-inst-1")
        self.assertEqual(reconstructed_plan.beats[0].evidence_refs, ("doc-ref-1",))
        self.assertEqual(reconstructed_plan.beats[1].beat_id, "beat-inst-2")
        self.assertEqual(reconstructed_plan.beats[1].evidence_refs, ("doc-ref-2", "doc-ref-3"))

    def test_shot_with_multiple_shot_revisions_preserves_history(self):
        """Invariant 2: Shot + multiple ShotRevisions can be persisted without overwriting historical revisions."""
        # Setup prerequisite Plan and Beat
        plan_orm = ContentPlanRevisionORM(
            content_plan_revision_id="plan-1",
            revision_number=1,
            topic="Physics",
            overall_target_duration=10.0,
            created_at=ContentPlanRevision(
                content_plan_revision_id="plan-1",
                revision_number=1,
                topic="Physics",
                overall_target_duration=10.0,
            ).created_at,
        )
        beat_orm = ContentBeatORM(
            beat_id="beat-1",
            content_plan_revision_id="plan-1",
            beat_lineage_id="lin-beat-1",
            beat_type="HOOK",
            order=1,
            intent="Intro",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=[],
        )
        shot = Shot(shot_id="shot-1", beat_lineage_id="lin-beat-1", local_order=1)

        self.session.add(plan_orm)
        self.session.add(beat_orm)
        self.session.add(shot_to_orm(shot))
        self.session.commit()

        # Add revision 1
        rev_1 = ShotRevision(
            shot_revision_id="rev-1",
            shot_id="shot-1",
            revision_number=1,
            beat_lineage_id="lin-beat-1",
            created_from_beat_instance_id="beat-1",
            narration="First draft narration",
            target_duration=4.0,
            visual_goal="Show pendulum",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="Pendulum swinging",
            generation_prompt="swinging pendulum",
            camera_movement="static",
        )
        self.session.add(shot_revision_to_orm(rev_1))
        self.session.commit()

        # Add revision 2
        rev_2 = ShotRevision(
            shot_revision_id="rev-2",
            shot_id="shot-1",
            revision_number=2,
            beat_lineage_id="lin-beat-1",
            created_from_beat_instance_id="beat-1",
            narration="Second draft narration with more detail",
            target_duration=4.5,
            visual_goal="Show pendulum with vector overlay",
            visual_type=VisualType.DIAGRAM,
            scene_description="Pendulum with force vectors",
            generation_prompt="force vectors diagram",
            camera_movement="zoom-in",
        )
        self.session.add(shot_revision_to_orm(rev_2))
        self.session.commit()

        # Verify both revisions exist and neither was overwritten
        loaded_shot = self.session.get(ShotORM, "shot-1")
        self.assertIsNotNone(loaded_shot)
        self.assertEqual(len(loaded_shot.revisions), 2)

        rev_orms = {r.revision_number: r for r in loaded_shot.revisions}
        self.assertEqual(rev_orms[1].narration, "First draft narration")
        self.assertEqual(rev_orms[1].visual_type, "STOCK_VIDEO")
        self.assertEqual(rev_orms[2].narration, "Second draft narration with more detail")
        self.assertEqual(rev_orms[2].visual_type, "DIAGRAM")

    def test_unique_shot_id_and_revision_number_enforced(self):
        """Invariant 3: UNIQUE(shot_id, revision_number) is enforced by the database."""
        plan_orm = ContentPlanRevisionORM(
            content_plan_revision_id="plan-u",
            revision_number=1,
            topic="Test",
            overall_target_duration=5.0,
            created_at=ContentPlanRevision(
                content_plan_revision_id="plan-u",
                revision_number=1,
                topic="Test",
                overall_target_duration=5.0,
            ).created_at,
        )
        beat_orm = ContentBeatORM(
            beat_id="beat-u",
            content_plan_revision_id="plan-u",
            beat_lineage_id="lin-u",
            beat_type="HOOK",
            order=1,
            intent="Intro",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=[],
        )
        shot_orm = ShotORM(
            shot_id="shot-u",
            beat_lineage_id="lin-u",
            local_order=1,
            created_at=plan_orm.created_at,
        )
        self.session.add_all([plan_orm, beat_orm, shot_orm])
        self.session.commit()

        rev_1 = ShotRevisionORM(
            shot_revision_id="rev-u1",
            shot_id="shot-u",
            revision_number=1,
            beat_lineage_id="lin-u",
            created_from_beat_instance_id="beat-u",
            narration="Draft 1",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type="AI_VIDEO",
            scene_description="Scene",
            generation_prompt="Prompt",
            camera_movement="None",
            evidence_refs=[],
            created_at=plan_orm.created_at,
        )
        rev_duplicate = ShotRevisionORM(
            shot_revision_id="rev-u2",
            shot_id="shot-u",
            revision_number=1,  # Same shot_id and revision_number
            beat_lineage_id="lin-u",
            created_from_beat_instance_id="beat-u",
            narration="Conflicting revision 1",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type="AI_VIDEO",
            scene_description="Scene",
            generation_prompt="Prompt",
            camera_movement="None",
            evidence_refs=[],
            created_at=plan_orm.created_at,
        )
        self.session.add(rev_1)
        self.session.commit()

        self.session.add(rev_duplicate)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_storyboard_snapshot_references_exact_shot_revisions(self):
        """Invariant 4: StoryboardSnapshot freezes exact ShotRevision IDs, distinguishing DRAFT and APPROVED."""
        plan_orm = ContentPlanRevisionORM(
            content_plan_revision_id="plan-sb",
            revision_number=1,
            topic="Test SB",
            overall_target_duration=10.0,
            created_at=ContentPlanRevision(
                content_plan_revision_id="plan-sb",
                revision_number=1,
                topic="Test SB",
                overall_target_duration=10.0,
            ).created_at,
        )
        beat_orm = ContentBeatORM(
            beat_id="beat-sb",
            content_plan_revision_id="plan-sb",
            beat_lineage_id="lin-sb",
            beat_type="HOOK",
            order=1,
            intent="Intro",
            target_duration=5.0,
            importance=0.8,
            evidence_refs=[],
        )
        shot_orm = ShotORM(
            shot_id="shot-sb",
            beat_lineage_id="lin-sb",
            local_order=1,
            created_at=plan_orm.created_at,
        )
        rev_1 = ShotRevisionORM(
            shot_revision_id="rev-sb-1",
            shot_id="shot-sb",
            revision_number=1,
            beat_lineage_id="lin-sb",
            created_from_beat_instance_id="beat-sb",
            narration="Rev 1",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type="STOCK_VIDEO",
            scene_description="Scene",
            generation_prompt="Prompt",
            camera_movement="None",
            evidence_refs=[],
            created_at=plan_orm.created_at,
        )
        rev_2 = ShotRevisionORM(
            shot_revision_id="rev-sb-2",
            shot_id="shot-sb",
            revision_number=2,
            beat_lineage_id="lin-sb",
            created_from_beat_instance_id="beat-sb",
            narration="Rev 2",
            target_duration=3.0,
            visual_goal="Goal",
            visual_type="STOCK_VIDEO",
            scene_description="Scene",
            generation_prompt="Prompt",
            camera_movement="None",
            evidence_refs=[],
            created_at=plan_orm.created_at,
        )
        self.session.add_all([plan_orm, beat_orm, shot_orm, rev_1, rev_2])
        self.session.commit()

        # Freeze snapshot to rev-1 specifically
        domain_snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-approved-1",
            content_plan_revision_id="plan-sb",
            shot_revision_ids=("rev-sb-1",),
            snapshot_state=StoryboardSnapshotState.APPROVED,
        )
        self.session.add(storyboard_snapshot_to_orm(domain_snap))
        self.session.commit()

        # Query back and verify it refers to rev-1, not rev-2
        loaded_snap_orm = self.session.get(StoryboardSnapshotORM, "snap-approved-1")
        self.assertIsNotNone(loaded_snap_orm)
        reconstructed_snap = storyboard_snapshot_from_orm(loaded_snap_orm)

        self.assertEqual(reconstructed_snap.shot_revision_ids, ("rev-sb-1",))
        self.assertEqual(reconstructed_snap.snapshot_state, StoryboardSnapshotState.APPROVED)

    def test_beat_lineage_shared_across_content_plan_revisions(self):
        """Invariant 5: Multiple ContentPlan revisions may share beat_lineage_id without schema violation."""
        plan_1 = ContentPlanRevision(
            content_plan_revision_id="plan-v1",
            revision_number=1,
            topic="Version 1",
            overall_target_duration=5.0,
            beats=(
                ContentBeat(
                    beat_id="beat-instance-v1",
                    beat_lineage_id="lineage-stable-100",
                    beat_type=BeatType.HOOK,
                    order=1,
                    intent="Initial Hook",
                    target_duration=5.0,
                    importance=0.9,
                ),
            ),
        )
        plan_2 = ContentPlanRevision(
            content_plan_revision_id="plan-v2",
            revision_number=2,
            topic="Version 2",
            overall_target_duration=6.0,
            beats=(
                ContentBeat(
                    beat_id="beat-instance-v2",
                    beat_lineage_id="lineage-stable-100",  # Shared lineage across revisions
                    beat_type=BeatType.HOOK,
                    order=1,
                    intent="Refined Hook",
                    target_duration=6.0,
                    importance=0.95,
                ),
            ),
        )

        self.session.add(content_plan_revision_to_orm(plan_1))
        self.session.add(content_plan_revision_to_orm(plan_2))
        self.session.commit()

        # Query beats across revisions by lineage
        stmt = select(ContentBeatORM).where(ContentBeatORM.beat_lineage_id == "lineage-stable-100")
        results = self.session.scalars(stmt).all()
        self.assertEqual(len(results), 2)
        beat_ids = {b.beat_id for b in results}
        self.assertEqual(beat_ids, {"beat-instance-v1", "beat-instance-v2"})

    def test_deleting_storyboard_snapshot_does_not_destroy_revision_history(self):
        """Invariant 6: Deleting a StoryboardSnapshot does not cascade to ShotRevisions, Shots, or Beats."""
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-del",
            revision_number=1,
            topic="Delete Test",
            overall_target_duration=5.0,
            beats=(
                ContentBeat(
                    beat_id="beat-del",
                    beat_lineage_id="lin-del",
                    beat_type=BeatType.HOOK,
                    order=1,
                    intent="Hook",
                    target_duration=5.0,
                    importance=1.0,
                ),
            ),
        )
        self.session.add(content_plan_revision_to_orm(plan))
        shot = Shot(shot_id="shot-del", beat_lineage_id="lin-del", local_order=1)
        self.session.add(shot_to_orm(shot))
        rev = ShotRevision(
            shot_revision_id="rev-del",
            shot_id="shot-del",
            revision_number=1,
            beat_lineage_id="lin-del",
            created_from_beat_instance_id="beat-del",
            narration="Narr",
            target_duration=5.0,
            visual_goal="Goal",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="Desc",
            generation_prompt="Prompt",
            camera_movement="None",
        )
        self.session.add(shot_revision_to_orm(rev))
        snap = StoryboardSnapshot(
            storyboard_snapshot_id="snap-del",
            content_plan_revision_id="plan-del",
            shot_revision_ids=("rev-del",),
        )
        self.session.add(storyboard_snapshot_to_orm(snap))
        self.session.commit()

        # Delete only the storyboard snapshot
        snap_orm = self.session.get(StoryboardSnapshotORM, "snap-del")
        self.session.delete(snap_orm)
        self.session.commit()

        # Verify underlying revision history is completely intact
        self.assertIsNone(self.session.get(StoryboardSnapshotORM, "snap-del"))
        self.assertIsNotNone(self.session.get(ShotRevisionORM, "rev-del"))
        self.assertIsNotNone(self.session.get(ShotORM, "shot-del"))
        self.assertIsNotNone(self.session.get(ContentBeatORM, "beat-del"))
        self.assertIsNotNone(self.session.get(ContentPlanRevisionORM, "plan-del"))

    def test_foreign_key_restricts_deleting_beat_referenced_by_shot_revision(self):
        """Invariant 7: ForeignKey RESTRICT protects historical beat provenance from accidental deletion."""
        plan = ContentPlanRevision(
            content_plan_revision_id="plan-prot",
            revision_number=1,
            topic="Provenance Protection",
            overall_target_duration=5.0,
            beats=(
                ContentBeat(
                    beat_id="beat-prot",
                    beat_lineage_id="lin-prot",
                    beat_type=BeatType.HOOK,
                    order=1,
                    intent="Hook",
                    target_duration=5.0,
                    importance=1.0,
                ),
            ),
        )
        self.session.add(content_plan_revision_to_orm(plan))
        shot = Shot(shot_id="shot-prot", beat_lineage_id="lin-prot", local_order=1)
        self.session.add(shot_to_orm(shot))
        rev = ShotRevision(
            shot_revision_id="rev-prot",
            shot_id="shot-prot",
            revision_number=1,
            beat_lineage_id="lin-prot",
            created_from_beat_instance_id="beat-prot",
            narration="Narr",
            target_duration=5.0,
            visual_goal="Goal",
            visual_type=VisualType.STOCK_VIDEO,
            scene_description="Desc",
            generation_prompt="Prompt",
            camera_movement="None",
        )
        self.session.add(shot_revision_to_orm(rev))
        self.session.commit()

        # Attempt to delete the beat that is referenced by active ShotRevision
        beat_to_delete = self.session.get(ContentBeatORM, "beat-prot")
        self.session.delete(beat_to_delete)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()


if __name__ == "__main__":
    unittest.main()
