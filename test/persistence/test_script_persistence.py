import unittest
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.enums import BeatType
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.persistence.models import Base
from app.persistence.repositories import (
    KnowledgeVideoTaskRepository,
    ScriptRepository,
)


class TestScriptPersistence(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

    def tear_down(self):
        Base.metadata.drop_all(self.engine)

    def test_save_and_get_script_revision(self):
        """Saves ScriptRevision with child ScriptSegments and retrieves them truthfully."""
        task_id = str(uuid4())
        plan_id = str(uuid4())

        with self.session_factory() as session:
            task_repo = KnowledgeVideoTaskRepository(session)
            task = KnowledgeVideoTask.create(task_id=task_id, topic="Self-Attention")
            task_repo.save_task(task)

            # Insert dummy plan revision row
            from app.persistence.models import ContentPlanRevisionORM

            plan_orm = ContentPlanRevisionORM(
                content_plan_revision_id=plan_id,
                revision_number=1,
                topic="Self-Attention",
                overall_target_duration=30.0,
                created_at=datetime.now(UTC),
            )
            session.add(plan_orm)
            session.commit()

        seg1 = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-1",
            beat_lineage_id="lin-1",
            order=1,
            narration_text="Why do modern transformers rely on attention?",
            target_duration=5.0,
            beat_type=BeatType.HOOK,
            evidence_refs=(),
        )
        seg2 = ScriptSegment(
            script_revision_id="rev-1",
            content_beat_id="beat-2",
            beat_lineage_id="lin-2",
            order=2,
            narration_text="Self-attention allows tokens to build context-aware representations.",
            target_duration=20.0,
            beat_type=BeatType.KNOWLEDGE,
            evidence_refs=("ev-101", "ev-102"),
        )
        rev = ScriptRevision.create(
            script_revision_id="rev-1",
            task_id=task_id,
            content_plan_revision_id=plan_id,
            segments=[seg1, seg2],
            overall_target_duration=25.0,
            language="zh",
            revision_number=1,
        )

        with self.session_factory() as session:
            repo = ScriptRepository(session)
            saved = repo.save_revision(rev)
            session.commit()

            self.assertEqual(saved.script_revision_id, "rev-1")
            self.assertEqual(saved.task_id, task_id)
            self.assertEqual(len(saved.segments), 2)
            self.assertEqual(saved.segments[0].order, 1)
            self.assertEqual(saved.segments[0].beat_type, BeatType.HOOK)
            self.assertEqual(saved.segments[1].order, 2)
            self.assertEqual(saved.segments[1].evidence_refs, ("ev-101", "ev-102"))

        # Query back in a clean session
        with self.session_factory() as session:
            repo = ScriptRepository(session)
            loaded = repo.get_revision("rev-1")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.content_fingerprint, rev.content_fingerprint)
            self.assertEqual(len(loaded.segments), 2)
            self.assertEqual(loaded.segments[1].narration_text, seg2.narration_text)

            latest = repo.get_latest_revision_for_task(task_id)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.script_revision_id, "rev-1")

    def test_unique_constraint_on_segment_order(self):
        """Duplicate segment order within the same revision raises IntegrityError."""
        task_id = str(uuid4())
        plan_id = str(uuid4())

        with self.session_factory() as session:
            task_repo = KnowledgeVideoTaskRepository(session)
            task = KnowledgeVideoTask.create(task_id=task_id, topic="Self-Attention")
            task_repo.save_task(task)

            from app.persistence.models import ContentPlanRevisionORM, ScriptRevisionORM, ScriptSegmentORM

            plan_orm = ContentPlanRevisionORM(
                content_plan_revision_id=plan_id,
                revision_number=1,
                topic="Self-Attention",
                overall_target_duration=30.0,
                created_at=datetime.now(UTC),
            )
            session.add(plan_orm)

            rev_orm = ScriptRevisionORM(
                script_revision_id="rev-dup",
                task_id=task_id,
                content_plan_revision_id=plan_id,
                revision_number=1,
                overall_target_duration=10.0,
                language="zh",
                content_fingerprint="fp123",
                created_at=datetime.now(UTC),
            )
            session.add(rev_orm)
            session.flush()

            # Two segments with same order = 1
            s1 = ScriptSegmentORM(
                script_segment_id="s1",
                script_revision_id="rev-dup",
                content_beat_id="b1",
                order=1,
                narration_text="text 1",
                target_duration=5.0,
                evidence_refs=[],
                created_at=datetime.now(UTC),
            )
            s2 = ScriptSegmentORM(
                script_segment_id="s2",
                script_revision_id="rev-dup",
                content_beat_id="b2",
                order=1,
                narration_text="text 2",
                target_duration=5.0,
                evidence_refs=[],
                created_at=datetime.now(UTC),
            )
            session.add(s1)
            session.add(s2)
            with self.assertRaises(IntegrityError):
                session.commit()


if __name__ == "__main__":
    unittest.main()
