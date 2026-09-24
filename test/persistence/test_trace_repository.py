import os
import unittest
from datetime import UTC, datetime

from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.domain.trace import (
    TraceContext,
    TraceEvent,
    TraceEventStatus,
    TraceEventType,
    TraceRoot,
)
from app.persistence.models import Base
from app.persistence.repositories import TraceRepository


class TestTraceRepository(unittest.TestCase):

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
        self.trace_repo = TraceRepository(self.session)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_add_and_get_trace_root(self):
        """Verify persistence and retrieval of TraceRoot."""
        root = TraceRoot(
            trace_id="tr-repo-1",
            root_type="PRODUCTION_WORKFLOW",
            root_reference_id="ref-task-1",
            metadata_version="v1",
        )
        saved = self.trace_repo.add_root(root)
        self.assertEqual(saved.trace_id, "tr-repo-1")

        loaded = self.trace_repo.get_root("tr-repo-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.trace_id, "tr-repo-1")
        self.assertEqual(loaded.root_reference_id, "ref-task-1")
        self.assertEqual(loaded.metadata_version, "v1")

    def test_add_and_get_trace_event(self):
        """Verify persistence and retrieval of TraceEvent with explicit context."""
        root = TraceRoot(trace_id="tr-repo-2")
        self.trace_repo.add_root(root)

        ctx = TraceContext(
            trace_id="tr-repo-2",
            content_plan_revision_id="cpr-1",
            shot_id="shot-1",
            execution_run_id="run-1",
        )
        event = TraceEvent(
            trace_event_id="ev-repo-1",
            trace_id="tr-repo-2",
            event_type=TraceEventType.CONTENT_PLANNER_STARTED,
            status=TraceEventStatus.STARTED,
            context=ctx,
            attributes={"topic": "Quantum Computing", "target_duration": 45.0},
        )
        self.trace_repo.add_event(event)

        loaded = self.trace_repo.get_event("ev-repo-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.trace_event_id, "ev-repo-1")
        self.assertEqual(loaded.event_type, TraceEventType.CONTENT_PLANNER_STARTED)
        self.assertEqual(loaded.status, TraceEventStatus.STARTED)
        self.assertEqual(loaded.context.content_plan_revision_id, "cpr-1")
        self.assertEqual(loaded.attributes["topic"], "Quantum Computing")

    def test_complete_event_lifecycle(self):
        """Verify lifecycle completion of a previously STARTED event."""
        root = TraceRoot(trace_id="tr-repo-3")
        self.trace_repo.add_root(root)

        ctx = TraceContext(trace_id="tr-repo-3")
        event = TraceEvent(
            trace_event_id="ev-repo-2",
            trace_id="tr-repo-3",
            event_type=TraceEventType.STORYBOARD_GENERATION_STARTED,
            status=TraceEventStatus.STARTED,
            context=ctx,
            attributes={"step": "generating"},
        )
        self.trace_repo.add_event(event)

        t_complete = datetime(2026, 9, 11, 10, 5, 0, tzinfo=UTC)
        completed = self.trace_repo.complete_event(
            trace_event_id="ev-repo-2",
            status=TraceEventStatus.SUCCEEDED,
            completed_at=t_complete,
            duration_ms=4520.0,
            attributes_update={"shot_count": 5},
        )
        self.assertEqual(completed.status, TraceEventStatus.SUCCEEDED)
        self.assertEqual(completed.duration_ms, 4520.0)
        self.assertEqual(completed.attributes["shot_count"], 5)
        self.assertEqual(completed.attributes["step"], "generating")

        # Second completion must fail
        with self.assertRaises(ValueError):
            self.trace_repo.complete_event(
                trace_event_id="ev-repo-2",
                status=TraceEventStatus.SUCCEEDED,
                completed_at=t_complete,
                duration_ms=4520.0,
            )

    def test_list_events_by_trace_chronological(self):
        """Verify list_events_by_trace reconstructs all events for a trace in order."""
        root = TraceRoot(trace_id="tr-repo-4")
        self.trace_repo.add_root(root)

        ctx = TraceContext(trace_id="tr-repo-4")
        e1 = TraceEvent(
            trace_event_id="ev-1",
            trace_id="tr-repo-4",
            event_type=TraceEventType.CONTENT_PLANNER_STARTED,
            started_at=datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC),
            context=ctx,
        )
        e2 = TraceEvent(
            trace_event_id="ev-2",
            trace_id="tr-repo-4",
            parent_event_id="ev-1",
            event_type=TraceEventType.CONTENT_PLANNER_COMPLETED,
            started_at=datetime(2026, 9, 11, 10, 0, 2, tzinfo=UTC),
            context=ctx,
        )
        e3 = TraceEvent(
            trace_event_id="ev-3",
            trace_id="tr-repo-4",
            parent_event_id="ev-2",
            event_type=TraceEventType.STORYBOARD_GENERATION_STARTED,
            started_at=datetime(2026, 9, 11, 10, 0, 3, tzinfo=UTC),
            context=ctx,
        )
        self.trace_repo.add_event(e1)
        self.trace_repo.add_event(e2)
        self.trace_repo.add_event(e3)

        events = self.trace_repo.list_events_by_trace("tr-repo-4")
        self.assertEqual(len(events), 3)
        self.assertEqual([e.trace_event_id for e in events], ["ev-1", "ev-2", "ev-3"])
        self.assertEqual(events[1].parent_event_id, "ev-1")
        self.assertEqual(events[2].parent_event_id, "ev-2")

    def test_foreign_key_enforcement(self):
        """Verify event cannot be added for non-existent TraceRoot."""
        ctx = TraceContext(trace_id="non-existent-root")
        event = TraceEvent(
            trace_event_id="ev-orphan",
            trace_id="non-existent-root",
            event_type=TraceEventType.EXECUTION_RUN_STARTED,
            context=ctx,
        )
        with self.assertRaises(IntegrityError):
            self.trace_repo.add_event(event)
            self.session.commit()
