import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.domain.trace import (
    TraceContext,
    TraceEvent,
    TraceEventStatus,
    TraceEventType,
)
from app.persistence.models import Base
from app.persistence.repositories import TraceRepository
from app.services.trace_service import TraceReader, TraceService


class TestTraceView(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = self.session_factory()
        self.repo = TraceRepository(self.session)
        self.service = TraceService(repository=self.repo)

    def tearDown(self):
        self.session.close()

    def test_trace_timeline_item_and_detail_view(self):
        """Verify timeline items and full detail view construction."""
        t0 = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)
        self.service.create_root(trace_id="tr-view-1", root_reference_id="ref-123")

        ctx1 = TraceContext(trace_id="tr-view-1", shot_id="shot-1", execution_attempt_id="att-1")
        ev1 = TraceEvent(
            trace_id="tr-view-1",
            event_type=TraceEventType.EXECUTION_ATTEMPT_STARTED,
            status=TraceEventStatus.STARTED,
            started_at=t0,
            context=ctx1,
            attributes={"attempt_number": 1},
        )
        self.repo.add_event(ev1)

        t1 = t0 + timedelta(seconds=2)
        self.repo.complete_event(
            trace_event_id=ev1.trace_event_id,
            status=TraceEventStatus.SUCCEEDED,
            completed_at=t1,
            duration_ms=2000.0,
            attributes_update={"resulting_shot_asset_version_id": "asset-1"},
        )

        ctx2 = TraceContext(
            trace_id="tr-view-1",
            shot_id="shot-1",
            shot_asset_version_id="asset-1",
            evaluation_snapshot_id="eval-snap-1",
        )
        ev2 = TraceEvent(
            trace_id="tr-view-1",
            event_type=TraceEventType.EVALUATION_COMPLETED,
            status=TraceEventStatus.SUCCEEDED,
            started_at=t1,
            completed_at=t1 + timedelta(seconds=1),
            duration_ms=1000.0,
            context=ctx2,
            attributes={"decision": "PASS"},
        )
        self.repo.add_event(ev2)
        self.session.commit()

        # Test TraceReader
        reader = TraceReader(self.repo)
        timeline = reader.get_timeline("tr-view-1")
        self.assertEqual(len(timeline), 2)
        self.assertEqual(timeline[0].event_type, TraceEventType.EXECUTION_ATTEMPT_STARTED)
        self.assertEqual(timeline[0].shot_id, "shot-1")
        self.assertEqual(timeline[0].execution_attempt_id, "att-1")
        self.assertEqual(timeline[1].event_type, TraceEventType.EVALUATION_COMPLETED)
        self.assertEqual(timeline[1].evaluation_snapshot_id, "eval-snap-1")

        # Test Detail View
        detail = reader.get_trace_detail_view("tr-view-1")
        self.assertIsNotNone(detail)
        self.assertEqual(detail.trace_id, "tr-view-1")
        self.assertEqual(detail.total_events, 2)
        self.assertEqual(detail.shot_ids, ("shot-1",))
        self.assertEqual(detail.timeline[0].attributes["resulting_shot_asset_version_id"], "asset-1")

        # Test filtering by shot, attempt, evaluation
        shot_events = reader.list_events_by_shot("shot-1")
        self.assertEqual(len(shot_events), 2)

        attempt_events = reader.list_events_by_attempt("att-1")
        self.assertEqual(len(attempt_events), 1)
        self.assertEqual(attempt_events[0].event_type, TraceEventType.EXECUTION_ATTEMPT_STARTED)

        eval_events = reader.list_events_by_evaluation("eval-snap-1")
        self.assertEqual(len(eval_events), 1)
        self.assertEqual(eval_events[0].event_type, TraceEventType.EVALUATION_COMPLETED)
