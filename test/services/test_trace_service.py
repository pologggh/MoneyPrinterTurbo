import unittest
from unittest.mock import MagicMock

from app.domain.trace import (
    TraceContext,
    TraceEventStatus,
    TraceEventType,
)
from app.services.trace_service import TraceWriter


class TestTraceService(unittest.TestCase):

    def test_start_and_complete_event_with_monotonic_duration(self):
        """Verify start_event and complete_event measure monotonic duration in milliseconds."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        root = writer.create_root(trace_id="tr-srv-1")
        self.assertEqual(root.trace_id, "tr-srv-1")
        mock_repo.add_root.assert_called_once()

        ctx = TraceContext(trace_id="tr-srv-1")
        event = writer.start_event(
            context=ctx,
            event_type=TraceEventType.CONTENT_PLANNER_STARTED,
            attributes={"topic": "Black Holes"},
        )
        self.assertEqual(event.status, TraceEventStatus.STARTED)
        mock_repo.add_event.assert_called_once()

        completed = writer.complete_event(
            event=event,
            status=TraceEventStatus.SUCCEEDED,
            attributes_update={"beat_count": 3},
        )
        self.assertEqual(completed.status, TraceEventStatus.SUCCEEDED)
        self.assertIsNotNone(completed.duration_ms)
        self.assertGreaterEqual(completed.duration_ms, 0.0)
        self.assertEqual(completed.attributes["beat_count"], 3)
        mock_repo.complete_event.assert_called_once()

    def test_fail_event_convenience(self):
        """Verify fail_event marks status FAILED and attaches error_code."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        ctx = TraceContext(trace_id="tr-srv-2")
        event = writer.start_event(
            context=ctx,
            event_type=TraceEventType.CONTENT_PLANNER_STARTED,
        )
        failed = writer.fail_event(
            event=event,
            error_code="LLM_TIMEOUT",
            attributes_update={"retries_exhausted": True},
        )
        self.assertEqual(failed.status, TraceEventStatus.FAILED)
        self.assertEqual(failed.error_code, "LLM_TIMEOUT")
        self.assertTrue(failed.attributes["retries_exhausted"])

    def test_record_event_atomic(self):
        """Verify record_event creates an instantaneous completed event."""
        mock_repo = MagicMock()
        writer = TraceWriter(repository=mock_repo)

        ctx = TraceContext(trace_id="tr-srv-3")
        rec = writer.record_event(
            context=ctx,
            event_type=TraceEventType.SHOT_ASSET_CREATED,
            status=TraceEventStatus.SUCCEEDED,
            attributes={"file_hash": "b" * 64},
        )
        self.assertEqual(rec.status, TraceEventStatus.SUCCEEDED)
        self.assertEqual(rec.attributes["file_hash"], "b" * 64)
        mock_repo.add_event.assert_called_once()

    def test_degraded_trace_policy(self):
        """Invariant 24: Trace persistence error degrades gracefully without crashing production flow."""
        failing_repo = MagicMock()
        failing_repo.add_event.side_effect = RuntimeError("Database connection lost")
        failing_repo.complete_event.side_effect = RuntimeError("Database connection lost")

        # Non-strict mode (default) must catch error and return valid event in memory
        writer_lenient = TraceWriter(repository=failing_repo, strict=False)
        ctx = TraceContext(trace_id="tr-srv-4")
        event = writer_lenient.start_event(
            context=ctx,
            event_type=TraceEventType.ASSET_ROUTE_DECISION_CREATED,
        )
        self.assertIsNotNone(event)
        completed = writer_lenient.complete_event(event=event)
        self.assertEqual(completed.status, TraceEventStatus.SUCCEEDED)

        # Strict mode must raise
        writer_strict = TraceWriter(repository=failing_repo, strict=True)
        with self.assertRaises(RuntimeError):
            writer_strict.start_event(
                context=ctx,
                event_type=TraceEventType.ASSET_ROUTE_DECISION_CREATED,
            )
