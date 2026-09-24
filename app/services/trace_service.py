from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel

from app.domain.trace import (
    TraceContext,
    TraceDetailView,
    TraceEvent,
    TraceEventStatus,
    TraceEventType,
    TraceRoot,
    TraceTimelineItem,
    sanitize_attributes,
)
from app.persistence.repositories import TraceRepository


class TraceWriter:
    """
    Focused service for recording lifecycle-managed and instantaneous TraceEvents.
    Adheres to the degraded-trace policy: non-critical trace persistence errors
    will not corrupt or abort valid business operations.
    """

    def __init__(
        self,
        repository: TraceRepository | None = None,
        strict: bool = False,
    ):
        self._repository = repository
        self._strict = strict
        self._active_timers: dict[str, float] = {}  # trace_event_id -> perf_counter start

    def create_root(
        self,
        trace_id: str | None = None,
        root_type: str = "PRODUCTION_WORKFLOW",
        root_reference_id: str | None = None,
    ) -> TraceRoot:
        """Creates and persists a new TraceRoot for a production run."""
        tid = trace_id or str(uuid4())
        root = TraceRoot(
            trace_id=tid,
            root_type=root_type,
            root_reference_id=root_reference_id,
        )
        if self._repository is not None:
            try:
                self._repository.add_root(root)
            except Exception as exc:
                logger.warning(f"Trace persistence degraded on create_root: {exc}")
                if self._strict:
                    raise
        return root

    def start_event(
        self,
        context: TraceContext,
        event_type: TraceEventType,
        attributes: BaseModel | dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        """Starts a new TraceEvent in STARTED status and begins monotonic timing."""
        raw_attrs = (
            attributes.model_dump(mode="json")
            if isinstance(attributes, BaseModel)
            else (attributes or {})
        )
        parent_id = parent_event_id or context.parent_event_id

        event = TraceEvent(
            trace_id=context.trace_id,
            parent_event_id=parent_id,
            event_type=event_type,
            status=TraceEventStatus.STARTED,
            started_at=datetime.now(UTC),
            context=context,
            attributes=raw_attrs,
        )

        self._active_timers[event.trace_event_id] = time.perf_counter()

        if self._repository is not None:
            try:
                self._repository.add_event(event)
            except Exception as exc:
                logger.warning(f"Trace persistence degraded on start_event: {exc}")
                if self._strict:
                    raise

        return event

    def complete_event(
        self,
        event: TraceEvent,
        status: TraceEventStatus = TraceEventStatus.SUCCEEDED,
        attributes_update: BaseModel | dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> TraceEvent:
        """Completes an in-flight TraceEvent, calculating monotonic duration in ms."""
        t_now = datetime.now(UTC)
        duration_ms: float | None = None

        t0 = self._active_timers.pop(event.trace_event_id, None)
        if t0 is not None:
            duration_ms = max(0.0, (time.perf_counter() - t0) * 1000.0)

        raw_update = (
            attributes_update.model_dump(mode="json")
            if isinstance(attributes_update, BaseModel)
            else (attributes_update or {})
        )

        completed_event = event.complete(
            status=status,
            completed_at=t_now,
            duration_ms=duration_ms,
            attributes_update=raw_update,
            error_code=error_code,
        )

        if self._repository is not None:
            try:
                self._repository.complete_event(
                    trace_event_id=completed_event.trace_event_id,
                    status=status,
                    completed_at=t_now,
                    duration_ms=completed_event.duration_ms or 0.0,
                    attributes_update=sanitize_attributes(raw_update),
                    error_code=error_code,
                )
            except Exception as exc:
                logger.warning(f"Trace persistence degraded on complete_event: {exc}")
                if self._strict:
                    raise

        return completed_event

    def fail_event(
        self,
        event: TraceEvent,
        error_code: str,
        attributes_update: BaseModel | dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Convenience method to complete an event in FAILED status."""
        return self.complete_event(
            event=event,
            status=TraceEventStatus.FAILED,
            attributes_update=attributes_update,
            error_code=error_code,
        )

    def record_event(
        self,
        context: TraceContext,
        event_type: TraceEventType,
        status: TraceEventStatus = TraceEventStatus.SUCCEEDED,
        attributes: BaseModel | dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error_code: str | None = None,
        parent_event_id: str | None = None,
    ) -> TraceEvent:
        """Records an instantaneous, completed event in a single atomic step."""
        raw_attrs = (
            attributes.model_dump(mode="json")
            if isinstance(attributes, BaseModel)
            else (attributes or {})
        )
        parent_id = parent_event_id or context.parent_event_id
        now = datetime.now(UTC)

        event = TraceEvent(
            trace_id=context.trace_id,
            parent_event_id=parent_id,
            event_type=event_type,
            status=status,
            started_at=now,
            completed_at=now,
            duration_ms=duration_ms,
            context=context,
            attributes=raw_attrs,
            error_code=error_code,
        )

        if self._repository is not None:
            try:
                self._repository.add_event(event)
            except Exception as exc:
                logger.warning(f"Trace persistence degraded on record_event: {exc}")
                if self._strict:
                    raise

        return event


class TraceReader:
    """
    Focused read model service for querying traces, timelines, and entity events.
    """

    def __init__(self, repository: TraceRepository):
        self._repository = repository

    def get_root(self, trace_id: str) -> TraceRoot | None:
        return self._repository.get_root(trace_id)

    def get_timeline(self, trace_id: str) -> list[TraceTimelineItem]:
        events = self._repository.list_events_by_trace(trace_id)
        return [TraceTimelineItem.from_event(e) for e in events]

    def get_trace_detail_view(self, trace_id: str) -> TraceDetailView | None:
        root = self._repository.get_root(trace_id)
        events = self._repository.list_events_by_trace(trace_id)
        if not root and not events:
            return None

        timeline = tuple(TraceTimelineItem.from_event(e) for e in events)
        started_at = events[0].started_at if events else (root.created_at if root else None)
        completed_at = None
        for e in reversed(events):
            if e.completed_at:
                completed_at = e.completed_at
                break

        total_duration_ms = None
        if started_at and completed_at:
            total_duration_ms = max(0.0, (completed_at - started_at).total_seconds() * 1000.0)

        shot_ids = tuple(
            dict.fromkeys(e.context.shot_id for e in events if e.context.shot_id)
        )

        return TraceDetailView(
            trace_id=trace_id,
            root=root,
            total_events=len(events),
            started_at=started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration_ms,
            timeline=timeline,
            shot_ids=shot_ids,
        )

    def list_events_by_trace(self, trace_id: str) -> list[TraceEvent]:
        return self._repository.list_events_by_trace(trace_id)

    def list_events_by_shot(self, shot_id: str, trace_id: str | None = None) -> list[TraceEvent]:
        return self._repository.list_events_by_shot(shot_id, trace_id=trace_id)

    def list_events_by_attempt(
        self, execution_attempt_id: str, trace_id: str | None = None
    ) -> list[TraceEvent]:
        return self._repository.list_events_by_attempt(
            execution_attempt_id, trace_id=trace_id
        )

    def list_events_by_evaluation(
        self, evaluation_snapshot_id: str, trace_id: str | None = None
    ) -> list[TraceEvent]:
        return self._repository.list_events_by_evaluation(
            evaluation_snapshot_id, trace_id=trace_id
        )


class TraceService:
    """
    Unified service coordinating both trace writing and read-model queries.
    """

    def __init__(
        self,
        repository: TraceRepository | None = None,
        strict: bool = False,
    ):
        self._repository = repository
        self.writer = TraceWriter(repository=repository, strict=strict)
        self.reader = TraceReader(repository=repository) if repository is not None else None

    # Convenience delegators to writer
    def create_root(self, *args: Any, **kwargs: Any) -> TraceRoot:
        return self.writer.create_root(*args, **kwargs)

    def start_event(self, *args: Any, **kwargs: Any) -> TraceEvent:
        return self.writer.start_event(*args, **kwargs)

    def complete_event(self, *args: Any, **kwargs: Any) -> TraceEvent:
        return self.writer.complete_event(*args, **kwargs)

    def record_event(self, *args: Any, **kwargs: Any) -> TraceEvent:
        return self.writer.record_event(*args, **kwargs)

    # Convenience delegators to reader
    def get_timeline(self, trace_id: str) -> list[TraceTimelineItem]:
        if self.reader is None:
            return []
        return self.reader.get_timeline(trace_id)

    def get_trace_detail_view(self, trace_id: str) -> TraceDetailView | None:
        if self.reader is None:
            return None
        return self.reader.get_trace_detail_view(trace_id)

