from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enums
# =============================================================================


class CostEstimateStatus(str, Enum):
    """Status of cost / usage estimate for an executed event."""

    OBSERVED = "OBSERVED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class TraceEventStatus(str, Enum):
    """Lifecycle status of a TraceEvent."""

    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


class TraceEventType(str, Enum):
    """Explicit, stable event types for business decision tracing."""

    # Content Planner
    CONTENT_PLANNER_STARTED = "CONTENT_PLANNER_STARTED"
    CONTENT_PLANNER_COMPLETED = "CONTENT_PLANNER_COMPLETED"
    CONTENT_PLANNER_FAILED = "CONTENT_PLANNER_FAILED"

    # Storyboard
    STORYBOARD_GENERATION_STARTED = "STORYBOARD_GENERATION_STARTED"
    STORYBOARD_GENERATION_COMPLETED = "STORYBOARD_GENERATION_COMPLETED"
    STORYBOARD_REPLAN_COMPLETED = "STORYBOARD_REPLAN_COMPLETED"
    STORYBOARD_EDITED = "STORYBOARD_EDITED"
    STORYBOARD_APPROVED = "STORYBOARD_APPROVED"
    STORYBOARD_STAGE_STARTED = "STORYBOARD_STAGE_STARTED"
    STORYBOARD_CREATED = "STORYBOARD_CREATED"
    STORYBOARD_VALIDATION_FAILED = "STORYBOARD_VALIDATION_FAILED"
    STORYBOARD_WAITING_APPROVAL = "STORYBOARD_WAITING_APPROVAL"

    # Asset Router
    ASSET_ROUTE_PLAN_CREATED = "ASSET_ROUTE_PLAN_CREATED"
    ASSET_ROUTE_DECISION_CREATED = "ASSET_ROUTE_DECISION_CREATED"

    # Execution Lifecycle
    EXECUTION_RUN_STARTED = "EXECUTION_RUN_STARTED"
    SHOT_EXECUTION_STARTED = "SHOT_EXECUTION_STARTED"
    EXECUTION_ATTEMPT_STARTED = "EXECUTION_ATTEMPT_STARTED"
    EXECUTION_ATTEMPT_COMPLETED = "EXECUTION_ATTEMPT_COMPLETED"
    EXECUTION_FALLBACK_SELECTED = "EXECUTION_FALLBACK_SELECTED"
    EXECUTION_RECOVERY_REQUIRED = "EXECUTION_RECOVERY_REQUIRED"
    SHOT_ASSET_CREATED = "SHOT_ASSET_CREATED"
    EXECUTION_RUN_COMPLETED = "EXECUTION_RUN_COMPLETED"

    # Evaluation
    EVALUATION_STARTED = "EVALUATION_STARTED"
    DIMENSION_EVALUATION_COMPLETED = "DIMENSION_EVALUATION_COMPLETED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"

    # Quality Remediation
    QUALITY_REMEDIATION_DECIDED = "QUALITY_REMEDIATION_DECIDED"
    CONTROLLED_VISUAL_REPLAN_CREATED = "CONTROLLED_VISUAL_REPLAN_CREATED"
    QUALITY_ASSET_ACCEPTED = "QUALITY_ASSET_ACCEPTED"

    # Web Research & Evidence
    WEB_RESEARCH_STARTED = "WEB_RESEARCH_STARTED"
    WEB_SEARCH_COMPLETED = "WEB_SEARCH_COMPLETED"
    WEB_SOURCE_FETCHED = "WEB_SOURCE_FETCHED"
    WEB_SOURCE_REJECTED = "WEB_SOURCE_REJECTED"
    WEB_RESEARCH_COMPLETED = "WEB_RESEARCH_COMPLETED"
    WEB_RESEARCH_EXHAUSTED = "WEB_RESEARCH_EXHAUSTED"

    # Knowledge Plan Stage
    KNOWLEDGE_PLAN_STARTED = "KNOWLEDGE_PLAN_STARTED"
    KNOWLEDGE_PLAN_COMPLETED = "KNOWLEDGE_PLAN_COMPLETED"
    KNOWLEDGE_PLAN_VALIDATION_FAILED = "KNOWLEDGE_PLAN_VALIDATION_FAILED"
    KNOWLEDGE_PLAN_NEEDS_EVIDENCE = "KNOWLEDGE_PLAN_NEEDS_EVIDENCE"

    # Script Stage
    SCRIPT_STAGE_STARTED = "SCRIPT_STAGE_STARTED"
    SCRIPT_REVISION_CREATED = "SCRIPT_REVISION_CREATED"
    SCRIPT_VALIDATION_FAILED = "SCRIPT_VALIDATION_FAILED"
    SCRIPT_STAGE_NEEDS_EVIDENCE = "SCRIPT_STAGE_NEEDS_EVIDENCE"


# =============================================================================
# Sanitization & Data Safety
# =============================================================================

_SECRET_PATTERN = re.compile(
    r"(api[_-]?key|secret|password|bearer|auth|token|private[_-]?key)",
    re.IGNORECASE,
)
_COT_PATTERN = re.compile(
    r"(chain[_-]?of[_-]?thought|thought|internal[_-]?monologue|hidden[_-]?reasoning)",
    re.IGNORECASE,
)
_MAX_ATTR_STRING_LEN = 10000  # Prohibit dumping entire media or large base64 blobs


class ChainOfThoughtPersistenceError(ValueError):
    """Raised when an attempt is made to persist Chain-of-Thought or hidden reasoning."""


def sanitize_attribute_value(val: Any) -> Any:
    """Recursively sanitize attribute values, rejecting CoT and stripping secrets."""
    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if _COT_PATTERN.search(str(k)):
                raise ChainOfThoughtPersistenceError(
                    f"Chain-of-Thought / hidden reasoning key '{k}' is forbidden in Trace attributes."
                )
            if _SECRET_PATTERN.search(str(k)):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_attribute_value(v)
        return sanitized
    elif isinstance(val, list | tuple):
        return [sanitize_attribute_value(item) for item in val]
    elif isinstance(val, str):
        if len(val) > _MAX_ATTR_STRING_LEN:
            return val[:200] + f"... [TRUNCATED: original length {len(val)} bytes]"
        return val
    elif isinstance(val, bytes):
        return f"[BINARY CONTENT: {len(val)} bytes redacted]"
    return val


def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Sanitize attributes dictionary, ensuring secret redaction and CoT rejection."""
    return sanitize_attribute_value(attributes)


# =============================================================================
# Trace Root
# =============================================================================


class TraceRoot(BaseModel):
    """
    Stable root concept for a single logical production workflow trace.
    One production run shares exactly one trace_id across all subsystems.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    root_type: str = "PRODUCTION_WORKFLOW"
    root_reference_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_version: str = "v1"


# =============================================================================
# Trace Context
# =============================================================================


class TraceContext(BaseModel):
    """
    Small immutable trace context holding explicit domain IDs.
    Does NOT use generic arbitrary context maps as the primary interface.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    parent_event_id: str | None = None
    content_plan_revision_id: str | None = None
    storyboard_snapshot_id: str | None = None
    asset_route_plan_id: str | None = None
    execution_run_id: str | None = None
    shot_id: str | None = None
    shot_revision_id: str | None = None
    execution_attempt_id: str | None = None
    shot_asset_version_id: str | None = None
    evaluation_snapshot_id: str | None = None

    def child_context(
        self,
        parent_event_id: str,
        **kwargs: Any,
    ) -> TraceContext:
        """Create a child context propagating the trace_id and updating IDs."""
        data = self.model_dump()
        data["parent_event_id"] = parent_event_id
        for k, v in kwargs.items():
            if k in data and v is not None:
                data[k] = v
        return TraceContext(**data)

    def with_ids(self, **kwargs: Any) -> TraceContext:
        """Derive a new TraceContext with specified explicit domain IDs updated."""
        data = self.model_dump()
        for k, v in kwargs.items():
            if k in data:
                data[k] = v
        return TraceContext(**data)


# =============================================================================
# Typed Trace Attributes
# =============================================================================


class CostUsageTraceData(BaseModel):
    """Cost and usage trace metadata with explicit status."""

    model_config = ConfigDict(frozen=True)

    cost_status: CostEstimateStatus = CostEstimateStatus.UNKNOWN
    cost_amount: float | None = None
    currency: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_name: str | None = None

    @model_validator(mode="after")
    def check_unknown_cost_not_zero(self) -> CostUsageTraceData:
        if self.cost_status == CostEstimateStatus.UNKNOWN and self.cost_amount == 0.0:
            # UNKNOWN must never be fabricated into zero cost
            raise ValueError("UNKNOWN cost status must not be converted to 0.0.")
        return self


class PlannerStartedTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    topic: str
    target_duration: float
    provider: str | None = None
    model: str | None = None
    planner_version: str = "v1"


class PlannerCompletedTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_plan_revision_id: str
    beat_count: int
    evidence_ref_count: int = 0
    repair_retry_count: int = 0
    model_reference: str | None = None
    usage: CostUsageTraceData | None = None


class StoryboardGenerationTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_plan_revision_id: str
    storyboard_snapshot_id: str
    beat_lineage_id: str | None = None
    shot_ids: tuple[str, ...] = ()
    shot_revision_ids: tuple[str, ...] = ()
    shot_count: int = 0
    visual_type_distribution: dict[str, int] = Field(default_factory=dict)
    model_reference: str | None = None
    repair_retry_count: int = 0


class StoryboardEditTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    old_snapshot_id: str
    new_snapshot_id: str
    shot_id: str
    old_shot_revision_id: str
    new_shot_revision_id: str


class StoryboardApprovalTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    draft_snapshot_id: str
    approved_snapshot_id: str
    approval_record_id: str
    shot_count: int
    approval_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RouterDecisionTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_route_plan_id: str
    shot_id: str
    shot_revision_id: str
    requested_visual_type: str
    strategy: str
    selection_mode: str
    selected_provider: str
    selected_model: str
    generation_mode: str
    selected_score: float | None = None
    eligible_candidate_count: int = 0
    rejected_candidate_count: int = 0
    key_reason_codes: tuple[str, ...] = ()
    routing_policy_version: str = "v1"


class ExecutionAttemptTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_run_id: str
    shot_execution_id: str
    execution_attempt_id: str
    shot_revision_id: str
    provider: str
    model: str
    generation_mode: str
    attempt_number: int
    technical_failure_category: str | None = None
    retry_fallback_decision: str | None = None
    has_provider_receipt: bool = False
    submission_outcome_unknown: bool = False
    recovery_outcome: str | None = None
    resulting_shot_asset_version_id: str | None = None
    usage: CostUsageTraceData | None = None


class ExecutionFallbackTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_run_id: str
    shot_id: str
    from_provider: str
    from_model: str
    to_provider: str
    to_model: str
    reason_code: str


class EvaluationCompletedTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluation_target_id: str
    evaluation_snapshot_id: str
    shot_asset_version_id: str
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    evaluation_policy_version: str
    dimension_statuses: dict[str, str]
    dimension_scores: dict[str, float | None]
    decision: str
    reused_dimension_count: int = 0
    evaluator_call_count: int = 0


class RemediationTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    remediation_decision_id: str
    quality_chain_id: str
    source_evaluation_snapshot_id: str
    action: str
    reason_codes: tuple[str, ...] = ()
    quality_attempt_index: int = 0
    old_shot_asset_version_id: str | None = None
    new_shot_asset_version_id: str | None = None
    old_route_candidate: str | None = None
    new_route_candidate: str | None = None
    controlled_replan_shot_revision_id: str | None = None


class QualityAssetAcceptedTraceData(BaseModel):
    model_config = ConfigDict(frozen=True)

    quality_selection_id: str
    shot_id: str
    shot_revision_id: str
    accepted_shot_asset_version_id: str
    evaluation_snapshot_id: str


# =============================================================================
# Trace Event
# =============================================================================


class TraceEvent(BaseModel):
    """
    Immutable append-only representation of a business decision trace event.
    Historical completed events cannot have business attributes rewritten.
    """

    model_config = ConfigDict(frozen=True)

    trace_event_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    parent_event_id: str | None = None
    event_type: TraceEventType
    event_version: int = 1
    status: TraceEventStatus = TraceEventStatus.STARTED
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_ms: float | None = None
    context: TraceContext
    attributes: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None

    @field_validator("attributes", mode="before")
    @classmethod
    def validate_attributes(cls, val: Any) -> dict[str, Any]:
        if isinstance(val, BaseModel):
            val = val.model_dump(mode="json")
        if not isinstance(val, dict):
            val = {}
        return sanitize_attributes(val)

    def complete(
        self,
        status: TraceEventStatus = TraceEventStatus.SUCCEEDED,
        completed_at: datetime | None = None,
        duration_ms: float | None = None,
        attributes_update: BaseModel | dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> TraceEvent:
        """
        Finalize a started event into a completed event (lifecycle transition).
        Returns a new completed TraceEvent representing completion of the SAME event.
        """
        if self.status != TraceEventStatus.STARTED:
            raise ValueError(f"Cannot complete event already in status {self.status.value}")

        end_time = completed_at or datetime.now(UTC)
        calc_duration = duration_ms
        if calc_duration is None:
            calc_duration = max(0.0, (end_time - self.started_at).total_seconds() * 1000.0)

        new_attrs = dict(self.attributes)
        if attributes_update is not None:
            if isinstance(attributes_update, BaseModel):
                update_dict = attributes_update.model_dump(mode="json")
            else:
                update_dict = attributes_update
            new_attrs.update(sanitize_attributes(update_dict))

        return TraceEvent(
            trace_event_id=self.trace_event_id,
            trace_id=self.trace_id,
            parent_event_id=self.parent_event_id,
            event_type=self.event_type,
            event_version=self.event_version,
            status=status,
            started_at=self.started_at,
            completed_at=end_time,
            duration_ms=calc_duration,
            context=self.context,
            attributes=new_attrs,
            error_code=error_code or self.error_code,
        )


# =============================================================================
# Trace Read / Query Models
# =============================================================================


class TraceTimelineItem(BaseModel):
    """
    Flattened, ordered timeline representation of a single TraceEvent
    for user visualization and benchmark analysis.
    """
    model_config = ConfigDict(frozen=True)

    trace_event_id: str
    trace_id: str
    parent_event_id: str | None = None
    event_type: TraceEventType
    status: TraceEventStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = None
    shot_id: str | None = None
    shot_revision_id: str | None = None
    execution_attempt_id: str | None = None
    shot_asset_version_id: str | None = None
    evaluation_snapshot_id: str | None = None
    error_code: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_event(cls, event: TraceEvent) -> TraceTimelineItem:
        return cls(
            trace_event_id=event.trace_event_id,
            trace_id=event.trace_id,
            parent_event_id=event.parent_event_id,
            event_type=event.event_type,
            status=event.status,
            started_at=event.started_at,
            completed_at=event.completed_at,
            duration_ms=event.duration_ms,
            shot_id=event.context.shot_id,
            shot_revision_id=event.context.shot_revision_id,
            execution_attempt_id=event.context.execution_attempt_id,
            shot_asset_version_id=event.context.shot_asset_version_id,
            evaluation_snapshot_id=event.context.evaluation_snapshot_id,
            error_code=event.error_code,
            attributes=dict(event.attributes),
        )


class TraceDetailView(BaseModel):
    """
    Complete chronological read model of an end-to-end production workflow trace.
    Serves as the foundation for Phase 7.2 Benchmark Runner and Phase 7.3 Aggregate Metrics.
    """
    model_config = ConfigDict(frozen=True)

    trace_id: str
    root: TraceRoot | None = None
    total_events: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_duration_ms: float | None = None
    timeline: tuple[TraceTimelineItem, ...] = ()
    shot_ids: tuple[str, ...] = ()
