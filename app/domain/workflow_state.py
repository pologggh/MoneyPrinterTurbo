from __future__ import annotations

from enum import Enum


class TaskStatus(str, Enum):
    """Authoritative lifecycle status of a KnowledgeVideoTask."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_USER = "WAITING_USER"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


class Stage(str, Enum):
    """Linear execution stages of a KnowledgeVideoTask."""
    EVIDENCE = "EVIDENCE"
    KNOWLEDGE_PLAN = "KNOWLEDGE_PLAN"
    SCRIPT = "SCRIPT"
    STORYBOARD = "STORYBOARD"
    PRODUCTION_PLAN = "PRODUCTION_PLAN"
    ASSET = "ASSET"
    AUDIO = "AUDIO"
    COMPOSITION = "COMPOSITION"
    QUALITY_REVIEW = "QUALITY_REVIEW"
    DELIVERY = "DELIVERY"


STAGE_ORDER: tuple[Stage, ...] = (
    Stage.EVIDENCE,
    Stage.KNOWLEDGE_PLAN,
    Stage.SCRIPT,
    Stage.STORYBOARD,
    Stage.PRODUCTION_PLAN,
    Stage.ASSET,
    Stage.AUDIO,
    Stage.COMPOSITION,
    Stage.QUALITY_REVIEW,
    Stage.DELIVERY,
)


def get_next_stage(stage: Stage) -> Stage | None:
    """Returns the immediately subsequent stage in the workflow pipeline."""
    try:
        idx = STAGE_ORDER.index(stage)
        if idx + 1 < len(STAGE_ORDER):
            return STAGE_ORDER[idx + 1]
        return None
    except ValueError:
        return None


class JobStatus(str, Enum):
    """Lifecycle status of a persistent stage WorkflowJob."""
    QUEUED = "QUEUED"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    RETRYABLE = "RETRYABLE"
    WAITING_USER = "WAITING_USER"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class WorkflowPolicyType(str, Enum):
    """Execution policy controlling automatic continuation vs user approval pauses."""
    AUTO = "AUTO"
    REVIEW = "REVIEW"


class JobErrorType(str, Enum):
    """Categorized error types influencing recovery and retry semantics."""
    RETRYABLE = "RETRYABLE"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    NEEDS_USER = "NEEDS_USER"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    FATAL = "FATAL"


class ArtifactType(str, Enum):
    """Explicit artifact reference types linking KnowledgeVideoTask to domain entities."""
    EVIDENCE_SNAPSHOT = "EVIDENCE_SNAPSHOT"
    CONTENT_PLAN_REVISION = "CONTENT_PLAN_REVISION"
    SCRIPT_REVISION = "SCRIPT_REVISION"
    STORYBOARD_SNAPSHOT = "STORYBOARD_SNAPSHOT"
    ASSET_ROUTE_PLAN = "ASSET_ROUTE_PLAN"
    EXECUTION_RUN = "EXECUTION_RUN"
    AUDIO_OUTPUT = "AUDIO_OUTPUT"
    COMPOSITION_OUTPUT = "COMPOSITION_OUTPUT"
    EVALUATION_SNAPSHOT = "EVALUATION_SNAPSHOT"
    DELIVERY_MANIFEST = "DELIVERY_MANIFEST"


# =============================================================================
# Domain Exceptions
# =============================================================================


class WorkflowDomainError(Exception):
    """Base exception for workflow domain rules."""


class InvalidStateTransitionError(WorkflowDomainError):
    """Raised when an illegal task status transition is attempted."""


class TerminalStateImmutableError(InvalidStateTransitionError):
    """Raised when an update is attempted on a completed, failed, or cancelled task."""


class InvalidStageTransitionError(WorkflowDomainError):
    """Raised when an illegal stage progression is attempted."""


class WorkflowConflictError(WorkflowDomainError):
    """Raised when a concurrent modification or idempotent state conflict occurs."""


class StageOrderViolationError(InvalidStageTransitionError):
    """Raised when stage progression violates sequential ordering rules."""
