from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.domain.asset_router import (
    AssetRouteCandidate,
    AssetRouteDecision,
    GenerationMode,
)


class ExecutionStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    # Backwards-compatibility aliases for Phase 5.1
    PENDING = "CREATED"  # noqa: PIE796
    SUCCEEDED = "COMPLETED"  # noqa: PIE796
    PARTIAL = "FAILED"  # noqa: PIE796


class ExecutionAttemptStatus(str, Enum):
    CREATED = "CREATED"
    READY_TO_SUBMIT = "READY_TO_SUBMIT"
    SUBMITTING = "SUBMITTING"
    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUBMISSION_OUTCOME_UNKNOWN = "SUBMISSION_OUTCOME_UNKNOWN"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    # Backwards-compatibility alias
    PENDING = "CREATED"  # noqa: PIE796


class ShotExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    EXHAUSTED = "EXHAUSTED"
    REUSED = "REUSED"
    # Backwards-compatibility alias for Phase 5.1
    SUBMISSION_OUTCOME_UNKNOWN = "NEEDS_RECOVERY"  # noqa: PIE796


class ProviderOutcomeType(str, Enum):
    SUCCESS = "SUCCESS"
    DEFINITIVE_TECHNICAL_FAILURE = "DEFINITIVE_TECHNICAL_FAILURE"
    CAPABILITY_INCOMPATIBLE = "CAPABILITY_INCOMPATIBLE"
    SUBMISSION_OUTCOME_UNKNOWN = "SUBMISSION_OUTCOME_UNKNOWN"


class FailureCategory(str, Enum):
    TRANSIENT_TECHNICAL = "TRANSIENT_TECHNICAL"
    PERMANENT_TECHNICAL = "PERMANENT_TECHNICAL"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    SUBMISSION_AMBIGUOUS = "SUBMISSION_AMBIGUOUS"


class RetryAction(str, Enum):
    RETRY_SAME_CANDIDATE = "RETRY_SAME_CANDIDATE"
    FALLBACK_NEXT_CANDIDATE = "FALLBACK_NEXT_CANDIDATE"
    WAIT_FOR_RECOVERY = "WAIT_FOR_RECOVERY"
    STOP_EXHAUSTED = "STOP_EXHAUSTED"
    STOP_FAILED = "STOP_FAILED"


class RecoveryAction(str, Enum):
    RESOLVED_SUCCEEDED = "RESOLVED_SUCCEEDED"
    RESOLVED_FAILED = "RESOLVED_FAILED"
    STILL_UNKNOWN = "STILL_UNKNOWN"
    SAFE_TO_RESUBMIT_SAME_ATTEMPT = "SAFE_TO_RESUBMIT_SAME_ATTEMPT"


class RecoveryDecision(BaseModel):
    execution_attempt_id: str
    action: RecoveryAction
    evidence: dict[str, Any] = Field(default_factory=dict)
    resolved_state: ExecutionAttemptStatus | None = None
    reason_code: str


class AssetMediaType(str, Enum):
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"


class InvalidAttemptStateTransitionError(ValueError):
    """Raised when an attempt transitions to an illegal state."""

    def __init__(self, from_state: str, to_state: str):
        super().__init__(f"Cannot transition attempt from {from_state} to {to_state}")
        self.from_state = from_state
        self.to_state = to_state


class BlindSubmissionForbiddenError(RuntimeError):
    """Raised when re-submission is attempted on a shot with SUBMISSION_OUTCOME_UNKNOWN / NEEDS_RECOVERY."""

    def __init__(self, message: str = "Blind re-submission is strictly forbidden when submission outcome is unknown"):
        super().__init__(message)


class RoutePlanNotReadyError(ValueError):
    """Raised when attempting to execute an AssetRoutePlan that is not in READY status."""

    def __init__(self, message: str = "AssetRoutePlan is not in READY status"):
        super().__init__(message)


ALLOWED_ATTEMPT_TRANSITIONS: dict[ExecutionAttemptStatus, set[ExecutionAttemptStatus]] = {
    ExecutionAttemptStatus.CREATED: {
        ExecutionAttemptStatus.READY_TO_SUBMIT,
        ExecutionAttemptStatus.SUBMITTING,
        ExecutionAttemptStatus.RUNNING,
        ExecutionAttemptStatus.CANCELLED,
    },
    ExecutionAttemptStatus.READY_TO_SUBMIT: {
        ExecutionAttemptStatus.SUBMITTING,
        ExecutionAttemptStatus.CANCELLED,
    },
    ExecutionAttemptStatus.SUBMITTING: {
        ExecutionAttemptStatus.ACCEPTED,
        ExecutionAttemptStatus.RUNNING,
        ExecutionAttemptStatus.SUCCEEDED,
        ExecutionAttemptStatus.FAILED,
        ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
        ExecutionAttemptStatus.CANCEL_PENDING,
    },
    ExecutionAttemptStatus.ACCEPTED: {
        ExecutionAttemptStatus.RUNNING,
        ExecutionAttemptStatus.SUCCEEDED,
        ExecutionAttemptStatus.FAILED,
        ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
        ExecutionAttemptStatus.CANCEL_PENDING,
    },
    ExecutionAttemptStatus.RUNNING: {
        ExecutionAttemptStatus.SUCCEEDED,
        ExecutionAttemptStatus.FAILED,
        ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
        ExecutionAttemptStatus.CANCEL_PENDING,
    },
    ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN: {
        ExecutionAttemptStatus.SUCCEEDED,
        ExecutionAttemptStatus.FAILED,
    },
    ExecutionAttemptStatus.CANCEL_PENDING: {
        ExecutionAttemptStatus.CANCELLED,
        ExecutionAttemptStatus.FAILED,
        ExecutionAttemptStatus.SUCCEEDED,
    },
    ExecutionAttemptStatus.SUCCEEDED: set(),
    ExecutionAttemptStatus.FAILED: set(),
    ExecutionAttemptStatus.CANCELLED: set(),
}


ALLOWED_SHOT_TRANSITIONS: dict[ShotExecutionStatus, set[ShotExecutionStatus]] = {
    ShotExecutionStatus.PENDING: {
        ShotExecutionStatus.RUNNING,
        ShotExecutionStatus.FAILED,
        ShotExecutionStatus.NEEDS_RECOVERY,
        ShotExecutionStatus.REUSED,
    },
    ShotExecutionStatus.RUNNING: {
        ShotExecutionStatus.SUCCEEDED,
        ShotExecutionStatus.FAILED,
        ShotExecutionStatus.NEEDS_RECOVERY,
        ShotExecutionStatus.EXHAUSTED,
    },
    ShotExecutionStatus.NEEDS_RECOVERY: {
        ShotExecutionStatus.RUNNING,
        ShotExecutionStatus.SUCCEEDED,
        ShotExecutionStatus.FAILED,
        ShotExecutionStatus.EXHAUSTED,
    },
    ShotExecutionStatus.SUCCEEDED: set(),
    ShotExecutionStatus.FAILED: set(),
    ShotExecutionStatus.EXHAUSTED: set(),
    ShotExecutionStatus.REUSED: set(),
}


def validate_attempt_transition(from_state: ExecutionAttemptStatus, to_state: ExecutionAttemptStatus) -> None:
    allowed = ALLOWED_ATTEMPT_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidAttemptStateTransitionError(from_state.value, to_state.value)


def validate_shot_transition(from_state: ShotExecutionStatus, to_state: ShotExecutionStatus) -> None:
    allowed = ALLOWED_SHOT_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise ValueError(f"Cannot transition shot execution from {from_state.value} to {to_state.value}")


class ExecutionRetryPolicy(BaseModel):
    policy_version: str = "v1"
    max_attempts_per_candidate: int = 2
    max_total_attempts_per_shot: int = 4


class RetryDecision(BaseModel):
    action: RetryAction
    candidate: AssetRouteCandidate | None = None
    reason: str
    next_attempt_number: int
    remaining_budget: int


def evaluate_retry_decision(
    failure_category: FailureCategory,
    current_candidate: AssetRouteCandidate,
    candidate_attempt_count: int,
    total_shot_attempts: int,
    policy: ExecutionRetryPolicy,
    model_selection_mode: str = "AUTO",
    next_candidate: AssetRouteCandidate | None = None,
) -> RetryDecision:
    remaining_budget = max(0, policy.max_total_attempts_per_shot - total_shot_attempts)

    # 1. Ambiguous submission must NEVER retry or fallback automatically
    if failure_category == FailureCategory.SUBMISSION_AMBIGUOUS:
        return RetryDecision(
            action=RetryAction.WAIT_FOR_RECOVERY,
            candidate=current_candidate,
            reason="SUBMISSION_OUTCOME_UNKNOWN",
            next_attempt_number=total_shot_attempts,
            remaining_budget=remaining_budget,
        )

    # 2. Check total budget
    if remaining_budget <= 0:
        return RetryDecision(
            action=RetryAction.STOP_EXHAUSTED,
            candidate=None,
            reason="SHOT_EXECUTION_EXHAUSTED",
            next_attempt_number=total_shot_attempts + 1,
            remaining_budget=0,
        )

    is_pinned = model_selection_mode.upper() == "PINNED"

    # 3. Safe transient technical failure
    if failure_category == FailureCategory.TRANSIENT_TECHNICAL:
        if (
            candidate_attempt_count < policy.max_attempts_per_candidate
            and total_shot_attempts < policy.max_total_attempts_per_shot
        ):
            return RetryDecision(
                action=RetryAction.RETRY_SAME_CANDIDATE,
                candidate=current_candidate,
                reason="RETRY_TRANSIENT_FAILURE",
                next_attempt_number=total_shot_attempts + 1,
                remaining_budget=remaining_budget,
            )
        # Candidate local budget exhausted
        if is_pinned:
            return RetryDecision(
                action=RetryAction.STOP_FAILED,
                candidate=None,
                reason="PINNED_MODEL_FAILED",
                next_attempt_number=total_shot_attempts + 1,
                remaining_budget=remaining_budget,
            )
        if next_candidate is not None and total_shot_attempts < policy.max_total_attempts_per_shot:
            return RetryDecision(
                action=RetryAction.FALLBACK_NEXT_CANDIDATE,
                candidate=next_candidate,
                reason="CANDIDATE_RETRY_EXHAUSTED_FALLBACK",
                next_attempt_number=total_shot_attempts + 1,
                remaining_budget=remaining_budget,
            )
        return RetryDecision(
            action=RetryAction.STOP_EXHAUSTED,
            candidate=None,
            reason="SHOT_EXECUTION_EXHAUSTED",
            next_attempt_number=total_shot_attempts + 1,
            remaining_budget=remaining_budget,
        )

    # 4. Permanent technical failure or unsupported capability
    if failure_category in (FailureCategory.PERMANENT_TECHNICAL, FailureCategory.UNSUPPORTED_CAPABILITY):
        if is_pinned:
            return RetryDecision(
                action=RetryAction.STOP_FAILED,
                candidate=None,
                reason="PINNED_MODEL_FAILED",
                next_attempt_number=total_shot_attempts + 1,
                remaining_budget=remaining_budget,
            )
        if next_candidate is not None and total_shot_attempts < policy.max_total_attempts_per_shot:
            reason = (
                "PERMANENT_FAILURE_FALLBACK"
                if failure_category == FailureCategory.PERMANENT_TECHNICAL
                else "UNSUPPORTED_CAPABILITY_FALLBACK"
            )
            return RetryDecision(
                action=RetryAction.FALLBACK_NEXT_CANDIDATE,
                candidate=next_candidate,
                reason=reason,
                next_attempt_number=total_shot_attempts + 1,
                remaining_budget=remaining_budget,
            )
        return RetryDecision(
            action=RetryAction.STOP_FAILED,
            candidate=None,
            reason="NO_FALLBACK_CANDIDATE",
            next_attempt_number=total_shot_attempts + 1,
            remaining_budget=remaining_budget,
        )

    return RetryDecision(
        action=RetryAction.STOP_FAILED,
        candidate=None,
        reason="UNKNOWN_FAILURE",
        next_attempt_number=total_shot_attempts + 1,
        remaining_budget=remaining_budget,
    )


_SECRET_KEY_PATTERN = re.compile(r"(key|secret|token|auth|password|credential)", re.IGNORECASE)


def _sanitize_dict(data: Any) -> Any:
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if _SECRET_KEY_PATTERN.search(str(k)):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = _sanitize_dict(v)
        return sanitized
    if isinstance(data, list):
        return [_sanitize_dict(item) for item in data]
    return data


class AttemptRequest(BaseModel):
    """
    Immutable persisted snapshot of what was intended to be submitted to the external provider.
    Strictly forbids persisting raw credentials/secrets.
    """
    execution_attempt_id: str
    provider: str
    model: str
    generation_mode: GenerationMode
    idempotency_key: str
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)
    request_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create_sanitized(
        cls,
        execution_attempt_id: str,
        provider: str,
        model: str,
        generation_mode: GenerationMode,
        idempotency_key: str,
        raw_payload: dict[str, Any],
    ) -> AttemptRequest:
        sanitized = _sanitize_dict(raw_payload)
        payload_bytes = json.dumps(sanitized, sort_keys=True).encode("utf-8")
        req_hash = hashlib.sha256(payload_bytes).hexdigest()
        return cls(
            execution_attempt_id=execution_attempt_id,
            provider=provider,
            model=model,
            generation_mode=generation_mode,
            idempotency_key=idempotency_key,
            sanitized_payload=sanitized,
            request_hash=req_hash,
        )


class ProviderReceipt(BaseModel):
    """
    Immutable acknowledgment that provider accepted the task request.
    Does NOT indicate generation success.
    """
    execution_attempt_id: str
    provider: str
    provider_job_id: str | None = None
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider_status: str
    sanitized_metadata: dict[str, Any] = Field(default_factory=dict)


class AttemptResult(BaseModel):
    """
    Immutable final outcome record of an execution attempt.
    """
    execution_attempt_id: str
    outcome: ProviderOutcomeType
    provider_status: str | None = None
    asset_reference: str | None = None
    error_category: FailureCategory | None = None
    error_code: str | None = None
    diagnostic_summary: str | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionTransition(BaseModel):
    """
    Append-only transition audit event for ShotExecution or ExecutionAttempt.
    """
    transition_id: str = Field(default_factory=lambda: str(uuid4()))
    entity_type: str  # "EXECUTION_ATTEMPT" | "SHOT_EXECUTION"
    entity_id: str
    from_state: str
    to_state: str
    reason_code: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionRun(BaseModel):
    """
    Top-level execution run coordinating the asset production for an AssetRoutePlan.
    """
    execution_run_id: str = Field(default_factory=lambda: str(uuid4()))
    asset_route_plan_id: str
    storyboard_snapshot_id: str
    status: ExecutionStatus = ExecutionStatus.CREATED
    total_shots: int = Field(ge=1)
    succeeded_shots: int = Field(default=0, ge=0)
    reused_shots: int = Field(default=0, ge=0)
    failed_shots: int = Field(default=0, ge=0)
    recovery_required_shots: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def calculate_aggregated_status(self) -> ExecutionStatus:
        """Determines the aggregate status based on current shot completion counts."""
        if self.recovery_required_shots > 0:
            return ExecutionStatus.NEEDS_RECOVERY
        completed = self.succeeded_shots + self.reused_shots + self.failed_shots
        if completed < self.total_shots:
            return ExecutionStatus.RUNNING if completed > 0 else self.status
        if self.succeeded_shots + self.reused_shots == self.total_shots:
            return ExecutionStatus.COMPLETED
        if self.failed_shots == self.total_shots:
            return ExecutionStatus.FAILED
        return ExecutionStatus.PARTIAL


class ExecutionAttempt(BaseModel):
    """
    An immutable audit record of a single execution invocation against a provider.
    """
    execution_attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_run_id: str
    shot_id: str
    shot_revision_id: str
    attempt_number: int = Field(ge=1)
    provider: str
    model: str
    generation_mode: GenerationMode
    status: ExecutionAttemptStatus = ExecutionAttemptStatus.CREATED
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error_code: str | None = None
    error_message: str | None = None
    raw_provider_response: dict[str, Any] | None = None
    provider_receipt: ProviderReceipt | None = None
    attempt_result: AttemptResult | None = None

    def mark_ready_to_submit(self) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.READY_TO_SUBMIT)
        return self.model_copy(update={"status": ExecutionAttemptStatus.READY_TO_SUBMIT})

    def mark_submitting(self) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.SUBMITTING)
        return self.model_copy(update={"status": ExecutionAttemptStatus.SUBMITTING})

    def mark_accepted(self, receipt: ProviderReceipt) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.ACCEPTED)
        return self.model_copy(
            update={
                "status": ExecutionAttemptStatus.ACCEPTED,
                "provider_receipt": receipt,
            }
        )

    def mark_running(self) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.RUNNING)
        return self.model_copy(update={"status": ExecutionAttemptStatus.RUNNING})

    def _calculate_duration(self, end_time: datetime) -> float:
        s_at = self.started_at
        if s_at is None:
            return 0.0
        if end_time.tzinfo is not None and s_at.tzinfo is None:
            s_at = s_at.replace(tzinfo=UTC)
        elif end_time.tzinfo is None and s_at.tzinfo is not None:
            end_time = end_time.replace(tzinfo=UTC)
        return max(0.0, (end_time - s_at).total_seconds())

    def mark_succeeded(
        self,
        finished_at: datetime | None = None,
        duration_seconds: float | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.SUCCEEDED)
        end_time = finished_at or datetime.now(UTC)
        dur = duration_seconds if duration_seconds is not None else self._calculate_duration(end_time)
        return self.model_copy(
            update={
                "status": ExecutionAttemptStatus.SUCCEEDED,
                "finished_at": end_time,
                "duration_seconds": dur,
                "raw_provider_response": raw_response or self.raw_provider_response,
            }
        )

    def mark_failed(
        self,
        error_code: str,
        error_message: str,
        finished_at: datetime | None = None,
        duration_seconds: float | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.FAILED)
        end_time = finished_at or datetime.now(UTC)
        dur = duration_seconds if duration_seconds is not None else self._calculate_duration(end_time)
        return self.model_copy(
            update={
                "status": ExecutionAttemptStatus.FAILED,
                "finished_at": end_time,
                "duration_seconds": dur,
                "error_code": error_code,
                "error_message": error_message,
                "raw_provider_response": raw_response or self.raw_provider_response,
            }
        )

    def mark_submission_unknown(
        self,
        error_code: str = "SUBMISSION_OUTCOME_UNKNOWN",
        error_message: str = "Task submission outcome is unknown",
        raw_response: dict[str, Any] | None = None,
    ) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN)
        return self.model_copy(
            update={
                "status": ExecutionAttemptStatus.SUBMISSION_OUTCOME_UNKNOWN,
                "error_code": error_code,
                "error_message": error_message,
                "raw_provider_response": raw_response or self.raw_provider_response,
            }
        )

    def mark_cancel_pending(self) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.CANCEL_PENDING)
        return self.model_copy(update={"status": ExecutionAttemptStatus.CANCEL_PENDING})

    def mark_cancelled(self) -> ExecutionAttempt:
        validate_attempt_transition(self.status, ExecutionAttemptStatus.CANCELLED)
        return self.model_copy(
            update={
                "status": ExecutionAttemptStatus.CANCELLED,
                "finished_at": datetime.now(UTC),
            }
        )


_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ShotAssetVersion(BaseModel):
    """
    An immutable media asset version bound to a specific ShotRevision.
    Produced by a successful ExecutionAttempt and verified on disk.
    """
    shot_asset_version_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_id: str
    shot_revision_id: str
    execution_attempt_id: str
    file_path: str
    file_hash: str
    file_size_bytes: int = Field(gt=0)
    media_type: AssetMediaType
    mime_type: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(ge=0.0)
    fps: float | None = Field(default=None, ge=0.0)
    provider: str
    model: str
    generation_mode: GenerationMode
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("file_hash")
    @classmethod
    def validate_file_hash(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if not _SHA256_HEX_PATTERN.match(v_lower):
            raise ValueError(f"file_hash must be a 64-character lowercase hexadecimal SHA-256 string, got: {v!r}")
        return v_lower

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("file_path must not be empty")
        return v.strip()


class AdapterExecutionResult(BaseModel):
    """Normalized outcome returned by an AssetExecutionAdapter."""
    outcome_type: ProviderOutcomeType
    status: str | None = None
    file_path: str | None = None
    remote_task_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


class ShotExecution(BaseModel):
    """
    Manages the execution lifecycle and attempt history for a single ShotRoutePlanEntry.
    """
    shot_execution_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_run_id: str
    shot_id: str
    shot_revision_id: str
    route_decision: AssetRouteDecision
    status: ShotExecutionStatus = ShotExecutionStatus.PENDING
    current_candidate_index: int = 0
    attempt_ids: list[str] = Field(default_factory=list)
    produced_asset_version_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    unconfirmed_remote_task_id: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def mark_running(self) -> ShotExecution:
        if self.status == ShotExecutionStatus.NEEDS_RECOVERY:
            raise BlindSubmissionForbiddenError(
                f"Cannot run shot {self.shot_id} because previous submission outcome is unknown / needs recovery"
            )
        validate_shot_transition(self.status, ShotExecutionStatus.RUNNING)
        return self.model_copy(update={"status": ShotExecutionStatus.RUNNING})

    def record_attempt(self, attempt_id: str) -> ShotExecution:
        return self.model_copy(update={"attempt_ids": [*self.attempt_ids, attempt_id]})

    def advance_candidate(self) -> ShotExecution:
        return self.model_copy(update={"current_candidate_index": self.current_candidate_index + 1})

    def mark_succeeded(self, version_id: str, attempt_id: str | None = None) -> ShotExecution:
        validate_shot_transition(self.status, ShotExecutionStatus.SUCCEEDED)
        new_attempts = [*self.attempt_ids, attempt_id] if attempt_id and attempt_id not in self.attempt_ids else self.attempt_ids
        return self.model_copy(
            update={
                "status": ShotExecutionStatus.SUCCEEDED,
                "produced_asset_version_id": version_id,
                "attempt_ids": new_attempts,
                "finished_at": datetime.now(UTC),
            }
        )

    def mark_failed(self, error_code: str, error_message: str, attempt_id: str | None = None) -> ShotExecution:
        validate_shot_transition(self.status, ShotExecutionStatus.FAILED)
        new_attempts = [*self.attempt_ids, attempt_id] if attempt_id and attempt_id not in self.attempt_ids else self.attempt_ids
        return self.model_copy(
            update={
                "status": ShotExecutionStatus.FAILED,
                "error_code": error_code,
                "error_message": error_message,
                "attempt_ids": new_attempts,
                "finished_at": datetime.now(UTC),
            }
        )

    def mark_submission_unknown(
        self,
        error_code: str,
        error_message: str,
        remote_task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> ShotExecution:
        validate_shot_transition(self.status, ShotExecutionStatus.NEEDS_RECOVERY)
        new_attempts = [*self.attempt_ids, attempt_id] if attempt_id and attempt_id not in self.attempt_ids else self.attempt_ids
        return self.model_copy(
            update={
                "status": ShotExecutionStatus.NEEDS_RECOVERY,
                "error_code": error_code,
                "error_message": error_message,
                "unconfirmed_remote_task_id": remote_task_id,
                "attempt_ids": new_attempts,
                "finished_at": datetime.now(UTC),
            }
        )

    def mark_exhausted(
        self,
        error_code: str = "SHOT_EXECUTION_EXHAUSTED",
        error_message: str = "Total retry and fallback budget exhausted",
        attempt_id: str | None = None,
    ) -> ShotExecution:
        validate_shot_transition(self.status, ShotExecutionStatus.EXHAUSTED)
        new_attempts = [*self.attempt_ids, attempt_id] if attempt_id and attempt_id not in self.attempt_ids else self.attempt_ids
        return self.model_copy(
            update={
                "status": ShotExecutionStatus.EXHAUSTED,
                "error_code": error_code,
                "error_message": error_message,
                "attempt_ids": new_attempts,
                "finished_at": datetime.now(UTC),
            }
        )

    def mark_reused(self, version_id: str) -> ShotExecution:
        validate_shot_transition(self.status, ShotExecutionStatus.REUSED)
        return self.model_copy(
            update={
                "status": ShotExecutionStatus.REUSED,
                "produced_asset_version_id": version_id,
                "finished_at": datetime.now(UTC),
            }
        )


class AssetReuseMode(str, Enum):
    REUSE_COMPATIBLE = "REUSE_COMPATIBLE"
    FORCE_REGENERATE = "FORCE_REGENERATE"


class AssetReuseDecisionType(str, Enum):
    REUSE = "REUSE"
    EXECUTE = "EXECUTE"


class AssetReuseReasonCode(str, Enum):
    EXACT_COMPATIBLE_ASSET = "EXACT_COMPATIBLE_ASSET"
    NO_PREVIOUS_ASSET = "NO_PREVIOUS_ASSET"
    SHOT_REVISION_CHANGED = "SHOT_REVISION_CHANGED"
    VISUAL_TYPE_CHANGED = "VISUAL_TYPE_CHANGED"
    GENERATION_MODE_CHANGED = "GENERATION_MODE_CHANGED"
    ASPECT_RATIO_INCOMPATIBLE = "ASPECT_RATIO_INCOMPATIBLE"
    DURATION_INCOMPATIBLE = "DURATION_INCOMPATIBLE"
    ASSET_MISSING = "ASSET_MISSING"
    ASSET_INTEGRITY_FAILED = "ASSET_INTEGRITY_FAILED"
    FORCE_REGENERATE = "FORCE_REGENERATE"
    UNRESOLVED_OUTCOME_UNSAFE = "UNRESOLVED_OUTCOME_UNSAFE"


class AssetReuseDecision(BaseModel):
    """
    Typed immutable decision recording whether a shot asset can be technically reused.
    Does NOT imply semantic quality pass (Phase 6 boundary).
    """
    shot_id: str
    shot_revision_id: str
    candidate_asset_version_id: str | None = None
    decision: AssetReuseDecisionType
    reason_codes: tuple[AssetReuseReasonCode, ...]
    policy_version: str = "v1.0"


class ShotExecutionSummary(BaseModel):
    """
    Summary outcome of a single Shot within a completed or partial ExecutionRun.
    """
    shot_id: str
    shot_revision_id: str
    shot_execution_id: str
    asset_version_id: str | None = None
    result_type: str  # "REUSED", "GENERATED", "FAILED", "NEEDS_RECOVERY"
    status: ShotExecutionStatus
    failure_reason: str | None = None
    attempts_count: int = 0


class ExecutionRunResult(BaseModel):
    """
    Typed aggregated outcome of an entire AssetRoutePlan execution run.
    """
    execution_run_id: str
    asset_route_plan_id: str
    storyboard_snapshot_id: str
    state: ExecutionStatus
    total_shots: int
    generated_shots: int
    reused_shots: int
    failed_shots: int
    recovery_required_shots: int
    shot_results: tuple[ShotExecutionSummary, ...]
    started_at: datetime
    finished_at: datetime

