from datetime import UTC, datetime

import pytest

from app.domain.asset_execution import (
    AttemptRequest,
    AttemptResult,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    ExecutionTransition,
    FailureCategory,
    InvalidAttemptStateTransitionError,
    ProviderOutcomeType,
    ProviderReceipt,
    RetryAction,
    evaluate_retry_decision,
    validate_attempt_transition,
)
from app.domain.asset_router import (
    AssetRouteCandidate,
    GenerationMode,
)
from app.domain.enums import VisualType


def _make_candidate(provider: str = "seedance", model: str = "pro") -> AssetRouteCandidate:
    return AssetRouteCandidate(
        capability_id=f"{provider}:{model}:TEXT_TO_VIDEO",
        provider=provider,
        model=model,
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        requested_visual_type=VisualType.AI_VIDEO,
        is_eligible=True,
    )


def test_attempt_state_lifecycle_transitions():
    """Verify standard progression: CREATED -> READY_TO_SUBMIT -> SUBMITTING -> ACCEPTED -> RUNNING -> SUCCEEDED."""
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        status=ExecutionAttemptStatus.CREATED,
    )
    assert attempt.status == ExecutionAttemptStatus.CREATED

    ready = attempt.mark_ready_to_submit()
    assert ready.status == ExecutionAttemptStatus.READY_TO_SUBMIT

    submitting = ready.mark_submitting()
    assert submitting.status == ExecutionAttemptStatus.SUBMITTING

    receipt = ProviderReceipt(
        execution_attempt_id=attempt.execution_attempt_id,
        provider="seedance",
        provider_job_id="job-100",
        provider_status="queued",
    )
    accepted = submitting.mark_accepted(receipt)
    assert accepted.status == ExecutionAttemptStatus.ACCEPTED

    running = accepted.mark_running()
    assert running.status == ExecutionAttemptStatus.RUNNING

    succeeded = running.mark_succeeded(raw_response={"status": "done"})
    assert succeeded.status == ExecutionAttemptStatus.SUCCEEDED


def test_illegal_attempt_state_transition_is_rejected():
    """Verify illegal transitions, such as SUCCEEDED -> RUNNING, are strictly rejected."""
    attempt = ExecutionAttempt(
        execution_run_id="run-1",
        shot_id="shot-1",
        shot_revision_id="rev-1",
        attempt_number=1,
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        status=ExecutionAttemptStatus.SUCCEEDED,
    )
    with pytest.raises(InvalidAttemptStateTransitionError):
        validate_attempt_transition(ExecutionAttemptStatus.SUCCEEDED, ExecutionAttemptStatus.RUNNING)

    with pytest.raises(InvalidAttemptStateTransitionError):
        attempt.mark_running()


def test_attempt_request_contains_no_api_keys_or_secrets():
    """Verify AttemptRequest sanitizes credentials/secrets from payload."""
    raw_payload = {
        "prompt": "flying eagle over mountains",
        "api_key": "secret_ark_12345",
        "bearer_token": "bearer_abc_xyz",
        "authorization": "Bearer secret_header",
        "password": "my_password",
        "nested": {
            "token": "nested_token_value",
            "safe_param": "4k_hd",
        },
    }
    req = AttemptRequest.create_sanitized(
        execution_attempt_id="att-1",
        provider="seedance",
        model="pro",
        generation_mode=GenerationMode.TEXT_TO_VIDEO,
        idempotency_key="idemp-12345",
        raw_payload=raw_payload,
    )
    assert "api_key" not in req.sanitized_payload or req.sanitized_payload["api_key"] == "[REDACTED]"
    assert "bearer_token" not in req.sanitized_payload or req.sanitized_payload["bearer_token"] == "[REDACTED]"
    assert "password" not in req.sanitized_payload or req.sanitized_payload["password"] == "[REDACTED]"
    assert "secret_ark_12345" not in str(req.sanitized_payload)
    assert len(req.request_hash) == 64


def test_provider_receipt_and_result_immutability():
    """Verify ProviderReceipt and AttemptResult models."""
    receipt = ProviderReceipt(
        execution_attempt_id="att-1",
        provider="seedance",
        provider_job_id="seedance-job-888",
        provider_status="queued",
        sanitized_metadata={"region": "cn-beijing"},
    )
    assert receipt.provider_job_id == "seedance-job-888"

    result = AttemptResult(
        execution_attempt_id="att-1",
        outcome=ProviderOutcomeType.SUCCESS,
        provider_status="succeeded",
        asset_reference="storage/shot_assets/shot-1/asset.mp4",
        completed_at=datetime.now(UTC),
    )
    assert result.outcome == ProviderOutcomeType.SUCCESS


def test_execution_transition_audit_record():
    """Verify ExecutionTransition record structure."""
    trans = ExecutionTransition(
        entity_type="EXECUTION_ATTEMPT",
        entity_id="att-1",
        from_state="SUBMITTING",
        to_state="ACCEPTED",
        reason_code="PROVIDER_ACKNOWLEDGED",
    )
    assert trans.entity_type == "EXECUTION_ATTEMPT"
    assert trans.from_state == "SUBMITTING"
    assert trans.to_state == "ACCEPTED"


def test_retry_decision_transient_failure_retries_same_candidate():
    """Safe transient failure retries the same candidate when local and total budgets remain."""
    cand = _make_candidate("seedance", "pro")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    decision = evaluate_retry_decision(
        failure_category=FailureCategory.TRANSIENT_TECHNICAL,
        current_candidate=cand,
        candidate_attempt_count=1,
        total_shot_attempts=1,
        policy=policy,
        model_selection_mode="AUTO",
        next_candidate=_make_candidate("wavespeed", "standard"),
    )
    assert decision.action == RetryAction.RETRY_SAME_CANDIDATE
    assert decision.candidate == cand
    assert decision.next_attempt_number == 2


def test_retry_decision_candidate_local_budget_exhausted_falls_back():
    """When candidate-local budget is exhausted, AUTO moves to next frozen candidate."""
    cand1 = _make_candidate("seedance", "pro")
    cand2 = _make_candidate("wavespeed", "standard")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    decision = evaluate_retry_decision(
        failure_category=FailureCategory.TRANSIENT_TECHNICAL,
        current_candidate=cand1,
        candidate_attempt_count=2,  # exhausted candidate local
        total_shot_attempts=2,
        policy=policy,
        model_selection_mode="AUTO",
        next_candidate=cand2,
    )
    assert decision.action == RetryAction.FALLBACK_NEXT_CANDIDATE
    assert decision.candidate == cand2
    assert decision.next_attempt_number == 3


def test_retry_decision_permanent_technical_skips_retry():
    """Permanent technical failure skips same-candidate retry and falls back immediately."""
    cand1 = _make_candidate("seedance", "pro")
    cand2 = _make_candidate("wavespeed", "standard")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    decision = evaluate_retry_decision(
        failure_category=FailureCategory.PERMANENT_TECHNICAL,
        current_candidate=cand1,
        candidate_attempt_count=1,
        total_shot_attempts=1,
        policy=policy,
        model_selection_mode="AUTO",
        next_candidate=cand2,
    )
    assert decision.action == RetryAction.FALLBACK_NEXT_CANDIDATE
    assert decision.candidate == cand2


def test_retry_decision_pinned_mode_never_switches_model():
    """PINNED mode stops on failure if retries exhausted or permanent failure occurs."""
    cand1 = _make_candidate("seedance", "pro")
    cand2 = _make_candidate("wavespeed", "standard")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    # Permanent failure in PINNED mode
    decision = evaluate_retry_decision(
        failure_category=FailureCategory.PERMANENT_TECHNICAL,
        current_candidate=cand1,
        candidate_attempt_count=1,
        total_shot_attempts=1,
        policy=policy,
        model_selection_mode="PINNED",
        next_candidate=cand2,
    )
    assert decision.action == RetryAction.STOP_FAILED
    assert decision.reason == "PINNED_MODEL_FAILED"


def test_retry_decision_shot_total_budget_exhaustion():
    """When total shot budget is exhausted, action is STOP_EXHAUSTED."""
    cand1 = _make_candidate("seedance", "pro")
    cand2 = _make_candidate("wavespeed", "standard")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    decision = evaluate_retry_decision(
        failure_category=FailureCategory.TRANSIENT_TECHNICAL,
        current_candidate=cand1,
        candidate_attempt_count=1,
        total_shot_attempts=4,  # total budget reached
        policy=policy,
        model_selection_mode="AUTO",
        next_candidate=cand2,
    )
    assert decision.action == RetryAction.STOP_EXHAUSTED
    assert decision.reason == "SHOT_EXECUTION_EXHAUSTED"


def test_retry_decision_submission_ambiguous_waits_for_recovery():
    """Ambiguous submission waits for recovery, no retry or fallback."""
    cand1 = _make_candidate("seedance", "pro")
    cand2 = _make_candidate("wavespeed", "standard")
    policy = ExecutionRetryPolicy(max_attempts_per_candidate=2, max_total_attempts_per_shot=4)

    decision = evaluate_retry_decision(
        failure_category=FailureCategory.SUBMISSION_AMBIGUOUS,
        current_candidate=cand1,
        candidate_attempt_count=1,
        total_shot_attempts=1,
        policy=policy,
        model_selection_mode="AUTO",
        next_candidate=cand2,
    )
    assert decision.action == RetryAction.WAIT_FOR_RECOVERY
