from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from app.domain.asset_execution import (
    AdapterExecutionResult,
    AttemptRequest,
    BlindSubmissionForbiddenError,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ExecutionRetryPolicy,
    FailureCategory,
    ProviderOutcomeType,
    ProviderReceipt,
    RetryAction,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
    evaluate_retry_decision,
)
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_probe_service import probe_media_file


class ShotExecutionService:
    """
    Orchestrates the execution lifecycle for a single ShotExecution.
    Implements:
    - Pre-submission AttemptRequest persistence with secret redaction and idempotency keys
    - ProviderReceipt creation and async lifecycle tracking
    - Two-level retry budget enforcement (candidate-local limit + shot-total limit)
    - Fallback along frozen Phase 4 candidate ordering (preserving VisualType)
    - Strict submission halt and guard on SUBMISSION_OUTCOME_UNKNOWN (NEEDS_RECOVERY)
    - PINNED mode enforcement (never switches provider/model)
    - Asset probing and ShotAssetVersion creation on SUCCESS
    """

    def __init__(
        self,
        adapter_registry: AdapterRegistry | None = None,
        max_retries_per_candidate: int = 2,
        policy: ExecutionRetryPolicy | None = None,
        session_factory: Any | None = None,
    ):
        self._adapter_registry = adapter_registry or AdapterRegistry()
        self._session_factory = session_factory
        if policy is not None:
            self._policy = policy
        else:
            self._policy = ExecutionRetryPolicy(
                max_attempts_per_candidate=max_retries_per_candidate,
                max_total_attempts_per_shot=max(4, max_retries_per_candidate * 2),
            )

    @property
    def policy(self) -> ExecutionRetryPolicy:
        return self._policy

    def execute_shot(
        self,
        shot_execution: ShotExecution,
        request: AssetRoutingRequest,
        storage_base_dir: Path | str,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
        session_factory: Any | None = None,
    ) -> tuple[ShotExecution, list[ExecutionAttempt], ShotAssetVersion | None]:
        # 1. Guard against blind re-submission if outcome is unknown / needs recovery
        if shot_execution.status in (
            ShotExecutionStatus.NEEDS_RECOVERY,
            ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
        ):
            raise BlindSubmissionForbiddenError(
                f"Shot {shot_execution.shot_id} has {shot_execution.status.value} status. "
                "Blind re-submission is strictly forbidden to prevent duplicate billing."
            )

        storage_path = Path(storage_base_dir).resolve() / "shot_assets" / shot_execution.shot_id
        storage_path.mkdir(parents=True, exist_ok=True)

        # 2. Build candidate list strictly from frozen Phase 4 decision
        # NEVER call HybridAssetRouter.route(...) during execution/retry/fallback!
        candidates: list[AssetRouteCandidate] = []
        if shot_execution.route_decision.selected_candidate:
            candidates.append(shot_execution.route_decision.selected_candidate)

        for elig in shot_execution.route_decision.eligible_candidates:
            if not any(c.capability_id == elig.capability_id for c in candidates):
                candidates.append(elig)

        # Preserve requested_visual_type (NO VisualType fallback)
        expected_visual_type = shot_execution.route_decision.requested_visual_type
        candidates = [c for c in candidates if c.requested_visual_type == expected_visual_type]

        if not candidates:
            failed_exec = shot_execution.mark_failed(
                error_code="NO_CANDIDATES",
                error_message="No eligible candidates available in route decision matching visual type",
            )
            return failed_exec, [], None

        model_mode = getattr(shot_execution.route_decision, "model_selection_mode", "AUTO")
        # In PINNED mode, only candidate 0 (the pinned candidate) is allowed
        if str(model_mode).upper() == "PINNED":
            candidates = candidates[:1]

        executed_attempts: list[ExecutionAttempt] = []
        current_idx = shot_execution.current_candidate_index
        current_exec = shot_execution.mark_running()

        while current_idx < len(candidates):
            candidate = candidates[current_idx]

            # Adapter lookup check with understandable diagnostic error
            if not self._adapter_registry.has_adapter(candidate.provider):
                available_providers = (
                    self._adapter_registry.list_providers()
                    if hasattr(self._adapter_registry, "list_providers")
                    else []
                )
                err_msg = (
                    f"No execution adapter registered for provider '{candidate.provider}'. "
                    f"Available providers: {available_providers}"
                )
                logger.warning(err_msg)
                attempt_num = len(executed_attempts) + 1
                failed_att = ExecutionAttempt(
                    execution_run_id=current_exec.execution_run_id,
                    shot_id=current_exec.shot_id,
                    shot_revision_id=current_exec.shot_revision_id,
                    attempt_number=attempt_num,
                    provider=candidate.provider,
                    model=candidate.model,
                    generation_mode=candidate.generation_mode,
                    status=ExecutionAttemptStatus.FAILED,
                    error_code="ADAPTER_NOT_FOUND",
                    error_message=err_msg,
                )
                executed_attempts.append(failed_att)
                current_exec = current_exec.record_attempt(failed_att.execution_attempt_id)
                next_cand = (
                    candidates[current_idx + 1]
                    if (current_idx + 1) < len(candidates)
                    else None
                )
                retry_dec = evaluate_retry_decision(
                    failure_category=FailureCategory.UNSUPPORTED_CAPABILITY,
                    current_candidate=candidate,
                    candidate_attempt_count=1,
                    total_shot_attempts=len(executed_attempts),
                    policy=self._policy,
                    model_selection_mode=str(model_mode),
                    next_candidate=next_cand,
                )
                if retry_dec.action == RetryAction.FALLBACK_NEXT_CANDIDATE:
                    current_exec = current_exec.advance_candidate()
                    current_idx = current_exec.current_candidate_index
                    continue
                else:
                    final_failed = current_exec.mark_exhausted(
                        error_code="ADAPTER_NOT_FOUND",
                        error_message=err_msg,
                        attempt_id=failed_att.execution_attempt_id,
                    )
                    return final_failed, executed_attempts, None

            adapter = self._adapter_registry.get_adapter(candidate.provider)
            candidate_attempts = 0

            while True:
                candidate_attempts += 1
                attempt_num = len(executed_attempts) + 1
                idempotency_key = f"idemp_{current_exec.shot_id}_{attempt_num}_{uuid4().hex[:8]}"

                attempt = ExecutionAttempt(
                    execution_run_id=current_exec.execution_run_id,
                    shot_id=current_exec.shot_id,
                    shot_revision_id=current_exec.shot_revision_id,
                    attempt_number=attempt_num,
                    provider=candidate.provider,
                    model=candidate.model,
                    generation_mode=candidate.generation_mode,
                    status=ExecutionAttemptStatus.CREATED,
                )

                # Persist AttemptRequest BEFORE external provider submission (with secrets redacted)
                raw_payload = {
                    "prompt": request.generation_prompt or request.visual_goal,
                    "target_duration": request.target_duration,
                    "aspect_ratio": getattr(request, "aspect_ratio", None) or "16:9",
                    "visual_type": request.requested_visual_type.value,
                    "camera_movement": request.camera_movement,
                }
                _attempt_req = AttemptRequest.create_sanitized(
                    execution_attempt_id=attempt.execution_attempt_id,
                    provider=candidate.provider,
                    model=candidate.model,
                    generation_mode=candidate.generation_mode,
                    idempotency_key=idempotency_key,
                    raw_payload=raw_payload,
                )

                # Advance attempt states: CREATED -> READY_TO_SUBMIT -> SUBMITTING
                attempt = attempt.mark_ready_to_submit()
                submitting_attempt = attempt.mark_submitting()
                current_exec = current_exec.record_attempt(submitting_attempt.execution_attempt_id)

                # Persist ExecutionAttempt and AttemptRequest in the SAME short transaction BEFORE submitting
                active_sf = session_factory or self._session_factory
                try:
                    from app.persistence.repositories import ExecutionRepository
                    from app.persistence.session import get_session

                    with get_session(active_sf) as sess:
                        repo = ExecutionRepository(sess)
                        repo.save_or_update_execution_attempt(submitting_attempt)
                        repo.save_attempt_request(_attempt_req)
                except Exception as persist_exc:
                    logger.error(
                        f"Pre-submission persistence failed for shot '{current_exec.shot_id}', "
                        f"attempt '{submitting_attempt.execution_attempt_id}': {persist_exc}. "
                        "Halting execution to prevent unpersisted external provider call."
                    )
                    raise

                attempt_ctx = None
                attempt_start_ev = None
                if trace_writer is not None and trace_context is not None:
                    from app.domain.trace import (
                        CostEstimateStatus,
                        CostUsageTraceData,
                        ExecutionAttemptTraceData,
                        ExecutionFallbackTraceData,
                        TraceEventStatus,
                        TraceEventType,
                    )

                    attempt_ctx = trace_context.child_context(
                        parent_event_id=trace_context.parent_event_id,
                        execution_attempt_id=attempt.execution_attempt_id,
                        shot_id=current_exec.shot_id,
                        shot_revision_id=current_exec.shot_revision_id,
                        execution_run_id=current_exec.execution_run_id,
                    )
                    attempt_start_ev = trace_writer.start_event(
                        context=attempt_ctx,
                        event_type=TraceEventType.EXECUTION_ATTEMPT_STARTED,
                        attributes={
                            "execution_run_id": current_exec.execution_run_id,
                            "shot_id": current_exec.shot_id,
                            "shot_revision_id": current_exec.shot_revision_id,
                            "attempt_number": attempt_num,
                            "provider": candidate.provider,
                            "model": candidate.model,
                            "generation_mode": candidate.generation_mode.value
                            if hasattr(candidate.generation_mode, "value")
                            else str(candidate.generation_mode),
                        },
                    )

                # Invoke provider adapter submit/execute with exception protection
                try:
                    submission_result = adapter.submit(
                        request, candidate, storage_path, idempotency_key=idempotency_key
                    )
                except Exception as sub_exc:  # noqa: BLE001
                    logger.exception(
                        f"Unexpected exception calling adapter.submit for provider {candidate.provider}: {sub_exc}"
                    )
                    submission_result = AdapterExecutionResult(
                        outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                        error_code="UNEXPECTED_SUBMISSION_EXCEPTION",
                        error_message=f"Adapter execution crashed with exception: {sub_exc}",
                    )

                # Handle async ProviderReceipt returned directly
                receipt: ProviderReceipt | None = None
                if isinstance(submission_result, ProviderReceipt):
                    receipt = submission_result
                    running_attempt = submitting_attempt.mark_accepted(receipt).mark_running()
                    # Query initial status for the async job
                    adapter_result = adapter.get_status(receipt.provider_job_id, storage_path)
                else:
                    adapter_result = submission_result
                    if adapter_result.remote_task_id:
                        receipt = ProviderReceipt(
                            execution_attempt_id=submitting_attempt.execution_attempt_id,
                            provider=candidate.provider,
                            provider_job_id=adapter_result.remote_task_id,
                            provider_status=adapter_result.status or "submitted",
                            sanitized_metadata=adapter_result.raw_response or {},
                        )
                        running_attempt = submitting_attempt.mark_accepted(receipt).mark_running()
                    else:
                        running_attempt = submitting_attempt.mark_running()

                # If async job returned and still RUNNING, return running status
                if adapter_result.status == "RUNNING":
                    executed_attempts.append(running_attempt)
                    return current_exec, executed_attempts, None

                # Branch 1: SUCCESS
                if adapter_result.outcome_type == ProviderOutcomeType.SUCCESS:
                    raw_file_path = adapter_result.file_path or ""
                    local_file_path: str | None = None
                    download_or_probe_failed = False
                    failure_err_code = "TECHNICAL_FAILURE"
                    failure_err_msg = ""

                    if raw_file_path.startswith(("http://", "https://")):
                        try:
                            from app.services.asset_probe_service import (
                                download_media_file,
                            )
                            dest_path = download_media_file(
                                url=raw_file_path,
                                target_dir=storage_path,
                                filename_prefix=f"attempt_{submitting_attempt.execution_attempt_id}",
                            )
                            local_file_path = str(dest_path)
                        except Exception as dl_exc:  # noqa: BLE001
                            download_or_probe_failed = True
                            failure_err_code = "DOWNLOAD_FAILED"
                            failure_err_msg = f"Failed to download media URL: {dl_exc}"
                    else:
                        local_file_path = raw_file_path
                        if not local_file_path or not os.path.isfile(local_file_path) or os.path.getsize(local_file_path) == 0:
                            download_or_probe_failed = True
                            failure_err_code = "EMPTY_MEDIA_FILE" if (local_file_path and os.path.exists(local_file_path)) else "ASSET_FILE_NOT_FOUND"
                            failure_err_msg = f"Local media file missing or empty: {local_file_path}"

                    probe = None
                    if not download_or_probe_failed and local_file_path:
                        try:
                            probe = probe_media_file(local_file_path)
                        except Exception as pr_exc:  # noqa: BLE001
                            download_or_probe_failed = True
                            failure_err_code = "INVALID_ASSET_FORMAT"
                            failure_err_msg = f"Media file probing failed: {pr_exc}"

                    if download_or_probe_failed or probe is None:
                        logger.warning(
                            f"Asset acquisition failed for attempt {submitting_attempt.execution_attempt_id}: "
                            f"{failure_err_code} - {failure_err_msg}"
                        )
                        adapter_result = AdapterExecutionResult(
                            outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                            error_code=failure_err_code,
                            error_message=failure_err_msg,
                            raw_response=adapter_result.raw_response,
                        )
                    else:
                        succeeded_attempt = running_attempt.mark_succeeded(
                            raw_response=adapter_result.raw_response
                        )
                        executed_attempts.append(succeeded_attempt)

                        asset_version = ShotAssetVersion(
                            shot_id=current_exec.shot_id,
                            shot_revision_id=current_exec.shot_revision_id,
                            execution_attempt_id=succeeded_attempt.execution_attempt_id,
                            file_path=probe.file_path,
                            file_hash=probe.file_hash,
                            file_size_bytes=probe.file_size_bytes,
                            media_type=probe.media_type,
                            mime_type=probe.mime_type,
                            width=probe.width,
                            height=probe.height,
                            duration_seconds=probe.duration_seconds,
                            fps=probe.fps,
                            provider=candidate.provider,
                            model=candidate.model,
                            generation_mode=candidate.generation_mode,
                        )
                        final_exec = current_exec.mark_succeeded(
                            version_id=asset_version.shot_asset_version_id,
                            attempt_id=succeeded_attempt.execution_attempt_id,
                        )

                        if trace_writer is not None and attempt_ctx is not None:
                            if attempt_start_ev is not None:
                                trace_writer.complete_event(
                                    event=attempt_start_ev,
                                    status=TraceEventStatus.SUCCEEDED,
                                    attributes_update={
                                        "resulting_shot_asset_version_id": asset_version.shot_asset_version_id,
                                    },
                                )
                            cost_usage = CostUsageTraceData(
                                cost_status=CostEstimateStatus.UNKNOWN,
                                raw_usage={"provider": candidate.provider, "model": candidate.model},
                            )
                            attempt_data = ExecutionAttemptTraceData(
                                execution_run_id=current_exec.execution_run_id,
                                shot_execution_id=current_exec.shot_execution_id,
                                execution_attempt_id=succeeded_attempt.execution_attempt_id,
                                shot_revision_id=current_exec.shot_revision_id,
                                provider=candidate.provider,
                                model=candidate.model,
                                generation_mode=candidate.generation_mode.value
                                if hasattr(candidate.generation_mode, "value")
                                else str(candidate.generation_mode),
                                attempt_number=attempt_num,
                                has_provider_receipt=receipt is not None,
                                submission_outcome_unknown=False,
                                resulting_shot_asset_version_id=asset_version.shot_asset_version_id,
                                usage=cost_usage,
                            )
                            trace_writer.record_event(
                                context=attempt_ctx,
                                event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
                                status=TraceEventStatus.SUCCEEDED,
                                attributes=attempt_data,
                                parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                            )
                            trace_writer.record_event(
                                context=attempt_ctx.child_context(
                                    parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                                    shot_asset_version_id=asset_version.shot_asset_version_id,
                                ),
                                event_type=TraceEventType.SHOT_ASSET_CREATED,
                                status=TraceEventStatus.SUCCEEDED,
                                attributes={
                                    "shot_asset_version_id": asset_version.shot_asset_version_id,
                                    "shot_id": current_exec.shot_id,
                                    "shot_revision_id": current_exec.shot_revision_id,
                                    "execution_attempt_id": succeeded_attempt.execution_attempt_id,
                                    "file_path": asset_version.file_path,
                                    "file_hash": asset_version.file_hash,
                                    "file_size_bytes": asset_version.file_size_bytes,
                                    "width": asset_version.width,
                                    "height": asset_version.height,
                                    "duration_seconds": asset_version.duration_seconds,
                                    "provider": candidate.provider,
                                    "model": candidate.model,
                                },
                            )

                        return final_exec, executed_attempts, asset_version

                # Branch 2: SUBMISSION_OUTCOME_UNKNOWN (NEEDS_RECOVERY)
                # Strictly halt immediately! Never retry or fallback automatically.
                if adapter_result.outcome_type == ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN:
                    unknown_attempt = running_attempt.mark_submission_unknown(
                        error_code="SUBMISSION_OUTCOME_UNKNOWN",
                        error_message=adapter_result.error_message or "Task submission outcome unconfirmed",
                        raw_response=adapter_result.raw_response,
                    )
                    executed_attempts.append(unknown_attempt)

                    needs_recovery_exec = current_exec.mark_submission_unknown(
                        error_code="SUBMISSION_OUTCOME_UNKNOWN",
                        error_message=adapter_result.error_message or "Task submission outcome unconfirmed",
                        remote_task_id=adapter_result.remote_task_id,
                        attempt_id=unknown_attempt.execution_attempt_id,
                    )

                    if trace_writer is not None and attempt_ctx is not None:
                        if attempt_start_ev is not None:
                            trace_writer.complete_event(
                                event=attempt_start_ev,
                                status=TraceEventStatus.INDETERMINATE,
                                error_code="SUBMISSION_OUTCOME_UNKNOWN",
                            )
                        attempt_data = ExecutionAttemptTraceData(
                            execution_run_id=current_exec.execution_run_id,
                            shot_execution_id=current_exec.shot_execution_id,
                            execution_attempt_id=unknown_attempt.execution_attempt_id,
                            shot_revision_id=current_exec.shot_revision_id,
                            provider=candidate.provider,
                            model=candidate.model,
                            generation_mode=candidate.generation_mode.value
                            if hasattr(candidate.generation_mode, "value")
                            else str(candidate.generation_mode),
                            attempt_number=attempt_num,
                            has_provider_receipt=receipt is not None,
                            submission_outcome_unknown=True,
                            technical_failure_category="SUBMISSION_OUTCOME_UNKNOWN",
                        )
                        trace_writer.record_event(
                            context=attempt_ctx,
                            event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
                            status=TraceEventStatus.INDETERMINATE,
                            attributes=attempt_data,
                            error_code="SUBMISSION_OUTCOME_UNKNOWN",
                            parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                        )
                        trace_writer.record_event(
                            context=attempt_ctx,
                            event_type=TraceEventType.EXECUTION_RECOVERY_REQUIRED,
                            status=TraceEventStatus.INDETERMINATE,
                            attributes={
                                "execution_run_id": current_exec.execution_run_id,
                                "shot_id": current_exec.shot_id,
                                "shot_revision_id": current_exec.shot_revision_id,
                                "execution_attempt_id": unknown_attempt.execution_attempt_id,
                                "remote_task_id": adapter_result.remote_task_id,
                                "error_code": "SUBMISSION_OUTCOME_UNKNOWN",
                                "error_message": adapter_result.error_message or "Task submission outcome unconfirmed",
                            },
                            error_code="SUBMISSION_OUTCOME_UNKNOWN",
                            parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                        )

                    return needs_recovery_exec, executed_attempts, None

                # Branch 3: CAPABILITY_INCOMPATIBLE or DEFINITIVE_TECHNICAL_FAILURE
                if adapter_result.outcome_type == ProviderOutcomeType.CAPABILITY_INCOMPATIBLE:
                    failed_attempt = running_attempt.mark_failed(
                        error_code=adapter_result.error_code or "CAPABILITY_INCOMPATIBLE",
                        error_message=adapter_result.error_message or "Capability incompatible",
                        raw_response=adapter_result.raw_response,
                    )
                    failure_cat = FailureCategory.UNSUPPORTED_CAPABILITY
                else:
                    failed_attempt = running_attempt.mark_failed(
                        error_code=adapter_result.error_code or "DEFINITIVE_TECHNICAL_FAILURE",
                        error_message=adapter_result.error_message or "Technical failure",
                        raw_response=adapter_result.raw_response,
                    )
                    failure_cat = FailureCategory.TRANSIENT_TECHNICAL

                executed_attempts.append(failed_attempt)

                if trace_writer is not None and attempt_ctx is not None:
                    if attempt_start_ev is not None:
                        trace_writer.complete_event(
                            event=attempt_start_ev,
                            status=TraceEventStatus.FAILED,
                            error_code=failed_attempt.error_code,
                        )
                    attempt_data = ExecutionAttemptTraceData(
                        execution_run_id=current_exec.execution_run_id,
                        shot_execution_id=current_exec.shot_execution_id,
                        execution_attempt_id=failed_attempt.execution_attempt_id,
                        shot_revision_id=current_exec.shot_revision_id,
                        provider=candidate.provider,
                        model=candidate.model,
                        generation_mode=candidate.generation_mode.value
                        if hasattr(candidate.generation_mode, "value")
                        else str(candidate.generation_mode),
                        attempt_number=attempt_num,
                        has_provider_receipt=receipt is not None,
                        technical_failure_category=failure_cat.value
                        if hasattr(failure_cat, "value")
                        else str(failure_cat),
                    )
                    trace_writer.record_event(
                        context=attempt_ctx,
                        event_type=TraceEventType.EXECUTION_ATTEMPT_COMPLETED,
                        status=TraceEventStatus.FAILED,
                        attributes=attempt_data,
                        error_code=failed_attempt.error_code,
                        parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                    )

                # Determine next candidate for retry decision evaluation
                next_cand = candidates[current_idx + 1] if (current_idx + 1) < len(candidates) else None

                decision = evaluate_retry_decision(
                    failure_category=failure_cat,
                    current_candidate=candidate,
                    candidate_attempt_count=candidate_attempts,
                    total_shot_attempts=len(executed_attempts),
                    policy=self._policy,
                    model_selection_mode=str(model_mode),
                    next_candidate=next_cand,
                )

                if decision.action == RetryAction.RETRY_SAME_CANDIDATE:
                    # Retry same candidate within budget
                    continue
                elif decision.action == RetryAction.FALLBACK_NEXT_CANDIDATE:
                    if trace_writer is not None and attempt_ctx is not None:
                        fallback_data = ExecutionFallbackTraceData(
                            execution_run_id=current_exec.execution_run_id,
                            shot_id=current_exec.shot_id,
                            from_provider=candidate.provider,
                            from_model=candidate.model,
                            to_provider=next_cand.provider if next_cand else "NONE",
                            to_model=next_cand.model if next_cand else "NONE",
                            reason_code=decision.reason,
                        )
                        trace_writer.record_event(
                            context=attempt_ctx,
                            event_type=TraceEventType.EXECUTION_FALLBACK_SELECTED,
                            status=TraceEventStatus.SUCCEEDED,
                            attributes=fallback_data,
                            parent_event_id=attempt_start_ev.trace_event_id if attempt_start_ev else None,
                        )
                    # Advance candidate index and break inner retry loop
                    break
                elif decision.action == RetryAction.STOP_EXHAUSTED:
                    exhausted_exec = current_exec.mark_exhausted(
                        error_code="SHOT_EXECUTION_EXHAUSTED",
                        error_message=f"Total attempt budget ({self._policy.max_total_attempts_per_shot}) exhausted",
                        attempt_id=failed_attempt.execution_attempt_id,
                    )
                    return exhausted_exec, executed_attempts, None
                else:  # STOP_FAILED
                    failed_exec = current_exec.mark_failed(
                        error_code=decision.reason,
                        error_message=f"Execution stopped with reason: {decision.reason}",
                        attempt_id=failed_attempt.execution_attempt_id,
                    )
                    return failed_exec, executed_attempts, None

            # Advance to next candidate (fallback)
            current_exec = current_exec.advance_candidate()
            current_idx = current_exec.current_candidate_index

        # All candidates exhausted
        exhausted_exec = current_exec.mark_failed(
            error_code="ALL_CANDIDATES_EXHAUSTED",
            error_message="All eligible candidates and retries have failed",
        )
        return exhausted_exec, executed_attempts, None

    def poll_attempt(
        self,
        shot_execution: ShotExecution,
        attempt: ExecutionAttempt,
        storage_base_dir: Path | str,
    ) -> tuple[ShotExecution, ExecutionAttempt, ShotAssetVersion | None]:
        """
        Polls the status of an active asynchronous execution attempt.
        """
        adapter = self._adapter_registry.get_adapter(attempt.provider)
        receipt = attempt.provider_receipt
        job_id = (receipt.provider_job_id if receipt else None) or shot_execution.unconfirmed_remote_task_id
        if not job_id:
            return shot_execution, attempt, None

        storage_path = Path(storage_base_dir).resolve() / "shot_assets" / shot_execution.shot_id
        storage_path.mkdir(parents=True, exist_ok=True)

        result = adapter.get_status(job_id, storage_path)
        if result.outcome_type == ProviderOutcomeType.SUCCESS and result.file_path and result.status == "SUCCEEDED":
            probe = probe_media_file(result.file_path)
            asset_version = ShotAssetVersion(
                shot_id=shot_execution.shot_id,
                shot_revision_id=shot_execution.shot_revision_id,
                execution_attempt_id=attempt.execution_attempt_id,
                file_path=probe.file_path,
                file_hash=probe.file_hash,
                file_size_bytes=probe.file_size_bytes,
                media_type=probe.media_type,
                mime_type=probe.mime_type,
                width=probe.width,
                height=probe.height,
                duration_seconds=probe.duration_seconds,
                fps=probe.fps,
                provider=attempt.provider,
                model=attempt.model,
                generation_mode=attempt.generation_mode,
            )
            succeeded_attempt = attempt.mark_succeeded(raw_response=result.raw_response)
            succeeded_exec = shot_execution.mark_succeeded(
                version_id=asset_version.shot_asset_version_id,
                attempt_id=attempt.execution_attempt_id,
            )
            return succeeded_exec, succeeded_attempt, asset_version

        if result.outcome_type in (
            ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
            ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
        ) or result.status == "FAILED":
            failed_attempt = attempt.mark_failed(
                error_code=result.error_code or "REMOTE_JOB_FAILED",
                error_message=result.error_message or "Asynchronous job failed remotely",
                raw_response=result.raw_response,
            )
            failed_exec = shot_execution.mark_failed(
                error_code=result.error_code or "REMOTE_JOB_FAILED",
                error_message=result.error_message or "Asynchronous job failed remotely",
                attempt_id=attempt.execution_attempt_id,
            )
            return failed_exec, failed_attempt, None

        return shot_execution, attempt, None
