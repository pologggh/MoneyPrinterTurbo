from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.asset_execution import (
    AttemptRequest,
    ExecutionAttempt,
    ExecutionAttemptStatus,
    ProviderOutcomeType,
    ProviderReceipt,
    RecoveryAction,
    RecoveryDecision,
    ShotAssetVersion,
    ShotExecution,
    ShotExecutionStatus,
)
from app.services.asset_adapters.registry import AdapterRegistry
from app.services.asset_probe_service import probe_media_file


class AttemptRecoveryService:
    """
    Deterministic service for recovering attempts whose submission outcome was uncertain.
    Distinguishes recovery (discovering outcome of existing external request) from retry.
    Never creates a new idempotency key; reuses the same idempotency key for the same attempt.
    """

    def __init__(self, adapter_registry: AdapterRegistry | None = None):
        self._adapter_registry = adapter_registry or AdapterRegistry()

    def recover(
        self,
        shot_execution: ShotExecution,
        attempt: ExecutionAttempt,
        attempt_request: AttemptRequest,
        receipt: ProviderReceipt | None = None,
        target_dir: Path | str | None = None,
    ) -> tuple[RecoveryDecision, ShotExecution, ExecutionAttempt, ShotAssetVersion | None]:
        if hasattr(self._adapter_registry, "has_adapter") and not self._adapter_registry.has_adapter(attempt.provider):
            available = getattr(self._adapter_registry, "list_providers", list)()
            err_msg = (
                f"No adapter registered for provider '{attempt.provider}' during recovery. "
                f"Available: {available}"
            )
            rec_dec = RecoveryDecision(
                execution_attempt_id=attempt.execution_attempt_id,
                action=RecoveryAction.RESOLVED_FAILED,
                error_code="ADAPTER_NOT_FOUND",
                error_message=err_msg,
            )
            failed_attempt = attempt.mark_failed(
                error_code="ADAPTER_NOT_FOUND",
                error_message=err_msg,
            )
            failed_exec = shot_execution.record_failure(
                error_code="ADAPTER_NOT_FOUND",
                error_message=err_msg,
            )
            return rec_dec, failed_exec, failed_attempt, None

        adapter = self._adapter_registry.get_adapter(attempt.provider)
        caps = adapter.get_capabilities()
        job_id = (
            (receipt.provider_job_id if receipt else None)
            or shot_execution.unconfirmed_remote_task_id
        )

        # Path A: Provider job ID exists and provider supports async status query
        if job_id and caps.supports_async_status:
            try:
                result = adapter.get_status(
                    job_id, Path(target_dir) if target_dir else None
                )
                if (
                    result.outcome_type == ProviderOutcomeType.SUCCESS
                    and result.file_path
                ):
                    # File produced! If remote URL, download before probing
                    local_path = result.file_path
                    if local_path.startswith(("http://", "https://")):
                        from app.services.asset_probe_service import download_media_file
                        t_dir = Path(target_dir) if target_dir else Path("storage/recovery")
                        local_path = str(download_media_file(local_path, t_dir, f"recovery_{attempt.execution_attempt_id}"))
                    probe = probe_media_file(local_path)
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
                    # Transition attempt and shot to SUCCEEDED
                    resolved_attempt = attempt.mark_succeeded(
                        raw_response=result.raw_response
                    )
                    resolved_exec = shot_execution.mark_succeeded(
                        version_id=asset_version.shot_asset_version_id,
                        attempt_id=attempt.execution_attempt_id,
                    )
                    decision = RecoveryDecision(
                        execution_attempt_id=attempt.execution_attempt_id,
                        action=RecoveryAction.RESOLVED_SUCCEEDED,
                        evidence={
                            "provider_job_id": job_id,
                            "file_path": result.file_path,
                        },
                        resolved_state=ExecutionAttemptStatus.SUCCEEDED,
                        reason_code="RECOVERY_RESOLVED_SUCCESS",
                    )
                    return decision, resolved_exec, resolved_attempt, asset_version

                if result.outcome_type in (
                    ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                ) or result.status == "FAILED":
                    # Task definitely failed remotely. Mark attempt FAILED.
                    resolved_attempt = attempt.mark_failed(
                        error_code=result.error_code or "REMOTE_TASK_FAILED",
                        error_message=result.error_message
                        or "Remote task failed on provider",
                        raw_response=result.raw_response,
                    )
                    # Shot execution can now transition from NEEDS_RECOVERY to RUNNING
                    # allowing safe subsequent retry/fallback
                    resolved_exec = shot_execution.model_copy(
                        update={
                            "status": ShotExecutionStatus.RUNNING,
                            "unconfirmed_remote_task_id": None,
                        }
                    )
                    decision = RecoveryDecision(
                        execution_attempt_id=attempt.execution_attempt_id,
                        action=RecoveryAction.RESOLVED_FAILED,
                        evidence={
                            "provider_job_id": job_id,
                            "error_code": result.error_code,
                        },
                        resolved_state=ExecutionAttemptStatus.FAILED,
                        reason_code="RECOVERY_RESOLVED_FAILURE",
                    )
                    return decision, resolved_exec, resolved_attempt, None

                # Still running or unconfirmed
                decision = RecoveryDecision(
                    execution_attempt_id=attempt.execution_attempt_id,
                    action=RecoveryAction.STILL_UNKNOWN,
                    evidence={"provider_job_id": job_id, "status": result.status},
                    resolved_state=attempt.status,
                    reason_code="REMOTE_TASK_STILL_RUNNING",
                )
                return decision, shot_execution, attempt, None

            except Exception as exc:  # noqa: BLE001
                decision = RecoveryDecision(
                    execution_attempt_id=attempt.execution_attempt_id,
                    action=RecoveryAction.STILL_UNKNOWN,
                    evidence={"error": str(exc)},
                    resolved_state=attempt.status,
                    reason_code="ASYNC_STATUS_QUERY_FAILED",
                )
                return decision, shot_execution, attempt, None

        # Path B / C: Provider supports safe resubmission with same key
        if caps.supports_safe_resubmission_with_same_key:
            decision = RecoveryDecision(
                execution_attempt_id=attempt.execution_attempt_id,
                action=RecoveryAction.SAFE_TO_RESUBMIT_SAME_ATTEMPT,
                evidence={"idempotency_key": attempt_request.idempotency_key},
                resolved_state=attempt.status,
                reason_code="SAFE_TO_RESUBMIT_WITH_SAME_KEY",
            )
            return decision, shot_execution, attempt, None

        # Path D: Provider offers no safe way to determine outcome -> remains NEEDS_RECOVERY
        decision = RecoveryDecision(
            execution_attempt_id=attempt.execution_attempt_id,
            action=RecoveryAction.STILL_UNKNOWN,
            evidence={"provider": attempt.provider},
            resolved_state=attempt.status,
            reason_code="RECOVERY_NOT_SUPPORTED",
        )
        return decision, shot_execution, attempt, None

    def recover_run(
        self,
        run_id: str,
        session: Any,
        storage_base_dir: Path | str | None = None,
        force_timeout: bool = False,
    ) -> dict[str, Any]:
        """
        Scans all shot executions in a run that need recovery, executes recovery decisions,
        persists updated attempts/executions/assets, and reconciles the overall ExecutionRun status.
        """
        from app.domain.asset_execution import ExecutionStatus
        from app.persistence.repositories import ExecutionRepository

        repo = ExecutionRepository(session)
        run = repo.get_execution_run(run_id)
        if run is None:
            raise ValueError(f"ExecutionRun '{run_id}' not found")

        shot_execs = repo.list_shot_executions_for_run(run_id)
        recovery_results = []
        resolved_count = 0

        target_dir = Path(storage_base_dir) if storage_base_dir else Path("storage/recovery")
        target_dir.mkdir(parents=True, exist_ok=True)

        for se in shot_execs:
            if se.status in (
                ShotExecutionStatus.NEEDS_RECOVERY,
                ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN,
            ):
                if not se.attempt_ids:
                    continue
                latest_attempt_id = se.attempt_ids[-1]
                attempt = repo.get_execution_attempt(latest_attempt_id)
                attempt_req = repo.get_attempt_request(latest_attempt_id)
                receipt = repo.get_provider_receipt(latest_attempt_id)

                if attempt is None or attempt_req is None:
                    continue

                decision, updated_exec, updated_attempt, new_asset = self.recover(
                    shot_execution=se,
                    attempt=attempt,
                    attempt_request=attempt_req,
                    receipt=receipt,
                    target_dir=target_dir,
                )

                repo.update_execution_attempt(updated_attempt)
                repo.update_shot_execution(updated_exec)
                if new_asset is not None:
                    repo.add_shot_asset_version(new_asset)

                recovery_results.append({
                    "shot_id": se.shot_id,
                    "attempt_id": latest_attempt_id,
                    "action": decision.action.value,
                    "reason_code": decision.reason_code,
                    "status": updated_attempt.status.value,
                })

                if decision.action in (
                    RecoveryAction.RESOLVED_SUCCEEDED,
                    RecoveryAction.RESOLVED_FAILED,
                ):
                    resolved_count += 1

        # Re-evaluate ExecutionRun status
        refreshed_execs = repo.list_shot_executions_for_run(run_id)
        succeeded = sum(1 for e in refreshed_execs if e.status == ShotExecutionStatus.SUCCEEDED)
        reused = sum(1 for e in refreshed_execs if e.status == ShotExecutionStatus.REUSED)
        failed = sum(1 for e in refreshed_execs if e.status in (ShotExecutionStatus.FAILED, ShotExecutionStatus.EXHAUSTED))
        still_needs_rec = sum(
            1 for e in refreshed_execs
            if e.status in (ShotExecutionStatus.NEEDS_RECOVERY, ShotExecutionStatus.SUBMISSION_OUTCOME_UNKNOWN)
        )

        if still_needs_rec > 0:
            new_run_status = ExecutionStatus.NEEDS_RECOVERY
        elif succeeded + reused == len(refreshed_execs):
            new_run_status = ExecutionStatus.COMPLETED
        else:
            new_run_status = ExecutionStatus.FAILED

        repo.update_execution_run_status(
            run_id=run_id,
            status=new_run_status,
            succeeded_shots=succeeded,
            reused_shots=reused,
            failed_shots=failed,
            recovery_required_shots=still_needs_rec,
            finished_at=datetime.now(UTC),
        )

        return {
            "execution_run_id": run_id,
            "resolved_count": resolved_count,
            "recovery_results": recovery_results,
            "overall_status": new_run_status.value,
            "succeeded_shots": succeeded,
            "reused_shots": reused,
            "failed_shots": failed,
            "recovery_required_shots": still_needs_rec,
        }
