from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.domain.asset_execution import (
    AdapterExecutionResult,
    ProviderOutcomeType,
    ProviderReceipt,
)
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.aliyun_wan import (
    AliyunWanClient,
    AliyunWanConfigurationError,
    AliyunWanRequestRejectedError,
)
from app.services.asset_adapters.base import (
    AssetExecutionAdapter,
    ProviderExecutionCapabilities,
)


class AliyunWanAdapter(AssetExecutionAdapter):
    """Executes Alibaba Cloud Model Studio Wan video generation jobs."""

    def __init__(
        self,
        client: AliyunWanClient | Any | None = None,
        *,
        poll_interval: float | None = None,
        max_polls: int | None = None,
    ) -> None:
        from app.config import config

        self._client = client or AliyunWanClient()
        self._poll_interval = (
            float(config.app.get("aliyun_wan_poll_interval", 5))
            if poll_interval is None
            else poll_interval
        )
        if max_polls is None:
            timeout = int(config.app.get("aliyun_wan_run_timeout", 1800))
            divisor = max(self._poll_interval, 1)
            max_polls = max(1, int(timeout / divisor))
        self._max_polls = max_polls

    @property
    def capabilities(self) -> ProviderExecutionCapabilities:
        return ProviderExecutionCapabilities(
            synchronous_completion=True,
            supports_async_status=True,
            supports_cancel=False,
            supports_idempotency=False,
            supports_safe_resubmission_with_same_key=False,
        )

    def submit(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
        idempotency_key: str | None = None,
    ) -> AdapterExecutionResult | ProviderReceipt:
        del idempotency_key
        del candidate
        target_dir.mkdir(parents=True, exist_ok=True)
        prompt = (request.generation_prompt or request.visual_goal).strip()
        try:
            submission = self._client.submit_video_task(
                prompt=prompt,
                duration=round(request.target_duration),
                ratio=request.aspect_ratio or "16:9",
            )
        except AliyunWanConfigurationError as exc:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                error_code="PROVIDER_NOT_CONFIGURED",
                error_message=str(exc),
            )
        except AliyunWanRequestRejectedError as exc:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="SUBMISSION_REJECTED",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                error_code="SUBMISSION_OUTCOME_UNKNOWN",
                error_message=f"Wan task submission outcome is unknown: {exc}",
            )

        task_id = submission.get("task_id")
        if not task_id:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                error_code="MISSING_TASK_ID",
                error_message="Wan accepted submission without a task ID",
                raw_response=submission,
            )
        return ProviderReceipt(
            execution_attempt_id="",
            provider="aliyun_wan",
            provider_job_id=task_id,
            provider_status=str(submission.get("status") or "PENDING"),
            sanitized_metadata={
                "request_id": str(submission.get("request_id") or "")
            },
        )

    def _normalize_status(self, snapshot: dict[str, str]) -> AdapterExecutionResult:
        task_id = snapshot.get("task_id") or None
        status = str(snapshot.get("status") or "UNKNOWN").upper()
        if status in {"PENDING", "RUNNING"}:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="RUNNING",
                remote_task_id=task_id,
                raw_response=snapshot,
            )
        if status == "SUCCEEDED" and snapshot.get("video_url"):
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                status="SUCCEEDED",
                file_path=snapshot["video_url"],
                remote_task_id=task_id,
                raw_response=snapshot,
            )
        if status in {"FAILED", "CANCELED"}:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                status=status,
                remote_task_id=task_id,
                error_code=snapshot.get("error_code") or "TASK_FAILED",
                error_message=snapshot.get("error_message")
                or f"Wan task terminated with status {status}",
                raw_response=snapshot,
            )
        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            status=status,
            remote_task_id=task_id,
            error_code="UNKNOWN_TASK_STATUS",
            error_message=f"Wan task returned unknown status {status}",
            raw_response=snapshot,
        )

    def get_status(
        self,
        provider_job_id: str,
        target_dir: Path | None = None,
    ) -> AdapterExecutionResult:
        del target_dir
        for poll_index in range(self._max_polls):
            try:
                result = self._normalize_status(
                    self._client.get_video_task(provider_job_id)
                )
            except Exception as exc:  # noqa: BLE001
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                    remote_task_id=provider_job_id,
                    error_code="STATUS_QUERY_FAILED",
                    error_message=f"Failed to query Wan task status: {exc}",
                )
            if result.status != "RUNNING":
                return result
            if poll_index + 1 < self._max_polls and self._poll_interval > 0:
                time.sleep(self._poll_interval)

        return AdapterExecutionResult(
            outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
            status="RUNNING",
            remote_task_id=provider_job_id,
            error_code="POLL_TIMEOUT",
            error_message="Wan task did not reach a terminal state before timeout",
        )

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        submission = self.submit(request, candidate, target_dir)
        if isinstance(submission, AdapterExecutionResult):
            return submission
        if not submission.provider_job_id:
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                error_code="MISSING_TASK_ID",
                error_message="Wan accepted submission without a task ID",
            )
        return self.get_status(submission.provider_job_id, target_dir)
