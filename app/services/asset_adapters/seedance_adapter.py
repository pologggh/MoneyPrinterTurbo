from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.domain.asset_execution import AdapterExecutionResult, ProviderOutcomeType
from app.domain.asset_router import AssetRouteCandidate, AssetRoutingRequest
from app.services.asset_adapters.base import (
    AssetExecutionAdapter,
    ProviderExecutionCapabilities,
)
from app.services.volcengine_seedance import (
    VolcEngineSeedanceDownloadError,
    VolcEngineSeedanceError,
    VolcEngineSeedanceUnconfirmedTaskError,
)


class SeedanceAdapter(AssetExecutionAdapter):
    """
    Adapter wrapping VolcEngine Seedance AI video generation.
    """

    def __init__(
        self,
        generator_func: Callable[..., Any] | None = None,
        status_func: Callable[..., Any] | None = None,
        submit_func: Callable[..., Any] | None = None,
    ):
        self._generator_func = generator_func
        self._status_func = status_func
        self._submit_func = submit_func

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
    ) -> AdapterExecutionResult:
        if self._submit_func is not None:
            return self._submit_func(request, candidate, target_dir, idempotency_key)
        return self.execute(request, candidate, target_dir)

    def get_status(
        self,
        provider_job_id: str,
        target_dir: Path | None = None,
    ) -> AdapterExecutionResult:
        if self._status_func is not None:
            return self._status_func(provider_job_id, target_dir)
        try:
            import requests

            from app.config import config
            from app.services.volcengine_seedance import (
                _base_url,
                _tls_verify,
                get_api_key,
            )

            api_key = get_api_key()
            if not api_key:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    error_code="MISSING_API_KEY",
                    error_message="VolcEngine API key not configured",
                )
            url = f"{_base_url()}/contents/generations/tasks/{provider_job_id}"
            headers = {"Authorization": f"Bearer {api_key}"}
            res = requests.get(
                url,
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=30,
            )
            if not (200 <= res.status_code < 300):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                    remote_task_id=provider_job_id,
                    error_code=f"HTTP_{res.status_code}",
                    error_message=f"Query failed with HTTP {res.status_code}",
                )
            body = res.json()
            status = str(body.get("status", "")).lower()
            if status in ("queued", "running"):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUCCESS,
                    status="RUNNING",
                    remote_task_id=provider_job_id,
                    raw_response=body,
                )
            elif status == "succeeded":
                content = body.get("content") or {}
                video_url = content.get("video_url")
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.SUCCESS,
                    status="SUCCEEDED",
                    file_path=video_url,
                    remote_task_id=provider_job_id,
                    raw_response=body,
                )
            else:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    status="FAILED",
                    remote_task_id=provider_job_id,
                    error_code="TASK_FAILED",
                    error_message=f"Task terminated with status: {status}",
                    raw_response=body,
                )
        except Exception as exc:  # noqa: BLE001
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                remote_task_id=provider_job_id,
                error_code="QUERY_ERROR",
                error_message=f"Failed to query Seedance task: {exc}",
            )

    def execute(
        self,
        request: AssetRoutingRequest,
        candidate: AssetRouteCandidate,
        target_dir: Path,
    ) -> AdapterExecutionResult:
        target_dir.mkdir(parents=True, exist_ok=True)
        prompt = request.generation_prompt or request.visual_goal

        try:
            if self._generator_func is not None:
                video_paths = self._generator_func(
                    search_term=prompt,
                    minimum_duration=max(1, int(request.target_duration)),
                    target_dir=str(target_dir),
                )
            else:
                from app.models.schema import VideoAspect
                from app.services.volcengine_seedance import generate_videos

                aspect_str = getattr(request, "aspect_ratio", None) or getattr(request, "target_aspect_ratio", None)
                aspect_enum = (
                    VideoAspect.portrait
                    if aspect_str == "9:16"
                    else VideoAspect.landscape
                )
                items = generate_videos(
                    search_term=prompt,
                    minimum_duration=max(1, int(request.target_duration)),
                    video_aspect=aspect_enum,
                )
                # In real execution, downloaded item URL or local file path
                video_paths = [items[0].url] if items else []

            if not video_paths:
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                    error_code="EMPTY_GENERATION_RESULT",
                    error_message="Seedance returned no video items",
                )

            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUCCESS,
                file_path=str(video_paths[0]),
                raw_response={"provider": candidate.provider, "model": candidate.model},
            )

        except VolcEngineSeedanceUnconfirmedTaskError as exc:
            # Payment task may have been created remotely; status unknown.
            # Strictly prohibit blind re-submission!
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.SUBMISSION_OUTCOME_UNKNOWN,
                remote_task_id=getattr(exc, "task_id", None),
                error_code="SUBMISSION_OUTCOME_UNKNOWN",
                error_message=f"Seedance task submission outcome unconfirmed: {exc}",
                raw_response={"task_id": getattr(exc, "task_id", "")},
            )

        except VolcEngineSeedanceDownloadError as exc:
            # Video was generated remotely, but download failed. Technical/network error.
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                remote_task_id=getattr(exc, "task_id", None),
                error_code="DOWNLOAD_FAILED",
                error_message=f"Seedance download failed: {exc}",
            )

        except VolcEngineSeedanceError as exc:
            msg = str(exc).lower()
            # If rejected by parameter constraints or content safety
            if any(k in msg for k in ("invalid parameter", "unsupported", "content policy", "sensitive", "rejected")):
                return AdapterExecutionResult(
                    outcome_type=ProviderOutcomeType.CAPABILITY_INCOMPATIBLE,
                    error_code="CAPABILITY_INCOMPATIBLE",
                    error_message=f"Seedance rejected request parameters or content: {exc}",
                )
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="TECHNICAL_FAILURE",
                error_message=f"Seedance technical failure: {exc}",
            )

        except Exception as exc:  # noqa: BLE001
            return AdapterExecutionResult(
                outcome_type=ProviderOutcomeType.DEFINITIVE_TECHNICAL_FAILURE,
                error_code="UNEXPECTED_ERROR",
                error_message=f"Unexpected error in Seedance adapter: {exc}",
            )
