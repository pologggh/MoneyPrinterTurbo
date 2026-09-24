"""Alibaba Cloud Model Studio Wan video-generation client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_MODEL = "wan2.7-t2v"
DEFAULT_REGION = "cn-beijing"
SUPPORTED_REGIONS = {"cn-beijing", "ap-southeast-1"}


class AliyunWanError(RuntimeError):
    """Base error for Alibaba Cloud Wan API operations."""


class AliyunWanConfigurationError(AliyunWanError):
    """Raised when required Wan provider settings are incomplete or invalid."""


class AliyunWanRequestRejectedError(AliyunWanError):
    """Raised when the provider definitively rejects a task submission."""


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """Return whether the minimum credentials required by Wan are configured."""
    if settings is None:
        from app.config import config

        settings = config.app
    api_key = str(settings.get("aliyun_wan_api_key", "") or "").strip()
    workspace_id = str(settings.get("aliyun_wan_workspace_id", "") or "").strip()
    region = str(
        settings.get("aliyun_wan_region", DEFAULT_REGION) or DEFAULT_REGION
    ).strip()
    model = str(settings.get("aliyun_wan_model", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    return bool(
        api_key
        and workspace_id
        and region in SUPPORTED_REGIONS
        and model == DEFAULT_MODEL
    )


class AliyunWanClient:
    """Client boundary for Alibaba Cloud Model Studio Wan APIs."""

    def __init__(
        self,
        settings: Mapping[str, Any] | None = None,
        http_client: Any | None = None,
    ) -> None:
        if settings is None:
            from app.config import config

            settings = config.app
        if http_client is None:
            import requests

            http_client = requests
        self._settings = settings
        self._http = http_client

    @property
    def _api_key(self) -> str:
        return str(self._settings.get("aliyun_wan_api_key", "") or "").strip()

    @property
    def _workspace_id(self) -> str:
        return str(self._settings.get("aliyun_wan_workspace_id", "") or "").strip()

    @property
    def _base_url(self) -> str:
        region = str(
            self._settings.get("aliyun_wan_region", DEFAULT_REGION) or DEFAULT_REGION
        ).strip()
        return f"https://{self._workspace_id}.{region}.maas.aliyuncs.com/api/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

    def _validate_configuration(self) -> None:
        if not self._api_key:
            raise AliyunWanConfigurationError("Aliyun Wan API key is not configured")
        if not self._workspace_id:
            raise AliyunWanConfigurationError(
                "Aliyun Wan workspace ID is not configured"
            )
        region = str(
            self._settings.get("aliyun_wan_region", DEFAULT_REGION) or DEFAULT_REGION
        ).strip()
        if region not in SUPPORTED_REGIONS:
            raise AliyunWanConfigurationError(
                f"Unsupported Aliyun Wan region: {region}"
            )
        model = str(
            self._settings.get("aliyun_wan_model", DEFAULT_MODEL) or DEFAULT_MODEL
        ).strip()
        if model != DEFAULT_MODEL:
            raise AliyunWanConfigurationError(
                f"Phase 1 supports only the {DEFAULT_MODEL} model"
            )

    def submit_video_task(
        self,
        *,
        prompt: str,
        duration: int,
        ratio: str,
    ) -> dict[str, str]:
        self._validate_configuration()
        if not 2 <= duration <= 15:
            raise ValueError("Wan video duration must be between 2 and 15 seconds")
        model = str(
            self._settings.get("aliyun_wan_model", DEFAULT_MODEL) or DEFAULT_MODEL
        ).strip()
        payload = {
            "model": model,
            "input": {"prompt": prompt},
            "parameters": {
                "resolution": str(
                    self._settings.get("aliyun_wan_resolution", "720P") or "720P"
                ),
                "ratio": ratio,
                "prompt_extend": bool(
                    self._settings.get("aliyun_wan_prompt_extend", True)
                ),
                "watermark": bool(self._settings.get("aliyun_wan_watermark", False)),
                "duration": duration,
            },
        }
        response = self._http.post(
            f"{self._base_url}/services/aigc/video-generation/video-synthesis",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if 400 <= response.status_code < 500:
            raise AliyunWanRequestRejectedError(
                f"Wan task submission was rejected with HTTP {response.status_code}"
            )
        if not 200 <= response.status_code < 300:
            raise AliyunWanError(
                f"Wan task submission failed with HTTP {response.status_code}"
            )
        body = response.json()
        output = body.get("output") or {}
        task_id = str(output.get("task_id") or "").strip()
        if not task_id:
            raise AliyunWanError("Wan task submission response did not include task_id")
        return {
            "task_id": task_id,
            "status": str(output.get("task_status") or "PENDING").upper(),
            "request_id": str(body.get("request_id") or ""),
        }

    def get_video_task(self, task_id: str) -> dict[str, str]:
        self._validate_configuration()
        response = self._http.get(
            f"{self._base_url}/tasks/{task_id}",
            headers=self._headers(),
            timeout=30,
        )
        if not 200 <= response.status_code < 300:
            raise AliyunWanError(
                f"Wan task status query failed with HTTP {response.status_code}"
            )
        body = response.json()
        output = body.get("output") or {}
        return {
            "task_id": str(output.get("task_id") or task_id),
            "status": str(output.get("task_status") or "UNKNOWN").upper(),
            "video_url": str(output.get("video_url") or ""),
            "request_id": str(body.get("request_id") or ""),
            "error_code": str(output.get("code") or body.get("code") or ""),
            "error_message": str(
                output.get("message") or body.get("message") or ""
            ),
        }
