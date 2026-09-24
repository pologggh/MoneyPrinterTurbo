"""Knowledge Video Agent HTTP API Client.

Architecture Rule:
This module must NEVER import from app.persistence, app.domain,
app.application, app.services, or sqlalchemy.
It communicates with the MoneyPrinterTurbo backend strictly via HTTP.
"""

from __future__ import annotations

import os
from typing import Any
import requests


class ApiClientError(Exception):
    """Exception raised for API request failures."""

    def __init__(self, message: str, status_code: int | None = None, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class KnowledgeVideoApiClient:
    """HTTP client for Knowledge Video Agent endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        raw_url = (
            base_url
            or os.environ.get("KNOWLEDGE_VIDEO_API_BASE_URL")
            or os.environ.get("API_BASE_URL")
            or "http://127.0.0.1:8080/api/v1"
        )
        self.base_url = raw_url.rstrip("/")
        self.api_key = api_key or os.environ.get("API_KEY")
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: Any = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Network error connecting to API: {exc}") from exc

        try:
            data = resp.json()
        except ValueError:
            data = {"raw": resp.text}

        if not resp.ok:
            error_detail = (
                data.get("detail")
                or data.get("message")
                or f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
            raise ApiClientError(
                str(error_detail),
                status_code=resp.status_code,
                response_data=data,
            )

        if isinstance(data, dict):
            if "data" in data:
                return data["data"]
            if "status" in data and 200 <= data["status"] < 300:
                return data.get("data", {})
        return data

    def create_task(
        self,
        topic: str,
        target_duration: float = 60.0,
        aspect_ratio: str = "16:9",
        language: str = "zh",
        workflow_policy: str = "AUTO",
        task_metadata: dict[str, Any] | None = None,
        allow_research: bool = False,
    ) -> dict[str, Any]:
        """Creates a new Knowledge Video Task."""
        payload = {
            "topic": topic,
            "target_duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "language": language,
            "workflow_policy": workflow_policy,
            "task_metadata": task_metadata or {},
            "allow_research": allow_research,
        }
        return self._request("POST", "/knowledge-video-tasks", json_data=payload)

    def list_tasks(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Lists recent Knowledge Video Tasks."""
        res = self._request("GET", "/knowledge-video-tasks", params={"limit": limit, "offset": offset})
        if isinstance(res, list):
            return res
        return []

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Gets detailed status and history for a Knowledge Video Task."""
        return self._request("GET", f"/knowledge-video-tasks/{task_id}")

    def approve_task(self, task_id: str) -> dict[str, Any]:
        """Approves a paused task in WAITING_USER to advance to next stage."""
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/approve")

    def retry_task(self, task_id: str, reason: str | None = None) -> dict[str, Any]:
        """Retries a task currently in NEEDS_RECOVERY or NEEDS_EVIDENCE."""
        payload = {"reason": reason} if reason else {}
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/retry", json_data=payload)

    def cancel_task(self, task_id: str, reason: str | None = None) -> dict[str, Any]:
        """Cancels an active or queued Knowledge Video Task."""
        payload = {"reason": reason} if reason else {}
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/cancel", json_data=payload)

    def authorize_research(self, task_id: str, resume_if_waiting: bool = True) -> dict[str, Any]:
        """Authorizes open-web research augmentation for the task."""
        payload = {"resume_if_waiting": resume_if_waiting}
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/authorize-research", json_data=payload)

    def register_evidence(
        self,
        task_id: str,
        source_type: str,
        text_content: str | None = None,
        url: str | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> dict[str, Any]:
        """Registers external evidence (TEXT, URL, KNOWLEDGE_BASE) to the task."""
        payload = {
            "source_type": source_type,
            "text_content": text_content,
            "url": url,
            "title": title,
            "author": author,
            "metadata": metadata or {},
            "kb_id": kb_id,
        }
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/evidence", json_data=payload)

    def process_knowledge(self, task_id: str, source_document_id: str | None = None) -> dict[str, Any]:
        """Processes registered evidence sources into knowledge chunks."""
        payload = {"source_document_id": source_document_id} if source_document_id else {}
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/knowledge/process", json_data=payload)

    def retrieve_knowledge(
        self,
        task_id: str,
        query: str,
        top_k: int = 10,
        source_scope_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Performs lexical search over processed knowledge chunks."""
        payload = {
            "query": query,
            "top_k": top_k,
            "source_scope_ids": source_scope_ids,
        }
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/knowledge/retrieve", json_data=payload)

    def list_task_retrievals(self, task_id: str) -> list[dict[str, Any]]:
        """Lists retrieval snapshots for the task."""
        res = self._request("GET", f"/knowledge-video-tasks/{task_id}/knowledge/retrievals")
        if isinstance(res, list):
            return res
        return []

    def list_task_chunks(self, task_id: str) -> list[dict[str, Any]]:
        """Lists processed chunks for the task."""
        res = self._request("GET", f"/knowledge-video-tasks/{task_id}/knowledge/chunks")
        if isinstance(res, list):
            return res
        return []

    def get_task_events(self, task_id: str) -> list[dict[str, Any]]:
        """Gets chronological execution trace events for the task."""
        res = self._request("GET", f"/knowledge-video-tasks/{task_id}/events")
        if isinstance(res, list):
            return res
        return []

    def get_task_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """Gets all task artifact references for the task."""
        res = self._request("GET", f"/knowledge-video-tasks/{task_id}/artifacts")
        if isinstance(res, list):
            return res
        return []

    def revise_task(
        self,
        task_id: str,
        target_stage: str | None = None,
        feedback: str | None = None,
        topic: str | None = None,
        target_duration: float | None = None,
        aspect_ratio: str | None = None,
    ) -> dict[str, Any]:
        """Applies feedback or revisions and reruns workflow from target stage."""
        payload: dict[str, Any] = {}
        if target_stage:
            payload["target_stage"] = target_stage
        if feedback:
            payload["feedback"] = feedback
        if topic:
            payload["topic"] = topic
        if target_duration is not None:
            payload["target_duration"] = target_duration
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        return self._request("POST", f"/knowledge-video-tasks/{task_id}/revise", json_data=payload)

    def get_delivery_manifest(self, task_id: str) -> dict[str, Any]:
        """Gets the authoritative delivery manifest for a completed task."""
        return self._request("GET", f"/knowledge-video-tasks/{task_id}/delivery")

    def get_download_url(self, task_id: str, target: str = "video") -> str:
        """Returns the download URL for a delivery artifact."""
        return f"{self.base_url}/knowledge-video-tasks/{task_id}/delivery/download?target={target}"

    def download_delivery_file(self, task_id: str, target: str = "video") -> bytes:
        """Downloads the binary or text content of a delivery artifact."""
        url = self.get_download_url(task_id, target=target)
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"Failed to download delivery artifact '{target}': {exc}") from exc
