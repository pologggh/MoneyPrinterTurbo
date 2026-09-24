from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.application.task_command_service import TaskCommandService
from app.application.task_query_service import TaskQueryService
from app.controllers import base
from app.controllers.v1.base import new_router
from app.domain.workflow_state import (
    InvalidStateTransitionError,
    TerminalStateImmutableError,
    WorkflowConflictError,
    WorkflowPolicyType,
)
from app.persistence.session import get_session
from app.utils import utils

router = new_router(dependencies=[Depends(base.verify_token)])

SENSITIVE_KEY_SUBSTRINGS = {"api_key", "secret", "token", "password", "authorization"}


def redact_sensitive_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redacts sensitive keys such as API keys or secret tokens."""
    redacted: dict[str, Any] = {}
    for k, v in data.items():
        k_lower = k.lower()
        if any(sub in k_lower for sub in SENSITIVE_KEY_SUBSTRINGS):
            redacted[k] = "***"
        elif isinstance(v, dict):
            redacted[k] = redact_sensitive_dict(v)
        elif isinstance(v, list):
            redacted[k] = [
                redact_sensitive_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            redacted[k] = v
    return redacted


class CreateKnowledgeVideoTaskRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    target_duration: float = Field(default=60.0, gt=0.0, le=3600.0)
    aspect_ratio: str = Field(default="16:9")
    language: str = Field(default="zh")
    workflow_policy: WorkflowPolicyType = Field(default=WorkflowPolicyType.AUTO)
    task_metadata: dict[str, Any] = Field(default_factory=dict)


class RetryKnowledgeVideoTaskRequest(BaseModel):
    reason: str | None = None


class CancelKnowledgeVideoTaskRequest(BaseModel):
    reason: str | None = None


@router.post(
    "/knowledge-video-tasks",
    status_code=202,
    summary="Create a new Knowledge Video Task",
)
def create_task(body: CreateKnowledgeVideoTaskRequest):
    """
    Creates an authoritative KnowledgeVideoTask and enqueues its initial EVIDENCE job.
    Returns 202 Accepted.
    """
    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.create_task(
                topic=body.topic,
                target_duration=body.target_duration,
                aspect_ratio=body.aspect_ratio,
                language=body.language,
                workflow_policy=body.workflow_policy,
                task_metadata=body.task_metadata,
            )
            # Find initial job
            from app.persistence.repositories import WorkflowJobRepository
            job_repo = WorkflowJobRepository(session)
            current_job = job_repo.get_current_job_for_task(task.task_id)

            session.commit()

            return utils.get_response(
                202,
                data={
                    "task_id": task.task_id,
                    "topic": task.topic,
                    "task_status": task.task_status.value,
                    "current_stage": task.current_stage.value,
                    "workflow_policy": task.workflow_policy.value,
                    "initial_job_id": current_job.job_id if current_job else None,
                    "created_at": task.created_at.isoformat(),
                },
                message="Task accepted",
            )
        except Exception as exc:
            logger.error(f"Failed to create KnowledgeVideoTask: {exc}")
            raise


@router.get(
    "/knowledge-video-tasks/{task_id}",
    summary="Get Knowledge Video Task details",
)
def get_task(task_id: str):
    """
    Retrieves current state, active job, and execution history for a KnowledgeVideoTask.
    """
    with get_session() as session:
        query_service = TaskQueryService(session)
        detail = query_service.get_task_detail(task_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

        # Ensure credentials in metadata are redacted
        if "task_metadata" in detail and isinstance(detail["task_metadata"], dict):
            detail["task_metadata"] = redact_sensitive_dict(detail["task_metadata"])

        return utils.get_response(200, data=detail)


@router.post(
    "/knowledge-video-tasks/{task_id}/approve",
    summary="Approve a paused Knowledge Video Task",
)
def approve_task(task_id: str):
    """
    Approves a task paused in WAITING_USER, advancing it to the next stage and queuing its job.
    """
    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.approve_task(task_id)
            session.commit()
            return utils.get_response(
                200,
                data={
                    "task_id": task.task_id,
                    "task_status": task.task_status.value,
                    "current_stage": task.current_stage.value,
                },
                message="Task approved and advanced",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkflowConflictError, InvalidStateTransitionError, TerminalStateImmutableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/knowledge-video-tasks/{task_id}/retry",
    summary="Retry a failed or recoverable Knowledge Video Task",
)
def retry_task(task_id: str, body: RetryKnowledgeVideoTaskRequest | None = None):
    """
    Retries a task currently in NEEDS_RECOVERY or NEEDS_EVIDENCE.
    """
    reason = body.reason if body else None
    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.retry_task(task_id, reason=reason)
            session.commit()
            return utils.get_response(
                200,
                data={
                    "task_id": task.task_id,
                    "task_status": task.task_status.value,
                    "current_stage": task.current_stage.value,
                },
                message="Task retried",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkflowConflictError, InvalidStateTransitionError, TerminalStateImmutableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/knowledge-video-tasks/{task_id}/cancel",
    summary="Cancel a Knowledge Video Task",
)
def cancel_task(task_id: str, body: CancelKnowledgeVideoTaskRequest | None = None):
    """
    Cancels an active or queued Knowledge Video Task and aborts active jobs.
    """
    reason = body.reason if body else None
    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.cancel_task(task_id, reason=reason)
            session.commit()
            return utils.get_response(
                200,
                data={
                    "task_id": task.task_id,
                    "task_status": task.task_status.value,
                },
                message="Task cancelled",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkflowConflictError, InvalidStateTransitionError, TerminalStateImmutableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
