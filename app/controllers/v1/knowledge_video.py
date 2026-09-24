from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.application.evidence_service import TaskEvidenceCommandService
from app.application.knowledge_retrieval_service import (
    KnowledgeProcessingService,
    KnowledgeRetrievalService,
)
from app.application.task_command_service import TaskCommandService
from app.application.task_query_service import TaskQueryService
from app.controllers import base
from app.controllers.v1.base import new_router
from app.domain.evidence import (
    EvidenceDomainError,
    EvidenceNotFoundError,
    SourceDocument,
    SourceStatus,
    SourceType,
    UnsupportedSourceTypeError,
)
import os
from pathlib import Path
from app.domain.workflow_state import (
    InvalidStateTransitionError,
    Stage,
    TerminalStateImmutableError,
    WorkflowConflictError,
    WorkflowPolicyType,
)
from app.persistence.repositories import EvidenceRepository
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
    allow_research: bool = Field(default=False)


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
                allow_research=body.allow_research,
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
    "/knowledge-video-tasks",
    summary="List Knowledge Video Tasks",
)
def list_tasks(limit: int = 50, offset: int = 0):
    """
    Lists recent KnowledgeVideoTasks ordered by creation time descending.
    """
    with get_session() as session:
        query_service = TaskQueryService(session)
        tasks = query_service.list_tasks(limit=limit, offset=offset)
        return utils.get_response(200, data=tasks, message="Tasks listed")


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


class AuthorizeResearchRequest(BaseModel):
    resume_if_waiting: bool = Field(
        default=True,
        description="Whether to automatically resume task if waiting in NEEDS_EVIDENCE",
    )


@router.post(
    "/knowledge-video-tasks/{task_id}/authorize-research",
    summary="Authorize open-web research augmentation for a Knowledge Video Task",
)
def authorize_research(task_id: str, body: AuthorizeResearchRequest | None = None):
    """
    Authorizes open-web research for a task. If the task is in NEEDS_EVIDENCE or WAITING_USER,
    it is automatically re-enqueued for an EVIDENCE stage retry.
    """
    resume = body.resume_if_waiting if body is not None else True
    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.authorize_research(task_id, resume_if_waiting=resume)
            session.commit()
            return utils.get_response(
                200,
                data={
                    "task_id": task.task_id,
                    "task_status": task.task_status.value,
                    "current_stage": task.current_stage.value,
                    "is_research_authorized": task.is_research_authorized,
                },
                message="Web research authorized",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (
            WorkflowConflictError,
            InvalidStateTransitionError,
            TerminalStateImmutableError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


class RegisterEvidenceRequest(BaseModel):
    source_type: SourceType = Field(..., description="Source type: TEXT, URL, FILE, or KNOWLEDGE_BASE")
    text_content: str | None = Field(default=None, description="Raw text for TEXT source type")
    url: str | None = Field(default=None, description="URL for URL source type")
    title: str | None = Field(default=None, description="Optional title or label for the source")
    author: str | None = Field(default=None, description="Optional author attribution")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata for the source")
    kb_id: str | None = Field(default=None, description="Knowledge Base ID for KNOWLEDGE_BASE source type")


@router.post(
    "/knowledge-video-tasks/{task_id}/evidence",
    status_code=201,
    summary="Register evidence source for a Knowledge Video Task",
)
def register_evidence(task_id: str, body: RegisterEvidenceRequest):
    """
    Registers an authoritative external source (TEXT, URL, etc.) and associates it with the task.
    """
    with get_session() as session:
        command_service = TaskEvidenceCommandService(session)
        try:
            if body.source_type == SourceType.TEXT:
                if not body.text_content:
                    raise HTTPException(status_code=400, detail="text_content is required for TEXT source type.")
                doc = command_service.add_text_source(
                    task_id=task_id,
                    text=body.text_content,
                    title=body.title,
                    author=body.author,
                    metadata=body.metadata,
                )
            elif body.source_type == SourceType.URL:
                if not body.url:
                    raise HTTPException(status_code=400, detail="url is required for URL source type.")
                doc = command_service.register_url_source(
                    task_id=task_id,
                    url=body.url,
                    title=body.title,
                    author=body.author,
                    metadata=body.metadata,
                )
            elif body.source_type == SourceType.KNOWLEDGE_BASE:
                if not body.kb_id:
                    raise HTTPException(status_code=400, detail="kb_id is required for KNOWLEDGE_BASE source type.")
                doc = command_service.register_knowledge_base_source(
                    task_id=task_id,
                    kb_id=body.kb_id,
                    metadata=body.metadata,
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported source type: {body.source_type}")

            session.commit()

            return utils.get_response(
                201,
                data={
                    "source_document_id": doc.source_document_id,
                    "source_type": doc.source_type.value,
                    "title": doc.title,
                    "source_locator": doc.source_locator,
                    "content_hash": doc.content_hash,
                    "source_fingerprint": doc.source_fingerprint,
                    "status": doc.status.value,
                    "media_type": doc.media_type,
                    "metadata": redact_sensitive_dict(doc.metadata_json),
                    "created_at": doc.created_at.isoformat(),
                },
                message="Evidence source registered",
            )
        except HTTPException:
            raise
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
        except TerminalStateImmutableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedSourceTypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


class ProcessKnowledgeRequest(BaseModel):
    source_document_id: str | None = Field(
        default=None,
        description="Optional specific source to process. If omitted, processes all registered sources for task.",
    )


class RetrieveKnowledgeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Max candidate chunks to return")
    source_scope_ids: list[str] | None = Field(
        default=None,
        description="Optional restrict retrieval to specific source document IDs",
    )


@router.post(
    "/knowledge-video-tasks/{task_id}/knowledge/process",
    summary="Parse and chunk registered evidence sources for a task",
)
def process_knowledge(task_id: str, body: ProcessKnowledgeRequest | None = None):
    with get_session() as session:
        proc_service = KnowledgeProcessingService(session)
        source_id = body.source_document_id if body else None
        try:
            if source_id:
                chunks = proc_service.process_source_document(source_id)
                data = {
                    "task_id": task_id,
                    "processed_sources": 1,
                    "total_chunks": len(chunks),
                }
            else:
                mapping = proc_service.process_sources_for_task(task_id)
                total_chunks = sum(len(c) for c in mapping.values())
                data = {
                    "task_id": task_id,
                    "processed_sources": len(mapping),
                    "total_chunks": total_chunks,
                }
            return utils.get_response(200, data=data, message="Knowledge sources processed")
        except EvidenceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvidenceDomainError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/knowledge-video-tasks/{task_id}/knowledge/retrieve",
    summary="Execute scoped lexical retrieval for a task",
)
def retrieve_knowledge(task_id: str, body: RetrieveKnowledgeRequest):
    with get_session() as session:
        retrieval_service = KnowledgeRetrievalService(session)
        try:
            snapshot = retrieval_service.retrieve(
                task_id=task_id,
                query=body.query,
                top_k=body.top_k,
                source_scope_ids=body.source_scope_ids,
            )
            return utils.get_response(
                200,
                data={
                    "retrieval_snapshot_id": snapshot.retrieval_snapshot_id,
                    "task_id": snapshot.task_id,
                    "query": snapshot.query,
                    "source_scope_ids": list(snapshot.source_scope_ids),
                    "candidate_count": len(snapshot.candidates),
                    "candidates": [c.model_dump() for c in snapshot.candidates],
                    "selected_evidence_ids": list(snapshot.selected_evidence_ids),
                    "content_fingerprint": snapshot.content_fingerprint,
                    "created_at": snapshot.created_at.isoformat(),
                },
                message="Retrieval executed",
            )
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg.lower():
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
        except EvidenceDomainError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/knowledge-video-tasks/{task_id}/knowledge/retrievals",
    summary="List retrieval snapshots for a task",
)
def list_task_retrievals(task_id: str):
    with get_session() as session:
        ev_repo = EvidenceRepository(session)
        snapshots = ev_repo.list_retrieval_snapshots_for_task(task_id)
        data = [
            {
                "retrieval_snapshot_id": s.retrieval_snapshot_id,
                "task_id": s.task_id,
                "query": s.query,
                "source_scope_ids": list(s.source_scope_ids),
                "candidate_count": len(s.candidates),
                "selected_evidence_ids": list(s.selected_evidence_ids),
                "content_fingerprint": s.content_fingerprint,
                "created_at": s.created_at.isoformat(),
            }
            for s in snapshots
        ]
        return utils.get_response(200, data=data, message="Retrieval snapshots listed")


@router.get(
    "/knowledge-video-tasks/{task_id}/knowledge/chunks",
    summary="List knowledge chunks for a task",
)
def list_task_chunks(task_id: str):
    with get_session() as session:
        ev_repo = EvidenceRepository(session)
        chunks = ev_repo.list_chunks_for_task(task_id)
        data = [
            {
                "chunk_id": c.chunk_id,
                "source_document_id": c.source_document_id,
                "chunk_index": c.chunk_index,
                "normalized_text": c.normalized_text,
                "locator": c.locator,
                "content_fingerprint": c.content_fingerprint,
                "created_at": c.created_at.isoformat(),
            }
            for c in chunks
        ]
        return utils.get_response(200, data=data, message="Chunks listed")


@router.get(
    "/knowledge-video-tasks/{task_id}/events",
    summary="Get execution trace events for a Knowledge Video Task",
)
def get_task_events(task_id: str):
    """
    Returns chronological domain trace events recorded for this task.
    """
    with get_session() as session:
        query_service = TaskQueryService(session)
        events = query_service.get_task_events(task_id)
        return utils.get_response(200, data=events, message="Events retrieved")


@router.get(
    "/knowledge-video-tasks/{task_id}/artifacts",
    summary="Get artifact references for a Knowledge Video Task",
)
def get_task_artifacts(task_id: str):
    """
    Returns all task artifact references (typed pointers to domain outputs) for this task.
    """
    with get_session() as session:
        query_service = TaskQueryService(session)
        artifacts = query_service.get_task_artifacts(task_id)
        return utils.get_response(200, data=artifacts, message="Artifacts retrieved")


class ReviseKnowledgeVideoTaskRequest(BaseModel):
    target_stage: Stage | None = Field(default=None, description="Stage to restart from (e.g. SCRIPT, STORYBOARD)")
    feedback: str | None = Field(default=None, description="User revision instructions or feedback")
    topic: str | None = Field(default=None, description="Updated topic")
    target_duration: float | None = Field(default=None, gt=0, description="Updated target duration in seconds")
    aspect_ratio: str | None = Field(default=None, description="Updated aspect ratio")


@router.post(
    "/knowledge-video-tasks/{task_id}/revise",
    summary="Revise or rerun a Knowledge Video Task",
)
def revise_task(task_id: str, body: ReviseKnowledgeVideoTaskRequest | None = None):
    """
    Applies user revisions or feedback and reruns the workflow from a designated stage.
    """
    target_stage = body.target_stage if body else None
    feedback = body.feedback if body else None
    topic = body.topic if body else None
    target_duration = body.target_duration if body else None
    aspect_ratio = body.aspect_ratio if body else None

    with get_session() as session:
        command_service = TaskCommandService(session)
        try:
            task = command_service.revise_task(
                task_id=task_id,
                target_stage=target_stage,
                feedback=feedback,
                topic=topic,
                target_duration=target_duration,
                aspect_ratio=aspect_ratio,
            )
            session.commit()
            return utils.get_response(
                200,
                data={
                    "task_id": task.task_id,
                    "task_status": task.task_status.value,
                    "current_stage": task.current_stage.value,
                    "is_partial_rerun": task.task_metadata.get("is_partial_rerun", False),
                },
                message="Task revision accepted and re-enqueued",
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkflowConflictError, InvalidStateTransitionError, TerminalStateImmutableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/knowledge-video-tasks/{task_id}/delivery",
    summary="Get final delivery manifest for a Knowledge Video Task",
)
def get_delivery_manifest(task_id: str):
    """
    Retrieves the authoritative delivery manifest for a completed task.
    """
    with get_session() as session:
        query_service = TaskQueryService(session)
        manifest = query_service.get_delivery_manifest(task_id)
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Delivery manifest not found for task '{task_id}'.",
            )
        return utils.get_response(200, data=manifest, message="Delivery manifest retrieved")


@router.get(
    "/knowledge-video-tasks/{task_id}/delivery/download",
    summary="Safely download delivery artifact file for a task",
)
def download_delivery_file(
    task_id: str,
    target: str = "video",
):
    """
    Downloads an authoritative delivery artifact for the task.
    Enforces path-traversal prevention and strict task-isolation safety.
    """
    from fastapi.responses import FileResponse
    from app.utils.file_security import resolve_path_within_directory

    with get_session() as session:
        query_service = TaskQueryService(session)
        manifest = query_service.get_delivery_manifest(task_id)
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Delivery manifest not found for task '{task_id}'.",
            )

    target_map = {
        "video": (manifest.get("final_video_path"), "video/mp4"),
        "subtitle": (manifest.get("subtitle_path"), "text/plain"),
        "source_report": (manifest.get("source_report_path"), "text/markdown"),
        "execution_report": (manifest.get("execution_report_path"), "text/markdown"),
    }

    target_info = target_map.get(target.lower())
    if target_info is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target '{target}'. Allowed: video, subtitle, source_report, execution_report.",
        )

    file_path, media_type = target_info
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact for target '{target}' does not exist for task '{task_id}'.",
        )

    # Validate file existence on disk
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Artifact file for target '{target}' not found on storage disk.",
        )

    # Path-traversal & task-isolation check:
    allowed_bases = [
        utils.task_dir(task_id),
        utils.task_dir(),
        utils.storage_dir(),
    ]
    resolved_path = None
    for base in allowed_bases:
        try:
            resolved_path = resolve_path_within_directory(base, file_path, require_file=True)
            break
        except ValueError:
            continue

    if resolved_path is None:
        real_file = os.path.realpath(file_path)
        if ".." in file_path or task_id not in real_file:
            raise HTTPException(
                status_code=403,
                detail="Access denied: file path violates task isolation or security constraints.",
            )
        resolved_path = real_file

    filename = Path(resolved_path).name
    return FileResponse(
        path=resolved_path,
        filename=filename,
        media_type=media_type,
    )

