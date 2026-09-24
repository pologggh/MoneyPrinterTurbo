from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.application.knowledge_base_service import (
    FileSizeExceededError,
    InvalidFileFormatError,
    KnowledgeBaseArchivedError,
    KnowledgeBaseAssociationNotFoundError,
    KnowledgeBaseCommandService,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseServiceError,
    MAX_FILE_SIZE_BYTES,
    SecurityValidationError,
)
from app.controllers import base
from app.controllers.v1.base import new_router
from app.domain.knowledge_base import KnowledgeBaseStatus
from app.persistence.session import get_session
from app.utils import utils

CHUNK_SIZE = 64 * 1024  # 64KB
MAX_UPLOAD_SIZE_BYTES: int = MAX_FILE_SIZE_BYTES

ALLOWED_MIME_TYPES_BY_EXT: dict[str, set[str]] = {
    ".txt": {"text/plain", "application/octet-stream", "text/*"},
    ".md": {
        "text/markdown",
        "text/plain",
        "text/x-markdown",
        "application/octet-stream",
    },
    ".pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
        "application/msword",
        "application/octet-stream",
    },
    ".html": {"text/html", "application/xhtml+xml", "text/plain", "application/octet-stream"},
}

router = new_router(dependencies=[Depends(base.verify_token)])


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Name of the knowledge base")
    description: str | None = Field(default=None, max_length=2000, description="Optional description")
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateKnowledgeBaseRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


@router.post(
    "/knowledge-bases",
    status_code=201,
    summary="Create a new Knowledge Base",
)
def create_knowledge_base(body: CreateKnowledgeBaseRequest):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            kb = service.create_knowledge_base(
                name=body.name,
                description=body.description,
                metadata=body.metadata,
            )
            data = {
                "knowledge_base_id": kb.knowledge_base_id,
                "name": kb.name,
                "description": kb.description,
                "status": kb.status.value,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
            }
            return utils.get_response(201, data=data, message="Knowledge base created successfully")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/knowledge-bases",
    summary="List all Knowledge Bases",
)
def list_knowledge_bases(
    status: KnowledgeBaseStatus | None = Query(None, description="Optional status filter: ACTIVE or ARCHIVED")
):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        kbs = service.list_knowledge_bases(status=status.value if status else None)
        results = []
        for kb in kbs:
            sources = service.kb_repo.list_sources_for_kb(kb.knowledge_base_id)
            results.append({
                "knowledge_base_id": kb.knowledge_base_id,
                "name": kb.name,
                "description": kb.description,
                "status": kb.status.value,
                "document_count": len(sources),
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
            })
        return utils.get_response(200, data={"items": results, "total": len(results)}, message="Knowledge bases listed")


@router.get(
    "/knowledge-bases/{kb_id}",
    summary="Get details of a Knowledge Base",
)
def get_knowledge_base(kb_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            kb = service.get_knowledge_base(kb_id)
            sources = service.kb_repo.list_sources_for_kb(kb_id)
            data = {
                "knowledge_base_id": kb.knowledge_base_id,
                "name": kb.name,
                "description": kb.description,
                "status": kb.status.value,
                "document_count": len(sources),
                "metadata_json": kb.metadata_json,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
            }
            return utils.get_response(200, data=data, message="Knowledge base retrieved")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch(
    "/knowledge-bases/{kb_id}",
    summary="Update a Knowledge Base",
)
def update_knowledge_base(kb_id: str, body: UpdateKnowledgeBaseRequest):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            kb = service.update_knowledge_base(
                kb_id=kb_id,
                name=body.name,
                description=body.description,
            )
            data = {
                "knowledge_base_id": kb.knowledge_base_id,
                "name": kb.name,
                "description": kb.description,
                "status": kb.status.value,
                "updated_at": kb.updated_at.isoformat(),
            }
            return utils.get_response(200, data=data, message="Knowledge base updated successfully")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/knowledge-bases/{kb_id}",
    summary="Archive / soft-delete a Knowledge Base",
)
def archive_knowledge_base(kb_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            service.archive_knowledge_base(kb_id)
            data = {"status": "ARCHIVED", "knowledge_base_id": kb_id}
            return utils.get_response(200, data=data, message="Knowledge Base archived successfully.")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _read_upload_with_limit(
    file: UploadFile,
    max_size: int,
    chunk_size: int,
) -> bytes:
    """Reads an uploaded file stream chunk-by-chunk up to max_size bytes.

    Immediately halts reading upon exceeding max_size and raises HTTPException(413).
    Guarantees no further chunks are read past the limit.
    """
    chunks: list[bytes] = []
    total_size = 0
    read_size = min(chunk_size, max(1, max_size))
    while True:
        try:
            chunk = await file.read(read_size)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {max_size} bytes.",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return content


@router.post(
    "/knowledge-bases/{kb_id}/documents",
    status_code=201,
    summary="Upload a document into a Knowledge Base",
)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(None),
):
    try:
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if not ext or ext not in ALLOWED_MIME_TYPES_BY_EXT:
            allowed_exts = ", ".join(sorted(ALLOWED_MIME_TYPES_BY_EXT.keys()))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {allowed_exts}.",
            )

        # Compatibility policy:
        # 1. Missing or empty content_type is accepted as a compatibility fallback (e.g. CLI/curl tools).
        # 2. 'application/octet-stream' is accepted as generic binary fallback for all supported extensions.
        # True document content and structure are authoritatively validated downstream by DocumentParser.
        # However, explicit conflicting MIME declarations (e.g. .pdf declared as image/png) are rejected immediately.
        if file.content_type:
            raw_mime = file.content_type.lower().split(";")[0].strip()
            allowed_mimes = ALLOWED_MIME_TYPES_BY_EXT[ext]
            matched = (
                raw_mime in allowed_mimes
                or ("text/*" in allowed_mimes and raw_mime.startswith("text/"))
            )
            if not matched:
                raise HTTPException(
                    status_code=400,
                    detail=f"MIME type '{file.content_type}' conflicts with file extension '{ext}'.",
                )

        content = await _read_upload_with_limit(
            file=file,
            max_size=MAX_UPLOAD_SIZE_BYTES,
            chunk_size=CHUNK_SIZE,
        )

        with get_session() as session:
            service = KnowledgeBaseCommandService(session)
            try:
                doc = service.upload_document(
                    kb_id=kb_id,
                    file_bytes=content,
                    filename=filename,
                    title=title,
                )
                doc_chunks = service.ev_repo.list_chunks_for_source(doc.source_document_id)
                data = {
                    "source_document_id": doc.source_document_id,
                    "knowledge_base_id": kb_id,
                    "title": doc.title,
                    "status": doc.status.value,
                    "media_type": doc.media_type,
                    "chunk_count": len(doc_chunks),
                    "error_message": doc.metadata_json.get("processing_error"),
                    "created_at": doc.created_at.isoformat(),
                }
                return utils.get_response(201, data=data, message="Document uploaded successfully")
            except KnowledgeBaseNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except KnowledgeBaseArchivedError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except InvalidFileFormatError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except FileSizeExceededError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except SecurityValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except KnowledgeBaseServiceError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get(
    "/knowledge-bases/{kb_id}/documents",
    summary="List all documents in a Knowledge Base",
)
def list_documents(kb_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            docs = service.list_documents(kb_id)
            return utils.get_response(200, data={"items": docs, "total": len(docs)}, message="Documents listed")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/knowledge-bases/{kb_id}/documents/{doc_id}/retry",
    summary="Retry document parsing, chunking, and embedding",
)
def retry_document(kb_id: str, doc_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            chunks = service.retry_document(kb_id, doc_id)
            doc = service.ev_repo.get_source_document(doc_id)
            data = {
                "source_document_id": doc_id,
                "knowledge_base_id": kb_id,
                "status": doc.status.value if doc else "UNKNOWN",
                "chunk_count": len(chunks),
                "error_message": doc.metadata_json.get("processing_error") if doc else None,
            }
            return utils.get_response(200, data=data, message="Document processing retried")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KnowledgeBaseServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/tasks/{task_id}/knowledge-bases/{kb_id}",
    summary="Attach a Knowledge Base to a Knowledge Video Task",
)
def attach_knowledge_base_to_task(task_id: str, kb_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            service.attach_to_task(task_id, kb_id)
            data = {"status": "ATTACHED", "task_id": task_id, "knowledge_base_id": kb_id}
            return utils.get_response(200, data=data, message="Knowledge base attached to task")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KnowledgeBaseArchivedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KnowledgeBaseServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/tasks/{task_id}/knowledge-bases/{kb_id}",
    summary="Detach a Knowledge Base from a Knowledge Video Task",
)
def detach_knowledge_base_from_task(task_id: str, kb_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            service.detach_from_task(task_id, kb_id)
            data = {"status": "DETACHED", "task_id": task_id, "knowledge_base_id": kb_id}
            return utils.get_response(200, data=data, message="Knowledge base detached from task")
        except (KnowledgeBaseNotFoundError, KnowledgeBaseAssociationNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except KnowledgeBaseServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/tasks/{task_id}/knowledge-bases",
    summary="List all Knowledge Bases attached to a Knowledge Video Task",
)
def list_task_knowledge_bases(task_id: str):
    with get_session() as session:
        service = KnowledgeBaseCommandService(session)
        try:
            kbs = service.list_task_knowledge_bases(task_id)
            results = []
            for kb in kbs:
                sources = service.kb_repo.list_sources_for_kb(kb.knowledge_base_id)
                results.append({
                    "knowledge_base_id": kb.knowledge_base_id,
                    "name": kb.name,
                    "description": kb.description,
                    "status": kb.status.value,
                    "document_count": len(sources),
                    "created_at": kb.created_at.isoformat(),
                })
            return utils.get_response(200, data={"items": results, "total": len(results)}, message="Task knowledge bases listed")
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
