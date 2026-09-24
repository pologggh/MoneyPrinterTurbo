from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import re
from typing import Any, Sequence
from uuid import uuid4

from sqlalchemy.orm import Session

from app.application.knowledge_retrieval_service import KnowledgeProcessingService
from app.config import config
from app.domain.evidence import (
    DocumentTextNotExtractableError,
    KnowledgeChunk,
    SourceDocument,
    SourceStatus,
    SourceType,
    compute_sha256,
    compute_source_fingerprint,
)
from app.domain.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseStatus,
)
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

MIME_MAP = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


class KnowledgeBaseServiceError(Exception):
    """Base exception for KnowledgeBaseService operations."""


class KnowledgeBaseNotFoundError(KnowledgeBaseServiceError):
    """Raised when a Knowledge Base cannot be found."""


class InvalidFileFormatError(KnowledgeBaseServiceError):
    """Raised when an unsupported file format is uploaded."""


class FileSizeExceededError(KnowledgeBaseServiceError):
    """Raised when an uploaded file exceeds size limits."""


class SecurityValidationError(KnowledgeBaseServiceError):
    """Raised when filename or path validation detects security risks."""


class KnowledgeBaseArchivedError(KnowledgeBaseServiceError):
    """Raised when an operation cannot be performed because a Knowledge Base is archived."""


class KnowledgeBaseAssociationNotFoundError(KnowledgeBaseServiceError):
    """Raised when an association between a task and a Knowledge Base is not found."""


def sanitize_filename(filename: str) -> str:
    """Sanitizes an uploaded filename to prevent directory traversal and illegal characters."""
    if not filename:
        return "unnamed_document.txt"

    # Strip null bytes
    clean = filename.replace("\0", "")

    # Strip directory components (both / and \)
    clean = os.path.basename(clean)
    clean = Path(clean).name

    # Check for path traversal attempt sequences
    if ".." in clean or "/" in clean or "\\" in clean:
        raise SecurityValidationError(f"Path traversal detected in filename: '{filename}'")

    # Replace forbidden Windows / UNIX characters: < > : " / \ | ? *
    clean = re.sub(r'[<>:"/\\|?*]', "_", clean).strip()

    if not clean or clean.startswith("."):
        clean = f"document{clean if clean.startswith('.') else '.txt'}"

    return clean


class KnowledgeBaseCommandService:
    """Application service for managing Knowledge Bases and ingesting multipart documents."""

    def __init__(
        self,
        session: Session,
        storage_dir: Path | None = None,
        processing_service: KnowledgeProcessingService | None = None,
    ) -> None:
        self._session = session
        self.kb_repo = KnowledgeBaseRepository(session)
        self.ev_repo = EvidenceRepository(session)
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.processing_service = processing_service or KnowledgeProcessingService(session)

        base_dir = storage_dir or (Path(config.root_dir) / "storage" / "evidence_sources")
        self.storage_dir = Path(base_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_knowledge_base(
        self,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        """Creates a new active Knowledge Base."""
        kb = KnowledgeBase.create(
            name=name,
            description=description,
            metadata_json=metadata,
        )
        saved = self.kb_repo.save_knowledge_base(kb)
        self._session.commit()
        return saved

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Retrieves a Knowledge Base or raises KnowledgeBaseNotFoundError."""
        kb = self.kb_repo.get_knowledge_base(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge Base '{kb_id}' not found.")
        return kb

    def list_knowledge_bases(self, status: str | None = None) -> list[KnowledgeBase]:
        """Lists all Knowledge Bases, optionally filtered by status."""
        return self.kb_repo.list_knowledge_bases(status=status)

    def update_knowledge_base(
        self,
        kb_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBase:
        """Updates name and/or description of a Knowledge Base."""
        kb = self.get_knowledge_base(kb_id)
        updated_dict: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ValueError("Knowledge Base name cannot be empty.")
            updated_dict["name"] = cleaned
        if description is not None:
            updated_dict["description"] = description.strip() or None

        updated_kb = kb.model_copy(update=updated_dict)
        saved = self.kb_repo.save_knowledge_base(updated_kb)
        self._session.commit()
        return saved

    def archive_knowledge_base(self, kb_id: str) -> None:
        """Soft-deletes / archives a Knowledge Base."""
        kb = self.get_knowledge_base(kb_id)
        self.kb_repo.archive_knowledge_base(kb.knowledge_base_id)
        self._session.commit()

    def upload_document(
        self,
        kb_id: str,
        file_bytes: bytes,
        filename: str,
        title: str | None = None,
    ) -> SourceDocument:
        """Securely validates, stores, parses, chunks, embeds, and attaches a document to a Knowledge Base."""
        kb = self.get_knowledge_base(kb_id)
        if kb.status != KnowledgeBaseStatus.ACTIVE:
            raise KnowledgeBaseArchivedError(f"Cannot upload document to archived Knowledge Base '{kb_id}'.")

        # 1. Size Validation
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise FileSizeExceededError(
                f"File size ({len(file_bytes)} bytes) exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes (50MB)."
            )

        # 2. Filename & Extension Sanitization
        clean_name = sanitize_filename(filename)
        ext = os.path.splitext(clean_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileFormatError(
                f"Unsupported file format '{ext}'. Allowed formats: {sorted(ALLOWED_EXTENSIONS)}"
            )

        # 3. Content Hash & Deduplication check
        c_hash = compute_sha256(file_bytes)
        existing_sources = self.kb_repo.list_sources_for_kb(kb_id)
        for existing in existing_sources:
            if existing.content_hash == c_hash:
                # Document already exists in this KB
                return existing

        # 4. Generate persistent SourceDocument record & write to storage
        doc_id = f"src_{uuid4().hex[:24]}"
        stored_file_path = self.storage_dir / f"{doc_id}_{clean_name}"

        try:
            with open(stored_file_path, "wb") as f:
                f.write(file_bytes)
        except Exception as exc:
            raise KnowledgeBaseServiceError(f"Failed to write file to storage: {exc}") from exc

        media_type = MIME_MAP.get(ext, "application/octet-stream")
        source_locator = str(stored_file_path.as_posix())
        fingerprint = compute_source_fingerprint(SourceType.FILE, source_locator, c_hash)

        doc = SourceDocument(
            source_document_id=doc_id,
            source_type=SourceType.FILE,
            title=title.strip() if title else clean_name,
            source_locator=source_locator,
            content_snapshot=None,
            content_hash=c_hash,
            source_fingerprint=fingerprint,
            media_type=media_type,
            status=SourceStatus.REGISTERED,
            metadata_json={"original_filename": clean_name, "file_size_bytes": len(file_bytes)},
        )

        try:
            self.ev_repo.save_source_document(doc)
            self.kb_repo.associate_source(kb_id, doc_id)
            self._session.commit()
        except Exception as exc:
            # Transactional cleanup on disk if DB save fails
            if stored_file_path.exists():
                stored_file_path.unlink(missing_ok=True)
            self._session.rollback()
            raise KnowledgeBaseServiceError(f"Database error registering source document: {exc}") from exc

        # 5. Process document (parse -> chunk -> embed)
        try:
            self.processing_service.process_source_document(doc_id)
        except DocumentTextNotExtractableError as exc:
            # Expected graceful failure for scanned / image-only PDFs
            pass
        except Exception as exc:
            # Other errors already set SourceStatus.FAILED in processing_service
            pass

        # Return latest persisted state
        updated_doc = self.ev_repo.get_source_document(doc_id)
        return updated_doc or doc

    def list_documents(self, kb_id: str) -> list[dict[str, Any]]:
        """Lists all documents associated with a Knowledge Base with chunk count and status."""
        kb = self.get_knowledge_base(kb_id)
        sources = self.kb_repo.list_sources_for_kb(kb.knowledge_base_id)
        result: list[dict[str, Any]] = []

        for src in sources:
            chunks = self.ev_repo.list_chunks_for_source(src.source_document_id)
            err = src.metadata_json.get("processing_error")
            result.append({
                "source_document_id": src.source_document_id,
                "title": src.title,
                "status": src.status.value if hasattr(src.status, "value") else str(src.status),
                "source_type": src.source_type.value if hasattr(src.source_type, "value") else str(src.source_type),
                "media_type": src.media_type,
                "chunk_count": len(chunks),
                "error_message": err,
                "created_at": src.created_at.isoformat() if src.created_at else None,
                "content_hash": src.content_hash,
            })
        return result

    def retry_document(self, kb_id: str, source_document_id: str) -> tuple[KnowledgeChunk, ...]:
        """Retries parsing, chunking, and embedding for a failed or registered document."""
        kb = self.get_knowledge_base(kb_id)
        doc = self.ev_repo.get_source_document(source_document_id)
        if doc is None:
            raise KnowledgeBaseServiceError(f"Document '{source_document_id}' not found.")

        # Verify document belongs to KB
        sources = self.kb_repo.list_sources_for_kb(kb_id)
        if not any(s.source_document_id == source_document_id for s in sources):
            raise KnowledgeBaseServiceError(
                f"Document '{source_document_id}' is not associated with Knowledge Base '{kb_id}'."
            )

        return self.processing_service.process_source_document(source_document_id)

    def attach_to_task(self, task_id: str, kb_id: str) -> None:
        """Attaches a Knowledge Base to a task."""
        kb = self.get_knowledge_base(kb_id)
        if kb.status != KnowledgeBaseStatus.ACTIVE:
            raise KnowledgeBaseArchivedError(f"Cannot attach archived Knowledge Base '{kb_id}' to task.")
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise KnowledgeBaseNotFoundError(f"Task '{task_id}' not found.")

        self.kb_repo.attach_to_task(task_id, kb_id)
        self._session.commit()

    def detach_from_task(self, task_id: str, kb_id: str) -> None:
        """Detaches a Knowledge Base from a task."""
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise KnowledgeBaseNotFoundError(f"Task '{task_id}' not found.")
        kb = self.get_knowledge_base(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge Base '{kb_id}' not found.")

        attached_kbs = self.kb_repo.list_kbs_for_task(task_id)
        if not any(k.knowledge_base_id == kb_id for k in attached_kbs):
            raise KnowledgeBaseAssociationNotFoundError(
                f"Knowledge Base '{kb_id}' is not attached to task '{task_id}'."
            )

        self.kb_repo.detach_from_task(task_id, kb_id)
        self._session.commit()

    def list_task_knowledge_bases(self, task_id: str) -> list[KnowledgeBase]:
        """Lists all Knowledge Bases attached to a task."""
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise KnowledgeBaseNotFoundError(f"Task '{task_id}' not found.")
        return self.kb_repo.list_kbs_for_task(task_id)
