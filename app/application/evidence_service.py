from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app.domain.evidence import (
    ClaimType,
    EvidenceDomainError,
    EvidenceItem,
    EvidenceNotFoundError,
    EvidenceSnapshot,
    KnowledgeClaim,
    SourceDocument,
    SourceStatus,
    SourceType,
    UnsupportedSourceTypeError,
    compute_sha256,
)
from app.application.knowledge_base_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseServiceError,
)
from app.domain.knowledge_base import KnowledgeBaseStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_state import (
    ArtifactType,
    Stage,
    TerminalStateImmutableError,
)
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
    TaskArtifactRepository,
)
from app.utils.utils import storage_dir


class EvidenceResolutionService:
    """Service for resolving lightweight evidence references to full provenance records."""

    def __init__(self, session: Session) -> None:
        self._repo = EvidenceRepository(session)

    def resolve_evidence(self, evidence_id: str) -> tuple[EvidenceItem, SourceDocument]:
        """Resolves a single evidence_id to its EvidenceItem and originating SourceDocument.

        Raises EvidenceNotFoundError if evidence_id or source document does not exist.
        """
        item = self._repo.get_evidence_item(evidence_id)
        if item is None:
            raise EvidenceNotFoundError(f"Evidence '{evidence_id}' could not be resolved.")
        doc = self._repo.get_source_document(item.source_document_id)
        if doc is None:
            raise EvidenceNotFoundError(
                f"SourceDocument '{item.source_document_id}' not found for evidence '{evidence_id}'."
            )
        return item, doc

    def resolve_evidence_refs(
        self, evidence_refs: Sequence[str]
    ) -> dict[str, tuple[EvidenceItem, SourceDocument]]:
        """Resolves a collection of evidence reference IDs.

        Returns a mapping of evidence_id -> (EvidenceItem, SourceDocument).
        Raises EvidenceNotFoundError if any reference cannot be resolved.
        """
        results: dict[str, tuple[EvidenceItem, SourceDocument]] = {}
        missing: list[str] = []
        for ref in evidence_refs:
            try:
                results[ref] = self.resolve_evidence(ref)
            except EvidenceNotFoundError:
                missing.append(ref)
        if missing:
            raise EvidenceNotFoundError(
                f"Failed to resolve {len(missing)} evidence reference(s): {', '.join(missing)}"
            )
        return results


class TaskEvidenceCommandService:
    """Application service for registering and managing evidence sources for KnowledgeVideoTasks."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.evidence_repo = EvidenceRepository(session)
        self.artifact_repo = TaskArtifactRepository(session)
        self.kb_repo = KnowledgeBaseRepository(session)

    def _assert_task_modifiable(self, task_id: str) -> KnowledgeVideoTask:
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found")
        if task.task_status.is_terminal:
            raise TerminalStateImmutableError(
                f"Cannot register evidence for task '{task_id}' in terminal state '{task.task_status.value}'."
            )
        return task

    def add_text_source(
        self,
        task_id: str,
        text: str,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        self._assert_task_modifiable(task_id)
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text content cannot be empty.")
        doc = SourceDocument.create_text(
            text=clean_text,
            title=title,
            author=author,
            metadata_json=metadata,
            now=now,
        )
        saved_doc = self.evidence_repo.save_source_document(doc)
        self.evidence_repo.associate_task_source(task_id, saved_doc.source_document_id, role="PRIMARY", now=now)
        self._session.flush()
        return saved_doc

    def register_file_source(
        self,
        task_id: str,
        filename: str,
        content: bytes,
        media_type: str = "application/octet-stream",
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        self._assert_task_modifiable(task_id)
        if not content:
            raise ValueError("File content cannot be empty.")

        target_dir = storage_dir("evidence_sources", create=True)
        os.makedirs(target_dir, exist_ok=True)
        file_hash = compute_sha256(content)
        safe_name = os.path.basename(filename).replace(" ", "_")
        storage_filename = f"{file_hash[:16]}_{safe_name}"
        storage_path = os.path.join(target_dir, storage_filename)

        with open(storage_path, "wb") as f:
            f.write(content)

        relative_path = os.path.join("storage", "evidence_sources", storage_filename)

        doc = SourceDocument.register_file(
            filename=filename,
            file_bytes=content,
            media_type=media_type,
            storage_path=relative_path,
            title=title,
            author=author,
            metadata_json=metadata,
            now=now,
        )
        saved_doc = self.evidence_repo.save_source_document(doc)
        self.evidence_repo.associate_task_source(task_id, saved_doc.source_document_id, role="PRIMARY", now=now)
        self._session.flush()
        return saved_doc

    def register_url_source(
        self,
        task_id: str,
        url: str,
        title: str | None = None,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> SourceDocument:
        self._assert_task_modifiable(task_id)
        norm_url = url.strip()
        if not (norm_url.startswith("http://") or norm_url.startswith("https://")):
            raise ValueError(f"Invalid URL '{url}'. Must start with http:// or https://")

        doc = SourceDocument.register_url(
            url=norm_url,
            title=title,
            author=author,
            metadata_json=metadata,
            now=now,
        )
        saved_doc = self.evidence_repo.save_source_document(doc)
        self.evidence_repo.associate_task_source(task_id, saved_doc.source_document_id, role="PRIMARY", now=now)
        self._session.flush()
        return saved_doc

    def register_knowledge_base_source(
        self,
        task_id: str,
        kb_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._assert_task_modifiable(task_id)
        kb = self.kb_repo.get_knowledge_base(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(f"Knowledge Base '{kb_id}' not found.")
        if kb.status != KnowledgeBaseStatus.ACTIVE:
            raise KnowledgeBaseServiceError(
                f"Cannot attach archived Knowledge Base '{kb_id}' to task."
            )
        self.kb_repo.attach_to_task(task_id, kb_id)
        self._session.flush()

    def create_task_evidence_snapshot(
        self,
        task_id: str,
        knowledge_claims: Sequence[KnowledgeClaim] = (),
        now: datetime | None = None,
    ) -> tuple[EvidenceSnapshot, TaskArtifactRef]:
        """Creates an immutable EvidenceSnapshot of all currently associated sources and items for the task."""
        ts = now or datetime.now(UTC)
        self._assert_task_modifiable(task_id)

        sources = self.evidence_repo.list_sources_for_task(task_id)
        source_ids = [s.source_document_id for s in sources]

        evidence_ids: list[str] = []
        for s in sources:
            items = self.evidence_repo.list_evidence_items_for_source(s.source_document_id)
            evidence_ids.extend([item.evidence_id for item in items])

        claim_ids: list[str] = []
        for c in knowledge_claims:
            saved_claim = self.evidence_repo.save_knowledge_claim(c)
            claim_ids.append(saved_claim.knowledge_claim_id)

        latest_snap = self.evidence_repo.get_latest_snapshot_for_task(task_id)
        next_version = (latest_snap.snapshot_version + 1) if latest_snap else 1

        snapshot = EvidenceSnapshot.create(
            task_id=task_id,
            source_document_ids=source_ids,
            evidence_ids=evidence_ids,
            knowledge_claim_ids=claim_ids,
            snapshot_version=next_version,
            now=ts,
        )
        saved_snapshot = self.evidence_repo.save_evidence_snapshot(snapshot)

        art_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=saved_snapshot.evidence_snapshot_id,
            artifact_version=str(saved_snapshot.snapshot_version),
            metadata_json={
                "source_count": len(source_ids),
                "evidence_count": len(evidence_ids),
                "claim_count": len(claim_ids),
                "content_fingerprint": saved_snapshot.content_fingerprint,
            },
            now=ts,
        )
        saved_ref = self.artifact_repo.save_artifact_ref(art_ref)
        self._session.flush()
        return saved_snapshot, saved_ref
