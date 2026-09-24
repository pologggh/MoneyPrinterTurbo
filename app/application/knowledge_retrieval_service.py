from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app.domain.evidence import (
    EvidenceDomainError,
    EvidenceItem,
    EvidenceNotFoundError,
    EvidenceRole,
    KnowledgeChunk,
    RetrievalCandidate,
    RetrievalSnapshot,
    SourceDocument,
    SourceStatus,
    SourceType,
    compute_evidence_id,
    compute_sha256,
)
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
)
from app.services.knowledge.bm25_retriever import (
    DEFAULT_RETRIEVAL_POLICY_VERSION,
    BM25Index,
)
from app.services.knowledge.chunking import (
    ChunkingPolicy,
    KnowledgeChunker,
)
from app.services.knowledge.document_parser import (
    DEFAULT_PROCESSING_VERSION,
    DocumentParser,
    ParsedKnowledgeDocument,
)
from app.services.knowledge.source_fetcher import SourceFetcher


class KnowledgeProcessingService:
    """Acquires, parses, and chunks registered source documents into deterministic KnowledgeChunks."""

    def __init__(
        self,
        session: Session,
        fetcher: SourceFetcher | None = None,
        parser: DocumentParser | None = None,
        chunker: KnowledgeChunker | None = None,
    ) -> None:
        self._session = session
        self.evidence_repo = EvidenceRepository(session)
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.fetcher = fetcher or SourceFetcher()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or KnowledgeChunker()

    def process_source_document(
        self,
        source_document_id: str,
        raw_content: bytes | str | None = None,
    ) -> tuple[KnowledgeChunk, ...]:
        """Processes a single registered SourceDocument into normalized KnowledgeChunks.

        - If SourceDocument is a URL and not yet fetched, fetches it securely with bounded limits.
        - Parses raw content / snapshot into structural blocks.
        - Splits into deterministic KnowledgeChunks.
        - Persists chunks to storage.
        - Marks SourceDocument READY (or FAILED if error occurs).
        """
        source = self.evidence_repo.get_source_document(source_document_id)
        if source is None:
            raise EvidenceNotFoundError(f"SourceDocument '{source_document_id}' not found.")

        try:
            content_to_parse = raw_content

            # URL Acquisition
            if source.source_type == SourceType.URL and not source.content_snapshot and not raw_content:
                fetched = self.fetcher.fetch_url(source.source_locator)
                content_to_parse = fetched.extracted_text
                new_title = (
                    source.title
                    if source.title and source.title != source.source_locator
                    else (fetched.title or source.title)
                )
                c_hash = compute_sha256(content_to_parse)
                source = source.model_copy(
                    update={
                        "title": new_title,
                        "content_snapshot": content_to_parse,
                        "content_hash": c_hash,
                        "status": SourceStatus.READY,
                    }
                )
                self.evidence_repo.save_source_document(source)

            # Document Parsing
            parsed = self.parser.parse_source_document(source, content_to_parse)

            # Chunking
            chunks = self.chunker.chunk_document(source, parsed)

            # Persistence
            if chunks:
                self.evidence_repo.save_knowledge_chunks(chunks)

            # Update Source status to READY
            if source.status != SourceStatus.READY:
                source = source.model_copy(update={"status": SourceStatus.READY})
                self.evidence_repo.save_source_document(source)

            self._session.commit()
            return chunks

        except Exception as exc:
            # Mark source as FAILED with error details
            try:
                meta = dict(source.metadata_json)
                meta["processing_error"] = str(exc)
                source = source.model_copy(
                    update={
                        "status": SourceStatus.FAILED,
                        "metadata_json": meta,
                    }
                )
                self.evidence_repo.save_source_document(source)
                self._session.commit()
            except Exception:
                self._session.rollback()
            raise

    def process_sources_for_task(self, task_id: str) -> dict[str, tuple[KnowledgeChunk, ...]]:
        """Processes all registered sources for a given task."""
        sources = self.evidence_repo.list_sources_for_task(task_id)
        results: dict[str, tuple[KnowledgeChunk, ...]] = {}
        for src in sources:
            existing_chunks = self.evidence_repo.list_chunks_for_source(src.source_document_id)
            if existing_chunks:
                results[src.source_document_id] = tuple(existing_chunks)
            else:
                chunks = self.process_source_document(src.source_document_id)
                results[src.source_document_id] = chunks
        return results


class KnowledgeRetrievalService:
    """Scoped knowledge retrieval engine backed by deterministic Okapi BM25 and RetrievalSnapshots."""

    def __init__(
        self,
        session: Session,
        retrieval_policy_version: str = DEFAULT_RETRIEVAL_POLICY_VERSION,
    ) -> None:
        self._session = session
        self.evidence_repo = EvidenceRepository(session)
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.retrieval_policy_version = retrieval_policy_version

    def retrieve(
        self,
        task_id: str,
        query: str,
        top_k: int = 10,
        source_scope_ids: Sequence[str] | None = None,
        auto_create_evidence_items: bool = True,
    ) -> RetrievalSnapshot:
        """Executes scoped lexical retrieval for a task, creating a frozen RetrievalSnapshot and stable EvidenceItems.

        - Strictly enforces task scoping: only sources associated with task_id can be queried.
        - If source_scope_ids is provided, verifies that each source belongs to task_id.
        - Searches using BM25Index.
        - Reuses or creates EvidenceItem records for top candidate chunks.
        - Freezes and persists RetrievalSnapshot.
        """
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        # Get all sources associated with this task
        task_sources = self.evidence_repo.list_sources_for_task(task_id)
        valid_source_ids = {s.source_document_id for s in task_sources}
        source_doc_map = {s.source_document_id: s for s in task_sources}

        # Validate requested scope
        if source_scope_ids is not None:
            for s_id in source_scope_ids:
                if s_id not in valid_source_ids:
                    raise EvidenceDomainError(
                        f"Source document '{s_id}' is not associated with task '{task_id}'."
                    )
            effective_scopes = tuple(sorted(set(source_scope_ids)))
        else:
            effective_scopes = tuple(sorted(valid_source_ids))

        if not effective_scopes:
            snapshot = RetrievalSnapshot.create(
                task_id=task_id,
                query=query,
                source_scope_ids=(),
                candidates=(),
                selected_evidence_ids=(),
                retrieval_policy_version=self.retrieval_policy_version,
            )
            self.evidence_repo.save_retrieval_snapshot(snapshot)
            self._session.commit()
            return snapshot

        # Load chunks strictly for the allowed sources
        chunks = self.evidence_repo.list_chunks_for_sources(effective_scopes)
        if not chunks:
            snapshot = RetrievalSnapshot.create(
                task_id=task_id,
                query=query,
                source_scope_ids=effective_scopes,
                candidates=(),
                selected_evidence_ids=(),
                retrieval_policy_version=self.retrieval_policy_version,
            )
            self.evidence_repo.save_retrieval_snapshot(snapshot)
            self._session.commit()
            return snapshot

        # Build BM25 index and search
        index = BM25Index(chunks, retrieval_policy_version=self.retrieval_policy_version)
        candidates = index.search(query=query, top_k=top_k, source_scope_ids=effective_scopes)

        # Stable EvidenceItem creation / resolution
        selected_evidence_ids: list[str] = []
        if auto_create_evidence_items and candidates:
            for cand in candidates:
                src_doc = source_doc_map.get(cand.source_document_id)
                if not src_doc:
                    continue
                new_ev = EvidenceItem.create(
                    source_document=src_doc,
                    original_excerpt=cand.excerpt,
                    locator=cand.locator,
                    evidence_role=EvidenceRole.FACTUAL_SUPPORT,
                    extraction_method="RETRIEVAL_LEXICAL_BM25",
                    confidence=1.0,
                )
                ev_id = new_ev.evidence_id
                existing_ev = self.evidence_repo.get_evidence_item(ev_id)
                if existing_ev is None:
                    self.evidence_repo.save_evidence_item(new_ev)
                selected_evidence_ids.append(ev_id)

        # Create immutable RetrievalSnapshot
        snapshot = RetrievalSnapshot.create(
            task_id=task_id,
            query=query,
            source_scope_ids=effective_scopes,
            candidates=candidates,
            selected_evidence_ids=selected_evidence_ids,
            retrieval_policy_version=self.retrieval_policy_version,
        )
        self.evidence_repo.save_retrieval_snapshot(snapshot)
        self._session.commit()
        return snapshot
