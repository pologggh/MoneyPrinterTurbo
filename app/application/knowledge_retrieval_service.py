from __future__ import annotations

from datetime import UTC, datetime
import os
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
from app.domain.knowledge_base import ChunkEmbedding
from app.persistence.repositories import (
    EmbeddingRepository,
    EvidenceRepository,
    KnowledgeBaseRepository,
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
from app.services.knowledge.embedding_provider import (
    EmbeddingProvider,
    get_embedding_provider,
)
from app.services.knowledge.hybrid_retriever import (
    DEFAULT_HYBRID_POLICY_VERSION,
    HybridRetriever,
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
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._session = session
        self.evidence_repo = EvidenceRepository(session)
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.embedding_repo = EmbeddingRepository(session)
        self.fetcher = fetcher or SourceFetcher()
        self.parser = parser or DocumentParser()
        self.chunker = chunker or KnowledgeChunker()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def _generate_and_save_embeddings(self, chunks: Sequence[KnowledgeChunk]) -> None:
        if not self.embedding_provider or not chunks:
            return
        provider_name = self.embedding_provider.provider_name
        model_name = self.embedding_provider.model_name
        expected_dim = self.embedding_provider.dimension

        existing = self.embedding_repo.list_embeddings_for_chunks(
            [c.chunk_id for c in chunks],
            provider=provider_name,
            model=model_name,
        )
        existing_map = {e.chunk_id: e for e in existing}
        needed_chunks = [
            c for c in chunks
            if c.chunk_id not in existing_map
            or existing_map[c.chunk_id].text_hash != c.text_hash
            or existing_map[c.chunk_id].dimension != expected_dim
        ]
        if not needed_chunks:
            return

        texts = [c.normalized_text for c in needed_chunks]
        vectors = self.embedding_provider.embed_texts(texts)
        embeddings_to_save: list[ChunkEmbedding] = []
        for chunk, vec in zip(needed_chunks, vectors):
            emb = ChunkEmbedding.create(
                chunk_id=chunk.chunk_id,
                provider=provider_name,
                model=model_name,
                vector=vec,
                text_hash=chunk.text_hash,
            )
            embeddings_to_save.append(emb)
        self.embedding_repo.save_chunk_embeddings(embeddings_to_save)

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
        - Generates and persists chunk vector embeddings.
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

            # File Acquisition
            if source.source_type == SourceType.FILE and not content_to_parse and not source.content_snapshot:
                if source.source_locator and os.path.isfile(source.source_locator):
                    with open(source.source_locator, "rb") as f:
                        content_to_parse = f.read()

            # Document Parsing
            parsed = self.parser.parse_source_document(source, content_to_parse)

            # Chunking
            chunks = self.chunker.chunk_document(source, parsed)

            # Persistence
            if chunks:
                self.evidence_repo.save_knowledge_chunks(chunks)
                if self.embedding_provider:
                    try:
                        self._generate_and_save_embeddings(chunks)
                    except Exception as emb_exc:
                        from loguru import logger
                        logger.warning(f"Failed to generate embeddings for source '{source_document_id}': {emb_exc}")

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
    """Scoped knowledge retrieval engine backed by deterministic Okapi BM25, exact Cosine Vector search, and RRF."""

    def __init__(
        self,
        session: Session,
        retrieval_policy_version: str = DEFAULT_HYBRID_POLICY_VERSION,
        embedding_provider: EmbeddingProvider | None = None,
        min_vector_similarity: float = 0.5,
    ) -> None:
        self._session = session
        self.evidence_repo = EvidenceRepository(session)
        self.task_repo = KnowledgeVideoTaskRepository(session)
        self.kb_repo = KnowledgeBaseRepository(session)
        self.embedding_repo = EmbeddingRepository(session)
        self.retrieval_policy_version = retrieval_policy_version
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.min_vector_similarity = min_vector_similarity

    def retrieve(
        self,
        task_id: str,
        query: str,
        top_k: int = 10,
        source_scope_ids: Sequence[str] | None = None,
        auto_create_evidence_items: bool = True,
        retrieval_mode: str | None = None,
    ) -> RetrievalSnapshot:
        """Executes scoped hybrid retrieval for a task, creating a frozen RetrievalSnapshot and stable EvidenceItems.

        - Strictly enforces task scoping: only sources associated with task_id (directly or via active KBs) can be queried.
        - If source_scope_ids is provided, verifies that each source belongs to task_id.
        - Searches using HybridRetriever (BM25 + Cosine Vector + RRF).
        - Reuses or creates EvidenceItem records for top candidate chunks.
        - Freezes and persists RetrievalSnapshot.
        """
        task = self.task_repo.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")

        # Get all sources associated with this task (merging direct task sources + attached KB sources)
        task_sources = self.kb_repo.list_sources_for_task_with_kbs(task_id)
        if not task_sources:
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

        req_mode = (retrieval_mode or "").strip().upper()

        # Load embeddings for the allowed sources if provider is available and mode allows vector
        embeddings: list[ChunkEmbedding] = []
        if self.embedding_provider and req_mode != "BM25_ONLY":
            raw_embs = self.embedding_repo.list_embeddings_for_sources(
                effective_scopes,
                provider=self.embedding_provider.provider_name,
                model=self.embedding_provider.model_name,
            )
            chunk_hash_map = {c.chunk_id: c.text_hash for c in chunks}
            expected_dim = self.embedding_provider.dimension
            embeddings = [
                e for e in raw_embs
                if e.dimension == expected_dim
                and chunk_hash_map.get(e.chunk_id) == e.text_hash
            ]

        # Build HybridRetriever and search
        retriever = HybridRetriever(
            chunks=chunks,
            embeddings=embeddings,
            embedding_provider=self.embedding_provider if req_mode != "BM25_ONLY" else None,
            retrieval_policy_version=self.retrieval_policy_version,
            min_vector_similarity=self.min_vector_similarity,
        )
        candidates, mode = retriever.search(
            query=query,
            top_k=top_k,
            source_scope_ids=effective_scopes,
            mode=retrieval_mode,
        )

        # Stable EvidenceItem creation / resolution
        selected_evidence_ids: list[str] = []
        if auto_create_evidence_items and candidates:
            for cand in candidates:
                src_doc = source_doc_map.get(cand.source_document_id)
                if not src_doc:
                    continue
                extraction_method = f"RETRIEVAL_{cand.retrieval_method}"
                new_ev = EvidenceItem.create(
                    source_document=src_doc,
                    original_excerpt=cand.excerpt,
                    locator=cand.locator,
                    evidence_role=EvidenceRole.FACTUAL_SUPPORT,
                    extraction_method=extraction_method,
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
            effective_retrieval_mode=mode,
        )
        self.evidence_repo.save_retrieval_snapshot(snapshot)
        self._session.commit()
        return snapshot
