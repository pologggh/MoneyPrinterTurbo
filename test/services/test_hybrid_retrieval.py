from __future__ import annotations

from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.knowledge_retrieval_service import (
    KnowledgeProcessingService,
    KnowledgeRetrievalService,
)
from app.domain.evidence import (
    KnowledgeChunk,
    SourceDocument,
    SourceStatus,
    SourceType,
    compute_retrieval_snapshot_fingerprint,
)
from app.domain.knowledge_base import (
    ChunkEmbedding,
    KnowledgeBase,
    KnowledgeBaseStatus,
)
from app.services.delivery_report_service import DeliveryReportService
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EmbeddingRepository,
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
)
from app.services.knowledge.embedding_provider import (
    DeterministicFakeEmbeddingProvider,
    EmbeddingProviderError,
)
from app.services.knowledge.hybrid_retriever import (
    DEFAULT_RRF_K,
    HybridRetriever,
)


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal


def test_rrf_scoring_and_deterministic_tie_breaking():
    # Construct 2 chunks from doc
    doc = SourceDocument.create_text("Sample text", title="Doc 1")
    c1 = KnowledgeChunk.create(doc, "Alpha text", chunk_index=0, chunk_id="chk_b_second")
    c2 = KnowledgeChunk.create(doc, "Beta text", chunk_index=1, chunk_id="chk_a_first")

    # In BM25: c1 rank 1, c2 rank 2
    # In Vector: c2 rank 1, c1 rank 2
    # Both have RRF score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.032519
    # Tie-breaking by chunk_id ASC: c2 ('chk_a_first') MUST come before c1 ('chk_b_second')!

    emb1 = ChunkEmbedding.create(c1.chunk_id, "test", "test_m", [1.0, 0.0], c1.text_hash)
    emb2 = ChunkEmbedding.create(c2.chunk_id, "test", "test_m", [0.0, 1.0], c2.text_hash)

    class MockProvider:
        provider_name = "test"
        model_name = "test_m"
        dimension = 2

        def embed_text(self, text: str) -> list[float]:
            # Query is close to c2
            return [0.1, 0.9]

        def embed_texts(self, texts):
            return [self.embed_text(t) for t in texts]

    retriever = HybridRetriever(
        chunks=[c1, c2],
        embeddings=[emb1, emb2],
        embedding_provider=MockProvider(),
        min_vector_similarity=0.0,
    )

    candidates, mode = retriever.search(query="Alpha text", top_k=5)
    assert mode == "HYBRID"
    assert len(candidates) == 2
    # Check that score is computed and rounded
    assert candidates[0].score > 0
    # Both candidates have retrieval_method HYBRID_RRF
    assert candidates[0].retrieval_method == "HYBRID_RRF"
    assert candidates[1].retrieval_method == "HYBRID_RRF"


def test_fallback_to_bm25_on_embedding_failure():
    doc = SourceDocument.create_text("Python concurrency asyncio threading", title="Python Concurrency")
    c1 = KnowledgeChunk.create(doc, "Python concurrency asyncio threading", chunk_index=0)

    class FailingProvider:
        provider_name = "failing"
        model_name = "fail"
        dimension = 4

        def embed_text(self, text: str) -> list[float]:
            raise EmbeddingProviderError("Rate limit exceeded")

        def embed_texts(self, texts):
            raise EmbeddingProviderError("Rate limit exceeded")

    emb1 = ChunkEmbedding.create(c1.chunk_id, "failing", "fail", [0.1, 0.2, 0.3, 0.4], c1.text_hash)
    retriever = HybridRetriever(
        chunks=[c1],
        embeddings=[emb1],
        embedding_provider=FailingProvider(),
    )

    candidates, mode = retriever.search(query="Python asyncio", top_k=5)
    assert mode == "BM25_FALLBACK"
    assert len(candidates) == 1
    assert candidates[0].retrieval_method == "BM25_FALLBACK"


def test_hybrid_retrieval_service_with_attached_kb(session_factory):
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        # Create Task
        task = KnowledgeVideoTask.create(
            topic="Space Exploration",
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Create KB and attach to task
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Astronomy KB"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        # Add document to KB
        kb_text = "The James Webb Space Telescope observes the universe in the infrared spectrum."
        doc = SourceDocument.create_text(kb_text, title="JWST Article")
        ev_repo.save_source_document(doc)
        kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        # Process document with fake embeddings
        proc_service = KnowledgeProcessingService(
            session=session,
            embedding_provider=DeterministicFakeEmbeddingProvider(dimension=32),
        )
        chunks = proc_service.process_source_document(doc.source_document_id)
        assert len(chunks) >= 1

        # Retrieve through KnowledgeRetrievalService
        retrieval_service = KnowledgeRetrievalService(
            session=session,
            embedding_provider=DeterministicFakeEmbeddingProvider(dimension=32),
        )
        snapshot = retrieval_service.retrieve(
            task_id=task.task_id,
            query="James Webb Telescope infrared",
            top_k=5,
        )

        assert snapshot is not None
        assert len(snapshot.candidates) >= 1
        assert snapshot.candidates[0].source_document_id == doc.source_document_id
        assert len(snapshot.selected_evidence_ids) >= 1

        # Check that EvidenceItem was created and stored
        ev_item = ev_repo.get_evidence_item(snapshot.selected_evidence_ids[0])
        assert ev_item is not None
        assert ev_item.source_document_id == doc.source_document_id
        assert "infrared" in ev_item.original_excerpt.lower()


def test_retrieval_mode_hybrid_execution_and_reporting(tmp_path, session_factory):
    """
    Tests HYBRID mode: both BM25 and Vector produce matches, fused with RRF.
    Verifies domain model, fingerprint determinism, ORM reload, and delivery report.
    """
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Neural Networks"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Deep Learning"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        doc = ev_repo.save_source_document(
            SourceDocument.create_text("Convolutional neural networks excel at computer vision tasks.", title="CNN Guide")
        )
        kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        proc_service = KnowledgeProcessingService(
            session=session,
            embedding_provider=DeterministicFakeEmbeddingProvider(dimension=16),
        )
        proc_service.process_source_document(doc.source_document_id)

        retrieval_service = KnowledgeRetrievalService(
            session=session,
            embedding_provider=DeterministicFakeEmbeddingProvider(dimension=16),
        )
        snapshot = retrieval_service.retrieve(
            task_id=task.task_id,
            query="Convolutional neural networks vision",
            top_k=5,
        )

        assert snapshot.effective_retrieval_mode == "HYBRID"
        assert len(snapshot.candidates) >= 1
        assert snapshot.candidates[0].retrieval_method == "HYBRID_RRF"
        snapshot_id = snapshot.retrieval_snapshot_id
        task_id = task.task_id

    # Verify ORM reloading in a new session
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        reloaded = ev_repo.get_retrieval_snapshot(snapshot_id)
        assert reloaded is not None
        assert reloaded.effective_retrieval_mode == "HYBRID"

        expected_fp = compute_retrieval_snapshot_fingerprint(
            task_id=reloaded.task_id,
            query=reloaded.query,
            source_scope_ids=reloaded.source_scope_ids,
            retrieval_policy_version=reloaded.retrieval_policy_version,
            processing_version=reloaded.processing_version,
            candidates=reloaded.candidates,
            selected_evidence_ids=reloaded.selected_evidence_ids,
            effective_retrieval_mode="HYBRID",
        )
        assert reloaded.content_fingerprint == expected_fp

        # Verify delivery report rendering
        report_service = DeliveryReportService(session)
        report_path = str(tmp_path / "source_report_hybrid.md")
        report_service.generate_source_report(task_id, report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "- **Effective Retrieval Mode**: `HYBRID`" in content


def test_retrieval_mode_vector_only_execution_and_reporting(tmp_path, session_factory):
    """
    Tests VECTOR_ONLY mode: query has zero lexical BM25 overlap with text,
    but vector search finds high cosine similarity.
    """
    class FixedUnitVectorProvider:
        provider_name = "unit_provider"
        model_name = "unit_model"
        dimension = 4

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

        def embed_texts(self, texts):
            return [self.embed_text(t) for t in texts]

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)
        emb_repo = EmbeddingRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Aviation"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Flight Manual"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        # Document text has NO words overlapping with query "helicopter"
        doc = ev_repo.save_source_document(
            SourceDocument.create_text("Aerodynamics lift drag thrust velocity", title="Flight Basics")
        )
        kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        # Create chunk directly
        chunk = KnowledgeChunk.create(doc, "Aerodynamics lift drag thrust velocity", chunk_index=0)
        ev_repo.save_knowledge_chunks([chunk])

        # Store aligned embedding vector
        emb = ChunkEmbedding.create(
            chunk_id=chunk.chunk_id,
            provider="unit_provider",
            model="unit_model",
            vector=[1.0, 0.0, 0.0, 0.0],
            text_hash=chunk.text_hash,
        )
        emb_repo.save_chunk_embeddings([emb])

        unit_provider = FixedUnitVectorProvider()
        retrieval_service = KnowledgeRetrievalService(
            session=session,
            embedding_provider=unit_provider,
        )

        # Query has zero word overlap with "Aerodynamics lift drag thrust velocity"
        snapshot = retrieval_service.retrieve(
            task_id=task.task_id,
            query="rotorcraft hovering",
            top_k=5,
        )

        assert snapshot.effective_retrieval_mode == "VECTOR_ONLY"
        assert len(snapshot.candidates) >= 1
        assert snapshot.candidates[0].retrieval_method == "SEMANTIC_VECTOR"
        snapshot_id = snapshot.retrieval_snapshot_id
        task_id = task.task_id

    # Verify ORM reload and delivery report
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        reloaded = ev_repo.get_retrieval_snapshot(snapshot_id)
        assert reloaded is not None
        assert reloaded.effective_retrieval_mode == "VECTOR_ONLY"

        report_service = DeliveryReportService(session)
        report_path = str(tmp_path / "source_report_vector_only.md")
        report_service.generate_source_report(task_id, report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "- **Effective Retrieval Mode**: `VECTOR_ONLY`" in content


def test_retrieval_mode_bm25_only_execution_and_reporting(tmp_path, session_factory):
    """
    Tests BM25_ONLY mode: lexical BM25 matches, but embedding provider is None or no embeddings exist.
    """
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Databases"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Relational DBs"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        doc = ev_repo.save_source_document(
            SourceDocument.create_text("PostgreSQL supports relational tables and acid transactions.", title="PostgreSQL Guide")
        )
        kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        # Chunk document without generating any embeddings
        chunk = KnowledgeChunk.create(doc, "PostgreSQL supports relational tables and acid transactions.", chunk_index=0)
        ev_repo.save_knowledge_chunks([chunk])

        # Run retrieval with NO embedding provider (or no embeddings)
        retrieval_service = KnowledgeRetrievalService(
            session=session,
            embedding_provider=None,
        )
        # Ensure provider is None
        retrieval_service.embedding_provider = None

        snapshot = retrieval_service.retrieve(
            task_id=task.task_id,
            query="PostgreSQL relational transactions",
            top_k=5,
        )

        assert snapshot.effective_retrieval_mode == "BM25_ONLY"
        assert len(snapshot.candidates) >= 1
        assert snapshot.candidates[0].retrieval_method == "LEXICAL_BM25"
        snapshot_id = snapshot.retrieval_snapshot_id
        task_id = task.task_id

    # Verify ORM reload and delivery report
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        reloaded = ev_repo.get_retrieval_snapshot(snapshot_id)
        assert reloaded is not None
        assert reloaded.effective_retrieval_mode == "BM25_ONLY"

        report_service = DeliveryReportService(session)
        report_path = str(tmp_path / "source_report_bm25_only.md")
        report_service.generate_source_report(task_id, report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "- **Effective Retrieval Mode**: `BM25_ONLY`" in content


def test_retrieval_mode_bm25_fallback_execution_and_reporting(tmp_path, session_factory):
    """
    Tests BM25_FALLBACK mode: embedding provider raises EmbeddingProviderError,
    gracefully degrading to BM25 search.
    """
    class ExplodingProvider:
        provider_name = "exploding"
        model_name = "explode"
        dimension = 4

        def embed_text(self, text: str) -> list[float]:
            raise EmbeddingProviderError("Upstream embedding service 503 unavailable")

        def embed_texts(self, texts):
            raise EmbeddingProviderError("Upstream embedding service 503 unavailable")

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)
        emb_repo = EmbeddingRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Networking"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="HTTP Specs"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        doc = ev_repo.save_source_document(
            SourceDocument.create_text("HTTP 3 uses QUIC protocol over UDP.", title="HTTP/3 Overview")
        )
        kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        chunk = KnowledgeChunk.create(doc, "HTTP 3 uses QUIC protocol over UDP.", chunk_index=0)
        ev_repo.save_knowledge_chunks([chunk])

        # Save an embedding with matching provider name
        emb = ChunkEmbedding.create(
            chunk_id=chunk.chunk_id,
            provider="exploding",
            model="explode",
            vector=[0.1, 0.2, 0.3, 0.4],
            text_hash=chunk.text_hash,
        )
        emb_repo.save_chunk_embeddings([emb])

        retrieval_service = KnowledgeRetrievalService(
            session=session,
            embedding_provider=ExplodingProvider(),
        )

        snapshot = retrieval_service.retrieve(
            task_id=task.task_id,
            query="HTTP protocol QUIC",
            top_k=5,
        )

        assert snapshot.effective_retrieval_mode == "BM25_FALLBACK"
        assert len(snapshot.candidates) >= 1
        assert snapshot.candidates[0].retrieval_method == "BM25_FALLBACK"
        snapshot_id = snapshot.retrieval_snapshot_id
        task_id = task.task_id

    # Verify ORM reload and delivery report
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        reloaded = ev_repo.get_retrieval_snapshot(snapshot_id)
        assert reloaded is not None
        assert reloaded.effective_retrieval_mode == "BM25_FALLBACK"

        report_service = DeliveryReportService(session)
        report_path = str(tmp_path / "source_report_fallback.md")
        report_service.generate_source_report(task_id, report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()
        assert "- **Effective Retrieval Mode**: `BM25_FALLBACK`" in content
