from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.knowledge_retrieval_service import (
    KnowledgeProcessingService,
    KnowledgeRetrievalService,
)
from app.domain.evidence import KnowledgeChunk, SourceDocument, SourceType
from app.domain.knowledge_base import KnowledgeBase
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
    OpenAICompatibleEmbeddingProvider,
    get_embedding_provider,
    redact_secrets,
)
from app.services.knowledge.hybrid_retriever import HybridRetriever


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


def test_embedding_provider_honest_degradation_when_disabled():
    # Empty config
    prov = get_embedding_provider(config_dict={}, prefer_fake=False)
    assert prov is None

    # Disabled config
    prov2 = get_embedding_provider(
        config_dict={"enabled": False, "api_key": "sk-real-key-123"},
        prefer_fake=False,
    )
    assert prov2 is None

    # Enabled but missing API key
    prov3 = get_embedding_provider(
        config_dict={"enabled": True, "api_key": ""},
        prefer_fake=False,
    )
    assert prov3 is None


def test_embedding_provider_fake_when_explicitly_requested():
    # Via prefer_fake
    prov_fake1 = get_embedding_provider(config_dict={}, prefer_fake=True)
    assert isinstance(prov_fake1, DeterministicFakeEmbeddingProvider)

    # Via provider = "fake" in config
    prov_fake2 = get_embedding_provider(
        config_dict={"provider": "fake", "dimension": 64},
        prefer_fake=False,
    )
    assert isinstance(prov_fake2, DeterministicFakeEmbeddingProvider)
    assert prov_fake2.dimension == 64


def test_embedding_provider_openai_when_enabled():
    prov = get_embedding_provider(
        config_dict={
            "enabled": True,
            "provider": "openai",
            "api_key": "sk-prod-secret-999",
            "base_url": "https://api.openai.com/v1",
            "model": "text-embedding-3-small",
            "dimension": 1536,
        },
        prefer_fake=False,
    )
    assert isinstance(prov, OpenAICompatibleEmbeddingProvider)
    assert prov.model_name == "text-embedding-3-small"
    assert prov.dimension == 1536


def test_credential_redaction():
    leak_text = "Authorization failed with Bearer sk-1234567890abcdef and api_key='sk-abcdef1234567890'"
    redacted = redact_secrets(leak_text)
    assert "sk-1234567890" not in redacted
    assert "sk-abcdef123456" not in redacted
    assert "***" in redacted


def test_processing_service_skips_embedding_when_provider_is_none(session_factory):
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        doc = SourceDocument.create_text(
            text="General relativity describes gravitation as a geometric property of spacetime.",
            title="Relativity",
        )
        ev_repo.save_source_document(doc)
        session.commit()
        doc_id = doc.source_document_id

    with session_factory() as session:
        proc = KnowledgeProcessingService(session, embedding_provider=None)
        # Explicitly ensure proc has no embedding provider
        proc.embedding_provider = None
        chunks = proc.process_source_document(doc_id)
        session.commit()

        assert len(chunks) >= 1
        emb_repo = EmbeddingRepository(session)
        embs = emb_repo.list_embeddings_for_chunks(
            [c.chunk_id for c in chunks],
            provider="deterministic_fake",
            model="fake-embedding-v1",
        )
        assert len(embs) == 0


def test_hybrid_retriever_degrades_to_bm25_only_when_no_provider():
    chunk = KnowledgeChunk.create(
        source_document=SourceDocument.create_text("Black holes form when massive stars collapse."),
        chunk_index=0,
        normalized_text="Black holes form when massive stars collapse.",
    )
    retriever = HybridRetriever(
        chunks=[chunk],
        embeddings=[],
        embedding_provider=None,
    )
    candidates, mode = retriever.search(query="massive stars collapse", top_k=5)
    assert mode == "BM25_ONLY"
    assert len(candidates) == 1
    assert candidates[0].chunk_id == chunk.chunk_id
