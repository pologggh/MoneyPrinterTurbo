from __future__ import annotations

import json
from typing import Any
import httpx
import numpy as np
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
    SourceType,
)
from app.domain.knowledge_base import (
    ChunkEmbedding,
    KnowledgeBase,
)
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
    OpenAICompatibleEmbeddingProvider,
    get_embedding_provider,
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


# ---------------------------------------------------------------------------
# Test 1: Real / Deterministic Dense Retrieval Precision Smoke Test
# ---------------------------------------------------------------------------
def test_dense_retrieval_top1_precision_three_domain_chunks(session_factory):
    """
    Validates core loop:
    3 chunks (A: Python indentation, B: Transformer attention, C: FFmpeg codec).
    Query: 'Python 如何定义代码块结构？'
    In DENSE_ONLY mode:
    - Chunk A must be ranked Top-1 with significant margin.
    - Candidate retrieval_method must be SEMANTIC_VECTOR.
    - Snapshot effective_retrieval_mode must be VECTOR_ONLY.
    """
    text_python = "Python 使用严格的缩进（通常是 4 个空格）来定义代码块结构，而不是使用花括号或关键字。"
    text_transformer = "Transformer 架构依赖多头自注意力机制（Self-Attention）来捕捉序列中不同位置之间的全局依赖关系。"
    text_ffmpeg = "FFmpeg 包含众多的音视频编解码器，支持 libx264 视频编码和 aac 音频编码转换。"

    query = "Python 如何定义代码块结构？"

    # Semantic mock vectors (128-dim normalized) where Python is strongly aligned with query
    np.random.seed(42)
    base_query_vec = np.random.randn(128).astype(np.float32)
    base_query_vec /= np.linalg.norm(base_query_vec)

    # Chunk A is query + small orthogonal noise -> high cosine similarity (~0.95)
    noise_a = np.random.randn(128).astype(np.float32)
    noise_a /= np.linalg.norm(noise_a)
    vec_a = 0.9 * base_query_vec + 0.1 * noise_a
    vec_a /= np.linalg.norm(vec_a)

    # Chunk B is orthogonal noise -> low cosine similarity (~0.1)
    vec_b = np.random.randn(128).astype(np.float32)
    vec_b /= np.linalg.norm(vec_b)

    # Chunk C is orthogonal noise -> low cosine similarity (~0.05)
    vec_c = np.random.randn(128).astype(np.float32)
    vec_c /= np.linalg.norm(vec_c)

    vector_store = {
        query: base_query_vec.tolist(),
        text_python: vec_a.tolist(),
        text_transformer: vec_b.tolist(),
        text_ffmpeg: vec_c.tolist(),
    }

    class MockSemanticEmbeddingProvider:
        provider_name = "mock_semantic"
        model_name = "semantic-v1"
        dimension = 128

        def embed_text(self, text: str) -> list[float]:
            t = text.lower()
            if "python" in t and "缩进" in t:
                return vec_a.tolist()
            if "transformer" in t or "attention" in t:
                return vec_b.tolist()
            if "ffmpeg" in t or "libx264" in t:
                return vec_c.tolist()
            if "代码块" in t:
                return base_query_vec.tolist()
            if "python" in t:
                return vec_a.tolist()
            # Default to random unit vector
            v = np.random.randn(128).astype(np.float32)
            return (v / np.linalg.norm(v)).tolist()

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(t) for t in texts]

    provider = MockSemanticEmbeddingProvider()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Programming & AI"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Tech KB"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        doc_py = ev_repo.save_source_document(SourceDocument.create_text(text_python, title="Python"))
        doc_tf = ev_repo.save_source_document(SourceDocument.create_text(text_transformer, title="Transformer"))
        doc_ff = ev_repo.save_source_document(SourceDocument.create_text(text_ffmpeg, title="FFmpeg"))

        for doc in (doc_py, doc_tf, doc_ff):
            kb_repo.associate_source(kb.knowledge_base_id, doc.source_document_id)

        # Ingestion & Embedding
        proc = KnowledgeProcessingService(session=session, embedding_provider=provider)
        proc.process_source_document(doc_py.source_document_id)
        proc.process_source_document(doc_tf.source_document_id)
        proc.process_source_document(doc_ff.source_document_id)

        # Retrieval in DENSE_ONLY mode
        retrieval = KnowledgeRetrievalService(
            session=session,
            embedding_provider=provider,
            min_vector_similarity=0.2,
        )
        snapshot = retrieval.retrieve(
            task_id=task.task_id,
            query=query,
            top_k=3,
            retrieval_mode="DENSE_ONLY",
        )

        assert snapshot.effective_retrieval_mode == "VECTOR_ONLY"
        assert len(snapshot.candidates) >= 1
        top_cand = snapshot.candidates[0]
        assert top_cand.source_document_id == doc_py.source_document_id
        assert top_cand.retrieval_method == "SEMANTIC_VECTOR"
        assert "缩进" in top_cand.excerpt
        assert top_cand.score > 0.8


# ---------------------------------------------------------------------------
# Test 2: Mode Isolation - BM25_ONLY vs DENSE_ONLY
# ---------------------------------------------------------------------------
def test_retrieval_mode_isolation(session_factory):
    """
    Verifies that BM25_ONLY skips vector search completely,
    and DENSE_ONLY skips BM25 lexical search.
    """
    call_log = {"bm25": 0, "vector": 0}

    class InstrumentedProvider:
        provider_name = "instrumented"
        model_name = "inst-v1"
        dimension = 4

        def embed_text(self, text: str) -> list[float]:
            call_log["vector"] += 1
            return [1.0, 0.0, 0.0, 0.0]

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            call_log["vector"] += len(texts)
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    doc = SourceDocument.create_text("Distributed consensus Paxos Raft algorithms.", title="Consensus")
    chunk = KnowledgeChunk.create(doc, "Distributed consensus Paxos Raft algorithms.", chunk_index=0)
    emb = ChunkEmbedding.create(
        chunk_id=chunk.chunk_id,
        provider="instrumented",
        model="inst-v1",
        vector=[1.0, 0.0, 0.0, 0.0],
        text_hash=chunk.text_hash,
    )

    provider = InstrumentedProvider()
    retriever = HybridRetriever(
        chunks=[chunk],
        embeddings=[emb],
        embedding_provider=provider,
    )

    # 1. BM25_ONLY: vector search must NOT be invoked
    call_log["vector"] = 0
    cands, mode = retriever.search(query="Paxos Raft", top_k=5, mode="BM25_ONLY")
    assert mode == "BM25_ONLY"
    assert len(cands) == 1
    assert cands[0].retrieval_method == "LEXICAL_BM25"
    assert call_log["vector"] == 0

    # 2. DENSE_ONLY: vector search must be invoked
    call_log["vector"] = 0
    cands, mode = retriever.search(query="Paxos Raft", top_k=5, mode="DENSE_ONLY")
    assert mode == "VECTOR_ONLY"
    assert len(cands) == 1
    assert cands[0].retrieval_method == "SEMANTIC_VECTOR"
    assert call_log["vector"] >= 1


# ---------------------------------------------------------------------------
# Test 3: Stale Vector Detection and Re-Embedding
# ---------------------------------------------------------------------------
def test_stale_embedding_detection_and_reembedding(session_factory):
    """
    Verifies that when a chunk's text or hash changes, or dimension changes,
    the processing service updates the database with a fresh embedding,
    and the retrieval service does not return stale embeddings.
    """
    provider_v1 = DeterministicFakeEmbeddingProvider(dimension=16, model_name="v1")
    provider_v2 = DeterministicFakeEmbeddingProvider(dimension=32, model_name="v2")

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        emb_repo = EmbeddingRepository(session)

        doc = ev_repo.save_source_document(SourceDocument.create_text("Initial document content version 1", title="Doc"))
        chunk = KnowledgeChunk.create(doc, "Initial document content version 1", chunk_index=0)
        ev_repo.save_knowledge_chunks([chunk])

        # Manually create a stale embedding with mismatched text_hash
        stale_emb = ChunkEmbedding.create(
            chunk_id=chunk.chunk_id,
            provider=provider_v1.provider_name,
            model=provider_v1.model_name,
            vector=[0.1] * 16,
            text_hash="obsolete_hash_12345",
        )
        emb_repo.save_chunk_embeddings([stale_emb])
        session.commit()

        # Processing service should detect the stale hash and re-embed
        proc = KnowledgeProcessingService(session=session, embedding_provider=provider_v1)
        proc._generate_and_save_embeddings([chunk])
        session.commit()

        # Check that DB row was updated with current chunk.text_hash
        updated_embs = emb_repo.list_embeddings_for_chunks(
            [chunk.chunk_id],
            provider=provider_v1.provider_name,
            model=provider_v1.model_name,
        )
        assert len(updated_embs) == 1
        assert updated_embs[0].text_hash == chunk.text_hash

        # Now test dimension change: provider_v2 has dimension 32
        proc_v2 = KnowledgeProcessingService(session=session, embedding_provider=provider_v2)
        proc_v2._generate_and_save_embeddings([chunk])
        session.commit()

        updated_v2_embs = emb_repo.list_embeddings_for_chunks(
            [chunk.chunk_id],
            provider=provider_v2.provider_name,
            model=provider_v2.model_name,
        )
        assert len(updated_v2_embs) == 1
        assert updated_v2_embs[0].dimension == 32


# ---------------------------------------------------------------------------
# Test 4: Dimension Mismatch Rejection in Search
# ---------------------------------------------------------------------------
def test_dimension_mismatch_safety_in_retriever():
    """
    If a stored embedding has dimension 16, but current provider produces dimension 32,
    HybridRetriever must safely reject or skip the incompatible embedding without crashing.
    """
    doc = SourceDocument.create_text("Microservices event-driven architecture.", title="Arch")
    chunk = KnowledgeChunk.create(doc, "Microservices event-driven architecture.", chunk_index=0)

    # Incompatible embedding with dimension 16
    emb_dim16 = ChunkEmbedding.create(
        chunk_id=chunk.chunk_id,
        provider="test_prov",
        model="test_model",
        vector=[0.1] * 16,
        text_hash=chunk.text_hash,
    )

    class ProviderDim32:
        provider_name = "test_prov"
        model_name = "test_model"
        dimension = 32

        def embed_text(self, text: str) -> list[float]:
            return [0.1] * 32

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_text(t) for t in texts]

    retriever = HybridRetriever(
        chunks=[chunk],
        embeddings=[emb_dim16],
        embedding_provider=ProviderDim32(),
    )

    # In HYBRID mode, mismatched embedding is ignored, falls back gracefully to BM25_ONLY
    cands, mode = retriever.search(query="Microservices architecture", top_k=5)
    assert mode in ("BM25_ONLY", "BM25_FALLBACK")
    assert len(cands) == 1
    assert cands[0].chunk_id == chunk.chunk_id

    # In DENSE_ONLY mode, mismatched embedding is skipped -> zero candidates, no crash
    cands_dense, mode_dense = retriever.search(query="Microservices architecture", top_k=5, mode="DENSE_ONLY")
    assert mode_dense == "VECTOR_ONLY"
    assert len(cands_dense) == 0


# ---------------------------------------------------------------------------
# Test 5: Provider Name Propagation in get_embedding_provider
# ---------------------------------------------------------------------------
def test_provider_name_propagation():
    """
    Ensures that when provider='qwen' or 'siliconflow' is configured,
    the instantiated provider preserves the provider_name rather than generic 'openai_compatible'.
    """
    prov_qwen = get_embedding_provider(
        config_dict={
            "enabled": True,
            "provider": "qwen",
            "api_key": "sk-qwen-test",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v3",
            "dimension": 1024,
        },
        prefer_fake=False,
    )
    assert prov_qwen is not None
    assert prov_qwen.provider_name == "qwen"
    assert prov_qwen.dimension == 1024
    assert prov_qwen.model_name == "text-embedding-v3"

    prov_sf = get_embedding_provider(
        config_dict={
            "enabled": True,
            "provider": "siliconflow",
            "api_key": "sk-sf-test",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-large-zh-v1.5",
            "dimension": 1024,
        },
        prefer_fake=False,
    )
    assert prov_sf is not None
    assert prov_sf.provider_name == "siliconflow"


# ---------------------------------------------------------------------------
# Test 6: OpenAI-Compatible HTTP Mock Contract (Zero Remote Calls)
# ---------------------------------------------------------------------------
def test_openai_compatible_http_mock_contract():
    """
    Tests OpenAICompatibleEmbeddingProvider over an httpx MockTransport:
    - Verifies JSON body sent to /embeddings
    - Verifies Bearer header
    - Verifies response parsing
    - Zero remote network calls
    """
    recorded_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        assert request.headers.get("Authorization") == "Bearer sk-contract-mock-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "text-embedding-3-small"
        inputs = body["input"]
        # Return mock 4-dim embeddings
        mock_data = [
            {"index": idx, "embedding": [0.5, 0.5, 0.5, 0.5]}
            for idx, _ in enumerate(inputs)
        ]
        return httpx.Response(200, json={"data": mock_data, "model": "text-embedding-3-small"})

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    prov = OpenAICompatibleEmbeddingProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-contract-mock-key",
        model="text-embedding-3-small",
        dimension=4,
        http_client=mock_client,
    )

    vectors = prov.embed_texts(["Hello world", "Dense retrieval baseline"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    assert len(recorded_requests) == 1
    assert recorded_requests[0].url.path == "/v1/embeddings"
