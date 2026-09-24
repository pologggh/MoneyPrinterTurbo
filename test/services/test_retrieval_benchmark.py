from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
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
from app.domain.retrieval_benchmark import (
    QueryCategory,
    QueryDifficulty,
    RetrievalGoldenDataset,
    RetrievalGoldenQuery,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    EmbeddingRepository,
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
)
from app.services.knowledge.embedding_provider import DeterministicFakeEmbeddingProvider
from app.services.knowledge.retrieval_benchmark import (
    RetrievalBenchmarkRunner,
    RetrievalDatasetLoader,
    RetrievalDatasetValidator,
    RetrievalMetricsCalculator,
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


# ---------------------------------------------------------------------------
# Test 1: Recall@K calculation
# ---------------------------------------------------------------------------
def test_recall_at_k_calculation():
    calc = RetrievalMetricsCalculator
    relevance_map = {"c1": 3, "c2": 2, "c3": 1, "c_irrelevant": 0}
    # Total relevant: c1, c2, c3 (size 3)
    retrieved = ["c1", "c_other", "c2", "c_irrelevant", "c3"]

    assert calc.compute_recall_at_k(retrieved, relevance_map, k=1) == pytest.approx(1.0 / 3.0)
    assert calc.compute_recall_at_k(retrieved, relevance_map, k=3) == pytest.approx(2.0 / 3.0)
    assert calc.compute_recall_at_k(retrieved, relevance_map, k=5) == pytest.approx(3.0 / 3.0)
    assert calc.compute_recall_at_k(retrieved, relevance_map, k=0) == 0.0


# ---------------------------------------------------------------------------
# Test 2: Precision@K calculation
# ---------------------------------------------------------------------------
def test_precision_at_k_calculation():
    calc = RetrievalMetricsCalculator
    relevance_map = {"c1": 3, "c2": 1, "c3": 0}
    # Top 5 contains 2 relevant chunks (c1, c2)
    retrieved = ["c1", "c_other", "c2", "c_none", "c_extra"]

    p5 = calc.compute_precision_at_k(retrieved, relevance_map, k=5)
    assert p5 == pytest.approx(2.0 / 5.0)

    p1 = calc.compute_precision_at_k(retrieved, relevance_map, k=1)
    assert p1 == pytest.approx(1.0 / 1.0)


# ---------------------------------------------------------------------------
# Test 3: MRR calculation
# ---------------------------------------------------------------------------
def test_mrr_calculation():
    calc = RetrievalMetricsCalculator
    relevance_map = {"c_target": 2}

    # Case 1: First hit at rank 1
    assert calc.compute_mrr(["c_target", "c2"], relevance_map) == 1.0

    # Case 2: First hit at rank 2
    assert calc.compute_mrr(["c_other", "c_target", "c3"], relevance_map) == 0.5

    # Case 3: First hit at rank 4
    assert calc.compute_mrr(["c1", "c2", "c3", "c_target"], relevance_map) == 0.25

    # Case 4: No hit
    assert calc.compute_mrr(["c1", "c2", "c3"], relevance_map) == 0.0


# ---------------------------------------------------------------------------
# Test 4: Graded nDCG@5 calculation
# ---------------------------------------------------------------------------
def test_graded_ndcg5_calculation():
    calc = RetrievalMetricsCalculator
    # Ground truth: c1 has grade 3, c2 has grade 2, c3 has grade 1
    relevance_map = {"c1": 3, "c2": 2, "c3": 1}

    # Ideal ranking: ["c1", "c2", "c3"]
    ideal_ndcg = calc.compute_ndcg_at_k(["c1", "c2", "c3"], relevance_map, k=5)
    assert ideal_ndcg == pytest.approx(1.0)

    # Sub-optimal ranking: ["c2", "c1", "c_none", "c3", "c_other"]
    # DCG = (2^2 - 1)/log2(2) + (2^3 - 1)/log2(3) + 0 + (2^1 - 1)/log2(5) + 0
    #     = 3/1 + 7/1.5849625 + 0 + 1/2.321928 + 0 ≈ 3 + 4.4165 + 0.4307 = 7.8472
    # IDCG = 7/1 + 3/1.5849625 + 1/2.0 ≈ 7 + 1.8928 + 0.5 = 9.3928
    # nDCG ≈ 7.8472 / 9.3928 ≈ 0.8354
    ndcg_sub = calc.compute_ndcg_at_k(["c2", "c1", "c_none", "c3", "c_other"], relevance_map, k=5)
    assert 0.80 < ndcg_sub < 0.90


# ---------------------------------------------------------------------------
# Test 5: Multiple relevant chunks
# ---------------------------------------------------------------------------
def test_multiple_relevant_chunks():
    calc = RetrievalMetricsCalculator
    # Query with 4 relevant chunks of varying importance
    relevance_map = {"chunk_a": 3, "chunk_b": 2, "chunk_c": 1, "chunk_d": 1}
    retrieved = ["chunk_a", "chunk_b", "chunk_other", "chunk_c", "chunk_extra"]

    recall_5 = calc.compute_recall_at_k(retrieved, relevance_map, k=5)
    # Retrieved 3 out of 4 relevant chunks
    assert recall_5 == pytest.approx(3.0 / 4.0)

    precision_5 = calc.compute_precision_at_k(retrieved, relevance_map, k=5)
    assert precision_5 == pytest.approx(3.0 / 5.0)


# ---------------------------------------------------------------------------
# Test 6: Zero-hit query
# ---------------------------------------------------------------------------
def test_zero_hit_query():
    calc = RetrievalMetricsCalculator
    relevance_map = {"chunk_true": 3}
    retrieved = ["chunk_w1", "chunk_w2", "chunk_w3", "chunk_w4", "chunk_w5"]

    assert calc.compute_recall_at_k(retrieved, relevance_map, k=5) == 0.0
    assert calc.compute_precision_at_k(retrieved, relevance_map, k=5) == 0.0
    assert calc.compute_mrr(retrieved, relevance_map) == 0.0
    assert calc.compute_ndcg_at_k(retrieved, relevance_map, k=5) == 0.0

    failures = calc.identify_failure_types(
        ranked_chunk_ids=retrieved,
        relevance_map=relevance_map,
        mode="BM25_ONLY",
    )
    assert "ZERO_RECALL_AT_5" in failures


# ---------------------------------------------------------------------------
# Test 7: Duplicate query ID validation
# ---------------------------------------------------------------------------
def test_duplicate_query_id_validation():
    data = {
        "dataset_name": "test_dup",
        "queries": [
            {
                "query_id": "q001",
                "query": "First query",
                "relevance": {"c1": 3},
            },
            {
                "query_id": "q001",
                "query": "Second duplicate query",
                "relevance": {"c2": 2},
            },
        ],
    }
    with pytest.raises(ValueError, match="Duplicate query_id 'q001'"):
        RetrievalDatasetValidator.validate_dict(data)


# ---------------------------------------------------------------------------
# Test 8: Invalid relevance grade validation
# ---------------------------------------------------------------------------
def test_invalid_relevance_grade_validation():
    # Negative grade
    data_neg = {
        "dataset_name": "test_grade",
        "queries": [
            {
                "query_id": "q001",
                "query": "Some query",
                "relevance": {"c1": -1},
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid relevance grade -1"):
        RetrievalDatasetValidator.validate_dict(data_neg)

    # Out-of-bounds grade (> 3)
    data_high = {
        "dataset_name": "test_grade",
        "queries": [
            {
                "query_id": "q001",
                "query": "Some query",
                "relevance": {"c1": 4},
            }
        ],
    }
    with pytest.raises(ValueError, match="invalid relevance grade 4"):
        RetrievalDatasetValidator.validate_dict(data_high)


# ---------------------------------------------------------------------------
# Test 9: Missing ground-truth chunk detection
# ---------------------------------------------------------------------------
def test_missing_ground_truth_chunk_detection():
    data = {
        "dataset_name": "test_corpus_check",
        "queries": [
            {
                "query_id": "q001",
                "query": "Explain DNS",
                "relevance": {"chunk_dns_01": 3, "chunk_non_existent": 2},
                "hard_negative_chunk_ids": ["chunk_also_missing"],
            }
        ],
    }
    dataset = RetrievalDatasetValidator.validate_dict(data)
    corpus_chunk_ids = ["chunk_dns_01", "chunk_dns_02"]

    with pytest.raises(ValueError, match="chunk IDs not present in corpus"):
        RetrievalDatasetValidator.validate_against_corpus(dataset, corpus_chunk_ids, strict=True)

    missing = RetrievalDatasetValidator.validate_against_corpus(dataset, corpus_chunk_ids, strict=False)
    assert set(missing) == {"chunk_non_existent", "chunk_also_missing"}


# ---------------------------------------------------------------------------
# Test Setup Helper for Runner Integration Tests (10 - 17)
# ---------------------------------------------------------------------------
@pytest.fixture
def runner_test_env(session_factory):
    """Sets up a deterministic scoped test environment using test_fixture_v1.json."""
    fixture_path = Path("fixtures/benchmark/retrieval/test_fixture_v1.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    dataset = RetrievalDatasetLoader.load_from_file(fixture_path)
    provider = DeterministicFakeEmbeddingProvider(dimension=64)

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)
        emb_repo = EmbeddingRepository(session)

        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Networking & Deep Learning"))
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Benchmark KB"))
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)

        source_doc_ids: list[str] = []
        created_chunk_ids: list[str] = []

        # Create source docs and chunks matching the fixture
        for chunk_item in fixture_data["corpus_chunks"]:
            cid = chunk_item["chunk_id"]
            text = chunk_item["text"]

            src = ev_repo.save_source_document(
                SourceDocument.create_text(text=text, title=f"Doc for {cid}")
            )
            source_doc_ids.append(src.source_document_id)
            kb_repo.associate_source(kb.knowledge_base_id, src.source_document_id)

            chunk = KnowledgeChunk.create(
                source_document=src,
                normalized_text=text,
                chunk_index=0,
                chunk_id=cid,
            )
            ev_repo.save_knowledge_chunks([chunk])
            created_chunk_ids.append(cid)

            # Generate and persist deterministic embeddings
            vec = provider.embed_text(text)
            emb = ChunkEmbedding.create(
                chunk_id=cid,
                provider=provider.provider_name,
                model=provider.model_name,
                vector=vec,
                text_hash=chunk.text_hash,
            )
            emb_repo.save_chunk_embeddings([emb])

        session.commit()

        yield {
            "session": session,
            "task_id": task.task_id,
            "source_doc_ids": source_doc_ids,
            "chunk_ids": created_chunk_ids,
            "dataset": dataset,
            "provider": provider,
        }


# ---------------------------------------------------------------------------
# Test 10: Benchmark runner uses production retrieval interface
# ---------------------------------------------------------------------------
def test_runner_uses_production_retrieval_interface(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    with patch.object(
        runner.retrieval_service, "retrieve", wraps=runner.retrieval_service.retrieve
    ) as spy_retrieve:
        report = runner.run_benchmark(dataset=env["dataset"], modes=["BM25_ONLY"])
        assert spy_retrieve.called
        assert spy_retrieve.call_count == len(env["dataset"].queries)
        assert len(report.modes) == 1
        assert "BM25_ONLY" in report.modes


# ---------------------------------------------------------------------------
# Test 11: Runner executes BM25_ONLY
# ---------------------------------------------------------------------------
def test_runner_executes_bm25_only(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    report = runner.run_benchmark(dataset=env["dataset"], modes=["BM25_ONLY"])
    bm25_summary = report.modes.get("BM25_ONLY")
    assert bm25_summary is not None
    assert bm25_summary.query_count == 3

    # Check candidates have LEXICAL_BM25 retrieval method
    for res in report.per_query_results:
        if res.mode == "BM25_ONLY":
            for item in res.ranked_items:
                assert item.retrieval_method in ("LEXICAL_BM25", "BM25_FALLBACK")


# ---------------------------------------------------------------------------
# Test 12: Runner executes DENSE_ONLY
# ---------------------------------------------------------------------------
def test_runner_executes_dense_only(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    report = runner.run_benchmark(dataset=env["dataset"], modes=["DENSE_ONLY"])
    dense_summary = report.modes.get("DENSE_ONLY")
    assert dense_summary is not None
    assert dense_summary.query_count == 3

    # Check candidates have SEMANTIC_VECTOR retrieval method
    for res in report.per_query_results:
        if res.mode == "DENSE_ONLY":
            for item in res.ranked_items:
                assert item.retrieval_method == "SEMANTIC_VECTOR"


# ---------------------------------------------------------------------------
# Test 13: Runner executes HYBRID
# ---------------------------------------------------------------------------
def test_runner_executes_hybrid(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    report = runner.run_benchmark(dataset=env["dataset"], modes=["HYBRID"])
    hybrid_summary = report.modes.get("HYBRID")
    assert hybrid_summary is not None
    assert hybrid_summary.query_count == 3

    for res in report.per_query_results:
        if res.mode == "HYBRID":
            for item in res.ranked_items:
                assert item.retrieval_method == "HYBRID_RRF"


# ---------------------------------------------------------------------------
# Test 14: One top-5 retrieval call supports all K metrics
# ---------------------------------------------------------------------------
def test_one_top5_retrieval_call_supports_all_k_metrics(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    call_records: list[dict] = []

    def spy_retrieve(*args, **kwargs):
        call_records.append(kwargs)
        return orig_retrieve(*args, **kwargs)

    orig_retrieve = runner.retrieval_service.retrieve
    runner.retrieval_service.retrieve = spy_retrieve

    report = runner.run_benchmark(dataset=env["dataset"], modes=["HYBRID"])

    # Exactly 3 queries, so exactly 3 retrieval calls
    assert len(call_records) == 3
    for call in call_records:
        assert call.get("top_k") == 5

    # Check each per_query_result contains all required K metrics
    for res in report.per_query_results:
        m = res.metrics
        assert "recall@1" in m
        assert "recall@3" in m
        assert "recall@5" in m
        assert "precision@5" in m
        assert "mrr" in m
        assert "ndcg@5" in m


# ---------------------------------------------------------------------------
# Test 15: Per-query raw rankings persisted
# ---------------------------------------------------------------------------
def test_per_query_raw_rankings_persisted(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    report = runner.run_benchmark(dataset=env["dataset"], modes=["BM25_ONLY", "HYBRID"])

    assert len(report.per_query_results) == 6  # 3 queries * 2 modes
    for res in report.per_query_results:
        assert res.query_id.startswith("q_")
        assert len(res.ranked_items) > 0
        for item in res.ranked_items:
            assert item.rank >= 1
            assert item.chunk_id != ""
            assert isinstance(item.score, float)
            assert isinstance(item.is_hit, bool)
            assert isinstance(item.expected_relevance_grade, int)


# ---------------------------------------------------------------------------
# Test 16: Latency summary generated
# ---------------------------------------------------------------------------
def test_latency_summary_generated(runner_test_env):
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    report = runner.run_benchmark(dataset=env["dataset"], modes=["BM25_ONLY", "DENSE_ONLY", "HYBRID"])

    for mode in ("BM25_ONLY", "DENSE_ONLY", "HYBRID"):
        summary = report.modes[mode]
        assert summary.mean_latency_ms >= 0.0
        assert summary.p95_latency_ms >= 0.0
        assert summary.p95_latency_ms >= summary.mean_latency_ms or np.isclose(summary.p95_latency_ms, summary.mean_latency_ms)


# ---------------------------------------------------------------------------
# Test 17: No tests call external embedding API
# ---------------------------------------------------------------------------
def test_no_tests_call_external_embedding_api(runner_test_env):
    """Guarantees that benchmark execution never attempts external network requests."""
    env = runner_test_env
    runner = RetrievalBenchmarkRunner(
        session=env["session"],
        task_id=env["task_id"],
        source_scope_ids=env["source_doc_ids"],
        embedding_provider=env["provider"],
    )

    with patch.object(httpx.Client, "post") as mock_post:
        mock_post.side_effect = AssertionError("External HTTP request detected in benchmark execution!")
        report = runner.run_benchmark(dataset=env["dataset"], modes=["BM25_ONLY", "DENSE_ONLY", "HYBRID"])
        assert len(report.modes) == 3
        mock_post.assert_not_called()
