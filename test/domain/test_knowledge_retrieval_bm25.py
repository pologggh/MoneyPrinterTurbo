from __future__ import annotations

import pytest

from app.domain.evidence import (
    EvidenceLocator,
    KnowledgeChunk,
    SourceDocument,
)
from app.services.knowledge.bm25_retriever import (
    BM25Index,
    tokenize,
)


def _make_chunk(source_id: str, text: str, index: int, chunk_id: str | None = None) -> KnowledgeChunk:
    doc = SourceDocument.create_text(text=text, title="Doc", source_document_id=source_id)
    return KnowledgeChunk.create(
        source_document=doc,
        normalized_text=text,
        chunk_index=index,
        chunk_id=chunk_id,
    )


def test_tokenize_multilingual():
    text = "Transformer 模型 Attention 机制 2024!"
    tokens = tokenize(text)
    assert "transformer" in tokens
    assert "模" in tokens
    assert "型" in tokens
    assert "attention" in tokens
    assert "机" in tokens
    assert "制" in tokens
    assert "2024" in tokens


def test_bm25_ranking_accuracy():
    """Verify that a document directly relevant to the query scores highest."""
    chunk_attention = _make_chunk(
        "src_ai",
        "Attention mechanisms in transformer architectures dynamically weigh input representations.",
        0,
        "chk_attention",
    )
    chunk_dns = _make_chunk(
        "src_net",
        "Domain Name System (DNS) translates human readable hostnames into IP addresses.",
        0,
        "chk_dns",
    )
    chunk_tcp = _make_chunk(
        "src_net",
        "Transmission Control Protocol provides reliable, ordered, and error-checked byte stream delivery.",
        1,
        "chk_tcp",
    )

    index = BM25Index([chunk_attention, chunk_dns, chunk_tcp])

    # Query for transformer attention
    results_ai = index.search("transformer attention", top_k=3)
    assert len(results_ai) >= 1
    assert results_ai[0].chunk_id == "chk_attention"
    assert results_ai[0].score > 0

    # Query for DNS IP addresses
    results_net = index.search("DNS IP addresses", top_k=3)
    assert len(results_net) >= 1
    assert results_net[0].chunk_id == "chk_dns"
    assert results_net[0].score > 0


def test_bm25_deterministic_tie_breaking():
    """Verify identical scores tie-break strictly by chunk_id ascending."""
    # Two identical texts on different chunks
    chunk_b = _make_chunk("src_1", "identical text for score test", 0, "chk_b")
    chunk_a = _make_chunk("src_1", "identical text for score test", 1, "chk_a")

    # Index in reverse order
    index = BM25Index([chunk_b, chunk_a])
    results = index.search("identical text", top_k=2)

    assert len(results) == 2
    assert results[0].score == results[1].score
    # Must be chk_a first because 'chk_a' < 'chk_b'
    assert results[0].chunk_id == "chk_a"
    assert results[1].chunk_id == "chk_b"


def test_bm25_source_scoping_filter():
    """Verify source_scope_ids strictly excludes chunks from other sources."""
    chunk_1 = _make_chunk("src_allowed", "Database transactions ACID properties", 0, "chk_1")
    chunk_2 = _make_chunk("src_blocked", "Database transactions isolation levels", 0, "chk_2")

    index = BM25Index([chunk_1, chunk_2])
    results = index.search("database transactions", source_scope_ids=["src_allowed"])

    assert len(results) == 1
    assert results[0].chunk_id == "chk_1"
    assert results[0].source_document_id == "src_allowed"
