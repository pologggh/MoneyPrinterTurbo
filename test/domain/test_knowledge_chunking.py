from __future__ import annotations

import pytest

from app.domain.evidence import (
    EvidenceLocator,
    SourceDocument,
    SourceType,
)
from app.services.knowledge.chunking import (
    ChunkingPolicy,
    KnowledgeChunker,
)
from app.services.knowledge.document_parser import (
    DocumentParser,
    ParsedKnowledgeDocument,
    StructuralBlock,
)


def _sample_source() -> SourceDocument:
    return SourceDocument.create_text(
        text="Sample text",
        title="Sample Knowledge Document",
    )


def test_chunking_determinism():
    source = _sample_source()
    content = """# Attention Mechanisms

Attention in deep learning mimics cognitive attention.
It allows neural networks to focus on specific subsets of inputs dynamically.

## Transformer Architecture

Transformers rely entirely on self-attention mechanisms without recurrence.
The scaled dot-product attention computes query-key similarity matrices.
Multi-head attention allows the model to jointly attend to information at different positions.
"""
    parser = DocumentParser()
    parsed = parser.parse_markdown(source, content)

    chunker = KnowledgeChunker(ChunkingPolicy(target_size=200, max_size=400, overlap=30))
    run1 = chunker.chunk_document(source, parsed)
    run2 = chunker.chunk_document(source, parsed)

    assert len(run1) > 0
    assert len(run1) == len(run2)

    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.normalized_text == c2.normalized_text
        assert c1.text_hash == c2.text_hash
        assert c1.content_fingerprint == c2.content_fingerprint
        assert c1.locator == c2.locator


def test_chunking_oversized_block_splitting():
    source = _sample_source()
    # Generate an oversized paragraph > 1000 characters
    sentences = [
        f"This is sentence number {i} containing important factual details about computing."
        for i in range(25)
    ]
    long_para = " ".join(sentences)
    assert len(long_para) > 1000

    block = StructuralBlock(
        block_index=1,
        block_type="PARAGRAPH",
        text=long_para,
        locator=EvidenceLocator(paragraph=1, section="Overview"),
    )
    parsed = ParsedKnowledgeDocument(
        source_document_id=source.source_document_id,
        processing_version="knowledge_processing_v1",
        title="Long Doc",
        blocks=(block,),
    )

    chunker = KnowledgeChunker(ChunkingPolicy(target_size=300, max_size=500, overlap=40))
    chunks = chunker.chunk_document(source, parsed)

    assert len(chunks) >= 2
    for c in chunks:
        # Every chunk should be bounded
        assert len(c.normalized_text) <= 500
        # Preserves section locator
        assert c.locator.get("section") == "Overview"


def test_chunking_empty_document():
    source = _sample_source()
    parsed = ParsedKnowledgeDocument(
        source_document_id=source.source_document_id,
        processing_version="knowledge_processing_v1",
        title="Empty",
        blocks=(),
    )
    chunker = KnowledgeChunker()
    chunks = chunker.chunk_document(source, parsed)
    assert chunks == ()


def test_chunking_preserves_locators_and_index():
    source = _sample_source()
    blocks = [
        StructuralBlock(
            block_index=1,
            block_type="PARAGRAPH",
            text="First small block on page 1.",
            locator=EvidenceLocator(page=1, paragraph=1, section="Introduction"),
        ),
        StructuralBlock(
            block_index=2,
            block_type="PARAGRAPH",
            text="Second block on page 2.",
            locator=EvidenceLocator(page=2, paragraph=2, section="Methods"),
        ),
    ]
    parsed = ParsedKnowledgeDocument(
        source_document_id=source.source_document_id,
        processing_version="knowledge_processing_v1",
        title="Two Pages",
        blocks=tuple(blocks),
    )

    # Use small target_size to ensure separate chunks
    chunker = KnowledgeChunker(ChunkingPolicy(target_size=50, max_size=100))
    chunks = chunker.chunk_document(source, parsed)

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].locator.get("page") == 1
    assert chunks[0].locator.get("section") == "Introduction"
    assert chunks[1].locator.get("page") == 2
    assert chunks[1].locator.get("section") == "Methods"
