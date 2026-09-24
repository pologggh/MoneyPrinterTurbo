from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.knowledge_base import (
    ChunkEmbedding,
    KnowledgeBase,
    KnowledgeBaseSource,
    KnowledgeBaseStatus,
    TaskKnowledgeBase,
    compute_embedding_id,
)


def test_knowledge_base_creation_and_defaults():
    kb = KnowledgeBase.create(name="  Physics 101  ", description="General physics lecture notes")
    assert kb.name == "Physics 101"
    assert kb.description == "General physics lecture notes"
    assert kb.status == KnowledgeBaseStatus.ACTIVE
    assert kb.knowledge_base_id.startswith("kb_")
    assert kb.created_at is not None
    assert kb.updated_at is not None


def test_knowledge_base_empty_name_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        KnowledgeBase.create(name="   ")


def test_knowledge_base_immutability():
    kb = KnowledgeBase.create(name="AI Systems")
    with pytest.raises(ValidationError):
        kb.name = "New Name"  # type: ignore


def test_knowledge_base_source_and_task_kb_creation():
    kbs = KnowledgeBaseSource(knowledge_base_id="kb_123", source_document_id="src_456")
    assert kbs.knowledge_base_id == "kb_123"
    assert kbs.source_document_id == "src_456"
    assert kbs.associated_at is not None

    tkb = TaskKnowledgeBase(task_id="task_abc", knowledge_base_id="kb_123")
    assert tkb.task_id == "task_abc"
    assert tkb.knowledge_base_id == "kb_123"
    assert tkb.attached_at is not None


def test_chunk_embedding_creation_and_validation():
    emb = ChunkEmbedding.create(
        chunk_id="chk_001",
        provider="openai",
        model="text-embedding-3-small",
        vector=[0.1, -0.2, 0.5, 0.0],
        text_hash="abc123hash",
    )
    assert emb.dimension == 4
    assert emb.vector == (0.1, -0.2, 0.5, 0.0)
    assert emb.provider == "openai"
    assert emb.model == "text-embedding-3-small"
    assert emb.embedding_id == compute_embedding_id("chk_001", "openai", "text-embedding-3-small")

    # Empty vector should be rejected
    with pytest.raises(ValueError, match="cannot be empty"):
        ChunkEmbedding.create(
            chunk_id="chk_001",
            provider="openai",
            model="text-embedding-3-small",
            vector=[],
            text_hash="abc123hash",
        )
