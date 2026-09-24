from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import hashlib
from typing import Any, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseStatus(str, Enum):
    """Lifecycle status for a Knowledge Base."""
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class KnowledgeBase(BaseModel):
    """Domain model representing a persistent Knowledge Base collection."""
    model_config = ConfigDict(frozen=True)

    knowledge_base_id: str
    name: str
    description: str | None = None
    status: KnowledgeBaseStatus = KnowledgeBaseStatus.ACTIVE
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        name: str,
        description: str | None = None,
        metadata_json: dict[str, Any] | None = None,
        knowledge_base_id: str | None = None,
        now: datetime | None = None,
    ) -> KnowledgeBase:
        ts = now or datetime.now(UTC)
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Knowledge base name cannot be empty.")
        return cls(
            knowledge_base_id=knowledge_base_id or f"kb_{uuid4().hex[:24]}",
            name=cleaned_name,
            description=description.strip() if description else None,
            status=KnowledgeBaseStatus.ACTIVE,
            metadata_json=dict(metadata_json or {}),
            created_at=ts,
            updated_at=ts,
        )


class KnowledgeBaseSource(BaseModel):
    """Domain association between a Knowledge Base and a SourceDocument."""
    model_config = ConfigDict(frozen=True)

    knowledge_base_id: str
    source_document_id: str
    associated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskKnowledgeBase(BaseModel):
    """Domain association between a Knowledge Video Task and an attached Knowledge Base."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    knowledge_base_id: str
    attached_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def compute_embedding_id(chunk_id: str, provider: str, model: str) -> str:
    """Computes a deterministic unique identifier for a chunk embedding record."""
    payload = f"{chunk_id}:{provider.strip()}:{model.strip()}".encode("utf-8")
    return f"emb_{hashlib.sha256(payload).hexdigest()[:24]}"


class ChunkEmbedding(BaseModel):
    """Domain record representing a semantic vector embedding for a KnowledgeChunk."""
    model_config = ConfigDict(frozen=True)

    embedding_id: str
    chunk_id: str
    provider: str
    model: str
    dimension: int
    vector: tuple[float, ...]
    text_hash: str
    retrieval_policy_version: str = "vector_v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        chunk_id: str,
        provider: str,
        model: str,
        vector: Sequence[float],
        text_hash: str,
        retrieval_policy_version: str = "vector_v1",
        embedding_id: str | None = None,
        now: datetime | None = None,
    ) -> ChunkEmbedding:
        ts = now or datetime.now(UTC)
        vec_tuple = tuple(float(v) for v in vector)
        dim = len(vec_tuple)
        if dim == 0:
            raise ValueError("Embedding vector cannot be empty.")
        emb_id = embedding_id or compute_embedding_id(chunk_id, provider, model)
        return cls(
            embedding_id=emb_id,
            chunk_id=chunk_id,
            provider=provider.strip(),
            model=model.strip(),
            dimension=dim,
            vector=vec_tuple,
            text_hash=text_hash,
            retrieval_policy_version=retrieval_policy_version,
            created_at=ts,
        )
