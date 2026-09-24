from __future__ import annotations

from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.evidence import (
    KnowledgeChunk,
    SourceDocument,
    SourceStatus,
    SourceType,
)
from app.domain.knowledge_base import (
    ChunkEmbedding,
    KnowledgeBase,
    KnowledgeBaseStatus,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import Stage, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EmbeddingRepository,
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
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


def test_knowledge_base_crud_and_archiving(session_factory):
    with session_factory() as session:
        repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Quantum Computing", description="Papers and guides")
        saved = repo.save_knowledge_base(kb)
        session.commit()

        assert saved.knowledge_base_id == kb.knowledge_base_id
        assert saved.status == KnowledgeBaseStatus.ACTIVE

        # Fetch
        fetched = repo.get_knowledge_base(kb.knowledge_base_id)
        assert fetched is not None
        assert fetched.name == "Quantum Computing"

        # List active
        active_list = repo.list_knowledge_bases(status="ACTIVE")
        assert len(active_list) == 1

        # Archive
        repo.archive_knowledge_base(kb.knowledge_base_id)
        session.commit()

        archived = repo.get_knowledge_base(kb.knowledge_base_id)
        assert archived is not None
        assert archived.status == KnowledgeBaseStatus.ARCHIVED

        active_after = repo.list_knowledge_bases(status="ACTIVE")
        assert len(active_after) == 0


def test_knowledge_base_source_association(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        # Create KB
        kb = repo_kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="AI Safety"))

        # Create SourceDocuments
        doc1 = SourceDocument.create_text(text="Paper on alignment.", title="Doc 1")
        doc2 = SourceDocument.create_text(text="Paper on robustness.", title="Doc 2")
        ev_repo.save_source_document(doc1)
        ev_repo.save_source_document(doc2)

        # Associate
        kb_repo.associate_source(kb.knowledge_base_id, doc1.source_document_id)
        kb_repo.associate_source(kb.knowledge_base_id, doc2.source_document_id)
        session.commit()

        # List sources for KB
        sources = kb_repo.list_sources_for_kb(kb.knowledge_base_id)
        assert len(sources) == 2
        assert {s.source_document_id for s in sources} == {
            doc1.source_document_id,
            doc2.source_document_id,
        }
        assert kb_repo.count_sources_for_kb(kb.knowledge_base_id) == 2

        # Dissociate
        kb_repo.dissociate_source(kb.knowledge_base_id, doc1.source_document_id)
        session.commit()

        sources_after = kb_repo.list_sources_for_kb(kb.knowledge_base_id)
        assert len(sources_after) == 1
        assert sources_after[0].source_document_id == doc2.source_document_id


def test_task_attachment_and_merged_source_retrieval(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)

        # Create task
        task = KnowledgeVideoTask.create(
            topic="Test Topic",
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Create direct task source
        direct_doc = SourceDocument.create_text(text="Direct task note.", title="Direct Doc")
        ev_repo.save_source_document(direct_doc)
        ev_repo.associate_task_source(task.task_id, direct_doc.source_document_id)

        # Create Knowledge Base with 2 sources (one overlapping with direct)
        kb = kb_repo.save_knowledge_base(KnowledgeBase.create(name="Shared KB"))
        kb_doc2 = SourceDocument.create_text(text="Shared KB text.", title="KB Doc 2")
        ev_repo.save_source_document(kb_doc2)

        kb_repo.associate_source(kb.knowledge_base_id, direct_doc.source_document_id)
        kb_repo.associate_source(kb.knowledge_base_id, kb_doc2.source_document_id)

        # Attach KB to task
        kb_repo.attach_to_task(task.task_id, kb.knowledge_base_id)
        session.commit()

        attached_kbs = kb_repo.list_kbs_for_task(task.task_id)
        assert len(attached_kbs) == 1
        assert attached_kbs[0].knowledge_base_id == kb.knowledge_base_id

        # Query merged sources: direct_doc + kb_doc2 (direct_doc should be deduplicated!)
        merged_sources = kb_repo.list_sources_for_task_with_kbs(task.task_id)
        assert len(merged_sources) == 2
        merged_ids = [s.source_document_id for s in merged_sources]
        assert direct_doc.source_document_id in merged_ids
        assert kb_doc2.source_document_id in merged_ids

        # If KB is archived, its sources should not be included
        kb_repo.archive_knowledge_base(kb.knowledge_base_id)
        session.commit()

        merged_after_archive = kb_repo.list_sources_for_task_with_kbs(task.task_id)
        assert len(merged_after_archive) == 1
        assert merged_after_archive[0].source_document_id == direct_doc.source_document_id


def test_embedding_repository_operations(session_factory):
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        emb_repo = EmbeddingRepository(session)

        # Create document & chunk
        doc = SourceDocument.create_text(text="Deep learning fundamentals.", title="DL Doc")
        ev_repo.save_source_document(doc)

        chunk1 = KnowledgeChunk.create(
            source_document=doc,
            normalized_text="Deep learning fundamentals introduction.",
            chunk_index=0,
        )
        chunk2 = KnowledgeChunk.create(
            source_document=doc,
            normalized_text="Backpropagation algorithm details.",
            chunk_index=1,
        )
        ev_repo.save_knowledge_chunks([chunk1, chunk2])
        session.commit()

        # Save embeddings
        emb1 = ChunkEmbedding.create(
            chunk_id=chunk1.chunk_id,
            provider="fake_provider",
            model="fake_model",
            vector=[0.1, 0.2, 0.3],
            text_hash=chunk1.text_hash,
        )
        emb2 = ChunkEmbedding.create(
            chunk_id=chunk2.chunk_id,
            provider="fake_provider",
            model="fake_model",
            vector=[0.4, 0.5, 0.6],
            text_hash=chunk2.text_hash,
        )
        emb_repo.save_chunk_embeddings([emb1, emb2])
        session.commit()

        # Retrieve single embedding
        fetched = emb_repo.get_chunk_embedding(chunk1.chunk_id, "fake_provider", "fake_model")
        assert fetched is not None
        assert fetched.dimension == 3
        assert fetched.vector == (0.1, 0.2, 0.3)

        # List by chunk IDs
        chunks_embs = emb_repo.list_embeddings_for_chunks(
            [chunk1.chunk_id, chunk2.chunk_id],
            "fake_provider",
            "fake_model",
        )
        assert len(chunks_embs) == 2

        # List by source document IDs
        source_embs = emb_repo.list_embeddings_for_sources(
            [doc.source_document_id],
            "fake_provider",
            "fake_model",
        )
        assert len(source_embs) == 2
