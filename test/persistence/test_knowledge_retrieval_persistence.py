from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.evidence import (
    EvidenceLocator,
    KnowledgeChunk,
    RetrievalCandidate,
    RetrievalSnapshot,
    SourceDocument,
    SourceType,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
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


def test_knowledge_chunks_persistence_and_querying(session_factory):
    """Verify persisting and querying KnowledgeChunk records."""
    source_a = SourceDocument.create_text(text="Source A content", title="Doc A")
    source_b = SourceDocument.create_text(text="Source B content", title="Doc B")

    chunk_a0 = KnowledgeChunk.create(
        source_document=source_a,
        normalized_text="Chunk A-0 content about transformers.",
        chunk_index=0,
        locator=EvidenceLocator(paragraph=1, section="Intro"),
    )
    chunk_a1 = KnowledgeChunk.create(
        source_document=source_a,
        normalized_text="Chunk A-1 content about self-attention.",
        chunk_index=1,
        locator=EvidenceLocator(paragraph=2, section="Deep Dive"),
    )
    chunk_b0 = KnowledgeChunk.create(
        source_document=source_b,
        normalized_text="Chunk B-0 content about computer networks.",
        chunk_index=0,
        locator=EvidenceLocator(paragraph=1, section="Protocols"),
    )

    with session_factory() as session:
        repo = EvidenceRepository(session)
        repo.save_source_document(source_a)
        repo.save_source_document(source_b)
        repo.save_knowledge_chunks([chunk_a0, chunk_a1, chunk_b0])
        session.commit()

    with session_factory() as session:
        repo = EvidenceRepository(session)

        # Query single source
        chunks_a = repo.list_chunks_for_source(source_a.source_document_id)
        assert len(chunks_a) == 2
        assert chunks_a[0].chunk_id == chunk_a0.chunk_id
        assert chunks_a[0].chunk_index == 0
        assert chunks_a[0].locator.get("section") == "Intro"
        assert chunks_a[1].chunk_id == chunk_a1.chunk_id
        assert chunks_a[1].chunk_index == 1

        # Query multiple sources
        chunks_both = repo.list_chunks_for_sources([source_a.source_document_id, source_b.source_document_id])
        assert len(chunks_both) == 3

        # Idempotent re-save
        re_saved = repo.save_knowledge_chunks([chunk_a0])
        session.commit()
        assert len(re_saved) == 1
        assert re_saved[0].chunk_id == chunk_a0.chunk_id

        # Total count still 2 for source A
        assert len(repo.list_chunks_for_source(source_a.source_document_id)) == 2


def test_task_scoping_chunks_isolation(session_factory):
    """Verify list_chunks_for_task strictly adheres to task associations."""
    task_1 = f"task_{uuid4().hex[:8]}"
    task_2 = f"task_{uuid4().hex[:8]}"

    source_1 = SourceDocument.create_text(text="Content for Task 1", title="Task 1 Doc")
    source_2 = SourceDocument.create_text(text="Content for Task 2", title="Task 2 Doc")

    chunk_1 = KnowledgeChunk.create(
        source_document=source_1,
        normalized_text="Attention is all you need.",
        chunk_index=0,
    )
    chunk_2 = KnowledgeChunk.create(
        source_document=source_2,
        normalized_text="DNS maps hostnames to IP addresses.",
        chunk_index=0,
    )

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_1, topic="AI", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_2, topic="Networking", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))

        ev_repo = EvidenceRepository(session)
        ev_repo.save_source_document(source_1)
        ev_repo.save_source_document(source_2)
        ev_repo.associate_task_source(task_id=task_1, source_document_id=source_1.source_document_id)
        ev_repo.associate_task_source(task_id=task_2, source_document_id=source_2.source_document_id)
        ev_repo.save_knowledge_chunks([chunk_1, chunk_2])
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)

        task_1_chunks = ev_repo.list_chunks_for_task(task_1)
        assert len(task_1_chunks) == 1
        assert task_1_chunks[0].chunk_id == chunk_1.chunk_id
        assert "Attention" in task_1_chunks[0].normalized_text

        task_2_chunks = ev_repo.list_chunks_for_task(task_2)
        assert len(task_2_chunks) == 1
        assert task_2_chunks[0].chunk_id == chunk_2.chunk_id
        assert "DNS" in task_2_chunks[0].normalized_text


def test_retrieval_snapshot_persistence(session_factory):
    """Verify saving and retrieving RetrievalSnapshot with candidate rankings."""
    task_id = f"task_{uuid4().hex[:8]}"

    candidate_1 = RetrievalCandidate(
        rank=1,
        chunk_id="chk_001",
        source_document_id="src_001",
        score=2.45,
        retrieval_method="LEXICAL_BM25",
        locator={"paragraph": 1},
        excerpt="Key transformer sentence.",
    )
    candidate_2 = RetrievalCandidate(
        rank=2,
        chunk_id="chk_002",
        source_document_id="src_001",
        score=1.12,
        retrieval_method="LEXICAL_BM25",
        locator={"paragraph": 2},
        excerpt="Second transformer sentence.",
    )

    snapshot = RetrievalSnapshot.create(
        task_id=task_id,
        query="transformer attention",
        source_scope_ids=["src_001"],
        candidates=[candidate_1, candidate_2],
        selected_evidence_ids=["evi_001"],
    )

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))

        ev_repo = EvidenceRepository(session)
        ev_repo.save_retrieval_snapshot(snapshot)
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)

        loaded = ev_repo.get_retrieval_snapshot(snapshot.retrieval_snapshot_id)
        assert loaded is not None
        assert loaded.task_id == task_id
        assert loaded.query == "transformer attention"
        assert loaded.source_scope_ids == ("src_001",)
        assert len(loaded.candidates) == 2
        assert loaded.candidates[0].chunk_id == "chk_001"
        assert loaded.candidates[0].score == 2.45
        assert loaded.candidates[0].rank == 1
        assert loaded.selected_evidence_ids == ("evi_001",)
        assert loaded.content_fingerprint == snapshot.content_fingerprint

        task_snapshots = ev_repo.list_retrieval_snapshots_for_task(task_id)
        assert len(task_snapshots) == 1
        assert task_snapshots[0].retrieval_snapshot_id == snapshot.retrieval_snapshot_id


def test_alembic_migration_0017_upgrade_downgrade(tmp_path, monkeypatch):
    """Verify Alembic migration 0017 upgrades from 0016, downgrades, and re-upgrades cleanly."""
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "test_migration_0017.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    project_root = Path(__file__).resolve().parent.parent.parent

    alembic_ini_path = project_root / "alembic.ini"
    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(project_root / "migrations"))

    # Upgrade to 0016 first
    command.upgrade(cfg, "0016_evidence_provenance_foundation")

    # Upgrade to 0017
    command.upgrade(cfg, "0017_knowledge_processing_and_retrieval")

    # Downgrade back to 0016
    command.downgrade(cfg, "0016_evidence_provenance_foundation")

    # Re-upgrade to 0017
    command.upgrade(cfg, "0017_knowledge_processing_and_retrieval")
