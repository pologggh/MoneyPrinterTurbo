from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.evidence import (
    SearchResult,
    WebResearchSnapshot,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.persistence.converters import (
    web_research_snapshot_from_orm,
    web_research_snapshot_to_orm,
)
from app.persistence.models import Base, WebResearchSnapshotORM
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


def test_web_research_snapshot_persistence_and_querying(session_factory):
    """Verify persisting and querying WebResearchSnapshot records via repository."""
    task_id = f"task_{uuid4().hex[:12]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="Quantum Computing"))

        ev_repo = EvidenceRepository(session)
        snap1 = WebResearchSnapshot.create(
            task_id=task_id,
            stage_attempt=1,
            query="Quantum Computing Qubits",
            provider="TavilySearchProvider",
            search_results=[
                SearchResult(
                    result_id="res_1",
                    title="Qubit Overview",
                    url="https://example.com/qubits",
                    snippet="Introduction to superposition and entanglement",
                    provider_rank=1,
                )
            ],
            selected_urls=["https://example.com/qubits"],
            fetch_outcomes=[
                {
                    "url": "https://example.com/qubits",
                    "status": "SUCCESS",
                    "quality_tier": "PRIMARY_OFFICIAL",
                }
            ],
            created_source_document_ids=["src_q1"],
        )
        saved_snap1 = ev_repo.save_web_research_snapshot(snap1)
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        loaded = ev_repo.get_web_research_snapshot(saved_snap1.web_research_snapshot_id)
        assert loaded is not None
        assert loaded.task_id == task_id
        assert loaded.query == "Quantum Computing Qubits"
        assert loaded.provider == "TavilySearchProvider"
        assert len(loaded.search_results) == 1
        assert loaded.search_results[0].title == "Qubit Overview"
        assert loaded.selected_urls == ("https://example.com/qubits",)
        assert loaded.created_source_document_ids == ("src_q1",)
        assert loaded.content_fingerprint == saved_snap1.content_fingerprint

        all_snaps = ev_repo.list_web_research_snapshots_for_task(task_id)
        assert len(all_snaps) == 1
        assert all_snaps[0].web_research_snapshot_id == saved_snap1.web_research_snapshot_id


def test_web_research_snapshot_converter_roundtrip():
    """Verify bidirectional conversion between domain model and ORM model."""
    snap = WebResearchSnapshot.create(
        task_id="task_123",
        stage_attempt=2,
        query="Transformer Attention",
        provider="TavilySearchProvider",
        search_results=[
            SearchResult(
                result_id="r1",
                title="Attention",
                url="https://example.com/attention",
                snippet="Snippet",
                provider_rank=1,
            )
        ],
        selected_urls=["https://example.com/attention"],
        fetch_outcomes=[{"url": "https://example.com/attention", "status": "SUCCESS"}],
        created_source_document_ids=["src_100"],
    )

    orm = web_research_snapshot_to_orm(snap)
    assert orm.web_research_snapshot_id == snap.web_research_snapshot_id
    assert orm.task_id == snap.task_id
    assert orm.query == snap.query
    assert orm.provider == snap.provider

    reconstructed = web_research_snapshot_from_orm(orm)
    assert reconstructed.web_research_snapshot_id == snap.web_research_snapshot_id
    assert reconstructed.task_id == snap.task_id
    assert reconstructed.content_fingerprint == snap.content_fingerprint
    assert len(reconstructed.search_results) == 1
    assert reconstructed.search_results[0].url == "https://example.com/attention"
    assert reconstructed.selected_urls == ("https://example.com/attention",)
