from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import config
from app.controllers.v1.knowledge_video import router
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import KnowledgeVideoTaskRepository


@pytest.fixture
def test_db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    session = session_factory()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db_session, monkeypatch):
    config.app["api_key"] = ""

    @contextmanager
    def mock_get_session():
        yield test_db_session

    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", mock_get_session)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_process_and_retrieve_knowledge_api(client, test_db_session):
    """Full API test: register evidence -> process chunks -> retrieve -> list snapshots & chunks."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Deep Learning", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    # 1. Register evidence
    reg_resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "Transformers use self-attention to process entire sequences in parallel.",
            "title": "Transformer Notes",
        },
    )
    assert reg_resp.status_code == 201

    # 2. Process knowledge sources
    proc_resp = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/knowledge/process")
    assert proc_resp.status_code == 200
    proc_data = proc_resp.json()["data"]
    assert proc_data["task_id"] == task_id
    assert proc_data["processed_sources"] == 1
    assert proc_data["total_chunks"] >= 1

    # 3. Retrieve knowledge
    ret_resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/knowledge/retrieve",
        json={
            "query": "self-attention parallel sequences",
            "top_k": 5,
        },
    )
    assert ret_resp.status_code == 200
    ret_data = ret_resp.json()["data"]
    assert ret_data["task_id"] == task_id
    assert ret_data["candidate_count"] >= 1
    assert len(ret_data["candidates"]) >= 1
    assert ret_data["candidates"][0]["score"] > 0
    assert len(ret_data["selected_evidence_ids"]) >= 1

    # 4. List retrieval snapshots
    list_ret_resp = client.get(f"/api/v1/knowledge-video-tasks/{task_id}/knowledge/retrievals")
    assert list_ret_resp.status_code == 200
    snaps = list_ret_resp.json()["data"]
    assert len(snaps) == 1
    assert snaps[0]["retrieval_snapshot_id"] == ret_data["retrieval_snapshot_id"]

    # 5. List chunks
    list_chunks_resp = client.get(f"/api/v1/knowledge-video-tasks/{task_id}/knowledge/chunks")
    assert list_chunks_resp.status_code == 200
    chunks = list_chunks_resp.json()["data"]
    assert len(chunks) >= 1
    assert "self-attention" in chunks[0]["normalized_text"]


def test_retrieve_knowledge_not_found(client):
    """POST /knowledge/retrieve returns 404 for non-existent task."""
    resp = client.post(
        "/api/v1/knowledge-video-tasks/task_nonexistent/knowledge/retrieve",
        json={"query": "hello"},
    )
    assert resp.status_code == 404
