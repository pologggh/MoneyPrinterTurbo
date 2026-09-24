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
from app.domain.knowledge_base import KnowledgeBase
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import TaskStatus, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import KnowledgeBaseRepository, KnowledgeVideoTaskRepository


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


def test_register_text_evidence_success(client, test_db_session):
    """POST /knowledge-video-tasks/{id}/evidence registers TEXT source with status READY."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Astrophysics", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "A black hole is a region of spacetime where gravity is so strong that nothing can escape.",
            "title": "Black Holes Intro",
            "author": "Dr. Thorne",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == 201
    assert body["message"] == "Evidence source registered"
    data = body["data"]
    assert data["source_type"] == "TEXT"
    assert data["status"] == "READY"
    assert data["title"] == "Black Holes Intro"
    assert data["source_locator"].startswith("inline:sha256:")
    assert len(data["content_hash"]) == 64
    assert len(data["source_fingerprint"]) == 64


def test_register_url_evidence_success(client, test_db_session):
    """POST /knowledge-video-tasks/{id}/evidence registers URL source with status REGISTERED."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Biology", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "URL",
            "url": "https://www.nature.com/articles/cell-mitochondria",
            "title": "Mitochondria Nature Paper",
        },
    )
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["source_type"] == "URL"
    assert data["status"] == "REGISTERED"
    assert data["source_locator"] == "https://www.nature.com/articles/cell-mitochondria"


def test_register_evidence_missing_or_invalid_fields(client, test_db_session):
    """POST /evidence returns 400 on missing required fields or invalid URL."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Genetics", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    # TEXT without text_content
    resp1 = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={"source_type": "TEXT"},
    )
    assert resp1.status_code == 400
    assert "text_content is required" in resp1.json()["detail"]

    # URL without url
    resp2 = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={"source_type": "URL"},
    )
    assert resp2.status_code == 400
    assert "url is required" in resp2.json()["detail"]

    # Invalid URL scheme
    resp3 = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={"source_type": "URL", "url": "ftp://bad-url"},
    )
    assert resp3.status_code == 400
    assert "Invalid URL" in resp3.json()["detail"]


def test_register_evidence_nonexistent_task(client):
    """POST /evidence on unknown task returns 404."""
    resp = client.post(
        "/api/v1/knowledge-video-tasks/task_nonexistent_12345/evidence",
        json={"source_type": "TEXT", "text_content": "Some text"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_register_evidence_terminal_task(client, test_db_session):
    """POST /evidence on completed or cancelled task returns 409 Conflict."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task = KnowledgeVideoTask.create(task_id=task_id, topic="Completed Topic", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    task.task_status = TaskStatus.COMPLETED
    task_repo.save_task(task)
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={"source_type": "TEXT", "text_content": "Late submission"},
    )
    assert resp.status_code == 409
    assert "terminal state" in resp.json()["detail"].lower()


def test_register_evidence_metadata_credential_redaction(client, test_db_session):
    """POST /evidence redacts sensitive keys in returned metadata."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Security", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "Security protocols and cryptographic algorithms.",
            "metadata": {
                "api_key": "super_secret_token_123",
                "custom_notes": "public research notes",
            },
        },
    )
    assert resp.status_code == 201
    meta = resp.json()["data"]["metadata"]
    assert meta["api_key"] == "***"
    assert meta["custom_notes"] == "public research notes"


def test_register_knowledge_base_source_api_success(client, test_db_session):
    """POST /evidence with KNOWLEDGE_BASE source attaches active KB to task."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="KB Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    kb_repo = KnowledgeBaseRepository(test_db_session)
    kb = KnowledgeBase.create(name="AI Systems KB")
    kb_repo.save_knowledge_base(kb)
    test_db_session.commit()
    kb_id = kb.knowledge_base_id

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "KNOWLEDGE_BASE",
            "kb_id": kb_id,
        },
    )
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["task_id"] == task_id
    assert data["knowledge_base_id"] == kb_id
    assert data["status"] == "ATTACHED"


def test_register_knowledge_base_source_api_not_found(client, test_db_session):
    """POST /evidence with missing KNOWLEDGE_BASE returns 404."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="KB Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "KNOWLEDGE_BASE",
            "kb_id": "kb_nonexistent_999",
        },
    )
    assert resp.status_code == 404


def test_register_knowledge_base_source_api_archived_conflict(client, test_db_session):
    """POST /evidence with archived KNOWLEDGE_BASE returns 409."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="KB Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    kb_repo = KnowledgeBaseRepository(test_db_session)
    kb = KnowledgeBase.create(name="Archived KB")
    kb_repo.save_knowledge_base(kb)
    kb_repo.archive_knowledge_base(kb.knowledge_base_id)
    test_db_session.commit()
    kb_id = kb.knowledge_base_id

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "KNOWLEDGE_BASE",
            "kb_id": kb_id,
        },
    )
    assert resp.status_code == 409


def test_register_url_with_whitespace_trimmed(client, test_db_session):
    """POST /evidence trims leading and trailing whitespace from URLs."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Web Research", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "URL",
            "url": "  https://example.com/trimmed   ",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["data"]["source_locator"] == "https://example.com/trimmed"


def test_register_text_empty_content_raises_400(client, test_db_session):
    """POST /evidence with whitespace-only text raises 400 Bad Request."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Empty Text", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "    ",
        },
    )
    assert resp.status_code == 400
    assert "cannot be empty" in resp.json()["detail"].lower()


def test_register_evidence_unsupported_source_type(client, test_db_session):
    """POST /evidence with synthetic or unknown source_type returns 422 Unprocessable Entity."""
    task_id = f"task_{uuid4().hex[:8]}"
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task_repo.save_task(
        KnowledgeVideoTask.create(task_id=task_id, topic="Synthetic", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
    )
    test_db_session.commit()

    resp = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "MODEL_MEMORY",
            "text_content": "Hallucinated knowledge",
        },
    )
    assert resp.status_code == 422

