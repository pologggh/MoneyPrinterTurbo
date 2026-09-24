from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import config
from app.controllers.v1.knowledge_video import router
from app.domain.workflow_state import TaskStatus
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


def test_create_task_returns_202_accepted(client):
    """POST /api/v1/knowledge-video-tasks returns 202 Accepted with task and job IDs."""
    response = client.post(
        "/api/v1/knowledge-video-tasks",
        json={
            "topic": "Quantum Computing",
            "target_duration": 90.0,
            "workflow_policy": "AUTO",
        },
    )
    assert response.status_code == 202
    data = response.json()["data"]
    assert "task_id" in data
    assert data["topic"] == "Quantum Computing"
    assert data["task_status"] == "CREATED"
    assert data["current_stage"] == "EVIDENCE"
    assert "initial_job_id" in data


def test_create_task_validation_error_returns_422(client):
    """Validation errors on request payload return 422 Unprocessable Entity."""
    # Empty topic
    response = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": ""},
    )
    assert response.status_code == 422

    # Negative duration
    response = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Physics", "target_duration": -10.0},
    )
    assert response.status_code == 422


def test_get_task_returns_200_and_404(client):
    """GET /api/v1/knowledge-video-tasks/{id} returns 200 or 404."""
    # 404 Not Found
    res404 = client.get("/api/v1/knowledge-video-tasks/nonexistent_id")
    assert res404.status_code == 404

    # Create task
    create_res = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Relativity"},
    )
    task_id = create_res.json()["data"]["task_id"]

    # 200 OK
    res200 = client.get(f"/api/v1/knowledge-video-tasks/{task_id}")
    assert res200.status_code == 200
    detail = res200.json()["data"]
    assert detail["task_id"] == task_id
    assert detail["topic"] == "Relativity"
    assert detail["current_job"] is not None


def test_approve_task_success_and_conflict(client, test_db_session):
    """POST .../approve returns 200 when WAITING_USER, 409 otherwise."""
    create_res = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Gene Editing", "workflow_policy": "REVIEW"},
    )
    task_id = create_res.json()["data"]["task_id"]

    # In CREATED state -> approve should return 409
    res_conflict = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/approve")
    assert res_conflict.status_code == 409

    # Transition task to WAITING_USER in DB
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task = task_repo.get_task(task_id)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.WAITING_USER, reason="Review required after KNOWLEDGE_PLAN")
    task_repo.save_task(task)
    test_db_session.commit()

    # Now approve succeeds
    res_approve = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/approve")
    assert res_approve.status_code == 200
    data = res_approve.json()["data"]
    assert data["task_status"] == "RUNNING"


def test_retry_task_success_and_conflict(client, test_db_session):
    """POST .../retry returns 200 when recoverable, 409 when invalid."""
    create_res = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Robotics"},
    )
    task_id = create_res.json()["data"]["task_id"]

    # In CREATED state -> retry should return 409
    res_conflict = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/retry")
    assert res_conflict.status_code == 409

    # Transition to NEEDS_RECOVERY in DB
    task_repo = KnowledgeVideoTaskRepository(test_db_session)
    task = task_repo.get_task(task_id)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.NEEDS_RECOVERY, reason="Recoverable timeout")
    task_repo.save_task(task)
    test_db_session.commit()

    # Retry succeeds
    res_retry = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/retry")
    assert res_retry.status_code == 200
    data = res_retry.json()["data"]
    assert data["task_status"] == "RUNNING"


def test_cancel_task_success_and_terminal_immutability(client):
    """POST .../cancel cancels task; subsequent cancellation returns 409."""
    create_res = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Astrobiology"},
    )
    task_id = create_res.json()["data"]["task_id"]

    # Cancel
    cancel_res = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/cancel",
        json={"reason": "User cancelled"},
    )
    assert cancel_res.status_code == 200
    assert cancel_res.json()["data"]["task_status"] == "CANCELLED"

    # Second cancel -> 409 Conflict (terminal state immutable)
    cancel_again = client.post(f"/api/v1/knowledge-video-tasks/{task_id}/cancel")
    assert cancel_again.status_code == 409


def test_sensitive_credentials_redacted(client):
    """Verifies that API keys or auth tokens in task_metadata are redacted in responses."""
    create_res = client.post(
        "/api/v1/knowledge-video-tasks",
        json={
            "topic": "Security Testing",
            "task_metadata": {
                "api_key": "sk-secret-key-123456",
                "secret_token": "token-abcdef",
                "regular_field": "safe_value",
            },
        },
    )
    assert create_res.status_code == 202
    task_id = create_res.json()["data"]["task_id"]

    detail_res = client.get(f"/api/v1/knowledge-video-tasks/{task_id}")
    assert detail_res.status_code == 200
    metadata = detail_res.json()["data"]["task_metadata"]

    assert metadata["regular_field"] == "safe_value"
    assert "sk-secret-key-123456" not in str(metadata)
    assert metadata["api_key"] == "***"
    assert metadata["secret_token"] == "***"
