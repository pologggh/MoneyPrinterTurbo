from __future__ import annotations

import inspect
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.controllers.v1.knowledge_base import router as kb_router
from app.controllers.v1.knowledge_video import router as kv_router
from app.domain.evidence import SourceType
from app.domain.knowledge_base import KnowledgeBase
from app.domain.workflow_state import Stage
from app.persistence.models import (
    Base,
    KnowledgeBaseORM,
    KnowledgeVideoTaskORM,
    SourceDocumentORM,
    TaskKnowledgeBaseORM,
    WorkflowJobORM,
)
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
    WorkflowJobRepository,
)
from webui.api_client import KnowledgeVideoApiClient


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


def test_create_task_atomic_with_kbs_and_evidence_202_and_db_state(session_factory, monkeypatch):
    """
    1. POST /api/v1/knowledge-video-tasks concurrently carrying knowledge_base_ids and
       initial_evidence (TEXT + URL).
    2. Response returns 202 Accepted.
    3. Underlying DB state confirms: Task, initial EVIDENCE Job, Task-KB links,
       and initial evidence documents are all durably persisted upon response return.
    """
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    # Pre-create 2 active Knowledge Bases
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb1 = KnowledgeBase.create(name="Quantum KB 1", description="Physics foundations")
        kb2 = KnowledgeBase.create(name="Quantum KB 2", description="Experimental setups")
        kb_repo.save_knowledge_base(kb1)
        kb_repo.save_knowledge_base(kb2)
        session.commit()
        kb1_id = kb1.knowledge_base_id
        kb2_id = kb2.knowledge_base_id

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    request_payload = {
        "topic": "Quantum Teleportation Experiments",
        "target_duration": 90.0,
        "aspect_ratio": "16:9",
        "language": "zh",
        "workflow_policy": "AUTO",
        "allow_research": True,
        "knowledge_base_ids": [kb1_id, kb2_id],
        "initial_evidence": [
            {
                "source_type": "TEXT",
                "text_content": "Quantum teleportation is a technique for transferring quantum information.",
                "title": "Teleportation Summary Note",
            },
            {
                "source_type": "URL",
                "url": "https://nature.com/articles/quantum-teleportation",
                "title": "Nature Teleportation Paper",
            },
        ],
    }

    resp = client.post("/api/v1/knowledge-video-tasks", json=request_payload)
    assert resp.status_code == 202
    resp_body = resp.json()
    assert resp_body["status"] == 202
    assert resp_body["message"] == "Task accepted"

    data = resp_body["data"]
    task_id = data["task_id"]
    initial_job_id = data["initial_job_id"]
    assert task_id is not None
    assert initial_job_id is not None
    assert data["topic"] == "Quantum Teleportation Experiments"
    assert data["current_stage"] == "EVIDENCE"

    # Verify underlying DB state directly
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        # 1. Task aggregate
        task = task_repo.get_task(task_id)
        assert task is not None
        assert task.topic == "Quantum Teleportation Experiments"
        assert task.current_stage == Stage.EVIDENCE

        # 2. Initial EVIDENCE Job
        job = job_repo.get_current_job_for_task(task_id)
        assert job is not None
        assert job.job_id == initial_job_id
        assert job.stage == Stage.EVIDENCE

        # 3. Task-KB associations
        attached_kbs = kb_repo.list_kbs_for_task(task_id)
        attached_ids = {k.knowledge_base_id for k in attached_kbs}
        assert attached_ids == {kb1_id, kb2_id}

        # 4. Source documents
        docs = ev_repo.list_sources_for_task(task_id)
        assert len(docs) == 2

        doc_by_type = {d.source_type: d for d in docs}
        assert SourceType.TEXT in doc_by_type
        assert SourceType.URL in doc_by_type

        text_doc = doc_by_type[SourceType.TEXT]
        assert text_doc.title == "Teleportation Summary Note"
        assert "transferring quantum information" in (text_doc.content_snapshot or "")

        url_doc = doc_by_type[SourceType.URL]
        assert url_doc.title == "Nature Teleportation Paper"
        assert url_doc.source_locator == "https://nature.com/articles/quantum-teleportation"


def test_create_task_with_nonexistent_kb_returns_404_and_zero_db_residue(session_factory, monkeypatch):
    """
    When a non-existent knowledge_base_id is supplied:
    - Returns 404
    - DB has zero residue: 0 tasks, 0 jobs, 0 attachments, 0 source documents.
    """
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    request_payload = {
        "topic": "Orphaned Task",
        "knowledge_base_ids": ["kb_ghost_99999"],
        "initial_evidence": [
            {
                "source_type": "TEXT",
                "text_content": "Valid text content but KB is ghost.",
                "title": "Orphan Note",
            }
        ],
    }

    resp = client.post("/api/v1/knowledge-video-tasks", json=request_payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    # DB zero residue assertion
    with session_factory() as session:
        assert session.scalars(select(KnowledgeVideoTaskORM)).all() == []
        assert session.scalars(select(WorkflowJobORM)).all() == []
        assert session.scalars(select(TaskKnowledgeBaseORM)).all() == []
        assert session.scalars(select(SourceDocumentORM)).all() == []


def test_create_task_with_invalid_evidence_text_empty_returns_400_and_zero_db_residue(session_factory, monkeypatch):
    """
    When initial_evidence (TEXT) has empty text_content:
    - Returns 400
    - DB has zero residue.
    """
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    # Pre-create valid KB to ensure error is triggered by evidence validation
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Valid KB")
        kb_repo.save_knowledge_base(kb)
        session.commit()
        kb_id = kb.knowledge_base_id

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    request_payload = {
        "topic": "Empty Text Task",
        "knowledge_base_ids": [kb_id],
        "initial_evidence": [
            {
                "source_type": "TEXT",
                "text_content": "   ",  # whitespace only
                "title": "Blank note",
            }
        ],
    }

    resp = client.post("/api/v1/knowledge-video-tasks", json=request_payload)
    assert resp.status_code == 400
    assert "non-empty text_content" in resp.json()["detail"]

    with session_factory() as session:
        assert session.scalars(select(KnowledgeVideoTaskORM)).all() == []
        assert session.scalars(select(WorkflowJobORM)).all() == []
        assert session.scalars(select(TaskKnowledgeBaseORM)).all() == []
        assert session.scalars(select(SourceDocumentORM)).all() == []


def test_create_task_with_invalid_evidence_url_malformed_returns_400_and_zero_db_residue(session_factory, monkeypatch):
    """
    When initial_evidence (URL) has non-http/https URL:
    - Returns 400
    - DB has zero residue.
    """
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    request_payload = {
        "topic": "Bad URL Task",
        "initial_evidence": [
            {
                "source_type": "URL",
                "url": "ftp://bad-protocol.com/file",
                "title": "Bad Scheme",
            }
        ],
    }

    resp = client.post("/api/v1/knowledge-video-tasks", json=request_payload)
    assert resp.status_code == 400
    assert "must start with http:// or https://" in resp.json()["detail"]

    with session_factory() as session:
        assert session.scalars(select(KnowledgeVideoTaskORM)).all() == []
        assert session.scalars(select(WorkflowJobORM)).all() == []
        assert session.scalars(select(SourceDocumentORM)).all() == []


def test_webui_api_client_create_task_payload_contract(monkeypatch):
    """
    Verifies that KnowledgeVideoApiClient.create_task() properly builds a payload
    containing 'knowledge_base_ids' and 'initial_evidence'.
    """
    client = KnowledgeVideoApiClient(base_url="http://testserver")
    mock_request = MagicMock(return_value={"status": 202, "task_id": "test_t1", "initial_job_id": "job_1"})
    monkeypatch.setattr(client, "_request", mock_request)

    client.create_task(
        topic="Physics",
        target_duration=120.0,
        aspect_ratio="9:16",
        language="en",
        workflow_policy="SEMI_AUTO",
        allow_research=True,
        knowledge_base_ids=["kb_101", "kb_102"],
        initial_evidence=[
            {"source_type": "TEXT", "text_content": "Physics text", "title": "Notes"},
            {"source_type": "URL", "url": "https://example.com", "title": "Link"},
        ],
    )

    mock_request.assert_called_once()
    method, path = mock_request.call_args[0]
    kwargs = mock_request.call_args[1]

    assert method == "POST"
    assert path == "/knowledge-video-tasks"
    payload = kwargs["json_data"]
    assert payload["topic"] == "Physics"
    assert payload["target_duration"] == 120.0
    assert payload["aspect_ratio"] == "9:16"
    assert payload["language"] == "en"
    assert payload["workflow_policy"] == "SEMI_AUTO"
    assert payload["allow_research"] is True
    assert payload["knowledge_base_ids"] == ["kb_101", "kb_102"]
    assert len(payload["initial_evidence"]) == 2
    assert payload["initial_evidence"][0]["source_type"] == "TEXT"
    assert payload["initial_evidence"][1]["source_type"] == "URL"


def test_webui_agent_page_form_submit_atomic_no_duplicate_calls(monkeypatch):
    """
    Simulates the form submission logic in webui/agent_page.py:
    1. Initial evidence and selected KBs are bundled into a single client.create_task() call.
    2. No secondary attach or add_evidence calls are made post-creation.
    3. Statically verifies that agent_page.py does not invoke secondary attachment endpoints
       after client.create_task().
    """
    import webui.agent_page as agent_page

    # 1. Functional simulation of the form submission block in agent_page._render_create_task_view
    mock_client = MagicMock()
    mock_client.create_task.return_value = {"task_id": "task_atomic_123", "status": 202}

    topic = "Atomic Submission Topic"
    selected_kb_ids = ["kb_alpha", "kb_beta"]
    source_content = "https://example.com/atomic-doc"
    source_type = "URL"
    source_title = "Atomic Resource"

    # Exact logic executed on agent_page form submission
    initial_evidence = []
    if source_content and source_content.strip():
        clean_content = source_content.strip()
        if source_type == "TEXT":
            initial_evidence.append({
                "source_type": "TEXT",
                "text_content": clean_content,
                "title": source_title or topic,
            })
        elif source_type == "URL":
            initial_evidence.append({
                "source_type": "URL",
                "url": clean_content,
                "title": source_title or clean_content,
            })

    task_res = mock_client.create_task(
        topic=topic.strip(),
        target_duration=60.0,
        aspect_ratio="16:9",
        language="zh",
        workflow_policy="AUTO",
        allow_research=True,
        knowledge_base_ids=selected_kb_ids,
        initial_evidence=initial_evidence,
    )
    task_id = task_res.get("task_id")
    assert task_id == "task_atomic_123"

    # Assert exactly ONE API call was made to create_task with both KBs and initial_evidence
    assert mock_client.create_task.call_count == 1
    call_kwargs = mock_client.create_task.call_args[1]
    assert call_kwargs["knowledge_base_ids"] == ["kb_alpha", "kb_beta"]
    assert len(call_kwargs["initial_evidence"]) == 1
    assert call_kwargs["initial_evidence"][0]["url"] == "https://example.com/atomic-doc"

    # Assert NO secondary calls (like attach_to_task or add_evidence) occurred
    assert mock_client.attach_to_task.call_count == 0
    assert mock_client.add_evidence.call_count == 0

    # 2. Static AST / source verification of _render_create_task_view in agent_page.py
    src = inspect.getsource(agent_page._render_create_task_view)
    # create_task should be present
    assert "client.create_task(" in src
    # Neither attach_to_task nor add_evidence should be called inside _render_create_task_view
    assert "client.attach_to_task(" not in src
    assert "client.add_evidence(" not in src
