from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from webui.api_client import ApiClientError, KnowledgeVideoApiClient


@pytest.fixture
def client():
    return KnowledgeVideoApiClient(base_url="http://testserver/api/v1", api_key="secret-token")


def test_create_knowledge_base(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "data": {
                "knowledge_base_id": "kb_123",
                "name": "Quantum Physics",
                "description": "Quantum notes",
                "status": "ACTIVE",
            }
        }
        mock_req.return_value = mock_resp

        res = client.create_knowledge_base(name="Quantum Physics", description="Quantum notes")
        assert res["knowledge_base_id"] == "kb_123"
        assert res["name"] == "Quantum Physics"

        mock_req.assert_called_once()
        args, kwargs = mock_req.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "http://testserver/api/v1/knowledge-bases"
        assert kwargs["json"] == {"name": "Quantum Physics", "description": "Quantum notes"}
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["x-api-key"] == "secret-token"


def test_list_knowledge_bases(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "items": [
                {"knowledge_base_id": "kb_1", "name": "KB 1", "document_count": 5},
                {"knowledge_base_id": "kb_2", "name": "KB 2", "document_count": 0},
            ]
        }
        mock_req.return_value = mock_resp

        res = client.list_knowledge_bases(status="ACTIVE")
        assert len(res) == 2
        assert res[0]["knowledge_base_id"] == "kb_1"
        assert res[0]["document_count"] == 5

        args, kwargs = mock_req.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["params"] == {"status": "ACTIVE"}


def test_get_knowledge_base(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "data": {"knowledge_base_id": "kb_456", "name": "Detailed KB"}
        }
        mock_req.return_value = mock_resp

        res = client.get_knowledge_base("kb_456")
        assert res["knowledge_base_id"] == "kb_456"
        assert mock_req.call_args[1]["url"] == "http://testserver/api/v1/knowledge-bases/kb_456"


def test_update_knowledge_base(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "data": {"knowledge_base_id": "kb_456", "name": "Updated KB Name"}
        }
        mock_req.return_value = mock_resp

        res = client.update_knowledge_base("kb_456", name="Updated KB Name")
        assert res["name"] == "Updated KB Name"
        assert mock_req.call_args[1]["method"] == "PATCH"
        assert mock_req.call_args[1]["json"] == {"name": "Updated KB Name"}


def test_archive_knowledge_base(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "ARCHIVED", "knowledge_base_id": "kb_456"}
        mock_req.return_value = mock_resp

        res = client.archive_knowledge_base("kb_456")
        assert res["status"] == "ARCHIVED"
        assert mock_req.call_args[1]["method"] == "DELETE"


def test_upload_knowledge_base_document(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "data": {
                "source_document_id": "doc_789",
                "title": "Quantum Doc",
                "status": "READY",
                "chunk_count": 4,
            }
        }
        mock_req.return_value = mock_resp

        file_content = b"Quantum physics lecture content."
        res = client.upload_knowledge_base_document(
            kb_id="kb_123",
            file_name="quantum.txt",
            file_bytes=file_content,
            content_type="text/plain",
            title="Quantum Doc",
        )
        assert res["source_document_id"] == "doc_789"
        assert res["chunk_count"] == 4

        args, kwargs = mock_req.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == "http://testserver/api/v1/knowledge-bases/kb_123/documents"
        assert "files" in kwargs
        assert kwargs["files"]["file"] == ("quantum.txt", file_content, "text/plain")
        assert kwargs["data"] == {"title": "Quantum Doc"}
        # For multipart file uploads, Content-Type should not be set to application/json
        assert "Content-Type" not in kwargs["headers"]
        assert kwargs["headers"]["x-api-key"] == "secret-token"


def test_list_knowledge_base_documents(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "items": [
                {"source_document_id": "doc_1", "status": "READY", "chunk_count": 3},
                {"source_document_id": "doc_2", "status": "FAILED", "chunk_count": 0},
            ]
        }
        mock_req.return_value = mock_resp

        docs = client.list_knowledge_base_documents("kb_123")
        assert len(docs) == 2
        assert docs[0]["status"] == "READY"
        assert docs[1]["status"] == "FAILED"


def test_retry_knowledge_base_document(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "data": {"source_document_id": "doc_2", "status": "READY", "chunk_count": 2}
        }
        mock_req.return_value = mock_resp

        res = client.retry_knowledge_base_document("kb_123", "doc_2")
        assert res["status"] == "READY"
        assert res["chunk_count"] == 2
        assert mock_req.call_args[1]["url"] == "http://testserver/api/v1/knowledge-bases/kb_123/documents/doc_2/retry"


def test_attach_and_detach_knowledge_base_to_task(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"status": "ATTACHED", "task_id": "t1", "knowledge_base_id": "kb1"}
        mock_req.return_value = mock_resp

        res = client.attach_knowledge_base_to_task("t1", "kb1")
        assert res["status"] == "ATTACHED"
        assert mock_req.call_args[1]["method"] == "POST"
        assert mock_req.call_args[1]["url"] == "http://testserver/api/v1/tasks/t1/knowledge-bases/kb1"

        mock_resp.json.return_value = {"status": "DETACHED", "task_id": "t1", "knowledge_base_id": "kb1"}
        res2 = client.detach_knowledge_base_from_task("t1", "kb1")
        assert res2["status"] == "DETACHED"
        assert mock_req.call_args[1]["method"] == "DELETE"
        assert mock_req.call_args[1]["url"] == "http://testserver/api/v1/tasks/t1/knowledge-bases/kb1"


def test_list_task_knowledge_bases(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "items": [
                {"knowledge_base_id": "kb1", "name": "Physics", "document_count": 3}
            ]
        }
        mock_req.return_value = mock_resp

        res = client.list_task_knowledge_bases("t1")
        assert len(res) == 1
        assert res[0]["knowledge_base_id"] == "kb1"
        assert mock_req.call_args[1]["url"] == "http://testserver/api/v1/tasks/t1/knowledge-bases"


def test_api_client_error_handling(client):
    with patch("requests.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"detail": "Knowledge Base 'kb_none' not found."}
        mock_req.return_value = mock_resp

        with pytest.raises(ApiClientError) as exc_info:
            client.get_knowledge_base("kb_none")
        assert "not found" in str(exc_info.value)
        assert exc_info.value.status_code == 404


def test_task_kb_controls_use_distinct_widget_keys_for_each_render_scope(monkeypatch):
    """Rendering the same task controls in two dashboard regions must not reuse keys."""
    from webui import agent_page

    class FakeStreamlit:
        def __init__(self):
            self.widget_keys: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def markdown(self, *_args, **_kwargs):
            return None

        def info(self, *_args, **_kwargs):
            return None

        def columns(self, widths):
            return [self for _ in widths]

        def button(self, _label, *, key, **_kwargs):
            self.widget_keys.append(key)
            return False

    class FakeClient:
        def list_task_knowledge_bases(self, _task_id):
            return [
                {
                    "knowledge_base_id": "kb_7b035f2824b7488badb4b365",
                    "name": "Test KB",
                    "document_count": 1,
                }
            ]

        def list_knowledge_bases(self, status):
            assert status == "ACTIVE"
            return []

    fake_st = FakeStreamlit()
    monkeypatch.setattr(agent_page, "st", fake_st)

    agent_page._render_task_kb_controls(
        FakeClient(),
        "task_1",
        key_scope="checkpoint",
    )
    agent_page._render_task_kb_controls(
        FakeClient(),
        "task_1",
        key_scope="attached_tab",
    )

    assert len(fake_st.widget_keys) == 2
    assert len(set(fake_st.widget_keys)) == 2
