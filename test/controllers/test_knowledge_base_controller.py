from __future__ import annotations

import io
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.controllers.v1.knowledge_base import router as kb_router
from app.controllers.v1.knowledge_video import router as kv_router
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import KnowledgeVideoTaskRepository


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


def test_knowledge_base_controller_endpoints(session_factory, monkeypatch):
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    # 1. Create Knowledge Base
    resp_create = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "Astronomy", "description": "Space science documents"},
    )
    assert resp_create.status_code == 201
    res = resp_create.json()
    assert res["status"] == 201
    kb_data = res["data"]
    kb_id = kb_data["knowledge_base_id"]
    assert kb_data["name"] == "Astronomy"
    assert kb_data["status"] == "ACTIVE"

    # 2. List Knowledge Bases
    resp_list = client.get("/api/v1/knowledge-bases")
    assert resp_list.status_code == 200
    res_list = resp_list.json()
    assert res_list["status"] == 200
    items = res_list["data"]["items"]
    assert len(items) == 1
    assert items[0]["knowledge_base_id"] == kb_id

    # 3. Get KB details
    resp_get = client.get(f"/api/v1/knowledge-bases/{kb_id}")
    assert resp_get.status_code == 200
    res_get = resp_get.json()
    assert res_get["status"] == 200
    assert res_get["data"]["document_count"] == 0

    # 4. Update KB
    resp_patch = client.patch(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={"name": "Astrophysics & Cosmology", "description": "Updated description"},
    )
    assert resp_patch.status_code == 200
    res_patch = resp_patch.json()
    assert res_patch["status"] == 200
    assert res_patch["data"]["name"] == "Astrophysics & Cosmology"

    # 5. Upload document
    file_bytes = b"Hubble and James Webb are orbiting optical and infrared space telescopes."
    resp_upload = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("telescopes.txt", io.BytesIO(file_bytes), "text/plain")},
        data={"title": "Space Telescopes Overview"},
    )
    assert resp_upload.status_code == 201
    res_upload = resp_upload.json()
    assert res_upload["status"] == 201
    doc_data = res_upload["data"]
    doc_id = doc_data["source_document_id"]
    assert doc_data["status"] == "READY"
    assert doc_data["chunk_count"] >= 1

    # 6. List documents in KB
    resp_docs = client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
    assert resp_docs.status_code == 200
    res_docs = resp_docs.json()
    assert res_docs["status"] == 200
    doc_items = res_docs["data"]["items"]
    assert len(doc_items) == 1
    assert doc_items[0]["source_document_id"] == doc_id
    assert doc_items[0]["status"] == "READY"

    # 7. Retry document
    resp_retry = client.post(f"/api/v1/knowledge-bases/{kb_id}/documents/{doc_id}/retry")
    assert resp_retry.status_code == 200
    res_retry = resp_retry.json()
    assert res_retry["status"] == 200
    assert res_retry["data"]["status"] == "READY"

    # 8. Create task and attach KB
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        t_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="Telescopes", workflow_policy=WorkflowPolicyType.AUTO))
        session.commit()

    resp_attach = client.post(f"/api/v1/tasks/{task_id}/knowledge-bases/{kb_id}")
    assert resp_attach.status_code == 200
    res_attach = resp_attach.json()
    assert res_attach["status"] == 200
    assert res_attach["data"]["status"] == "ATTACHED"

    # 9. List KBs attached to task
    resp_task_kbs = client.get(f"/api/v1/tasks/{task_id}/knowledge-bases")
    assert resp_task_kbs.status_code == 200
    res_task_kbs = resp_task_kbs.json()
    assert res_task_kbs["status"] == 200
    task_kbs = res_task_kbs["data"]["items"]
    assert len(task_kbs) == 1
    assert task_kbs[0]["knowledge_base_id"] == kb_id

    # 10. Detach KB from task
    resp_detach = client.delete(f"/api/v1/tasks/{task_id}/knowledge-bases/{kb_id}")
    assert resp_detach.status_code == 200
    res_detach = resp_detach.json()
    assert res_detach["status"] == 200
    assert res_detach["data"]["status"] == "DETACHED"

    resp_task_kbs_after = client.get(f"/api/v1/tasks/{task_id}/knowledge-bases")
    assert len(resp_task_kbs_after.json()["data"]["items"]) == 0

    # 11. Archive KB
    resp_del = client.delete(f"/api/v1/knowledge-bases/{kb_id}")
    assert resp_del.status_code == 200
    res_del = resp_del.json()
    assert res_del["status"] == 200
    assert res_del["data"]["status"] == "ARCHIVED"

    # 12. Non-existent KB returns 404
    resp_404 = client.get("/api/v1/knowledge-bases/kb_nonexistent")
    assert resp_404.status_code == 404


def test_knowledge_base_controller_validation_and_errors(session_factory, monkeypatch):
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    app = FastAPI()
    app.include_router(kb_router)
    app.include_router(kv_router)
    client = TestClient(app)

    # 1. Invalid status query param returns 422
    resp_invalid_status = client.get("/api/v1/knowledge-bases?status=NOT_A_STATUS")
    assert resp_invalid_status.status_code == 422

    # 2. Valid status query param returns 200
    resp_valid_status = client.get("/api/v1/knowledge-bases?status=ACTIVE")
    assert resp_valid_status.status_code == 200
    assert resp_valid_status.json()["data"]["items"] == []

    # 3. Create active KB
    resp_create = client.post("/api/v1/knowledge-bases", json={"name": "Science KB"})
    assert resp_create.status_code == 201
    kb_id = resp_create.json()["data"]["knowledge_base_id"]

    # 4. Upload file exceeding limit returns 413 (tested with 1KB limit to avoid 50MB buffer allocation)
    monkeypatch.setattr("app.controllers.v1.knowledge_base.MAX_UPLOAD_SIZE_BYTES", 1024)
    monkeypatch.setattr("app.application.knowledge_base_service.MAX_FILE_SIZE_BYTES", 1024)
    oversized_bytes = b"X" * 1025
    resp_413 = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("large.txt", io.BytesIO(oversized_bytes), "text/plain")},
    )
    assert resp_413.status_code == 413
    assert "exceeds maximum limit" in resp_413.json()["detail"]

    # 5. Archive KB
    client.delete(f"/api/v1/knowledge-bases/{kb_id}")

    # 6. Upload to archived KB returns 409
    resp_upload_archived = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")},
    )
    assert resp_upload_archived.status_code == 409

    # 7. Create task and attempt to attach archived KB -> 409
    task_id = f"task_{uuid4().hex[:8]}"
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        t_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="Test", workflow_policy=WorkflowPolicyType.AUTO))
        session.commit()

    resp_attach_archived = client.post(f"/api/v1/tasks/{task_id}/knowledge-bases/{kb_id}")
    assert resp_attach_archived.status_code == 409

    # 8. Detach non-attached KB from task returns 404
    resp_detach_not_attached = client.delete(f"/api/v1/tasks/{task_id}/knowledge-bases/{kb_id}")
    assert resp_detach_not_attached.status_code == 404

    # 9. Detach from non-existent task returns 404
    resp_detach_no_task = client.delete(f"/api/v1/tasks/task_missing/knowledge-bases/{kb_id}")
    assert resp_detach_no_task.status_code == 404

    # 10. Detach non-existent KB returns 404
    resp_detach_no_kb = client.delete(f"/api/v1/tasks/{task_id}/knowledge-bases/kb_missing")
    assert resp_detach_no_kb.status_code == 404


def test_knowledge_base_upload_boundary_and_mime_validation(session_factory, monkeypatch):
    """
    Tests upload boundary conditions (1023, 1024, 1025 bytes with 1KB limit),
    MIME vs extension conflict rejection, and guaranteed file.close() invocations.
    """
    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    # 1KB upload limit for lightweight testing (zero 50MB allocations)
    monkeypatch.setattr("app.controllers.v1.knowledge_base.MAX_UPLOAD_SIZE_BYTES", 1024)
    monkeypatch.setattr("app.application.knowledge_base_service.MAX_FILE_SIZE_BYTES", 1024)

    close_call_count = 0
    from starlette.datastructures import UploadFile as StarletteUploadFile
    original_close = StarletteUploadFile.close

    async def tracking_close(self):
        nonlocal close_call_count
        close_call_count += 1
        return await original_close(self)

    monkeypatch.setattr(StarletteUploadFile, "close", tracking_close)

    app = FastAPI()
    app.include_router(kb_router)
    client = TestClient(app)

    # Create KB
    res_kb = client.post("/api/v1/knowledge-bases", json={"name": "Boundary Test KB"})
    assert res_kb.status_code == 201
    kb_id = res_kb.json()["data"]["knowledge_base_id"]

    # --- Boundary Test: 1023 bytes (limit - 1) -> 201 OK ---
    prev_close = close_call_count
    resp_1023 = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("doc_1023.txt", io.BytesIO(b"A" * 1023), "text/plain")},
    )
    assert resp_1023.status_code == 201
    assert close_call_count > prev_close, "file.close() must be called on 201 success"

    # --- Boundary Test: 1024 bytes (exact limit) -> 201 OK ---
    prev_close = close_call_count
    resp_1024 = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("doc_1024.txt", io.BytesIO(b"B" * 1024), "text/plain")},
    )
    assert resp_1024.status_code == 201
    assert close_call_count > prev_close, "file.close() must be called on 201 limit match"

    # --- Boundary Test: 1025 bytes (limit + 1) -> 413 Payload Too Large ---
    prev_close = close_call_count
    resp_1025 = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("doc_1025.txt", io.BytesIO(b"C" * 1025), "text/plain")},
    )
    assert resp_1025.status_code == 413
    assert "exceeds maximum limit" in resp_1025.json()["detail"]
    assert close_call_count > prev_close, "file.close() must be called on 413 limit exceeded"

    # --- MIME Conflict: .pdf extension with image/png MIME -> 400 ---
    prev_close = close_call_count
    resp_mime_conflict = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4 fake"), "image/png")},
    )
    assert resp_mime_conflict.status_code == 400
    assert "conflicts with file extension" in resp_mime_conflict.json()["detail"]
    assert close_call_count > prev_close, "file.close() must be called on 400 MIME conflict"

    # --- MIME Conflict: .txt extension with application/pdf MIME -> 400 ---
    prev_close = close_call_count
    resp_mime_conflict2 = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "application/pdf")},
    )
    assert resp_mime_conflict2.status_code == 400
    assert "conflicts with file extension" in resp_mime_conflict2.json()["detail"]
    assert close_call_count > prev_close, "file.close() must be called on 400 MIME conflict"

    # --- Unsupported Extension: .exe -> 400 ---
    prev_close = close_call_count
    resp_unsupported = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        files={"file": ("malware.exe", io.BytesIO(b"binary data"), "application/octet-stream")},
    )
    assert resp_unsupported.status_code == 400
    assert "Unsupported file extension" in resp_unsupported.json()["detail"]
    assert close_call_count > prev_close, "file.close() must be called on 400 unsupported extension"


class RecordingFakeUploadFile:
    """
    Test double for UploadFile that records all read sizes, chunk lengths,
    close calls, and can simulate read failures at specific call indices.
    """

    def __init__(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        fail_on_read_index: int | None = None,
    ):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.fail_on_read_index = fail_on_read_index
        self._cursor = 0
        self.read_calls: list[int] = []
        self.chunks_returned: list[int] = []
        self.close_calls = 0

    async def read(self, size: int = -1) -> bytes:
        current_idx = len(self.read_calls)
        self.read_calls.append(size)
        if self.fail_on_read_index is not None and current_idx == self.fail_on_read_index:
            raise OSError(f"Simulated read error at call index {current_idx}")
        if self._cursor >= len(self.content):
            self.chunks_returned.append(0)
            return b""
        if size == -1:
            chunk = self.content[self._cursor :]
            self._cursor = len(self.content)
        else:
            chunk = self.content[self._cursor : self._cursor + size]
            self._cursor += len(chunk)
        self.chunks_returned.append(len(chunk))
        return chunk

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_read_upload_with_limit_chunked_behavior_isolated():
    """
    Directly tests _read_upload_with_limit chunked reading contract:
    - Under limit: reads in chunks of chunk_size and ends at EOF.
    - Exact limit: reads all chunks up to limit.
    - Over limit: halts IMMEDIATELY upon exceeding max_size without reading remaining chunks.
    - Read exception: wraps in HTTPException(400).
    - Empty stream: raises HTTPException(400).
    """
    from app.controllers.v1.knowledge_base import _read_upload_with_limit

    # 1. Under limit: limit 1024, chunk 256, data 700 bytes -> 4 reads: 256, 256, 188, 0
    fake = RecordingFakeUploadFile("a.txt", b"x" * 700)
    data = await _read_upload_with_limit(file=fake, max_size=1024, chunk_size=256)
    assert len(data) == 700
    assert fake.read_calls == [256, 256, 256, 256]
    assert fake.chunks_returned == [256, 256, 188, 0]

    # 2. Exact limit: limit 1024, chunk 256, data 1024 bytes -> 5 reads: 256, 256, 256, 256, 0
    fake = RecordingFakeUploadFile("b.txt", b"y" * 1024)
    data = await _read_upload_with_limit(file=fake, max_size=1024, chunk_size=256)
    assert len(data) == 1024
    assert fake.read_calls == [256, 256, 256, 256, 256]
    assert fake.chunks_returned == [256, 256, 256, 256, 0]

    # 3. Over limit: limit 1024, chunk 256, data 2048 bytes
    # Chunks: 256 (256), 256 (512), 256 (768), 256 (1024), 256 (1280 > 1024) -> raises 413!
    # Crucially: halts on the 5th chunk immediately. Remaining 768 bytes in fake buffer are NEVER read!
    fake = RecordingFakeUploadFile("c.txt", b"z" * 2048)
    with pytest.raises(HTTPException) as exc_info:
        await _read_upload_with_limit(file=fake, max_size=1024, chunk_size=256)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum limit" in exc_info.value.detail
    assert len(fake.read_calls) == 5
    assert fake.chunks_returned == [256, 256, 256, 256, 256]
    assert fake._cursor == 1280  # only 1280 bytes consumed, remainder untouched

    # 4. Read exception on 2nd chunk -> raises 400
    fake = RecordingFakeUploadFile("d.txt", b"d" * 700, fail_on_read_index=1)
    with pytest.raises(HTTPException) as exc_info:
        await _read_upload_with_limit(file=fake, max_size=1024, chunk_size=256)
    assert exc_info.value.status_code == 400
    assert "Failed to read uploaded file" in exc_info.value.detail

    # 5. Empty file -> raises 400
    fake = RecordingFakeUploadFile("empty.txt", b"")
    with pytest.raises(HTTPException) as exc_info:
        await _read_upload_with_limit(file=fake, max_size=1024, chunk_size=256)
    assert exc_info.value.status_code == 400
    assert "Uploaded file is empty" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_document_controller_chunked_lifecycle_and_guarantees(session_factory, monkeypatch):
    """
    Tests upload_document controller workflow with RecordingFakeUploadFile:
    - Case A: Under-limit chunked read (count and chunk sizes verified, close called, 201)
    - Case B: Exact-limit chunked read (count and chunk sizes verified, close called, 201)
    - Case C: Over-limit immediate early halt (raises 413, business service uncalled, close called)
    - Case D: Read exception (raises 400, business service uncalled, close called)
    - Case E: Business service exception (raises 400, close called)
    - Case F: MIME/extension conflict (raises 400, 0 reads performed, close called)
    - Case G: Empty file (raises 400, close called)
    - Case H: Compatibility for content_type=None and application/octet-stream (accepted, close called)
    """
    from app.application.knowledge_base_service import KnowledgeBaseCommandService, KnowledgeBaseServiceError
    from app.controllers.v1.knowledge_base import upload_document
    from app.domain.knowledge_base import KnowledgeBase
    from app.persistence.repositories import KnowledgeBaseRepository

    monkeypatch.setattr("app.controllers.v1.knowledge_base.get_session", session_factory)
    monkeypatch.setattr("app.controllers.v1.knowledge_base.CHUNK_SIZE", 256)
    monkeypatch.setattr("app.controllers.v1.knowledge_base.MAX_UPLOAD_SIZE_BYTES", 1024)
    monkeypatch.setattr("app.application.knowledge_base_service.MAX_FILE_SIZE_BYTES", 1024)

    # Pre-create active KB
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Upload Test KB")
        kb_repo.save_knowledge_base(kb)
        session.commit()
        kb_id = kb.knowledge_base_id

    # --- Case A: Under-limit (700 bytes, chunk 256, limit 1024) ---
    fake_a = RecordingFakeUploadFile("doc_a.txt", b"A" * 700, content_type="text/plain")
    res_a = await upload_document(kb_id=kb_id, file=fake_a, title="Doc A")
    assert res_a["status"] == 201
    assert fake_a.read_calls == [256, 256, 256, 256]
    assert fake_a.chunks_returned == [256, 256, 188, 0]
    assert fake_a.close_calls == 1

    # --- Case B: Exact-limit (1024 bytes, chunk 256, limit 1024) ---
    fake_b = RecordingFakeUploadFile("doc_b.txt", b"B" * 1024, content_type="text/plain")
    res_b = await upload_document(kb_id=kb_id, file=fake_b, title="Doc B")
    assert res_b["status"] == 201
    assert fake_b.read_calls == [256, 256, 256, 256, 256]
    assert fake_b.chunks_returned == [256, 256, 256, 256, 0]
    assert fake_b.close_calls == 1

    # --- Case C: Over-limit (2048 bytes > 1024 limit) ---
    # Service should NOT be called; reading should halt at 5th chunk; file.close() must be called
    service_called = False
    orig_upload = KnowledgeBaseCommandService.upload_document

    def spy_upload(self, *args, **kwargs):
        nonlocal service_called
        service_called = True
        return orig_upload(self, *args, **kwargs)

    monkeypatch.setattr(KnowledgeBaseCommandService, "upload_document", spy_upload)

    fake_c = RecordingFakeUploadFile("doc_c.txt", b"C" * 2048, content_type="text/plain")
    with pytest.raises(HTTPException) as exc_c:
        await upload_document(kb_id=kb_id, file=fake_c, title="Doc C")
    assert exc_c.value.status_code == 413
    assert len(fake_c.read_calls) == 5  # early break on 5th chunk (1280 > 1024)
    assert fake_c._cursor == 1280  # did not read remaining bytes
    assert not service_called, "Business service must NOT be called when file is oversized"
    assert fake_c.close_calls == 1, "file.close() must be called on 413 error"

    # --- Case D: Read exception (fails on 2nd chunk) ---
    service_called = False
    fake_d = RecordingFakeUploadFile("doc_d.txt", b"D" * 700, content_type="text/plain", fail_on_read_index=1)
    with pytest.raises(HTTPException) as exc_d:
        await upload_document(kb_id=kb_id, file=fake_d, title="Doc D")
    assert exc_d.value.status_code == 400
    assert "Failed to read uploaded file" in exc_d.value.detail
    assert not service_called, "Business service must NOT be called on read failure"
    assert fake_d.close_calls == 1, "file.close() must be called on read error"

    # --- Case E: Business service exception ---
    def failing_upload(self, *args, **kwargs):
        raise KnowledgeBaseServiceError("Simulated ingestion pipeline crash")

    monkeypatch.setattr(KnowledgeBaseCommandService, "upload_document", failing_upload)
    fake_e = RecordingFakeUploadFile("doc_e.txt", b"E" * 500, content_type="text/plain")
    with pytest.raises(HTTPException) as exc_e:
        await upload_document(kb_id=kb_id, file=fake_e, title="Doc E")
    assert exc_e.value.status_code == 400
    assert "Simulated ingestion pipeline crash" in exc_e.value.detail
    assert fake_e.close_calls == 1, "file.close() must be called on service error"

    # Restore original upload
    monkeypatch.setattr(KnowledgeBaseCommandService, "upload_document", orig_upload)

    # --- Case F: MIME/extension conflict (0 reads executed, close called) ---
    fake_f = RecordingFakeUploadFile("fake.pdf", b"Some random content", content_type="image/png")
    with pytest.raises(HTTPException) as exc_f:
        await upload_document(kb_id=kb_id, file=fake_f, title="Doc F")
    assert exc_f.value.status_code == 400
    assert "conflicts with file extension" in exc_f.value.detail
    assert len(fake_f.read_calls) == 0, "No read should occur when MIME conflict is detected upfront"
    assert fake_f.close_calls == 1, "file.close() must be called on MIME conflict error"

    # --- Case G: Empty file ---
    fake_g = RecordingFakeUploadFile("empty.txt", b"", content_type="text/plain")
    with pytest.raises(HTTPException) as exc_g:
        await upload_document(kb_id=kb_id, file=fake_g, title="Doc G")
    assert exc_g.value.status_code == 400
    assert "Uploaded file is empty" in exc_g.value.detail
    assert fake_g.close_calls == 1, "file.close() must be called on empty file error"

    # --- Case H: Content type compatibility ---
    # 1. content_type is None
    fake_h1 = RecordingFakeUploadFile("no_type.txt", b"Hello plain text", content_type=None)
    res_h1 = await upload_document(kb_id=kb_id, file=fake_h1, title="No Type Doc")
    assert res_h1["status"] == 201
    assert fake_h1.close_calls == 1

    # 2. content_type is application/octet-stream for .txt
    fake_h2 = RecordingFakeUploadFile("stream.txt", b"Hello octet txt", content_type="application/octet-stream")
    res_h2 = await upload_document(kb_id=kb_id, file=fake_h2, title="Octet TXT")
    assert res_h2["status"] == 201
    assert fake_h2.close_calls == 1

    # 3. content_type is application/octet-stream for .md
    fake_h3 = RecordingFakeUploadFile("notes.md", b"# Markdown Header\nSome notes", content_type="application/octet-stream")
    res_h3 = await upload_document(kb_id=kb_id, file=fake_h3, title="Octet MD")
    assert res_h3["status"] == 201
    assert fake_h3.close_calls == 1
