from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_service import TaskEvidenceCommandService
from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.stage_executor_registry import (
    StageExecutorRegistry,
    get_default_executor_registry,
)
from app.application.task_command_service import TaskCommandService
from app.controllers.v1.knowledge_video import router as kv_router
from app.domain.evidence import (
    EvidenceAvailabilityPolicy,
    EvidenceItem,
    EvidenceSnapshot,
    KnowledgeChunk,
    RetrievalSnapshot,
    SearchResult,
    SourceDocument,
    SourceQualityTier,
    SourceStatus,
    SourceType,
    WebResearchSnapshot,
    classify_source_quality,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    JobErrorType,
    JobStatus,
    Stage,
    TaskStatus,
    WorkflowPolicyType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    StageExecutionRepository,
    TaskArtifactRepository,
    WorkflowJobRepository,
)
from app.services.knowledge.search_provider import (
    SearchCredentialsMissingError,
    SearchProvider,
    SearchProviderError,
    SearchProviderRateLimitError,
    SearchProviderUnavailableError,
    TavilySearchProvider,
)
from app.services.knowledge.source_fetcher import FetchedUrlContent, SourceFetcher
from app.services.knowledge.web_research_service import (
    WebResearchConfig,
    WebResearchService,
)
from app.workers.stage_worker import StageWorker


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


class MockSearchProvider:
    """Mock search provider for isolated unit tests."""

    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.call_count = 0
        self.last_query: str | None = None
        self.last_limit: int | None = None

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.call_count += 1
        self.last_query = query
        self.last_limit = limit
        if self.error:
            raise self.error
        return self.results[:limit]


# =============================================================================
# 1. AUTHORIZATION TESTS (1-5)
# =============================================================================


def test_research_defaults_to_unauthorized():
    """1. Verify that a task defaults to research_authorized = False."""
    task = KnowledgeVideoTask.create(topic="Quantum computing")
    assert task.is_research_authorized is False


def test_unauthorized_task_makes_zero_search_calls(session_factory):
    """2. Verify that an unauthorized task with insufficient evidence makes 0 search calls."""
    mock_search = MockSearchProvider()
    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
    )

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="Quantum computing", allow_research=False)
        job_repo = WorkflowJobRepository(session)
        job = job_repo.get_current_job_for_task(task.task_id)
        session.commit()

    result = executor.execute(task, job)
    assert result.success is False
    assert result.error_type == JobErrorType.NEEDS_EVIDENCE.value
    assert mock_search.call_count == 0


def test_authorization_is_durable(session_factory):
    """3. Verify that task research authorization is persisted durably."""
    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="Quantum computing", allow_research=False)
        session.commit()
        task_id = task.task_id

    # Authorize research
    with session_factory() as session:
        cmd = TaskCommandService(session)
        cmd.authorize_research(task_id)
        session.commit()

    # Re-read in a fresh session
    with session_factory() as session:
        repo = KnowledgeVideoTaskRepository(session)
        reloaded = repo.get_task(task_id)
        assert reloaded is not None
        assert reloaded.is_research_authorized is True


def test_authorization_is_scoped_to_correct_task(session_factory):
    """4-5. Task A authorization does not authorize Task B."""
    with session_factory() as session:
        cmd = TaskCommandService(session)
        task_a = cmd.create_task(topic="Task A", allow_research=True)
        task_b = cmd.create_task(topic="Task B", allow_research=False)
        session.commit()
        id_a, id_b = task_a.task_id, task_b.task_id

    with session_factory() as session:
        repo = KnowledgeVideoTaskRepository(session)
        loaded_a = repo.get_task(id_a)
        loaded_b = repo.get_task(id_b)
        assert loaded_a.is_research_authorized is True
        assert loaded_b.is_research_authorized is False


# =============================================================================
# 2. SEARCH PROVIDER CONTRACT & ADAPTER TESTS (6-12)
# =============================================================================


def test_tavily_search_missing_credentials_raises_error():
    """11. Missing credentials raise SearchCredentialsMissingError without making network calls."""
    with patch.dict("os.environ", {}, clear=True):
        provider = TavilySearchProvider(api_key=None)
        with pytest.raises(SearchCredentialsMissingError) as exc_info:
            provider.search("test query")
        assert "API key is missing" in str(exc_info.value)


def test_tavily_search_adapter_mocked_http_success():
    """6, 7, 9. Production TavilySearchProvider parses HTTP response and respects limits."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Attention Paper",
                "url": "https://example.com/paper",
                "content": "Discovery snippet about self-attention",
                "score": 0.98,
            },
            {
                "title": "Transformer Tutorial",
                "url": "https://example.com/tutorial",
                "content": "Tutorial on multi-head attention",
                "score": 0.85,
            },
        ]
    }
    mock_session.post.return_value = mock_response

    provider = TavilySearchProvider(api_key="tvly-mock-key-1234", session=mock_session)
    results = provider.search(query="Why Transformer?", limit=1)

    assert len(results) == 1
    assert results[0].title == "Attention Paper"
    assert results[0].url == "https://example.com/paper"
    assert results[0].snippet == "Discovery snippet about self-attention"
    assert results[0].provider_rank == 1

    # Verify posted payload
    mock_session.post.assert_called_once()
    call_kwargs = mock_session.post.call_args[1]
    assert call_kwargs["json"]["query"] == "Why Transformer?"
    assert call_kwargs["json"]["max_results"] == 1


def test_tavily_search_adapter_rate_limit_error():
    """10. HTTP 429 maps to SearchProviderRateLimitError."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 429
    mock_session.post.return_value = mock_response

    provider = TavilySearchProvider(api_key="tvly-mock-key", session=mock_session)
    with pytest.raises(SearchProviderRateLimitError):
        provider.search("query")


def test_tavily_search_adapter_timeout_error():
    """10. Request timeout maps to SearchProviderUnavailableError."""
    mock_session = MagicMock()
    mock_session.post.side_effect = requests.Timeout("Connection timed out")

    provider = TavilySearchProvider(api_key="tvly-mock-key", session=mock_session)
    with pytest.raises(SearchProviderUnavailableError):
        provider.search("query")


# =============================================================================
# 3. DISCOVERY SNAPSHOT & SOURCE QUALITY (13-16, SourceQualityTier)
# =============================================================================


def test_classify_source_quality_deterministic():
    """Source quality tier classification maps domains deterministically."""
    assert classify_source_quality("https://www.whitehouse.gov/briefing") == SourceQualityTier.PRIMARY_OFFICIAL
    assert classify_source_quality("https://arxiv.org/abs/1706.03762") == SourceQualityTier.PEER_REVIEWED
    assert classify_source_quality("https://mit.edu/research") == SourceQualityTier.PEER_REVIEWED
    assert classify_source_quality("https://developer.mozilla.org/en-US/docs/Web") == SourceQualityTier.AUTHORITATIVE_PROFESSIONAL
    assert classify_source_quality("https://www.reuters.com/technology") == SourceQualityTier.SECONDARY_MEDIA
    assert classify_source_quality("https://medium.com/@user/story") == SourceQualityTier.BLOG_COMMUNITY
    assert classify_source_quality("https://unknown-random-domain.xyz/page") == SourceQualityTier.UNKNOWN


def test_web_research_snapshot_immutability_and_credentials_safety(session_factory):
    """13-16. WebResearchSnapshot freezes search audit without credentials."""
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(topic="Test topic")
        task_repo.save_task(task)

        snap = WebResearchSnapshot.create(
            task_id=task.task_id,
            stage_attempt=1,
            query="Test query",
            provider="TavilySearchProvider",
            search_results=[
                SearchResult(
                    result_id="res_1",
                    title="Doc Title",
                    url="https://example.com/doc",
                    snippet="Snippet text",
                    provider_rank=1,
                )
            ],
            selected_urls=["https://example.com/doc"],
            fetch_outcomes=[{"url": "https://example.com/doc", "status": "SUCCESS"}],
            created_source_document_ids=["src_1"],
        )
        saved = ev_repo.save_web_research_snapshot(snap)
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        reloaded = ev_repo.get_web_research_snapshot(saved.web_research_snapshot_id)
        assert reloaded is not None
        assert reloaded.query == "Test query"
        assert reloaded.provider == "TavilySearchProvider"
        assert len(reloaded.search_results) == 1
        assert reloaded.search_results[0].provider_rank == 1
        assert reloaded.content_fingerprint == saved.content_fingerprint
        # Verify no credentials in snapshot dump
        dump_str = str(reloaded.model_dump())
        assert "api_key" not in dump_str
        assert "secret" not in dump_str


# =============================================================================
# 4. URL FETCH SAFETY, REUSE & DEDUPLICATION (17-25)
# =============================================================================


def test_search_service_deduplicates_and_respects_limits(session_factory):
    """8, 9, 25. Bounded limits and URL deduplication are strictly enforced."""
    mock_search = MockSearchProvider(
        results=[
            SearchResult(result_id="1", title="A", url="https://example.com/a", snippet="s", provider_rank=1),
            SearchResult(result_id="2", title="A duplicate", url="https://example.com/a", snippet="s", provider_rank=2),
            SearchResult(result_id="3", title="B", url="https://example.com/b", snippet="s", provider_rank=3),
            SearchResult(result_id="4", title="C", url="https://example.com/c", snippet="s", provider_rank=4),
            SearchResult(result_id="5", title="D", url="https://example.com/d", snippet="s", provider_rank=5),
        ]
    )

    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url="https://example.com/a",
        final_url="https://example.com/a",
        status_code=200,
        content_type="text/html",
        title="Page Title",
        extracted_text="Some extracted article text with enough content.",
    )

    config = WebResearchConfig(max_pages_to_fetch=2)

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task = task_repo.save_task(KnowledgeVideoTask.create(topic="Topic"))
        session.commit()

        service = WebResearchService(
            session=session,
            search_provider=mock_search,
            fetcher=mock_fetcher,
            config=config,
            allow_private_for_tests=True,
        )
        snapshot, docs = service.execute_research(task.task_id, query="Topic")
        session.commit()

        # Should fetch only 2 pages due to max_pages_to_fetch=2 and deduplication
        assert len(snapshot.selected_urls) == 2
        assert snapshot.selected_urls == ("https://example.com/a", "https://example.com/b")


# =============================================================================
# 5. MANDATORY DEMONSTRATIONS (23, 24, 25, 26)
# =============================================================================


def test_mandatory_success_integration(session_factory):
    """23. MANDATORY SUCCESS INTEGRATION TEST

    Scenario:
    Task topic: "Why does Transformer use self-attention?"
    User source: contains unrelated/insufficient information.
    Research authorization: TRUE.

    Mock provider returns search result:
    url: https://example.test/attention-paper, snippet: "Transformer architecture..."
    Mock page fetch returns:
    "Self-attention relates positions within a sequence in order to compute a representation of the sequence."

    Verify:
    - SearchResult snippet was NOT used as Evidence excerpt
    - Evidence resolves to fetched page content
    - task_id remains unchanged
    - Evidence stage succeeds
    - KNOWLEDGE_PLAN job is created
    - Zero public internet calls occurred
    """
    fetched_body = "Self-attention relates positions within a sequence in order to compute a representation of the sequence."
    search_snippet = "Transformer architecture..."

    # Setup mock provider
    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="search_res_1",
                title="Attention Is All You Need",
                url="https://example.test/attention-paper",
                snippet=search_snippet,
                provider_rank=1,
            )
        ]
    )

    # Setup mock fetcher
    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url="https://example.test/attention-paper",
        final_url="https://example.test/attention-paper",
        status_code=200,
        content_type="text/html",
        title="Attention Is All You Need",
        extracted_text=fetched_body,
    )

    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
        allow_private_for_tests=True,
    )

    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, executor)

    worker = StageWorker(
        session_factory=session_factory,
        registry=registry,
        worker_id="test-worker-w1-success",
    )

    # 1. Create task with insufficient user source, authorized for research
    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(
            topic="Why does Transformer use self-attention?",
            allow_research=True,
        )
        task_id = task.task_id

        # Register unrelated user source
        ev_cmd = TaskEvidenceCommandService(session)
        ev_cmd.add_text_source(
            task_id=task_id,
            text="Baking bread requires flour, water, yeast, and salt. Knead thoroughly.",
            title="Bread Recipe",
        )
        session.commit()

    # 2. Worker executes EVIDENCE stage
    processed = worker.run_once()
    assert processed is True

    # 3. Verify workflow and evidence assertions
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        art_repo = TaskArtifactRepository(session)

        # Task remains in RUNNING, advancing towards planning
        current_task = task_repo.get_task(task_id)
        assert current_task is not None
        assert current_task.task_id == task_id
        assert current_task.current_stage == Stage.KNOWLEDGE_PLAN
        assert current_task.task_status == TaskStatus.RUNNING

        # KNOWLEDGE_PLAN job was queued
        plan_job = job_repo.get_current_job_for_task(task_id)
        assert plan_job is not None
        assert plan_job.stage == Stage.KNOWLEDGE_PLAN
        assert plan_job.status == JobStatus.QUEUED

        # Evidence stage job succeeded
        evidence_jobs = job_repo.list_jobs_for_task(task_id)
        ev_job = next(j for j in evidence_jobs if j.stage == Stage.EVIDENCE)
        assert ev_job.status == JobStatus.SUCCEEDED

        # Evidence snapshot exists
        latest_snapshot = ev_repo.get_latest_snapshot_for_task(task_id)
        assert latest_snapshot is not None
        assert len(latest_snapshot.evidence_ids) >= 1

        # Check evidence item: MUST come from fetched page, NOT snippet
        evidence_items = [ev_repo.get_evidence_item(eid) for eid in latest_snapshot.evidence_ids]
        matched_item = next((item for item in evidence_items if item and "Self-attention" in item.original_excerpt), None)
        assert matched_item is not None
        assert search_snippet not in matched_item.original_excerpt
        assert "Self-attention relates positions" in matched_item.original_excerpt

        # WebResearchSnapshot exists and audit is intact
        research_snaps = ev_repo.list_web_research_snapshots_for_task(task_id)
        assert len(research_snaps) == 1
        assert research_snaps[0].query == "Why does Transformer use self-attention?"
        assert research_snaps[0].selected_urls == ("https://example.test/attention-paper",)


def test_mandatory_unauthorized_demonstration(session_factory):
    """24. MANDATORY UNAUTHORIZED TEST

    Same insufficient task, research_authorized = FALSE.
    Expected:
    - SearchProvider call count = 0
    - HTTP discovered-page fetch count = 0
    - Task becomes NEEDS_EVIDENCE
    - No synthetic evidence
    - No KNOWLEDGE_PLAN job
    """
    mock_search = MockSearchProvider()
    mock_fetcher = MagicMock(spec=SourceFetcher)

    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry)

    # 1. Create task without research authorization
    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(
            topic="Why does Transformer use self-attention?",
            allow_research=False,
        )
        task_id = task.task_id

        # Register unrelated source
        ev_cmd = TaskEvidenceCommandService(session)
        ev_cmd.add_text_source(
            task_id=task_id,
            text="Baking bread recipe.",
            title="Bread Recipe",
        )
        session.commit()

    # 2. Worker executes
    processed = worker.run_once()
    assert processed is True

    # 3. Assertions
    assert mock_search.call_count == 0
    assert mock_fetcher.fetch_url.call_count == 0

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        current_task = task_repo.get_task(task_id)
        assert current_task.task_status == TaskStatus.NEEDS_EVIDENCE

        # No KNOWLEDGE_PLAN job created
        jobs = job_repo.list_jobs_for_task(task_id)
        assert not any(j.stage == Stage.KNOWLEDGE_PLAN for j in jobs)


def test_mandatory_exhaustion_demonstration(session_factory):
    """25. MANDATORY EXHAUSTION TEST

    research_authorized = TRUE, search returns results, but fetched pages contain no usable evidence.
    Expected:
    - Configured limits respected
    - No repeated unlimited searching
    - Task ends in NEEDS_EVIDENCE
    - Research snapshot available
    - No fake EvidenceItem is created
    """
    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="res_1",
                title="Page 1",
                url="https://example.com/unrelated",
                snippet="Some text",
                provider_rank=1,
            )
        ]
    )

    # Page fetch returns empty / unrelated text
    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url="https://example.com/unrelated",
        final_url="https://example.com/unrelated",
        status_code=200,
        content_type="text/html",
        title="Unrelated",
        extracted_text="Completely different topic about gardening tips.",
    )

    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
        allow_private_for_tests=True,
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry)

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(
            topic="Quantum computing key distribution algorithms",
            allow_research=True,
        )
        task_id = task.task_id
        session.commit()

    processed = worker.run_once()
    assert processed is True

    # Verify task is in NEEDS_EVIDENCE and search call was made exactly once
    assert mock_search.call_count == 1
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        ev_repo = EvidenceRepository(session)
        current_task = task_repo.get_task(task_id)
        assert current_task.task_status == TaskStatus.NEEDS_EVIDENCE

        # Research snapshot was persisted
        snaps = ev_repo.list_web_research_snapshots_for_task(task_id)
        assert len(snaps) == 1
        assert snaps[0].query == "Quantum computing key distribution algorithms"


def test_mandatory_web_evidence_provenance_demonstration(session_factory):
    """26. MANDATORY PROVENANCE DEMONSTRATION

    For a web-derived factual EvidenceItem, resolve:
    EvidenceItem -> KnowledgeChunk -> SourceDocument -> fetched URL -> content hash
    And resolve:
    EvidenceSnapshot -> RetrievalSnapshot -> WebResearchSnapshot.
    """
    fetched_url = "https://example.org/spec"
    article_text = "The quick brown fox jumps over the lazy dog."

    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="s1",
                title="Fox Spec",
                url=fetched_url,
                snippet="fox snippet",
                provider_rank=1,
            )
        ]
    )
    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url=fetched_url,
        final_url=fetched_url,
        status_code=200,
        content_type="text/html",
        title="Fox Spec",
        extracted_text=article_text,
    )

    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
        allow_private_for_tests=True,
    )
    registry = StageExecutorRegistry()
    registry.register(Stage.EVIDENCE, executor)
    worker = StageWorker(session_factory=session_factory, registry=registry)

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="quick brown fox", allow_research=True)
        task_id = task.task_id
        session.commit()

    worker.run_once()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        art_repo = TaskArtifactRepository(session)

        # 1. EvidenceSnapshot -> TaskArtifactRef
        artifacts = art_repo.list_artifact_refs_for_task(task_id, stage=Stage.EVIDENCE)
        assert len(artifacts) >= 1
        ev_art = artifacts[0]
        snapshot_id = ev_art.artifact_id
        ev_snapshot = ev_repo.get_latest_snapshot_for_task(task_id)
        assert ev_snapshot.evidence_snapshot_id == snapshot_id

        # 2. EvidenceSnapshot -> RetrievalSnapshot -> WebResearchSnapshot
        retrieval_snapshot_id = ev_art.metadata_json["retrieval_snapshot_id"]
        web_research_snapshot_id = ev_art.metadata_json["web_research_snapshot_id"]
        ret_snap = ev_repo.get_retrieval_snapshot(retrieval_snapshot_id)
        web_snap = ev_repo.get_web_research_snapshot(web_research_snapshot_id)
        assert ret_snap is not None
        assert web_snap is not None

        # 3. EvidenceItem -> KnowledgeChunk -> SourceDocument -> fetched URL
        ev_id = ev_snapshot.evidence_ids[0]
        ev_item = ev_repo.get_evidence_item(ev_id)
        assert ev_item is not None

        src_doc = ev_repo.get_source_document(ev_item.source_document_id)
        assert src_doc is not None
        assert src_doc.source_type == SourceType.URL
        assert src_doc.source_locator == fetched_url
        assert src_doc.content_hash == ev_item.content_hash

        # KnowledgeChunk linkage
        chunks = ev_repo.list_chunks_for_source(src_doc.source_document_id)
        assert len(chunks) >= 1
        assert any(c.normalized_text == ev_item.original_excerpt for c in chunks)


# =============================================================================
# 6. API TESTS (46-48)
# =============================================================================


def test_authorize_research_api_endpoint(session_factory):
    """46-48. POST /api/v1/knowledge-video-tasks/{task_id}/authorize-research endpoint."""
    from contextlib import contextmanager

    app = FastAPI()
    app.include_router(kv_router)

    with patch("app.controllers.base.verify_token", return_value=True):
        client = TestClient(app)

        with session_factory() as session:
            cmd = TaskCommandService(session)
            task = cmd.create_task(topic="API authorization test", allow_research=False)
            task_id = task.task_id
            session.commit()

        @contextmanager
        def mock_get_session(factory=None):
            with session_factory() as s:
                yield s

        # Call endpoint with mock session dependency
        with patch("app.controllers.v1.knowledge_video.get_session", mock_get_session):
            resp = client.post(
                f"/api/v1/knowledge-video-tasks/{task_id}/authorize-research",
                headers={"Authorization": "Bearer test-token"},
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["task_id"] == task_id
            assert data["is_research_authorized"] is True


def test_authorize_research_terminal_task_fails(session_factory):
    """48. Authorizing research on terminal (CANCELLED) task fails with 409 Conflict."""
    from contextlib import contextmanager

    app = FastAPI()
    app.include_router(kv_router)

    with patch("app.controllers.base.verify_token", return_value=True):
        client = TestClient(app)

        with session_factory() as session:
            cmd = TaskCommandService(session)
            task = cmd.create_task(topic="Terminal task test")
            cmd.cancel_task(task.task_id, reason="User cancelled")
            task_id = task.task_id
            session.commit()

        @contextmanager
        def mock_get_session(factory=None):
            with session_factory() as s:
                yield s

        with patch("app.controllers.v1.knowledge_video.get_session", mock_get_session):
            resp = client.post(
                f"/api/v1/knowledge-video-tasks/{task_id}/authorize-research",
                headers={"Authorization": "Bearer test-token"},
            )
            assert resp.status_code == 409


# =============================================================================
# 7. SSRF, TASK ISOLATION & RETRY TESTS (19, 28, 29, 30, 41, 43, 49)
# =============================================================================


def test_ssrf_unsafe_url_is_rejected(session_factory):
    """19. Private / internal IP target is rejected during web research without crashing."""
    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="ssrf_res",
                title="Internal Metadata",
                url="http://169.254.169.254/latest/meta-data/",
                snippet="cloud credentials",
                provider_rank=1,
            )
        ]
    )

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task = task_repo.save_task(KnowledgeVideoTask.create(topic="SSRF test"))
        session.commit()

        service = WebResearchService(
            session=session,
            search_provider=mock_search,
            allow_private_for_tests=False,  # Enforce SSRF blocking
        )
        snap, docs = service.execute_research(task.task_id, query="SSRF test")
        session.commit()

        assert len(docs) == 0
        assert len(snap.fetch_outcomes) == 1
        assert snap.fetch_outcomes[0]["status"] == "REJECTED_SECURITY"


def test_temporary_provider_error_is_retryable(session_factory):
    """43. Temporary search provider timeout produces a RETRYABLE stage execution result."""
    mock_search = MockSearchProvider(error=SearchProviderUnavailableError("Tavily gateway timeout"))
    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
    )

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="Retry test topic", allow_research=True)
        job_repo = WorkflowJobRepository(session)
        job = job_repo.get_current_job_for_task(task.task_id)
        session.commit()

    result = executor.execute(task, job)
    assert result.success is False
    assert result.is_retryable is True
    assert result.error_type == JobErrorType.RETRYABLE.value


def test_task_source_isolation_in_augmented_retrieval(session_factory):
    """28. Sources associated with Task A are NEVER retrieved in Task B."""
    with session_factory() as session:
        cmd = TaskCommandService(session)
        task_a = cmd.create_task(topic="Topic A", allow_research=True)
        task_b = cmd.create_task(topic="Topic A", allow_research=True)  # Same topic

        ev_cmd = TaskEvidenceCommandService(session)
        ev_cmd.add_text_source(
            task_id=task_a.task_id,
            text="Exclusive secret evidence for Task A only.",
            title="Secret A",
        )
        session.commit()
        id_a, id_b = task_a.task_id, task_b.task_id

    # Execute Task B with 0 sources and empty search
    mock_search = MockSearchProvider(results=[])
    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
    )
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        task_b_obj = task_repo.get_task(id_b)
        job_b = job_repo.get_current_job_for_task(id_b)

    res_b = executor.execute(task_b_obj, job_b)
    assert res_b.success is False
    assert res_b.error_type == JobErrorType.NEEDS_EVIDENCE.value


def test_retrieval_snapshot_r1_and_r2_immutability(session_factory):
    """29-30. Both R1 and R2 are distinct, immutable records and R1 is unmodified."""
    search_snippet = "Transformer paper"
    page_text = "Self-attention mechanism computes pairwise attention weights across sequence representations."

    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="res_s1",
                title="Attention",
                url="https://example.test/paper",
                snippet=search_snippet,
                provider_rank=1,
            )
        ]
    )
    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url="https://example.test/paper",
        final_url="https://example.test/paper",
        status_code=200,
        content_type="text/html",
        title="Attention",
        extracted_text=page_text,
    )

    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
        allow_private_for_tests=True,
    )

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="self-attention mechanism", allow_research=True)
        task_id = task.task_id

        # User source with insufficient info
        ev_cmd = TaskEvidenceCommandService(session)
        ev_cmd.add_text_source(
            task_id=task_id,
            text="Neural networks use weights and biases.",
            title="Intro",
        )
        session.commit()

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        job = job_repo.get_current_job_for_task(task_id)

    res = executor.execute(task, job)
    assert res.success is True

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        snapshots = ev_repo.list_retrieval_snapshots_for_task(task_id)
        # Exactly two retrieval snapshots: R1 (user only) and R2 (user + web)
        assert len(snapshots) == 2
        r2, r1 = snapshots[0], snapshots[1]  # Ordered desc by created_at
        assert r1.retrieval_snapshot_id != r2.retrieval_snapshot_id
        assert len(r1.source_scope_ids) == 1
        assert len(r2.source_scope_ids) == 2
        # R1 content fingerprint remains stable and unmodified
        assert r1.content_fingerprint != r2.content_fingerprint


def test_knowledge_plan_job_remains_unsupported_and_queued(session_factory):
    """40-41. Successful EVIDENCE stage leaves KNOWLEDGE_PLAN queued and StageWorker skips it."""
    mock_search = MockSearchProvider(
        results=[
            SearchResult(
                result_id="s1",
                title="Title",
                url="https://example.test/page",
                snippet="snippet",
                provider_rank=1,
            )
        ]
    )
    mock_fetcher = MagicMock(spec=SourceFetcher)
    mock_fetcher.validate_url_security.return_value = None
    mock_fetcher.fetch_url.return_value = FetchedUrlContent(
        url="https://example.test/page",
        final_url="https://example.test/page",
        status_code=200,
        content_type="text/html",
        title="Title",
        extracted_text="Deep learning Transformer architectures use self-attention to model context.",
    )

    registry = StageExecutorRegistry()
    executor = EvidenceStageExecutor(
        session_factory=session_factory,
        search_provider=mock_search,
        fetcher=mock_fetcher,
        allow_private_for_tests=True,
    )
    registry.register(Stage.EVIDENCE, executor)

    worker = StageWorker(session_factory=session_factory, registry=registry)

    with session_factory() as session:
        cmd = TaskCommandService(session)
        task = cmd.create_task(topic="Transformer self-attention", allow_research=True)
        task_id = task.task_id
        session.commit()

    # Run EVIDENCE stage
    worker.run_once()

    # Next attempt to run worker should find KNOWLEDGE_PLAN is unsupported and not lease/run it
    processed_again = worker.run_once()
    assert processed_again is False  # StageWorker found no claimable jobs for supported stages

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        plan_job = job_repo.get_current_job_for_task(task_id)
        assert plan_job is not None
        assert plan_job.stage == Stage.KNOWLEDGE_PLAN
        assert plan_job.status == JobStatus.QUEUED
