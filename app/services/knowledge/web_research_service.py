from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any, Sequence
from uuid import uuid4

from loguru import logger
from sqlalchemy.orm import Session

from app.domain.evidence import (
    SearchResult,
    SourceDocument,
    SourceQualityTier,
    SourceStatus,
    SourceType,
    WebResearchSnapshot,
    classify_source_quality,
    compute_sha256,
    compute_source_fingerprint,
)
from app.domain.trace import TraceEventType
from app.persistence.repositories import EvidenceRepository
from app.services.knowledge.search_provider import SearchProvider
from app.services.knowledge.source_fetcher import SourceFetcher
from app.services.trace_service import TraceWriter


@dataclass(frozen=True)
class WebResearchConfig:
    """Bounded limits for web research execution."""

    max_queries: int = 1
    max_results_per_query: int = 5
    max_pages_to_fetch: int = 3
    fetch_timeout_seconds: float = 10.0
    fetch_max_bytes: int = 5 * 1024 * 1024


class WebResearchService:
    """Service executing bounded, safe web research augmentation for a task.

    1. Discovers candidate URLs via SearchProvider (discovery metadata only).
    2. Enforces strict query, result count, and page fetch budgets.
    3. Deduplicates URLs against existing task sources and within the search results.
    4. Fetches original pages via real E2 SourceFetcher (SSRF, timeout, max bytes).
    5. Saves successfully fetched pages as task-associated SourceDocuments.
    6. Creates and persists an immutable WebResearchSnapshot audit record.
    """

    def __init__(
        self,
        session: Session,
        search_provider: SearchProvider,
        fetcher: SourceFetcher | None = None,
        config: WebResearchConfig | None = None,
        trace_writer: TraceWriter | None = None,
        allow_private_for_tests: bool = False,
    ) -> None:
        self._session = session
        self.evidence_repo = EvidenceRepository(session)
        self.search_provider = search_provider
        self.fetcher = fetcher or SourceFetcher(
            timeout=(config or WebResearchConfig()).fetch_timeout_seconds,
            max_bytes=(config or WebResearchConfig()).fetch_max_bytes,
        )
        self.config = config or WebResearchConfig()
        self.trace_writer = trace_writer
        self.allow_private_for_tests = allow_private_for_tests

    def execute_research(
        self,
        task_id: str,
        query: str,
        stage_attempt: int = 1,
        now: datetime | None = None,
    ) -> tuple[WebResearchSnapshot, list[SourceDocument]]:
        ts = now or datetime.now(UTC)
        provider_name = self.search_provider.__class__.__name__

        logger.info(
            f"[WebResearchService] Starting web research for task '{task_id}', attempt {stage_attempt} using provider '{provider_name}'."
        )

        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        if self.trace_writer:
            self.trace_writer.record_instant(
                event_type=TraceEventType.WEB_RESEARCH_STARTED,
                attributes={
                    "task_id": task_id,
                    "stage_attempt": stage_attempt,
                    "query_hash": query_hash,
                    "provider": provider_name,
                },
            )

        # 1. Bounded Search Execution
        search_results: Sequence[SearchResult] = ()
        try:
            search_results = self.search_provider.search(
                query=query,
                limit=self.config.max_results_per_query,
            )
        except Exception as exc:
            logger.warning(
                f"[WebResearchService] Search query failed for task '{task_id}': {exc}"
            )
            if self.trace_writer:
                self.trace_writer.record_instant(
                    event_type=TraceEventType.WEB_RESEARCH_EXHAUSTED,
                    attributes={
                        "task_id": task_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            raise

        if self.trace_writer:
            self.trace_writer.record_instant(
                event_type=TraceEventType.WEB_SEARCH_COMPLETED,
                attributes={
                    "task_id": task_id,
                    "result_count": len(search_results),
                    "provider": provider_name,
                },
            )

        if not search_results:
            logger.info(
                f"[WebResearchService] 0 search results returned for task '{task_id}'."
            )
            empty_snap = WebResearchSnapshot.create(
                task_id=task_id,
                stage_attempt=stage_attempt,
                query=query,
                provider=provider_name,
                search_results=(),
                selected_urls=(),
                fetch_outcomes=(),
                created_source_document_ids=(),
                now=ts,
            )
            saved_empty_snap = self.evidence_repo.save_web_research_snapshot(empty_snap)
            if self.trace_writer:
                self.trace_writer.record_instant(
                    event_type=TraceEventType.WEB_RESEARCH_EXHAUSTED,
                    attributes={"task_id": task_id, "reason": "NO_RESULTS"},
                )
            return saved_empty_snap, []

        # 2. Deduplication & Selection
        existing_sources = self.evidence_repo.list_sources_for_task(task_id)
        existing_locators = {
            s.source_locator.strip().lower() for s in existing_sources
        }

        selected_urls: list[str] = []
        seen_urls: set[str] = set()

        for res in search_results:
            clean_u = res.url.strip()
            norm_u = clean_u.lower()
            if not clean_u:
                continue
            if norm_u in existing_locators or norm_u in seen_urls:
                continue
            seen_urls.add(norm_u)
            selected_urls.append(clean_u)
            if len(selected_urls) >= self.config.max_pages_to_fetch:
                break

        # 3. Safe Page Acquisition
        created_source_docs: list[SourceDocument] = []
        fetch_outcomes: list[dict[str, Any]] = []

        for target_url in selected_urls:
            # 3a. Security validation
            try:
                self.fetcher.validate_url_security(
                    target_url,
                    allow_private_for_tests=self.allow_private_for_tests,
                )
            except Exception as sec_exc:
                logger.warning(
                    f"[WebResearchService] URL security validation failed for '{target_url}': {sec_exc}"
                )
                fetch_outcomes.append({
                    "url": target_url,
                    "status": "REJECTED_SECURITY",
                    "error": str(sec_exc),
                    "error_type": type(sec_exc).__name__,
                })
                if self.trace_writer:
                    self.trace_writer.record_instant(
                        event_type=TraceEventType.WEB_SOURCE_REJECTED,
                        attributes={
                            "task_id": task_id,
                            "url": target_url,
                            "reason": "SECURITY_VIOLATION",
                        },
                    )
                continue

            # 3b. Bounded fetch and text extraction
            try:
                fetched = self.fetcher.fetch_url(
                    target_url,
                    allow_private_for_tests=self.allow_private_for_tests,
                )
                if fetched.status_code >= 400:
                    fetch_outcomes.append({
                        "url": target_url,
                        "final_url": fetched.final_url,
                        "status": "FETCH_FAILED",
                        "status_code": fetched.status_code,
                    })
                    if self.trace_writer:
                        self.trace_writer.record_instant(
                            event_type=TraceEventType.WEB_SOURCE_REJECTED,
                            attributes={
                                "task_id": task_id,
                                "url": target_url,
                                "reason": f"HTTP_{fetched.status_code}",
                            },
                        )
                    continue

                extracted_text = (fetched.extracted_text or "").strip()
                if not extracted_text:
                    fetch_outcomes.append({
                        "url": target_url,
                        "final_url": fetched.final_url,
                        "status": "NO_TEXT_EXTRACTED",
                    })
                    if self.trace_writer:
                        self.trace_writer.record_instant(
                            event_type=TraceEventType.WEB_SOURCE_REJECTED,
                            attributes={
                                "task_id": task_id,
                                "url": target_url,
                                "reason": "EMPTY_EXTRACTED_TEXT",
                            },
                        )
                    continue

                quality_tier = classify_source_quality(
                    url=fetched.final_url or target_url
                )
                c_hash = compute_sha256(extracted_text)
                canonical_loc = fetched.final_url or target_url
                fp = compute_source_fingerprint(
                    source_type=SourceType.URL,
                    canonical_locator=canonical_loc,
                    content_hash=c_hash,
                )

                # Check if identical content hash already exists for this task
                if any(
                    s.content_hash == c_hash
                    for s in existing_sources + created_source_docs
                ):
                    fetch_outcomes.append({
                        "url": target_url,
                        "final_url": canonical_loc,
                        "status": "DUPLICATE_CONTENT",
                        "content_hash": c_hash,
                    })
                    continue

                source_doc = SourceDocument(
                    source_document_id=f"src_{uuid4().hex[:24]}",
                    source_type=SourceType.URL,
                    title=fetched.title or f"Web Source: {target_url}",
                    source_locator=canonical_loc,
                    content_snapshot=extracted_text,
                    content_hash=c_hash,
                    source_fingerprint=fp,
                    author=None,
                    published_at=None,
                    captured_at=ts,
                    media_type="text/html",
                    status=SourceStatus.READY,
                    metadata_json={
                        "original_search_url": target_url,
                        "final_url": fetched.final_url,
                        "quality_tier": quality_tier.value,
                        "content_type": fetched.content_type,
                        "retrieved_at": ts.isoformat(),
                        "is_web_research": True,
                    },
                    created_at=ts,
                )
                saved_doc = self.evidence_repo.save_source_document(source_doc)
                self.evidence_repo.associate_task_source(
                    task_id=task_id,
                    source_document_id=saved_doc.source_document_id,
                    role="RESEARCH",
                    now=ts,
                )
                created_source_docs.append(saved_doc)
                fetch_outcomes.append({
                    "url": target_url,
                    "final_url": canonical_loc,
                    "status": "SUCCESS",
                    "source_document_id": saved_doc.source_document_id,
                    "quality_tier": quality_tier.value,
                    "text_bytes": len(extracted_text.encode("utf-8")),
                })
                if self.trace_writer:
                    self.trace_writer.record_instant(
                        event_type=TraceEventType.WEB_SOURCE_FETCHED,
                        attributes={
                            "task_id": task_id,
                            "source_document_id": saved_doc.source_document_id,
                            "url": target_url,
                            "quality_tier": quality_tier.value,
                        },
                    )
            except Exception as fetch_exc:
                logger.warning(
                    f"[WebResearchService] Fetch exception for '{target_url}': {fetch_exc}"
                )
                fetch_outcomes.append({
                    "url": target_url,
                    "status": "FETCH_ERROR",
                    "error": str(fetch_exc),
                    "error_type": type(fetch_exc).__name__,
                })
                if self.trace_writer:
                    self.trace_writer.record_instant(
                        event_type=TraceEventType.WEB_SOURCE_REJECTED,
                        attributes={
                            "task_id": task_id,
                            "url": target_url,
                            "error": str(fetch_exc),
                        },
                    )

        # 4. Create and persist immutable WebResearchSnapshot
        snapshot = WebResearchSnapshot.create(
            task_id=task_id,
            stage_attempt=stage_attempt,
            query=query,
            provider=provider_name,
            search_results=search_results,
            selected_urls=selected_urls,
            fetch_outcomes=fetch_outcomes,
            created_source_document_ids=[
                s.source_document_id for s in created_source_docs
            ],
            now=ts,
        )
        saved_snapshot = self.evidence_repo.save_web_research_snapshot(snapshot)

        if not created_source_docs:
            logger.warning(
                f"[WebResearchService] Research for task '{task_id}' yielded 0 usable SourceDocuments."
            )
            if self.trace_writer:
                self.trace_writer.record_instant(
                    event_type=TraceEventType.WEB_RESEARCH_EXHAUSTED,
                    attributes={
                        "task_id": task_id,
                        "snapshot_id": saved_snapshot.web_research_snapshot_id,
                        "selected_url_count": len(selected_urls),
                    },
                )
        else:
            logger.info(
                f"[WebResearchService] Research for task '{task_id}' successfully created {len(created_source_docs)} SourceDocuments."
            )
            if self.trace_writer:
                self.trace_writer.record_instant(
                    event_type=TraceEventType.WEB_RESEARCH_COMPLETED,
                    attributes={
                        "task_id": task_id,
                        "snapshot_id": saved_snapshot.web_research_snapshot_id,
                        "source_document_count": len(created_source_docs),
                    },
                )

        return saved_snapshot, created_source_docs
