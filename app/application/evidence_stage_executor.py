from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from app.application.knowledge_retrieval_service import (
    KnowledgeProcessingService,
    KnowledgeRetrievalService,
)
from app.application.stage_executor_protocol import (
    StageExecutionResult,
    StageExecutorProtocol,
)
from app.domain.evidence import (
    EvidenceAvailabilityPolicy,
    EvidenceItem,
    EvidenceSnapshot,
    EvidenceSufficiencyResult,
    SourceDocument,
    SourceStatus,
    WebResearchSnapshot,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobErrorType, Stage
from app.persistence.repositories import EvidenceRepository, KnowledgeBaseRepository
from app.persistence.session import get_session
from app.services.knowledge.bm25_retriever import DEFAULT_RETRIEVAL_POLICY_VERSION
from app.services.knowledge.chunking import KnowledgeChunker
from app.services.knowledge.document_parser import DocumentParser
from app.services.knowledge.search_provider import (
    SearchCredentialsMissingError,
    SearchProvider,
    SearchProviderRateLimitError,
    SearchProviderUnavailableError,
    TavilySearchProvider,
)
from app.services.knowledge.source_fetcher import SourceFetcher
from app.services.knowledge.web_research_service import (
    WebResearchConfig,
    WebResearchService,
)
from app.services.trace_service import TraceWriter


class EvidenceStageExecutor:
    """Production StageExecutor for the EVIDENCE stage in the unified workflow.

    Executes the prioritized evidence pipeline:
    1. USER SOURCES FIRST: Loads and processes registered task-scoped sources.
    2. Runs lexical retrieval over processed user sources.
    3. Evaluates evidence sufficiency via deterministic EvidenceAvailabilityPolicy.
    4. IF SUFFICIENT: Completes immediately with zero web calls.
    5. IF INSUFFICIENT AND NOT AUTHORIZED: Returns NEEDS_EVIDENCE with zero web calls.
    6. IF INSUFFICIENT AND AUTHORIZED: Executes bounded WebResearchService.
    7. Processes newly fetched web sources through E2 chunking.
    8. Reruns scoped retrieval over all sources (user + web) -> new RetrievalSnapshot.
    9. Re-evaluates evidence sufficiency -> new EvidenceSnapshot or structured NEEDS_EVIDENCE.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
        policy: EvidenceAvailabilityPolicy | None = None,
        top_k: int = 10,
        retrieval_policy_version: str = DEFAULT_RETRIEVAL_POLICY_VERSION,
        fetcher: SourceFetcher | None = None,
        parser: DocumentParser | None = None,
        chunker: KnowledgeChunker | None = None,
        search_provider: SearchProvider | None = None,
        web_research_service: WebResearchService | None = None,
        research_config: WebResearchConfig | None = None,
        trace_writer: TraceWriter | None = None,
        allow_private_for_tests: bool = False,
    ) -> None:
        self._session_factory = session_factory
        self.policy = policy or EvidenceAvailabilityPolicy()
        self.top_k = top_k
        self.retrieval_policy_version = retrieval_policy_version
        self.fetcher = fetcher
        self.parser = parser
        self.chunker = chunker
        self.search_provider = search_provider
        self.web_research_service = web_research_service
        self.research_config = research_config or WebResearchConfig()
        self.trace_writer = trace_writer
        self.allow_private_for_tests = allow_private_for_tests

    def execute(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
    ) -> StageExecutionResult:
        task_id = task.task_id
        logger.info(
            f"[EvidenceStageExecutor] Starting execution for task '{task_id}', job '{job.job_id}', attempt {job.attempt_number}."
        )

        # ---------------------------------------------------------------------
        # 1. Short TX: Load registered task-scoped sources (USER SOURCES FIRST)
        # ---------------------------------------------------------------------
        sources: list[SourceDocument] = []
        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            kb_repo = KnowledgeBaseRepository(session)
            sources = kb_repo.list_sources_for_task_with_kbs(task_id)
            if not sources:
                sources = ev_repo.list_sources_for_task(task_id)

        task_source_count = len(sources)
        if task_source_count == 0:
            eval_res = self.policy.evaluate(
                task_source_count=0,
                processed_source_ids=(),
                selected_evidence_items=(),
            )
            logger.warning(
                f"[EvidenceStageExecutor] Task '{task_id}' has 0 registered sources -> insufficient ({eval_res.reason_code})."
            )
            if not task.is_research_authorized:
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message=eval_res.message or "No evidence sources registered.",
                    is_retryable=False,
                    metadata_json={
                        "reason_code": eval_res.reason_code,
                        "task_source_count": 0,
                        "processed_source_count": 0,
                        "selected_evidence_count": 0,
                        "failed_sources": [],
                    },
                )
            # Proceed to web research
            return self._execute_web_research_augmentation(
                task=task,
                job=job,
                initial_sources=[],
                initial_processed_source_ids=[],
                initial_failed_sources=[],
                initial_retrieval_snapshot_id=None,
                initial_eval_res=eval_res,
            )

        # ---------------------------------------------------------------------
        # 2. Process / parse / chunk registered sources (Short TX per source)
        # ---------------------------------------------------------------------
        processed_source_ids: list[str] = []
        failed_sources: list[dict[str, Any]] = []

        for source in sources:
            s_id = source.source_document_id
            try:
                with get_session(self._session_factory) as session:
                    ev_repo = EvidenceRepository(session)
                    existing_chunks = ev_repo.list_chunks_for_source(s_id)
                    current_src = ev_repo.get_source_document(s_id)
                    if existing_chunks and current_src and current_src.status == SourceStatus.READY:
                        processed_source_ids.append(s_id)
                        continue

                with get_session(self._session_factory) as session:
                    proc_service = KnowledgeProcessingService(
                        session=session,
                        fetcher=self.fetcher,
                        parser=self.parser,
                        chunker=self.chunker,
                    )
                    chunks = proc_service.process_source_document(s_id)
                    if chunks:
                        processed_source_ids.append(s_id)
                    else:
                        failed_sources.append({
                            "source_document_id": s_id,
                            "error": "Source produced 0 chunks after parsing.",
                            "error_type": "EmptyChunksError",
                        })
            except Exception as exc:
                logger.warning(f"[EvidenceStageExecutor] Failed to process source '{s_id}': {exc}")
                failed_sources.append({
                    "source_document_id": s_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })

        # ---------------------------------------------------------------------
        # 3. Check processable sources sufficiency
        # ---------------------------------------------------------------------
        if not processed_source_ids:
            eval_res = self.policy.evaluate(
                task_source_count=task_source_count,
                processed_source_ids=(),
                selected_evidence_items=(),
            )
            logger.warning(
                f"[EvidenceStageExecutor] Task '{task_id}' has no processable sources -> insufficient ({eval_res.reason_code})."
            )
            if not task.is_research_authorized:
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message=eval_res.message or "None of the registered sources could be parsed or processed.",
                    is_retryable=False,
                    metadata_json={
                        "reason_code": eval_res.reason_code,
                        "task_source_count": task_source_count,
                        "processed_source_count": 0,
                        "selected_evidence_count": 0,
                        "failed_sources": failed_sources,
                    },
                )
            # Proceed to web research
            return self._execute_web_research_augmentation(
                task=task,
                job=job,
                initial_sources=sources,
                initial_processed_source_ids=[],
                initial_failed_sources=failed_sources,
                initial_retrieval_snapshot_id=None,
                initial_eval_res=eval_res,
            )

        # ---------------------------------------------------------------------
        # 4. Short TX: Execute retrieval over processed sources (R1)
        # ---------------------------------------------------------------------
        query = task.topic
        retrieval_snapshot_id: str | None = None
        selected_evidence_items: list[EvidenceItem] = []

        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            retrieval_service = KnowledgeRetrievalService(
                session=session,
                retrieval_policy_version=self.retrieval_policy_version,
            )
            ret_snap = retrieval_service.retrieve(
                task_id=task_id,
                query=query,
                top_k=self.top_k,
                source_scope_ids=processed_source_ids,
                auto_create_evidence_items=True,
            )
            retrieval_snapshot_id = ret_snap.retrieval_snapshot_id
            for eid in ret_snap.selected_evidence_ids:
                item = ev_repo.get_evidence_item(eid)
                if item is not None:
                    selected_evidence_items.append(item)

        # ---------------------------------------------------------------------
        # 5. Evaluate evidence sufficiency via deterministic policy
        # ---------------------------------------------------------------------
        eval_res = self.policy.evaluate(
            task_source_count=task_source_count,
            processed_source_ids=processed_source_ids,
            selected_evidence_items=selected_evidence_items,
        )

        if not eval_res.is_sufficient:
            logger.warning(
                f"[EvidenceStageExecutor] Evidence sufficiency check failed for task '{task_id}': {eval_res.reason_code} - {eval_res.message}"
            )
            if not task.is_research_authorized:
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.NEEDS_EVIDENCE.value,
                    error_message=eval_res.message or "Insufficient evidence.",
                    is_retryable=False,
                    metadata_json={
                        "reason_code": eval_res.reason_code,
                        "task_source_count": task_source_count,
                        "processed_source_count": eval_res.processed_source_count,
                        "selected_evidence_count": eval_res.selected_evidence_count,
                        "failed_sources": failed_sources,
                        "retrieval_snapshot_id": retrieval_snapshot_id,
                    },
                )
            # Web research authorized: augment evidence
            return self._execute_web_research_augmentation(
                task=task,
                job=job,
                initial_sources=sources,
                initial_processed_source_ids=processed_source_ids,
                initial_failed_sources=failed_sources,
                initial_retrieval_snapshot_id=retrieval_snapshot_id,
                initial_eval_res=eval_res,
            )

        # ---------------------------------------------------------------------
        # 6. User evidence sufficient -> Create EvidenceSnapshot and TaskArtifactRef
        # ---------------------------------------------------------------------
        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            latest_snap = ev_repo.get_latest_snapshot_for_task(task_id)
            next_version = (latest_snap.snapshot_version + 1) if latest_snap else 1

            snapshot = EvidenceSnapshot.create(
                task_id=task_id,
                source_document_ids=processed_source_ids,
                evidence_ids=[item.evidence_id for item in selected_evidence_items],
                knowledge_claim_ids=(),
                snapshot_version=next_version,
            )
            saved_snapshot = ev_repo.save_evidence_snapshot(snapshot)

            art_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.EVIDENCE,
                artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
                artifact_id=saved_snapshot.evidence_snapshot_id,
                artifact_version=str(saved_snapshot.snapshot_version),
                metadata_json={
                    "retrieval_snapshot_id": retrieval_snapshot_id,
                    "source_count": len(processed_source_ids),
                    "evidence_count": len(selected_evidence_items),
                    "content_fingerprint": saved_snapshot.content_fingerprint,
                    "failed_sources_count": len(failed_sources),
                },
            )

        logger.info(
            f"[EvidenceStageExecutor] EVIDENCE stage succeeded for task '{task_id}'. "
            f"Snapshot: '{saved_snapshot.evidence_snapshot_id}', ArtifactRef: '{art_ref.task_artifact_ref_id}'."
        )

        return StageExecutionResult(
            success=True,
            output_artifact_ref=art_ref,
            output_task_artifact_ref_id=art_ref.task_artifact_ref_id,
            metadata_json={
                "evidence_snapshot_id": saved_snapshot.evidence_snapshot_id,
                "retrieval_snapshot_id": retrieval_snapshot_id,
                "snapshot_version": saved_snapshot.snapshot_version,
                "processed_source_count": len(processed_source_ids),
                "selected_evidence_count": len(selected_evidence_items),
                "failed_sources": failed_sources,
            },
        )

    def _execute_web_research_augmentation(
        self,
        task: KnowledgeVideoTask,
        job: WorkflowJob,
        initial_sources: list[SourceDocument],
        initial_processed_source_ids: list[str],
        initial_failed_sources: list[dict[str, Any]],
        initial_retrieval_snapshot_id: str | None,
        initial_eval_res: EvidenceSufficiencyResult,
    ) -> StageExecutionResult:
        """Executes bounded web research augmentation and reruns scoped retrieval."""
        task_id = task.task_id
        logger.info(
            f"[EvidenceStageExecutor] Executing authorized web research for task '{task_id}' (attempt {job.attempt_number})."
        )

        # 1. Execute bounded WebResearchService
        research_snapshot: WebResearchSnapshot
        web_sources: list[SourceDocument] = []
        with get_session(self._session_factory) as session:
            sp = self.search_provider or TavilySearchProvider()
            web_service = self.web_research_service or WebResearchService(
                session=session,
                search_provider=sp,
                fetcher=self.fetcher,
                config=self.research_config,
                trace_writer=self.trace_writer,
                allow_private_for_tests=self.allow_private_for_tests,
            )
            try:
                research_snapshot, web_sources = web_service.execute_research(
                    task_id=task_id,
                    query=task.topic,
                    stage_attempt=job.attempt_number,
                )
            except SearchCredentialsMissingError as cred_exc:
                logger.error(f"[EvidenceStageExecutor] Search credentials missing: {cred_exc}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.FATAL.value,
                    error_message=str(cred_exc),
                    is_retryable=False,
                    metadata_json={
                        "error": str(cred_exc),
                        "category": "CONFIG_ERROR",
                        "retrieval_snapshot_id": initial_retrieval_snapshot_id,
                    },
                )
            except (SearchProviderUnavailableError, SearchProviderRateLimitError) as retry_exc:
                logger.warning(f"[EvidenceStageExecutor] Search provider temporary failure: {retry_exc}")
                return StageExecutionResult(
                    success=False,
                    error_type=JobErrorType.RETRYABLE.value,
                    error_message=str(retry_exc),
                    is_retryable=True,
                    metadata_json={
                        "error": str(retry_exc),
                        "retrieval_snapshot_id": initial_retrieval_snapshot_id,
                    },
                )

        # 2. Check if web research yielded usable sources
        if not web_sources:
            logger.warning(
                f"[EvidenceStageExecutor] Web research produced 0 usable sources for task '{task_id}' -> NEEDS_EVIDENCE."
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.NEEDS_EVIDENCE.value,
                error_message="Web research exhausted: no usable sources could be acquired.",
                is_retryable=False,
                metadata_json={
                    "reason_code": "RESEARCH_EXHAUSTED",
                    "web_research_snapshot_id": research_snapshot.web_research_snapshot_id,
                    "retrieval_snapshot_id": initial_retrieval_snapshot_id,
                    "task_source_count": len(initial_sources),
                    "processed_source_count": len(initial_processed_source_ids),
                    "selected_evidence_count": 0,
                    "failed_sources": initial_failed_sources,
                },
            )

        # 3. Process / parse / chunk new web sources
        augmented_processed_source_ids = list(initial_processed_source_ids)
        failed_web_sources: list[dict[str, Any]] = []

        for w_src in web_sources:
            s_id = w_src.source_document_id
            try:
                with get_session(self._session_factory) as session:
                    proc_service = KnowledgeProcessingService(
                        session=session,
                        fetcher=self.fetcher,
                        parser=self.parser,
                        chunker=self.chunker,
                    )
                    chunks = proc_service.process_source_document(s_id)
                    if chunks:
                        if s_id not in augmented_processed_source_ids:
                            augmented_processed_source_ids.append(s_id)
                    else:
                        failed_web_sources.append({
                            "source_document_id": s_id,
                            "error": "Web source produced 0 chunks after parsing.",
                            "error_type": "EmptyChunksError",
                        })
            except Exception as exc:
                logger.warning(f"[EvidenceStageExecutor] Failed to process web source '{s_id}': {exc}")
                failed_web_sources.append({
                    "source_document_id": s_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })

        # 4. Rerun lexical retrieval over all processed sources (R2)
        retrieval_snapshot_id_2: str | None = None
        selected_evidence_items_2: list[EvidenceItem] = []

        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            retrieval_service = KnowledgeRetrievalService(
                session=session,
                retrieval_policy_version=self.retrieval_policy_version,
            )
            ret_snap_2 = retrieval_service.retrieve(
                task_id=task_id,
                query=task.topic,
                top_k=self.top_k,
                source_scope_ids=augmented_processed_source_ids,
                auto_create_evidence_items=True,
            )
            retrieval_snapshot_id_2 = ret_snap_2.retrieval_snapshot_id
            for eid in ret_snap_2.selected_evidence_ids:
                item = ev_repo.get_evidence_item(eid)
                if item is not None:
                    selected_evidence_items_2.append(item)

        # 5. Re-evaluate evidence sufficiency
        all_failed_sources = initial_failed_sources + failed_web_sources
        total_source_count = len(initial_sources) + len(web_sources)
        eval_res_2 = self.policy.evaluate(
            task_source_count=total_source_count,
            processed_source_ids=augmented_processed_source_ids,
            selected_evidence_items=selected_evidence_items_2,
        )

        if not eval_res_2.is_sufficient:
            logger.warning(
                f"[EvidenceStageExecutor] Evidence still insufficient after web research for task '{task_id}': {eval_res_2.reason_code}"
            )
            return StageExecutionResult(
                success=False,
                error_type=JobErrorType.NEEDS_EVIDENCE.value,
                error_message=eval_res_2.message or "Evidence still insufficient after web research.",
                is_retryable=False,
                metadata_json={
                    "reason_code": eval_res_2.reason_code,
                    "web_research_snapshot_id": research_snapshot.web_research_snapshot_id,
                    "retrieval_snapshot_id": retrieval_snapshot_id_2,
                    "initial_retrieval_snapshot_id": initial_retrieval_snapshot_id,
                    "task_source_count": total_source_count,
                    "processed_source_count": eval_res_2.processed_source_count,
                    "selected_evidence_count": eval_res_2.selected_evidence_count,
                    "failed_sources": all_failed_sources,
                },
            )

        # 6. Web research succeeded -> create new EvidenceSnapshot and TaskArtifactRef
        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            latest_snap = ev_repo.get_latest_snapshot_for_task(task_id)
            next_version = (latest_snap.snapshot_version + 1) if latest_snap else 1

            snapshot = EvidenceSnapshot.create(
                task_id=task_id,
                source_document_ids=augmented_processed_source_ids,
                evidence_ids=[item.evidence_id for item in selected_evidence_items_2],
                knowledge_claim_ids=(),
                snapshot_version=next_version,
            )
            saved_snapshot = ev_repo.save_evidence_snapshot(snapshot)

            art_ref = TaskArtifactRef.create(
                task_id=task_id,
                stage=Stage.EVIDENCE,
                artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
                artifact_id=saved_snapshot.evidence_snapshot_id,
                artifact_version=str(saved_snapshot.snapshot_version),
                metadata_json={
                    "retrieval_snapshot_id": retrieval_snapshot_id_2,
                    "initial_retrieval_snapshot_id": initial_retrieval_snapshot_id,
                    "web_research_snapshot_id": research_snapshot.web_research_snapshot_id,
                    "source_count": len(augmented_processed_source_ids),
                    "evidence_count": len(selected_evidence_items_2),
                    "content_fingerprint": saved_snapshot.content_fingerprint,
                    "failed_sources_count": len(all_failed_sources),
                },
            )

        logger.info(
            f"[EvidenceStageExecutor] EVIDENCE stage succeeded with web research for task '{task_id}'. "
            f"Snapshot: '{saved_snapshot.evidence_snapshot_id}', ArtifactRef: '{art_ref.task_artifact_ref_id}'."
        )

        return StageExecutionResult(
            success=True,
            output_artifact_ref=art_ref,
            output_task_artifact_ref_id=art_ref.task_artifact_ref_id,
            metadata_json={
                "evidence_snapshot_id": saved_snapshot.evidence_snapshot_id,
                "retrieval_snapshot_id": retrieval_snapshot_id_2,
                "web_research_snapshot_id": research_snapshot.web_research_snapshot_id,
                "snapshot_version": saved_snapshot.snapshot_version,
                "processed_source_count": len(augmented_processed_source_ids),
                "selected_evidence_count": len(selected_evidence_items_2),
                "failed_sources": all_failed_sources,
            },
        )
