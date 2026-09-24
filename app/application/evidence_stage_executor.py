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
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import ArtifactType, TaskArtifactRef
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobErrorType, Stage
from app.persistence.repositories import EvidenceRepository
from app.persistence.session import get_session
from app.services.knowledge.bm25_retriever import DEFAULT_RETRIEVAL_POLICY_VERSION
from app.services.knowledge.chunking import KnowledgeChunker
from app.services.knowledge.document_parser import DocumentParser
from app.services.knowledge.source_fetcher import SourceFetcher


class EvidenceStageExecutor:
    """Production StageExecutor for the EVIDENCE stage in the unified workflow.

    Executes the end-to-end evidence pipeline:
    1. Loads task-associated source documents (strictly scoped to task_id).
    2. Reuses existing chunks or processes unprocessed sources into deterministic chunks.
    3. Handles partial source failures truthfully.
    4. Executes scoped lexical retrieval over processed sources.
    5. Evaluates evidence sufficiency via deterministic EvidenceAvailabilityPolicy.
    6. On insufficiency: returns structured NEEDS_EVIDENCE result.
    7. On sufficiency: creates immutable EvidenceSnapshot and TaskArtifactRef.
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
    ) -> None:
        self._session_factory = session_factory
        self.policy = policy or EvidenceAvailabilityPolicy()
        self.top_k = top_k
        self.retrieval_policy_version = retrieval_policy_version
        self.fetcher = fetcher
        self.parser = parser
        self.chunker = chunker

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
        # 1. Short TX: Load registered task-scoped sources
        # ---------------------------------------------------------------------
        sources: list[SourceDocument] = []
        with get_session(self._session_factory) as session:
            ev_repo = EvidenceRepository(session)
            sources = ev_repo.list_sources_for_task(task_id)

        task_source_count = len(sources)
        if task_source_count == 0:
            eval_res = self.policy.evaluate(
                task_source_count=0,
                processed_source_ids=(),
                selected_evidence_items=(),
            )
            logger.warning(
                f"[EvidenceStageExecutor] Task '{task_id}' has 0 registered sources -> NEEDS_EVIDENCE ({eval_res.reason_code})."
            )
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

        # ---------------------------------------------------------------------
        # 2. Process / parse / chunk registered sources (Short TX per source)
        # ---------------------------------------------------------------------
        processed_source_ids: list[str] = []
        failed_sources: list[dict[str, Any]] = []

        for source in sources:
            s_id = source.source_document_id
            try:
                # Check if already successfully processed under current version
                with get_session(self._session_factory) as session:
                    ev_repo = EvidenceRepository(session)
                    existing_chunks = ev_repo.list_chunks_for_source(s_id)
                    current_src = ev_repo.get_source_document(s_id)
                    if existing_chunks and current_src and current_src.status == SourceStatus.READY:
                        processed_source_ids.append(s_id)
                        continue

                # Process source in a dedicated short transaction
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
                f"[EvidenceStageExecutor] Task '{task_id}' has no processable sources -> NEEDS_EVIDENCE ({eval_res.reason_code})."
            )
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

        # ---------------------------------------------------------------------
        # 4. Short TX: Execute retrieval over processed sources
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

        # ---------------------------------------------------------------------
        # 6. Short TX: Create immutable EvidenceSnapshot and TaskArtifactRef
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
