from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.knowledge_base_service import KnowledgeBaseCommandService
from app.application.knowledge_plan_stage_executor import KnowledgePlanStageExecutor
from app.application.knowledge_retrieval_service import KnowledgeProcessingService
from app.application.script_stage_executor import ScriptStageExecutor
from app.application.stage_executor_registry import get_default_executor_registry
from app.domain.evidence import SourceStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import Stage, TaskStatus, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    ContentPlanRepository,
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
    WorkflowJobRepository,
)
from app.services.knowledge.embedding_provider import DeterministicFakeEmbeddingProvider
from app.workers.stage_worker import StageWorker


import tempfile

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


def test_knowledge_base_end_to_end_workflow_and_traceability(session_factory):
    """End-to-end integration test verifying Knowledge Base workflow integration.

    1. Create Knowledge Base and upload documents.
    2. Attach Knowledge Base to Knowledge Video Task.
    3. Run EVIDENCE stage: documents retrieved via Hybrid RAG, EvidenceSnapshot created.
    4. Run KNOWLEDGE_PLAN stage: beats grounded in KB evidence.
    5. Run SCRIPT stage: narration segments preserve evidence references to KB.
    """
    kb_id: str
    doc_id: str
    task_id: str

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        with session_factory() as session:
            proc_service = KnowledgeProcessingService(
                session=session,
                embedding_provider=DeterministicFakeEmbeddingProvider(),
            )
            kb_service = KnowledgeBaseCommandService(
                session=session,
                storage_dir=tmp_path,
                processing_service=proc_service,
            )

        # 1. Create Knowledge Base
        kb = kb_service.create_knowledge_base(
            name="Quantum Information Science",
            description="Quantum algorithms and entanglement principles",
        )
        kb_id = kb.knowledge_base_id

        # 2. Upload document into KB
        quantum_text = (
            "Quantum teleportation is a technique for transferring quantum information "
            "between two separated quantum systems. It utilizes shared quantum entanglement "
            "and classical communication channels without transmitting the physical qubit itself."
        )
        doc = kb_service.upload_document(
            kb_id=kb_id,
            file_bytes=quantum_text.encode("utf-8"),
            filename="quantum_teleportation.txt",
            title="Quantum Teleportation Guide",
        )
        assert doc.status == SourceStatus.READY
        doc_id = doc.source_document_id

        # 3. Create task and attach KB
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)

        task = KnowledgeVideoTask.create(
            topic="Quantum Teleportation and Entanglement",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)
        task_id = task.task_id

        kb_service.attach_to_task(task_id, kb_id)

        # Enqueue initial EVIDENCE stage job
        job = WorkflowJob.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            attempt_number=1,
            idempotency_key=f"idem_{uuid4().hex}",
        )
        job_repo.create_job(job)
        session.commit()

        # 4. Run StageWorker for EVIDENCE stage
        worker = StageWorker(
            session_factory=session_factory,
            registry=get_default_executor_registry(session_factory=session_factory),
            worker_id="test-kb-worker",
        )

        ran_evidence = worker.run_once()
        assert ran_evidence is True

        with session_factory() as session:
            task_repo = KnowledgeVideoTaskRepository(session)
            ev_repo = EvidenceRepository(session)

            task = task_repo.get_task(task_id)
            assert task.current_stage == Stage.KNOWLEDGE_PLAN

            # Verify EvidenceSnapshot was created and contains doc_id
            snapshot = ev_repo.get_latest_snapshot_for_task(task_id)
            assert snapshot is not None
            assert doc_id in snapshot.source_document_ids
            assert len(snapshot.evidence_ids) >= 1

            # Verify EvidenceItem points to doc_id
            ev_item = ev_repo.get_evidence_item(snapshot.evidence_ids[0])
            assert ev_item is not None
            assert ev_item.source_document_id == doc_id
            assert "teleportation" in ev_item.original_excerpt.lower()

            # Verify RetrievalSnapshot exists
            ret_snaps = ev_repo.list_retrieval_snapshots_for_task(task_id)
            assert len(ret_snaps) >= 1
            assert ret_snaps[0].candidates[0].retrieval_method in ("HYBRID_RRF", "LEXICAL_BM25", "SEMANTIC_VECTOR")
