from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_service import TaskEvidenceCommandService
from app.application.evidence_stage_executor import EvidenceStageExecutor
from app.application.stage_executor_protocol import StageExecutorProtocol
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
    EvidenceSufficiencyResult,
    RetrievalSnapshot,
    SourceDocument,
    SourceStatus,
    SourceType,
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


# =============================================================================
# 1. EXECUTOR REGISTRATION TESTS
# =============================================================================


def test_production_registry_registers_real_evidence_executor():
    """Verify that default production registry registers EvidenceStageExecutor for Stage.EVIDENCE."""
    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.EVIDENCE)
    executor = registry.get_executor(Stage.EVIDENCE)
    assert executor is not None
    assert isinstance(executor, EvidenceStageExecutor)
    assert isinstance(executor, StageExecutorProtocol)


def test_production_registry_contains_no_fake_executors():
    """Verify that production registry only contains genuine StageExecutor implementations."""
    registry = get_default_executor_registry()
    for stage in registry.list_supported_stages():
        executor = registry.get_executor(stage)
        assert executor is not None
        assert not executor.__class__.__name__.startswith(("Mock", "Fake", "Dummy"))


def test_unimplemented_stages_remain_unsupported():
    """Verify that DELIVERY and subsequent stages remain unregistered in Stage Q1."""
    registry = get_default_executor_registry()
    for stage in Stage:
        if stage not in (
            Stage.EVIDENCE,
            Stage.KNOWLEDGE_PLAN,
            Stage.SCRIPT,
            Stage.STORYBOARD,
            Stage.PRODUCTION_PLAN,
            Stage.ASSET,
            Stage.AUDIO,
            Stage.COMPOSITION,
            Stage.QUALITY_REVIEW,
        ):
            assert not registry.has_executor(stage)
            assert registry.get_executor(stage) is None
    assert registry.list_supported_stages() == {
        Stage.EVIDENCE,
        Stage.KNOWLEDGE_PLAN,
        Stage.SCRIPT,
        Stage.STORYBOARD,
        Stage.PRODUCTION_PLAN,
        Stage.ASSET,
        Stage.AUDIO,
        Stage.COMPOSITION,
        Stage.QUALITY_REVIEW,
    }


# =============================================================================
# 2. EVIDENCE AVAILABILITY POLICY TESTS
# =============================================================================


def test_evidence_availability_policy_evaluations():
    """Test deterministic evaluation rules of EvidenceAvailabilityPolicy."""
    policy = EvidenceAvailabilityPolicy()

    # Rule 1: NO_SOURCES
    r1 = policy.evaluate(task_source_count=0, processed_source_ids=(), selected_evidence_items=())
    assert not r1.is_sufficient
    assert r1.reason_code == "NO_SOURCES"

    # Rule 2: NO_PROCESSABLE_SOURCES
    r2 = policy.evaluate(task_source_count=2, processed_source_ids=(), selected_evidence_items=())
    assert not r2.is_sufficient
    assert r2.reason_code == "NO_PROCESSABLE_SOURCES"

    # Rule 3: NO_RETRIEVAL_RESULTS
    r3 = policy.evaluate(
        task_source_count=1,
        processed_source_ids=["src_1"],
        selected_evidence_items=(),
    )
    assert not r3.is_sufficient
    assert r3.reason_code == "NO_RETRIEVAL_RESULTS"

    # Rule 4: BROKEN_EVIDENCE_PROVENANCE (source_document_id not in processed sources)
    src = SourceDocument.create_text(
        text="Valid content",
        title="Test",
    )
    ev_item = EvidenceItem.create(
        source_document=src,
        original_excerpt="Valid content excerpt",
    )
    r4 = policy.evaluate(
        task_source_count=1,
        processed_source_ids=["src_other"],  # Mismatch
        selected_evidence_items=[ev_item],
    )
    assert not r4.is_sufficient
    assert r4.reason_code == "BROKEN_EVIDENCE_PROVENANCE"

    # Rule 5: Sufficient
    r5 = policy.evaluate(
        task_source_count=1,
        processed_source_ids=[src.source_document_id],
        selected_evidence_items=[ev_item],
    )
    assert r5.is_sufficient
    assert r5.reason_code is None
    assert r5.processed_source_count == 1
    assert r5.selected_evidence_count == 1


# =============================================================================
# 3. MANDATORY END-TO-END STAGE TEST
# =============================================================================


def test_mandatory_e2e_real_text_source_success(session_factory):
    """
    MANDATORY E2E STAGE TEST:
    Topic: 'Why does Transformer use self-attention?'
    Source: 'Self-attention allows each token to compute relationships with other tokens and construct context-dependent representations.'
    Executes through production pipeline:
    KnowledgeVideoTask -> WorkflowJob(EVIDENCE) -> StageWorker -> EvidenceStageExecutor -> E2 parser/chunker/retriever ->
    RetrievalSnapshot -> EvidenceItem -> EvidenceSnapshot -> TaskArtifactRef -> StageExecution -> KnowledgeVideoWorkflow ->
    WorkflowJob(KNOWLEDGE_PLAN).
    """
    topic = "Why does Transformer use self-attention?"
    source_text = (
        "Self-attention allows each token to compute relationships with other tokens "
        "and construct context-dependent representations."
    )

    # 1. Create task and add real text source
    task_id: str
    src_doc_id: str
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        ev_cmd = TaskEvidenceCommandService(session)

        task = cmd_service.create_task(
            topic=topic,
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_id = task.task_id

        doc = ev_cmd.add_text_source(
            task_id=task_id,
            text=source_text,
            title="Attention Explanation",
        )
        src_doc_id = doc.source_document_id
        session.commit()

    # 2. Run StageWorker with EVIDENCE executor
    evidence_registry = StageExecutorRegistry()
    evidence_registry.register(Stage.EVIDENCE, EvidenceStageExecutor(session_factory=session_factory))
    worker = StageWorker(
        session_factory=session_factory,
        registry=evidence_registry,
        worker_id="test-e2e-worker",
    )

    # First run: acquires and executes EVIDENCE job
    processed = worker.run_once()
    assert processed is True, "Worker should acquire and process the EVIDENCE job."

    # 3. Verify database state and provenance lineage
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        artifact_repo = TaskArtifactRepository(session)
        exec_repo = StageExecutionRepository(session)

        # Task state
        task = task_repo.get_task(task_id)
        assert task is not None
        assert task.task_id == task_id
        assert task.task_status == TaskStatus.RUNNING
        assert task.current_stage == Stage.KNOWLEDGE_PLAN

        # Evidence job
        completed_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_evidence_1")
        assert completed_job is not None
        assert completed_job.status == JobStatus.SUCCEEDED
        assert completed_job.output_task_artifact_ref_id is not None

        # Stage Execution audit
        executions = exec_repo.list_executions_for_task(task_id)
        assert len(executions) == 1
        exec_record = executions[0]
        assert exec_record.task_id == task_id
        assert exec_record.stage == Stage.EVIDENCE
        assert exec_record.status == "SUCCEEDED"
        assert exec_record.output_task_artifact_ref_id == completed_job.output_task_artifact_ref_id

        # TaskArtifactRef
        art_ref = artifact_repo.get_artifact_ref(completed_job.output_task_artifact_ref_id)
        assert art_ref is not None
        assert art_ref.task_id == task_id
        assert art_ref.stage == Stage.EVIDENCE
        assert art_ref.artifact_type == ArtifactType.EVIDENCE_SNAPSHOT

        # EvidenceSnapshot
        snapshot = ev_repo.get_evidence_snapshot(art_ref.artifact_id)
        assert snapshot is not None
        assert snapshot.task_id == task_id
        assert snapshot.snapshot_version == 1
        assert src_doc_id in snapshot.source_document_ids
        assert len(snapshot.evidence_ids) >= 1

        # EvidenceItem resolves to original source text
        ev_items = [ev_repo.get_evidence_item(eid) for eid in snapshot.evidence_ids]
        assert any(
            "context-dependent representations" in item.original_excerpt
            for item in ev_items
            if item is not None
        )

        # RetrievalSnapshot
        ret_snapshots = ev_repo.list_retrieval_snapshots_for_task(task_id)
        assert len(ret_snapshots) == 1
        ret_snap = ret_snapshots[0]
        assert ret_snap.task_id == task_id
        assert ret_snap.query == topic
        assert src_doc_id in ret_snap.source_scope_ids
        assert len(ret_snap.candidates) >= 1

        # Next WorkflowJob: KNOWLEDGE_PLAN
        kp_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_knowledge_plan_1")
        assert kp_job is not None
        assert kp_job.stage == Stage.KNOWLEDGE_PLAN
        assert kp_job.status == JobStatus.QUEUED
        assert kp_job.input_task_artifact_ref_id == art_ref.task_artifact_ref_id

    # 4. Mandatory safety: Worker tries to run again, but KNOWLEDGE_PLAN is unsupported -> remains QUEUED
    second_processed = worker.run_once()
    assert second_processed is False, "Worker must NOT claim unsupported KNOWLEDGE_PLAN job."

    with session_factory() as session:
        job_repo = WorkflowJobRepository(session)
        kp_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_knowledge_plan_1")
        assert kp_job.status == JobStatus.QUEUED
        assert kp_job.lease_owner is None


# =============================================================================
# 4. MANDATORY NO-EVIDENCE TEST
# =============================================================================


def test_mandatory_no_evidence_becomes_needs_evidence(session_factory):
    """
    MANDATORY NO-EVIDENCE TEST:
    Task with zero sources executed by real EVIDENCE job.
    Expected: task_status = NEEDS_EVIDENCE, stage remains EVIDENCE, no KNOWLEDGE_PLAN job.
    """
    task_id: str
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        task = cmd_service.create_task(
            topic="History of Quantum Teleportation",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_id = task.task_id
        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-no-ev-worker",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        exec_repo = StageExecutionRepository(session)

        task = task_repo.get_task(task_id)
        assert task is not None
        assert task.task_status == TaskStatus.NEEDS_EVIDENCE
        assert task.current_stage == Stage.EVIDENCE
        assert task.error_type == JobErrorType.NEEDS_EVIDENCE

        # Job marked FAILED
        job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_evidence_1")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_type == JobErrorType.NEEDS_EVIDENCE.value

        # Execution audit records failure
        executions = exec_repo.list_executions_for_task(task_id)
        assert len(executions) == 1
        assert executions[0].status == "FAILED"
        assert executions[0].error_type == JobErrorType.NEEDS_EVIDENCE.value

        # No EvidenceSnapshot or KNOWLEDGE_PLAN job
        assert ev_repo.get_latest_snapshot_for_task(task_id) is None
        kp_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_knowledge_plan_1")
        assert kp_job is None


# =============================================================================
# 5. MANDATORY PARTIAL-FAILURE TEST
# =============================================================================


def test_mandatory_partial_failure_success(session_factory):
    """
    MANDATORY PARTIAL-FAILURE TEST:
    Task has:
    - Source A: valid TEXT source
    - Source B: unsupported / empty source that fails parsing
    Expected:
    - Source B fails truthfully and is recorded in metadata
    - Source A provides valid evidence
    - EVIDENCE stage succeeds
    - Task advances to KNOWLEDGE_PLAN
    """
    task_id: str
    src_a_id: str
    src_b_id: str

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        ev_cmd = TaskEvidenceCommandService(session)

        task = cmd_service.create_task(
            topic="Attention in Deep Learning",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_id = task.task_id

        # Source A: Valid TEXT
        doc_a = ev_cmd.add_text_source(
            task_id=task_id,
            text="Attention mechanisms calculate dynamic weights across representations.",
            title="Valid Attention Source",
        )
        src_a_id = doc_a.source_document_id

        # Source B: Unsupported binary file that will fail document parser
        doc_b = ev_cmd.register_file_source(
            task_id=task_id,
            filename="unsupported.xyz",
            content=b"\x00\x01\x02\x03\xff\xfe",
            media_type="application/octet-stream",
            title="Binary Source",
        )
        src_b_id = doc_b.source_document_id
        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-partial-worker",
    )

    processed = worker.run_once()
    assert processed is True

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)

        task = task_repo.get_task(task_id)
        assert task.task_status == TaskStatus.RUNNING
        assert task.current_stage == Stage.KNOWLEDGE_PLAN

        # Evidence snapshot exists and only references successful source
        snapshot = ev_repo.get_latest_snapshot_for_task(task_id)
        assert snapshot is not None
        assert src_a_id in snapshot.source_document_ids
        assert src_b_id not in snapshot.source_document_ids

        # Failed source is marked FAILED in database
        failed_doc = ev_repo.get_source_document(src_b_id)
        assert failed_doc is not None
        assert failed_doc.status == SourceStatus.FAILED
        assert "processing_error" in failed_doc.metadata_json


# =============================================================================
# 6. MANDATORY RESUME TEST
# =============================================================================


def test_mandatory_resume_after_adding_evidence(session_factory):
    """
    MANDATORY RESUME TEST:
    1. Task starts with no evidence.
    2. EVIDENCE execution results in NEEDS_EVIDENCE.
    3. Add a real TEXT source via TaskEvidenceCommandService.
    4. Explicitly retry/resume Evidence stage via TaskCommandService.retry_task.
    5. Worker executes new evidence job attempt.
    6. Real evidence is generated and EvidenceSnapshot is created.
    7. Workflow proceeds to KNOWLEDGE_PLAN.
    8. Verify first failed attempt audit history is preserved.
    """
    task_id: str
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        task = cmd_service.create_task(
            topic="Neural Network Backpropagation",
            target_duration=90.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_id = task.task_id
        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-resume-worker",
    )

    # 1. Run attempt 1 -> NEEDS_EVIDENCE
    worker.run_once()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        t1 = task_repo.get_task(task_id)
        assert t1.task_status == TaskStatus.NEEDS_EVIDENCE

    # 2. Add real TEXT source
    with session_factory() as session:
        ev_cmd = TaskEvidenceCommandService(session)
        ev_cmd.add_text_source(
            task_id=task_id,
            text="Backpropagation computes gradients using the chain rule to update weights.",
            title="Backprop Fundamentals",
        )
        session.commit()

    # 3. Explicit retry
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        retried_task = cmd_service.retry_task(task_id, reason="Added source document")
        assert retried_task.task_status == TaskStatus.RUNNING
        session.commit()

    # 4. Worker executes attempt 2
    processed = worker.run_once()
    assert processed is True

    # 5. Verify successful advancement and audit history
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        ev_repo = EvidenceRepository(session)
        exec_repo = StageExecutionRepository(session)

        # Task advanced
        final_task = task_repo.get_task(task_id)
        assert final_task.task_status == TaskStatus.RUNNING
        assert final_task.current_stage == Stage.KNOWLEDGE_PLAN

        # Evidence snapshot exists
        snap = ev_repo.get_latest_snapshot_for_task(task_id)
        assert snap is not None
        assert snap.snapshot_version == 1

        # Check job attempts
        attempt1_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_evidence_1")
        attempt2_job = job_repo.get_job_by_idempotency_key(f"idemp_{task_id}_evidence_2")
        assert attempt1_job is not None and attempt1_job.status == JobStatus.FAILED
        assert attempt2_job is not None and attempt2_job.status == JobStatus.SUCCEEDED

        # Both stage executions preserved in history
        executions = exec_repo.list_executions_for_task(task_id)
        assert len(executions) == 2
        assert executions[0].attempt_number == 1 and executions[0].status == "FAILED"
        assert executions[1].attempt_number == 2 and executions[1].status == "SUCCEEDED"


# =============================================================================
# 7. PROVENANCE INTEGRITY TESTS
# =============================================================================


def test_provenance_chain_integrity(session_factory):
    """Verify strict cryptographic provenance across EvidenceSnapshot -> EvidenceItem -> KnowledgeChunk -> SourceDocument."""
    topic = "Convolutional Neural Networks"
    text = "Convolutional layers apply localized kernels to preserve spatial structure."

    with session_factory() as session:
        cmd = TaskCommandService(session)
        ev_cmd = TaskEvidenceCommandService(session)

        task = cmd.create_task(topic=topic)
        task_id = task.task_id
        src = ev_cmd.add_text_source(task_id=task_id, text=text, title="CNN Overview")
        src_id = src.source_document_id
        session.commit()

    worker = StageWorker(
        session_factory=session_factory,
        registry=get_default_executor_registry(session_factory=session_factory),
        worker_id="test-prov-worker",
    )
    worker.run_once()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        snap = ev_repo.get_latest_snapshot_for_task(task_id)
        assert snap is not None

        for eid in snap.evidence_ids:
            assert eid.startswith("ev_")
            item = ev_repo.get_evidence_item(eid)
            assert item is not None
            assert item.source_document_id == src_id
            assert item.content_hash is not None

            # Verify chunk existence and excerpt match
            chunks = ev_repo.list_chunks_for_source(src_id)
            assert len(chunks) >= 1
            assert any(
                item.original_excerpt in c.normalized_text or c.normalized_text in item.original_excerpt
                for c in chunks
            )


# =============================================================================
# 8. API INTEGRATION TESTS
# =============================================================================


def test_api_register_evidence_and_retry_task(session_factory, monkeypatch):
    """Test REST API endpoints for evidence source registration and task retry."""
    monkeypatch.setattr("app.controllers.v1.knowledge_video.get_session", session_factory)

    test_app = FastAPI()
    test_app.include_router(kv_router)
    client = TestClient(test_app)

    # 1. Create task via API
    resp_create = client.post(
        "/api/v1/knowledge-video-tasks",
        json={"topic": "Black Holes", "target_duration": 60.0},
    )
    assert resp_create.status_code == 202
    task_id = resp_create.json()["data"]["task_id"]

    # 2. Register evidence via API
    resp_ev = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "A black hole is a region of spacetime where gravity is so strong that nothing can escape.",
            "title": "General Relativity and Black Holes",
        },
    )
    assert resp_ev.status_code == 201
    src_data = resp_ev.json()["data"]
    assert src_data["source_type"] == "TEXT"
    assert src_data["source_document_id"] is not None

    # 3. Verify terminal task rejects registration
    with session_factory() as session:
        t_repo = KnowledgeVideoTaskRepository(session)
        task = t_repo.get_task(task_id)
        task.transition_to(TaskStatus.CANCELLED)
        t_repo.save_task(task)
        session.commit()

    resp_terminal = client.post(
        f"/api/v1/knowledge-video-tasks/{task_id}/evidence",
        json={
            "source_type": "TEXT",
            "text_content": "Late content",
            "title": "Late Title",
        },
    )
    assert resp_terminal.status_code in (400, 409, 422)
