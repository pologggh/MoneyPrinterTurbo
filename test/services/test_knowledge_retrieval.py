from __future__ import annotations

from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_service import TaskEvidenceCommandService
from app.application.knowledge_retrieval_service import (
    KnowledgeProcessingService,
    KnowledgeRetrievalService,
)
from app.application.stage_executor_registry import get_default_executor_registry
from app.domain.evidence import (
    EvidenceDomainError,
    EvidenceItem,
    EvidenceLocator,
    RetrievalSnapshot,
    SourceDocument,
    SourceStatus,
    SourceType,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import Stage, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
)


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


def test_attention_vs_dns_ranking_accuracy(session_factory):
    """MANDATORY TEST 1: Attention text vs DNS text ranking accuracy.

    Document A: self-attention mechanism in transformers.
    Document B: DNS and IP address resolution.
    Query: 'transformer attention mechanism'.
    Expected: Document A ranked at top with highest BM25 score.
    """
    task_id = f"task_{uuid4().hex[:8]}"

    doc_a_text = (
        "The transformer architecture relies entirely on the self-attention mechanism "
        "to compute representations of its input and output without using sequence-aligned RNNs or convolution. "
        "Multi-head self-attention allows the model to jointly attend to information from different representation subspaces."
    )
    doc_b_text = (
        "The Domain Name System (DNS) is a hierarchical and decentralized naming system for computers, "
        "services, or other resources connected to the Internet or a private network. "
        "It translates more readily memorized domain names to the numerical IP addresses needed for locating computer services."
    )

    with session_factory() as session:
        # Create task
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="AI vs Networks", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )

        # Register sources
        cmd_service = TaskEvidenceCommandService(session)
        src_a = cmd_service.add_text_source(task_id, doc_a_text, title="Transformer Attention Paper")
        src_b = cmd_service.add_text_source(task_id, doc_b_text, title="DNS Networking RFC")
        session.commit()

        # Ingest and chunk documents
        proc_service = KnowledgeProcessingService(session)
        chunks_a = proc_service.process_source_document(src_a.source_document_id)
        chunks_b = proc_service.process_source_document(src_b.source_document_id)
        assert len(chunks_a) >= 1
        assert len(chunks_b) >= 1

        # Execute retrieval
        retrieval_service = KnowledgeRetrievalService(session)
        snapshot = retrieval_service.retrieve(
            task_id=task_id,
            query="transformer attention mechanism",
            top_k=5,
        )

        assert snapshot is not None
        assert len(snapshot.candidates) >= 1
        # Top candidate must be from Document A
        top_cand = snapshot.candidates[0]
        assert top_cand.source_document_id == src_a.source_document_id
        assert "attention" in top_cand.excerpt.lower()
        assert top_cand.rank == 1
        assert top_cand.score > 0

        # Selected evidence items were created
        assert len(snapshot.selected_evidence_ids) >= 1


def test_task_scoping_strict_isolation(session_factory):
    """MANDATORY TEST 2: Strict isolation between Task A and Task B.

    Task A has Document A (Neural network backpropagation).
    Task B has Document B (Quantum entanglement).
    Query for 'neural network' on Task B must NOT return Document A.
    Attempting to force Document A scope on Task B must raise EvidenceDomainError.
    """
    task_a_id = f"task_{uuid4().hex[:8]}"
    task_b_id = f"task_{uuid4().hex[:8]}"

    doc_a_text = "Neural network backpropagation calculates the gradient of the loss function with respect to weights."
    doc_b_text = "Quantum entanglement is a physical phenomenon that occurs when a group of particles generate correlated states."

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_a_id, topic="AI", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_b_id, topic="Physics", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))

        cmd_service = TaskEvidenceCommandService(session)
        src_a = cmd_service.add_text_source(task_a_id, doc_a_text, title="Backprop")
        src_b = cmd_service.add_text_source(task_b_id, doc_b_text, title="Entanglement")
        session.commit()

        proc_service = KnowledgeProcessingService(session)
        proc_service.process_source_document(src_a.source_document_id)
        proc_service.process_source_document(src_b.source_document_id)

        retrieval_service = KnowledgeRetrievalService(session)

        # 1. Query on Task B for 'neural network' -> must yield 0 results
        snap_b = retrieval_service.retrieve(task_id=task_b_id, query="neural network backpropagation")
        assert len(snap_b.candidates) == 0
        for cand in snap_b.candidates:
            assert cand.source_document_id != src_a.source_document_id

        # 2. Attempt to explicitly query Document A on Task B -> must raise EvidenceDomainError
        with pytest.raises(EvidenceDomainError) as exc_info:
            retrieval_service.retrieve(
                task_id=task_b_id,
                query="neural network",
                source_scope_ids=[src_a.source_document_id],
            )
        assert "not associated with task" in str(exc_info.value)


def test_historical_retrieval_snapshot_immutability(session_factory):
    """MANDATORY TEST 3: Historical snapshot lineage and immutability.

    Retrieval R1 on Source V1 produces fingerprint F1.
    Source updated to V2 and re-chunked; retrieval R2 produces fingerprint F2.
    Assert R1 remains completely preserved and distinct from R2.
    """
    task_id = f"task_{uuid4().hex[:8]}"
    initial_text = "Early research into artificial intelligence focused on symbolic reasoning and expert systems."

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="AI History", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))

        cmd_service = TaskEvidenceCommandService(session)
        src = cmd_service.add_text_source(task_id, initial_text, title="AI History")
        session.commit()

        proc_service = KnowledgeProcessingService(session)
        proc_service.process_source_document(src.source_document_id)

        retrieval_service = KnowledgeRetrievalService(session)
        r1 = retrieval_service.retrieve(task_id=task_id, query="symbolic reasoning expert systems")
        f1 = r1.content_fingerprint
        r1_id = r1.retrieval_snapshot_id

    # Now simulate content update: create a new version of source content for the task
    second_text = (
        "Modern artificial intelligence shifted toward deep learning, neural networks, "
        "and large language models trained on massive internet datasets."
    )
    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        src2 = cmd_service.add_text_source(task_id, second_text, title="Modern AI")
        session.commit()

        proc_service = KnowledgeProcessingService(session)
        proc_service.process_source_document(src2.source_document_id)

        retrieval_service = KnowledgeRetrievalService(session)
        r2 = retrieval_service.retrieve(task_id=task_id, query="deep learning modern artificial intelligence")
        f2 = r2.content_fingerprint
        r2_id = r2.retrieval_snapshot_id

        # Verify fingerprints differ
        assert f1 != f2

        # Verify historical snapshot R1 is still unchanged and retrievable
        ev_repo = EvidenceRepository(session)
        loaded_r1 = ev_repo.get_retrieval_snapshot(r1_id)
        assert loaded_r1 is not None
        assert loaded_r1.content_fingerprint == f1
        assert loaded_r1.query == "symbolic reasoning expert systems"

        loaded_r2 = ev_repo.get_retrieval_snapshot(r2_id)
        assert loaded_r2 is not None
        assert loaded_r2.content_fingerprint == f2
        assert loaded_r2.query == "deep learning modern artificial intelligence"


def test_stable_evidence_item_reuse_across_queries(session_factory):
    """Verify that multiple retrievals selecting the same excerpt reuse the exact same EvidenceItem."""
    task_id = f"task_{uuid4().hex[:8]}"
    text = "The quick brown fox jumps over the lazy dog."

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(KnowledgeVideoTask.create(task_id=task_id, topic="Fox", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO))

        cmd_service = TaskEvidenceCommandService(session)
        src = cmd_service.add_text_source(task_id, text, title="Fox Story")
        session.commit()

        proc_service = KnowledgeProcessingService(session)
        proc_service.process_source_document(src.source_document_id)

        retrieval_service = KnowledgeRetrievalService(session)
        snap1 = retrieval_service.retrieve(task_id=task_id, query="quick brown fox")
        snap2 = retrieval_service.retrieve(task_id=task_id, query="lazy dog")

        assert len(snap1.selected_evidence_ids) == 1
        assert len(snap2.selected_evidence_ids) == 1
        # Both queries matched the same chunk and reused the same EvidenceItem ID
        assert snap1.selected_evidence_ids[0] == snap2.selected_evidence_ids[0]

        ev_repo = EvidenceRepository(session)
        all_items = ev_repo.list_evidence_items_for_source(src.source_document_id)
        # Exactly one EvidenceItem exists in DB, no duplicates
        assert len(all_items) == 1


def test_evidence_stage_executor_is_registered_in_stage_e3():
    """Verify that Stage.EVIDENCE executor is registered in Stage E3 production registry, while subsequent stages remain unregistered."""
    from app.application.evidence_stage_executor import EvidenceStageExecutor

    registry = get_default_executor_registry()
    executor = registry.get_executor(Stage.EVIDENCE)
    assert executor is not None
    assert isinstance(executor, EvidenceStageExecutor)
    assert registry.get_executor(Stage.STORYBOARD) is None
