from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.evidence_service import (
    EvidenceResolutionService,
    TaskEvidenceCommandService,
)
from app.application.knowledge_base_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseServiceError,
)
from app.application.stage_executor_registry import get_default_executor_registry
from app.domain.evidence import (
    ClaimType,
    EvidenceItem,
    EvidenceLocator,
    EvidenceNotFoundError,
    EvidenceRole,
    KnowledgeClaim,
    SourceDocument,
    SourceStatus,
    SourceType,
    UnsupportedSourceTypeError,
    VerificationStatus,
)
from app.domain.knowledge_base import KnowledgeBase, KnowledgeBaseStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.shot import ShotRevision
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import (
    ArtifactType,
    JobStatus,
    Stage,
    TaskStatus,
    TerminalStateImmutableError,
    WorkflowPolicyType,
)
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeBaseRepository,
    KnowledgeVideoTaskRepository,
    WorkflowJobRepository,
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


def test_add_text_source(session_factory):
    """Verify adding inline text source creates READY SourceDocument and task association."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Photosynthesis", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        session.commit()

    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        doc = cmd_service.add_text_source(
            task_id=task_id,
            text="Chlorophyll absorbs blue and red light while reflecting green light.",
            title="Chlorophyll Notes",
            author="Botanist A",
        )
        session.commit()

        assert doc.source_type == SourceType.TEXT
        assert doc.status == SourceStatus.READY
        assert doc.title == "Chlorophyll Notes"
        assert "Chlorophyll" in (doc.content_snapshot or "")

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        sources = ev_repo.list_sources_for_task(task_id)
        assert len(sources) == 1
        assert sources[0].source_document_id == doc.source_document_id


def test_register_file_source(session_factory, tmp_path, monkeypatch):
    """Verify registering a file source stores bytes to storage and creates REGISTERED SourceDocument."""
    task_id = f"task_{uuid4().hex[:8]}"
    storage_root = tmp_path / "storage"
    monkeypatch.setattr("app.application.evidence_service.storage_dir", lambda sub="", create=False: str(storage_root / sub) if sub else str(storage_root))

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Quantum Physics", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        session.commit()

    file_bytes = b"Raw binary contents of physics whitepaper."
    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        doc = cmd_service.register_file_source(
            task_id=task_id,
            filename="quantum_paper.pdf",
            content=file_bytes,
            media_type="application/pdf",
            title="Quantum Paper",
        )
        session.commit()

        assert doc.source_type == SourceType.FILE
        assert doc.status == SourceStatus.REGISTERED
        assert doc.content_snapshot is None  # Unparsed raw file in E1
        assert doc.metadata_json["original_filename"] == "quantum_paper.pdf"

    # Verify physical file existence
    stored_files = list((storage_root / "evidence_sources").glob("*quantum_paper.pdf"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == file_bytes


def test_register_url_source(session_factory):
    """Verify registering a URL source validates schema and creates REGISTERED SourceDocument without fetching."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Cosmology", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        session.commit()

    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)

        # Invalid URL scheme
        with pytest.raises(ValueError, match="Invalid URL"):
            cmd_service.register_url_source(task_id=task_id, url="ftp://example.com/data")

        # Valid URL
        doc = cmd_service.register_url_source(
            task_id=task_id,
            url="https://en.wikipedia.org/wiki/Big_Bang",
            title="Big Bang Article",
        )
        session.commit()

        assert doc.source_type == SourceType.URL
        assert doc.status == SourceStatus.REGISTERED
        assert doc.content_snapshot is None  # Unfetched in E1
        assert doc.source_locator == "https://en.wikipedia.org/wiki/Big_Bang"


def test_register_knowledge_base_source_success(session_factory):
    """Verify registering a valid active KB attaches it to the task."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="KB Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Deep Learning KB")
        kb_repo.save_knowledge_base(kb)
        session.commit()
        kb_id = kb.knowledge_base_id

    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        cmd_service.register_knowledge_base_source(task_id=task_id, kb_id=kb_id)
        session.commit()

    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        attached = kb_repo.list_kbs_for_task(task_id)
        assert len(attached) == 1
        assert attached[0].knowledge_base_id == kb_id


def test_register_knowledge_base_source_not_found(session_factory):
    """Verify registering a non-existent KB raises KnowledgeBaseNotFoundError."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="KB Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        session.commit()

    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        with pytest.raises(KnowledgeBaseNotFoundError, match="not found"):
            cmd_service.register_knowledge_base_source(task_id=task_id, kb_id="kb_missing_123")


def test_terminal_task_cannot_register_evidence(session_factory):
    """Verify registering evidence for a terminal task raises TerminalStateImmutableError."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task = KnowledgeVideoTask.create(
            task_id=task_id, topic="Done Task", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO
        )
        task.task_status = TaskStatus.COMPLETED
        task_repo.save_task(task)
        session.commit()

    with session_factory() as session:
        cmd_service = TaskEvidenceCommandService(session)
        with pytest.raises(TerminalStateImmutableError):
            cmd_service.add_text_source(task_id=task_id, text="Some late source")


def test_evidence_resolution_service(session_factory):
    """Verify EvidenceResolutionService resolves single and multiple evidence IDs."""
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        doc = SourceDocument.create_text("The Earth revolves around the Sun once every ~365.25 days.", title="Astronomy")
        ev_repo.save_source_document(doc)

        item1 = EvidenceItem.create(source_document=doc, original_excerpt="Earth revolves around the Sun", evidence_role=EvidenceRole.FACTUAL_SUPPORT)
        item2 = EvidenceItem.create(source_document=doc, original_excerpt="once every ~365.25 days", evidence_role=EvidenceRole.FACTUAL_SUPPORT)
        ev_repo.save_evidence_item(item1)
        ev_repo.save_evidence_item(item2)
        session.commit()

    with session_factory() as session:
        resolver = EvidenceResolutionService(session)

        # Single resolution
        res_item, res_doc = resolver.resolve_evidence(item1.evidence_id)
        assert res_item.evidence_id == item1.evidence_id
        assert res_doc.source_document_id == doc.source_document_id

        # Batch resolution
        resolved_map = resolver.resolve_evidence_refs([item1.evidence_id, item2.evidence_id])
        assert len(resolved_map) == 2
        assert item1.evidence_id in resolved_map
        assert item2.evidence_id in resolved_map

        # Broken ID resolution failure
        with pytest.raises(EvidenceNotFoundError, match="could not be resolved"):
            resolver.resolve_evidence("ev_nonexistent_id_12345678")

        # Broken ID in batch resolution
        with pytest.raises(EvidenceNotFoundError, match="Failed to resolve 1 evidence reference"):
            resolver.resolve_evidence_refs([item1.evidence_id, "ev_nonexistent_id_12345678"])


def test_phase_1_to_7_evidence_refs_compatibility(session_factory):
    """Verify Phase 1-7 domain models (ShotRevision) using evidence_refs link seamlessly to E1 resolution."""
    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        doc = SourceDocument.create_text(
            "Mitochondria generate most of the chemical energy needed to power the cell's biochemical reactions.",
            title="Cell Biology",
        )
        ev_repo.save_source_document(doc)

        item = EvidenceItem.create(
            source_document=doc,
            original_excerpt="Mitochondria generate most of the chemical energy",
            locator=EvidenceLocator(paragraph=1),
            evidence_role=EvidenceRole.FACTUAL_SUPPORT,
        )
        ev_repo.save_evidence_item(item)
        session.commit()

    from app.domain.content_plan import ContentBeat
    from app.domain.enums import BeatType, VisualType

    # Create Phase 1-7 ContentBeat citing this evidence_id
    beat = ContentBeat(
        beat_id="beat_001",
        beat_lineage_id="beat_lineage_001",
        beat_type=BeatType.KNOWLEDGE,
        order=1,
        intent="Explain cellular energy generation",
        target_duration=5.0,
        importance=0.9,
        evidence_refs=(item.evidence_id,),
    )

    # Create Phase 1-7 ShotRevision citing this evidence_id
    shot_rev = ShotRevision(
        shot_revision_id="shot_rev_001",
        shot_id="shot_001",
        revision_number=1,
        beat_lineage_id=beat.beat_lineage_id,
        created_from_beat_instance_id=beat.beat_id,
        narration="Inside each cell, mitochondria produce ATP energy to sustain life.",
        target_duration=5.0,
        visual_goal="Show glowing mitochondria inside cell structure",
        visual_type=VisualType.AI_IMAGE,
        scene_description="Macro view of a living cell with glowing mitochondria",
        generation_prompt="cinematic 3d animation of cell mitochondria glowing with energy",
        camera_movement="slow push in",
        evidence_refs=(item.evidence_id,),
    )

    with session_factory() as session:
        resolver = EvidenceResolutionService(session)

        # 1. Resolve from ContentBeat.evidence_refs
        beat_resolved = resolver.resolve_evidence_refs(beat.evidence_refs)
        assert item.evidence_id in beat_resolved
        b_item, b_doc = beat_resolved[item.evidence_id]
        assert b_item.original_excerpt == "Mitochondria generate most of the chemical energy"
        assert b_doc.title == "Cell Biology"

        # 2. Resolve from ShotRevision.evidence_refs
        shot_resolved = resolver.resolve_evidence_refs(shot_rev.evidence_refs)
        assert item.evidence_id in shot_resolved
        s_item, s_doc = shot_resolved[item.evidence_id]
        assert s_item.original_excerpt == "Mitochondria generate most of the chemical energy"
        assert s_doc.title == "Cell Biology"


def test_create_task_evidence_snapshot_and_artifact_ref(session_factory):
    """Verify creating a snapshot aggregates all sources/items and generates TaskArtifactRef."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Neuroscience", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        cmd = TaskEvidenceCommandService(session)
        doc1 = cmd.add_text_source(task_id, "Neurons communicate via synapses.", title="Neuro 1")
        doc2 = cmd.add_text_source(task_id, "Action potentials travel along the axon.", title="Neuro 2")

        ev_repo = EvidenceRepository(session)
        item1 = EvidenceItem.create(source_document=doc1, original_excerpt="Neurons communicate via synapses.")
        item2 = EvidenceItem.create(source_document=doc2, original_excerpt="Action potentials travel along the axon.")
        ev_repo.save_evidence_item(item1)
        ev_repo.save_evidence_item(item2)

        claim = KnowledgeClaim.create(
            claim_type=ClaimType.FACT,
            claim_text="Neurons transmit electrochemical signals.",
            evidence_refs=[item1.evidence_id, item2.evidence_id],
            verification_status=VerificationStatus.GROUNDED,
        )

        snapshot, art_ref = cmd.create_task_evidence_snapshot(
            task_id=task_id,
            knowledge_claims=[claim],
        )
        session.commit()

        assert snapshot.task_id == task_id
        assert snapshot.snapshot_version == 1
        assert len(snapshot.source_document_ids) == 2
        assert len(snapshot.evidence_ids) == 2
        assert len(snapshot.knowledge_claim_ids) == 1

        assert art_ref.task_id == task_id
        assert art_ref.stage == Stage.EVIDENCE
        assert art_ref.artifact_type == ArtifactType.EVIDENCE_SNAPSHOT
        assert art_ref.artifact_id == snapshot.evidence_snapshot_id


def test_evidence_stage_executor_registration_in_production():
    """Verify that default registry registers EvidenceStageExecutor and DeliveryStageExecutor."""
    from app.application.evidence_stage_executor import EvidenceStageExecutor

    registry = get_default_executor_registry()
    assert registry.has_executor(Stage.EVIDENCE)
    assert isinstance(registry.get_executor(Stage.EVIDENCE), EvidenceStageExecutor)
    assert registry.has_executor(Stage.QUALITY_REVIEW)
    assert registry.has_executor(Stage.DELIVERY)


def test_batch_resolution_with_empty_list(session_factory):
    """Verify resolve_evidence_refs with empty sequence returns empty mapping."""
    with session_factory() as session:
        resolver = EvidenceResolutionService(session)
        assert resolver.resolve_evidence_refs([]) == {}
        assert resolver.resolve_evidence_refs(()) == {}


def test_task_evidence_lineage_end_to_end(session_factory, tmp_path, monkeypatch):
    """Comprehensive provenance test: Task -> Sources -> EvidenceItems -> Claims -> Snapshot -> ArtifactRef -> Resolution."""
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(
        "app.application.evidence_service.storage_dir",
        lambda sub="", create=False: str(storage_root / sub) if sub else str(storage_root),
    )

    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(
                task_id=task_id, topic="End-to-End Provenance", target_duration=120.0, workflow_policy=WorkflowPolicyType.REVIEW
            )
        )
        session.commit()

    with session_factory() as session:
        cmd = TaskEvidenceCommandService(session)
        # Register 3 heterogeneous sources
        doc_text = cmd.add_text_source(task_id, "Fermat's Last Theorem was proven by Andrew Wiles in 1994.", title="Fermat Note")
        doc_file = cmd.register_file_source(task_id, "wiles_proof.pdf", b"%PDF-proof", media_type="application/pdf", title="Wiles Proof PDF")
        doc_url = cmd.register_url_source(task_id, "https://example.com/fermat", title="Fermat History")

        # Add evidence items
        ev_repo = EvidenceRepository(session)
        item_text = EvidenceItem.create(doc_text, "proven by Andrew Wiles in 1994", locator=EvidenceLocator(paragraph=1))
        ev_repo.save_evidence_item(item_text)

        # Create claim
        claim = KnowledgeClaim.create(
            claim_type=ClaimType.FACT,
            claim_text="Andrew Wiles proved Fermat's Last Theorem in 1994.",
            evidence_refs=[item_text.evidence_id],
        )

        # Create snapshot
        snapshot, art_ref = cmd.create_task_evidence_snapshot(task_id, knowledge_claims=[claim])
        session.commit()

        assert snapshot.snapshot_version == 1
        assert len(snapshot.source_document_ids) == 3
        assert len(snapshot.evidence_ids) == 1
        assert len(snapshot.knowledge_claim_ids) == 1

    with session_factory() as session:
        # Full backward lineage resolution: resolve evidence_id back to doc_text
        resolver = EvidenceResolutionService(session)
        resolved_item, resolved_doc = resolver.resolve_evidence(item_text.evidence_id)
        assert resolved_item.evidence_id == item_text.evidence_id
        assert resolved_doc.source_document_id == doc_text.source_document_id
        assert resolved_doc.title == "Fermat Note"


