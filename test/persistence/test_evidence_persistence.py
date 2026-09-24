from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.evidence import (
    ClaimType,
    EvidenceItem,
    EvidenceLocator,
    EvidenceRole,
    EvidenceSnapshot,
    KnowledgeClaim,
    SourceDocument,
    SourceStatus,
    SourceType,
    VerificationStatus,
)
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.task_artifact import TaskArtifactRef
from app.domain.workflow_state import ArtifactType, Stage, WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    TaskArtifactRepository,
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


def test_evidence_repository_source_documents(session_factory):
    """Verify saving and retrieving SourceDocuments of different types."""
    doc_text = SourceDocument.create_text(
        text="Einstein published the theory of general relativity in 1915.",
        title="General Relativity Note",
    )
    doc_file = SourceDocument.register_file(
        filename="relativity.pdf",
        file_bytes=b"%PDF-1.4 dummy",
        media_type="application/pdf",
        storage_path="storage/evidence_sources/relativity.pdf",
    )
    doc_url = SourceDocument.register_url(
        url="https://example.com/relativity",
        title="Relativity Reference",
    )

    with session_factory() as session:
        repo = EvidenceRepository(session)
        repo.save_source_document(doc_text)
        repo.save_source_document(doc_file)
        repo.save_source_document(doc_url)
        session.commit()

    with session_factory() as session:
        repo = EvidenceRepository(session)

        loaded_text = repo.get_source_document(doc_text.source_document_id)
        assert loaded_text is not None
        assert loaded_text.source_type == SourceType.TEXT
        assert loaded_text.status == SourceStatus.READY
        assert loaded_text.content_snapshot == doc_text.content_snapshot
        assert loaded_text.content_hash == doc_text.content_hash
        assert loaded_text.source_fingerprint == doc_text.source_fingerprint

        loaded_file = repo.get_source_document(doc_file.source_document_id)
        assert loaded_file is not None
        assert loaded_file.source_type == SourceType.FILE
        assert loaded_file.status == SourceStatus.REGISTERED
        assert loaded_file.content_snapshot is None
        assert loaded_file.media_type == "application/pdf"

        loaded_url = repo.get_source_document(doc_url.source_document_id)
        assert loaded_url is not None
        assert loaded_url.source_type == SourceType.URL
        assert loaded_url.status == SourceStatus.REGISTERED


def test_evidence_repository_task_sources_and_scoping(session_factory):
    """Verify task source associations and strict isolation between tasks."""
    task_a_id = f"task_{uuid4().hex[:8]}"
    task_b_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_a_id, topic="Task A", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_b_id, topic="Task B", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )

        ev_repo = EvidenceRepository(session)
        doc_a = SourceDocument.create_text("Document exclusively for Task A", title="Doc A")
        doc_b = SourceDocument.create_text("Document exclusively for Task B", title="Doc B")
        doc_shared = SourceDocument.create_text("Document shared across tasks", title="Doc Shared")

        ev_repo.save_source_document(doc_a)
        ev_repo.save_source_document(doc_b)
        ev_repo.save_source_document(doc_shared)

        ev_repo.associate_task_source(task_a_id, doc_a.source_document_id, role="PRIMARY")
        ev_repo.associate_task_source(task_a_id, doc_shared.source_document_id, role="SUPPLEMENTARY")
        ev_repo.associate_task_source(task_b_id, doc_b.source_document_id, role="PRIMARY")
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)

        sources_a = ev_repo.list_sources_for_task(task_a_id)
        assert len(sources_a) == 2
        a_ids = {s.source_document_id for s in sources_a}
        assert doc_a.source_document_id in a_ids
        assert doc_shared.source_document_id in a_ids
        assert doc_b.source_document_id not in a_ids

        sources_b = ev_repo.list_sources_for_task(task_b_id)
        assert len(sources_b) == 1
        assert sources_b[0].source_document_id == doc_b.source_document_id


def test_evidence_repository_evidence_items(session_factory):
    """Verify CRUD and querying for EvidenceItem."""
    doc = SourceDocument.create_text("Water is composed of hydrogen and oxygen (H2O).", title="Chemistry 101")

    with session_factory() as session:
        repo = EvidenceRepository(session)
        repo.save_source_document(doc)

        item = EvidenceItem.create(
            source_document=doc,
            original_excerpt="Water is composed of hydrogen and oxygen (H2O).",
            locator=EvidenceLocator(paragraph=1),
            evidence_role=EvidenceRole.FACTUAL_SUPPORT,
            confidence=0.98,
        )
        repo.save_evidence_item(item)
        session.commit()

    with session_factory() as session:
        repo = EvidenceRepository(session)
        loaded_item = repo.get_evidence_item(item.evidence_id)
        assert loaded_item is not None
        assert loaded_item.evidence_id == item.evidence_id
        assert loaded_item.source_document_id == doc.source_document_id
        assert loaded_item.confidence == 0.98
        assert loaded_item.locator == {"paragraph": 1}

        items = repo.list_evidence_items_for_source(doc.source_document_id)
        assert len(items) == 1
        assert items[0].evidence_id == item.evidence_id


def test_evidence_repository_knowledge_claims(session_factory):
    """Verify CRUD and querying for KnowledgeClaim."""
    claim = KnowledgeClaim.create(
        claim_type=ClaimType.FACT,
        claim_text="The speed of sound in air at 20°C is ~343 m/s.",
        evidence_refs=["ev_sound_343"],
        verification_status=VerificationStatus.GROUNDED,
    )

    with session_factory() as session:
        repo = EvidenceRepository(session)
        repo.save_knowledge_claim(claim)
        session.commit()

    with session_factory() as session:
        repo = EvidenceRepository(session)
        loaded = repo.get_knowledge_claim(claim.knowledge_claim_id)
        assert loaded is not None
        assert loaded.claim_type == ClaimType.FACT
        assert loaded.claim_text == claim.claim_text
        assert loaded.evidence_refs == ("ev_sound_343",)
        assert loaded.verification_status == VerificationStatus.GROUNDED


def test_evidence_repository_evidence_snapshots(session_factory):
    """Verify saving, versioning, and latest lookup for EvidenceSnapshot."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Snapshot Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )

        ev_repo = EvidenceRepository(session)
        snap1 = EvidenceSnapshot.create(
            task_id=task_id,
            source_document_ids=["src_1"],
            evidence_ids=["ev_1"],
            snapshot_version=1,
        )
        snap2 = EvidenceSnapshot.create(
            task_id=task_id,
            source_document_ids=["src_1", "src_2"],
            evidence_ids=["ev_1", "ev_2"],
            snapshot_version=2,
        )
        ev_repo.save_evidence_snapshot(snap1)
        ev_repo.save_evidence_snapshot(snap2)
        session.commit()

    with session_factory() as session:
        ev_repo = EvidenceRepository(session)
        latest = ev_repo.get_latest_snapshot_for_task(task_id)
        assert latest is not None
        assert latest.snapshot_version == 2
        assert latest.evidence_snapshot_id == snap2.evidence_snapshot_id
        assert latest.source_document_ids == ("src_1", "src_2")


def test_task_artifact_ref_linkage_with_evidence_snapshot(session_factory):
    """Verify TaskArtifactRef links task to EvidenceSnapshot using ArtifactType.EVIDENCE_SNAPSHOT."""
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        task_repo.save_task(
            KnowledgeVideoTask.create(task_id=task_id, topic="Artifact Ref Test", target_duration=60.0, workflow_policy=WorkflowPolicyType.AUTO)
        )

        ev_repo = EvidenceRepository(session)
        snapshot = EvidenceSnapshot.create(
            task_id=task_id,
            source_document_ids=["src_alpha"],
            evidence_ids=["ev_alpha"],
            snapshot_version=1,
        )
        ev_repo.save_evidence_snapshot(snapshot)

        art_repo = TaskArtifactRepository(session)
        art_ref = TaskArtifactRef.create(
            task_id=task_id,
            stage=Stage.EVIDENCE,
            artifact_type=ArtifactType.EVIDENCE_SNAPSHOT,
            artifact_id=snapshot.evidence_snapshot_id,
            artifact_version="1",
            metadata_json={"fingerprint": snapshot.content_fingerprint},
        )
        art_repo.save_artifact_ref(art_ref)
        session.commit()

    with session_factory() as session:
        art_repo = TaskArtifactRepository(session)
        ev_repo = EvidenceRepository(session)

        latest_ref = art_repo.get_latest_artifact_ref(task_id, stage=Stage.EVIDENCE, artifact_type=ArtifactType.EVIDENCE_SNAPSHOT)
        assert latest_ref is not None
        assert latest_ref.artifact_id == snapshot.evidence_snapshot_id

        loaded_snapshot = ev_repo.get_evidence_snapshot(latest_ref.artifact_id)
        assert loaded_snapshot is not None
        assert loaded_snapshot.content_fingerprint == latest_ref.metadata_json["fingerprint"]


def test_alembic_migration_0016_upgrade_downgrade(tmp_path, monkeypatch):
    """Verify Alembic migration 0016 upgrades from 0015, downgrades, and re-upgrades cleanly."""
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "test_migration_0016.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    project_root = Path(__file__).resolve().parent.parent.parent

    alembic_ini_path = project_root / "alembic.ini"
    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(project_root / "migrations"))

    # Upgrade to 0015 first
    command.upgrade(cfg, "0015_stage_worker_and_task_artifacts")

    # Upgrade to 0016
    command.upgrade(cfg, "0016_evidence_provenance_foundation")

    # Downgrade back to 0015
    command.downgrade(cfg, "0015_stage_worker_and_task_artifacts")

    # Re-upgrade to 0016
    command.upgrade(cfg, "0016_evidence_provenance_foundation")
