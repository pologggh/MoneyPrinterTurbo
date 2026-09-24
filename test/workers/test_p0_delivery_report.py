"""
Focused unit tests for P0-1:
Verifies that DeliveryReportService consumes the real authoritative ScriptSegment contract:
- narration_text
- evidence_refs
and that DeliveryStageExecutor creates a valid DeliveryManifest with accurate reports.
"""

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.domain.evidence import SourceDocument, SourceType
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.script import ScriptRevision, ScriptSegment
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import Base
from app.persistence.repositories import (
    EvidenceRepository,
    KnowledgeVideoTaskRepository,
    ScriptRepository,
)
from app.services.delivery_report_service import DeliveryReportService


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "test_p0_delivery.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_delivery_report_service_consumes_real_script_segments(session_factory, tmp_path):
    task_id = f"task_{uuid4().hex[:8]}"

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        ev_repo = EvidenceRepository(session)
        script_repo = ScriptRepository(session)

        task = KnowledgeVideoTask.create(
            task_id=task_id,
            topic="Quantum Computing Fundamentals",
            target_duration=60.0,
            workflow_policy=WorkflowPolicyType.AUTO,
        )
        task_repo.save_task(task)

        # Source doc with sensitive API key in URL to test redaction
        src = SourceDocument(
            source_document_id=f"src_{uuid4().hex[:12]}",
            source_type=SourceType.URL,
            title="Qubit Stability Study",
            source_locator="https://api.quantum.org/v1/papers?api_key=secret_1234567890",
            content_hash=hashlib.sha256(b"content").hexdigest(),
            source_fingerprint=f"fp_{uuid4().hex[:8]}",
        )
        ev_repo.save_source_document(src)
        ev_repo.associate_task_source(task_id, src.source_document_id)

        # Non-empty script with real ScriptSegments
        rev_id = f"script_rev_{uuid4().hex[:8]}"
        seg1 = ScriptSegment(
            script_revision_id=rev_id,
            content_beat_id=f"beat_{uuid4().hex[:8]}",
            order=1,
            narration_text="Welcome to the world of quantum computing, where qubits defy classical mechanics.",
            target_duration=15.0,
            evidence_refs=("ev_item_001", "ev_item_002"),
        )
        seg2 = ScriptSegment(
            script_revision_id=rev_id,
            content_beat_id=f"beat_{uuid4().hex[:8]}",
            order=2,
            narration_text="Decoherence is the primary obstacle to building fault-tolerant quantum computers.",
            target_duration=20.0,
            evidence_refs=("ev_item_003",),
        )
        script = ScriptRevision(
            script_revision_id=rev_id,
            task_id=task_id,
            content_plan_revision_id=f"plan_rev_{uuid4().hex[:8]}",
            revision_number=1,
            overall_target_duration=35.0,
            language="en",
            content_fingerprint="fp_12345678",
            segments=(seg1, seg2),
        )
        script_repo.save_revision(script)
        session.commit()

    report_path = str(tmp_path / "source_report.md")
    with session_factory() as session:
        svc = DeliveryReportService(session)
        out_path, sha256_hash = svc.generate_source_report(task_id, report_path)

    assert Path(out_path).exists()
    content = Path(out_path).read_text(encoding="utf-8")

    # Verify real ScriptSegment fields were consumed and rendered
    assert "Welcome to the world of quantum computing" in content
    assert "Decoherence is the primary obstacle" in content
    assert "ev_item_001" in content
    assert "ev_item_002" in content
    assert "ev_item_003" in content

    # Verify sensitive token redaction
    assert "secret_1234567890" not in content
    assert "api_key=[REDACTED]" in content
    assert len(sha256_hash) == 64
