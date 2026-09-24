from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.application.knowledge_base_service import (
    KnowledgeBaseNotFoundError,
    KnowledgeBaseServiceError,
)
from app.application.task_command_service import TaskCommandService
from app.domain.knowledge_base import KnowledgeBase, KnowledgeBaseStatus
from app.domain.knowledge_video_task import KnowledgeVideoTask, TaskStatus
from app.domain.workflow_job import WorkflowJob
from app.domain.workflow_state import JobStatus, Stage, WorkflowPolicyType
from app.persistence.models import Base, KnowledgeBaseORM, TaskKnowledgeBaseORM, WorkflowJobORM
from app.persistence.repositories import (
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


def test_create_task_with_valid_knowledge_bases_atomic(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb1 = KnowledgeBase.create(name="Biology KB")
        kb2 = KnowledgeBase.create(name="Chemistry KB")
        kb_repo.save_knowledge_base(kb1)
        kb_repo.save_knowledge_base(kb2)
        session.commit()

        kb1_id = kb1.knowledge_base_id
        kb2_id = kb2.knowledge_base_id

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        task = cmd_service.create_task(
            topic="Cellular Respiration",
            knowledge_base_ids=[kb1_id, kb2_id],
        )
        session.commit()
        task_id = task.task_id

    # Verify task, job, and attachments exist
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        kb_repo = KnowledgeBaseRepository(session)

        loaded_task = task_repo.get_task(task_id)
        assert loaded_task is not None
        assert loaded_task.topic == "Cellular Respiration"

        current_job = job_repo.get_current_job_for_task(task_id)
        assert current_job is not None
        assert current_job.stage == Stage.EVIDENCE

        attached_kbs = kb_repo.list_kbs_for_task(task_id)
        attached_ids = {k.knowledge_base_id for k in attached_kbs}
        assert attached_ids == {kb1_id, kb2_id}


def test_create_task_with_nonexistent_kb_fails_atomically(session_factory):
    non_existent_kb_id = "kb_missing_9999"

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        with pytest.raises(KnowledgeBaseNotFoundError, match="not found"):
            cmd_service.create_task(
                topic="Invalid Task",
                knowledge_base_ids=[non_existent_kb_id],
            )
        session.rollback()

    # Verify nothing was created in the database
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        tasks = task_repo.list_tasks(limit=10)
        assert len(tasks) == 0

        job_stmt = select(WorkflowJobORM)
        jobs = session.scalars(job_stmt).all()
        assert len(jobs) == 0


def test_create_task_with_archived_kb_fails_atomically(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Archived KB")
        kb_repo.save_knowledge_base(kb)
        kb_repo.archive_knowledge_base(kb.knowledge_base_id)
        session.commit()
        archived_id = kb.knowledge_base_id

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        with pytest.raises(KnowledgeBaseServiceError, match="archived|not active"):
            cmd_service.create_task(
                topic="Archived KB Task",
                knowledge_base_ids=[archived_id],
            )
        session.rollback()

    # Verify no task or job was saved
    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        tasks = task_repo.list_tasks(limit=10)
        assert len(tasks) == 0


def test_create_task_with_kb_and_initial_evidence_atomic(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="Quantum Physics")
        kb_repo.save_knowledge_base(kb)
        session.commit()
        kb_id = kb.knowledge_base_id

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        task = cmd_service.create_task(
            topic="Quantum Entanglement",
            knowledge_base_ids=[kb_id],
            initial_evidence=[
                {
                    "source_type": "TEXT",
                    "text_content": "Quantum entanglement is a physical phenomenon.",
                    "title": "Intro to Entanglement",
                },
                {
                    "source_type": "URL",
                    "url": "https://example.com/quantum",
                    "title": "Quantum Reference",
                },
            ],
        )
        session.commit()
        task_id = task.task_id

    # Verify atomic creation: task, job, KB attachment, and source documents all exist
    with session_factory() as session:
        from app.persistence.repositories import EvidenceRepository
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        loaded_task = task_repo.get_task(task_id)
        assert loaded_task is not None
        assert loaded_task.topic == "Quantum Entanglement"

        current_job = job_repo.get_current_job_for_task(task_id)
        assert current_job is not None
        assert current_job.stage == Stage.EVIDENCE

        attached_kbs = kb_repo.list_kbs_for_task(task_id)
        assert [k.knowledge_base_id for k in attached_kbs] == [kb_id]

        sources = ev_repo.list_sources_for_task(task_id)
        assert len(sources) == 2
        source_types = {s.source_type.value for s in sources}
        assert source_types == {"TEXT", "URL"}

        text_source = next(s for s in sources if s.source_type.value == "TEXT")
        assert "Quantum entanglement" in (text_source.content_snapshot or "")
        assert text_source.title == "Intro to Entanglement"

        url_source = next(s for s in sources if s.source_type.value == "URL")
        assert url_source.source_locator == "https://example.com/quantum"


def test_create_task_with_invalid_initial_evidence_rolls_back_everything(session_factory):
    with session_factory() as session:
        kb_repo = KnowledgeBaseRepository(session)
        kb = KnowledgeBase.create(name="AI Safety")
        kb_repo.save_knowledge_base(kb)
        session.commit()
        kb_id = kb.knowledge_base_id

    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        # Empty text_content should trigger validation error
        with pytest.raises(ValueError, match="non-empty text_content"):
            cmd_service.create_task(
                topic="AI Alignment",
                knowledge_base_ids=[kb_id],
                initial_evidence=[
                    {
                        "source_type": "TEXT",
                        "text_content": "   ",  # invalid whitespace
                    }
                ],
            )
        session.rollback()

    # Verify complete rollback: no task, no job, no KB association, no evidence
    with session_factory() as session:
        from app.persistence.repositories import EvidenceRepository
        task_repo = KnowledgeVideoTaskRepository(session)
        job_repo = WorkflowJobRepository(session)
        kb_repo = KnowledgeBaseRepository(session)
        ev_repo = EvidenceRepository(session)

        assert len(task_repo.list_tasks(limit=10)) == 0
        jobs = session.scalars(select(WorkflowJobORM)).all()
        assert len(jobs) == 0
        attachments = session.scalars(select(TaskKnowledgeBaseORM)).all()
        assert len(attachments) == 0


def test_create_task_with_invalid_url_evidence_rolls_back(session_factory):
    with session_factory() as session:
        cmd_service = TaskCommandService(session)
        # URL without http/https protocol
        with pytest.raises(ValueError, match="http:// or https://"):
            cmd_service.create_task(
                topic="Robotics",
                initial_evidence=[
                    {
                        "source_type": "URL",
                        "url": "ftp://example.com/robotics",
                    }
                ],
            )
        session.rollback()

    with session_factory() as session:
        task_repo = KnowledgeVideoTaskRepository(session)
        assert len(task_repo.list_tasks(limit=10)) == 0
