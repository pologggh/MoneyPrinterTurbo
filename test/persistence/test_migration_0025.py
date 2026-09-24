from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest
from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.config import config
from app.domain.evidence import KnowledgeChunk, RetrievalCandidate, RetrievalSnapshot, SourceDocument
from app.domain.knowledge_video_task import KnowledgeVideoTask
from app.domain.workflow_state import WorkflowPolicyType
from app.persistence.models import RetrievalSnapshotORM
from app.persistence.repositories import EvidenceRepository, KnowledgeVideoTaskRepository


def _setup_alembic_test_env(tmp_dir: str, monkeypatch: pytest.MonkeyPatch, db_name: str) -> tuple[Config, str, Path, float | None]:
    dev_db_path = Path(config.root_dir) / "storage" / "dev.db"
    dev_db_stat = dev_db_path.stat().st_mtime if dev_db_path.exists() else None

    db_path = Path(tmp_dir) / db_name
    db_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    assert os.getenv("DATABASE_URL") == db_url

    ini_path = Path(config.root_dir) / "alembic.ini"
    alembic_cfg = Config(str(ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", str(Path(config.root_dir) / "migrations"))

    return alembic_cfg, db_url, dev_db_path, dev_db_stat


def _assert_dev_db_untouched(dev_db_path: Path, dev_db_stat: float | None) -> None:
    if dev_db_stat is None:
        assert not dev_db_path.exists(), "storage/dev.db should not have been created by migration test"
    else:
        assert dev_db_path.exists()
        assert dev_db_path.stat().st_mtime == dev_db_stat, "storage/dev.db was modified by migration test"


def test_alembic_migration_0025_upgrade_downgrade_and_persistence(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0025_real.db")

        # 1. Upgrade to 0024
        command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        # 2. Verify effective_retrieval_mode does NOT exist in 0024
        engine = create_engine(db_url)
        inspector = inspect(engine)
        assert "retrieval_snapshots" in inspector.get_table_names()
        cols_0024 = {c["name"] for c in inspector.get_columns("retrieval_snapshots")}
        assert "effective_retrieval_mode" not in cols_0024

        # 3. Upgrade to 0025
        engine.dispose()
        command.upgrade(alembic_cfg, "0025_retrieval_snapshot_effective_mode")

        # 4. Verify column created
        engine_0025 = create_engine(db_url)
        inspector_0025 = inspect(engine_0025)
        cols_0025 = {c["name"] for c in inspector_0025.get_columns("retrieval_snapshots")}
        assert "effective_retrieval_mode" in cols_0025

        with engine_0025.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0025_retrieval_snapshot_effective_mode"

        # 5. Write and read the field to verify real persistence in migrated DB
        SessionLocal = sessionmaker(bind=engine_0025, autoflush=False, autocommit=False)
        with SessionLocal() as session:
            task_repo = KnowledgeVideoTaskRepository(session)
            task = KnowledgeVideoTask.create(
                topic="Effective Mode Real Migration Test",
                workflow_policy=WorkflowPolicyType.AUTO,
            )
            task_repo.save_task(task)

            ev_repo = EvidenceRepository(session)
            doc = SourceDocument.create_text(
                text="Quantum computing leverages superposition and entanglement.",
                title="Quantum Physics",
            )
            ev_repo.save_source_document(doc)

            cand = RetrievalCandidate(
                rank=1,
                chunk_id="chk_quantum_1",
                source_document_id=doc.source_document_id,
                score=0.95,
                retrieval_method="HYBRID_RRF",
                locator={},
                excerpt="superposition and entanglement",
            )

            snapshot = RetrievalSnapshot.create(
                task_id=task.task_id,
                query="quantum superposition",
                source_scope_ids=[doc.source_document_id],
                candidates=[cand],
                selected_evidence_ids=[],
                effective_retrieval_mode="HYBRID",
            )
            ev_repo.save_retrieval_snapshot(snapshot)
            session.commit()
            snapshot_id = snapshot.retrieval_snapshot_id

        # Verify loaded from DB
        with SessionLocal() as session:
            ev_repo = EvidenceRepository(session)
            loaded = ev_repo.get_retrieval_snapshot(snapshot_id)
            assert loaded is not None
            assert loaded.effective_retrieval_mode == "HYBRID"

            orm = session.get(RetrievalSnapshotORM, snapshot_id)
            assert orm is not None
            assert orm.effective_retrieval_mode == "HYBRID"

        # 6. Downgrade from 0025 to 0024
        engine_0025.dispose()
        command.downgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        engine_down = create_engine(db_url)
        inspector_down = inspect(engine_down)
        cols_down = {c["name"] for c in inspector_down.get_columns("retrieval_snapshots")}
        assert "effective_retrieval_mode" not in cols_down

        with engine_down.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0024_hybrid_rag_knowledge_base"

        # 7. Re-upgrade to 0025
        engine_down.dispose()
        command.upgrade(alembic_cfg, "0025_retrieval_snapshot_effective_mode")

        engine_reup = create_engine(db_url)
        inspector_reup = inspect(engine_reup)
        cols_reup = {c["name"] for c in inspector_reup.get_columns("retrieval_snapshots")}
        assert "effective_retrieval_mode" in cols_reup
        engine_reup.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0025_partial_ddl_column_exists(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0025_partial.db")

        # Upgrade to 0024
        command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        # Simulate partial DDL: column already created while alembic_version is still 0024
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE retrieval_snapshots ADD COLUMN effective_retrieval_mode VARCHAR(32) DEFAULT 'BM25_ONLY' NOT NULL"))
        engine.dispose()

        # Upgrade to 0025; must not fail with "duplicate column name"
        command.upgrade(alembic_cfg, "0025_retrieval_snapshot_effective_mode")

        engine_after = create_engine(db_url)
        inspector = inspect(engine_after)
        cols = {c["name"] for c in inspector.get_columns("retrieval_snapshots")}
        assert "effective_retrieval_mode" in cols

        with engine_after.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0025_retrieval_snapshot_effective_mode"
        engine_after.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)
