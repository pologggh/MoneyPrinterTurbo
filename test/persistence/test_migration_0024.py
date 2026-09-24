from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest
from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect

import importlib
from app.config import config

mod_0024 = importlib.import_module("migrations.versions.0024_hybrid_rag_knowledge_base")
IncompleteTableSchemaError = mod_0024.IncompleteTableSchemaError


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


def test_alembic_migration_0024_upgrade_downgrade_and_invariants(monkeypatch):
    """
    Test A & B:
    0023 -> 0024 full migration, verifies all 4 tables, columns, PKs, FKs, UQs, indexes.
    Downgrade 0024 -> 0023, verifies clean teardown.
    Re-upgrade 0023 -> 0024, verifies complete re-creation.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_norm.db")

        # 1. Upgrade to 0023
        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        # 2. Upgrade to 0024
        command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert {"knowledge_bases", "knowledge_base_sources", "task_knowledge_bases", "chunk_embeddings"}.issubset(tables)

        # Columns verification
        kb_cols = {c["name"] for c in inspector.get_columns("knowledge_bases")}
        assert {"knowledge_base_id", "name", "description", "status", "metadata_json", "created_at", "updated_at"}.issubset(kb_cols)

        kbs_cols = {c["name"] for c in inspector.get_columns("knowledge_base_sources")}
        assert {"knowledge_base_id", "source_document_id", "associated_at"}.issubset(kbs_cols)

        tkb_cols = {c["name"] for c in inspector.get_columns("task_knowledge_bases")}
        assert {"task_id", "knowledge_base_id", "attached_at"}.issubset(tkb_cols)

        emb_cols = {c["name"] for c in inspector.get_columns("chunk_embeddings")}
        assert {"embedding_id", "chunk_id", "provider", "model", "dimension", "embedding_vector", "text_hash", "retrieval_policy_version", "created_at"}.issubset(emb_cols)

        # Primary Key verification
        assert set(inspector.get_pk_constraint("knowledge_bases")["constrained_columns"]) == {"knowledge_base_id"}
        assert set(inspector.get_pk_constraint("knowledge_base_sources")["constrained_columns"]) == {"knowledge_base_id", "source_document_id"}
        assert set(inspector.get_pk_constraint("task_knowledge_bases")["constrained_columns"]) == {"task_id", "knowledge_base_id"}
        assert set(inspector.get_pk_constraint("chunk_embeddings")["constrained_columns"]) == {"embedding_id"}

        # Foreign Key verification
        kbs_fks = [(set(f["constrained_columns"]), f["referred_table"].lower()) for f in inspector.get_foreign_keys("knowledge_base_sources")]
        assert ({"knowledge_base_id"}, "knowledge_bases") in kbs_fks
        assert ({"source_document_id"}, "source_documents") in kbs_fks

        tkb_fks = [(set(f["constrained_columns"]), f["referred_table"].lower()) for f in inspector.get_foreign_keys("task_knowledge_bases")]
        assert ({"task_id"}, "knowledge_video_tasks") in tkb_fks
        assert ({"knowledge_base_id"}, "knowledge_bases") in tkb_fks

        emb_fks = [(set(f["constrained_columns"]), f["referred_table"].lower()) for f in inspector.get_foreign_keys("chunk_embeddings")]
        assert ({"chunk_id"}, "knowledge_chunks") in emb_fks

        # Unique Constraint verification
        emb_uqs = [set(u["column_names"]) for u in inspector.get_unique_constraints("chunk_embeddings")]
        emb_unique_indexes = [set(i["column_names"]) for i in inspector.get_indexes("chunk_embeddings") if i.get("unique")]
        assert {"chunk_id", "provider", "model"} in emb_uqs or {"chunk_id", "provider", "model"} in emb_unique_indexes

        # Indexes verification
        kb_indexes = {i["name"] for i in inspector.get_indexes("knowledge_bases")}
        assert {"ix_knowledge_bases_status", "ix_knowledge_bases_created_at"}.issubset(kb_indexes)

        kbs_indexes = {i["name"] for i in inspector.get_indexes("knowledge_base_sources")}
        assert {"ix_kb_sources_source_doc_id", "ix_kb_sources_associated_at"}.issubset(kbs_indexes)

        tkb_indexes = {i["name"] for i in inspector.get_indexes("task_knowledge_bases")}
        assert {"ix_task_kbs_knowledge_base_id", "ix_task_kbs_attached_at"}.issubset(tkb_indexes)

        emb_indexes = {i["name"] for i in inspector.get_indexes("chunk_embeddings")}
        assert {"ix_chunk_embeddings_chunk_id", "ix_chunk_embeddings_text_hash", "ix_chunk_embeddings_provider_model"}.issubset(emb_indexes)

        # alembic_version == 0024
        with engine.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0024_hybrid_rag_knowledge_base"

        # 3. Downgrade back to 0023
        engine.dispose()
        command.downgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine_down = create_engine(db_url)
        inspector_down = inspect(engine_down)
        tables_down = set(inspector_down.get_table_names())
        assert "knowledge_bases" not in tables_down
        assert "knowledge_base_sources" not in tables_down
        assert "task_knowledge_bases" not in tables_down
        assert "chunk_embeddings" not in tables_down

        with engine_down.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"

        # 4. Re-upgrade to 0024
        engine_down.dispose()
        command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        engine_reup = create_engine(db_url)
        inspector_reup = inspect(engine_reup)
        tables_reup = set(inspector_reup.get_table_names())
        assert {"knowledge_bases", "knowledge_base_sources", "task_knowledge_bases", "chunk_embeddings"}.issubset(tables_reup)
        engine_reup.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_partial_ddl_legitimate_recovery(monkeypatch):
    """
    Test C:
    Valid partial DDL recovery:
    - knowledge_bases is already completely created with 1 index.
    - knowledge_base_sources is already completely created WITH foreign keys.
    - Other tables and indexes do not exist.
    0024 completes missing tables and missing indexes without recreating existing ones.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_partial_valid.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(sa.text("CREATE INDEX ix_knowledge_bases_status ON knowledge_bases (status)"))
            # Structurally complete knowledge_base_sources with foreign keys
            conn.execute(sa.text("""
                CREATE TABLE knowledge_base_sources (
                    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(knowledge_base_id) ON DELETE CASCADE,
                    source_document_id VARCHAR(36) NOT NULL REFERENCES source_documents(source_document_id) ON DELETE CASCADE,
                    associated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (knowledge_base_id, source_document_id)
                )
            """))
        engine.dispose()

        # Run 0024 upgrade
        command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        engine_after = create_engine(db_url)
        inspector = inspect(engine_after)
        tables = set(inspector.get_table_names())
        assert {"knowledge_bases", "knowledge_base_sources", "task_knowledge_bases", "chunk_embeddings"}.issubset(tables)

        # Verify missing index ix_knowledge_bases_created_at was created
        kb_indexes = {idx["name"] for idx in inspector.get_indexes("knowledge_bases")}
        assert "ix_knowledge_bases_status" in kb_indexes
        assert "ix_knowledge_bases_created_at" in kb_indexes

        # Verify alembic_version is 0024
        with engine_after.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0024_hybrid_rag_knowledge_base"

        engine_after.dispose()
        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_missing_foreign_keys(monkeypatch):
    """
    Test D1:
    knowledge_base_sources exists but is missing foreign keys.
    Migration must fail, raise IncompleteTableSchemaError, and NOT update alembic_version.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_missing_fk.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            # Incomplete: table created without FOREIGN KEY constraints
            conn.execute(sa.text("""
                CREATE TABLE knowledge_base_sources (
                    knowledge_base_id VARCHAR(36) NOT NULL,
                    source_document_id VARCHAR(36) NOT NULL,
                    associated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (knowledge_base_id, source_document_id)
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_base_sources" in err_msg
        assert "Missing foreign key" in err_msg

        # Verify alembic_version is STILL 0023
        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_missing_required_columns(monkeypatch):
    """
    Test D2:
    knowledge_bases exists but is missing required column 'status'.
    Migration must fail and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_missing_col.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            # Missing 'status' and 'metadata_json'
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_bases" in err_msg
        assert "Missing required columns" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_chunk_embeddings_missing_unique_constraint(monkeypatch):
    """
    Test D3:
    chunk_embeddings exists but is missing unique constraint on (chunk_id, provider, model).
    Migration must fail and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_missing_uq.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            # Table has columns, PK, and FK, but lacks UNIQUE(chunk_id, provider, model)
            conn.execute(sa.text("""
                CREATE TABLE chunk_embeddings (
                    embedding_id VARCHAR(64) PRIMARY KEY,
                    chunk_id VARCHAR(64) NOT NULL REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE,
                    provider VARCHAR(64) NOT NULL,
                    model VARCHAR(128) NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding_vector JSON NOT NULL,
                    text_hash VARCHAR(64) NOT NULL,
                    retrieval_policy_version VARCHAR(64) NOT NULL DEFAULT 'vector_v1',
                    created_at TIMESTAMP NOT NULL
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "chunk_embeddings" in err_msg
        assert "Missing unique constraint" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_nullable_mismatch(monkeypatch):
    """
    Test D4:
    knowledge_bases exists with correct column names and PK, but business fields
    erroneously allow NULL (nullable=True).
    Migration must fail, raise IncompleteTableSchemaError, and NOT update alembic_version.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_nullable_flaw.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            # Defect: name, status, metadata_json, created_at, updated_at allow NULL
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255),
                    description TEXT,
                    status VARCHAR(32) DEFAULT 'ACTIVE',
                    metadata_json JSON DEFAULT '{}',
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_bases" in err_msg
        assert "nullability mismatch" in err_msg
        assert "name" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_column_type_and_length_mismatch(monkeypatch):
    """
    Test D5:
    knowledge_bases exists with correct column names, but with incorrect column types
    or string lengths (e.g. name VARCHAR(50) instead of VARCHAR(255), status INTEGER).
    Migration must fail and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_type_flaw.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    description TEXT,
                    status INTEGER NOT NULL DEFAULT 1,
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_bases" in err_msg
        assert "type mismatch" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_server_default_mismatch(monkeypatch):
    """
    Test D6:
    knowledge_bases exists with wrong server_default value (e.g. status default 'INACTIVE').
    Migration must fail and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_default_flaw.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'INACTIVE',
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_bases" in err_msg
        assert "server_default mismatch" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_foreign_key_missing_cascade(monkeypatch):
    """
    Test D7:
    knowledge_base_sources exists with foreign keys, but lacking ON DELETE CASCADE
    (e.g. ON DELETE SET NULL or default RESTRICT).
    Migration must fail and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_fk_no_cascade.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            # FK without ON DELETE CASCADE (default NO ACTION / RESTRICT)
            conn.execute(sa.text("""
                CREATE TABLE knowledge_base_sources (
                    knowledge_base_id VARCHAR(36) NOT NULL REFERENCES knowledge_bases(knowledge_base_id) ON DELETE SET NULL,
                    source_document_id VARCHAR(36) NOT NULL REFERENCES source_documents(source_document_id) ON DELETE CASCADE,
                    associated_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (knowledge_base_id, source_document_id)
                )
            """))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "knowledge_base_sources" in err_msg
        assert "Foreign key mismatch" in err_msg
        assert "ondelete mismatch" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_alembic_migration_0024_rejects_mismatched_index_columns_or_uniqueness(monkeypatch):
    """
    Test D8:
    Table exists and matches schema, but a pre-existing index has mismatched columns
    or mismatched uniqueness (e.g. ix_knowledge_bases_status built on created_at).
    Migration must fail, diagnose index mismatch, and NOT stamp 0024.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        alembic_cfg, db_url, dev_db_path, dev_db_stat = _setup_alembic_test_env(temp_dir, monkeypatch, "test_0024_bad_index.db")

        command.upgrade(alembic_cfg, "0023_delivery_manifest_artifact")

        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE knowledge_bases (
                    knowledge_base_id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                    metadata_json JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            # Existing index with WRONG column: indexed on 'created_at' instead of 'status'
            conn.execute(sa.text("CREATE INDEX ix_knowledge_bases_status ON knowledge_bases (created_at)"))
        engine.dispose()

        with pytest.raises(Exception) as exc_info:
            command.upgrade(alembic_cfg, "0024_hybrid_rag_knowledge_base")

        err_msg = str(exc_info.value)
        assert "ix_knowledge_bases_status" in err_msg
        assert "failed structural validation" in err_msg
        assert "column mismatch" in err_msg

        engine_check = create_engine(db_url)
        with engine_check.connect() as conn:
            current_rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar()
            assert current_rev == "0023_delivery_manifest_artifact"
        engine_check.dispose()

        _assert_dev_db_untouched(dev_db_path, dev_db_stat)


def test_validate_existing_table_rejects_unbounded_varchar():
    """An unbounded VARCHAR must not satisfy a migration VARCHAR length contract."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE knowledge_bases (
                knowledge_base_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                status VARCHAR NOT NULL DEFAULT 'ACTIVE',
                metadata_json JSON NOT NULL DEFAULT '{}',
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))

        with pytest.raises(IncompleteTableSchemaError, match="varchar length"):
            mod_0024._validate_existing_table(
                conn,
                "knowledge_bases",
                mod_0024.TABLE_SPECS["knowledge_bases"],
            )


def test_validate_existing_table_rejects_date_for_datetime():
    """A DATE column must not satisfy the DateTime/TIMESTAMP contract."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE knowledge_bases (
                knowledge_base_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
                metadata_json JSON NOT NULL DEFAULT '{}',
                created_at DATE NOT NULL,
                updated_at DATE NOT NULL
            )
        """))

        with pytest.raises(IncompleteTableSchemaError, match="datetime/timestamp"):
            mod_0024._validate_existing_table(
                conn,
                "knowledge_bases",
                mod_0024.TABLE_SPECS["knowledge_bases"],
            )


def test_get_foreign_keys_does_not_silence_sqlite_pragma_failure():
    """A failed SQLite FK inspection must surface a diagnostic error."""

    class FailingBind:
        class Dialect:
            name = "sqlite"

        dialect = Dialect()

        def execute(self, _statement):
            raise sa.exc.SQLAlchemyError("simulated pragma failure")

    class InspectorStub:
        def get_foreign_keys(self, _table_name):
            return []

    with pytest.raises(IncompleteTableSchemaError, match="foreign key metadata"):
        mod_0024._get_foreign_keys_with_options(
            FailingBind(),
            InspectorStub(),
            "knowledge_base_sources",
        )


def test_validate_index_rejects_same_name_with_wrong_uniqueness():
    """A same-name UNIQUE index must not satisfy a required non-unique index."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE knowledge_bases (status VARCHAR(32))"))
        conn.execute(sa.text(
            "CREATE UNIQUE INDEX ix_knowledge_bases_status ON knowledge_bases (status)"
        ))

        with pytest.raises(IncompleteTableSchemaError, match="uniqueness mismatch"):
            mod_0024._ensure_or_validate_index(
                conn,
                "knowledge_bases",
                "ix_knowledge_bases_status",
                ["status"],
                unique=False,
            )


def test_validate_existing_table_rejects_wrong_foreign_key_target_column():
    """A foreign key to the right table but wrong target column must be rejected."""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE knowledge_bases (
                knowledge_base_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) UNIQUE
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE source_documents (
                source_document_id VARCHAR(36) PRIMARY KEY,
                title VARCHAR(255) UNIQUE
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE knowledge_base_sources (
                knowledge_base_id VARCHAR(36) NOT NULL
                    REFERENCES knowledge_bases(name) ON DELETE CASCADE,
                source_document_id VARCHAR(36) NOT NULL
                    REFERENCES source_documents(source_document_id) ON DELETE CASCADE,
                associated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (knowledge_base_id, source_document_id)
            )
        """))

        with pytest.raises(IncompleteTableSchemaError, match="referred_columns mismatch"):
            mod_0024._validate_existing_table(
                conn,
                "knowledge_base_sources",
                mod_0024.TABLE_SPECS["knowledge_base_sources"],
            )
