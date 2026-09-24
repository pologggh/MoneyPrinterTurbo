"""hybrid rag subsystem: knowledge bases, chunk embeddings, and task attachments

Revision ID: 0024_hybrid_rag_knowledge_base
Revises: 0023_delivery_manifest_artifact
Create Date: 2026-09-18 21:05:00.000000

"""
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0024_hybrid_rag_knowledge_base"
down_revision: str | Sequence[str] | None = "0023_delivery_manifest_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class IncompleteTableSchemaError(RuntimeError):
    """Raised when an existing database table violates expected migration schema invariants."""
    pass


def _table_exists(bind, table_name: str) -> bool:
    return table_name in sa.inspect(bind).get_table_names()


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    if not _table_exists(bind, table_name):
        return False
    indexes = sa.inspect(bind).get_indexes(table_name)
    return any(idx.get("name") == index_name for idx in indexes)


def _normalize_default(raw: Any) -> str | None:
    """Normalizes default values across SQLite, PostgreSQL, and other database dialects.

    Handles wrapping quotes, parentheses, and dialect-specific type casts
    (e.g., 'ACTIVE'::character varying -> 'ACTIVE', ('ACTIVE') -> 'ACTIVE').
    """
    if raw is None:
        return None
    val = str(raw).strip()
    if "::" in val:
        val = val.split("::")[0].strip()
    while val.startswith("(") and val.endswith(")"):
        val = val[1:-1].strip()
    while len(val) >= 2 and ((val[0] == "'" and val[-1] == "'") or (val[0] == '"' and val[-1] == '"')):
        val = val[1:-1].strip()
    return val


def _is_type_compatible(actual_type: Any, expected_category: str, expected_length: int | None = None) -> tuple[bool, str]:
    """Checks cross-dialect type compatibility for a column.

    Categories:
    - 'STRING': VARCHAR/String/CHAR, strictly verified for length if expected_length is not None.
    - 'TEXT': Text.
    - 'INTEGER': Integer.
    - 'JSON': JSON / JSONB.
    - 'DATETIME': DateTime / TIMESTAMP.
    """
    type_str = str(actual_type).upper()
    cls_name = type(actual_type).__name__.upper()

    if expected_category == "STRING":
        if isinstance(actual_type, sa.types.Text) or "TEXT" in cls_name:
            return False, f"expected string/varchar, found text ({actual_type})"
        if not (isinstance(actual_type, sa.types.String) or "VARCHAR" in cls_name or "STRING" in cls_name or "CHAR" in cls_name or "VARCHAR" in type_str):
            return False, f"expected string/varchar type, found {actual_type}"
        if expected_length is not None:
            actual_len = getattr(actual_type, "length", None)
            if actual_len != expected_length:
                return False, f"expected varchar length {expected_length}, found {actual_len}"
        return True, ""

    if expected_category == "TEXT":
        if not (isinstance(actual_type, sa.types.Text) or "TEXT" in cls_name or "TEXT" in type_str):
            return False, f"expected text type, found {actual_type}"
        return True, ""

    if expected_category == "INTEGER":
        if not (isinstance(actual_type, sa.types.Integer) or "INT" in cls_name or "INT" in type_str):
            return False, f"expected integer type, found {actual_type}"
        return True, ""

    if expected_category == "JSON":
        if not (isinstance(actual_type, sa.types.JSON) or "JSON" in cls_name or "JSON" in type_str):
            return False, f"expected json type, found {actual_type}"
        return True, ""

    if expected_category == "DATETIME":
        if not (
            isinstance(actual_type, sa.types.DateTime)
            or "DATETIME" in cls_name
            or "TIMESTAMP" in cls_name
            or "DATETIME" in type_str
            or "TIMESTAMP" in type_str
        ):
            return False, f"expected datetime/timestamp type, found {actual_type}"
        return True, ""

    return False, f"unknown expected category {expected_category}"


def _get_foreign_keys_with_options(bind, inspector, table_name: str) -> list[dict[str, Any]]:
    """Retrieves foreign key definitions with reliable ondelete attributes across dialects.

    For SQLite, complements SQLAlchemy inspector with PRAGMA foreign_key_list to capture
    CASCADE/SET NULL options on all table and column constraint forms.
    """
    fks = inspector.get_foreign_keys(table_name)
    if bind.dialect.name == "sqlite":
        try:
            pragma_rows = bind.execute(sa.text(f"PRAGMA foreign_key_list('{table_name}')")).mappings().all()
            pragma_map: dict[tuple[str, str, str], str] = {}
            for r in pragma_rows:
                key = (r["from"], (r["table"] or "").lower(), r["to"])
                pragma_map[key] = (r.get("on_delete") or "").upper()
            for fk in fks:
                opts = fk.setdefault("options", {})
                if not opts.get("ondelete"):
                    cols = fk.get("constrained_columns") or []
                    ref_cols = fk.get("referred_columns") or []
                    ref_tbl = (fk.get("referred_table") or "").lower()
                    if cols and ref_cols:
                        pkey = (cols[0], ref_tbl, ref_cols[0])
                        if pkey in pragma_map:
                            opts["ondelete"] = pragma_map[pkey]
        except sa.exc.SQLAlchemyError as exc:
            raise IncompleteTableSchemaError(
                f"Failed to inspect SQLite foreign key metadata for table "
                f"'{table_name}': {exc}"
            ) from exc
    return fks


def _has_unique_constraint(inspector, table_name: str, expected_cols: list[str]) -> bool:
    exp_set = set(expected_cols)
    for uq in inspector.get_unique_constraints(table_name):
        if set(uq.get("column_names") or []) == exp_set:
            return True
    for idx in inspector.get_indexes(table_name):
        if idx.get("unique") and set(idx.get("column_names") or []) == exp_set:
            return True
    return False


def _ensure_or_validate_index(
    bind,
    table_name: str,
    index_name: str,
    expected_columns: list[str],
    unique: bool = False,
) -> None:
    """Ensures an index exists with the expected columns and uniqueness, or creates it safely.

    - If index does not exist: creates it via op.create_index().
    - If index exists with matching columns, order, and uniqueness: succeeds.
    - If index exists with mismatched columns, order, or uniqueness: raises IncompleteTableSchemaError.
    """
    inspector = sa.inspect(bind)
    indexes = inspector.get_indexes(table_name)
    existing = next((idx for idx in indexes if idx.get("name") == index_name), None)

    if existing is None:
        op.create_index(index_name, table_name, expected_columns, unique=unique)
        return

    actual_columns = list(existing.get("column_names") or [])
    actual_unique = bool(existing.get("unique"))

    issues: list[str] = []
    if actual_columns != expected_columns:
        issues.append(f"column mismatch (expected {expected_columns}, found {actual_columns})")
    if actual_unique != unique:
        issues.append(f"uniqueness mismatch (expected unique={unique}, found unique={actual_unique})")

    if issues:
        raise IncompleteTableSchemaError(
            f"Pre-existing index '{index_name}' on table '{table_name}' failed structural validation for migration 0024: "
            + "; ".join(issues)
            + ". Action required: The existing index definition conflicts with migration 0024 requirements. Please drop or align the index before migrating."
        )


def _validate_existing_table(bind, table_name: str, spec: dict[str, Any]) -> None:
    """Validates structural integrity of a pre-existing table before proceeding with migration.

    Raises IncompleteTableSchemaError with detailed diagnostics if critical invariants are violated:
    - Missing required columns
    - Column type or length incompatibility
    - Column nullability mismatch
    - Missing or mismatched server_default
    - Primary key constraint mismatch
    - Foreign key constraint target, referred column, or ondelete mismatch
    - Unique constraint mismatch
    """
    inspector = sa.inspect(bind)
    issues: list[str] = []

    actual_cols_map = {col["name"]: col for col in inspector.get_columns(table_name)}
    expected_cols = spec["columns"]
    expected_pk = spec["pk"]

    # 1. Columns existence and detailed attributes
    missing_cols = set(expected_cols.keys()) - set(actual_cols_map.keys())
    if missing_cols:
        issues.append(f"Missing required columns: {sorted(missing_cols)}")

    for col_name, col_spec in expected_cols.items():
        if col_name not in actual_cols_map:
            continue

        actual_col = actual_cols_map[col_name]
        actual_type = actual_col["type"]

        # Type and length check
        ok, err = _is_type_compatible(actual_type, col_spec["category"], col_spec.get("length"))
        if not ok:
            issues.append(f"Column '{col_name}' type mismatch: {err}")

        # Nullability (for non-primary key columns; PK is verified via get_pk_constraint)
        if col_name not in expected_pk:
            exp_nullable = col_spec["nullable"]
            act_nullable = actual_col["nullable"]
            if act_nullable != exp_nullable:
                issues.append(
                    f"Column '{col_name}' nullability mismatch: expected nullable={exp_nullable}, found nullable={act_nullable}"
                )

        # Server default check
        if col_spec.get("server_default") is not None:
            exp_default = col_spec["server_default"]
            act_default_raw = actual_col.get("default")
            act_default_norm = _normalize_default(act_default_raw)
            if act_default_raw is None:
                issues.append(
                    f"Column '{col_name}' missing expected server_default '{exp_default}' (found None)"
                )
            elif act_default_norm != exp_default:
                issues.append(
                    f"Column '{col_name}' server_default mismatch: expected '{exp_default}', found '{act_default_raw}' (normalized: '{act_default_norm}')"
                )

    # 2. Primary key
    actual_pk = list(inspector.get_pk_constraint(table_name).get("constrained_columns") or [])
    if set(actual_pk) != set(expected_pk):
        issues.append(f"Primary key mismatch: expected {sorted(expected_pk)}, found {sorted(actual_pk)}")

    # 3. Foreign keys
    expected_fks = spec.get("fks") or []
    if expected_fks:
        actual_fks = _get_foreign_keys_with_options(bind, inspector, table_name)
        for exp_fk in expected_fks:
            exp_cols = list(exp_fk["constrained_columns"])
            exp_ref_table = exp_fk["referred_table"].lower()
            exp_ref_cols = list(exp_fk["referred_columns"])
            exp_ondelete = exp_fk.get("ondelete", "CASCADE").upper()

            matched = False
            mismatch_reasons: list[str] = []
            for afk in actual_fks:
                afk_cols = list(afk.get("constrained_columns") or [])
                afk_ref_tbl = (afk.get("referred_table") or "").lower()
                afk_ref_cols = list(afk.get("referred_columns") or [])
                afk_opts = afk.get("options") or {}
                afk_ondelete = (afk_opts.get("ondelete") or afk.get("ondelete") or "").upper()

                if afk_cols == exp_cols and afk_ref_tbl == exp_ref_table:
                    if afk_ref_cols != exp_ref_cols:
                        mismatch_reasons.append(
                            f"referred_columns mismatch (expected {exp_ref_cols}, found {afk_ref_cols})"
                        )
                        continue
                    if afk_ondelete != exp_ondelete:
                        mismatch_reasons.append(
                            f"ondelete mismatch (expected {exp_ondelete}, found {afk_ondelete or 'NONE/RESTRICT'})"
                        )
                        continue
                    matched = True
                    break

            if not matched:
                if mismatch_reasons:
                    issues.append(
                        f"Foreign key mismatch on columns {exp_cols} referencing '{exp_ref_table}': "
                        + "; ".join(mismatch_reasons)
                    )
                else:
                    issues.append(
                        f"Missing foreign key on columns {sorted(exp_cols)} referencing '{exp_ref_table}'"
                    )

    # 4. Unique constraints
    expected_uqs = spec.get("uqs") or []
    for exp_uq in expected_uqs:
        if not _has_unique_constraint(inspector, table_name, list(exp_uq)):
            issues.append(f"Missing unique constraint on columns {sorted(exp_uq)}")

    if issues:
        issues_str = "\n".join(f"    * {issue}" for issue in issues)
        raise IncompleteTableSchemaError(
            f"Pre-existing table '{table_name}' failed structural validation for migration 0024:\n"
            f"  - Incomplete schema details:\n"
            f"{issues_str}\n"
            f"Action required: The existing table schema is incomplete or inconsistent with the 0024 specification.\n"
            f"Automatic schema reconstruction cannot be performed safely across database dialects without risk of data loss.\n"
            f"Please inspect and manually align the '{table_name}' schema or migrate existing records before retrying migration."
        )


TABLE_SPECS: dict[str, dict[str, Any]] = {
    "knowledge_bases": {
        "columns": {
            "knowledge_base_id": {"category": "STRING", "length": 36, "nullable": False},
            "name": {"category": "STRING", "length": 255, "nullable": False},
            "description": {"category": "TEXT", "length": None, "nullable": True},
            "status": {"category": "STRING", "length": 32, "nullable": False, "server_default": "ACTIVE"},
            "metadata_json": {"category": "JSON", "length": None, "nullable": False, "server_default": "{}"},
            "created_at": {"category": "DATETIME", "length": None, "nullable": False},
            "updated_at": {"category": "DATETIME", "length": None, "nullable": False},
        },
        "pk": ["knowledge_base_id"],
        "fks": [],
        "uqs": [],
    },
    "knowledge_base_sources": {
        "columns": {
            "knowledge_base_id": {"category": "STRING", "length": 36, "nullable": False},
            "source_document_id": {"category": "STRING", "length": 36, "nullable": False},
            "associated_at": {"category": "DATETIME", "length": None, "nullable": False},
        },
        "pk": ["knowledge_base_id", "source_document_id"],
        "fks": [
            {
                "constrained_columns": ["knowledge_base_id"],
                "referred_table": "knowledge_bases",
                "referred_columns": ["knowledge_base_id"],
                "ondelete": "CASCADE",
            },
            {
                "constrained_columns": ["source_document_id"],
                "referred_table": "source_documents",
                "referred_columns": ["source_document_id"],
                "ondelete": "CASCADE",
            },
        ],
        "uqs": [],
    },
    "task_knowledge_bases": {
        "columns": {
            "task_id": {"category": "STRING", "length": 36, "nullable": False},
            "knowledge_base_id": {"category": "STRING", "length": 36, "nullable": False},
            "attached_at": {"category": "DATETIME", "length": None, "nullable": False},
        },
        "pk": ["task_id", "knowledge_base_id"],
        "fks": [
            {
                "constrained_columns": ["task_id"],
                "referred_table": "knowledge_video_tasks",
                "referred_columns": ["task_id"],
                "ondelete": "CASCADE",
            },
            {
                "constrained_columns": ["knowledge_base_id"],
                "referred_table": "knowledge_bases",
                "referred_columns": ["knowledge_base_id"],
                "ondelete": "CASCADE",
            },
        ],
        "uqs": [],
    },
    "chunk_embeddings": {
        "columns": {
            "embedding_id": {"category": "STRING", "length": 64, "nullable": False},
            "chunk_id": {"category": "STRING", "length": 64, "nullable": False},
            "provider": {"category": "STRING", "length": 64, "nullable": False},
            "model": {"category": "STRING", "length": 128, "nullable": False},
            "dimension": {"category": "INTEGER", "length": None, "nullable": False},
            "embedding_vector": {"category": "JSON", "length": None, "nullable": False},
            "text_hash": {"category": "STRING", "length": 64, "nullable": False},
            "retrieval_policy_version": {"category": "STRING", "length": 64, "nullable": False, "server_default": "vector_v1"},
            "created_at": {"category": "DATETIME", "length": None, "nullable": False},
        },
        "pk": ["embedding_id"],
        "fks": [
            {
                "constrained_columns": ["chunk_id"],
                "referred_table": "knowledge_chunks",
                "referred_columns": ["chunk_id"],
                "ondelete": "CASCADE",
            },
        ],
        "uqs": [
            {"chunk_id", "provider", "model"},
        ],
    },
}


def upgrade() -> None:
    bind = op.get_bind()

    # 1. knowledge_bases
    if not _table_exists(bind, "knowledge_bases"):
        op.create_table(
            "knowledge_bases",
            sa.Column("knowledge_base_id", sa.String(36), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("status", sa.String(32), nullable=False, server_default="ACTIVE"),
            sa.Column("metadata_json", EvidenceType, nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    else:
        _validate_existing_table(bind, "knowledge_bases", TABLE_SPECS["knowledge_bases"])

    _ensure_or_validate_index(bind, "knowledge_bases", "ix_knowledge_bases_status", ["status"], unique=False)
    _ensure_or_validate_index(bind, "knowledge_bases", "ix_knowledge_bases_created_at", ["created_at"], unique=False)

    # 2. knowledge_base_sources
    if not _table_exists(bind, "knowledge_base_sources"):
        op.create_table(
            "knowledge_base_sources",
            sa.Column(
                "knowledge_base_id",
                sa.String(36),
                sa.ForeignKey(
                    "knowledge_bases.knowledge_base_id",
                    name="fk_knowledge_base_sources_kb_id",
                    ondelete="CASCADE",
                ),
                primary_key=True,
            ),
            sa.Column(
                "source_document_id",
                sa.String(36),
                sa.ForeignKey(
                    "source_documents.source_document_id",
                    name="fk_knowledge_base_sources_source_doc_id",
                    ondelete="CASCADE",
                ),
                primary_key=True,
            ),
            sa.Column("associated_at", sa.DateTime(timezone=True), nullable=False),
        )
    else:
        _validate_existing_table(bind, "knowledge_base_sources", TABLE_SPECS["knowledge_base_sources"])

    _ensure_or_validate_index(bind, "knowledge_base_sources", "ix_kb_sources_source_doc_id", ["source_document_id"], unique=False)
    _ensure_or_validate_index(bind, "knowledge_base_sources", "ix_kb_sources_associated_at", ["associated_at"], unique=False)

    # 3. task_knowledge_bases
    if not _table_exists(bind, "task_knowledge_bases"):
        op.create_table(
            "task_knowledge_bases",
            sa.Column(
                "task_id",
                sa.String(36),
                sa.ForeignKey(
                    "knowledge_video_tasks.task_id",
                    name="fk_task_knowledge_bases_task_id",
                    ondelete="CASCADE",
                ),
                primary_key=True,
            ),
            sa.Column(
                "knowledge_base_id",
                sa.String(36),
                sa.ForeignKey(
                    "knowledge_bases.knowledge_base_id",
                    name="fk_task_knowledge_bases_kb_id",
                    ondelete="CASCADE",
                ),
                primary_key=True,
            ),
            sa.Column("attached_at", sa.DateTime(timezone=True), nullable=False),
        )
    else:
        _validate_existing_table(bind, "task_knowledge_bases", TABLE_SPECS["task_knowledge_bases"])

    _ensure_or_validate_index(bind, "task_knowledge_bases", "ix_task_kbs_knowledge_base_id", ["knowledge_base_id"], unique=False)
    _ensure_or_validate_index(bind, "task_knowledge_bases", "ix_task_kbs_attached_at", ["attached_at"], unique=False)

    # 4. chunk_embeddings
    if not _table_exists(bind, "chunk_embeddings"):
        op.create_table(
            "chunk_embeddings",
            sa.Column("embedding_id", sa.String(64), primary_key=True),
            sa.Column(
                "chunk_id",
                sa.String(64),
                sa.ForeignKey(
                    "knowledge_chunks.chunk_id",
                    name="fk_chunk_embeddings_chunk_id",
                    ondelete="CASCADE",
                ),
                nullable=False,
            ),
            sa.Column("provider", sa.String(64), nullable=False),
            sa.Column("model", sa.String(128), nullable=False),
            sa.Column("dimension", sa.Integer(), nullable=False),
            sa.Column("embedding_vector", EvidenceType, nullable=False),
            sa.Column("text_hash", sa.String(64), nullable=False),
            sa.Column("retrieval_policy_version", sa.String(64), nullable=False, server_default="vector_v1"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("chunk_id", "provider", "model", name="uq_chunk_embeddings_chunk_provider_model"),
        )
    else:
        _validate_existing_table(bind, "chunk_embeddings", TABLE_SPECS["chunk_embeddings"])

    _ensure_or_validate_index(bind, "chunk_embeddings", "ix_chunk_embeddings_chunk_id", ["chunk_id"], unique=False)
    _ensure_or_validate_index(bind, "chunk_embeddings", "ix_chunk_embeddings_text_hash", ["text_hash"], unique=False)
    _ensure_or_validate_index(bind, "chunk_embeddings", "ix_chunk_embeddings_provider_model", ["provider", "model"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()

    # 4. chunk_embeddings
    if _table_exists(bind, "chunk_embeddings"):
        for idx_name in ("ix_chunk_embeddings_provider_model", "ix_chunk_embeddings_text_hash", "ix_chunk_embeddings_chunk_id"):
            if _index_exists(bind, "chunk_embeddings", idx_name):
                op.drop_index(idx_name, table_name="chunk_embeddings")
        op.drop_table("chunk_embeddings")

    # 3. task_knowledge_bases
    if _table_exists(bind, "task_knowledge_bases"):
        for idx_name in ("ix_task_kbs_attached_at", "ix_task_kbs_knowledge_base_id"):
            if _index_exists(bind, "task_knowledge_bases", idx_name):
                op.drop_index(idx_name, table_name="task_knowledge_bases")
        op.drop_table("task_knowledge_bases")

    # 2. knowledge_base_sources
    if _table_exists(bind, "knowledge_base_sources"):
        for idx_name in ("ix_kb_sources_associated_at", "ix_kb_sources_source_doc_id"):
            if _index_exists(bind, "knowledge_base_sources", idx_name):
                op.drop_index(idx_name, table_name="knowledge_base_sources")
        op.drop_table("knowledge_base_sources")

    # 1. knowledge_bases
    if _table_exists(bind, "knowledge_bases"):
        for idx_name in ("ix_knowledge_bases_created_at", "ix_knowledge_bases_status"):
            if _index_exists(bind, "knowledge_bases", idx_name):
                op.drop_index(idx_name, table_name="knowledge_bases")
        op.drop_table("knowledge_bases")
