"""knowledge processing and retrieval core: knowledge chunks and retrieval snapshots

Revision ID: 0017_knowledge_processing_and_retrieval
Revises: 0016_evidence_provenance_foundation
Create Date: 2026-09-12 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0017_knowledge_processing_and_retrieval"
down_revision: str | Sequence[str] | None = "0016_evidence_provenance_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create knowledge_chunks table
    op.create_table(
        "knowledge_chunks",
        sa.Column("chunk_id", sa.String(64), primary_key=True),
        sa.Column(
            "source_document_id",
            sa.String(36),
            sa.ForeignKey("source_documents.source_document_id", name="fk_knowledge_chunks_source_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("processing_version", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("locator_json", EvidenceType, nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_document_id", "processing_version", "chunk_index", name="uq_knowledge_chunks_source_ver_idx"),
    )
    op.create_index("ix_knowledge_chunks_source_document_id", "knowledge_chunks", ["source_document_id"])
    op.create_index("ix_knowledge_chunks_content_fingerprint", "knowledge_chunks", ["content_fingerprint"])
    op.create_index("ix_knowledge_chunks_created_at", "knowledge_chunks", ["created_at"])

    # 2. Create retrieval_snapshots table
    op.create_table(
        "retrieval_snapshots",
        sa.Column("retrieval_snapshot_id", sa.String(64), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", name="fk_retrieval_snapshots_task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("source_scope_ids_json", EvidenceType, nullable=False),
        sa.Column("retrieval_policy_version", sa.String(64), nullable=False),
        sa.Column("processing_version", sa.String(64), nullable=False),
        sa.Column("candidates_json", EvidenceType, nullable=False),
        sa.Column("selected_evidence_ids_json", EvidenceType, nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_retrieval_snapshots_task_id", "retrieval_snapshots", ["task_id"])
    op.create_index("ix_retrieval_snapshots_content_fingerprint", "retrieval_snapshots", ["content_fingerprint"])
    op.create_index("ix_retrieval_snapshots_created_at", "retrieval_snapshots", ["created_at"])


def downgrade() -> None:
    # 2. Drop retrieval_snapshots
    op.drop_index("ix_retrieval_snapshots_created_at", table_name="retrieval_snapshots")
    op.drop_index("ix_retrieval_snapshots_content_fingerprint", table_name="retrieval_snapshots")
    op.drop_index("ix_retrieval_snapshots_task_id", table_name="retrieval_snapshots")
    op.drop_table("retrieval_snapshots")

    # 1. Drop knowledge_chunks
    op.drop_index("ix_knowledge_chunks_created_at", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_content_fingerprint", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_source_document_id", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
