"""evidence provenance foundation: source documents, task sources, evidence items, knowledge claims, and evidence snapshots

Revision ID: 0016_evidence_provenance_foundation
Revises: 0015_stage_worker_and_task_artifacts
Create Date: 2026-09-12 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0016_evidence_provenance_foundation"
down_revision: str | Sequence[str] | None = "0015_stage_worker_and_task_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create source_documents table
    op.create_table(
        "source_documents",
        sa.Column("source_document_id", sa.String(36), primary_key=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column("content_snapshot", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source_fingerprint", sa.String(64), nullable=False),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("media_type", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metadata_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_source_documents_source_type", "source_documents", ["source_type"])
    op.create_index("ix_source_documents_source_fingerprint", "source_documents", ["source_fingerprint"])
    op.create_index("ix_source_documents_status", "source_documents", ["status"])
    op.create_index("ix_source_documents_created_at", "source_documents", ["created_at"])

    # 2. Create knowledge_video_task_sources association table
    op.create_table(
        "knowledge_video_task_sources",
        sa.Column("task_source_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", name="fk_kv_task_sources_task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_document_id",
            sa.String(36),
            sa.ForeignKey("source_documents.source_document_id", name="fk_kv_task_sources_source_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(32), nullable=False, server_default="PRIMARY"),
        sa.Column("associated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("task_id", "source_document_id", name="uq_kv_task_source"),
    )
    op.create_index("ix_kv_task_sources_task_id", "knowledge_video_task_sources", ["task_id"])
    op.create_index("ix_kv_task_sources_source_id", "knowledge_video_task_sources", ["source_document_id"])

    # 3. Create evidence_items table
    op.create_table(
        "evidence_items",
        sa.Column("evidence_id", sa.String(64), primary_key=True),
        sa.Column(
            "source_document_id",
            sa.String(36),
            sa.ForeignKey("source_documents.source_document_id", name="fk_evidence_items_source_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("locator_json", EvidenceType, nullable=False),
        sa.Column("original_excerpt", sa.Text(), nullable=False),
        sa.Column("normalized_fact", sa.Text(), nullable=True),
        sa.Column("evidence_role", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("extraction_method", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evidence_items_source_document_id", "evidence_items", ["source_document_id"])
    op.create_index("ix_evidence_items_evidence_role", "evidence_items", ["evidence_role"])
    op.create_index("ix_evidence_items_content_hash", "evidence_items", ["content_hash"])
    op.create_index("ix_evidence_items_created_at", "evidence_items", ["created_at"])

    # 4. Create knowledge_claims table
    op.create_table(
        "knowledge_claims",
        sa.Column("knowledge_claim_id", sa.String(36), primary_key=True),
        sa.Column("claim_type", sa.String(32), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("evidence_refs_json", EvidenceType, nullable=False),
        sa.Column("verification_status", sa.String(32), nullable=False),
        sa.Column("conflict_evidence_refs_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_knowledge_claims_claim_type", "knowledge_claims", ["claim_type"])
    op.create_index("ix_knowledge_claims_verification_status", "knowledge_claims", ["verification_status"])
    op.create_index("ix_knowledge_claims_created_at", "knowledge_claims", ["created_at"])

    # 5. Create evidence_snapshots table
    op.create_table(
        "evidence_snapshots",
        sa.Column("evidence_snapshot_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", name="fk_evidence_snapshots_task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source_document_ids_json", EvidenceType, nullable=False),
        sa.Column("evidence_ids_json", EvidenceType, nullable=False),
        sa.Column("knowledge_claim_ids_json", EvidenceType, nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_evidence_snapshots_task_id", "evidence_snapshots", ["task_id"])
    op.create_index("ix_evidence_snapshots_content_fingerprint", "evidence_snapshots", ["content_fingerprint"])
    op.create_index("ix_evidence_snapshots_created_at", "evidence_snapshots", ["created_at"])


def downgrade() -> None:
    # 5. Drop evidence_snapshots
    op.drop_index("ix_evidence_snapshots_created_at", table_name="evidence_snapshots")
    op.drop_index("ix_evidence_snapshots_content_fingerprint", table_name="evidence_snapshots")
    op.drop_index("ix_evidence_snapshots_task_id", table_name="evidence_snapshots")
    op.drop_table("evidence_snapshots")

    # 4. Drop knowledge_claims
    op.drop_index("ix_knowledge_claims_created_at", table_name="knowledge_claims")
    op.drop_index("ix_knowledge_claims_verification_status", table_name="knowledge_claims")
    op.drop_index("ix_knowledge_claims_claim_type", table_name="knowledge_claims")
    op.drop_table("knowledge_claims")

    # 3. Drop evidence_items
    op.drop_index("ix_evidence_items_created_at", table_name="evidence_items")
    op.drop_index("ix_evidence_items_content_hash", table_name="evidence_items")
    op.drop_index("ix_evidence_items_evidence_role", table_name="evidence_items")
    op.drop_index("ix_evidence_items_source_document_id", table_name="evidence_items")
    op.drop_table("evidence_items")

    # 2. Drop knowledge_video_task_sources
    op.drop_index("ix_kv_task_sources_source_id", table_name="knowledge_video_task_sources")
    op.drop_index("ix_kv_task_sources_task_id", table_name="knowledge_video_task_sources")
    op.drop_table("knowledge_video_task_sources")

    # 1. Drop source_documents
    op.drop_index("ix_source_documents_created_at", table_name="source_documents")
    op.drop_index("ix_source_documents_status", table_name="source_documents")
    op.drop_index("ix_source_documents_source_fingerprint", table_name="source_documents")
    op.drop_index("ix_source_documents_source_type", table_name="source_documents")
    op.drop_table("source_documents")
