"""web research augmentation: web research snapshots

Revision ID: 0018_web_research_augmentation
Revises: 0017_knowledge_processing_and_retrieval
Create Date: 2026-09-12 21:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0018_web_research_augmentation"
down_revision: str | Sequence[str] | None = "0017_knowledge_processing_and_retrieval"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "web_research_snapshots",
        sa.Column("web_research_snapshot_id", sa.String(64), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey(
                "knowledge_video_tasks.task_id",
                name="fk_web_research_snapshots_task_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("stage_attempt", sa.Integer(), nullable=False),
        sa.Column("query", sa.String(500), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("search_results_json", EvidenceType, nullable=False),
        sa.Column("selected_urls_json", EvidenceType, nullable=False),
        sa.Column("fetch_outcomes_json", EvidenceType, nullable=False),
        sa.Column("created_source_document_ids_json", EvidenceType, nullable=False),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_web_research_snapshots_task_id",
        "web_research_snapshots",
        ["task_id"],
    )
    op.create_index(
        "ix_web_research_snapshots_content_fingerprint",
        "web_research_snapshots",
        ["content_fingerprint"],
    )
    op.create_index(
        "ix_web_research_snapshots_created_at",
        "web_research_snapshots",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_web_research_snapshots_created_at", table_name="web_research_snapshots")
    op.drop_index(
        "ix_web_research_snapshots_content_fingerprint",
        table_name="web_research_snapshots",
    )
    op.drop_index("ix_web_research_snapshots_task_id", table_name="web_research_snapshots")
    op.drop_table("web_research_snapshots")
