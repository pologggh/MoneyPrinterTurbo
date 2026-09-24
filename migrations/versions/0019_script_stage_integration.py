"""script stage integration: script revisions and script segments

Revision ID: 0019_script_stage_integration
Revises: 0018_web_research_augmentation
Create Date: 2026-09-12 23:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0019_script_stage_integration"
down_revision: str | Sequence[str] | None = "0018_web_research_augmentation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "script_revisions",
        sa.Column("script_revision_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey(
                "knowledge_video_tasks.task_id",
                name="fk_script_revisions_task_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "content_plan_revision_id",
            sa.String(36),
            sa.ForeignKey(
                "content_plan_revisions.content_plan_revision_id",
                name="fk_script_revisions_plan_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("revision_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("overall_target_duration", sa.Float(), nullable=False),
        sa.Column("language", sa.String(32), nullable=False, server_default="zh"),
        sa.Column("content_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_script_revisions_task_id",
        "script_revisions",
        ["task_id"],
    )
    op.create_index(
        "ix_script_revisions_plan_id",
        "script_revisions",
        ["content_plan_revision_id"],
    )
    op.create_index(
        "ix_script_revisions_fingerprint",
        "script_revisions",
        ["content_fingerprint"],
    )

    op.create_table(
        "script_segments",
        sa.Column("script_segment_id", sa.String(36), primary_key=True),
        sa.Column(
            "script_revision_id",
            sa.String(36),
            sa.ForeignKey(
                "script_revisions.script_revision_id",
                name="fk_script_segments_revision_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("content_beat_id", sa.String(36), nullable=False),
        sa.Column("beat_lineage_id", sa.String(36), nullable=True),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("narration_text", sa.Text(), nullable=False),
        sa.Column("target_duration", sa.Float(), nullable=False),
        sa.Column("beat_type", sa.String(32), nullable=True),
        sa.Column("evidence_refs", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "script_revision_id",
            "order",
            name="uq_script_segments_revision_order",
        ),
    )
    op.create_index(
        "ix_script_segments_revision_id",
        "script_segments",
        ["script_revision_id"],
    )
    op.create_index(
        "ix_script_segments_beat_id",
        "script_segments",
        ["content_beat_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_script_segments_beat_id", table_name="script_segments")
    op.drop_index("ix_script_segments_revision_id", table_name="script_segments")
    op.drop_table("script_segments")
    op.drop_index("ix_script_revisions_fingerprint", table_name="script_revisions")
    op.drop_index("ix_script_revisions_plan_id", table_name="script_revisions")
    op.drop_index("ix_script_revisions_task_id", table_name="script_revisions")
    op.drop_table("script_revisions")
