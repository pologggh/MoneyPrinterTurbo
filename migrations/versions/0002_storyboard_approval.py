"""storyboard approval schema

Revision ID: 0002_storyboard_approval
Revises: 0001_initial_core_domain
Create Date: 2026-09-10 17:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_storyboard_approval"
down_revision: Union[str, Sequence[str], None] = "0001_initial_core_domain"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

evidence_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "storyboard_approval_records",
        sa.Column("storyboard_approval_id", sa.String(36), primary_key=True),
        sa.Column(
            "source_draft_snapshot_id",
            sa.String(36),
            sa.ForeignKey(
                "storyboard_snapshots.storyboard_snapshot_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column(
            "approved_storyboard_snapshot_id",
            sa.String(36),
            sa.ForeignKey(
                "storyboard_snapshots.storyboard_snapshot_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column(
            "content_plan_revision_id",
            sa.String(36),
            sa.ForeignKey(
                "content_plan_revisions.content_plan_revision_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column(
            "exact_shot_revision_ids",
            evidence_type,
            nullable=False,
        ),
        sa.Column("approved_by", sa.String(128), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_note", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_sb_approval_source_draft",
        "storyboard_approval_records",
        ["source_draft_snapshot_id"],
    )
    op.create_index(
        "ix_sb_approval_approved_snapshot",
        "storyboard_approval_records",
        ["approved_storyboard_snapshot_id"],
    )
    op.create_index(
        "ix_sb_approval_plan_revision",
        "storyboard_approval_records",
        ["content_plan_revision_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sb_approval_plan_revision",
        table_name="storyboard_approval_records",
    )
    op.drop_index(
        "ix_sb_approval_approved_snapshot",
        table_name="storyboard_approval_records",
    )
    op.drop_index(
        "ix_sb_approval_source_draft",
        table_name="storyboard_approval_records",
    )
    op.drop_table("storyboard_approval_records")
