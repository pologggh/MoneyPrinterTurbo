"""storyboard stage integration: add script_segment_id to shot_revisions

Revision ID: 0020_shot_revision_script_segment_id
Revises: 0019_script_stage_integration
Create Date: 2026-09-12 23:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020_shot_revision_script_segment_id"
down_revision: str | Sequence[str] | None = "0019_script_stage_integration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "shot_revisions",
        sa.Column(
            "script_segment_id",
            sa.String(36),
            sa.ForeignKey(
                "script_segments.script_segment_id",
                name="fk_shot_revisions_script_segment_id",
                ondelete="SET NULL",
            ),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_shot_revisions_script_segment_id",
        "shot_revisions",
        ["script_segment_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_shot_revisions_script_segment_id", table_name="shot_revisions")
    op.drop_column("shot_revisions", "script_segment_id")
