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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {
        column["name"] for column in inspector.get_columns("shot_revisions")
    }
    if "script_segment_id" not in existing_columns:
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("shot_revisions") as batch_op:
                batch_op.add_column(
                    sa.Column(
                        "script_segment_id",
                        sa.String(36),
                        sa.ForeignKey(
                            "script_segments.script_segment_id",
                            name="fk_shot_revisions_script_segment_id",
                            ondelete="SET NULL",
                        ),
                        nullable=True,
                    )
                )
        else:
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

    existing_indexes = {
        index["name"] for index in sa.inspect(bind).get_indexes("shot_revisions")
    }
    if "ix_shot_revisions_script_segment_id" not in existing_indexes:
        op.create_index(
            "ix_shot_revisions_script_segment_id",
            "shot_revisions",
            ["script_segment_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_index("ix_shot_revisions_script_segment_id", table_name="shot_revisions")
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("shot_revisions") as batch_op:
            batch_op.drop_column("script_segment_id")
    else:
        op.drop_column("shot_revisions", "script_segment_id")
