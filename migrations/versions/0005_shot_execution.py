"""shot execution schema

Revision ID: 0005_shot_execution
Revises: 0004_asset_execution
Create Date: 2026-09-10 18:52:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_shot_execution"
down_revision: str | Sequence[str] | None = "0004_asset_execution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shot_executions",
        sa.Column("shot_execution_id", sa.String(36), primary_key=True),
        sa.Column(
            "execution_run_id",
            sa.String(36),
            sa.ForeignKey("execution_runs.execution_run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "shot_id",
            sa.String(36),
            sa.ForeignKey("shots.shot_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "shot_revision_id",
            sa.String(36),
            sa.ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("current_candidate_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("produced_asset_version_id", sa.String(36), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("unconfirmed_remote_task_id", sa.String(128), nullable=True),
        sa.Column("route_decision_json", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_shot_execution_run_id",
        "shot_executions",
        ["execution_run_id"],
    )
    op.create_index(
        "ix_shot_execution_shot_rev_id",
        "shot_executions",
        ["shot_revision_id"],
    )
    op.create_index(
        "ix_shot_execution_status",
        "shot_executions",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("shot_executions")
