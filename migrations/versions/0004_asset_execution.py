"""asset execution schema

Revision ID: 0004_asset_execution
Revises: 0003_asset_route_plan
Create Date: 2026-09-10 18:10:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_asset_execution"
down_revision: str | Sequence[str] | None = "0003_asset_route_plan"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. execution_runs
    op.create_table(
        "execution_runs",
        sa.Column("execution_run_id", sa.String(36), primary_key=True),
        sa.Column(
            "asset_route_plan_id",
            sa.String(36),
            sa.ForeignKey("asset_route_plans.asset_route_plan_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "storyboard_snapshot_id",
            sa.String(36),
            sa.ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("total_shots", sa.Integer(), nullable=False),
        sa.Column("succeeded_shots", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_shots", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_execution_run_plan_id",
        "execution_runs",
        ["asset_route_plan_id"],
    )
    op.create_index(
        "ix_execution_run_status",
        "execution_runs",
        ["status"],
    )

    # 2. execution_attempts
    op.create_table(
        "execution_attempts",
        sa.Column("execution_attempt_id", sa.String(36), primary_key=True),
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
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("generation_mode", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="PENDING"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("raw_provider_response_json", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_exec_attempt_run_id",
        "execution_attempts",
        ["execution_run_id"],
    )
    op.create_index(
        "ix_exec_attempt_shot_rev_id",
        "execution_attempts",
        ["shot_revision_id"],
    )
    op.create_index(
        "ix_exec_attempt_status",
        "execution_attempts",
        ["status"],
    )

    # 3. shot_asset_versions
    op.create_table(
        "shot_asset_versions",
        sa.Column("shot_asset_version_id", sa.String(36), primary_key=True),
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
        sa.Column(
            "execution_attempt_id",
            sa.String(36),
            sa.ForeignKey("execution_attempts.execution_attempt_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(32), nullable=False),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("generation_mode", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_shot_asset_version_shot_rev_id",
        "shot_asset_versions",
        ["shot_revision_id"],
    )
    op.create_index(
        "ix_shot_asset_version_attempt_id",
        "shot_asset_versions",
        ["execution_attempt_id"],
    )
    op.create_index(
        "ix_shot_asset_version_hash",
        "shot_asset_versions",
        ["file_hash"],
    )


def downgrade() -> None:
    op.drop_table("shot_asset_versions")
    op.drop_table("execution_attempts")
    op.drop_table("execution_runs")
