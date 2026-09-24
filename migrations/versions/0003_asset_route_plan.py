"""asset route plan schema

Revision ID: 0003_asset_route_plan
Revises: 0002_storyboard_approval
Create Date: 2026-09-10 18:05:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_asset_route_plan"
down_revision: str | Sequence[str] | None = "0002_storyboard_approval"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "asset_route_plans",
        sa.Column("asset_route_plan_id", sa.String(36), primary_key=True),
        sa.Column(
            "storyboard_snapshot_id",
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
        sa.Column("routing_strategy", sa.String(64), nullable=False),
        sa.Column("routing_policy_version", sa.String(64), nullable=False),
        sa.Column("model_selection_mode", sa.String(64), nullable=False, server_default="AUTO"),
        sa.Column("selected_provider", sa.String(128), nullable=True),
        sa.Column("selected_model", sa.String(128), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="READY"),
        sa.Column("total_shots", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("routed_shots", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("blocked_shots", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_asset_route_plan_snapshot_id",
        "asset_route_plans",
        ["storyboard_snapshot_id"],
    )
    op.create_index(
        "ix_asset_route_plan_status",
        "asset_route_plans",
        ["status"],
    )

    op.create_table(
        "shot_route_plan_entries",
        sa.Column("entry_id", sa.String(36), primary_key=True),
        sa.Column(
            "asset_route_plan_id",
            sa.String(36),
            sa.ForeignKey(
                "asset_route_plans.asset_route_plan_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "shot_id",
            sa.String(36),
            sa.ForeignKey(
                "shots.shot_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column(
            "shot_revision_id",
            sa.String(36),
            sa.ForeignKey(
                "shot_revisions.shot_revision_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column("beat_lineage_id", sa.String(36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("requested_visual_type", sa.String(64), nullable=False),
        sa.Column("route_status", sa.String(32), nullable=False),
        sa.Column("asset_routing_request_json", sa.Text(), nullable=False),
        sa.Column("route_decision_json", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_shot_route_entry_plan_id",
        "shot_route_plan_entries",
        ["asset_route_plan_id"],
    )
    op.create_index(
        "ix_shot_route_entry_shot_rev_id",
        "shot_route_plan_entries",
        ["shot_revision_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_shot_route_entry_shot_rev_id", table_name="shot_route_plan_entries")
    op.drop_index("ix_shot_route_entry_plan_id", table_name="shot_route_plan_entries")
    op.drop_table("shot_route_plan_entries")

    op.drop_index("ix_asset_route_plan_status", table_name="asset_route_plans")
    op.drop_index("ix_asset_route_plan_snapshot_id", table_name="asset_route_plans")
    op.drop_table("asset_route_plans")
