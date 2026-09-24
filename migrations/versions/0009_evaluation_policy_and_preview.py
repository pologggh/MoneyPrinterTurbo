"""evaluation policy and composition preview schema

Revision ID: 0009_evaluation_policy_and_preview
Revises: 0008_evaluation_contracts
Create Date: 2026-09-11 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009_evaluation_policy_and_preview"
down_revision: str | Sequence[str] | None = "0008_evaluation_contracts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add overall_score to evaluation_snapshots
    op.add_column(
        "evaluation_snapshots",
        sa.Column("overall_score", sa.Float(), nullable=True),
    )

    # 2. Create composition_previews table
    op.create_table(
        "composition_previews",
        sa.Column("composition_preview_id", sa.String(36), primary_key=True),
        sa.Column(
            "shot_asset_version_id",
            sa.String(36),
            sa.ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("render_context_fingerprint", sa.String(64), nullable=False),
        sa.Column("preview_path", sa.String(512), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column(
            "preview_policy_version",
            sa.String(64),
            nullable=False,
            server_default="composition-preview-v1",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_comp_preview_shot_asset_version_id",
        "composition_previews",
        ["shot_asset_version_id"],
    )
    op.create_index(
        "ix_comp_preview_fingerprint",
        "composition_previews",
        ["shot_asset_version_id", "render_context_fingerprint", "preview_policy_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_comp_preview_fingerprint", table_name="composition_previews")
    op.drop_index("ix_comp_preview_shot_asset_version_id", table_name="composition_previews")
    op.drop_table("composition_previews")
    op.drop_column("evaluation_snapshots", "overall_score")
