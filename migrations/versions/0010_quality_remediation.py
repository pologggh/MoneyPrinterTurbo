"""quality remediation decisions and selections schema

Revision ID: 0010_quality_remediation
Revises: 0009_evaluation_policy_and_preview
Create Date: 2026-09-11 20:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0010_quality_remediation"
down_revision: str | Sequence[str] | None = "0009_evaluation_policy_and_preview"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create quality_remediation_decisions table
    op.create_table(
        "quality_remediation_decisions",
        sa.Column("remediation_decision_id", sa.String(36), primary_key=True),
        sa.Column("quality_chain_id", sa.String(36), nullable=False),
        sa.Column(
            "evaluation_snapshot_id",
            sa.String(36),
            sa.ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("shot_id", sa.String(36), nullable=False),
        sa.Column("shot_revision_id", sa.String(36), nullable=False),
        sa.Column("shot_asset_version_id", sa.String(36), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("reason_codes", EvidenceType, nullable=False),
        sa.Column("remediation_policy_version", sa.String(64), nullable=False),
        sa.Column("quality_attempt_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("selected_route_candidate_json", EvidenceType, nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_quality_remediation_chain_id",
        "quality_remediation_decisions",
        ["quality_chain_id"],
    )
    op.create_index(
        "ix_quality_remediation_shot_id",
        "quality_remediation_decisions",
        ["shot_id"],
    )
    op.create_index(
        "ix_quality_remediation_snapshot_id",
        "quality_remediation_decisions",
        ["evaluation_snapshot_id"],
    )

    # 2. Create shot_quality_selections table
    op.create_table(
        "shot_quality_selections",
        sa.Column("quality_selection_id", sa.String(36), primary_key=True),
        sa.Column("quality_chain_id", sa.String(36), nullable=False),
        sa.Column("shot_id", sa.String(36), nullable=False),
        sa.Column("shot_revision_id", sa.String(36), nullable=False),
        sa.Column(
            "shot_asset_version_id",
            sa.String(36),
            sa.ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_snapshot_id",
            sa.String(36),
            sa.ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("selection_source", sa.String(64), nullable=False, server_default="EVALUATION_PASS"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_shot_quality_selection_chain_id",
        "shot_quality_selections",
        ["quality_chain_id"],
    )
    op.create_index(
        "ix_shot_quality_selection_shot_id",
        "shot_quality_selections",
        ["shot_id"],
    )
    op.create_index(
        "ix_shot_quality_selection_asset_id",
        "shot_quality_selections",
        ["shot_asset_version_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_shot_quality_selection_asset_id", table_name="shot_quality_selections")
    op.drop_index("ix_shot_quality_selection_shot_id", table_name="shot_quality_selections")
    op.drop_index("ix_shot_quality_selection_chain_id", table_name="shot_quality_selections")
    op.drop_table("shot_quality_selections")

    op.drop_index("ix_quality_remediation_snapshot_id", table_name="quality_remediation_decisions")
    op.drop_index("ix_quality_remediation_shot_id", table_name="quality_remediation_decisions")
    op.drop_index("ix_quality_remediation_chain_id", table_name="quality_remediation_decisions")
    op.drop_table("quality_remediation_decisions")
