"""evaluation contracts schema

Revision ID: 0008_evaluation_contracts
Revises: 0007_plan_execution
Create Date: 2026-09-11 15:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008_evaluation_contracts"
down_revision: str | Sequence[str] | None = "0007_plan_execution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EvidenceType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    # 1. evaluation_targets
    op.create_table(
        "evaluation_targets",
        sa.Column("evaluation_target_id", sa.String(36), primary_key=True),
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
            "shot_asset_version_id",
            sa.String(36),
            sa.ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("evidence_refs_snapshot", EvidenceType, nullable=False),
        sa.Column("context_refs", EvidenceType, nullable=False),
        sa.Column("target_version", sa.String(32), nullable=False, server_default="v1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_eval_targets_shot_id",
        "evaluation_targets",
        ["shot_id"],
    )
    op.create_index(
        "ix_eval_targets_shot_rev_id",
        "evaluation_targets",
        ["shot_revision_id"],
    )
    op.create_index(
        "ix_eval_targets_asset_ver_id",
        "evaluation_targets",
        ["shot_asset_version_id"],
    )

    # 2. dimension_evaluation_results
    op.create_table(
        "dimension_evaluation_results",
        sa.Column("dimension_result_id", sa.String(36), primary_key=True),
        sa.Column(
            "evaluation_target_id",
            sa.String(36),
            sa.ForeignKey("evaluation_targets.evaluation_target_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dimension", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("reason_codes", EvidenceType, nullable=False),
        sa.Column("concise_summary", sa.Text(), nullable=True),
        sa.Column("evaluator_version", sa.String(64), nullable=False, server_default="v1.0"),
        sa.Column(
            "dimension_semantics_version",
            sa.String(64),
            nullable=False,
            server_default="v1.0",
        ),
        sa.Column("dependency_fingerprint", sa.String(128), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_dim_eval_results_target_id",
        "dimension_evaluation_results",
        ["evaluation_target_id"],
    )
    op.create_index(
        "ix_dim_eval_results_dimension",
        "dimension_evaluation_results",
        ["dimension"],
    )
    op.create_index(
        "ix_dim_eval_results_status",
        "dimension_evaluation_results",
        ["status"],
    )

    # 3. evaluation_snapshots
    op.create_table(
        "evaluation_snapshots",
        sa.Column("evaluation_snapshot_id", sa.String(36), primary_key=True),
        sa.Column(
            "evaluation_target_id",
            sa.String(36),
            sa.ForeignKey("evaluation_targets.evaluation_target_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(64), nullable=False, server_default="v1.0"),
        sa.Column("evaluator_version", sa.String(64), nullable=False, server_default="v1.0"),
        sa.Column("summary_reason_codes", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_eval_snapshots_target_id",
        "evaluation_snapshots",
        ["evaluation_target_id"],
    )
    op.create_index(
        "ix_eval_snapshots_decision",
        "evaluation_snapshots",
        ["decision"],
    )

    # 4. evaluation_snapshot_dimension_results
    op.create_table(
        "evaluation_snapshot_dimension_results",
        sa.Column(
            "evaluation_snapshot_id",
            sa.String(36),
            sa.ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "dimension_result_id",
            sa.String(36),
            sa.ForeignKey(
                "dimension_evaluation_results.dimension_result_id", ondelete="RESTRICT"
            ),
            primary_key=True,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_eval_snap_dim_res_dim_id",
        "evaluation_snapshot_dimension_results",
        ["dimension_result_id"],
    )


def downgrade() -> None:
    op.drop_table("evaluation_snapshot_dimension_results")
    op.drop_table("evaluation_snapshots")
    op.drop_table("dimension_evaluation_results")
    op.drop_table("evaluation_targets")
