"""delivery stage integration: delivery manifests

Revision ID: 0023_delivery_manifest_artifact
Revises: 0022_composition_output_artifact
Create Date: 2026-09-12 23:59:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0023_delivery_manifest_artifact"
down_revision: str | Sequence[str] | None = "0022_composition_output_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "delivery_manifests",
        sa.Column("delivery_manifest_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey(
                "knowledge_video_tasks.task_id",
                name="fk_delivery_manifests_task_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "composition_output_id",
            sa.String(36),
            sa.ForeignKey(
                "composition_outputs.composition_output_id",
                name="fk_delivery_manifests_composition_output_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "evaluation_snapshot_id",
            sa.String(36),
            sa.ForeignKey(
                "evaluation_snapshots.evaluation_snapshot_id",
                name="fk_delivery_manifests_evaluation_snapshot_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("final_video_path", sa.String(512), nullable=False),
        sa.Column("final_video_hash", sa.String(64), nullable=False),
        sa.Column("final_video_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("subtitle_path", sa.String(512), nullable=True),
        sa.Column("subtitle_hash", sa.String(64), nullable=True),
        sa.Column("source_report_path", sa.String(512), nullable=False),
        sa.Column("source_report_hash", sa.String(64), nullable=False),
        sa.Column("execution_report_path", sa.String(512), nullable=False),
        sa.Column("execution_report_hash", sa.String(64), nullable=False),
        sa.Column(
            "delivery_params_snapshot_json",
            EvidenceType,
            nullable=False,
            server_default="{}",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_delivery_manifests_task_id",
        "delivery_manifests",
        ["task_id"],
    )
    op.create_index(
        "ix_delivery_manifests_composition_id",
        "delivery_manifests",
        ["composition_output_id"],
    )
    op.create_index(
        "ix_delivery_manifests_eval_snapshot_id",
        "delivery_manifests",
        ["evaluation_snapshot_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_delivery_manifests_eval_snapshot_id", table_name="delivery_manifests")
    op.drop_index("ix_delivery_manifests_composition_id", table_name="delivery_manifests")
    op.drop_index("ix_delivery_manifests_task_id", table_name="delivery_manifests")
    op.drop_table("delivery_manifests")
