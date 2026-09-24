"""composition stage integration: composition outputs

Revision ID: 0022_composition_output_artifact
Revises: 0021_audio_output_artifact
Create Date: 2026-09-12 23:55:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0022_composition_output_artifact"
down_revision: str | Sequence[str] | None = "0021_audio_output_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "composition_outputs",
        sa.Column("composition_output_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey(
                "knowledge_video_tasks.task_id",
                name="fk_composition_outputs_task_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "storyboard_snapshot_id",
            sa.String(36),
            sa.ForeignKey(
                "storyboard_snapshots.storyboard_snapshot_id",
                name="fk_composition_outputs_storyboard_snapshot_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "execution_run_id",
            sa.String(36),
            sa.ForeignKey(
                "execution_runs.execution_run_id",
                name="fk_composition_outputs_execution_run_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "audio_output_id",
            sa.String(36),
            sa.ForeignKey(
                "audio_outputs.audio_output_id",
                name="fk_composition_outputs_audio_output_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("video_path", sa.String(512), nullable=False),
        sa.Column("video_hash", sa.String(64), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("video_codec", sa.String(64), nullable=True),
        sa.Column("audio_codec", sa.String(64), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column(
            "composition_params_snapshot_json",
            EvidenceType,
            nullable=False,
            server_default="{}",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_composition_outputs_task_id",
        "composition_outputs",
        ["task_id"],
    )
    op.create_index(
        "ix_composition_outputs_snapshot_id",
        "composition_outputs",
        ["storyboard_snapshot_id"],
    )
    op.create_index(
        "ix_composition_outputs_exec_run_id",
        "composition_outputs",
        ["execution_run_id"],
    )
    op.create_index(
        "ix_composition_outputs_audio_id",
        "composition_outputs",
        ["audio_output_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_composition_outputs_audio_id", table_name="composition_outputs")
    op.drop_index("ix_composition_outputs_exec_run_id", table_name="composition_outputs")
    op.drop_index("ix_composition_outputs_snapshot_id", table_name="composition_outputs")
    op.drop_index("ix_composition_outputs_task_id", table_name="composition_outputs")
    op.drop_table("composition_outputs")
