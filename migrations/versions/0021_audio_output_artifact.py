"""audio stage integration: audio outputs

Revision ID: 0021_audio_output_artifact
Revises: 0020_shot_revision_script_segment_id
Create Date: 2026-09-12 23:45:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0021_audio_output_artifact"
down_revision: str | Sequence[str] | None = "0020_shot_revision_script_segment_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audio_outputs",
        sa.Column("audio_output_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey(
                "knowledge_video_tasks.task_id",
                name="fk_audio_outputs_task_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "script_revision_id",
            sa.String(36),
            sa.ForeignKey(
                "script_revisions.script_revision_id",
                name="fk_audio_outputs_script_revision_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "execution_run_id",
            sa.String(36),
            sa.ForeignKey(
                "execution_runs.execution_run_id",
                name="fk_audio_outputs_execution_run_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("narration_audio_path", sa.String(512), nullable=False),
        sa.Column("narration_audio_hash", sa.String(64), nullable=False),
        sa.Column("actual_narration_duration", sa.Float(), nullable=False),
        sa.Column("subtitle_path", sa.String(512), nullable=True),
        sa.Column("subtitle_hash", sa.String(64), nullable=True),
        sa.Column("bgm_path", sa.String(512), nullable=True),
        sa.Column("bgm_volume", sa.Float(), nullable=True, server_default="0.2"),
        sa.Column(
            "voice_config_snapshot_json",
            EvidenceType,
            nullable=False,
            server_default="{}",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_audio_outputs_task_id",
        "audio_outputs",
        ["task_id"],
    )
    op.create_index(
        "ix_audio_outputs_script_rev_id",
        "audio_outputs",
        ["script_revision_id"],
    )
    op.create_index(
        "ix_audio_outputs_exec_run_id",
        "audio_outputs",
        ["execution_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_audio_outputs_exec_run_id", table_name="audio_outputs")
    op.drop_index("ix_audio_outputs_script_rev_id", table_name="audio_outputs")
    op.drop_index("ix_audio_outputs_task_id", table_name="audio_outputs")
    op.drop_table("audio_outputs")
