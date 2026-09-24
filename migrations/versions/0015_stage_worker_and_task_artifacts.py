"""stage worker, task artifact refs, and explicit workflow job artifact linkages

Revision ID: 0015_stage_worker_and_task_artifacts
Revises: 0014_knowledge_video_workflow
Create Date: 2026-09-12 07:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0015_stage_worker_and_task_artifacts"
down_revision: str | Sequence[str] | None = "0014_knowledge_video_workflow"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create task_artifact_refs table
    op.create_table(
        "task_artifact_refs",
        sa.Column("task_artifact_ref_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", name="fk_task_artifact_refs_task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("artifact_type", sa.String(64), nullable=False),
        sa.Column("artifact_id", sa.String(64), nullable=False),
        sa.Column("artifact_version", sa.String(64), nullable=True),
        sa.Column("metadata_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_task_artifact_refs_task_id", "task_artifact_refs", ["task_id"])
    op.create_index("ix_task_artifact_refs_stage", "task_artifact_refs", ["stage"])
    op.create_index("ix_task_artifact_refs_artifact_type", "task_artifact_refs", ["artifact_type"])
    op.create_index("ix_task_artifact_refs_created_at", "task_artifact_refs", ["created_at"])

    # 2. Add explicit artifact reference columns to workflow_jobs
    with op.batch_alter_table("workflow_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "input_task_artifact_ref_id",
                sa.String(36),
                sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_wf_jobs_input_art_ref", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "output_task_artifact_ref_id",
                sa.String(36),
                sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_wf_jobs_output_art_ref", ondelete="SET NULL"),
                nullable=True,
            )
        )

    # 3. Add explicit artifact reference columns to stage_executions
    with op.batch_alter_table("stage_executions") as batch_op:
        batch_op.add_column(
            sa.Column(
                "input_task_artifact_ref_id",
                sa.String(36),
                sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_stage_exec_input_art_ref", ondelete="SET NULL"),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "output_task_artifact_ref_id",
                sa.String(36),
                sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_stage_exec_output_art_ref", ondelete="SET NULL"),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("stage_executions") as batch_op:
        batch_op.drop_column("output_task_artifact_ref_id")
        batch_op.drop_column("input_task_artifact_ref_id")

    with op.batch_alter_table("workflow_jobs") as batch_op:
        batch_op.drop_column("output_task_artifact_ref_id")
        batch_op.drop_column("input_task_artifact_ref_id")

    op.drop_index("ix_task_artifact_refs_created_at", table_name="task_artifact_refs")
    op.drop_index("ix_task_artifact_refs_artifact_type", table_name="task_artifact_refs")
    op.drop_index("ix_task_artifact_refs_stage", table_name="task_artifact_refs")
    op.drop_index("ix_task_artifact_refs_task_id", table_name="task_artifact_refs")
    op.drop_table("task_artifact_refs")
