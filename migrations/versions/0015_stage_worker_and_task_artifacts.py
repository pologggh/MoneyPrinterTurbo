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
    bind = op.get_bind()

    # 1. Create task_artifact_refs table
    if "task_artifact_refs" not in sa.inspect(bind).get_table_names():
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

    existing_indexes = {
        index["name"] for index in sa.inspect(bind).get_indexes("task_artifact_refs")
    }
    for index_name, columns in (
        ("ix_task_artifact_refs_task_id", ["task_id"]),
        ("ix_task_artifact_refs_stage", ["stage"]),
        ("ix_task_artifact_refs_artifact_type", ["artifact_type"]),
        ("ix_task_artifact_refs_created_at", ["created_at"]),
    ):
        if index_name not in existing_indexes:
            op.create_index(index_name, "task_artifact_refs", columns)

    # 2. Add explicit artifact reference columns to workflow_jobs
    workflow_job_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("workflow_jobs")
    }
    if not {
        "input_task_artifact_ref_id",
        "output_task_artifact_ref_id",
    }.issubset(workflow_job_columns):
        with op.batch_alter_table("workflow_jobs") as batch_op:
            if "input_task_artifact_ref_id" not in workflow_job_columns:
                batch_op.add_column(
                    sa.Column(
                        "input_task_artifact_ref_id",
                        sa.String(36),
                        sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_wf_jobs_input_art_ref", ondelete="SET NULL"),
                        nullable=True,
                    )
                )
            if "output_task_artifact_ref_id" not in workflow_job_columns:
                batch_op.add_column(
                    sa.Column(
                        "output_task_artifact_ref_id",
                        sa.String(36),
                        sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_wf_jobs_output_art_ref", ondelete="SET NULL"),
                        nullable=True,
                    )
                )

    # 3. Add explicit artifact reference columns to stage_executions
    stage_execution_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("stage_executions")
    }
    if not {
        "input_task_artifact_ref_id",
        "output_task_artifact_ref_id",
    }.issubset(stage_execution_columns):
        with op.batch_alter_table("stage_executions") as batch_op:
            if "input_task_artifact_ref_id" not in stage_execution_columns:
                batch_op.add_column(
                    sa.Column(
                        "input_task_artifact_ref_id",
                        sa.String(36),
                        sa.ForeignKey("task_artifact_refs.task_artifact_ref_id", name="fk_stage_exec_input_art_ref", ondelete="SET NULL"),
                        nullable=True,
                    )
                )
            if "output_task_artifact_ref_id" not in stage_execution_columns:
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
