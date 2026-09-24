"""knowledge video workflow tasks, persistent jobs, and stage executions schema

Revision ID: 0014_knowledge_video_workflow
Revises: 0013_benchmark_reports
Create Date: 2026-09-12 00:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0014_knowledge_video_workflow"
down_revision: str | Sequence[str] | None = "0013_benchmark_reports"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create knowledge_video_tasks table
    op.create_table(
        "knowledge_video_tasks",
        sa.Column("task_id", sa.String(36), primary_key=True),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("task_status", sa.String(32), nullable=False),
        sa.Column("current_stage", sa.String(32), nullable=False),
        sa.Column("workflow_policy", sa.String(32), nullable=False),
        sa.Column("target_duration", sa.Float(), nullable=False, server_default="60.0"),
        sa.Column("aspect_ratio", sa.String(16), nullable=False, server_default="16:9"),
        sa.Column("language", sa.String(16), nullable=False, server_default="zh"),
        sa.Column("waiting_reason", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_kv_tasks_status", "knowledge_video_tasks", ["task_status"])
    op.create_index("ix_kv_tasks_current_stage", "knowledge_video_tasks", ["current_stage"])
    op.create_index("ix_kv_tasks_created_at", "knowledge_video_tasks", ["created_at"])

    # 2. Create workflow_jobs table
    op.create_table(
        "workflow_jobs",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("input_artifact_revision_id", sa.String(36), nullable=True),
        sa.Column("output_artifact_revision_id", sa.String(36), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_wf_jobs_idempotency_key"),
    )
    op.create_index("ix_wf_jobs_task_id", "workflow_jobs", ["task_id"])
    op.create_index("ix_wf_jobs_stage", "workflow_jobs", ["stage"])
    op.create_index("ix_wf_jobs_status", "workflow_jobs", ["status"])
    op.create_index("ix_wf_jobs_lease_expires_at", "workflow_jobs", ["lease_expires_at"])
    op.create_index("ix_wf_jobs_idempotency_key", "workflow_jobs", ["idempotency_key"])

    # 3. Create stage_executions table
    op.create_table(
        "stage_executions",
        sa.Column("stage_execution_id", sa.String(36), primary_key=True),
        sa.Column(
            "task_id",
            sa.String(36),
            sa.ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("workflow_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("input_artifact_revision_id", sa.String(36), nullable=True),
        sa.Column("output_artifact_revision_id", sa.String(36), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_stage_exec_task_id", "stage_executions", ["task_id"])
    op.create_index("ix_stage_exec_job_id", "stage_executions", ["job_id"])
    op.create_index("ix_stage_exec_stage", "stage_executions", ["stage"])


def downgrade() -> None:
    op.drop_table("stage_executions")
    op.drop_table("workflow_jobs")
    op.drop_table("knowledge_video_tasks")
