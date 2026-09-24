"""benchmark runs and case results schema for reproducible benchmark execution

Revision ID: 0012_benchmark_system
Revises: 0011_trace_system
Create Date: 2026-09-11 23:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0012_benchmark_system"
down_revision: str | Sequence[str] | None = "0011_trace_system"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create benchmark_runs table
    op.create_table(
        "benchmark_runs",
        sa.Column("benchmark_run_id", sa.String(36), primary_key=True),
        sa.Column("benchmark_suite_id", sa.String(64), nullable=False),
        sa.Column("suite_key", sa.String(64), nullable=False),
        sa.Column("suite_version", sa.String(32), nullable=False),
        sa.Column("suite_fingerprint", sa.String(64), nullable=False),
        sa.Column("variant_json", EvidenceType, nullable=False),
        sa.Column("execution_mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("application_version", sa.String(64), nullable=False, server_default="1.3.6"),
        sa.Column("git_commit", sa.String(64), nullable=True),
        sa.Column("total_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmark_runs_suite_key", "benchmark_runs", ["suite_key"])
    op.create_index("ix_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index("ix_benchmark_runs_started_at", "benchmark_runs", ["started_at"])

    # 2. Create benchmark_case_results table
    op.create_table(
        "benchmark_case_results",
        sa.Column("benchmark_case_result_id", sa.String(36), primary_key=True),
        sa.Column(
            "benchmark_run_id",
            sa.String(36),
            sa.ForeignKey("benchmark_runs.benchmark_run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("benchmark_case_id", sa.String(64), nullable=False),
        sa.Column("case_key", sa.String(64), nullable=False),
        sa.Column("case_version", sa.String(32), nullable=False),
        sa.Column("case_fingerprint", sa.String(64), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("failure_stage", sa.String(64), nullable=True),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("production_result_refs_json", EvidenceType, nullable=False),
        sa.Column("attributes_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmark_case_results_run_id", "benchmark_case_results", ["benchmark_run_id"])
    op.create_index("ix_benchmark_case_results_case_key", "benchmark_case_results", ["case_key"])
    op.create_index("ix_benchmark_case_results_trace_id", "benchmark_case_results", ["trace_id"])
    op.create_index("ix_benchmark_case_results_status", "benchmark_case_results", ["status"])


def downgrade() -> None:
    op.drop_index("ix_benchmark_case_results_status", table_name="benchmark_case_results")
    op.drop_index("ix_benchmark_case_results_trace_id", table_name="benchmark_case_results")
    op.drop_index("ix_benchmark_case_results_case_key", table_name="benchmark_case_results")
    op.drop_index("ix_benchmark_case_results_run_id", table_name="benchmark_case_results")
    op.drop_table("benchmark_case_results")

    op.drop_index("ix_benchmark_runs_started_at", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_status", table_name="benchmark_runs")
    op.drop_index("ix_benchmark_runs_suite_key", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
