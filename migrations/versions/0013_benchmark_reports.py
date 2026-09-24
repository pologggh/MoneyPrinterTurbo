"""benchmark reports and strategy comparison reports schema

Revision ID: 0013_benchmark_reports
Revises: 0012_benchmark_system
Create Date: 2026-09-11 23:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0013_benchmark_reports"
down_revision: str | Sequence[str] | None = "0012_benchmark_system"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create benchmark_reports table
    op.create_table(
        "benchmark_reports",
        sa.Column("report_id", sa.String(36), primary_key=True),
        sa.Column(
            "benchmark_run_id",
            sa.String(36),
            sa.ForeignKey("benchmark_runs.benchmark_run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("suite_key", sa.String(64), nullable=False),
        sa.Column("suite_version", sa.String(32), nullable=False),
        sa.Column("suite_fingerprint", sa.String(64), nullable=False),
        sa.Column("variant_key", sa.String(64), nullable=False),
        sa.Column("variant_json", EvidenceType, nullable=False),
        sa.Column("execution_mode", sa.String(32), nullable=False),
        sa.Column("metric_definition_set_version", sa.String(32), nullable=False),
        sa.Column("source_fingerprint", sa.String(64), nullable=False),
        sa.Column("metrics_json", EvidenceType, nullable=False),
        sa.Column("stage_latencies_json", EvidenceType, nullable=False),
        sa.Column("cost_summary_json", EvidenceType, nullable=False),
        sa.Column("provider_model_usage_json", EvidenceType, nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("shot_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmark_reports_run_id", "benchmark_reports", ["benchmark_run_id"])
    op.create_index("ix_benchmark_reports_suite_key", "benchmark_reports", ["suite_key"])
    op.create_index("ix_benchmark_reports_generated_at", "benchmark_reports", ["generated_at"])

    # 2. Create benchmark_comparison_reports table
    op.create_table(
        "benchmark_comparison_reports",
        sa.Column("comparison_id", sa.String(36), primary_key=True),
        sa.Column(
            "baseline_report_id",
            sa.String(36),
            sa.ForeignKey("benchmark_reports.report_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "candidate_report_id",
            sa.String(36),
            sa.ForeignKey("benchmark_reports.report_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("suite_key", sa.String(64), nullable=False),
        sa.Column("suite_version", sa.String(32), nullable=False),
        sa.Column("suite_fingerprint", sa.String(64), nullable=False),
        sa.Column("baseline_variant_key", sa.String(64), nullable=False),
        sa.Column("candidate_variant_key", sa.String(64), nullable=False),
        sa.Column("execution_mode", sa.String(32), nullable=False),
        sa.Column("metric_definition_set_version", sa.String(32), nullable=False),
        sa.Column("source_fingerprint", sa.String(64), nullable=False),
        sa.Column("metric_deltas_json", EvidenceType, nullable=False),
        sa.Column("latency_deltas_json", EvidenceType, nullable=False),
        sa.Column("cost_deltas_json", EvidenceType, nullable=False),
        sa.Column("tradeoff_summary_json", EvidenceType, nullable=False),
        sa.Column("case_drilldowns_json", EvidenceType, nullable=False),
        sa.Column("failure_breakdown_json", EvidenceType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_benchmark_comp_reports_baseline", "benchmark_comparison_reports", ["baseline_report_id"])
    op.create_index("ix_benchmark_comp_reports_candidate", "benchmark_comparison_reports", ["candidate_report_id"])
    op.create_index("ix_benchmark_comp_reports_suite_key", "benchmark_comparison_reports", ["suite_key"])
    op.create_index("ix_benchmark_comp_reports_created_at", "benchmark_comparison_reports", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_benchmark_comp_reports_created_at", table_name="benchmark_comparison_reports")
    op.drop_index("ix_benchmark_comp_reports_suite_key", table_name="benchmark_comparison_reports")
    op.drop_index("ix_benchmark_comp_reports_candidate", table_name="benchmark_comparison_reports")
    op.drop_index("ix_benchmark_comp_reports_baseline", table_name="benchmark_comparison_reports")
    op.drop_table("benchmark_comparison_reports")

    op.drop_index("ix_benchmark_reports_generated_at", table_name="benchmark_reports")
    op.drop_index("ix_benchmark_reports_suite_key", table_name="benchmark_reports")
    op.drop_index("ix_benchmark_reports_run_id", table_name="benchmark_reports")
    op.drop_table("benchmark_reports")
