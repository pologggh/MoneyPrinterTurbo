"""trace roots and events schema for end-to-end business decision tracing

Revision ID: 0011_trace_system
Revises: 0010_quality_remediation
Create Date: 2026-09-11 22:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

EvidenceType = sa.JSON().with_variant(JSONB(), "postgresql")

# revision identifiers, used by Alembic.
revision: str = "0011_trace_system"
down_revision: str | Sequence[str] | None = "0010_quality_remediation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Create trace_roots table
    op.create_table(
        "trace_roots",
        sa.Column("trace_id", sa.String(36), primary_key=True),
        sa.Column("root_type", sa.String(64), nullable=False, server_default="PRODUCTION_WORKFLOW"),
        sa.Column("root_reference_id", sa.String(64), nullable=True),
        sa.Column("metadata_version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trace_roots_created_at", "trace_roots", ["created_at"])

    # 2. Create trace_events table
    op.create_table(
        "trace_events",
        sa.Column("trace_event_id", sa.String(36), primary_key=True),
        sa.Column(
            "trace_id",
            sa.String(36),
            sa.ForeignKey("trace_roots.trace_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parent_event_id", sa.String(36), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("event_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("context_json", EvidenceType, nullable=False),
        sa.Column("attributes_json", EvidenceType, nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("execution_run_id", sa.String(36), nullable=True),
        sa.Column("shot_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_trace_events_trace_id", "trace_events", ["trace_id"])
    op.create_index("ix_trace_events_parent_event_id", "trace_events", ["parent_event_id"])
    op.create_index("ix_trace_events_event_type", "trace_events", ["event_type"])
    op.create_index("ix_trace_events_execution_run_id", "trace_events", ["execution_run_id"])
    op.create_index("ix_trace_events_shot_id", "trace_events", ["shot_id"])
    op.create_index("ix_trace_events_created_at", "trace_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_trace_events_created_at", table_name="trace_events")
    op.drop_index("ix_trace_events_shot_id", table_name="trace_events")
    op.drop_index("ix_trace_events_execution_run_id", table_name="trace_events")
    op.drop_index("ix_trace_events_event_type", table_name="trace_events")
    op.drop_index("ix_trace_events_parent_event_id", table_name="trace_events")
    op.drop_index("ix_trace_events_trace_id", table_name="trace_events")
    op.drop_table("trace_events")

    op.drop_index("ix_trace_roots_created_at", table_name="trace_roots")
    op.drop_table("trace_roots")
