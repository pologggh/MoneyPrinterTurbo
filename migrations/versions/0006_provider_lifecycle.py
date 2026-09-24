"""provider lifecycle schema

Revision ID: 0006_provider_lifecycle
Revises: 0005_shot_execution
Create Date: 2026-09-10 19:10:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006_provider_lifecycle"
down_revision: str | Sequence[str] | None = "0005_shot_execution"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. attempt_requests
    op.create_table(
        "attempt_requests",
        sa.Column("attempt_request_id", sa.String(36), primary_key=True),
        sa.Column(
            "execution_attempt_id",
            sa.String(36),
            sa.ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("generation_mode", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("sanitized_payload_json", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_attempt_request_attempt_id",
        "attempt_requests",
        ["execution_attempt_id"],
    )
    op.create_index(
        "ix_attempt_request_idempotency_key",
        "attempt_requests",
        ["idempotency_key"],
    )

    # 2. provider_receipts
    op.create_table(
        "provider_receipts",
        sa.Column("receipt_id", sa.String(36), primary_key=True),
        sa.Column(
            "execution_attempt_id",
            sa.String(36),
            sa.ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(128), nullable=False),
        sa.Column("provider_job_id", sa.String(128), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_status", sa.String(64), nullable=False),
        sa.Column(
            "sanitized_metadata_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )
    op.create_index(
        "ix_provider_receipt_attempt_id",
        "provider_receipts",
        ["execution_attempt_id"],
    )
    op.create_index(
        "ix_provider_receipt_job_id",
        "provider_receipts",
        ["provider_job_id"],
    )

    # 3. attempt_results
    op.create_table(
        "attempt_results",
        sa.Column("result_id", sa.String(36), primary_key=True),
        sa.Column(
            "execution_attempt_id",
            sa.String(36),
            sa.ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("outcome", sa.String(64), nullable=False),
        sa.Column("provider_status", sa.String(64), nullable=True),
        sa.Column("asset_reference", sa.String(512), nullable=True),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("diagnostic_summary", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_attempt_result_attempt_id",
        "attempt_results",
        ["execution_attempt_id"],
    )
    op.create_index(
        "ix_attempt_result_outcome",
        "attempt_results",
        ["outcome"],
    )

    # 4. execution_transitions
    op.create_table(
        "execution_transitions",
        sa.Column("transition_id", sa.String(36), primary_key=True),
        sa.Column("entity_type", sa.String(32), nullable=False),
        sa.Column("entity_id", sa.String(36), nullable=False),
        sa.Column("from_state", sa.String(64), nullable=False),
        sa.Column("to_state", sa.String(64), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_exec_transition_entity",
        "execution_transitions",
        ["entity_type", "entity_id"],
    )


def downgrade() -> None:
    op.drop_table("execution_transitions")
    op.drop_table("attempt_results")
    op.drop_table("provider_receipts")
    op.drop_table("attempt_requests")
