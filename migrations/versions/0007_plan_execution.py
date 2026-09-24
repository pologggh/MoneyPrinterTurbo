"""plan execution schema

Revision ID: 0007_plan_execution
Revises: 0006_provider_lifecycle
Create Date: 2026-09-11 14:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007_plan_execution"
down_revision: str | Sequence[str] | None = "0006_provider_lifecycle"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("execution_runs") as batch_op:
        batch_op.add_column(
            sa.Column("reused_shots", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("recovery_required_shots", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("execution_runs") as batch_op:
        batch_op.drop_column("recovery_required_shots")
        batch_op.drop_column("reused_shots")
