"""persist effective retrieval mode in retrieval snapshots

Revision ID: 0025_retrieval_snapshot_effective_mode
Revises: 0024_hybrid_rag_knowledge_base
Create Date: 2026-09-19 06:15:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0025_retrieval_snapshot_effective_mode"
down_revision: str | Sequence[str] | None = "0024_hybrid_rag_knowledge_base"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if "retrieval_snapshots" in tables:
        existing_cols = [c["name"] for c in inspector.get_columns("retrieval_snapshots")]
        if "effective_retrieval_mode" not in existing_cols:
            with op.batch_alter_table("retrieval_snapshots") as batch_op:
                batch_op.add_column(
                    sa.Column(
                        "effective_retrieval_mode",
                        sa.String(32),
                        nullable=False,
                        server_default="BM25_ONLY",
                    )
                )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if "retrieval_snapshots" in tables:
        existing_cols = [c["name"] for c in inspector.get_columns("retrieval_snapshots")]
        if "effective_retrieval_mode" in existing_cols:
            with op.batch_alter_table("retrieval_snapshots") as batch_op:
                batch_op.drop_column("effective_retrieval_mode")
