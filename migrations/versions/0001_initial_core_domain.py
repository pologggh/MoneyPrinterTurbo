"""initial core domain schema

Revision ID: 0001_initial_core_domain
Revises: 
Create Date: 2026-09-10 17:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_core_domain"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

evidence_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    # 1. content_plan_revisions
    op.create_table(
        "content_plan_revisions",
        sa.Column("content_plan_revision_id", sa.String(36), primary_key=True),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("overall_target_duration", sa.Float(), nullable=False),
        sa.Column("global_retrieval_snapshot_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 2. content_beats
    op.create_table(
        "content_beats",
        sa.Column("beat_id", sa.String(36), primary_key=True),
        sa.Column(
            "content_plan_revision_id",
            sa.String(36),
            sa.ForeignKey("content_plan_revisions.content_plan_revision_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("beat_lineage_id", sa.String(36), nullable=False),
        sa.Column("beat_type", sa.String(32), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("target_duration", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("evidence_refs", evidence_type, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "content_plan_revision_id",
            "order",
            name="uq_content_beats_revision_order",
        ),
    )
    op.create_index("ix_content_beats_revision_id", "content_beats", ["content_plan_revision_id"])
    op.create_index("ix_content_beats_lineage_id", "content_beats", ["beat_lineage_id"])

    # 3. shots
    op.create_table(
        "shots",
        sa.Column("shot_id", sa.String(36), primary_key=True),
        sa.Column("beat_lineage_id", sa.String(36), nullable=False),
        sa.Column("local_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_shots_beat_lineage_id", "shots", ["beat_lineage_id"])
    op.create_index("ix_shots_lineage_local_order", "shots", ["beat_lineage_id", "local_order"])

    # 4. shot_revisions
    op.create_table(
        "shot_revisions",
        sa.Column("shot_revision_id", sa.String(36), primary_key=True),
        sa.Column(
            "shot_id",
            sa.String(36),
            sa.ForeignKey("shots.shot_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("beat_lineage_id", sa.String(36), nullable=False),
        sa.Column(
            "created_from_beat_instance_id",
            sa.String(36),
            sa.ForeignKey("content_beats.beat_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("narration", sa.Text(), nullable=False),
        sa.Column("target_duration", sa.Float(), nullable=False),
        sa.Column("visual_goal", sa.Text(), nullable=False),
        sa.Column("visual_type", sa.String(32), nullable=False),
        sa.Column("scene_description", sa.Text(), nullable=False),
        sa.Column("generation_prompt", sa.Text(), nullable=False),
        sa.Column("camera_movement", sa.String(255), nullable=False),
        sa.Column("evidence_refs", evidence_type, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "shot_id",
            "revision_number",
            name="uq_shot_revisions_shot_id_revision_number",
        ),
    )
    op.create_index("ix_shot_revisions_shot_id", "shot_revisions", ["shot_id"])
    op.create_index("ix_shot_revisions_beat_lineage_id", "shot_revisions", ["beat_lineage_id"])
    op.create_index(
        "ix_shot_revisions_created_from_beat",
        "shot_revisions",
        ["created_from_beat_instance_id"],
    )

    # 5. storyboard_snapshots
    op.create_table(
        "storyboard_snapshots",
        sa.Column("storyboard_snapshot_id", sa.String(36), primary_key=True),
        sa.Column(
            "content_plan_revision_id",
            sa.String(36),
            sa.ForeignKey("content_plan_revisions.content_plan_revision_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("snapshot_state", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_storyboard_snapshots_plan_rev_id",
        "storyboard_snapshots",
        ["content_plan_revision_id"],
    )

    # 6. storyboard_snapshot_shot_revisions
    op.create_table(
        "storyboard_snapshot_shot_revisions",
        sa.Column(
            "storyboard_snapshot_id",
            sa.String(36),
            sa.ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "shot_revision_id",
            sa.String(36),
            sa.ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_sb_snap_shot_rev_snapshot_id",
        "storyboard_snapshot_shot_revisions",
        ["storyboard_snapshot_id"],
    )
    op.create_index(
        "ix_sb_snap_shot_rev_shot_revision_id",
        "storyboard_snapshot_shot_revisions",
        ["shot_revision_id"],
    )


def downgrade() -> None:
    op.drop_table("storyboard_snapshot_shot_revisions")
    op.drop_table("storyboard_snapshots")
    op.drop_table("shot_revisions")
    op.drop_table("shots")
    op.drop_table("content_beats")
    op.drop_table("content_plan_revisions")
