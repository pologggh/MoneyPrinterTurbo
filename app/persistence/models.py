from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# JSON column with native PostgreSQL JSONB support
EvidenceType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class ContentPlanRevisionORM(Base):
    __tablename__ = "content_plan_revisions"

    content_plan_revision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    overall_target_duration: Mapped[float] = mapped_column(Float, nullable=False)
    global_retrieval_snapshot_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    beats: Mapped[list["ContentBeatORM"]] = relationship(
        "ContentBeatORM",
        back_populates="content_plan_revision",
        cascade="all, delete-orphan",
        order_by="ContentBeatORM.order",
    )


class ContentBeatORM(Base):
    __tablename__ = "content_beats"

    beat_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    content_plan_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "content_plan_revisions.content_plan_revision_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    beat_lineage_id: Mapped[str] = mapped_column(String(36), nullable=False)
    beat_type: Mapped[str] = mapped_column(String(32), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    intent: Mapped[str] = mapped_column(Text, nullable=False)
    target_duration: Mapped[float] = mapped_column(Float, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_refs: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    content_plan_revision: Mapped["ContentPlanRevisionORM"] = relationship(
        "ContentPlanRevisionORM",
        back_populates="beats",
    )

    __table_args__ = (
        UniqueConstraint(
            "content_plan_revision_id",
            "order",
            name="uq_content_beats_revision_order",
        ),
        Index("ix_content_beats_revision_id", "content_plan_revision_id"),
        Index("ix_content_beats_lineage_id", "beat_lineage_id"),
    )


class ShotORM(Base):
    __tablename__ = "shots"

    shot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    beat_lineage_id: Mapped[str] = mapped_column(String(36), nullable=False)
    local_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    revisions: Mapped[list["ShotRevisionORM"]] = relationship(
        "ShotRevisionORM",
        back_populates="shot",
        cascade="all, delete-orphan",
        order_by="ShotRevisionORM.revision_number",
    )

    __table_args__ = (
        Index("ix_shots_beat_lineage_id", "beat_lineage_id"),
        Index("ix_shots_lineage_local_order", "beat_lineage_id", "local_order"),
    )


class ShotRevisionORM(Base):
    __tablename__ = "shot_revisions"

    shot_revision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    beat_lineage_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_from_beat_instance_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("content_beats.beat_id", ondelete="RESTRICT"),
        nullable=False,
    )
    narration: Mapped[str] = mapped_column(Text, nullable=False)
    target_duration: Mapped[float] = mapped_column(Float, nullable=False)
    visual_goal: Mapped[str] = mapped_column(Text, nullable=False)
    visual_type: Mapped[str] = mapped_column(String(32), nullable=False)
    scene_description: Mapped[str] = mapped_column(Text, nullable=False)
    generation_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    camera_movement: Mapped[str] = mapped_column(String(255), nullable=False)
    evidence_refs: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    shot: Mapped["ShotORM"] = relationship("ShotORM", back_populates="revisions")

    __table_args__ = (
        UniqueConstraint(
            "shot_id",
            "revision_number",
            name="uq_shot_revisions_shot_id_revision_number",
        ),
        Index("ix_shot_revisions_shot_id", "shot_id"),
        Index("ix_shot_revisions_beat_lineage_id", "beat_lineage_id"),
        Index("ix_shot_revisions_created_from_beat", "created_from_beat_instance_id"),
    )


class StoryboardSnapshotORM(Base):
    __tablename__ = "storyboard_snapshots"

    storyboard_snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    content_plan_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "content_plan_revisions.content_plan_revision_id", ondelete="RESTRICT"
        ),
        nullable=False,
    )
    snapshot_state: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    snapshot_revisions: Mapped[list["StoryboardSnapshotShotRevisionORM"]] = (
        relationship(
            "StoryboardSnapshotShotRevisionORM",
            back_populates="snapshot",
            cascade="all, delete-orphan",
            order_by="StoryboardSnapshotShotRevisionORM.sequence",
        )
    )

    __table_args__ = (
        Index("ix_storyboard_snapshots_plan_rev_id", "content_plan_revision_id"),
    )


class StoryboardSnapshotShotRevisionORM(Base):
    __tablename__ = "storyboard_snapshot_shot_revisions"

    storyboard_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="CASCADE"),
        primary_key=True,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    snapshot: Mapped["StoryboardSnapshotORM"] = relationship(
        "StoryboardSnapshotORM",
        back_populates="snapshot_revisions",
    )
    shot_revision: Mapped["ShotRevisionORM"] = relationship("ShotRevisionORM")

    __table_args__ = (
        Index("ix_sb_snap_shot_rev_snapshot_id", "storyboard_snapshot_id"),
        Index("ix_sb_snap_shot_rev_shot_revision_id", "shot_revision_id"),
    )


class StoryboardApprovalRecordORM(Base):
    __tablename__ = "storyboard_approval_records"

    storyboard_approval_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_draft_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    approved_storyboard_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    content_plan_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "content_plan_revisions.content_plan_revision_id", ondelete="RESTRICT"
        ),
        nullable=False,
    )
    exact_shot_revision_ids: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    approved_by: Mapped[str] = mapped_column(
        String(128), nullable=False, default="local_user"
    )
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_sb_approval_source_draft", "source_draft_snapshot_id"),
        Index("ix_sb_approval_approved_snapshot", "approved_storyboard_snapshot_id"),
        Index("ix_sb_approval_plan_revision", "content_plan_revision_id"),
    )


class AssetRoutePlanORM(Base):
    __tablename__ = "asset_route_plans"

    asset_route_plan_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    storyboard_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    content_plan_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "content_plan_revisions.content_plan_revision_id", ondelete="RESTRICT"
        ),
        nullable=False,
    )
    routing_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    routing_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_selection_mode: Mapped[str] = mapped_column(
        String(64), nullable=False, default="AUTO"
    )
    selected_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    selected_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="READY")
    total_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    routed_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blocked_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    entries: Mapped[list["ShotRoutePlanEntryORM"]] = relationship(
        "ShotRoutePlanEntryORM",
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="ShotRoutePlanEntryORM.sequence",
    )

    __table_args__ = (
        Index("ix_asset_route_plan_snapshot_id", "storyboard_snapshot_id"),
        Index("ix_asset_route_plan_status", "status"),
    )


class ShotRoutePlanEntryORM(Base):
    __tablename__ = "shot_route_plan_entries"

    entry_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_route_plan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("asset_route_plans.asset_route_plan_id", ondelete="CASCADE"),
        nullable=False,
    )
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    beat_lineage_id: Mapped[str] = mapped_column(String(36), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_visual_type: Mapped[str] = mapped_column(String(64), nullable=False)
    route_status: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_routing_request_json: Mapped[str] = mapped_column(Text, nullable=False)
    route_decision_json: Mapped[str] = mapped_column(Text, nullable=False)

    plan: Mapped["AssetRoutePlanORM"] = relationship(
        "AssetRoutePlanORM",
        back_populates="entries",
    )

    __table_args__ = (
        Index("ix_shot_route_entry_plan_id", "asset_route_plan_id"),
        Index("ix_shot_route_entry_shot_rev_id", "shot_revision_id"),
    )


class ExecutionRunORM(Base):
    __tablename__ = "execution_runs"

    execution_run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_route_plan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("asset_route_plans.asset_route_plan_id", ondelete="RESTRICT"),
        nullable=False,
    )
    storyboard_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("storyboard_snapshots.storyboard_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    total_shots: Mapped[int] = mapped_column(Integer, nullable=False)
    succeeded_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reused_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recovery_required_shots: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    attempts: Mapped[list["ExecutionAttemptORM"]] = relationship(
        "ExecutionAttemptORM",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_execution_run_plan_id", "asset_route_plan_id"),
        Index("ix_execution_run_status", "status"),
    )


class ExecutionAttemptORM(Base):
    __tablename__ = "execution_attempts"

    execution_attempt_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_runs.execution_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_provider_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped["ExecutionRunORM"] = relationship(
        "ExecutionRunORM",
        back_populates="attempts",
    )

    __table_args__ = (
        Index("ix_exec_attempt_run_id", "execution_run_id"),
        Index("ix_exec_attempt_shot_rev_id", "shot_revision_id"),
        Index("ix_exec_attempt_status", "status"),
    )


class ShotAssetVersionORM(Base):
    __tablename__ = "shot_asset_versions"

    shot_asset_version_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    execution_attempt_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_attempts.execution_attempt_id", ondelete="RESTRICT"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_shot_asset_version_shot_rev_id", "shot_revision_id"),
        Index("ix_shot_asset_version_attempt_id", "execution_attempt_id"),
        Index("ix_shot_asset_version_hash", "file_hash"),
    )


class ShotExecutionORM(Base):
    __tablename__ = "shot_executions"

    shot_execution_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_runs.execution_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    current_candidate_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    attempt_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    produced_asset_version_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    unconfirmed_remote_task_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    route_decision_json: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_shot_execution_run_id", "execution_run_id"),
        Index("ix_shot_execution_shot_rev_id", "shot_revision_id"),
        Index("ix_shot_execution_status", "status"),
    )


class AttemptRequestORM(Base):
    __tablename__ = "attempt_requests"

    attempt_request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_attempt_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    sanitized_payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_attempt_request_attempt_id", "execution_attempt_id"),
        Index("ix_attempt_request_idempotency_key", "idempotency_key"),
    )


class ProviderReceiptORM(Base):
    __tablename__ = "provider_receipts"

    receipt_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_attempt_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    provider_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    accepted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    provider_status: Mapped[str] = mapped_column(String(64), nullable=False)
    sanitized_metadata_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )

    __table_args__ = (
        Index("ix_provider_receipt_attempt_id", "execution_attempt_id"),
        Index("ix_provider_receipt_job_id", "provider_job_id"),
    )


class AttemptResultORM(Base):
    __tablename__ = "attempt_results"

    result_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    execution_attempt_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("execution_attempts.execution_attempt_id", ondelete="CASCADE"),
        nullable=False,
    )
    outcome: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_reference: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnostic_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_attempt_result_attempt_id", "execution_attempt_id"),
        Index("ix_attempt_result_outcome", "outcome"),
    )


class ExecutionTransitionORM(Base):
    __tablename__ = "execution_transitions"

    transition_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    from_state: Mapped[str] = mapped_column(String(64), nullable=False)
    to_state: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_exec_transition_entity", "entity_type", "entity_id"),
    )


class EvaluationTargetORM(Base):
    __tablename__ = "evaluation_targets"

    evaluation_target_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    shot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shots.shot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_revision_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_revisions.shot_revision_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_asset_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    evidence_refs_snapshot: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    context_refs: Mapped[list[dict]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    target_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="v1.0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_eval_targets_shot_id", "shot_id"),
        Index("ix_eval_targets_shot_rev_id", "shot_revision_id"),
        Index("ix_eval_targets_asset_ver_id", "shot_asset_version_id"),
    )


class DimensionEvaluationResultORM(Base):
    __tablename__ = "dimension_evaluation_results"

    dimension_result_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evaluation_target_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_targets.evaluation_target_id", ondelete="RESTRICT"),
        nullable=False,
    )
    dimension: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_codes: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    concise_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="v1.0"
    )
    dimension_semantics_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="v1.0"
    )
    dependency_fingerprint: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_dim_eval_results_target_id", "evaluation_target_id"),
        Index("ix_dim_eval_results_dimension", "dimension"),
        Index("ix_dim_eval_results_status", "status"),
    )


class EvaluationSnapshotORM(Base):
    __tablename__ = "evaluation_snapshots"

    evaluation_snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evaluation_target_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_targets.evaluation_target_id", ondelete="RESTRICT"),
        nullable=False,
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="v1.0"
    )
    evaluator_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="v1.0"
    )
    summary_reason_codes: Mapped[list[str]] = mapped_column(
        EvidenceType, nullable=False, default=list
    )
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    dimension_associations: Mapped[list["EvaluationSnapshotDimensionResultORM"]] = (
        relationship(
            "EvaluationSnapshotDimensionResultORM",
            back_populates="snapshot",
            cascade="all, delete-orphan",
            order_by="EvaluationSnapshotDimensionResultORM.sequence",
        )
    )

    __table_args__ = (
        Index("ix_eval_snapshots_target_id", "evaluation_target_id"),
        Index("ix_eval_snapshots_decision", "decision"),
    )


class EvaluationSnapshotDimensionResultORM(Base):
    __tablename__ = "evaluation_snapshot_dimension_results"

    evaluation_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="CASCADE"),
        primary_key=True,
    )
    dimension_result_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "dimension_evaluation_results.dimension_result_id", ondelete="RESTRICT"
        ),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    snapshot: Mapped["EvaluationSnapshotORM"] = relationship(
        "EvaluationSnapshotORM", back_populates="dimension_associations"
    )
    dimension_result: Mapped["DimensionEvaluationResultORM"] = relationship(
        "DimensionEvaluationResultORM"
    )

    __table_args__ = (
        Index("ix_eval_snap_dim_res_dim_id", "dimension_result_id"),
    )


class CompositionPreviewORM(Base):
    __tablename__ = "composition_previews"

    composition_preview_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    shot_asset_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    render_context_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    preview_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    preview_policy_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="composition-preview-v1"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ix_comp_preview_shot_asset_version_id", "shot_asset_version_id"),
        Index(
            "ix_comp_preview_fingerprint",
            "shot_asset_version_id",
            "render_context_fingerprint",
            "preview_policy_version",
        ),
    )


class QualityRemediationDecisionORM(Base):
    __tablename__ = "quality_remediation_decisions"

    remediation_decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    quality_chain_id: Mapped[str] = mapped_column(String(36), nullable=False)
    evaluation_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    shot_id: Mapped[str] = mapped_column(String(36), nullable=False)
    shot_revision_id: Mapped[str] = mapped_column(String(36), nullable=False)
    shot_asset_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    reason_codes: Mapped[list[str]] = mapped_column(EvidenceType, nullable=False, default=list)
    remediation_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_attempt_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    selected_route_candidate_json: Mapped[dict | None] = mapped_column(EvidenceType, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_quality_remediation_chain_id", "quality_chain_id"),
        Index("ix_quality_remediation_shot_id", "shot_id"),
        Index("ix_quality_remediation_snapshot_id", "evaluation_snapshot_id"),
    )


class ShotQualitySelectionORM(Base):
    __tablename__ = "shot_quality_selections"

    quality_selection_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    quality_chain_id: Mapped[str] = mapped_column(String(36), nullable=False)
    shot_id: Mapped[str] = mapped_column(String(36), nullable=False)
    shot_revision_id: Mapped[str] = mapped_column(String(36), nullable=False)
    shot_asset_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("shot_asset_versions.shot_asset_version_id", ondelete="RESTRICT"),
        nullable=False,
    )
    evaluation_snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evaluation_snapshots.evaluation_snapshot_id", ondelete="RESTRICT"),
        nullable=False,
    )
    selection_source: Mapped[str] = mapped_column(String(64), nullable=False, default="EVALUATION_PASS")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_shot_quality_selection_chain_id", "quality_chain_id"),
        Index("ix_shot_quality_selection_shot_id", "shot_id"),
        Index("ix_shot_quality_selection_asset_id", "shot_asset_version_id"),
    )


class TraceRootORM(Base):
    __tablename__ = "trace_roots"

    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    root_type: Mapped[str] = mapped_column(String(64), nullable=False, default="PRODUCTION_WORKFLOW")
    root_reference_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_trace_roots_created_at", "created_at"),
    )


class TraceEventORM(Base):
    __tablename__ = "trace_events"

    trace_event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("trace_roots.trace_id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    event_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    attributes_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)

    execution_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    shot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_trace_events_trace_id", "trace_id"),
        Index("ix_trace_events_parent_event_id", "parent_event_id"),
        Index("ix_trace_events_event_type", "event_type"),
        Index("ix_trace_events_execution_run_id", "execution_run_id"),
        Index("ix_trace_events_shot_id", "shot_id"),
        Index("ix_trace_events_created_at", "created_at"),
    )


class BenchmarkRunORM(Base):
    __tablename__ = "benchmark_runs"

    benchmark_run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    benchmark_suite_id: Mapped[str] = mapped_column(String(64), nullable=False)
    suite_key: Mapped[str] = mapped_column(String(64), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(32), nullable=False)
    suite_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    application_version: Mapped[str] = mapped_column(String(64), nullable=False, default="1.3.6")
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_benchmark_runs_suite_key", "suite_key"),
        Index("ix_benchmark_runs_status", "status"),
        Index("ix_benchmark_runs_started_at", "started_at"),
    )


class BenchmarkCaseResultORM(Base):
    __tablename__ = "benchmark_case_results"

    benchmark_case_result_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    benchmark_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runs.benchmark_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    benchmark_case_id: Mapped[str] = mapped_column(String(64), nullable=False)
    case_key: Mapped[str] = mapped_column(String(64), nullable=False)
    case_version: Mapped[str] = mapped_column(String(32), nullable=False)
    case_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    failure_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    production_result_refs_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    attributes_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_benchmark_case_results_run_id", "benchmark_run_id"),
        Index("ix_benchmark_case_results_case_key", "case_key"),
        Index("ix_benchmark_case_results_trace_id", "trace_id"),
        Index("ix_benchmark_case_results_status", "status"),
    )


class BenchmarkReportORM(Base):
    __tablename__ = "benchmark_reports"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    benchmark_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_runs.benchmark_run_id", ondelete="CASCADE"),
        nullable=False,
    )
    suite_key: Mapped[str] = mapped_column(String(64), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(32), nullable=False)
    suite_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_key: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_definition_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    stage_latencies_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    cost_summary_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    provider_model_usage_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_benchmark_reports_run_id", "benchmark_run_id"),
        Index("ix_benchmark_reports_suite_key", "suite_key"),
        Index("ix_benchmark_reports_generated_at", "generated_at"),
    )


class BenchmarkComparisonReportORM(Base):
    __tablename__ = "benchmark_comparison_reports"

    comparison_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    baseline_report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_reports.report_id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_report_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("benchmark_reports.report_id", ondelete="CASCADE"),
        nullable=False,
    )
    suite_key: Mapped[str] = mapped_column(String(64), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(32), nullable=False)
    suite_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    baseline_variant_key: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_variant_key: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_definition_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_deltas_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    latency_deltas_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    cost_deltas_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    tradeoff_summary_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    case_drilldowns_json: Mapped[list] = mapped_column(EvidenceType, nullable=False)
    failure_breakdown_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_benchmark_comp_reports_baseline", "baseline_report_id"),
        Index("ix_benchmark_comp_reports_candidate", "candidate_report_id"),
        Index("ix_benchmark_comp_reports_suite_key", "suite_key"),
        Index("ix_benchmark_comp_reports_created_at", "created_at"),
    )


class KnowledgeVideoTaskORM(Base):
    __tablename__ = "knowledge_video_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    task_status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(32), nullable=False)
    workflow_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    target_duration: Mapped[float] = mapped_column(Float, nullable=False, default=60.0)
    aspect_ratio: Mapped[str] = mapped_column(String(16), nullable=False, default="16:9")
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="zh")
    waiting_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_kv_tasks_status", "task_status"),
        Index("ix_kv_tasks_current_stage", "current_stage"),
        Index("ix_kv_tasks_created_at", "created_at"),
    )


class WorkflowJobORM(Base):
    __tablename__ = "workflow_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_artifact_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    output_artifact_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    input_task_artifact_ref_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("task_artifact_refs.task_artifact_ref_id", ondelete="SET NULL"),
        nullable=True,
    )
    output_task_artifact_ref_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("task_artifact_refs.task_artifact_ref_id", ondelete="SET NULL"),
        nullable=True,
    )
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_wf_jobs_task_id", "task_id"),
        Index("ix_wf_jobs_stage", "stage"),
        Index("ix_wf_jobs_status", "status"),
        Index("ix_wf_jobs_lease_expires_at", "lease_expires_at"),
        Index("ix_wf_jobs_idempotency_key", "idempotency_key"),
    )


class StageExecutionORM(Base):
    __tablename__ = "stage_executions"

    stage_execution_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workflow_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_artifact_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    output_artifact_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    input_task_artifact_ref_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("task_artifact_refs.task_artifact_ref_id", ondelete="SET NULL"),
        nullable=True,
    )
    output_task_artifact_ref_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("task_artifact_refs.task_artifact_ref_id", ondelete="SET NULL"),
        nullable=True,
    )
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_stage_exec_task_id", "task_id"),
        Index("ix_stage_exec_job_id", "job_id"),
        Index("ix_stage_exec_stage", "stage"),
    )


class TaskArtifactRefORM(Base):
    __tablename__ = "task_artifact_refs"

    task_artifact_ref_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_task_artifact_refs_task_id", "task_id"),
        Index("ix_task_artifact_refs_stage", "stage"),
        Index("ix_task_artifact_refs_artifact_type", "artifact_type"),
        Index("ix_task_artifact_refs_created_at", "created_at"),
    )


class SourceDocumentORM(Base):
    __tablename__ = "source_documents"

    source_document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_locator: Mapped[str] = mapped_column(Text, nullable=False)
    content_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_source_documents_source_type", "source_type"),
        Index("ix_source_documents_source_fingerprint", "source_fingerprint"),
        Index("ix_source_documents_status", "status"),
        Index("ix_source_documents_created_at", "created_at"),
    )


class TaskSourceORM(Base):
    __tablename__ = "knowledge_video_task_sources"

    task_source_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("source_documents.source_document_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="PRIMARY")
    associated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_kv_task_sources_task_id", "task_id"),
        Index("ix_kv_task_sources_source_id", "source_document_id"),
        UniqueConstraint("task_id", "source_document_id", name="uq_kv_task_source"),
    )


class EvidenceItemORM(Base):
    __tablename__ = "evidence_items"

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("source_documents.source_document_id", ondelete="CASCADE"),
        nullable=False,
    )
    locator_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False, default=dict)
    original_excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_fact: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_role: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    extraction_method: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_evidence_items_source_document_id", "source_document_id"),
        Index("ix_evidence_items_evidence_role", "evidence_role"),
        Index("ix_evidence_items_content_hash", "content_hash"),
        Index("ix_evidence_items_created_at", "created_at"),
    )


class KnowledgeClaimORM(Base):
    __tablename__ = "knowledge_claims"

    knowledge_claim_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    claim_type: Mapped[str] = mapped_column(String(32), nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_refs_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    verification_status: Mapped[str] = mapped_column(String(32), nullable=False)
    conflict_evidence_refs_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_knowledge_claims_claim_type", "claim_type"),
        Index("ix_knowledge_claims_verification_status", "verification_status"),
        Index("ix_knowledge_claims_created_at", "created_at"),
    )


class EvidenceSnapshotORM(Base):
    __tablename__ = "evidence_snapshots"

    evidence_snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_document_ids_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    evidence_ids_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    knowledge_claim_ids_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_evidence_snapshots_task_id", "task_id"),
        Index("ix_evidence_snapshots_content_fingerprint", "content_fingerprint"),
        Index("ix_evidence_snapshots_created_at", "created_at"),
    )


class KnowledgeChunkORM(Base):
    __tablename__ = "knowledge_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("source_documents.source_document_id", ondelete="CASCADE"),
        nullable=False,
    )
    processing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    locator_json: Mapped[dict] = mapped_column(EvidenceType, nullable=False, default=dict)
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_knowledge_chunks_source_document_id", "source_document_id"),
        Index("ix_knowledge_chunks_content_fingerprint", "content_fingerprint"),
        Index("ix_knowledge_chunks_created_at", "created_at"),
        UniqueConstraint("source_document_id", "processing_version", "chunk_index", name="uq_knowledge_chunks_source_ver_idx"),
    )


class RetrievalSnapshotORM(Base):
    __tablename__ = "retrieval_snapshots"

    retrieval_snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_video_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    source_scope_ids_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    retrieval_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    processing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    candidates_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    selected_evidence_ids_json: Mapped[list] = mapped_column(EvidenceType, nullable=False, default=list)
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_retrieval_snapshots_task_id", "task_id"),
        Index("ix_retrieval_snapshots_content_fingerprint", "content_fingerprint"),
        Index("ix_retrieval_snapshots_created_at", "created_at"),
    )
