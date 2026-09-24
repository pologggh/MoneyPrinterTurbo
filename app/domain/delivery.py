from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class DeliveryManifest(BaseModel):
    """
    Immutable representation of the final delivery manifest for a KnowledgeVideoTask.

    References the accepted final MP4 video, subtitle artifact, source/evidence report,
    and workflow execution report. Does NOT copy binary payloads or full database tables;
    acts as an authoritative, durable index of deliverable artifacts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delivery_manifest_id: str = Field(default_factory=lambda: f"dm_{uuid4().hex[:24]}")
    task_id: str
    composition_output_id: str
    evaluation_snapshot_id: str
    final_video_path: str
    final_video_hash: str
    final_video_size_bytes: int = Field(gt=0, description="Video size in bytes")
    subtitle_path: str | None = None
    subtitle_hash: str | None = None
    source_report_path: str
    source_report_hash: str
    execution_report_path: str
    execution_report_hash: str
    delivery_params_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        task_id: str,
        composition_output_id: str,
        evaluation_snapshot_id: str,
        final_video_path: str,
        final_video_hash: str,
        final_video_size_bytes: int,
        source_report_path: str,
        source_report_hash: str,
        execution_report_path: str,
        execution_report_hash: str,
        subtitle_path: str | None = None,
        subtitle_hash: str | None = None,
        delivery_params_snapshot: dict[str, Any] | None = None,
        delivery_manifest_id: str | None = None,
        created_at: datetime | None = None,
    ) -> DeliveryManifest:
        return cls(
            delivery_manifest_id=delivery_manifest_id or f"dm_{uuid4().hex[:24]}",
            task_id=task_id,
            composition_output_id=composition_output_id,
            evaluation_snapshot_id=evaluation_snapshot_id,
            final_video_path=final_video_path,
            final_video_hash=final_video_hash,
            final_video_size_bytes=final_video_size_bytes,
            subtitle_path=subtitle_path,
            subtitle_hash=subtitle_hash,
            source_report_path=source_report_path,
            source_report_hash=source_report_hash,
            execution_report_path=execution_report_path,
            execution_report_hash=execution_report_hash,
            delivery_params_snapshot=dict(delivery_params_snapshot or {}),
            created_at=created_at or datetime.now(UTC),
        )
