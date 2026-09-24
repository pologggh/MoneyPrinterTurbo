from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class CompositionOutput(BaseModel):
    """
    Immutable representation of the composition stage output for a KnowledgeVideoTask.

    Contains references and technical media metadata for the finalized rendered MP4 video.
    Does NOT store video binary payloads; only references durable file paths, sizes, and hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    composition_output_id: str = Field(default_factory=lambda: f"co_{uuid4().hex[:24]}")
    task_id: str
    storyboard_snapshot_id: str
    execution_run_id: str
    audio_output_id: str
    video_path: str
    video_hash: str
    duration: float = Field(gt=0, description="Measured video duration in seconds")
    width: int = Field(gt=0, description="Video pixel width")
    height: int = Field(gt=0, description="Video pixel height")
    file_size: int = Field(gt=0, description="Video file size in bytes")
    video_codec: str | None = None
    audio_codec: str | None = None
    fps: float | None = None
    composition_params_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        task_id: str,
        storyboard_snapshot_id: str,
        execution_run_id: str,
        audio_output_id: str,
        video_path: str,
        video_hash: str,
        duration: float,
        width: int,
        height: int,
        file_size: int,
        video_codec: str | None = None,
        audio_codec: str | None = None,
        fps: float | None = None,
        composition_params_snapshot: dict[str, Any] | None = None,
        composition_output_id: str | None = None,
        created_at: datetime | None = None,
    ) -> CompositionOutput:
        return cls(
            composition_output_id=composition_output_id or f"co_{uuid4().hex[:24]}",
            task_id=task_id,
            storyboard_snapshot_id=storyboard_snapshot_id,
            execution_run_id=execution_run_id,
            audio_output_id=audio_output_id,
            video_path=video_path,
            video_hash=video_hash,
            duration=duration,
            width=width,
            height=height,
            file_size=file_size,
            video_codec=video_codec,
            audio_codec=audio_codec,
            fps=fps,
            composition_params_snapshot=composition_params_snapshot or {},
            created_at=created_at or datetime.now(UTC),
        )
