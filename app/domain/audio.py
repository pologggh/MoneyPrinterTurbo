from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AudioOutput(BaseModel):
    """
    Immutable representation of the audio stage output for a KnowledgeVideoTask.

    Contains references and metadata for the synthesized narration audio,
    synchronized subtitles, and optional background music.
    Does NOT store media binary payloads; only references durable file paths and hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    audio_output_id: str = Field(default_factory=lambda: f"ao_{uuid4().hex[:24]}")
    task_id: str
    script_revision_id: str
    execution_run_id: str
    narration_audio_path: str
    narration_audio_hash: str
    actual_narration_duration: float = Field(
        gt=0, description="Real duration in seconds measured from media file"
    )
    subtitle_path: str | None = None
    subtitle_hash: str | None = None
    bgm_path: str | None = None
    bgm_volume: float | None = 0.2
    voice_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        task_id: str,
        script_revision_id: str,
        execution_run_id: str,
        narration_audio_path: str,
        narration_audio_hash: str,
        actual_narration_duration: float,
        subtitle_path: str | None = None,
        subtitle_hash: str | None = None,
        bgm_path: str | None = None,
        bgm_volume: float | None = 0.2,
        voice_config_snapshot: dict[str, Any] | None = None,
        audio_output_id: str | None = None,
        created_at: datetime | None = None,
    ) -> AudioOutput:
        return cls(
            audio_output_id=audio_output_id or f"ao_{uuid4().hex[:24]}",
            task_id=task_id,
            script_revision_id=script_revision_id,
            execution_run_id=execution_run_id,
            narration_audio_path=narration_audio_path,
            narration_audio_hash=narration_audio_hash,
            actual_narration_duration=actual_narration_duration,
            subtitle_path=subtitle_path,
            subtitle_hash=subtitle_hash,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            voice_config_snapshot=voice_config_snapshot or {},
            created_at=created_at or datetime.now(UTC),
        )
