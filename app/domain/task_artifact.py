from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.workflow_state import ArtifactType, Stage


class TaskArtifactRef(BaseModel):
    """
    Immutable typed reference linking a KnowledgeVideoTask to an authoritative
    Phase 1–7 or stage artifact.
    Does NOT store payload content; only references authoritative domain entities.
    """
    model_config = ConfigDict(frozen=True)

    task_artifact_ref_id: str
    task_id: str
    stage: Stage
    artifact_type: ArtifactType
    artifact_id: str
    artifact_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata_json: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        task_id: str,
        stage: Stage,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_version: str | None = None,
        task_artifact_ref_id: str | None = None,
        metadata_json: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> TaskArtifactRef:
        ts = now or datetime.now(UTC)
        return cls(
            task_artifact_ref_id=task_artifact_ref_id or uuid4().hex,
            task_id=task_id,
            stage=stage,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            created_at=ts,
            metadata_json=metadata_json or {},
        )
