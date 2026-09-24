from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import VisualType


class Shot(BaseModel):
    """
    Stable entity identity for a Shot.

    Every Shot belongs to exactly one Beat lineage and maintains a local order
    within that beat. Mutable or historical content is stored in ShotRevision,
    never directly on Shot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: str = Field(default_factory=lambda: str(uuid4()))
    beat_lineage_id: str = Field(
        description="Lineage ID of the ContentBeat this shot belongs to"
    )
    local_order: int = Field(ge=1, description="1-based local ordering within its beat")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ShotRevision(BaseModel):
    """
    Immutable revision content and provenance for a Shot.

    Freezes all creative, visual, and prompt specifications for a single revision
    of a Shot, linking back to the beat instance and lineage it was created from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_revision_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_id: str = Field(
        description="ID of the stable Shot entity this revision belongs to"
    )
    revision_number: int = Field(ge=1, description="1-based revision version number")
    beat_lineage_id: str = Field(description="Lineage ID of the ContentBeat")
    created_from_beat_instance_id: str = Field(
        description="ID of the specific ContentBeat instance that produced this revision"
    )
    narration: str
    target_duration: float = Field(gt=0, description="Target duration in seconds")
    visual_goal: str
    visual_type: VisualType
    scene_description: str
    generation_prompt: str
    camera_movement: str
    evidence_refs: tuple[str, ...] = Field(
        default=(),
        description="Evidence reference IDs or placeholder references",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
