from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import BeatType


class ContentBeat(BaseModel):
    """
    Immutable representation of a content beat within a ContentPlanRevision.

    A ContentBeat represents an atomic structural beat of the knowledge narrative.
    Its lineage ID is stable across plan revisions, while beat_id identifies this
    exact instance in this revision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_id: str = Field(default_factory=lambda: str(uuid4()))
    beat_lineage_id: str = Field(default_factory=lambda: str(uuid4()))
    beat_type: BeatType
    order: int = Field(
        ge=1, description="1-based ordering within the content plan revision"
    )
    intent: str
    target_duration: float = Field(gt=0, description="Target duration in seconds")
    importance: float = Field(ge=0.0, le=1.0, description="Importance score [0.0, 1.0]")
    evidence_refs: tuple[str, ...] = Field(
        default=(),
        description="Evidence reference IDs or placeholder references",
    )


class ContentPlanRevision(BaseModel):
    """
    Immutable revision of a content plan.

    Contains a frozen sequence of ContentBeats representing the planned narrative
    structure for the video.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content_plan_revision_id: str = Field(default_factory=lambda: str(uuid4()))
    revision_number: int = Field(ge=1, description="1-based revision version number")
    topic: str
    overall_target_duration: float = Field(
        gt=0, description="Overall target duration in seconds"
    )
    beats: tuple[ContentBeat, ...] = Field(
        default=(),
        description="Immutable sequence of content beats in this plan revision",
    )
    global_retrieval_snapshot_id: str | None = Field(
        default=None,
        description="Optional future reference to a retrieval snapshot",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
