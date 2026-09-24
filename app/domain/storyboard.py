from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.content_plan import ContentBeat
from app.domain.enums import StoryboardSnapshotState
from app.domain.shot import Shot


class StoryboardSnapshot(BaseModel):
    """
    Immutable snapshot freezing exact ShotRevision references.

    Captures a frozen historical state of the storyboard (DRAFT or APPROVED).
    Does NOT maintain an independent global shot order; final deterministic
    order is derived from ContentBeat.order + Shot.local_order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    content_plan_revision_id: str = Field(
        description="ID of the ContentPlanRevision this storyboard corresponds to"
    )
    shot_revision_ids: tuple[str, ...] = Field(
        default=(),
        description="Frozen collection of exact shot_revision_ids",
    )
    snapshot_state: StoryboardSnapshotState = Field(
        default=StoryboardSnapshotState.DRAFT,
        description="Snapshot state (DRAFT or APPROVED)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def derive_deterministic_shot_order(
    beats: Sequence[ContentBeat],
    shots: Sequence[Shot],
) -> tuple[Shot, ...]:
    """
    Derives deterministic global shot order from ContentBeat.order then Shot.local_order.

    In V1, global shot ordering is not duplicated inside StoryboardSnapshot.
    Deterministic ordering is defined by:
        primary sort key: ContentBeat.order
        secondary sort key: Shot.local_order
    """
    beat_order_map = {b.beat_lineage_id: b.order for b in beats}
    return tuple(
        sorted(
            shots,
            key=lambda s: (
                beat_order_map.get(s.beat_lineage_id, float("inf")),
                s.local_order,
            ),
        )
    )
