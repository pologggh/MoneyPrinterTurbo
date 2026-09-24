from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict

from app.domain.enums import BeatType, StoryboardSnapshotState, VisualType
from app.domain.shot import Shot, ShotRevision
from app.domain.storyboard import StoryboardSnapshot

# =============================================================================
# Exceptions
# =============================================================================


class StoryboardEditingError(Exception):
    """Base exception for storyboard editing domain operations."""

    def __init__(self, message: str, code: str = "STORYBOARD_EDITING_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class StoryboardNotFoundError(StoryboardEditingError):
    """Raised when the specified storyboard snapshot cannot be found."""

    def __init__(self, message: str):
        super().__init__(message, code="STORYBOARD_NOT_FOUND")


class ShotNotFoundError(StoryboardEditingError):
    """Raised when the specified shot entity does not exist."""

    def __init__(self, message: str):
        super().__init__(message, code="SHOT_NOT_FOUND")


class ShotNotInStoryboardError(StoryboardEditingError):
    """Raised when the specified shot is not part of the specified storyboard snapshot."""

    def __init__(self, message: str):
        super().__init__(message, code="SHOT_NOT_IN_STORYBOARD")


class StaleShotRevisionError(StoryboardEditingError):
    """Raised when an edit is based on a stale/outdated shot revision."""

    def __init__(self, message: str):
        super().__init__(message, code="STALE_SHOT_REVISION")


class InvalidShotEditError(StoryboardEditingError):
    """Raised when edited shot fields violate domain constraints."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_SHOT_EDIT")


class InvalidShotOrderError(StoryboardEditingError):
    """Raised when reordering shots within a beat is invalid, duplicate, or incomplete."""

    def __init__(self, message: str):
        super().__init__(message, code="INVALID_SHOT_ORDER")


class CrossBeatMoveNotAllowedError(StoryboardEditingError):
    """Raised when attempting to move a shot across different beat lineages."""

    def __init__(self, message: str):
        super().__init__(message, code="CROSS_BEAT_MOVE_NOT_ALLOWED")


class ApprovedSnapshotImmutableError(StoryboardEditingError):
    """Raised when attempting an in-place mutation of an approved snapshot."""

    def __init__(self, message: str):
        super().__init__(message, code="APPROVED_SNAPSHOT_IMMUTABLE")


# =============================================================================
# Request / View Models
# =============================================================================


class EditShotInput(BaseModel):
    """
    Input model for editing planning fields of a Shot within a StoryboardSnapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_snapshot_id: str
    shot_id: str
    base_shot_revision_id: str
    narration: str | None = None
    target_duration: float | None = None
    visual_goal: str | None = None
    visual_type: VisualType | None = None
    scene_description: str | None = None
    generation_prompt: str | None = None
    camera_movement: str | None = None


class ReorderShotsInput(BaseModel):
    """
    Input model for reordering Shots within a single ContentBeat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_snapshot_id: str
    beat_lineage_id: str
    ordered_shot_ids: tuple[str, ...]


class ShotEditingView(BaseModel):
    """
    Read model for a single Shot within an editing view.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: str
    shot_revision_id: str
    revision_number: int
    beat_lineage_id: str
    created_from_beat_instance_id: str
    local_order: int
    narration: str
    target_duration: float
    visual_goal: str
    visual_type: VisualType
    scene_description: str
    generation_prompt: str
    camera_movement: str = ""
    evidence_refs: tuple[str, ...] = ()
    created_at: datetime
    produced_asset_version_id: str | None = None
    asset_file_path: str | None = None
    asset_media_type: str | None = None
    asset_width: int | None = None
    asset_height: int | None = None
    asset_duration: float | None = None
    execution_status: str | None = None


class BeatEditingView(BaseModel):
    """
    Read model for a ContentBeat containing its ordered shots within the snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_id: str
    beat_lineage_id: str
    beat_type: BeatType
    order: int
    intent: str
    target_duration: float
    importance: float
    shots: tuple[ShotEditingView, ...] = ()


class StoryboardEditingView(BaseModel):
    """
    Complete UI-ready read model reconstructing the exact snapshot state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storyboard_snapshot_id: str
    content_plan_revision_id: str
    state: StoryboardSnapshotState
    topic: str = ""
    overall_target_duration: float = 0.0
    beats: tuple[BeatEditingView, ...] = ()
    created_at: datetime

    @property
    def total_shots(self) -> int:
        return sum(len(b.shots) for b in self.beats)

    @property
    def total_duration(self) -> float:
        return sum(s.target_duration for b in self.beats for s in b.shots)


# =============================================================================
# Storyboard Editing Service
# =============================================================================


class StoryboardEditingService:
    """
    Application service managing backend storyboard editing, shot revisions,
    same-beat reordering, and view reconstruction.
    """

    def __init__(
        self,
        plan_repository: Any,
        shot_repository: Any,
        storyboard_repository: Any,
        execution_repository: Any | None = None,
    ):
        self._plan_repo = plan_repository
        self._shot_repo = shot_repository
        self._storyboard_repo = storyboard_repository
        self._execution_repo = execution_repository

    def get_storyboard_for_editing(
        self, storyboard_snapshot_id: str
    ) -> StoryboardEditingView:
        """
        Reconstructs the complete editing view from the requested StoryboardSnapshot,
        resolving exact frozen ShotRevisions rather than latest database records.
        """
        snapshot: StoryboardSnapshot | None = self._storyboard_repo.get_snapshot(
            storyboard_snapshot_id
        )
        if snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{storyboard_snapshot_id}' not found."
            )

        plan = self._plan_repo.get_revision(snapshot.content_plan_revision_id)
        if plan is None:
            raise StoryboardNotFoundError(
                f"ContentPlanRevision '{snapshot.content_plan_revision_id}' not found for snapshot."
            )

        frozen_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
            snapshot.storyboard_snapshot_id
        )
        revisions_by_lineage: dict[str, list[ShotEditingView]] = defaultdict(list)

        for rev in frozen_revisions:
            shot: Shot | None = self._shot_repo.get_shot(rev.shot_id)
            local_order = shot.local_order if shot is not None else 1

            produced_asset_version_id = None
            asset_file_path = None
            asset_media_type = None
            asset_width = None
            asset_height = None
            asset_duration = None
            execution_status = None

            if self._execution_repo is not None:
                try:
                    asset_versions = (
                        self._execution_repo.list_asset_versions_for_shot_revision(
                            rev.shot_revision_id
                        )
                    )
                    if asset_versions:
                        latest_asset = asset_versions[0]
                        produced_asset_version_id = latest_asset.shot_asset_version_id
                        asset_file_path = latest_asset.file_path
                        asset_media_type = (
                            latest_asset.media_type.value
                            if hasattr(latest_asset.media_type, "value")
                            else str(latest_asset.media_type)
                        )
                        asset_width = getattr(latest_asset, "width", None)
                        asset_height = getattr(latest_asset, "height", None)
                        asset_duration = getattr(latest_asset, "duration_seconds", None)
                        execution_status = "SUCCEEDED"
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Could not load asset version for {rev.shot_revision_id}: {exc}")

            shot_view = ShotEditingView(
                shot_id=rev.shot_id,
                shot_revision_id=rev.shot_revision_id,
                revision_number=rev.revision_number,
                beat_lineage_id=rev.beat_lineage_id,
                created_from_beat_instance_id=rev.created_from_beat_instance_id,
                local_order=local_order,
                narration=rev.narration,
                target_duration=rev.target_duration,
                visual_goal=rev.visual_goal,
                visual_type=rev.visual_type,
                scene_description=rev.scene_description,
                generation_prompt=rev.generation_prompt,
                camera_movement=rev.camera_movement,
                evidence_refs=rev.evidence_refs,
                created_at=rev.created_at,
                produced_asset_version_id=produced_asset_version_id,
                asset_file_path=asset_file_path,
                asset_media_type=asset_media_type,
                asset_width=asset_width,
                asset_height=asset_height,
                asset_duration=asset_duration,
                execution_status=execution_status,
            )
            revisions_by_lineage[rev.beat_lineage_id].append(shot_view)

        # Build beats sorted by beat.order
        beat_views: list[BeatEditingView] = []
        for beat in sorted(plan.beats, key=lambda b: b.order):
            shots_in_beat = revisions_by_lineage.get(beat.beat_lineage_id, [])
            # Sort shots in beat by local_order
            shots_in_beat.sort(key=lambda s: s.local_order)

            beat_views.append(
                BeatEditingView(
                    beat_id=beat.beat_id,
                    beat_lineage_id=beat.beat_lineage_id,
                    beat_type=beat.beat_type,
                    order=beat.order,
                    intent=beat.intent,
                    target_duration=beat.target_duration,
                    importance=beat.importance,
                    shots=tuple(shots_in_beat),
                )
            )

        return StoryboardEditingView(
            storyboard_snapshot_id=snapshot.storyboard_snapshot_id,
            content_plan_revision_id=snapshot.content_plan_revision_id,
            state=snapshot.snapshot_state,
            topic=plan.topic,
            overall_target_duration=plan.overall_target_duration,
            beats=tuple(beat_views),
            created_at=snapshot.created_at,
        )

    def edit_shot(
        self,
        input_data: EditShotInput,
        *,
        trace_context: Any | None = None,
        trace_writer: Any | None = None,
    ) -> StoryboardSnapshot:
        """
        Safely applies planning edits to a Shot, producing a new ShotRevision
        and a new immutable DRAFT StoryboardSnapshot while preserving history.
        """
        snapshot: StoryboardSnapshot | None = self._storyboard_repo.get_snapshot(
            input_data.storyboard_snapshot_id
        )
        if snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{input_data.storyboard_snapshot_id}' not found."
            )

        frozen_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
            snapshot.storyboard_snapshot_id
        )
        target_rev: ShotRevision | None = None
        target_rev_index: int = -1

        for idx, rev in enumerate(frozen_revisions):
            if rev.shot_id == input_data.shot_id:
                target_rev = rev
                target_rev_index = idx
                break

        if target_rev is None or target_rev_index == -1:
            raise ShotNotInStoryboardError(
                f"Shot '{input_data.shot_id}' is not present in StoryboardSnapshot "
                f"'{input_data.storyboard_snapshot_id}'."
            )

        # Stale edit check
        if target_rev.shot_revision_id != input_data.base_shot_revision_id:
            raise StaleShotRevisionError(
                f"Stale edit: Shot '{input_data.shot_id}' in snapshot "
                f"'{input_data.storyboard_snapshot_id}' currently references revision "
                f"'{target_rev.shot_revision_id}', but base was '{input_data.base_shot_revision_id}'."
            )

        # Field validation
        if input_data.narration is not None:
            if not input_data.narration.strip():
                raise InvalidShotEditError(
                    "Narration cannot be empty or whitespace."
                )
            narration = input_data.narration.strip()
        else:
            narration = target_rev.narration

        if input_data.target_duration is not None:
            if input_data.target_duration <= 0:
                raise InvalidShotEditError(
                    "target_duration must be greater than 0."
                )
            target_duration = input_data.target_duration
        else:
            target_duration = target_rev.target_duration

        visual_type = input_data.visual_type or target_rev.visual_type
        visual_goal = (
            input_data.visual_goal
            if input_data.visual_goal is not None
            else target_rev.visual_goal
        )
        scene_description = (
            input_data.scene_description
            if input_data.scene_description is not None
            else target_rev.scene_description
        )
        generation_prompt = (
            input_data.generation_prompt
            if input_data.generation_prompt is not None
            else target_rev.generation_prompt
        )
        camera_movement = (
            input_data.camera_movement
            if input_data.camera_movement is not None
            else target_rev.camera_movement
        )

        # Determine next revision number
        existing_revisions = self._shot_repo.list_revisions(target_rev.shot_id)
        if existing_revisions:
            next_rev_num = (
                max(r.revision_number for r in existing_revisions) + 1
            )
        else:
            next_rev_num = target_rev.revision_number + 1

        new_shot_rev = ShotRevision(
            shot_revision_id=str(uuid4()),
            shot_id=target_rev.shot_id,
            revision_number=next_rev_num,
            beat_lineage_id=target_rev.beat_lineage_id,
            created_from_beat_instance_id=target_rev.created_from_beat_instance_id,
            narration=narration,
            target_duration=target_duration,
            visual_goal=visual_goal,
            visual_type=visual_type,
            scene_description=scene_description,
            generation_prompt=generation_prompt,
            camera_movement=camera_movement,
            evidence_refs=target_rev.evidence_refs,  # Read-only in Phase 3.1
            created_at=datetime.now(UTC),
        )

        # Persist new revision
        self._shot_repo.add_revision(new_shot_rev)

        # Assemble new snapshot with replaced revision
        new_shot_revision_ids = list(snapshot.shot_revision_ids)
        new_shot_revision_ids[target_rev_index] = new_shot_rev.shot_revision_id

        new_snapshot = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=snapshot.content_plan_revision_id,
            shot_revision_ids=tuple(new_shot_revision_ids),
            snapshot_state=StoryboardSnapshotState.DRAFT,  # Always DRAFT
            created_at=datetime.now(UTC),
        )

        self._storyboard_repo.add_snapshot(new_snapshot)

        if trace_writer is not None and trace_context is not None:
            from app.domain.trace import (
                StoryboardEditTraceData,
                TraceEventStatus,
                TraceEventType,
            )

            ctx = trace_context.with_ids(
                content_plan_revision_id=snapshot.content_plan_revision_id,
                storyboard_snapshot_id=new_snapshot.storyboard_snapshot_id,
                shot_id=target_rev.shot_id,
                shot_revision_id=new_shot_rev.shot_revision_id,
            )
            edit_data = StoryboardEditTraceData(
                old_snapshot_id=snapshot.storyboard_snapshot_id,
                new_snapshot_id=new_snapshot.storyboard_snapshot_id,
                shot_id=target_rev.shot_id,
                old_shot_revision_id=target_rev.shot_revision_id,
                new_shot_revision_id=new_shot_rev.shot_revision_id,
            )
            trace_writer.record_event(
                context=ctx,
                event_type=TraceEventType.STORYBOARD_EDITED,
                status=TraceEventStatus.SUCCEEDED,
                attributes=edit_data,
            )

        return new_snapshot

    def reorder_shots_within_beat(
        self, input_data: ReorderShotsInput
    ) -> StoryboardSnapshot:
        """
        Reorders Shots within a single ContentBeat, updating local_orders and
        producing a new DRAFT StoryboardSnapshot.
        """
        snapshot: StoryboardSnapshot | None = self._storyboard_repo.get_snapshot(
            input_data.storyboard_snapshot_id
        )
        if snapshot is None:
            raise StoryboardNotFoundError(
                f"StoryboardSnapshot '{input_data.storyboard_snapshot_id}' not found."
            )

        frozen_revisions = self._storyboard_repo.get_snapshot_shot_revisions(
            snapshot.storyboard_snapshot_id
        )
        revisions_in_beat: list[ShotRevision] = []
        other_revisions: list[ShotRevision] = []

        for rev in frozen_revisions:
            if rev.beat_lineage_id == input_data.beat_lineage_id:
                revisions_in_beat.append(rev)
            else:
                other_revisions.append(rev)

        existing_shot_ids_in_beat = {r.shot_id for r in revisions_in_beat}
        requested_shot_ids = input_data.ordered_shot_ids

        # Check for duplicates
        if len(requested_shot_ids) != len(set(requested_shot_ids)):
            raise InvalidShotOrderError(
                f"Duplicate shot_ids found in reorder request: {requested_shot_ids}"
            )

        # Check for cross-beat movement or missing shots
        requested_set = set(requested_shot_ids)
        if requested_set != existing_shot_ids_in_beat:
            other_shot_ids = {r.shot_id for r in other_revisions}
            if requested_set.intersection(other_shot_ids):
                raise CrossBeatMoveNotAllowedError(
                    "Moving shots across different beats is strictly not allowed."
                )
            raise InvalidShotOrderError(
                f"Reorder request must include all shots belonging to beat "
                f"'{input_data.beat_lineage_id}'. Expected {sorted(existing_shot_ids_in_beat)}, "
                f"got {sorted(requested_set)}."
            )

        # Update local_order on Shot entities
        shot_rev_map = {r.shot_id: r.shot_revision_id for r in revisions_in_beat}
        reordered_shot_rev_ids: list[str] = []

        for idx, shot_id in enumerate(requested_shot_ids, start=1):
            if hasattr(self._shot_repo, "update_shot_local_order"):
                self._shot_repo.update_shot_local_order(shot_id, idx)
            reordered_shot_rev_ids.append(shot_rev_map[shot_id])

        # Replace in snapshot sequence
        new_shot_revision_ids: list[str] = []
        reordered_inserted = False

        for rev in frozen_revisions:
            if rev.beat_lineage_id == input_data.beat_lineage_id:
                if not reordered_inserted:
                    new_shot_revision_ids.extend(reordered_shot_rev_ids)
                    reordered_inserted = True
            else:
                new_shot_revision_ids.append(rev.shot_revision_id)

        new_snapshot = StoryboardSnapshot(
            storyboard_snapshot_id=str(uuid4()),
            content_plan_revision_id=snapshot.content_plan_revision_id,
            shot_revision_ids=tuple(new_shot_revision_ids),
            snapshot_state=StoryboardSnapshotState.DRAFT,
            created_at=datetime.now(UTC),
        )

        self._storyboard_repo.add_snapshot(new_snapshot)
        return new_snapshot
