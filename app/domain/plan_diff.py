from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.content_plan import ContentPlanRevision
from app.domain.fingerprint import compute_beat_content_fingerprint


class BeatChangeType(str, Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    UNCHANGED = "UNCHANGED"
    MOVED = "MOVED"
    MODIFIED = "MODIFIED"


class ShotReuseDecision(str, Enum):
    REUSE_EXISTING_SHOTS = "REUSE_EXISTING_SHOTS"
    GENERATE_NEW_SHOTS = "GENERATE_NEW_SHOTS"
    NO_PREVIOUS_SHOTS = "NO_PREVIOUS_SHOTS"


class DuplicateBeatLineageError(ValueError):
    """Raised when a ContentPlanRevision contains duplicate beat_lineage_ids."""


def validate_content_plan_revision_lineages(plan: ContentPlanRevision) -> None:
    """
    Validates that each beat_lineage_id appears at most once in a ContentPlanRevision.

    Raises DuplicateBeatLineageError if any duplicate lineage IDs are detected.
    """
    seen_lineages: set[str] = set()
    duplicates: list[str] = []
    for beat in plan.beats:
        if beat.beat_lineage_id in seen_lineages:
            duplicates.append(beat.beat_lineage_id)
        seen_lineages.add(beat.beat_lineage_id)

    if duplicates:
        raise DuplicateBeatLineageError(
            f"ContentPlanRevision '{plan.content_plan_revision_id}' contains duplicate "
            f"beat_lineage_id(s): {sorted(duplicates)}"
        )


def generate_beat_lineage_id() -> str:
    """Generates a stable, system-owned lineage identifier for a ContentBeat."""
    return str(uuid4())


class BeatChange(BaseModel):
    """
    Represents the detected difference for a specific beat lineage across two plan revisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    beat_lineage_id: str
    change_type: BeatChangeType
    old_beat_instance_id: str | None = None
    new_beat_instance_id: str | None = None
    old_fingerprint: str | None = None
    new_fingerprint: str | None = None
    old_order: int | None = None
    new_order: int | None = None
    reason: str = ""


class ContentPlanDiff(BaseModel):
    """
    Immutable diff report comparing two revisions of a ContentPlan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    old_revision_id: str | None = None
    new_revision_id: str
    beat_changes: tuple[BeatChange, ...] = Field(default=())


def decide_shot_reuse(change_type: BeatChangeType) -> ShotReuseDecision:
    """
    Determines shot reuse / invalidation decision for a given beat change type.

    LOCKED RULES:
    - UNCHANGED -> REUSE_EXISTING_SHOTS
    - MOVED     -> REUSE_EXISTING_SHOTS
    - MODIFIED  -> GENERATE_NEW_SHOTS (all existing shots for that beat are invalidated)
    - ADDED     -> NO_PREVIOUS_SHOTS (new beat, no prior shots exist)
    - REMOVED   -> NO_PREVIOUS_SHOTS (or historical only, not part of new plan)
    """
    if change_type in (BeatChangeType.UNCHANGED, BeatChangeType.MOVED):
        return ShotReuseDecision.REUSE_EXISTING_SHOTS
    elif change_type == BeatChangeType.MODIFIED:
        return ShotReuseDecision.GENERATE_NEW_SHOTS
    else:  # ADDED or REMOVED
        return ShotReuseDecision.NO_PREVIOUS_SHOTS


def diff_content_plans(
    old_plan: ContentPlanRevision | None,
    new_plan: ContentPlanRevision,
) -> ContentPlanDiff:
    """
    Performs deterministic diffing between two ContentPlanRevisions based on beat_lineage_id.

    DIFF RULES (LOCKED):
    1. Lineage exists only in new revision -> ADDED
    2. Lineage exists only in old revision -> REMOVED
    3. Same lineage + same content fingerprint + same order -> UNCHANGED
    4. Same lineage + same content fingerprint + changed order -> MOVED
    5. Same lineage + changed content fingerprint -> MODIFIED
       (Even if order also changed, classification remains MODIFIED)
    """
    validate_content_plan_revision_lineages(new_plan)
    if old_plan is not None:
        validate_content_plan_revision_lineages(old_plan)

    new_beats_by_lineage = {b.beat_lineage_id: b for b in new_plan.beats}
    old_beats_by_lineage = (
        {b.beat_lineage_id: b for b in old_plan.beats} if old_plan is not None else {}
    )

    all_lineages = set(old_beats_by_lineage.keys()) | set(new_beats_by_lineage.keys())
    changes: list[BeatChange] = []

    for lineage_id in all_lineages:
        in_old = lineage_id in old_beats_by_lineage
        in_new = lineage_id in new_beats_by_lineage

        if in_new and not in_old:
            new_beat = new_beats_by_lineage[lineage_id]
            new_fp = compute_beat_content_fingerprint(new_beat)
            changes.append(
                BeatChange(
                    beat_lineage_id=lineage_id,
                    change_type=BeatChangeType.ADDED,
                    new_beat_instance_id=new_beat.beat_id,
                    new_fingerprint=new_fp,
                    new_order=new_beat.order,
                    reason="Beat added in new revision",
                )
            )
        elif in_old and not in_new:
            old_beat = old_beats_by_lineage[lineage_id]
            old_fp = compute_beat_content_fingerprint(old_beat)
            changes.append(
                BeatChange(
                    beat_lineage_id=lineage_id,
                    change_type=BeatChangeType.REMOVED,
                    old_beat_instance_id=old_beat.beat_id,
                    old_fingerprint=old_fp,
                    old_order=old_beat.order,
                    reason="Beat removed in new revision",
                )
            )
        else:
            old_beat = old_beats_by_lineage[lineage_id]
            new_beat = new_beats_by_lineage[lineage_id]
            old_fp = compute_beat_content_fingerprint(old_beat)
            new_fp = compute_beat_content_fingerprint(new_beat)

            same_content = old_fp == new_fp
            same_order = old_beat.order == new_beat.order

            if not same_content:
                # Rule 5: If content changed, classification is MODIFIED regardless of order
                changes.append(
                    BeatChange(
                        beat_lineage_id=lineage_id,
                        change_type=BeatChangeType.MODIFIED,
                        old_beat_instance_id=old_beat.beat_id,
                        new_beat_instance_id=new_beat.beat_id,
                        old_fingerprint=old_fp,
                        new_fingerprint=new_fp,
                        old_order=old_beat.order,
                        new_order=new_beat.order,
                        reason="Content modified",
                    )
                )
            elif not same_order:
                # Rule 4: Same content but changed order -> MOVED
                changes.append(
                    BeatChange(
                        beat_lineage_id=lineage_id,
                        change_type=BeatChangeType.MOVED,
                        old_beat_instance_id=old_beat.beat_id,
                        new_beat_instance_id=new_beat.beat_id,
                        old_fingerprint=old_fp,
                        new_fingerprint=new_fp,
                        old_order=old_beat.order,
                        new_order=new_beat.order,
                        reason=f"Moved from order {old_beat.order} to {new_beat.order}",
                    )
                )
            else:
                # Rule 3: Same content and same order -> UNCHANGED
                changes.append(
                    BeatChange(
                        beat_lineage_id=lineage_id,
                        change_type=BeatChangeType.UNCHANGED,
                        old_beat_instance_id=old_beat.beat_id,
                        new_beat_instance_id=new_beat.beat_id,
                        old_fingerprint=old_fp,
                        new_fingerprint=new_fp,
                        old_order=old_beat.order,
                        new_order=new_beat.order,
                        reason="Unchanged",
                    )
                )

    # Deterministic sorting of changes: by new_order (if present), then old_order, then lineage_id
    sorted_changes = tuple(
        sorted(
            changes,
            key=lambda c: (
                c.new_order if c.new_order is not None else float("inf"),
                c.old_order if c.old_order is not None else float("inf"),
                c.beat_lineage_id,
            ),
        )
    )

    return ContentPlanDiff(
        old_revision_id=old_plan.content_plan_revision_id if old_plan else None,
        new_revision_id=new_plan.content_plan_revision_id,
        beat_changes=sorted_changes,
    )
