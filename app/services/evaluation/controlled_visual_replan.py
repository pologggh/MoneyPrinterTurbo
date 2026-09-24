from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app.domain.content_plan import ContentBeat
from app.domain.enums import StoryboardSnapshotState, VisualType
from app.domain.evaluation import EvaluationSnapshot
from app.domain.shot import ShotRevision
from app.domain.storyboard import StoryboardSnapshot


class ControlledReplanViolationError(ValueError):
    """Base exception for controlled replan invariant violations."""


class FactualNarrationModificationForbiddenError(ControlledReplanViolationError):
    """Raised when controlled visual replan attempts to modify narration."""


class EvidenceModificationForbiddenError(ControlledReplanViolationError):
    """Raised when controlled visual replan attempts to modify evidence references."""


class StructuralModificationForbiddenError(ControlledReplanViolationError):
    """Raised when controlled replan attempts to alter duration, beat identity, order or shot count."""


class ApprovedSnapshotMutationForbiddenError(ControlledReplanViolationError):
    """Raised when an attempt is made to overwrite or alter an APPROVED StoryboardSnapshot."""


class ControlledVisualReplanRequest(BaseModel):
    """
    Input request to create a candidate ShotRevision for quality remediation.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_revision: ShotRevision
    beat: ContentBeat
    evaluation_snapshot: EvaluationSnapshot
    content_plan_revision_id: str = "cprev-1"
    visual_goal: str | None = None
    visual_type: VisualType | None = None
    scene_description: str | None = None
    generation_prompt: str | None = None
    camera_movement: str | None = None
    # Validation fields to ensure caller cannot secretly pass altered narration/evidence/duration
    proposed_narration: str | None = None
    proposed_evidence_refs: tuple[str, ...] | None = None
    proposed_target_duration: float | None = None


class ControlledVisualReplanResult(BaseModel):
    """
    Immutable result containing the new candidate ShotRevision and candidate DRAFT snapshot.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    original_shot_revision: ShotRevision
    candidate_shot_revision: ShotRevision
    candidate_draft_snapshot: StoryboardSnapshot
    reason_codes: tuple[str, ...] = ()


class ControlledVisualReplanService:
    """
    Service enforcing the strict invariants of Phase 6.3 Controlled Visual Replan:
    1. Preserves shot_id, beat_lineage_id, narration, evidence_refs, target_duration.
    2. Modifies ONLY: visual_goal, visual_type, scene_description, generation_prompt, camera_movement.
    3. Never modifies an APPROVED StoryboardSnapshot; creates a new candidate DRAFT snapshot.
    4. Replan candidate provenance is strictly DRAFT (requires human approval for official storyboard).
    """

    @classmethod
    def create_candidate_revision(
        cls,
        request: ControlledVisualReplanRequest,
        existing_approved_snapshot: StoryboardSnapshot | None = None,
    ) -> ControlledVisualReplanResult:
        orig = request.shot_revision

        # Invariant check 1: Narration cannot be altered
        if request.proposed_narration is not None and request.proposed_narration != orig.narration:
            raise FactualNarrationModificationForbiddenError(
                f"Controlled visual replan cannot modify narration. "
                f"Original: '{orig.narration}', Proposed: '{request.proposed_narration}'"
            )

        # Invariant check 2: Evidence refs cannot be altered
        if (
            request.proposed_evidence_refs is not None
            and tuple(request.proposed_evidence_refs) != tuple(orig.evidence_refs)
        ):
            raise EvidenceModificationForbiddenError(
                "Controlled visual replan cannot alter evidence refs."
            )

        # Invariant check 3: Duration cannot be altered
        if (
            request.proposed_target_duration is not None
            and abs(request.proposed_target_duration - orig.target_duration) > 1e-4
        ):
            raise StructuralModificationForbiddenError(
                f"Controlled visual replan cannot alter target duration ({orig.target_duration}s vs {request.proposed_target_duration}s)."
            )

        # Invariant check 4: Beat lineage must match
        if orig.beat_lineage_id != request.beat.beat_lineage_id:
            raise StructuralModificationForbiddenError(
                f"Beat lineage mismatch: revision {orig.beat_lineage_id} != beat {request.beat.beat_lineage_id}"
            )

        # Resolve updated visual fields (preserving original if not specified)
        new_visual_goal = request.visual_goal or orig.visual_goal
        new_visual_type = request.visual_type or orig.visual_type
        new_scene_description = (
            request.scene_description
            or (
                f"{orig.scene_description} (Reframed for visual clarity and composition compliance)"
                if request.scene_description is None and request.visual_goal is None
                else orig.scene_description
            )
        )
        new_generation_prompt = (
            request.generation_prompt
            or (
                f"{orig.generation_prompt}, centered composition, cinematic lighting, high quality"
                if request.generation_prompt is None
                else orig.generation_prompt
            )
        )
        new_camera_movement = request.camera_movement or orig.camera_movement

        # Create NEW ShotRevision for the SAME shot_id with incremented revision number
        candidate_revision = ShotRevision(
            shot_revision_id=str(uuid4()),
            shot_id=orig.shot_id,  # MUST be identical!
            revision_number=orig.revision_number + 1,
            beat_lineage_id=orig.beat_lineage_id,
            created_from_beat_instance_id=request.beat.beat_id,
            narration=orig.narration,  # STRICTLY LOCKED
            target_duration=orig.target_duration,  # STRICTLY LOCKED
            visual_goal=new_visual_goal,
            visual_type=new_visual_type,
            scene_description=new_scene_description,
            generation_prompt=new_generation_prompt,
            camera_movement=new_camera_movement,
            evidence_refs=orig.evidence_refs,  # STRICTLY LOCKED
        )

        # Packaging into candidate DRAFT snapshot (never modifying existing approved snapshot)
        if existing_approved_snapshot is not None:
            if existing_approved_snapshot.snapshot_state != StoryboardSnapshotState.APPROVED:
                pass  # not approved, but still frozen
            # Replace old shot_revision_id with new candidate_revision_id in a NEW DRAFT snapshot
            new_shot_rev_ids = tuple(
                candidate_revision.shot_revision_id
                if rid == orig.shot_revision_id
                else rid
                for rid in existing_approved_snapshot.shot_revision_ids
            )
            candidate_snapshot = StoryboardSnapshot(
                storyboard_snapshot_id=str(uuid4()),
                content_plan_revision_id=existing_approved_snapshot.content_plan_revision_id,
                shot_revision_ids=new_shot_rev_ids,
                snapshot_state=StoryboardSnapshotState.DRAFT,
            )
        else:
            candidate_snapshot = StoryboardSnapshot(
                storyboard_snapshot_id=str(uuid4()),
                content_plan_revision_id=request.content_plan_revision_id,
                shot_revision_ids=(candidate_revision.shot_revision_id,),
                snapshot_state=StoryboardSnapshotState.DRAFT,
            )

        return ControlledVisualReplanResult(
            original_shot_revision=orig,
            candidate_shot_revision=candidate_revision,
            candidate_draft_snapshot=candidate_snapshot,
            reason_codes=("CONTROLLED_VISUAL_REPLAN_CANDIDATE_CREATED",),
        )
